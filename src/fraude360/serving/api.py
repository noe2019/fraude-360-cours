"""ETAPE 08 - Deploiement : service de scoring temps reel.

API REST (FastAPI) qui reproduit le point de terminaison Databricks Model
Serving. Elle combine deux dispositifs, comme en production :

    score du MODELE  (probabilite calibree)     -> couverture large, apprise
    score des REGLES (expertise des enqueteurs) -> reaction immediate, explicable

La decision finale est ternaire :
    APPROUVER  : la transaction passe sans friction
    REVOIR     : elle part dans la file de l'equipe d'enquete
    BLOQUER    : elle est refusee et le membre est contacte

Toutes les decisions sont journalisees (artifacts/service/decisions.jsonl) :
c'est cette trace qui alimente la surveillance (etape 09) et le tableau de bord.

Lancement :  python -m fraude360.serving.api
Documentation interactive : http://127.0.0.1:8077/docs
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraude360.config import charger_config
from fraude360.modeling import registre
from fraude360.serving.regles import MoteurDeRegles, regle_nouvelle_typologie
from fraude360.utils.journal import lire_json, obtenir_journal

LOG = obtenir_journal("fraude360.service")
ETAT: Dict[str, Any] = {}


def charger_etat() -> Dict[str, Any]:
    """Charge le modele champion, le seuil et le moteur de regles."""
    cfg = charger_config()
    resume = lire_json(cfg.chemin("artefacts") / "modele" / "resume_courant.json")
    modele = registre.charger_par_alias(registre.ALIAS_CHAMPION, cfg)

    moteur = MoteurDeRegles()
    if resume.get("etiquette", "v1") != "v1":
        moteur.ajouter(regle_nouvelle_typologie())

    seuil_blocage = cfg["service"]["seuil_blocage"] or resume["seuil_decision"]
    journal = cfg.chemin("artefacts") / "service" / "decisions.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)

    etat = {
        "cfg": cfg,
        "modele": modele,
        "moteur": moteur,
        "variables": resume["variables_numeriques"] + resume["variables_categorielles"],
        "seuil_blocage": float(seuil_blocage),
        "version_modele": resume.get("version_registre"),
        "etiquette": resume.get("etiquette", "v1"),
        "modele_retenu": resume.get("modele_retenu"),
        "journal": journal,
        "demarre_a": datetime.now().isoformat(),
    }
    # Le seuil de revue est place a la moitie du seuil de blocage : zone grise
    etat["seuil_revue"] = round(etat["seuil_blocage"] / 2, 6)
    LOG.info("Modele champion v%s (%s, variables %s) charge. "
             "Seuil blocage=%.4f, seuil revue=%.4f",
             etat["version_modele"], etat["modele_retenu"], etat["etiquette"],
             etat["seuil_blocage"], etat["seuil_revue"])
    return etat


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    ETAT.update(charger_etat())
    yield
    ETAT.clear()


app = FastAPI(title="Fraude-360 - service de scoring",
              description="Detection de la fraude transactionnelle en temps reel",
              version="1.0.0", lifespan=cycle_de_vie)


# =============================================================================
#  Schemas
# =============================================================================
class DemandeScoring(BaseModel):
    transaction_id: str = Field(..., description="Identifiant unique de la transaction")
    variables: Dict[str, Any] = Field(
        ..., description="Variables calculees par la couche de variables (table GOLD)")


class ReponseScoring(BaseModel):
    transaction_id: str
    score_modele: float
    score_regles: float
    regles_declenchees: List[str]
    decision: str
    motif: str
    seuil_blocage: float
    version_modele: Optional[int]
    latence_ms: float


# =============================================================================
#  Logique de decision
# =============================================================================
def decider(score_modele: float, score_regles: float, regles: List[str],
            etat: Dict[str, Any]) -> tuple[str, str]:
    graves = {r.identifiant for r in etat["moteur"].regles if r.gravite == "elevee"}
    if score_modele >= etat["seuil_blocage"]:
        return "BLOQUER", f"score du modele {score_modele:.4f} >= seuil {etat['seuil_blocage']:.4f}"
    if regles and set(regles) & graves and score_regles >= 0.5:
        return "BLOQUER", f"regle(s) de gravite elevee : {', '.join(sorted(set(regles) & graves))}"
    if score_modele >= etat["seuil_revue"] or score_regles >= 0.35:
        return "REVOIR", "zone grise : verification humaine demandee"
    return "APPROUVER", "risque faible"


def scorer(demande: DemandeScoring, etat: Dict[str, Any]) -> ReponseScoring:
    t0 = time.perf_counter()
    ligne = pd.DataFrame([demande.variables])
    manquantes = [v for v in etat["variables"] if v not in ligne.columns]
    for v in manquantes:
        ligne[v] = None
    try:
        score_modele = float(etat["modele"].predict_proba(ligne[etat["variables"]])[:, 1][0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Variables invalides : {exc}")

    res_regles = etat["moteur"].evaluer_une(demande.variables)
    decision, motif = decider(score_modele, res_regles["score_regles"],
                              res_regles["regles_declenchees"], etat)
    latence = (time.perf_counter() - t0) * 1000

    reponse = ReponseScoring(
        transaction_id=demande.transaction_id,
        score_modele=round(score_modele, 6),
        score_regles=round(res_regles["score_regles"], 4),
        regles_declenchees=res_regles["regles_declenchees"],
        decision=decision, motif=motif,
        seuil_blocage=etat["seuil_blocage"],
        version_modele=etat["version_modele"],
        latence_ms=round(latence, 2))

    with open(etat["journal"], "a", encoding="utf-8") as f:
        trace = reponse.model_dump()
        trace["horodatage_decision"] = datetime.now().isoformat()
        trace["montant"] = demande.variables.get("montant")
        trace["variables_manquantes"] = len(manquantes)
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return reponse


# =============================================================================
#  Points de terminaison
# =============================================================================
@app.get("/health")
def sante():
    return {"statut": "operationnel", "version_modele": ETAT.get("version_modele"),
            "etiquette_variables": ETAT.get("etiquette"),
            "demarre_a": ETAT.get("demarre_a")}


@app.get("/modele")
def info_modele():
    return {"version_registre": ETAT.get("version_modele"),
            "algorithme": ETAT.get("modele_retenu"),
            "seuil_blocage": ETAT.get("seuil_blocage"),
            "seuil_revue": ETAT.get("seuil_revue"),
            "nb_variables": len(ETAT.get("variables", []))}


@app.get("/regles")
def info_regles():
    return [{"identifiant": r.identifiant, "libelle": r.libelle, "gravite": r.gravite,
             "typologie_visee": r.typologie_visee, "active": r.active}
            for r in ETAT["moteur"].regles]


@app.post("/score", response_model=ReponseScoring)
def score(demande: DemandeScoring):
    return scorer(demande, ETAT)


@app.post("/score_lot", response_model=List[ReponseScoring])
def score_lot(demandes: List[DemandeScoring]):
    return [scorer(d, ETAT) for d in demandes]


@app.post("/recharger")
def recharger():
    """Recharge le modele champion sans redemarrer le service (apres promotion)."""
    ETAT.update(charger_etat())
    return {"statut": "recharge", "version_modele": ETAT["version_modele"]}


def demarrer() -> None:
    import uvicorn
    cfg = charger_config()
    uvicorn.run("fraude360.serving.api:app", host=cfg["service"]["hote"],
                port=int(cfg["service"]["port"]), log_level="warning")


if __name__ == "__main__":
    demarrer()
