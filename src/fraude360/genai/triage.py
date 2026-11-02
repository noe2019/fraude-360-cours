"""Generation de la note de triage pour chaque alerte bloquee.

Ce que l'etape produit : pour chaque alerte, une note structuree (typologie
presumee, action recommandee, niveau de confiance, justification, politiques
citees) que l'enqueteur lit avant d'ouvrir le dossier.

Ce que l'etape ne fait pas : decider. La decision de blocage reste celle du
modele et des regles, a l'etape 08. Un modele de langage n'a pas sa place dans
une boucle de decision automatique sur de l'argent -- il n'est pas calibre, il
n'est pas stable, et on ne peut pas lui attribuer un cout d'erreur. Il sert ici
a reduire le temps de lecture d'un dossier, ce qui est mesurable et sans risque.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.genai import politiques as pol
from fraude360.genai.client import obtenir_client
from fraude360.serving.regles import regle_nouvelle_typologie, regles_de_base
from fraude360.utils.journal import chrono, obtenir_journal

LOG = obtenir_journal("fraude360.genai.triage")

ACTIONS = {"BLOQUER", "REVUE_MANUELLE", "MISE_EN_ATTENTE", "LIBERER"}

GABARIT = """Tu es analyste principal a la lutte a la fraude d'une cooperative financiere.
Tu rediges une note de triage destinee a un enqueteur. Tu t'appuies UNIQUEMENT sur
les extraits de politique fournis. Si les extraits ne couvrent pas le cas, tu le dis.

EXTRAITS DE POLITIQUE INTERNE
{extraits}

ALERTE A TRIER
transaction : {transaction_id}
montant : {montant}
canal : {canal}
score du modele : {score_modele}
seuil de blocage : {seuil}
regles declenchees : {regles}
Typologie visee : {typologie_regle}

Reponds en JSON strict, sans texte autour, avec exactement ces cles :
{{"typologie": "<typologie presumee>",
  "action": "<BLOQUER|REVUE_MANUELLE|MISE_EN_ATTENTE|LIBERER>",
  "confiance": "<faible|moyenne|elevee>",
  "justification": "<deux phrases maximum>",
  "politiques_citees": ["POL-XX"]}}"""


CANAL_MOTS = {
    "ECOM": "achat en ligne commerce electronique sans authentification 3-D Secure",
    "INTERAC": "virement Interac nouveau beneficiaire virement sortant",
    "POS": "achat de proximite terminal point de vente carte physique",
    "ATM": "retrait guichet automatique",
}


def _requete(alerte: dict, libelles_regles: Dict[str, str]) -> str:
    """Texte de recherche en langue naturelle.

    Les identifiants (R02, ECOM) ne portent aucun sens pour TF-IDF : on les
    remplace par le libelle de la regle et une glose du canal, qui partagent du
    vocabulaire avec les politiques. C'est ce qui fait passer la similarite
    moyenne de 0,10 a un niveau exploitable.
    """
    morceaux: List[str] = []
    for r in alerte.get("regles_declenchees", []):
        morceaux.append(libelles_regles.get(r, r))
    morceaux.append(CANAL_MOTS.get(str(alerte.get("canal", "")).upper(), ""))
    morceaux.append(str(alerte.get("categorie_marchand", "")))
    montant = float(alerte.get("montant", 0) or 0)
    morceaux.append("montant eleve" if montant > 400 else "petit montant")
    return " ".join(m for m in morceaux if m)


def _extraire_json(texte: str) -> Dict | None:
    """Recupere le premier objet JSON du texte. Les modeles bavardent."""
    m = re.search(r"\{.*\}", texte, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _normaliser(brut: Dict | None, alerte: dict, cites_valides: List[str]) -> Dict:
    """Applique les garde-fous : cles presentes, action connue, citations reelles."""
    d = brut or {}
    action = str(d.get("action", "")).strip().upper()
    citees = [c for c in d.get("politiques_citees", []) if c in pol.IDENTIFIANTS]
    return {
        "transaction_id": alerte["transaction_id"],
        "typologie": str(d.get("typologie", "indeterminee"))[:60],
        "action": action if action in ACTIONS else "REVUE_MANUELLE",
        "confiance": str(d.get("confiance", "faible")).lower()[:10],
        "justification": str(d.get("justification", ""))[:400],
        "politiques_citees": citees,
        "json_valide": brut is not None,
        "action_valide": action in ACTIONS,
        "citations_valides": bool(citees),
        "citations_recuperees": all(c in cites_valides for c in citees) if citees else False,
        "passages_fournis": cites_valides,
    }


COLONNES = ["transaction_id", "canal", "categorie_marchand", "montant",
            "score_modele", "score_regles", "regles_declenchees", "decision",
            "seuil_blocage", "est_fraude", "typologie_fraude"]


def charger_alertes(cfg: Config, limite: int) -> List[dict]:
    """Alertes a trier : table gold/decisions si presente, sinon le JSONL du service.

    La table gold porte le canal, la categorie du marchand et surtout `est_fraude`,
    l'etiquette reelle. Elle permet donc de mesurer le triage contre la verite
    terrain, ce que le journal du service ne permet pas.
    """
    table = cfg.chemin_table("gold", "decisions")
    if Path(table).exists():
        df = pd.read_parquet(table, columns=[c for c in COLONNES])
        df = df[df.decision.isin(["BLOQUER", "REVOIR"])]
        # Echantillon STRATIFIE sur le score, pas les N meilleurs : un jeu
        # d'evaluation compose uniquement des scores les plus eleves ne contient
        # qu'une classe et ne mesure aucun pouvoir de discrimination.
        df = df.assign(_strate=pd.qcut(df.score_modele, q=6, duplicates="drop",
                                       labels=False))
        par_strate = max(1, limite // max(1, df._strate.nunique()))
        df = (df.groupby("_strate", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), par_strate),
                                          random_state=cfg.seed))
                .drop(columns="_strate")
                .sort_values("score_modele", ascending=False))
        df["regles_declenchees"] = df.regles_declenchees.map(
            lambda v: [x for x in str(v).split(",") if x] if v is not None else [])
        return df.to_dict("records")

    fichier = cfg.chemin("artefacts") / "service" / "decisions.jsonl"
    if not fichier.exists():
        raise FileNotFoundError(
            f"Ni {table} ni {fichier} : executer d'abord `python -m fraude360 scorer`.")
    lignes = []
    with open(fichier, "r", encoding="utf-8") as f:
        for ligne in f:
            if not ligne.strip():
                continue
            try:
                d = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            if d.get("decision") in ("BLOQUER", "REVOIR"):
                lignes.append(d)
    lignes.sort(key=lambda d: d.get("score_modele", 0.0), reverse=True)
    return lignes[:limite]


def libelles_regles() -> Dict[str, str]:
    """Identifiant de regle -> libelle et typologie, pour enrichir la requete RAG."""
    toutes = regles_de_base() + [regle_nouvelle_typologie()]
    return {r.identifiant: f"{r.libelle} {r.typologie_visee.replace('_', ' ')}"
            for r in toutes}


def trier(cfg: Config, alertes: List[dict], index: pol.Index, client) -> pd.DataFrame:
    notes = []
    libelles = libelles_regles()
    for a in alertes:
        trouves = index.rechercher(_requete(a, libelles), k=3)
        extraits = "\n\n".join(
            f"[{p.identifiant}] {p.titre}\n{p.texte}" for p, _ in trouves)
        invite = GABARIT.format(
            extraits=extraits or "(aucun extrait pertinent)",
            transaction_id=a.get("transaction_id", ""),
            montant=a.get("montant", 0.0),
            canal=a.get("canal", "n.d."),
            score_modele=round(float(a.get("score_modele", 0.0)), 4),
            seuil=round(float(a.get("seuil_blocage", 0.0)), 4),
            regles=", ".join(a.get("regles_declenchees", [])) or "aucune",
            typologie_regle=a.get("typologie_regle", "indeterminee"))
        try:
            reponse = client.completer(invite)
        except Exception as exc:
            LOG.warning("Echec du modele sur %s : %s", a.get("transaction_id"), exc)
            reponse = ""
        note = _normaliser(_extraire_json(reponse), a,
                           [p.identifiant for p, _ in trouves])
        note["score_modele"] = float(a.get("score_modele", 0.0))
        note["montant"] = float(a.get("montant", 0.0))
        note["canal"] = a.get("canal", "n.d.")
        note["decision_pipeline"] = a.get("decision", "n.d.")
        note["similarite_max"] = round(trouves[0][1], 4) if trouves else 0.0
        # verite terrain, presente uniquement via la table gold
        if a.get("est_fraude") is not None:
            note["est_fraude"] = int(a["est_fraude"])
        notes.append(note)
    return pd.DataFrame(notes)


def executer(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or charger_config()
    g = cfg.get("genai", {}) or {}
    alertes = charger_alertes(cfg, int(g.get("nb_alertes", 60)))
    index = pol.Index()
    client = obtenir_client(g)
    with chrono(LOG, f"Triage de {len(alertes)} alertes par modele de langage"):
        notes = trier(cfg, alertes, index, client)
    notes.attrs["client"] = client.nom
    return notes
