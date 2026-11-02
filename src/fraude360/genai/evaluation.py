"""ETAPE 13 - Evaluation du triage : garde-fous deterministes + LLM-as-judge.

Un module de langage qu'on ne mesure pas est un module qu'on ne peut pas
deployer. Deux niveaux de mesure, dans cet ordre :

  1. GARDE-FOUS DETERMINISTES -- calculables sans second modele, donc fiables.
     - taux de JSON valide
     - taux d'action appartenant au vocabulaire autorise
     - taux de citation d'une politique EXISTANTE (detection d'invention)
     - taux de citation d'une politique effectivement RECUPEREE pour cette alerte
       (l'ecart avec le precedent mesure l'ancrage : citer une vraie politique
       qu'on ne nous a pas donnee reste une invention)

  2. LLM-AS-JUDGE -- un second appel note l'ancrage et l'utilite de 1 a 5.
     Il est faillible et corrélé au generateur : il ne remplace pas les
     garde-fous, il les complete sur ce qu'ils ne voient pas (la note est-elle
     lisible et actionnable ?). C'est pourquoi les deux sont rapportes
     separement et jamais fusionnes en un score unique.

Sortie : artifacts/genai/ (rapport JSON, notes CSV, figure) + table pour Power BI.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.genai import triage
from fraude360.genai.client import obtenir_client
from fraude360.utils.journal import chrono, ecrire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.genai.evaluation")

NB_MIN_PAR_STRATE = 5   # effectif minimal pour qu'une action entre au test d'ordre

GABARIT_JUGE = """Tu evalues la note de triage d'un analyste fraude.

EXTRAITS DE POLITIQUE DISPONIBLES : {passages}

NOTE A EVALUER
typologie : {typologie}
action : {action}
justification : {justification}
politiques citees : {citees}

Note deux criteres de 1 a 5 :
- ancrage : la justification decoule-t-elle des politiques citees ?
- utilite : un enqueteur peut-il agir immediatement a partir de cette note ?

Reponds en JSON strict : {{"ancrage": <1-5>, "utilite": <1-5>}}"""


def garde_fous(notes: pd.DataFrame) -> Dict[str, float]:
    n = len(notes)
    if n == 0:
        return {}
    return {
        "nb_alertes_triees": n,
        "taux_json_valide_pct": round(100 * notes.json_valide.mean(), 1),
        "taux_action_valide_pct": round(100 * notes.action_valide.mean(), 1),
        "taux_citation_existante_pct": round(100 * notes.citations_valides.mean(), 1),
        "taux_citation_recuperee_pct": round(100 * notes.citations_recuperees.mean(), 1),
        "similarite_moyenne_rag": round(float(notes.similarite_max.mean()), 4),
    }


def contre_verite_terrain(notes: pd.DataFrame) -> Dict[str, object]:
    """Le triage separe-t-il reellement la fraude ? Mesure sans second modele.

    C'est l'evaluation la plus solide des trois : elle ne depend ni du format de
    sortie ni du jugement d'un modele, seulement de l'etiquette reelle. Si le
    taux de fraude ne croit pas de MISE_EN_ATTENTE vers BLOQUER, le triage
    n'ordonne rien et n'a aucune valeur operationnelle.
    """
    if "est_fraude" not in notes.columns:
        return {"disponible": False}
    g = notes.groupby("action").agg(
        nb=("est_fraude", "size"),
        taux_fraude_pct=("est_fraude", lambda s: round(100 * s.mean(), 1)),
        montant=("montant", "sum"))
    ordre = ["MISE_EN_ATTENTE", "REVUE_MANUELLE", "BLOQUER"]
    # Un taux calcule sur une poignee d'observations n'est pas comparable a un
    # taux calcule sur trente : les strates trop peu peuplees sont exclues du
    # test d'ordonnancement et signalees, plutot que de le faire echouer ou
    # reussir au hasard.
    n_min = NB_MIN_PAR_STRATE
    retenues = [a for a in ordre if a in g.index and int(g.loc[a, "nb"]) >= n_min]
    ignorees = [a for a in ordre if a in g.index and int(g.loc[a, "nb"]) < n_min]
    taux = [float(g.loc[a, "taux_fraude_pct"]) for a in retenues]
    return {
        "disponible": True,
        "par_action": {a: {"nb": int(g.loc[a, "nb"]),
                           "taux_fraude_pct": float(g.loc[a, "taux_fraude_pct"]),
                           "montant": round(float(g.loc[a, "montant"]), 2)}
                       for a in g.index},
        "ordonnancement_respecte": (taux == sorted(taux)) if len(taux) >= 2 else None,
        "actions_testees": retenues,
        "actions_ignorees_effectif_insuffisant": ignorees,
        "effectif_minimal_par_action": n_min,
        "taux_fraude_global_pct": round(100 * float(notes.est_fraude.mean()), 1),
    }


def juger(notes: pd.DataFrame, client, limite: int) -> pd.DataFrame:
    """Second appel : note l'ancrage et l'utilite. Silencieux en cas d'echec."""
    lignes: List[dict] = []
    for _, r in notes.head(limite).iterrows():
        invite = GABARIT_JUGE.format(
            passages=", ".join(r.passages_fournis) or "aucun",
            typologie=r.typologie, action=r.action,
            justification=r.justification,
            citees=", ".join(r.politiques_citees) or "aucune")
        try:
            brut = client.completer(invite)
            m = re.search(r"\{.*\}", brut, re.S)
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
        anc, uti = d.get("ancrage"), d.get("utilite")
        if isinstance(anc, (int, float)) and isinstance(uti, (int, float)):
            lignes.append({"transaction_id": r.transaction_id,
                           "ancrage": float(anc), "utilite": float(uti)})
    return pd.DataFrame(lignes)


def _figure(notes: pd.DataFrame, gf: Dict[str, float], chemin) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    cles = ["taux_json_valide_pct", "taux_action_valide_pct",
            "taux_citation_existante_pct", "taux_citation_recuperee_pct"]
    libelles = ["JSON\nvalide", "Action\nautorisee", "Politique\nexistante",
                "Politique\nrecuperee"]
    axes[0].bar(libelles, [gf.get(c, 0) for c in cles], color="#1f4e79")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("%")
    axes[0].set_title("Garde-fous deterministes")
    axes[0].axhline(95, ls="--", c="crimson", lw=1)

    ordre = notes.action.value_counts()
    axes[1].barh(ordre.index[::-1], ordre.values[::-1], color="#2e7d32")
    axes[1].set_title("Actions recommandees")
    axes[1].set_xlabel("nombre d'alertes")

    axes[2].scatter(notes.score_modele, notes.similarite_max, s=18, alpha=0.6,
                    color="#b8860b")
    axes[2].set_xlabel("score du modele")
    axes[2].set_ylabel("similarite RAG")
    axes[2].set_title("Ancrage documentaire selon le score")

    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)


def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("ETAPE 13 - Triage des alertes par modele de langage et evaluation")
    g = cfg.get("genai", {}) or {}
    dossier = cfg.chemin("artefacts") / "genai"
    dossier.mkdir(parents=True, exist_ok=True)

    notes = triage.executer(cfg)
    if notes.empty:
        LOG.warning("Aucune alerte bloquee a trier.")
        return {"statut": "AUCUNE_ALERTE"}

    gf = garde_fous(notes)
    LOG.info("JSON valide %.1f %% | politique existante %.1f %% | recuperee %.1f %%",
             gf["taux_json_valide_pct"], gf["taux_citation_existante_pct"],
             gf["taux_citation_recuperee_pct"])

    jugements = pd.DataFrame()
    if g.get("activer_juge", True):
        with chrono(LOG, "Evaluation LLM-as-judge"):
            jugements = juger(notes, obtenir_client(g), int(g.get("nb_juges", 25)))

    verite = contre_verite_terrain(notes)
    if verite.get("disponible"):
        ordo = verite["ordonnancement_respecte"]
        LOG.info("Ordonnancement contre la verite terrain : %s  (actions testees : %s)",
                 {True: "respecte", False: "NON respecte", None: "non concluant"}[ordo],
                 ", ".join(verite["actions_testees"]) or "aucune")

    rapport = {
        "client_generateur": notes.attrs.get("client", "inconnu"),
        "garde_fous": gf,
        "contre_verite_terrain": verite,
        "repartition_actions": notes.action.value_counts().to_dict(),
        "repartition_confiance": notes.confiance.value_counts().to_dict(),
        "juge": ({"nb_evalues": len(jugements),
                  "ancrage_moyen": round(float(jugements.ancrage.mean()), 2),
                  "utilite_moyenne": round(float(jugements.utilite.mean()), 2)}
                 if not jugements.empty else {"nb_evalues": 0}),
        "montant_total_alertes": round(float(notes.montant.sum()), 2),
    }

    notes.drop(columns=["passages_fournis"]).to_csv(
        dossier / "notes_triage.csv", index=False, encoding="utf-8")
    if not jugements.empty:
        jugements.to_csv(dossier / "jugements.csv", index=False, encoding="utf-8")
    _figure(notes, gf, dossier / "30_triage_genai.png")
    ecrire_json(dossier / "rapport_triage.json", rapport)

    # table de faits pour Power BI, meme convention que bi/export.py
    faits = notes[["transaction_id", "typologie", "action", "confiance",
                   "score_modele", "montant", "similarite_max"]].copy()
    faits["politiques_citees"] = notes.politiques_citees.map(lambda l: ",".join(l))
    faits.to_csv(cfg.chemin("exports_bi") / "fait_triage_genai.csv",
                 index=False, encoding="utf-8-sig")

    LOG.info("Rapport ecrit dans %s", dossier)
    return rapport
