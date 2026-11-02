"""Analyse d'impact disparate du modele de fraude.

Pourquoi c'est necessaire. Le modele ne recoit ni age, ni sexe, ni origine :
il n'y a pas de discrimination directe possible. Cela ne suffit pas. Des
variables neutres en apparence peuvent etre correlees a un attribut protege --
le canal (les retraits au guichet ne servent pas la meme population que le
commerce electronique), la categorie de marchand, la geographie. Le seul moyen
de le savoir est de mesurer le taux de blocage par sous-groupe et de regarder
si l'ecart s'explique par le taux de fraude reel du sous-groupe.

Deux indicateurs, volontairement simples et lisibles :

  RATIO D'IMPACT (regle des quatre cinquiemes)
      taux de blocage du groupe / taux de blocage du groupe de reference.
      En deca de 0,80 ou au-dela de 1,25, l'ecart merite une justification.

  ECART DE TAUX DE FAUX POSITIFS
      Un ratio d'impact eleve est legitime si le groupe fraude reellement
      davantage. Ce qui ne l'est pas, c'est de bloquer plus souvent des
      transactions LEGITIMES dans un groupe que dans un autre. C'est
      l'indicateur qui compte, et c'est celui qu'on oublie.

Aucun seuil n'est bloquant ici : l'analyse produit un constat documente, que
la ligne de defense 2 arbitre. Un prototype qui declarerait l'equite
"validee" automatiquement serait moins credible, pas plus.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.utils.journal import ecrire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.gouvernance.equite")

SEUIL_BAS, SEUIL_HAUT = 0.80, 1.25
NB_MIN = 100  # sous ce volume, un taux n'est pas interpretable


def _table(d: pd.DataFrame, axe: str) -> pd.DataFrame:
    g = d.groupby(axe, observed=True).agg(
        nb=("bloquee", "size"),
        taux_blocage=("bloquee", "mean"),
        taux_fraude_reel=("est_fraude", "mean"),
    )
    # faux positifs : bloquees alors que legitimes, rapporte aux legitimes
    legit = d[d.est_fraude == 0]
    g["taux_faux_positifs"] = legit.groupby(axe, observed=True).bloquee.mean()
    # non detectees : fraudes non bloquees, rapporte aux fraudes
    fraudes = d[d.est_fraude == 1]
    g["taux_fraude_non_bloquee"] = 1 - fraudes.groupby(axe, observed=True).bloquee.mean()
    return g[g.nb >= NB_MIN].copy()


def analyser_axe(d: pd.DataFrame, axe: str) -> Dict[str, object]:
    g = _table(d, axe)
    if g.empty or len(g) < 2:
        return {"axe": axe, "exploitable": False,
                "motif": f"moins de deux sous-groupes d'au moins {NB_MIN} transactions"}
    # groupe de reference : le plus nombreux
    ref = g.nb.idxmax()
    base = g.loc[ref, "taux_blocage"] or 1e-9
    base_fp = g.loc[ref, "taux_faux_positifs"] or 1e-9
    g["ratio_impact"] = g.taux_blocage / base
    g["ratio_faux_positifs"] = g.taux_faux_positifs / base_fp

    signales = g[(g.ratio_impact < SEUIL_BAS) | (g.ratio_impact > SEUIL_HAUT)]
    # un ecart de blocage n'est preoccupant que s'il porte sur des LEGITIMES
    preoccupants = signales[(signales.ratio_faux_positifs > SEUIL_HAUT)]

    return {
        "axe": axe,
        "exploitable": True,
        "groupe_reference": str(ref),
        "sous_groupes": {
            str(k): {"nb": int(r.nb),
                     "taux_blocage_pct": round(100 * r.taux_blocage, 2),
                     "taux_fraude_reel_pct": round(100 * r.taux_fraude_reel, 2),
                     "taux_faux_positifs_pct": round(100 * r.taux_faux_positifs, 3),
                     "ratio_impact": round(r.ratio_impact, 3),
                     "ratio_faux_positifs": round(r.ratio_faux_positifs, 3)}
            for k, r in g.iterrows()},
        "hors_intervalle_quatre_cinquiemes": [str(k) for k in signales.index],
        "ecart_sur_transactions_legitimes": [str(k) for k in preoccupants.index],
    }


def executer(cfg: Config | None = None, axes: List[str] | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("GOUVERNANCE - Analyse d'impact disparate")
    import pandas as pd
    chemin = cfg.chemin_table("gold", "decisions")
    d = pd.read_parquet(chemin, columns=["canal", "categorie_marchand", "montant",
                                         "bloquee", "est_fraude"])
    d["bloquee"] = d.bloquee.astype(int)
    d["est_fraude"] = d.est_fraude.astype(int)
    d["tranche_montant"] = pd.cut(d.montant, [0, 50, 200, 500, 2000, 1e9],
                                  labels=["0-50", "50-200", "200-500",
                                          "500-2000", "2000+"])
    axes = axes or ["canal", "categorie_marchand", "tranche_montant"]

    resultats = [analyser_axe(d, a) for a in axes]
    preoccupants = {r["axe"]: r["ecart_sur_transactions_legitimes"]
                    for r in resultats
                    if r.get("exploitable") and r["ecart_sur_transactions_legitimes"]}

    rapport = {
        "nb_transactions": int(len(d)),
        "regle": f"ratio d'impact hors [{SEUIL_BAS}, {SEUIL_HAUT}] -> justification requise",
        "effectif_minimal_par_sous_groupe": NB_MIN,
        "axes": resultats,
        "sous_groupes_a_justifier": preoccupants,
        "conclusion": ("Aucun ecart sur transactions legitimes." if not preoccupants
                       else "Ecarts a justifier devant la ligne de defense 2."),
    }
    dossier = cfg.chemin("artefacts") / "gouvernance"
    dossier.mkdir(parents=True, exist_ok=True)
    ecrire_json(dossier / "rapport_equite.json", rapport)
    LOG.info("%s", rapport["conclusion"])
    return rapport
