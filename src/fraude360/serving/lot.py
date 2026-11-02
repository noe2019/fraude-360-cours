"""Scoring par lots - production des decisions sur une periode.

Le service temps reel traite une transaction a la fois ; ce module applique
exactement la meme logique (meme modele, memes regles, memes seuils) a un
volume complet. Il produit la table `gold/decisions`, consommee par la
surveillance (etape 09) et par Power BI (etape 12).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.modeling import donnees, registre
from fraude360.serving.regles import MoteurDeRegles, regle_nouvelle_typologie
from fraude360.utils.journal import chrono, obtenir_journal, lire_json, titre
from fraude360.utils.spark import ecrire_table, obtenir_spark

LOG = obtenir_journal("fraude360.scoring_lot")


def scorer_dataframe(df: pd.DataFrame, cfg: Config,
                     avec_regle_nouvelle: bool | None = None) -> pd.DataFrame:
    resume = lire_json(cfg.chemin("artefacts") / "modele" / "resume_courant.json")
    variables = resume["variables_numeriques"] + resume["variables_categorielles"]
    seuil = cfg["service"]["seuil_blocage"] or resume["seuil_decision"]
    seuil_revue = round(seuil / 2, 6)

    modele = registre.charger_par_alias(registre.ALIAS_CHAMPION, cfg)
    moteur = MoteurDeRegles()
    if avec_regle_nouvelle is None:
        avec_regle_nouvelle = resume.get("etiquette", "v1") != "v1"
    if avec_regle_nouvelle:
        moteur.ajouter(regle_nouvelle_typologie())

    manquantes = [v for v in variables if v not in df.columns]
    for v in manquantes:
        df[v] = np.nan
    if manquantes:
        LOG.warning("%d variable(s) absente(s) du lot, imputee(s) : %s",
                    len(manquantes), ", ".join(manquantes[:5]))

    with chrono(LOG, f"Scoring de {len(df):,} transactions"):
        scores = modele.predict_proba(df[variables])[:, 1]
        regles = moteur.evaluer(df)

    graves = {r.identifiant for r in moteur.regles if r.gravite == "elevee"}
    a_regle_grave = regles["regles_declenchees"].map(
        lambda s: bool(set(s.split(",")) & graves) if s else False)

    decision = np.where(scores >= seuil, "BLOQUER",
                np.where(a_regle_grave & (regles["score_regles"].values >= 0.5), "BLOQUER",
                np.where((scores >= seuil_revue) | (regles["score_regles"].values >= 0.35),
                         "REVOIR", "APPROUVER")))

    sortie = pd.DataFrame({
        "transaction_id": df["transaction_id"].values,
        "horodatage": df["horodatage"].values,
        "carte_id": df["carte_id"].values,
        "client_id": df["client_id"].values,
        "marchand_id": df["marchand_id"].values,
        "categorie_marchand": df["categorie_marchand"].values,
        "canal": df["canal"].values,
        "montant": df["montant"].values,
        "score_modele": scores,
        "score_regles": regles["score_regles"].values,
        "regles_declenchees": regles["regles_declenchees"].values,
        "decision": decision,
        "alerte": np.isin(decision, ["BLOQUER", "REVOIR"]).astype(int),
        "bloquee": (decision == "BLOQUER").astype(int),
        "est_fraude": df["est_fraude"].values,
        "typologie_fraude": df["typologie_fraude"].values,
        "version_modele": resume.get("version_registre"),
        "etiquette_variables": resume.get("etiquette", "v1"),
        "seuil_blocage": seuil,
        "horodatage_scoring": datetime.now(),
    })
    return sortie


def executer(cfg: Config | None = None, depuis_jour: int | None = None) -> str:
    cfg = cfg or charger_config()
    titre("ETAPE 08b - Scoring par lots de la periode de production")
    pdf = donnees.charger_gold(cfg)
    d = cfg["decoupage"]
    debut = depuis_jour if depuis_jour is not None else (
        d["jours_entrainement"] + d["jours_validation"])
    lot = pdf[pdf.jour_index >= debut].copy()

    sortie = scorer_dataframe(lot, cfg)
    spark = obtenir_spark(cfg)
    sdf = spark.createDataFrame(sortie)
    cible = cfg.chemin_table("gold", "decisions")
    ecrire_table(sdf, cible, cfg, mode="overwrite")

    taux_alerte = 100 * sortie.alerte.mean()
    detection = 100 * sortie.loc[sortie.est_fraude == 1, "alerte"].mean()
    LOG.info("Decisions ecrites : %d lignes | taux d'alerte %.2f %% | "
             "detection %.1f %% des fraudes | %d blocages",
             len(sortie), taux_alerte, detection, int(sortie.bloquee.sum()))
    return str(cible)


if __name__ == "__main__":
    executer()
