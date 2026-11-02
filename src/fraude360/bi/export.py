"""ETAPE 12 - Alimentation du tableau de bord Power BI.

Power BI ne doit jamais lire la couche transactionnelle brute : on lui expose un
modele en etoile deja agrege, stable et documente. C'est ce que produit ce
module, dans powerbi/data/.

  dim_date.csv                calendrier (indispensable au time intelligence DAX)
  dim_typologie.csv           libelles et descriptions des typologies de fraude
  fait_kpi_quotidien.csv      grain = jour : volumes, alertes, dollars, detection
  fait_alertes.csv            grain = transaction alertee (file d'enquete)
  fait_typologie_quotidien.csv grain = jour x typologie
  fait_derive.csv             grain = jour x variable (PSI, KS)
  fait_performance_modele.csv grain = version de modele (metriques de test)
  fait_performance_regles.csv grain = regle (precision, couverture)
  fait_ab_test.csv            grain = experience x bras

Sur Databricks, ces memes tables sont exposees en Delta dans le schema `gold` et
consommees en Direct Lake : aucun export de fichier n'est necessaire.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.utils.journal import chrono, lire_json, obtenir_journal, titre
from fraude360.utils.spark import lire_table, obtenir_spark

LOG = obtenir_journal("fraude360.bi")

DESCRIPTIONS = {
    "carte_perdue_volee": "Carte physique volee, rafale d'achats sans contact",
    "achat_en_ligne_cnp": "Achat en ligne sans presentation de la carte, sans 3DS",
    "usurpation_de_compte": "Prise de controle du compte, virements nocturnes",
    "test_de_carte": "Micro-montants repetes pour valider un numero de carte",
    "arnaque_virement_instantane": "Membre manipule, virements Interac autorises par la victime",
    "contestation_non_fondee": "Contestation d'un membre sur une transaction en realite legitime",
    "aucune": "Transaction legitime",
}


def _ecrire(df: pd.DataFrame, dossier: Path, nom: str) -> int:
    df.to_csv(dossier / nom, index=False, encoding="utf-8-sig")
    LOG.info("   %-34s %7d lignes", nom, len(df))
    return len(df)


def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("ETAPE 12 - Exports pour le tableau de bord Power BI")
    dossier = cfg.chemin("exports_bi")
    spark = obtenir_spark(cfg)

    with chrono(LOG, "Lecture des tables GOLD"):
        dec = lire_table(spark, cfg.chemin_table("gold", "decisions"), cfg).toPandas()
    dec["horodatage"] = pd.to_datetime(dec["horodatage"])
    dec["date"] = dec["horodatage"].dt.date

    tailles = {}

    # --- dimensions ----------------------------------------------------------
    dates = pd.date_range(dec.date.min(), dec.date.max(), freq="D")
    mois_fr = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet",
               "aout", "septembre", "octobre", "novembre", "decembre"]
    jours_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    dim_date = pd.DataFrame({
        "date": dates.date,
        "annee": dates.year,
        "mois": dates.month,
        "nom_mois": [mois_fr[m - 1] for m in dates.month],
        "annee_mois": dates.strftime("%Y-%m"),
        "semaine": dates.isocalendar().week.values,
        "jour_semaine": [jours_fr[j] for j in dates.dayofweek],
        "est_fin_semaine": (dates.dayofweek >= 5).astype(int),
        "numero_jour": range(len(dates)),
    })
    tailles["dim_date.csv"] = _ecrire(dim_date, dossier, "dim_date.csv")

    typologies = sorted(set(dec.typologie_fraude.unique()) | set(DESCRIPTIONS))
    dim_typo = pd.DataFrame({
        "typologie": typologies,
        "description": [DESCRIPTIONS.get(t, "Typologie non documentee") for t in typologies],
        "connue_du_modele_v1": [1 if t in cfg["generation"]["typologies"] else 0
                                for t in typologies],
    })
    tailles["dim_typologie.csv"] = _ecrire(dim_typo, dossier, "dim_typologie.csv")

    # --- fait : indicateurs quotidiens --------------------------------------
    q = (dec.groupby("date")
         .agg(transactions=("transaction_id", "count"),
              montant_total=("montant", "sum"),
              alertes=("alerte", "sum"),
              blocages=("bloquee", "sum"),
              fraudes=("est_fraude", "sum"),
              score_moyen=("score_modele", "mean"))
         .reset_index())
    detectees = (dec[dec.est_fraude == 1].groupby("date")["alerte"].sum()
                 .reindex(q.date).fillna(0).values)
    montant_fraude = (dec[dec.est_fraude == 1].groupby("date")["montant"].sum()
                      .reindex(q.date).fillna(0).values)
    montant_evite = (dec[(dec.est_fraude == 1) & (dec.alerte == 1)].groupby("date")["montant"]
                     .sum().reindex(q.date).fillna(0).values)
    q["fraudes_detectees"] = detectees
    q["montant_fraude"] = montant_fraude.round(2)
    q["montant_fraude_evite"] = montant_evite.round(2)
    q["montant_fraude_subi"] = (montant_fraude - montant_evite).round(2)
    q["faux_positifs"] = q.alertes - q.fraudes_detectees
    eco = cfg["modelisation"]["economie"]
    q["cout_revues_manuelles"] = (q.faux_positifs * eco["cout_faux_positif"]
                                  + q.fraudes_detectees * eco["cout_vrai_positif"]).round(2)
    q["cout_total"] = (q.montant_fraude_subi + q.cout_revues_manuelles).round(2)
    q["montant_total"] = q.montant_total.round(2)
    q["score_moyen"] = q.score_moyen.round(6)
    tailles["fait_kpi_quotidien.csv"] = _ecrire(q, dossier, "fait_kpi_quotidien.csv")

    # --- fait : file d'enquete (transactions alertees) -----------------------
    alertes = dec[dec.alerte == 1].copy()
    colonnes = ["transaction_id", "date", "horodatage", "carte_id", "client_id",
                "marchand_id", "categorie_marchand", "canal", "montant",
                "score_modele", "score_regles", "regles_declenchees", "decision",
                "est_fraude", "typologie_fraude", "version_modele"]
    alertes = alertes[colonnes].sort_values("score_modele", ascending=False)
    alertes["score_modele"] = alertes.score_modele.round(6)
    alertes["priorite"] = pd.cut(alertes.score_modele,
                                 bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
                                 labels=["4-faible", "3-moyenne", "2-haute", "1-critique"])
    alertes["statut_enquete"] = np.where(alertes.est_fraude == 1, "fraude confirmee",
                                         "faux positif")
    tailles["fait_alertes.csv"] = _ecrire(alertes, dossier, "fait_alertes.csv")

    # --- fait : jour x typologie --------------------------------------------
    typo = (dec[dec.est_fraude == 1]
            .groupby(["date", "typologie_fraude"])
            .agg(cas=("transaction_id", "count"),
                 detectes=("alerte", "sum"),
                 montant=("montant", "sum"),
                 score_moyen=("score_modele", "mean"))
            .reset_index())
    typo["montant"] = typo.montant.round(2)
    typo["score_moyen"] = typo.score_moyen.round(6)
    typo["taux_detection"] = (typo.detectes / typo.cas).round(4)
    tailles["fait_typologie_quotidien.csv"] = _ecrire(
        typo, dossier, "fait_typologie_quotidien.csv")

    # --- fait : derive -------------------------------------------------------
    f_derive = cfg.chemin("artefacts") / "surveillance" / "derive_donnees.csv"
    if f_derive.exists():
        tailles["fait_derive.csv"] = _ecrire(pd.read_csv(f_derive), dossier,
                                             "fait_derive.csv")

    # --- fait : performance des modeles --------------------------------------
    lignes = []
    for etiquette in ("v1", "v2"):
        f = cfg.chemin("artefacts") / "modele" / f"resume_{etiquette}.json"
        if not f.exists():
            continue
        r = lire_json(f)
        m = r["metriques_test"]
        lignes.append({
            "version_modele": r["version_registre"], "jeu_variables": etiquette,
            "algorithme": r["modele_retenu"], "nb_variables": r["nb_variables"],
            "seuil_decision": round(r["seuil_decision"], 6),
            "pr_auc": round(m["pr_auc"], 4), "roc_auc": round(m["roc_auc"], 4),
            "precision": round(m["precision"], 4), "rappel": round(m["rappel"], 4),
            "f1": round(m["f1"], 4),
            "taux_alerte_pct": round(m["taux_alerte_pct"], 3),
            "montant_fraude_evite": m["montant_fraude_evite"],
            "montant_fraude_subi": m["montant_fraude_subi"],
            "cout_par_1000_tx": m["cout_par_1000_tx"],
            "economie_vs_sans_modele": r["economie_vs_sans_modele"],
        })
    if lignes:
        tailles["fait_performance_modele.csv"] = _ecrire(
            pd.DataFrame(lignes), dossier, "fait_performance_modele.csv")

    # --- fait : performance des regles ---------------------------------------
    f_regles = cfg.chemin("artefacts") / "abtest" / "performance_des_regles.csv"
    if f_regles.exists():
        tailles["fait_performance_regles.csv"] = _ecrire(
            pd.read_csv(f_regles), dossier, "fait_performance_regles.csv")

    # --- fait : tests A/B ----------------------------------------------------
    lignes = []
    for f in sorted((cfg.chemin("artefacts") / "abtest").glob("rapport_*.json")):
        r = lire_json(f)
        for bras, v in r["bras"].items():
            lignes.append({"experience": r["experience"], "bras": bras, **v,
                           "decision_experience": r["decision"],
                           "valeur_p": r["test_taux_detection"]["valeur_p"]})
    if lignes:
        tailles["fait_ab_test.csv"] = _ecrire(pd.DataFrame(lignes), dossier,
                                              "fait_ab_test.csv")

    # --- fait : alertes de surveillance --------------------------------------
    f_surv = cfg.chemin("artefacts") / "surveillance" / "rapport_surveillance.json"
    if f_surv.exists():
        r = lire_json(f_surv)
        if r["alertes"]:
            tailles["fait_alertes_surveillance.csv"] = _ecrire(
                pd.DataFrame(r["alertes"]), dossier, "fait_alertes_surveillance.csv")

    manifeste = {
        "genere_le": datetime.now().isoformat(),
        "dossier": str(dossier),
        "fichiers": tailles,
        "grain": {
            "fait_kpi_quotidien.csv": "1 ligne par jour",
            "fait_alertes.csv": "1 ligne par transaction alertee",
            "fait_typologie_quotidien.csv": "1 ligne par jour et typologie",
            "fait_derive.csv": "1 ligne par jour et variable surveillee",
        },
    }
    from fraude360.utils.journal import ecrire_json
    ecrire_json(dossier / "manifeste.json", manifeste)
    LOG.info("%d fichiers exportes vers %s", len(tailles), dossier)
    return manifeste


if __name__ == "__main__":
    executer()
