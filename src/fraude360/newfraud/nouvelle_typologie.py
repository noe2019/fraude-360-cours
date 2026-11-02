"""ETAPE 10 - Prise en charge d'un nouveau cas de fraude.

Scenario joue par le prototype, calque sur un incident reel de l'industrie :

  J+0   Une nouvelle typologie apparait : l'arnaque au virement instantane.
        Le fraudeur ne vole pas la carte, il manipule le membre par telephone
        pour qu'il envoie lui-meme des virements Interac. Consequence : appareil
        connu, geolocalisation habituelle, heure normale, aucune velocite
        anormale. Toutes les variables de la version 1 voient une transaction
        parfaitement legitime.

  J+1   La surveillance leve l'alerte : les dollars non detectes explosent alors
        que le PSI des variables historiques bouge a peine. C'est la signature
        d'un angle mort, pas d'une derive.

  J+2   MITIGATION IMMEDIATE : une regle experte est deployee dans le moteur de
        regles (quelques heures), pour arreter l'hemorragie.

  J+7   REPONSE STRUCTURELLE : de nouvelles variables sont ajoutees a la couche
        GOLD (nouveau beneficiaire, cumul de virements sortants), le modele est
        reentraine (etape 11), teste en A/B, puis promu s'il gagne.

Ce module joue les quatre temps et produit un rapport d'incident.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.generation.simulateur import bruiter_etiquettes, degrader
from fraude360.utils.journal import chrono, ecrire_json, lire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.nouvelle_fraude")


# =============================================================================
#  1. Injection de la nouvelle typologie dans le flux
# =============================================================================
def generer_periode(cfg: Config) -> int:
    """Genere `nouvelle_fraude.jours_supplementaires` jours de flux contenant la
    nouvelle typologie, et les depose dans la zone d'atterrissage."""
    from fraude360.generation import simulateur

    nf = cfg["nouvelle_fraude"]
    g = cfg["generation"]
    jour_debut = g["nb_jours"]
    nb_jours = nf["jours_supplementaires"]
    rng = np.random.default_rng(cfg.seed + 777)

    ref_dir = cfg.chemin("racine_donnees") / "referentiels"
    ref = simulateur.Referentiels(
        clients=pd.read_csv(ref_dir / "clients.csv"),
        cartes=pd.read_csv(ref_dir / "cartes.csv"),
        marchands=pd.read_csv(ref_dir / "marchands.csv"))

    n_tx = nf["nb_transactions"]
    n_fraudes = int(n_tx * g["taux_fraude_cible"])
    n_nouvelles = int(n_fraudes * nf["part_dans_la_fraude"])
    n_anciennes = n_fraudes - n_nouvelles

    LOG.info("Nouvelle periode : %d jours, %d transactions, %d fraudes "
             "dont %d de la typologie '%s'.",
             nb_jours, n_tx, n_fraudes, n_nouvelles, nf["nom"])

    legit = simulateur.generer_legitimes(cfg, ref, rng, n_tx - n_fraudes,
                                         jour_debut, nb_jours)
    anciennes = simulateur.generer_fraudes(cfg, ref, rng, n_anciennes, jour_debut,
                                           nb_jours, g["typologies"])
    nouvelles = _generer_arnaque_virement(cfg, ref, rng, n_nouvelles, jour_debut, nb_jours)

    df = pd.concat([legit, anciennes, nouvelles], ignore_index=True)
    df = df.sort_values("horodatage").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TX{jour_debut:03d}{i:09d}" for i in range(len(df))])

    delai = int(g["delai_labellisation_jours"])
    df["horodatage_label"] = pd.NaT
    m = df["est_fraude"] == 1
    df.loc[m, "horodatage_label"] = df.loc[m, "horodatage"] + pd.to_timedelta(
        rng.integers(1, delai * 2, m.sum()), unit="D")

    df = bruiter_etiquettes(df, rng)
    df = degrader(df, rng)
    dossier = cfg.chemin("atterrissage")
    df["_jour"] = pd.to_datetime(df["horodatage"]).dt.date
    for jour, bloc in df.groupby("_jour"):
        bloc.drop(columns=["_jour"]).to_csv(
            dossier / f"transactions_{jour}.csv", index=False, encoding="utf-8")
    LOG.info("%d transactions deposees dans la zone d'atterrissage.", len(df))
    return len(df)


def _generer_arnaque_virement(cfg: Config, ref, rng, n_cible: int,
                              jour_debut: int, nb_jours: int) -> pd.DataFrame:
    """Arnaque au virement instantane : la victime autorise elle-meme l'operation.

    Signature volontairement INVISIBLE pour les variables de la version 1 :
      - appareil habituel du membre (pas de nouvel appareil) ;
      - geolocalisation du domicile ;
      - heure de bureau ;
      - pas de rafale : 2 a 4 virements etales sur 24 a 48 heures.
    Ce qui la trahit : le beneficiaire est nouveau, et le cumul des virements
    sortants sur 24 h est anormal pour ce membre. D'ou les variables v2.
    """
    nom = cfg["nouvelle_fraude"]["nom"]
    t0 = pd.Timestamp(cfg["generation"]["date_debut"]) + pd.Timedelta(days=jour_debut)
    clients = ref.clients.set_index("client_id")
    lignes: List[dict] = []

    while len(lignes) < n_cible:
        i_carte = int(rng.integers(0, len(ref.cartes)))
        carte = ref.cartes.iloc[i_carte]
        cli = clients.loc[carte["client_id"]]
        depart = (t0 + pd.Timedelta(seconds=int(rng.random() * nb_jours * 86400))
                  ).normalize() + pd.Timedelta(hours=int(rng.integers(9, 19)))
        n = int(rng.integers(2, 5))
        benefs = ref.marchands.sample(n, replace=False,
                                      random_state=int(rng.integers(1e9)))
        # Montants eleves mais du meme ordre qu'un achat legitime important
        # (voyage, ameublement) : l'anomalie de montant seule ne suffit pas.
        montants = np.round(np.clip(rng.lognormal(5.7, 0.6, n), 90, 2200), 2)
        deltas = np.cumsum(rng.integers(1800, 36000, n))
        # Appareil habituel du membre : meme formule que le flux legitime.
        appareil = f"APP{(i_carte * 7 % 900000):07d}"
        for k in range(n):
            b = benefs.iloc[k]
            lignes.append(dict(
                horodatage=depart + pd.Timedelta(seconds=int(deltas[k])),
                carte_id=carte["carte_id"], client_id=carte["client_id"],
                marchand_id=b["marchand_id"], categorie_marchand=b["categorie"],
                canal="INTERAC", mode_saisie="ecom", montant=float(montants[k]),
                devise="CAD", pays="CA",
                latitude=round(float(cli["latitude_domicile"]) + rng.normal(0, 0.01), 5),
                longitude=round(float(cli["longitude_domicile"]) + rng.normal(0, 0.01), 5),
                authentification_3ds=True, appareil_id=appareil,
                adresse_ip=f"24.{rng.integers(0,255)}.{rng.integers(0,255)}."
                           f"{rng.integers(0,255)}",
                duree_session_s=int(rng.integers(120, 900)),
                est_fraude=1, typologie_fraude=nom))
    return pd.DataFrame(lignes[:n_cible])


# =============================================================================
#  2. Diagnostic : pourquoi le modele n'a-t-il rien vu ?
# =============================================================================
def diagnostiquer(cfg: Config) -> dict:
    from fraude360.modeling import donnees
    from fraude360.utils.spark import lire_table, obtenir_spark

    spark = obtenir_spark(cfg)
    dec = lire_table(spark, cfg.chemin_table("gold", "decisions"), cfg).toPandas()
    nom = cfg["nouvelle_fraude"]["nom"]

    par_typologie = (dec[dec.est_fraude == 1]
                     .groupby("typologie_fraude")
                     .agg(cas=("transaction_id", "count"),
                          detectees=("alerte", "sum"),
                          montant=("montant", "sum"),
                          score_moyen=("score_modele", "mean"))
                     .reset_index())
    par_typologie["taux_detection_pct"] = (100 * par_typologie.detectees
                                           / par_typologie.cas).round(1)
    par_typologie["montant_non_detecte"] = par_typologie.apply(
        lambda r: float(dec[(dec.typologie_fraude == r.typologie_fraude)
                            & (dec.est_fraude == 1) & (dec.alerte == 0)].montant.sum()), axis=1)

    nouvelle = par_typologie[par_typologie.typologie_fraude == nom]
    anciennes = par_typologie[par_typologie.typologie_fraude != nom]

    diag = {
        "typologie": nom,
        "cas_observes": int(nouvelle.cas.iloc[0]) if len(nouvelle) else 0,
        "taux_detection_nouvelle_pct": float(nouvelle.taux_detection_pct.iloc[0])
                                       if len(nouvelle) else 0.0,
        "taux_detection_typologies_connues_pct": round(
            float(100 * anciennes.detectees.sum() / max(anciennes.cas.sum(), 1)), 1),
        "score_moyen_nouvelle": round(float(nouvelle.score_moyen.iloc[0]), 5)
                                if len(nouvelle) else 0.0,
        "montant_non_detecte": round(float(nouvelle.montant_non_detecte.iloc[0]), 2)
                               if len(nouvelle) else 0.0,
        "detail_par_typologie": par_typologie.to_dict("records"),
        "cause_racine": (
            "Les variables de la version 1 decrivent le comportement de la CARTE "
            "(velocite, geographie, appareil). Or dans cette typologie la victime "
            "agit elle-meme depuis son appareil habituel : aucune de ces variables "
            "ne s'ecarte de la normale. Le modele n'a pas derive, il est aveugle : "
            "l'information discriminante n'existe pas dans son espace de variables."),
        "reponse": [
            "Immediat (heures) : regle experte R11 sur le couple "
            "nouveau beneficiaire x cumul de virements sortants.",
            "Court terme (jours) : six nouvelles variables de virement ajoutees "
            "a la couche GOLD (jeu de variables v2).",
            "Structurel (semaine) : reentrainement, test A/B contre le champion, "
            "promotion conditionnelle.",
        ],
    }
    return diag


# =============================================================================
#  3. Orchestration de la prise en charge
# =============================================================================
def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("ETAPE 10 - Prise en charge d'une nouvelle typologie de fraude")

    from fraude360.features import gold
    from fraude360.ingestion import bronze
    from fraude360.monitoring import surveillance
    from fraude360.quality import silver
    from fraude360.serving import lot

    LOG.info("--- J+0 : la nouvelle typologie apparait dans le flux ---")
    with chrono(LOG, "Injection de la nouvelle periode"):
        n = generer_periode(cfg)

    LOG.info("--- Le pipeline quotidien tourne normalement ---")
    bronze.executer(cfg)                    # ingestion incrementale
    silver.executer(cfg)
    gold.executer(cfg, version=1)           # toujours le jeu de variables v1
    lot.executer(cfg)                       # scoring avec le champion en place

    LOG.info("--- J+1 : la surveillance rend son verdict ---")
    rapport_surveillance = surveillance.executer(cfg)

    LOG.info("--- Diagnostic ---")
    diag = diagnostiquer(cfg)
    LOG.info("Detection des typologies connues : %.1f %%",
             diag["taux_detection_typologies_connues_pct"])
    LOG.info("Detection de la nouvelle typologie : %.1f %% (%d cas, %.0f $ perdus)",
             diag["taux_detection_nouvelle_pct"], diag["cas_observes"],
             diag["montant_non_detecte"])
    LOG.info("Cause racine : %s", diag["cause_racine"])

    LOG.info("--- J+2 : mitigation immediate par une regle experte ---")
    impact_regle = _mesurer_regle(cfg)
    LOG.info("Regle R11 : %d cas captes sur %d (%.0f %%), %d faux positifs, "
             "precision %.1f %%",
             impact_regle["fraudes_captees"], impact_regle["fraudes_nouvelle_typologie"],
             impact_regle["couverture_pct"], impact_regle["faux_positifs"],
             impact_regle["precision_pct"])

    LOG.info("--- J+7 : reponse structurelle, nouvelles variables (v2) ---")
    with chrono(LOG, "Reconstruction de la couche GOLD avec les variables v2"):
        spec = gold.executer(cfg, version=2)
    LOG.info("Jeu de variables v2 : %d variables (%d de plus qu'en v1).",
             spec["nb_variables"], spec["nb_variables"] - 44)

    rapport = {
        "horodatage": datetime.now().isoformat(),
        "transactions_ajoutees": n,
        "diagnostic": diag,
        "alertes_surveillance": rapport_surveillance["alertes"],
        "mitigation_immediate": impact_regle,
        "nouvelles_variables": [v for v in spec["variables_numeriques"]
                                if v.startswith(("est_virement", "nb_virements",
                                                 "montant_virements", "est_nouveau_benef",
                                                 "ratio_virement"))],
        "etape_suivante": "Reentrainement du modele sur le jeu de variables v2 (etape 11).",
    }
    ecrire_json(cfg.chemin("artefacts") / "incidents" / "rapport_incident.json", rapport)
    return rapport


def _mesurer_regle(cfg: Config) -> dict:
    """Mesure ce qu'aurait capte la regle R11 sur la periode ecoulee."""
    from fraude360.modeling import donnees
    from fraude360.features import gold as gold_mod
    from fraude360.serving.regles import regle_nouvelle_typologie
    from fraude360.utils.spark import lire_table, obtenir_spark

    # La regle a besoin des variables de virement : on les calcule a la volee.
    spark = obtenir_spark(cfg)
    silver_df = lire_table(spark, cfg.chemin_table("silver", "transactions"), cfg)
    enrichi = gold_mod.variables_virement(gold_mod.variables_de_base(silver_df))
    pdf = enrichi.select("transaction_id", "horodatage", "canal", "montant",
                         "est_fraude", "typologie_fraude", "est_nouveau_beneficiaire",
                         "nb_virements_client_24h").toPandas()
    pdf["horodatage"] = pd.to_datetime(pdf["horodatage"])
    debut = pdf.horodatage.max() - pd.Timedelta(days=cfg["nouvelle_fraude"]["jours_supplementaires"])
    recent = pdf[pdf.horodatage >= debut]

    regle = regle_nouvelle_typologie()
    m = regle.condition(recent).fillna(False).astype(bool)
    nom = cfg["nouvelle_fraude"]["nom"]
    cible = recent.typologie_fraude == nom
    captees = int((m & cible).sum())
    return {
        "identifiant": regle.identifiant,
        "libelle": regle.libelle,
        "transactions_touchees": int(m.sum()),
        "fraudes_nouvelle_typologie": int(cible.sum()),
        "fraudes_captees": captees,
        "couverture_pct": round(100 * captees / max(int(cible.sum()), 1), 1),
        "faux_positifs": int((m & (recent.est_fraude == 0)).sum()),
        "precision_pct": round(100 * int((m & (recent.est_fraude == 1)).sum())
                               / max(int(m.sum()), 1), 1),
        "montant_protege": round(float(recent.loc[m & cible, "montant"].sum()), 2),
        "delai_de_mise_en_oeuvre": "quelques heures (aucun reentrainement requis)",
    }


if __name__ == "__main__":
    executer()
