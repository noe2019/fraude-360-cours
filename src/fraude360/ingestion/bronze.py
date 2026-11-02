"""ETAPE 01 - Ingestion / chargement (couche BRONZE).

Principe : copie fidele du flux source, sans transformation metier, enrichie de
metadonnees techniques permettant l'audit et le rejeu.

  - schema declare explicitement (jamais d'inference en production) ;
  - ingestion INCREMENTALE : un registre des fichiers deja traites evite les
    doubles chargements et permet de relancer le job sans effet de bord ;
  - sur Databricks, ce meme code s'execute avec Auto Loader
    (spark.readStream.format("cloudFiles")) : voir notebooks/01_ingestion.py.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from pyspark.sql import functions as F
from pyspark.sql.types import (BooleanType, IntegerType, StringType, StructField,
                               StructType)

from fraude360.config import Config, charger_config
from fraude360.utils.journal import chrono, ecrire_json, lire_json, obtenir_journal, titre
from fraude360.utils.spark import ecrire_table, lire_table, obtenir_spark

LOG = obtenir_journal("fraude360.ingestion")

# --- Schema contractuel du flux source ---------------------------------------
SCHEMA_SOURCE = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("horodatage", StringType(), True),
    StructField("carte_id", StringType(), True),
    StructField("client_id", StringType(), True),
    StructField("marchand_id", StringType(), True),
    StructField("categorie_marchand", StringType(), True),
    StructField("canal", StringType(), True),
    StructField("mode_saisie", StringType(), True),
    StructField("montant", StringType(), True),        # texte : le nettoyage suit
    StructField("devise", StringType(), True),
    StructField("pays", StringType(), True),
    StructField("latitude", StringType(), True),
    StructField("longitude", StringType(), True),
    StructField("authentification_3ds", StringType(), True),
    StructField("appareil_id", StringType(), True),
    StructField("adresse_ip", StringType(), True),
    StructField("duree_session_s", StringType(), True),
    StructField("est_fraude", IntegerType(), True),
    StructField("typologie_fraude", StringType(), True),
    StructField("horodatage_label", StringType(), True),
])


def _registre(cfg: Config) -> Path:
    return cfg.chemin("bronze") / "_registre_ingestion.json"


def fichiers_a_traiter(cfg: Config) -> List[Path]:
    dossier = cfg.chemin("atterrissage")
    tous = sorted(dossier.glob("*.csv"))
    reg = _registre(cfg)
    deja = set(lire_json(reg)["fichiers"]) if reg.exists() else set()
    return [f for f in tous if f.name not in deja]


def executer(cfg: Config | None = None, complet: bool = False) -> Path:
    cfg = cfg or charger_config()
    titre("ETAPE 01 - Ingestion vers la couche BRONZE")
    spark = obtenir_spark(cfg)

    cible = cfg.chemin_table("bronze", "transactions")
    reg_path = _registre(cfg)
    if complet and reg_path.exists():
        reg_path.unlink()

    nouveaux = fichiers_a_traiter(cfg)
    if not nouveaux:
        LOG.info("Aucun nouveau fichier a ingerer. Bronze deja a jour.")
        return cible

    LOG.info("%d nouveau(x) fichier(s) detecte(s) dans la zone d'atterrissage.",
             len(nouveaux))

    with chrono(LOG, "Lecture des fichiers sources"):
        df = (spark.read
              .option("header", True)
              .option("mode", "PERMISSIVE")
              .option("columnNameOfCorruptRecord", "_enregistrement_corrompu")
              .schema(SCHEMA_SOURCE)
              .csv([str(f) for f in nouveaux]))

        df = (df
              .withColumn("_fichier_source", F.input_file_name())
              .withColumn("_horodatage_ingestion", F.current_timestamp())
              .withColumn("_lot_id", F.lit(datetime.now().strftime("%Y%m%d%H%M%S")))
              .withColumn("_date_partition",
                          F.to_date(F.substring(F.col("horodatage"), 1, 10))))

    mode = "overwrite" if (complet or not cible.exists()) else "append"
    with chrono(LOG, f"Ecriture BRONZE (mode {mode})"):
        ecrire_table(df, cible, cfg, mode=mode, partitions=["_date_partition"])

    total = lire_table(spark, cible, cfg).count()
    LOG.info("BRONZE : %d lignes au total (%d ingerees dans ce lot).",
             total, df.count())

    deja = set(lire_json(reg_path)["fichiers"]) if reg_path.exists() else set()
    deja.update(f.name for f in nouveaux)
    ecrire_json(reg_path, {"fichiers": sorted(deja),
                           "derniere_execution": datetime.now().isoformat(),
                           "lignes_totales": total})
    return cible


if __name__ == "__main__":
    executer()
