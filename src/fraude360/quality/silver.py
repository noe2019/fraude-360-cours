"""ETAPE 02 - Nettoyage et controles de qualite (couche SILVER).

Regles appliquees :
  1. normalisation des types (montant francophone "12,50" -> 12.50, booleens,
     horodatages, geolocalisation) ;
  2. normalisation des chaines (espaces parasites, casse, valeurs inconnues) ;
  3. dedoublonnage sur la cle metier (transaction_id) en gardant la version la
     plus recemment ingeree ;
  4. sept controles de qualite explicites ; toute ligne qui echoue part en
     QUARANTAINE avec le motif du rejet -- rien n'est supprime silencieusement ;
  5. jonction avec les referentiels client / carte / marchand ;
  6. journal de qualite (data_quality) versionne, consommable dans Power BI.
"""
from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from fraude360.config import Config, charger_config
from fraude360.utils.journal import chrono, ecrire_json, obtenir_journal, titre
from fraude360.utils.spark import ecrire_table, lire_table, obtenir_spark

LOG = obtenir_journal("fraude360.qualite")


def normaliser(df: DataFrame) -> DataFrame:
    """Types et chaines : on rend les donnees exploitables sans juger leur validite."""
    return (df
            .withColumn("transaction_id", F.trim(F.col("transaction_id")))
            .withColumn("carte_id", F.upper(F.trim(F.col("carte_id"))))
            .withColumn("client_id", F.upper(F.trim(F.col("client_id"))))
            .withColumn("marchand_id", F.upper(F.trim(F.col("marchand_id"))))
            .withColumn("horodatage", F.to_timestamp("horodatage"))
            .withColumn("horodatage_label", F.to_timestamp("horodatage_label"))
            .withColumn("montant",
                        F.regexp_replace(F.trim(F.col("montant")), ",", ".").cast("double"))
            .withColumn("devise", F.upper(F.trim(F.col("devise"))))
            .withColumn("pays", F.upper(F.trim(F.col("pays"))))
            .withColumn("canal", F.upper(F.trim(F.col("canal"))))
            .withColumn("mode_saisie", F.lower(F.trim(F.col("mode_saisie"))))
            .withColumn("categorie_marchand",
                        F.coalesce(F.lower(F.trim(F.col("categorie_marchand"))),
                                   F.lit("inconnue")))
            .withColumn("latitude", F.col("latitude").cast("double"))
            .withColumn("longitude", F.col("longitude").cast("double"))
            .withColumn("authentification_3ds",
                        F.when(F.lower(F.col("authentification_3ds")).isin("true", "1", "vrai"),
                               F.lit(True)).otherwise(F.lit(False)))
            .withColumn("duree_session_s",
                        F.coalesce(F.col("duree_session_s").cast("int"), F.lit(0)))
            .withColumn("appareil_id",
                        F.when(F.trim(F.col("appareil_id")) == "", None)
                         .otherwise(F.trim(F.col("appareil_id"))))
            .withColumn("adresse_ip",
                        F.when(F.trim(F.col("adresse_ip")) == "", None)
                         .otherwise(F.trim(F.col("adresse_ip"))))
            .withColumn("est_fraude", F.coalesce(F.col("est_fraude"), F.lit(0)))
            .withColumn("typologie_fraude",
                        F.when(F.col("est_fraude") == 1,
                               F.coalesce(F.col("typologie_fraude"), F.lit("inconnue")))
                         .otherwise(F.lit("aucune"))))


def dedoublonner(df: DataFrame) -> DataFrame:
    """Une transaction = une ligne. On garde l'ingestion la plus recente."""
    w = Window.partitionBy("transaction_id").orderBy(F.col("_horodatage_ingestion").desc())
    return (df.withColumn("_rang", F.row_number().over(w))
              .filter(F.col("_rang") == 1)
              .drop("_rang"))


def controler(df: DataFrame, cfg: Config) -> DataFrame:
    """Ajoute la colonne `motif_rejet` (null = ligne conforme)."""
    q = cfg["qualite"]
    obligatoires = q["colonnes_obligatoires"]

    manque = None
    for c in obligatoires:
        cond = F.col(c).isNull()
        manque = cond if manque is None else (manque | cond)

    return df.withColumn(
        "motif_rejet",
        F.when(manque, F.lit("champ_obligatoire_manquant"))
         .when(F.col("montant").isNull(), F.lit("montant_illisible"))
         .when(F.col("montant") < q["montant_min"], F.lit("montant_negatif_ou_nul"))
         .when(F.col("montant") > q["montant_max"], F.lit("montant_hors_bornes"))
         .when(~F.col("devise").isin(q["devises_autorisees"]), F.lit("devise_non_autorisee"))
         .when(F.col("horodatage").isNull(), F.lit("horodatage_illisible"))
         .when(F.col("horodatage") > F.current_timestamp() + F.expr("INTERVAL 1 DAY"),
               F.lit("horodatage_dans_le_futur"))
         .when((F.col("latitude").isNull()) | (F.col("latitude").between(-90, 90) == False),
               F.lit("geolocalisation_invalide"))
         .otherwise(F.lit(None).cast("string")))


def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("ETAPE 02 - Nettoyage et controles de qualite (SILVER)")
    spark = obtenir_spark(cfg)

    bronze = lire_table(spark, cfg.chemin_table("bronze", "transactions"), cfg)
    n_bronze = bronze.count()

    with chrono(LOG, "Normalisation des types et des chaines"):
        df = normaliser(bronze)

    with chrono(LOG, "Dedoublonnage sur transaction_id"):
        df = dedoublonner(df)
        n_dedup = df.count()

    with chrono(LOG, "Application des controles de qualite"):
        df = controler(df, cfg)
        rejets = df.filter(F.col("motif_rejet").isNotNull())
        conformes = df.filter(F.col("motif_rejet").isNull()).drop("motif_rejet")
        n_rejets = rejets.count()

    taux_rejet = n_rejets / max(n_dedup, 1)
    detail = (rejets.groupBy("motif_rejet").count()
              .orderBy(F.col("count").desc()).collect())

    # --- jonction avec les referentiels --------------------------------------
    ref = cfg.chemin("racine_donnees") / "referentiels"
    clients = (spark.read.option("header", True).option("inferSchema", True)
               .csv(str(ref / "clients.csv"))
               .select("client_id", "age", "province", "segment",
                       F.col("latitude_domicile").cast("double"),
                       F.col("longitude_domicile").cast("double"),
                       F.col("anciennete_compte_jours").cast("int")))
    cartes = (spark.read.option("header", True).option("inferSchema", True)
              .csv(str(ref / "cartes.csv"))
              .select("carte_id", "type_carte", "reseau", F.col("limite").cast("double")))

    with chrono(LOG, "Jonction avec les referentiels client et carte"):
        silver = (conformes
                  .join(F.broadcast(clients), on="client_id", how="left")
                  .join(F.broadcast(cartes), on="carte_id", how="left")
                  .withColumn("_date_partition", F.to_date("horodatage")))

    cible = cfg.chemin_table("silver", "transactions")
    with chrono(LOG, "Ecriture SILVER"):
        ecrire_table(silver, cible, cfg, mode="overwrite", partitions=["_date_partition"])

    if n_rejets:
        with chrono(LOG, "Ecriture de la QUARANTAINE"):
            ecrire_table(rejets, cfg.chemin_table("quarantaine", "rejets"), cfg,
                         mode="overwrite")

    rapport = {
        "horodatage": datetime.now().isoformat(),
        "lignes_bronze": n_bronze,
        "lignes_apres_dedoublonnage": n_dedup,
        "doublons_supprimes": n_bronze - n_dedup,
        "lignes_rejetees": n_rejets,
        "taux_rejet": round(taux_rejet, 5),
        "lignes_silver": silver.count(),
        "detail_rejets": {r["motif_rejet"]: r["count"] for r in detail},
        "statut": "ECHEC" if taux_rejet > cfg["qualite"]["seuil_rejet_bloquant"] else "SUCCES",
    }
    ecrire_json(cfg.chemin("artefacts") / "qualite" / "rapport_qualite.json", rapport)

    LOG.info("Doublons supprimes : %d | Rejets : %d (%.2f %%) | Silver : %d lignes",
             rapport["doublons_supprimes"], n_rejets, 100 * taux_rejet,
             rapport["lignes_silver"])
    for motif, n in rapport["detail_rejets"].items():
        LOG.info("   rejet %-28s : %d", motif, n)

    if rapport["statut"] == "ECHEC":
        raise RuntimeError(
            f"Taux de rejet {taux_rejet:.2%} superieur au seuil bloquant "
            f"{cfg['qualite']['seuil_rejet_bloquant']:.2%} - lot refuse.")
    return rapport


if __name__ == "__main__":
    executer()
