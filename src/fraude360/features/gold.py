"""ETAPE 03 - Ingenierie des variables (couche GOLD).

C'est le coeur analytique : on transforme un journal de transactions en un jeu de
variables predictives. Toutes les variables sont calculees en PySpark avec des
fenetres temporelles glissantes ANTERIEURES a la transaction courante : aucune
information du futur n'entre dans le modele (pas de fuite de donnees).

Trois familles de variables :
  A. comportement de la carte    -> velocite, montants, diversite
  B. contexte de la transaction  -> heure, geographie, canal, appareil
  C. reputation du marchand      -> taux de fraude decale de 1 seconde a 30 jours

Version 2 (activee a l'etape 10, apres l'apparition d'une nouvelle typologie) :
  D. comportement de virement Interac -> nouveau beneficiaire, cumul sortant
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from fraude360.config import Config, charger_config
from fraude360.utils.journal import chrono, ecrire_json, obtenir_journal, titre
from fraude360.utils.spark import ecrire_table, lire_table, obtenir_spark

LOG = obtenir_journal("fraude360.variables")

H1, H24, J7, J30 = 3600, 86400, 604800, 2592000


def _fenetre(cle: str, debut: int, fin: int = 0) -> Window:
    return (Window.partitionBy(cle).orderBy("ts_unix").rangeBetween(debut, fin))


def variables_de_base(df: DataFrame) -> DataFrame:
    df = df.withColumn("ts_unix", F.col("horodatage").cast("long"))

    w_carte_1h = _fenetre("carte_id", -H1)
    w_carte_24h = _fenetre("carte_id", -H24)
    w_carte_7j = _fenetre("carte_id", -J7)
    w_client_24h = _fenetre("client_id", -H24)
    w_marchand_1h = _fenetre("marchand_id", -H1)
    # decalage d'une seconde : la transaction courante ne se voit pas elle-meme
    w_marchand_30j = _fenetre("marchand_id", -J30, -1)
    w_carte_prec = Window.partitionBy("carte_id").orderBy("ts_unix")
    w_carte_marchand = (Window.partitionBy("carte_id", "marchand_id")
                        .orderBy("ts_unix").rowsBetween(Window.unboundedPreceding, -1))
    w_client_appareil = (Window.partitionBy("client_id", "appareil_id")
                         .orderBy("ts_unix").rowsBetween(Window.unboundedPreceding, -1))

    df = (df
          # ---- A. comportement de la carte -----------------------------------
          .withColumn("nb_tx_carte_1h", F.count("*").over(w_carte_1h))
          .withColumn("nb_tx_carte_24h", F.count("*").over(w_carte_24h))
          .withColumn("nb_tx_carte_7j", F.count("*").over(w_carte_7j))
          .withColumn("montant_somme_carte_1h", F.sum("montant").over(w_carte_1h))
          .withColumn("montant_somme_carte_24h", F.sum("montant").over(w_carte_24h))
          .withColumn("montant_moyen_carte_7j", F.avg("montant").over(w_carte_7j))
          .withColumn("montant_ecart_type_carte_7j",
                      F.coalesce(F.stddev("montant").over(w_carte_7j), F.lit(0.0)))
          .withColumn("montant_max_carte_7j", F.max("montant").over(w_carte_7j))
          .withColumn("nb_marchands_distincts_carte_24h",
                      F.size(F.collect_set("marchand_id").over(w_carte_24h)))
          .withColumn("nb_categories_distinctes_carte_24h",
                      F.size(F.collect_set("categorie_marchand").over(w_carte_24h)))
          .withColumn("nb_pays_distincts_carte_24h",
                      F.size(F.collect_set("pays").over(w_carte_24h)))
          .withColumn("nb_micro_tx_carte_1h",
                      F.sum(F.when(F.col("montant") < 5, 1).otherwise(0)).over(w_carte_1h))
          .withColumn("nb_tx_client_24h", F.count("*").over(w_client_24h))
          # ---- B. contexte : temps, geographie, appareil ---------------------
          .withColumn("ts_precedent", F.lag("ts_unix").over(w_carte_prec))
          .withColumn("lat_precedente", F.lag("latitude").over(w_carte_prec))
          .withColumn("lon_precedente", F.lag("longitude").over(w_carte_prec))
          .withColumn("montant_precedent", F.lag("montant").over(w_carte_prec))
          .withColumn("secondes_depuis_tx_precedente",
                      F.coalesce(F.col("ts_unix") - F.col("ts_precedent"), F.lit(999999)))
          .withColumn("heure", F.hour("horodatage"))
          .withColumn("jour_semaine", F.dayofweek("horodatage"))
          .withColumn("est_nuit", F.when(F.hour("horodatage").between(0, 5), 1).otherwise(0))
          .withColumn("est_fin_semaine",
                      F.when(F.dayofweek("horodatage").isin(1, 7), 1).otherwise(0))
          .withColumn("montant_log", F.log1p(F.col("montant")))
          # ---- C. reputation du marchand -------------------------------------
          .withColumn("nb_tx_marchand_1h", F.count("*").over(w_marchand_1h))
          .withColumn("taux_fraude_marchand_30j",
                      F.coalesce(F.avg(F.col("est_fraude").cast("double")).over(w_marchand_30j),
                                 F.lit(0.0)))
          .withColumn("nb_tx_marchand_30j", F.count("*").over(w_marchand_30j))
          .withColumn("est_nouveau_marchand_carte",
                      F.when(F.count("*").over(w_carte_marchand) == 0, 1).otherwise(0))
          .withColumn("est_nouvel_appareil_client",
                      F.when(F.col("appareil_id").isNull(), 0)
                       .when(F.count("*").over(w_client_appareil) == 0, 1).otherwise(0))
          )

    # --- distances -----------------------------------------------------------
    def _haversine(lat1, lon1, lat2, lon2):
        a = (F.pow(F.sin(F.radians(lat2 - lat1) / 2), 2)
             + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2))
             * F.pow(F.sin(F.radians(lon2 - lon1) / 2), 2))
        return F.lit(6371.0) * 2 * F.asin(F.sqrt(F.least(a, F.lit(1.0))))

    df = (df
          .withColumn("distance_km_tx_precedente",
                      F.coalesce(_haversine(F.col("lat_precedente"), F.col("lon_precedente"),
                                            F.col("latitude"), F.col("longitude")), F.lit(0.0)))
          .withColumn("distance_domicile_km",
                      F.coalesce(_haversine(F.col("latitude_domicile"),
                                            F.col("longitude_domicile"),
                                            F.col("latitude"), F.col("longitude")),
                                 F.lit(0.0)))
          .withColumn("vitesse_kmh",
                      F.when(F.col("secondes_depuis_tx_precedente") > 0,
                             F.col("distance_km_tx_precedente")
                             / (F.col("secondes_depuis_tx_precedente") / 3600.0))
                       .otherwise(F.lit(0.0)))
          .withColumn("vitesse_kmh", F.least(F.col("vitesse_kmh"), F.lit(5000.0)))
          # --- ecarts au comportement habituel --------------------------------
          .withColumn("ratio_montant_moyenne_carte",
                      F.col("montant") / (F.coalesce(F.col("montant_moyen_carte_7j"),
                                                     F.col("montant")) + F.lit(1.0)))
          .withColumn("ecart_type_montant_carte",
                      (F.col("montant") - F.coalesce(F.col("montant_moyen_carte_7j"),
                                                     F.col("montant")))
                      / (F.col("montant_ecart_type_carte_7j") + F.lit(1.0)))
          .withColumn("ratio_montant_limite",
                      F.col("montant") / (F.coalesce(F.col("limite"), F.lit(5000.0)) + F.lit(1.0)))
          .withColumn("part_utilisation_limite_24h",
                      F.col("montant_somme_carte_24h")
                      / (F.coalesce(F.col("limite"), F.lit(5000.0)) + F.lit(1.0)))
          )
    return df


def variables_virement(df: DataFrame) -> DataFrame:
    """Version 2 : variables dediees aux virements Interac sortants.

    Ajoutees a l'etape 10 en reponse a la typologie `arnaque_virement_instantane`,
    ou la victime autorise elle-meme le virement : appareil connu, geolocalisation
    normale, heure normale. Les variables de la version 1 sont aveugles a ce cas.
    """
    w_client_24h = _fenetre("client_id", -H24)
    w_client_7j = _fenetre("client_id", -J7)
    w_benef = (Window.partitionBy("client_id", "marchand_id")
               .orderBy("ts_unix").rowsBetween(Window.unboundedPreceding, -1))

    est_virement = (F.col("canal") == "INTERAC").cast("int")
    return (df
            .withColumn("est_virement_sortant", est_virement)
            .withColumn("nb_virements_client_24h",
                        F.sum(est_virement).over(w_client_24h))
            .withColumn("montant_virements_client_24h",
                        F.sum(F.when(F.col("canal") == "INTERAC", F.col("montant"))
                              .otherwise(0.0)).over(w_client_24h))
            .withColumn("nb_virements_client_7j", F.sum(est_virement).over(w_client_7j))
            .withColumn("est_nouveau_beneficiaire",
                        F.when(F.col("canal") != "INTERAC", 0)
                         .when(F.count("*").over(w_benef) == 0, 1).otherwise(0))
            .withColumn("ratio_virement_moyenne_client",
                        F.when(F.col("canal") != "INTERAC", F.lit(0.0))
                         .otherwise(F.col("montant")
                                    / (F.col("montant_moyen_carte_7j") + F.lit(1.0)))))


COLONNES_CLES = ["transaction_id", "horodatage", "horodatage_label", "carte_id", "client_id",
                 "marchand_id", "est_fraude", "typologie_fraude"]

VARIABLES_NUMERIQUES_V1 = [
    "montant", "montant_log",
    "nb_tx_carte_1h", "nb_tx_carte_24h", "nb_tx_carte_7j",
    "montant_somme_carte_1h", "montant_somme_carte_24h",
    "montant_moyen_carte_7j", "montant_ecart_type_carte_7j", "montant_max_carte_7j",
    "nb_marchands_distincts_carte_24h", "nb_categories_distinctes_carte_24h",
    "nb_pays_distincts_carte_24h", "nb_micro_tx_carte_1h", "nb_tx_client_24h",
    "secondes_depuis_tx_precedente", "distance_km_tx_precedente", "distance_domicile_km",
    "vitesse_kmh", "ratio_montant_moyenne_carte", "ecart_type_montant_carte",
    "ratio_montant_limite", "part_utilisation_limite_24h",
    "nb_tx_marchand_1h", "taux_fraude_marchand_30j", "nb_tx_marchand_30j",
    "est_nouveau_marchand_carte", "est_nouvel_appareil_client",
    "heure", "jour_semaine", "est_nuit", "est_fin_semaine",
    "duree_session_s", "age", "anciennete_compte_jours",
]

VARIABLES_CATEGORIELLES_V1 = [
    "canal", "mode_saisie", "categorie_marchand", "pays", "type_carte", "reseau",
    "province", "segment", "authentification_3ds",
]

VARIABLES_NUMERIQUES_V2 = VARIABLES_NUMERIQUES_V1 + [
    "est_virement_sortant", "nb_virements_client_24h", "montant_virements_client_24h",
    "nb_virements_client_7j", "est_nouveau_beneficiaire", "ratio_virement_moyenne_client",
]


def executer(cfg: Config | None = None, version: int | None = None) -> dict:
    cfg = cfg or charger_config()
    version = version or int(cfg.get("variables", {}).get("version", 1))
    titre(f"ETAPE 03 - Ingenierie des variables (GOLD) - jeu de variables v{version}")
    spark = obtenir_spark(cfg)

    silver = lire_table(spark, cfg.chemin_table("silver", "transactions"), cfg)

    with chrono(LOG, "Calcul des variables de base (fenetres glissantes PySpark)"):
        gold = variables_de_base(silver)

    if version >= 2:
        with chrono(LOG, "Ajout des variables de virement (v2)"):
            gold = variables_virement(gold)

    numeriques = VARIABLES_NUMERIQUES_V2 if version >= 2 else VARIABLES_NUMERIQUES_V1
    colonnes = COLONNES_CLES + numeriques + VARIABLES_CATEGORIELLES_V1
    gold = (gold.select(*colonnes)
                .withColumn("_date_partition", F.to_date("horodatage")))

    cible = cfg.chemin_table("gold", "caracteristiques")
    with chrono(LOG, "Ecriture GOLD"):
        ecrire_table(gold, cible, cfg, mode="overwrite", partitions=["_date_partition"])

    spec = {
        "version": version,
        "cible": cfg["modelisation"]["cible"],
        "variables_numeriques": numeriques,
        "variables_categorielles": VARIABLES_CATEGORIELLES_V1,
        "colonnes_cles": COLONNES_CLES,
        "nb_variables": len(numeriques) + len(VARIABLES_CATEGORIELLES_V1),
    }
    ecrire_json(cfg.chemin("artefacts") / "variables" / f"specification_v{version}.json", spec)
    ecrire_json(cfg.chemin("artefacts") / "variables" / "specification_courante.json", spec)

    n = gold.count()
    LOG.info("GOLD : %d lignes x %d variables (%d numeriques, %d categorielles).",
             n, spec["nb_variables"], len(numeriques), len(VARIABLES_CATEGORIELLES_V1))
    return spec


if __name__ == "__main__":
    executer()
