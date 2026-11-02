# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Ingenierie des variables (couche Gold)
# MAGIC
# MAGIC Version Databricks de `fraude360.features.gold`. Le code de calcul est
# MAGIC **strictement le meme** : la bibliotheque du projet est installee sur le
# MAGIC cluster (wheel), ce qui garantit qu'une variable calculee ici est identique
# MAGIC a celle calculee en local et a celle servie par l'API. C'est la seule facon
# MAGIC d'eviter le decalage entrainement / production (*training-serving skew*).
# MAGIC
# MAGIC Toutes les fenetres sont **anterieures** a la transaction courante :
# MAGIC aucune information du futur n'entre dans le modele.

# COMMAND ----------

# MAGIC %pip install /Volumes/fraude360/gold/bibliotheques/fraude360-1.0.0-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------

from fraude360.features.gold import (VARIABLES_CATEGORIELLES_V1, VARIABLES_NUMERIQUES_V1,
                                     VARIABLES_NUMERIQUES_V2, COLONNES_CLES,
                                     variables_de_base, variables_virement)

dbutils.widgets.text("catalogue", "fraude360")
dbutils.widgets.dropdown("version_variables", "1", ["1", "2"])
catalogue = dbutils.widgets.get("catalogue")
version = int(dbutils.widgets.get("version_variables"))
spark.sql(f"USE CATALOG {catalogue}")

# COMMAND ----------

silver = spark.table(f"{catalogue}.silver.transactions")
gold = variables_de_base(silver)
if version >= 2:
    gold = variables_virement(gold)

numeriques = VARIABLES_NUMERIQUES_V2 if version >= 2 else VARIABLES_NUMERIQUES_V1
gold = gold.select(*(COLONNES_CLES + numeriques + VARIABLES_CATEGORIELLES_V1))
print(f"Jeu de variables v{version} : {len(numeriques) + len(VARIABLES_CATEGORIELLES_V1)} variables")

# COMMAND ----------

(gold.write.mode("overwrite").option("overwriteSchema", "true")
     .partitionBy("horodatage")
     .saveAsTable(f"{catalogue}.gold.caracteristiques"))

spark.sql(f"OPTIMIZE {catalogue}.gold.caracteristiques ZORDER BY (carte_id, horodatage)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Documentation du modele de donnees
# MAGIC Les commentaires de colonnes sont visibles dans Catalog Explorer et dans
# MAGIC Power BI : c'est la documentation vivante du projet.

# COMMAND ----------

commentaires = {
    "nb_tx_carte_1h": "Nombre de transactions de la carte dans l'heure precedente",
    "vitesse_kmh": "Vitesse implicite entre deux transactions (deplacement impossible)",
    "taux_fraude_marchand_30j": "Taux de fraude du marchand sur 30 jours, decale d'une seconde",
    "est_nouveau_marchand_carte": "1 si la carte n'a jamais paye chez ce marchand",
    "est_nouveau_beneficiaire": "1 si le membre n'a jamais vire d'argent a ce beneficiaire (v2)",
    "nb_virements_client_24h": "Nombre de virements Interac sortants sur 24 h (v2)",
}
for colonne, texte in commentaires.items():
    try:
        spark.sql(f"ALTER TABLE {catalogue}.gold.caracteristiques "
                  f"ALTER COLUMN {colonne} COMMENT '{texte}'")
    except Exception as e:
        print(colonne, "->", e)

# COMMAND ----------

# MAGIC %md ## Controle anti-fuite : la cible ne doit pas etre reconstituable

# COMMAND ----------

from pyspark.sql import functions as F
controle = (spark.table(f"{catalogue}.gold.caracteristiques")
            .groupBy("est_fraude")
            .agg(F.count("*").alias("lignes"),
                 F.avg("taux_fraude_marchand_30j").alias("taux_marchand_moyen")))
display(controle)
print("Si taux_marchand_moyen vaut 1,0 pour les fraudes, il y a fuite : verifier le decalage.")
