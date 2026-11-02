# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingestion incrementale avec Auto Loader
# MAGIC
# MAGIC Version Databricks de l'etape 01 du pipeline local
# MAGIC (`fraude360.ingestion.bronze`).
# MAGIC
# MAGIC Auto Loader (`cloudFiles`) decouvre les nouveaux fichiers deposes dans le
# MAGIC volume d'atterrissage et ne traite que ceux-la. Le point de reprise
# MAGIC (checkpoint) garantit l'exactement-une-fois : relancer le job apres un
# MAGIC incident ne cree aucun doublon.
# MAGIC
# MAGIC Le mode `availableNow` traite tout le retard accumule puis s'arrete : c'est
# MAGIC le mode adapte a un job planifie. Passer en `trigger(processingTime=...)`
# MAGIC suffit pour basculer en continu, sans changer une ligne de logique.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

dbutils.widgets.text("catalogue", "fraude360")
catalogue = dbutils.widgets.get("catalogue")
spark.sql(f"USE CATALOG {catalogue}")

atterrissage = f"/Volumes/{catalogue}/bronze/atterrissage"
checkpoint = f"/Volumes/{catalogue}/bronze/atterrissage/_checkpoints/transactions"
table_bronze = f"{catalogue}.bronze.transactions"

# COMMAND ----------

# MAGIC %md ## Schema contractuel
# MAGIC Le schema est declare, jamais infere : une colonne ajoutee en amont ne doit
# MAGIC pas casser la production silencieusement. Les enregistrements non conformes
# MAGIC partent dans `_rescued_data`.

# COMMAND ----------

SCHEMA_SOURCE = StructType([
    StructField("transaction_id", StringType()),
    StructField("horodatage", StringType()),
    StructField("carte_id", StringType()),
    StructField("client_id", StringType()),
    StructField("marchand_id", StringType()),
    StructField("categorie_marchand", StringType()),
    StructField("canal", StringType()),
    StructField("mode_saisie", StringType()),
    StructField("montant", StringType()),
    StructField("devise", StringType()),
    StructField("pays", StringType()),
    StructField("latitude", StringType()),
    StructField("longitude", StringType()),
    StructField("authentification_3ds", StringType()),
    StructField("appareil_id", StringType()),
    StructField("adresse_ip", StringType()),
    StructField("duree_session_s", StringType()),
    StructField("est_fraude", IntegerType()),
    StructField("typologie_fraude", StringType()),
    StructField("horodatage_label", StringType()),
])

# COMMAND ----------

flux = (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", f"{checkpoint}/schema")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .option("header", "true")
        .schema(SCHEMA_SOURCE)
        .load(atterrissage)
        .withColumn("_fichier_source", F.col("_metadata.file_path"))
        .withColumn("_horodatage_ingestion", F.current_timestamp())
        .withColumn("_date_partition", F.to_date(F.substring(F.col("horodatage"), 1, 10))))

# COMMAND ----------

(flux.writeStream
     .format("delta")
     .outputMode("append")
     .option("checkpointLocation", checkpoint)
     .option("mergeSchema", "true")
     .partitionBy("_date_partition")
     .trigger(availableNow=True)
     .toTable(table_bronze)
     .awaitTermination())

# COMMAND ----------

spark.sql(f"OPTIMIZE {table_bronze}")
display(spark.sql(f"""
SELECT _date_partition, count(*) AS lignes, count(DISTINCT _fichier_source) AS fichiers
FROM {table_bronze} GROUP BY _date_partition ORDER BY _date_partition DESC LIMIT 10"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tracabilite
# MAGIC `DESCRIBE HISTORY` donne l'historique complet des versions de la table Delta :
# MAGIC qui a ecrit, quand, combien de lignes. C'est ce qui permet de rejouer une
# MAGIC journee (`VERSION AS OF`) lors d'une enquete.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {table_bronze} LIMIT 5"))
