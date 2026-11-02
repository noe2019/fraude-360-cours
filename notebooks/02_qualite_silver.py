# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Nettoyage, controles de qualite et couche Silver
# MAGIC
# MAGIC Version Databricks de `fraude360.quality.silver`.
# MAGIC
# MAGIC Deux principes non negociables :
# MAGIC 1. **Rien n'est supprime silencieusement.** Toute ligne rejetee part en
# MAGIC    quarantaine avec son motif ; le taux de rejet est un indicateur suivi.
# MAGIC 2. **L'ecriture est idempotente** : un `MERGE` sur la cle metier permet de
# MAGIC    rejouer un lot sans creer de doublon.

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.window import Window

dbutils.widgets.text("catalogue", "fraude360")
catalogue = dbutils.widgets.get("catalogue")
spark.sql(f"USE CATALOG {catalogue}")

bronze = f"{catalogue}.bronze.transactions"
silver = f"{catalogue}.silver.transactions"
quarantaine = f"{catalogue}.silver.quarantaine"

# COMMAND ----------

# MAGIC %md ## Normalisation

# COMMAND ----------

df = (spark.table(bronze)
      .withColumn("transaction_id", F.trim("transaction_id"))
      .withColumn("carte_id", F.upper(F.trim("carte_id")))
      .withColumn("client_id", F.upper(F.trim("client_id")))
      .withColumn("marchand_id", F.upper(F.trim("marchand_id")))
      .withColumn("horodatage", F.to_timestamp("horodatage"))
      .withColumn("horodatage_label", F.to_timestamp("horodatage_label"))
      .withColumn("montant", F.regexp_replace(F.trim("montant"), ",", ".").cast("double"))
      .withColumn("devise", F.upper(F.trim("devise")))
      .withColumn("canal", F.upper(F.trim("canal")))
      .withColumn("categorie_marchand",
                  F.coalesce(F.lower(F.trim("categorie_marchand")), F.lit("inconnue")))
      .withColumn("latitude", F.col("latitude").cast("double"))
      .withColumn("longitude", F.col("longitude").cast("double"))
      .withColumn("authentification_3ds",
                  F.lower(F.col("authentification_3ds")).isin("true", "1", "vrai"))
      .withColumn("duree_session_s", F.coalesce(F.col("duree_session_s").cast("int"), F.lit(0)))
      .withColumn("est_fraude", F.coalesce(F.col("est_fraude"), F.lit(0))))

# COMMAND ----------

# MAGIC %md ## Dedoublonnage : une transaction, une ligne

# COMMAND ----------

w = Window.partitionBy("transaction_id").orderBy(F.col("_horodatage_ingestion").desc())
df = df.withColumn("_rang", F.row_number().over(w)).filter("_rang = 1").drop("_rang")

# COMMAND ----------

# MAGIC %md ## Controles de qualite (equivalent des expectations DLT)

# COMMAND ----------

df = df.withColumn(
    "motif_rejet",
    F.when(F.col("transaction_id").isNull() | F.col("carte_id").isNull()
           | F.col("client_id").isNull(), "champ_obligatoire_manquant")
     .when(F.col("montant").isNull(), "montant_illisible")
     .when(F.col("montant") < 0.01, "montant_negatif_ou_nul")
     .when(F.col("montant") > 25000, "montant_hors_bornes")
     .when(~F.col("devise").isin("CAD", "USD", "EUR"), "devise_non_autorisee")
     .when(F.col("horodatage").isNull(), "horodatage_illisible")
     .when(F.col("latitude").isNull() | ~F.col("latitude").between(-90, 90),
           "geolocalisation_invalide")
     .otherwise(F.lit(None).cast("string")))

rejets = df.filter("motif_rejet IS NOT NULL")
conformes = df.filter("motif_rejet IS NULL").drop("motif_rejet")

n_total, n_rejets = df.count(), rejets.count()
taux = n_rejets / max(n_total, 1)
print(f"Lignes : {n_total} | rejets : {n_rejets} ({taux:.2%})")
display(rejets.groupBy("motif_rejet").count().orderBy(F.desc("count")))

if taux > 0.05:
    raise Exception(f"Taux de rejet {taux:.2%} au-dela du seuil bloquant de 5 % : lot refuse.")

# COMMAND ----------

rejets.write.mode("append").saveAsTable(quarantaine)

# COMMAND ----------

# MAGIC %md ## Jonction aux referentiels et ecriture idempotente (MERGE)

# COMMAND ----------

clients = spark.table(f"{catalogue}.silver.clients")
cartes = spark.table(f"{catalogue}.silver.cartes")

final = (conformes
         .join(F.broadcast(clients), "client_id", "left")
         .join(F.broadcast(cartes), "carte_id", "left")
         .withColumn("_date_partition", F.to_date("horodatage")))

if spark.catalog.tableExists(silver):
    (DeltaTable.forName(spark, silver).alias("c")
     .merge(final.alias("n"), "c.transaction_id = n.transaction_id")
     .whenMatchedUpdateAll()
     .whenNotMatchedInsertAll()
     .execute())
else:
    final.write.partitionBy("_date_partition").saveAsTable(silver)

spark.sql(f"OPTIMIZE {silver} ZORDER BY (carte_id, horodatage)")
print("Silver :", spark.table(silver).count(), "lignes")
