# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Deploiement : Databricks Model Serving
# MAGIC
# MAGIC Deux modes de deploiement coexistent en detection de fraude :
# MAGIC
# MAGIC | Mode | Latence | Usage |
# MAGIC |---|---|---|
# MAGIC | **Temps reel** (Model Serving) | < 100 ms | autorisation de la transaction au moment du paiement |
# MAGIC | **Par lots** (job planifie) | minutes | revue des transactions de la nuit, files d'enquete |
# MAGIC
# MAGIC Le point de terminaison sert l'alias `champion` : promouvoir une nouvelle
# MAGIC version au registre suffit a changer le modele servi, sans redeploiement.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (EndpointCoreConfigInput, ServedEntityInput,
                                            AutoCaptureConfigInput)

dbutils.widgets.text("catalogue", "fraude360")
catalogue = dbutils.widgets.get("catalogue")
nom_modele = f"{catalogue}.gold.modele_fraude"
nom_point = "fraude360-scoring"

w = WorkspaceClient()

# COMMAND ----------

config = EndpointCoreConfigInput(
    name=nom_point,
    served_entities=[ServedEntityInput(
        entity_name=nom_modele,
        entity_version=None,          # sert l'alias champion
        entity_alias="champion",
        workload_size="Small",
        scale_to_zero_enabled=True)],
    # Journalisation automatique des requetes et des reponses : indispensable
    # pour la surveillance de la derive et pour l'audit des decisions.
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name=catalogue, schema_name="surveillance",
        table_name_prefix="scoring", enabled=True))

points_existants = [p.name for p in w.serving_endpoints.list()]
if nom_point in points_existants:
    w.serving_endpoints.update_config(name=nom_point,
                                      served_entities=config.served_entities)
    print("Point de terminaison mis a jour :", nom_point)
else:
    w.serving_endpoints.create(name=nom_point, config=config)
    print("Point de terminaison cree :", nom_point)

# COMMAND ----------

# MAGIC %md ## Test du point de terminaison

# COMMAND ----------

import pandas as pd

exemple = (spark.table(f"{catalogue}.gold.caracteristiques")
           .filter("est_fraude = 1").limit(3).toPandas())
colonnes = [c for c in exemple.columns
            if c not in ("transaction_id", "horodatage", "horodatage_label", "carte_id",
                         "client_id", "marchand_id", "est_fraude", "typologie_fraude")]

reponse = w.serving_endpoints.query(
    name=nom_point, dataframe_records=exemple[colonnes].to_dict("records"))
print(reponse.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring par lots
# MAGIC Pour les volumes importants, on n'appelle pas l'API : on charge le modele
# MAGIC comme une fonction Spark et on score en parallele sur le cluster.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

scorer = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{nom_modele}@champion",
                                 result_type="double", env_manager="local")

from pyspark.sql import functions as F
decisions = (spark.table(f"{catalogue}.gold.caracteristiques")
             .withColumn("score_modele", scorer(*[F.col(c) for c in colonnes])))
(decisions.write.mode("overwrite").saveAsTable(f"{catalogue}.gold.decisions"))
print("Scoring par lots termine.")
