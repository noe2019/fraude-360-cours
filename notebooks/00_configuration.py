# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Configuration de l'espace de travail
# MAGIC
# MAGIC Cree le catalogue Unity Catalog, les trois schemas de l'architecture
# MAGIC medaillon et l'emplacement externe pointant vers le conteneur ADLS Gen2.
# MAGIC
# MAGIC **A executer une seule fois**, avec un compte administrateur du metastore.
# MAGIC
# MAGIC | Ressource Azure | Role dans le projet |
# MAGIC |---|---|
# MAGIC | Azure Data Lake Storage Gen2 | stockage des couches bronze / silver / gold |
# MAGIC | Azure Databricks | calcul Spark, MLflow, Model Serving |
# MAGIC | Azure Key Vault | secrets (chaines de connexion, jetons) |
# MAGIC | Power BI | restitution (Direct Lake sur les tables gold) |

# COMMAND ----------

dbutils.widgets.text("catalogue", "fraude360", "Catalogue Unity Catalog")
dbutils.widgets.text("compte_stockage", "stfraude360", "Compte de stockage ADLS Gen2")
dbutils.widgets.text("conteneur", "lakehouse", "Conteneur")

catalogue = dbutils.widgets.get("catalogue")
compte = dbutils.widgets.get("compte_stockage")
conteneur = dbutils.widgets.get("conteneur")
racine = f"abfss://{conteneur}@{compte}.dfs.core.windows.net"
print("Racine du lac de donnees :", racine)

# COMMAND ----------

# MAGIC %md ## Catalogue et schemas

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalogue} MANAGED LOCATION '{racine}/uc'")
for schema in ("bronze", "silver", "gold", "surveillance"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalogue}.{schema}")
    print("schema pret :", f"{catalogue}.{schema}")

spark.sql(f"USE CATALOG {catalogue}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Emplacement externe et informations d'identification
# MAGIC
# MAGIC A creer une fois via l'interface Databricks (Catalog Explorer > External Data)
# MAGIC ou par Terraform. L'identite managee du connecteur d'acces Databricks doit
# MAGIC porter le role **Storage Blob Data Contributor** sur le compte de stockage.

# COMMAND ----------

spark.sql(f"""
CREATE EXTERNAL LOCATION IF NOT EXISTS emplacement_fraude360
URL '{racine}'
WITH (STORAGE CREDENTIAL `connecteur_fraude360`)
COMMENT 'Lac de donnees du projet de detection de la fraude'
""")

# COMMAND ----------

# MAGIC %md ## Volume pour la zone d'atterrissage (fichiers deposes par les systemes sources)

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalogue}.bronze.atterrissage")
print(f"Volume : /Volumes/{catalogue}/bronze/atterrissage")

# COMMAND ----------

# MAGIC %md ## Secrets (Azure Key Vault adosse a une portee Databricks)
# MAGIC ```
# MAGIC databricks secrets create-scope fraude360 \
# MAGIC   --scope-backend-type AZURE_KEYVAULT \
# MAGIC   --resource-id /subscriptions/<id>/resourceGroups/rg-fraude360/providers/Microsoft.KeyVault/vaults/kv-fraude360 \
# MAGIC   --dns-name https://kv-fraude360.vault.azure.net/
# MAGIC ```

# COMMAND ----------

print("Configuration terminee. Passer au notebook 01_ingestion_autoloader.")
