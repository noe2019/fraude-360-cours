# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Entrainement, suivi MLflow et enregistrement Unity Catalog
# MAGIC
# MAGIC Version Databricks de `fraude360.modeling.entrainement`.
# MAGIC
# MAGIC Points d'attention specifiques a la fraude :
# MAGIC - decoupage **temporel**, jamais aleatoire ;
# MAGIC - metrique de selection **PR-AUC** (l'exactitude ne veut rien dire a 0,4 %
# MAGIC   de positifs) ;
# MAGIC - probabilites **calibrees**, sinon le seuil economique n'a pas de sens ;
# MAGIC - seuil de decision choisi en **dollars**, sous contrainte du budget
# MAGIC   quotidien de l'equipe d'enquete.

# COMMAND ----------

# MAGIC %pip install /Volumes/fraude360/gold/bibliotheques/fraude360-1.0.0-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------

import mlflow
from fraude360.config import charger_config
from fraude360.modeling import entrainement

dbutils.widgets.text("catalogue", "fraude360")
dbutils.widgets.text("etiquette", "v1")
catalogue = dbutils.widgets.get("catalogue")
etiquette = dbutils.widgets.get("etiquette")

cfg = charger_config()
cfg.brut["environnement"]["cible"] = "databricks"
cfg.brut["databricks"]["catalogue"] = catalogue

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/fraude360/experiences")

# COMMAND ----------

resume = entrainement.executer(cfg, etiquette=etiquette)

# COMMAND ----------

print("Modele retenu       :", resume["modele_retenu"])
print("Seuil de decision   :", round(resume["seuil_decision"], 5))
print("PR-AUC (test)       :", round(resume["metriques_test"]["pr_auc"], 4))
print("Rappel (test)       :", round(resume["metriques_test"]["rappel"], 4))
print("Economie annuelle   :", resume["economie_vs_sans_modele"], "$ sur la periode de test")
print("Version enregistree :", resume["version_registre"], "alias", resume["alias"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gouvernance du modele
# MAGIC Unity Catalog conserve la lignee complete : quelles tables ont produit
# MAGIC quelles variables, quel run a produit quelle version, qui l'a promue.
# MAGIC C'est ce qui rend le modele auditable par la deuxieme ligne de defense.

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
nom = f"{catalogue}.gold.modele_fraude"
for v in client.search_model_versions(f"name='{nom}'"):
    print(f"version {v.version} | alias {v.aliases} | cree {v.creation_timestamp} | {v.description}")
