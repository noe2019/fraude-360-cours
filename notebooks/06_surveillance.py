# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Surveillance continue
# MAGIC
# MAGIC Version Databricks de `fraude360.monitoring.surveillance`.
# MAGIC
# MAGIC Trois horizons de detection, du plus rapide au plus fiable :
# MAGIC 1. **derive des donnees** (immediat) - PSI et test de Kolmogorov-Smirnov ;
# MAGIC 2. **derive operationnelle** (immediat) - volume d'alertes, scores moyens ;
# MAGIC 3. **derive de performance** (differee du delai de labellisation) - rappel,
# MAGIC    precision, dollars non evites.
# MAGIC
# MAGIC Les resultats alimentent une table Delta et une alerte Databricks SQL qui
# MAGIC previent l'equipe par courriel ou Teams.

# COMMAND ----------

# MAGIC %pip install /Volumes/fraude360/gold/bibliotheques/fraude360-1.0.0-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------

from fraude360.config import charger_config
from fraude360.monitoring import surveillance

dbutils.widgets.text("catalogue", "fraude360")
cfg = charger_config()
cfg.brut["environnement"]["cible"] = "databricks"
cfg.brut["databricks"]["catalogue"] = dbutils.widgets.get("catalogue")

rapport = surveillance.executer(cfg)

# COMMAND ----------

print("Alertes levees :", rapport["nb_alertes"])
for a in rapport["alertes"]:
    print(f"  [{a['gravite']:<12}] {a['message']}")
print("\nReentrainement recommande :", rapport["reentrainement_recommande"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakehouse Monitoring (alternative native)
# MAGIC Databricks sait calculer ces metriques automatiquement sur une table de
# MAGIC scoring, avec un tableau de bord genere. On garde ici le calcul explicite
# MAGIC parce qu'il rend les seuils et la logique metier lisibles et versionnes.
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC w = WorkspaceClient()
# MAGIC w.quality_monitors.create(
# MAGIC     table_name="fraude360.gold.decisions",
# MAGIC     inference_log=MonitorInferenceLog(
# MAGIC         problem_type="PROBLEM_TYPE_CLASSIFICATION",
# MAGIC         prediction_col="score_modele", label_col="est_fraude",
# MAGIC         timestamp_col="horodatage", granularities=["1 day"],
# MAGIC         model_id_col="version_modele"),
# MAGIC     output_schema_name="fraude360.surveillance")
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## Declenchement conditionnel du reentrainement
# MAGIC Cette tache renvoie une valeur que la tache suivante du job lit pour decider
# MAGIC si elle doit s'executer (condition `if/else` dans Databricks Workflows).

# COMMAND ----------

dbutils.jobs.taskValues.set(key="reentrainer",
                            value=str(rapport["reentrainement_recommande"]).lower())
print("Valeur transmise au job :", rapport["reentrainement_recommande"])
