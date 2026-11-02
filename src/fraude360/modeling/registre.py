"""Enregistrement des modeles - MLflow.

En local : entrepot SQLite + artefacts sur disque (le registre de modeles MLflow
exige un backend transactionnel, un simple dossier ne suffit pas).
Sur Databricks : le meme code pointe vers le registre Unity Catalog, il suffit de
changer `environnement.cible` dans conf/config.yaml.

Le cycle de vie utilise des ALIAS plutot que des stages (approche recommandee
depuis MLflow 2.9) :
    champion   -> modele servi en production
    challenger -> modele candidat evalue en test A/B
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from fraude360.config import Config, charger_config
from fraude360.utils.journal import obtenir_journal

LOG = obtenir_journal("fraude360.registre")

NOM_MODELE = "fraude360_detection"
ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"


def configurer(cfg: Config | None = None) -> MlflowClient:
    cfg = cfg or charger_config()
    if cfg.sur_databricks:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(cfg["databricks"]["experience_mlflow"])
        return MlflowClient()

    racine = cfg.chemin("mlruns")
    base = f"sqlite:///{(racine / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(base)
    mlflow.set_registry_uri(base)
    artefacts = (racine / "artefacts").resolve()
    artefacts.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    nom_exp = "fraude360"
    exp = client.get_experiment_by_name(nom_exp)
    if exp is None:
        client.create_experiment(nom_exp, artifact_location=artefacts.as_uri())
    mlflow.set_experiment(nom_exp)
    return client


def enregistrer(uri_modele: str, cfg: Config | None = None,
                alias: Optional[str] = None, description: str = "") -> int:
    """Enregistre une version du modele et lui pose eventuellement un alias."""
    cfg = cfg or charger_config()
    client = MlflowClient()
    nom = cfg["databricks"]["modele_enregistre"] if cfg.sur_databricks else NOM_MODELE
    try:
        client.create_registered_model(nom)
    except Exception:
        pass
    version = mlflow.register_model(uri_modele, nom)
    numero = int(version.version)
    if description:
        client.update_model_version(nom, numero, description=description)
    if alias:
        client.set_registered_model_alias(nom, alias, numero)
        LOG.info("Modele %s version %d enregistre avec l'alias '%s'.", nom, numero, alias)
    return numero


def promouvoir(numero_version: int, alias: str = ALIAS_CHAMPION,
               cfg: Config | None = None) -> None:
    cfg = cfg or charger_config()
    client = MlflowClient()
    nom = cfg["databricks"]["modele_enregistre"] if cfg.sur_databricks else NOM_MODELE
    client.set_registered_model_alias(nom, alias, numero_version)
    LOG.info("Version %d promue '%s'.", numero_version, alias)


def version_par_alias(alias: str, cfg: Config | None = None) -> Optional[int]:
    cfg = cfg or charger_config()
    client = MlflowClient()
    nom = cfg["databricks"]["modele_enregistre"] if cfg.sur_databricks else NOM_MODELE
    try:
        return int(client.get_model_version_by_alias(nom, alias).version)
    except Exception:
        return None


def charger_par_alias(alias: str = ALIAS_CHAMPION, cfg: Config | None = None):
    cfg = cfg or charger_config()
    configurer(cfg)
    nom = cfg["databricks"]["modele_enregistre"] if cfg.sur_databricks else NOM_MODELE
    return mlflow.sklearn.load_model(f"models:/{nom}@{alias}")
