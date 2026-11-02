"""Session Spark unique, portable entre poste local et Databricks.

Sur Databricks, la session existe deja : on la recupere.
En local, on demarre un Spark local[*] et, si Delta Lake est disponible,
on active les extensions Delta ; sinon on retombe proprement sur Parquet.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from fraude360.config import Config, charger_config

_LOG = logging.getLogger(__name__)
_session: Optional[SparkSession] = None


def obtenir_spark(cfg: Config | None = None, nom_app: str = "fraude-360") -> SparkSession:
    global _session
    if _session is not None:
        return _session

    cfg = cfg or charger_config()
    env = cfg["environnement"]

    # Spark appelle "python3" pour lancer ses processus de calcul.
    # Sous Windows, un environnement virtuel ne cree pas de python3.exe :
    # l appel tombe sur le raccourci du Microsoft Store et echoue.
    # On impose donc l interprete du pilote aux executeurs.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        _session = SparkSession.builder.getOrCreate()
        return _session

    builder = (
        SparkSession.builder.appName(nom_app)
        .master(env["spark_master"])
        .config("spark.driver.memory", env["spark_driver_memory"])
        .config("spark.sql.shuffle.partitions", env["spark_shuffle_partitions"])
        .config("spark.sql.session.timeZone", "America/Toronto")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.host", "127.0.0.1")
    )

    if cfg.format_table == "delta":
        try:
            from delta import configure_spark_with_delta_pip  # type: ignore

            builder = (
                builder.config(
                    "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
                ).config(
                    "spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog",
                )
            )
            builder = configure_spark_with_delta_pip(builder)
        except Exception as exc:  # pragma: no cover
            _LOG.warning("Delta indisponible (%s) - repli sur Parquet.", exc)

    _session = builder.getOrCreate()
    _session.sparkContext.setLogLevel("ERROR")
    return _session


def ecrire_table(df: DataFrame, chemin, cfg: Config, mode: str = "overwrite",
                 partitions: list[str] | None = None) -> None:
    """Ecrit une table dans le format cible (Delta sur Databricks, Parquet en local)."""
    w = df.write.mode(mode)
    if partitions:
        w = w.partitionBy(*partitions)
    fmt = cfg.format_table
    if fmt == "delta":
        w.format("delta").option("overwriteSchema", "true").save(str(chemin))
    else:
        w.parquet(str(chemin))


def lire_table(spark: SparkSession, chemin, cfg: Config) -> DataFrame:
    if cfg.format_table == "delta":
        return spark.read.format("delta").load(str(chemin))
    return spark.read.parquet(str(chemin))


def arreter_spark() -> None:
    global _session
    if _session is not None:
        _session.stop()
        _session = None
