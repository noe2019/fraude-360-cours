"""Outils communs aux laboratoires du cours « Science des données appliquée : détection de la fraude ».

Chaque laboratoire appelle `preparer("<etape>")` : les étapes du pipeline Fraude-360 nécessaires
sont exécutées dans l'ordre, une seule fois (l'état est mémorisé dans artifacts/_etat_cours.json).
On peut donc ouvrir n'importe quel laboratoire sans avoir fait les précédents.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
CONFIG = RACINE / "conf" / "config_cours.yaml"

os.environ.setdefault("FRAUDE360_HOME", str(RACINE))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")   # aucun envoi de télémétrie
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if str(RACINE / "src") not in sys.path:
    sys.path.insert(0, str(RACINE / "src"))

ORDRE = ["generer", "ingerer", "nettoyer", "variables", "eda", "entrainer", "abtest",
         "scorer", "surveiller", "nouvelle-fraude", "reentrainer", "rescorer",
         "triage-genai", "equite", "carte-modele", "exporter-bi"]


def config():
    """Configuration du cours (conf/config_cours.yaml), rechargée depuis le disque."""
    from fraude360.config import charger_config
    return charger_config(CONFIG, forcer=True)


def _fichier_etat(cfg) -> Path:
    return cfg.chemin("artefacts") / "_etat_cours.json"


def etapes_faites() -> list[str]:
    f = _fichier_etat(config())
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def preparer(jusqu_a: str, verbeux: bool = False):
    """Exécute, si nécessaire, toutes les étapes du pipeline jusqu'à `jusqu_a` incluse.

    Retourne la configuration. Les étapes déjà faites ne sont pas refaites.
    """
    import logging
    from fraude360.pipeline import _etapes

    if jusqu_a not in ORDRE:
        raise ValueError(f"Étape inconnue : {jusqu_a}. Choix : {ORDRE}")
    if not verbeux:
        logging.getLogger("fraude360").setLevel(logging.WARNING)
        for nom in ("py4j", "mlflow", "alembic"):
            logging.getLogger(nom).setLevel(logging.ERROR)
    cfg = config()
    faites = etapes_faites()
    fonctions = {nom: (lib, f) for nom, lib, f in _etapes()}
    cible = ORDRE[: ORDRE.index(jusqu_a) + 1]
    a_faire = [e for e in cible if e not in faites]
    if not a_faire:
        print(f"✔ Pipeline déjà prêt jusqu'à « {jusqu_a} ».")
        return cfg
    for e in a_faire:
        lib, f = fonctions[e]
        t0 = time.time()
        print(f"▶ {lib} ...", end=" ", flush=True)
        f(cfg)
        faites.append(e)
        _fichier_etat(cfg).parent.mkdir(parents=True, exist_ok=True)
        _fichier_etat(cfg).write_text(json.dumps(faites), encoding="utf-8")
        print(f"ok ({time.time() - t0:.0f} s)")
        cfg = config()  # certaines étapes modifient la configuration (jeu de variables v2)
    return cfg


def reinitialiser() -> None:
    """Efface données, artefacts et registre MLflow du cours et remet le jeu de variables v1."""
    cfg = config()
    for cle in ("racine_donnees", "artefacts", "mlruns", "exports_bi"):
        dossier = cfg.chemin(cle)
        if dossier.exists() and RACINE in dossier.resolve().parents:
            shutil.rmtree(dossier)
    texte = CONFIG.read_text(encoding="utf-8")
    CONFIG.write_text(re.sub(r"(variables:\s*\n\s*version:\s*)\d+", r"\g<1>1", texte), encoding="utf-8")
    print("Cours réinitialisé.")


# --- rétroaction instantanée ------------------------------------------------------------
def verifier(condition, message="Vérification échouée"):
    assert condition, message
    print("✅ Réussi")


def verifier_proche(valeur, attendu, tol=1e-6, message=None):
    ok = abs(float(valeur) - float(attendu)) <= tol * max(1.0, abs(float(attendu)))
    assert ok, message or f"Attendu ≈ {attendu!r}, obtenu {valeur!r}"
    print("✅ Réussi")


def image(chemin, largeur=760):
    """Affiche une figure produite par le pipeline."""
    from IPython.display import Image, display
    display(Image(filename=str(chemin), width=largeur))


def lire_json(chemin):
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


def table(couche: str, nom: str):
    """Lit une table du lakehouse (Parquet partitionné) en pandas, sans Spark."""
    import pyarrow.dataset as ds
    chemin = config().chemin_table(couche, nom)
    df = ds.dataset(str(chemin), format="parquet", partitioning="hive",
                    ignore_prefixes=[".", "_SUCCESS", "_committed", "_started"]).to_table().to_pandas()
    # Spark stocke les horodatages en UTC et les affiche dans le fuseau de la session :
    # on revient à l'heure locale pour obtenir exactement les mêmes valeurs que Spark.
    import pandas as pd
    fuseau = "America/Toronto"
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            s = df[c]
            s = s.dt.tz_localize("UTC") if s.dt.tz is None else s
            df[c] = s.dt.tz_convert(fuseau).dt.tz_localize(None)
    return df
