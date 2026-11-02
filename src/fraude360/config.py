"""Chargement de la configuration et resolution des chemins du projet.

Une seule source de verite : conf/config.yaml.
Toute etape du pipeline appelle `charger_config()`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def racine_projet() -> Path:
    """Racine du depot, quelle que soit la maniere dont le module est appele."""
    env = os.environ.get("FRAUDE360_HOME")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    brut: Dict[str, Any]
    racine: Path
    fichier: Path | None = None          # fichier YAML effectivement charge

    # ---- acces confortable -------------------------------------------------
    def __getitem__(self, cle: str) -> Any:
        return self.brut[cle]

    def get(self, cle: str, defaut: Any = None) -> Any:
        return self.brut.get(cle, defaut)

    def chemin(self, cle: str) -> Path:
        """Chemin absolu d'un dossier declare dans la section `chemins`."""
        rel = self.brut["chemins"][cle]
        p = (self.racine / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def chemin_table(self, couche: str, nom: str) -> Path:
        """Emplacement d'une table dans une couche (bronze/silver/gold)."""
        return self.chemin(couche) / nom

    @property
    def format_table(self) -> str:
        return self.brut["environnement"]["format_table"]

    @property
    def sur_databricks(self) -> bool:
        return (
            self.brut["environnement"]["cible"] == "databricks"
            or "DATABRICKS_RUNTIME_VERSION" in os.environ
        )

    @property
    def seed(self) -> int:
        return int(self.brut["environnement"]["seed"])


_cache: Config | None = None


def charger_config(chemin: str | Path | None = None, forcer: bool = False) -> Config:
    global _cache
    if _cache is not None and not forcer:
        return _cache
    racine = racine_projet()
    fichier = Path(chemin) if chemin else racine / "conf" / "config.yaml"
    with open(fichier, "r", encoding="utf-8") as f:
        brut = yaml.safe_load(f)
    _cache = Config(brut=brut, racine=racine, fichier=fichier)
    return _cache
