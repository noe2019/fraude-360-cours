"""Journalisation homogene + petites aides d'affichage pour toutes les etapes."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

_CONFIGURE = False


def obtenir_journal(nom: str) -> logging.Logger:
    global _CONFIGURE
    if not _CONFIGURE:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        logging.getLogger("py4j").setLevel(logging.ERROR)
        logging.getLogger("pyspark").setLevel(logging.ERROR)
        _CONFIGURE = True
    return logging.getLogger(nom)


def titre(texte: str) -> None:
    barre = "=" * 78
    print(f"\n{barre}\n  {texte}\n{barre}", flush=True)


@contextmanager
def chrono(journal: logging.Logger, libelle: str):
    debut = time.time()
    journal.info("%s ...", libelle)
    yield
    journal.info("%s termine en %.1f s", libelle, time.time() - debut)


def ecrire_json(chemin: Path, contenu: Dict[str, Any]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(contenu, f, indent=2, ensure_ascii=False, default=str)


def lire_json(chemin: Path) -> Dict[str, Any]:
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)
