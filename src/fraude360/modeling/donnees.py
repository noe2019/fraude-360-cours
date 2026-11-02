"""Chargement de la table GOLD et decoupage temporel.

Regle absolue en detection de fraude : le decoupage est TEMPOREL, jamais
aleatoire. Un melange aleatoire ferait fuir de l'information du futur vers le
passe (une carte compromise apparaitrait des deux cotes) et donnerait des
performances irrealistes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.utils.journal import lire_json, obtenir_journal
from fraude360.utils.spark import lire_table, obtenir_spark

LOG = obtenir_journal("fraude360.donnees")


@dataclass
class JeuxDeDonnees:
    entrainement: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    variables_numeriques: List[str]
    variables_categorielles: List[str]
    cible: str

    @property
    def variables(self) -> List[str]:
        return self.variables_numeriques + self.variables_categorielles


def specification(cfg: Config) -> dict:
    return lire_json(cfg.chemin("artefacts") / "variables" / "specification_courante.json")


def charger_gold(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or charger_config()
    spark = obtenir_spark(cfg)
    pdf = lire_table(spark, cfg.chemin_table("gold", "caracteristiques"), cfg).toPandas()
    pdf["horodatage"] = pd.to_datetime(pdf["horodatage"])
    pdf = pdf.sort_values("horodatage").reset_index(drop=True)
    origine = pdf["horodatage"].min().normalize()
    pdf["jour_index"] = (pdf["horodatage"] - origine).dt.days
    # Les colonnes entieres sont converties en flottants : c'est ce que fera le
    # service de scoring en production (valeurs manquantes possibles), et cela
    # evite un ecart de schema entre l'entrainement et l'inference.
    spec = specification(cfg)
    for c in spec["variables_numeriques"]:
        if c in pdf.columns:
            pdf[c] = pdf[c].astype("float64")
    return pdf


def decouper(cfg: Config | None = None, pdf: pd.DataFrame | None = None,
             jusqu_au_jour: int | None = None) -> JeuxDeDonnees:
    cfg = cfg or charger_config()
    pdf = charger_gold(cfg) if pdf is None else pdf
    d = cfg["decoupage"]
    spec = specification(cfg)

    if jusqu_au_jour is not None:
        pdf = pdf[pdf.jour_index <= jusqu_au_jour]

    fin_train = d["jours_entrainement"]
    fin_valid = fin_train + d["jours_validation"]

    tr = pdf[pdf.jour_index < fin_train]
    va = pdf[(pdf.jour_index >= fin_train) & (pdf.jour_index < fin_valid)]
    te = pdf[pdf.jour_index >= fin_valid]

    LOG.info("Decoupage temporel : entrainement %d lignes (J0-J%d, %d fraudes) | "
             "validation %d (%d fraudes) | test %d (%d fraudes)",
             len(tr), fin_train - 1, tr.est_fraude.sum(),
             len(va), va.est_fraude.sum(), len(te), te.est_fraude.sum())

    return JeuxDeDonnees(
        entrainement=tr, validation=va, test=te,
        variables_numeriques=[c for c in spec["variables_numeriques"] if c in pdf.columns],
        variables_categorielles=[c for c in spec["variables_categorielles"] if c in pdf.columns],
        cible=spec["cible"])
