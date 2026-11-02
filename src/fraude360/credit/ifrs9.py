"""IFRS 9 : classement par etape et perte de credit attendue.

IFRS 9 remplace la provision pour perte SUBIE par une provision pour perte
ATTENDUE. Trois etapes :

  ETAPE 1  pret sain depuis l'octroi        -> ECL a 12 mois
  ETAPE 2  augmentation significative du risque de credit (ASRC)
                                             -> ECL a VIE
  ETAPE 3  en defaut                        -> ECL a vie, interets sur le net

Le point sensible, et celui que la validation independante regarde en premier,
est le critere d'ASRC. La norme ne le chiffre pas : chaque institution le
definit et doit le justifier. Ici, trois declencheurs, l'un suffit :

  - la PD a vie a plus que double depuis l'octroi ET a gagne au moins 50 pb
    (le double critere evite de basculer en etape 2 des prets tres sains dont
    la PD passe de 0,05 % a 0,11 %) ;
  - retard de paiement de 30 jours ou plus -- presomption refutable de la norme ;
  - degradation d'au moins trois classes de la grille interne.

    ECL = somme sur les periodes de  PD_marginale * LGD * EAD * facteur_actualisation

La PD a vie est construite depuis la PD a 12 mois par une courbe de survie a
hasard constant : c'est l'approximation usuelle en l'absence d'un modele
multi-horizon, et elle doit etre declaree comme telle dans le dossier de modele.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from fraude360.utils.journal import obtenir_journal

LOG = obtenir_journal("fraude360.credit.ifrs9")

TAUX_ACTUALISATION = 0.065     # taux d'interet effectif moyen du portefeuille
SEUIL_ASRC_MULTIPLE = 2.0      # PD a vie multipliee par
SEUIL_ASRC_ABSOLU = 0.005      # ET augmentation d'au moins 50 points de base
SEUIL_CLASSES = 3              # degradation de classes internes


def pd_a_vie(pd_12m: np.ndarray, duree_mois: np.ndarray) -> np.ndarray:
    """Survie a hasard constant : PD_vie = 1 - (1 - PD_12m) ^ (duree / 12)."""
    annees = np.clip(duree_mois / 12.0, 1.0 / 12, 10.0)
    return 1.0 - np.power(1.0 - np.clip(pd_12m, 1e-9, 0.999), annees)


def pd_marginales(pd_12m: np.ndarray, annees: int) -> np.ndarray:
    """Matrice (n, annees) des PD marginales : defaut en annee t, survie avant."""
    p = np.clip(pd_12m, 1e-9, 0.999)[:, None]
    t = np.arange(annees)[None, :]
    return np.power(1 - p, t) * p


def classer_etapes(pd_12m_courante: np.ndarray, pd_12m_octroi: np.ndarray,
                   duree_mois: np.ndarray, jours_retard: np.ndarray,
                   en_defaut: np.ndarray,
                   classes_degradees: np.ndarray | None = None) -> np.ndarray:
    vie_c = pd_a_vie(pd_12m_courante, duree_mois)
    vie_o = pd_a_vie(pd_12m_octroi, duree_mois)

    asrc_quantitatif = (vie_c > SEUIL_ASRC_MULTIPLE * vie_o) & \
                       ((vie_c - vie_o) >= SEUIL_ASRC_ABSOLU)
    asrc_arrieres = jours_retard >= 30
    asrc_classes = (classes_degradees >= SEUIL_CLASSES
                    if classes_degradees is not None
                    else np.zeros_like(asrc_arrieres, dtype=bool))

    etape = np.ones(len(pd_12m_courante), dtype=int)
    etape[asrc_quantitatif | asrc_arrieres | asrc_classes] = 2
    etape[en_defaut.astype(bool)] = 3
    return etape


def ecl(pd_12m: np.ndarray, lgd: np.ndarray, ead: np.ndarray,
        duree_mois: np.ndarray, etape: np.ndarray,
        taux: float = TAUX_ACTUALISATION) -> np.ndarray:
    """Perte de credit attendue, actualisee, selon l'etape."""
    horizon = np.maximum(1, np.ceil(np.clip(duree_mois, 12, 120) / 12)).astype(int)
    n_max = int(horizon.max())
    marg = pd_marginales(pd_12m, n_max)                      # (n, n_max)
    actual = 1.0 / np.power(1 + taux, np.arange(1, n_max + 1))[None, :]

    # masque : etape 1 = 1 an seulement, etapes 2 et 3 = duree residuelle
    t = np.arange(n_max)[None, :]
    limite = np.where(etape == 1, 1, horizon)[:, None]
    masque = (t < limite).astype(float)

    perte = (marg * masque * actual).sum(axis=1) * lgd * ead
    # en defaut : la perte n'est plus probabiliste, elle est encourue
    perte = np.where(etape == 3, lgd * ead, perte)
    return np.round(perte, 2)


def calculer(d: pd.DataFrame, pd_courante: np.ndarray,
             pd_octroi: np.ndarray) -> pd.DataFrame:
    jours_retard = np.where(d.nb_retards_12m.values >= 2, 45, 0)
    etape = classer_etapes(pd_courante, pd_octroi, d.duree_mois.values,
                           jours_retard, d.defaut_12m.values)
    perte = ecl(pd_courante, d.lgd_observee.values, d.exposition.values,
                d.duree_mois.values, etape)
    return d.assign(pd_12m=np.round(pd_courante, 6),
                    pd_octroi=np.round(pd_octroi, 6),
                    pd_vie=np.round(pd_a_vie(pd_courante, d.duree_mois.values), 6),
                    etape_ifrs9=etape, ecl=perte)


def synthese(d: pd.DataFrame) -> Dict[str, object]:
    g = d.groupby("etape_ifrs9").agg(
        nb=("pret_id", "size"),
        exposition=("exposition", "sum"),
        ecl=("ecl", "sum"),
        pd_moyenne=("pd_12m", "mean"))
    g["taux_couverture_pct"] = 100 * g.ecl / g.exposition
    total_exp, total_ecl = float(d.exposition.sum()), float(d.ecl.sum())
    return {
        "par_etape": {int(k): {"nb": int(r.nb),
                               "exposition": round(float(r.exposition), 2),
                               "ecl": round(float(r.ecl), 2),
                               "pd_moyenne_pct": round(100 * float(r.pd_moyenne), 3),
                               "taux_couverture_pct": round(float(r.taux_couverture_pct), 3)}
                     for k, r in g.iterrows()},
        "exposition_totale": round(total_exp, 2),
        "ecl_totale": round(total_ecl, 2),
        "taux_couverture_global_pct": round(100 * total_ecl / total_exp, 3),
        "hypotheses": {
            "taux_actualisation": TAUX_ACTUALISATION,
            "asrc_multiple_pd_vie": SEUIL_ASRC_MULTIPLE,
            "asrc_hausse_absolue": SEUIL_ASRC_ABSOLU,
            "pd_vie": "survie a hasard constant depuis la PD a 12 mois",
        },
    }
