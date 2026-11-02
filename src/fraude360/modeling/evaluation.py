"""Metriques et economie de la decision - utilise par l'entrainement, le test A/B,
la surveillance et le reentrainement.

Sur un probleme a 0,4 % de positifs, l'exactitude (accuracy) n'a aucun sens :
un modele qui ne detecte rien affiche 99,6 %. On raisonne donc avec
  - PR-AUC (aire sous la courbe precision-rappel) : metrique de selection ;
  - rappel a precision fixee et precision a budget d'alertes fixe ;
  - et surtout le COUT ATTENDU EN DOLLARS, qui est le langage de l'affaire.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)


@dataclass
class Metriques:
    pr_auc: float
    roc_auc: float
    brier: float
    seuil: float
    precision: float
    rappel: float
    f1: float
    vrais_positifs: int
    faux_positifs: int
    faux_negatifs: int
    vrais_negatifs: int
    nb_alertes: int
    taux_alerte_pct: float
    montant_fraude_total: float
    montant_fraude_evite: float
    montant_fraude_subi: float
    cout_total: float
    cout_par_1000_tx: float
    rappel_en_dollars: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def cout_metier(y: np.ndarray, prediction: np.ndarray, montants: np.ndarray,
                economie: dict) -> Dict[str, float]:
    """Traduit une matrice de confusion en dollars.

    - faux negatif  : la fraude passe -> on perd le montant de la transaction ;
    - faux positif  : revue manuelle inutile + friction pour le membre ;
    - vrai positif  : la fraude est bloquee, mais l'alerte a un cout de traitement.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(prediction).astype(int)
    m = np.asarray(montants, dtype=float)

    vp = (p == 1) & (y == 1)
    fp = (p == 1) & (y == 0)
    fn = (p == 0) & (y == 1)
    vn = (p == 0) & (y == 0)

    perte_fn = m[fn].sum() * economie["cout_faux_negatif_ratio"]
    cout_fp = fp.sum() * economie["cout_faux_positif"]
    cout_vp = vp.sum() * economie["cout_vrai_positif"]
    return {
        "vp": int(vp.sum()), "fp": int(fp.sum()),
        "fn": int(fn.sum()), "vn": int(vn.sum()),
        "montant_evite": float(m[vp].sum()),
        "montant_subi": float(m[fn].sum()),
        "cout_total": float(perte_fn + cout_fp + cout_vp),
    }


def seuil_optimal(y: np.ndarray, scores: np.ndarray, montants: np.ndarray,
                  economie: dict, budget_alertes: int | None = None,
                  nb_jours: int | None = None) -> tuple[float, pd.DataFrame]:
    """Balaye 200 seuils et retient celui qui minimise le cout attendu.

    Si un budget d'alertes quotidien est fourni, les seuils qui saturent la
    capacite de l'equipe d'enquete sont ecartes : un modele qu'on ne peut pas
    traiter operationnellement n'a aucune valeur.
    """
    # La grille combine des quantiles et les valeurs distinctes les plus hautes.
    # La calibration isotonique produit des scores en escalier : une grille
    # purement quantile s'effondrerait sur deux ou trois valeurs.
    grille = np.unique(np.concatenate([
        np.quantile(scores, np.linspace(0.85, 0.99999, 200)),
        np.unique(scores)[-150:],
    ]))
    lignes = []
    for s in grille:
        pred = (scores >= s).astype(int)
        c = cout_metier(y, pred, montants, economie)
        alertes = c["vp"] + c["fp"]
        alertes_jour = alertes / nb_jours if nb_jours else None
        lignes.append(dict(seuil=float(s), **c, nb_alertes=alertes,
                           alertes_par_jour=alertes_jour))
    tab = pd.DataFrame(lignes)
    admissibles = tab
    if budget_alertes and nb_jours:
        filtre = tab[tab.alertes_par_jour <= budget_alertes]
        if len(filtre) > 0:
            admissibles = filtre
    meilleur = admissibles.loc[admissibles.cout_total.idxmin()]
    return float(meilleur.seuil), tab


def evaluer(y: np.ndarray, scores: np.ndarray, montants: np.ndarray,
            seuil: float, economie: dict) -> Metriques:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= seuil).astype(int)
    c = cout_metier(y, pred, montants, economie)

    precision = c["vp"] / max(c["vp"] + c["fp"], 1)
    rappel = c["vp"] / max(c["vp"] + c["fn"], 1)
    f1 = 2 * precision * rappel / max(precision + rappel, 1e-9)
    total_fraude = c["montant_evite"] + c["montant_subi"]

    return Metriques(
        pr_auc=float(average_precision_score(y, scores)),
        roc_auc=float(roc_auc_score(y, scores)),
        brier=float(brier_score_loss(y, np.clip(scores, 0, 1))),
        seuil=float(seuil),
        precision=float(precision), rappel=float(rappel), f1=float(f1),
        vrais_positifs=c["vp"], faux_positifs=c["fp"],
        faux_negatifs=c["fn"], vrais_negatifs=c["vn"],
        nb_alertes=c["vp"] + c["fp"],
        taux_alerte_pct=float(100 * (c["vp"] + c["fp"]) / len(y)),
        montant_fraude_total=round(total_fraude, 2),
        montant_fraude_evite=round(c["montant_evite"], 2),
        montant_fraude_subi=round(c["montant_subi"], 2),
        cout_total=round(c["cout_total"], 2),
        cout_par_1000_tx=round(1000 * c["cout_total"] / len(y), 2),
        rappel_en_dollars=float(c["montant_evite"] / max(total_fraude, 1e-9)),
    )


def rappel_a_precision(y, scores, precision_cible: float = 0.5) -> float:
    """Rappel atteignable en maintenant au moins `precision_cible` de precision."""
    p, r, _ = precision_recall_curve(y, scores)
    ok = p[:-1] >= precision_cible
    return float(r[:-1][ok].max()) if ok.any() else 0.0


def precision_a_k(y, scores, k: int) -> float:
    """Precision sur les k transactions les plus suspectes (budget d'enquete)."""
    idx = np.argsort(-np.asarray(scores))[:k]
    return float(np.asarray(y)[idx].mean())
