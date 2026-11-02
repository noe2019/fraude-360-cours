"""Grille de score de credit : discretisation WOE, valeur d'information, points.

Pourquoi WOE et pas un encodage direct. La transformation en poids de la preuve
(weight of evidence) rend la relation lineaire dans le logit, absorbe les valeurs
extremes, traite les valeurs manquantes comme une classe a part entiere, et
surtout produit un modele que l'on peut lire ligne par ligne. C'est la forme
attendue dans un dossier de modele de credit.

        WOE(classe) = ln( part des bons dans la classe / part des mauvais )
        IV = somme sur les classes de (part_bons - part_mauvais) * WOE

Lecture usuelle de l'IV : < 0,02 inutile | 0,02-0,1 faible | 0,1-0,3 moyen |
0,3-0,5 fort | > 0,5 suspect (souvent une fuite d'information).

MONOTONIE. Un decoupage dont le WOE n'est pas monotone sur une variable
ordonnee est refuse et refusionne : sur `ratio_endettement`, le risque doit
croitre a chaque classe, sinon le decoupage capte du bruit d'echantillon.

MISE EN POINTS. La grille est exprimee en points, convention du metier :
    facteur = PDO / ln(2)      decalage = score_ref - facteur * ln(cote_ref)
avec par defaut 600 points pour une cote de 50 contre 1, et 20 points pour
doubler la cote.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraude360.utils.journal import obtenir_journal

LOG = obtenir_journal("fraude360.credit.scorecard")

EPS = 0.5  # correction de Laplace, evite un WOE infini sur une classe vide

VARIABLES_ORDONNEES = {
    "score_bureau": False,        # plus haut = moins risque -> WOE decroissant
    "ratio_endettement": True,
    "nb_retards_12m": True,
    "anciennete_emploi": False,
    "utilisation_marge": True,
    "ratio_pret_valeur": True,
    "duree_mois": True,
    "age": False,
    "revenu": False,
    "montant": True,
}


@dataclass
class Decoupage:
    variable: str
    bornes: List[float]
    table: pd.DataFrame            # classe, nb, nb_mauvais, woe, iv_partiel
    iv: float
    monotone: bool
    categorielle: bool = False
    modalites: Dict[str, float] = field(default_factory=dict)


def _woe_table(classes: pd.Series, y: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"classe": classes.astype(str), "y": y})
    g = d.groupby("classe", observed=True).agg(nb=("y", "size"), nb_mauvais=("y", "sum"))
    g["nb_bons"] = g.nb - g.nb_mauvais
    tot_b, tot_m = g.nb_bons.sum(), g.nb_mauvais.sum()
    g["part_bons"] = (g.nb_bons + EPS) / (tot_b + EPS * len(g))
    g["part_mauvais"] = (g.nb_mauvais + EPS) / (tot_m + EPS * len(g))
    g["woe"] = np.log(g.part_bons / g.part_mauvais)
    g["iv_partiel"] = (g.part_bons - g.part_mauvais) * g.woe
    g["taux_defaut"] = g.nb_mauvais / g.nb
    return g.reset_index()


def _monotone(taux: np.ndarray, croissant: bool) -> bool:
    d = np.diff(taux)
    return bool(np.all(d >= -1e-9) if croissant else np.all(d <= 1e-9))


def discretiser_numerique(x: pd.Series, y: np.ndarray, variable: str,
                          n_classes: int = 6, min_part: float = 0.05) -> Decoupage:
    """Decoupage par quantiles, puis fusion tant que la monotonie n'est pas atteinte."""
    croissant = VARIABLES_ORDONNEES.get(variable, True)
    manquant = x.isna()
    xv, yv = x[~manquant], y[~manquant]

    k = n_classes
    while k >= 2:
        try:
            classes, bornes = pd.qcut(xv, k, retbins=True, duplicates="drop",
                                      labels=False)
        except ValueError:
            k -= 1
            continue
        t = _woe_table(pd.Series(classes, index=xv.index), yv)
        assez_peuplees = (t.nb / len(xv) >= min_part).all()
        if assez_peuplees and _monotone(t.taux_defaut.values, croissant):
            break
        k -= 1
    else:  # pragma: no cover
        classes, bornes = pd.qcut(xv, 2, retbins=True, duplicates="drop", labels=False)
        t = _woe_table(pd.Series(classes, index=xv.index), yv)

    # les valeurs manquantes forment leur propre classe, jamais imputees
    if manquant.any():
        tm = _woe_table(pd.Series(["MANQUANT"] * int(manquant.sum()),
                                  index=x[manquant].index), y[manquant.values])
        t = pd.concat([t, tm], ignore_index=True)

    return Decoupage(variable=variable, bornes=list(np.round(bornes, 4)), table=t,
                     iv=float(t.iv_partiel.sum()),
                     monotone=_monotone(
                         t[t.classe != "MANQUANT"].taux_defaut.values, croissant))


def discretiser_categorielle(x: pd.Series, y: np.ndarray, variable: str) -> Decoupage:
    t = _woe_table(x.fillna("MANQUANT"), y)
    return Decoupage(variable=variable, bornes=[], table=t,
                     iv=float(t.iv_partiel.sum()), monotone=True, categorielle=True,
                     modalites=dict(zip(t.classe, t.woe)))


def appliquer(d: pd.DataFrame, dec: Decoupage) -> pd.Series:
    """Transforme une colonne brute en sa valeur WOE."""
    if dec.categorielle:
        m = dec.table.set_index("classe").woe
        return d[dec.variable].fillna("MANQUANT").astype(str).map(m).fillna(0.0)

    x = d[dec.variable]
    bornes = list(dec.bornes)
    bornes[0], bornes[-1] = -np.inf, np.inf
    idx = pd.cut(x, bins=bornes, labels=False, include_lowest=True)
    cle = pd.Series(np.where(x.isna(), "MANQUANT", pd.Series(idx, index=x.index)
                             .astype("Int64").astype(str)), index=x.index)
    m = dec.table.set_index("classe").woe
    return cle.map(m).astype(float).fillna(0.0)


@dataclass
class Grille:
    decoupages: Dict[str, Decoupage]
    coefficients: Dict[str, float]
    intercept: float
    facteur: float
    decalage: float
    points: Dict[str, Dict[str, float]]
    iv: Dict[str, float]


def ajuster(d: pd.DataFrame, y: np.ndarray, variables_num: List[str],
            variables_cat: List[str], iv_min: float = 0.02,
            pdo: int = 20, score_ref: int = 600, cote_ref: float = 50.0) -> Grille:
    decoupages: Dict[str, Decoupage] = {}
    for v in variables_num:
        decoupages[v] = discretiser_numerique(d[v], y, v)
    for v in variables_cat:
        decoupages[v] = discretiser_categorielle(d[v], y, v)

    retenues = [v for v, dec in decoupages.items() if dec.iv >= iv_min]
    ecartees = {v: round(dec.iv, 4) for v, dec in decoupages.items() if dec.iv < iv_min}
    if ecartees:
        LOG.info("Variables ecartees pour IV insuffisant : %s", ecartees)

    X = pd.DataFrame({v: appliquer(d, decoupages[v]) for v in retenues})
    modele = LogisticRegression(max_iter=1000, C=1e6)  # peu de regularisation :
    modele.fit(X, y)                                    # les WOE sont deja lisses

    coefs = dict(zip(retenues, modele.coef_[0]))
    facteur = pdo / np.log(2)
    decalage = score_ref - facteur * np.log(cote_ref)

    # repartition des points par classe : convention standard de l'industrie
    n = len(retenues)
    points: Dict[str, Dict[str, float]] = {}
    for v in retenues:
        points[v] = {
            str(r.classe): round(
                -(coefs[v] * r.woe + modele.intercept_[0] / n) * facteur + decalage / n, 2)
            for _, r in decoupages[v].table.iterrows()}

    return Grille(decoupages={v: decoupages[v] for v in retenues},
                  coefficients=coefs, intercept=float(modele.intercept_[0]),
                  facteur=float(facteur), decalage=float(decalage), points=points,
                  iv={v: round(decoupages[v].iv, 4) for v in decoupages})


def predire(d: pd.DataFrame, g: Grille) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (probabilite de defaut, score en points)."""
    logit = np.full(len(d), g.intercept, dtype=float)
    for v, dec in g.decoupages.items():
        logit += g.coefficients[v] * appliquer(d, dec).values
    pd_ = 1.0 / (1.0 + np.exp(-logit))
    # score = decalage + facteur * ln(cote de bon), cote = (1-pd)/pd
    score = g.decalage + g.facteur * np.log(np.clip((1 - pd_) / np.clip(pd_, 1e-9, 1), 1e-9, None))
    return pd_, np.round(score, 1)
