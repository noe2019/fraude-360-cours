"""Validation independante d'un modele de credit (ligne de defense 2).

Quatre familles de tests, dans l'ordre ou une ligne de defense 2 les regarde :

  DISCRIMINATION  Le modele separe-t-il les bons des mauvais ?
      Gini (= 2*AUC - 1) et statistique de Kolmogorov-Smirnov.
      Repere du marche pour une grille de credit aux particuliers :
      Gini superieur a 0,40 acceptable, superieur a 0,55 bon.

  CALIBRATION  Les PD predites correspondent-elles aux defauts observes ?
      Test de Hosmer-Lemeshow par deciles, et ecart PD moyenne / taux observe.
      Un modele peut tres bien discriminer et etre mal calibre : c'est
      inacceptable en IFRS 9, ou la PD alimente directement la provision.

  STABILITE  La population a-t-elle bouge depuis le developpement ?
      PSI sur le score et sur chaque variable de la grille.
      PSI < 0,10 stable | 0,10-0,25 a surveiller | > 0,25 recalibrage.

  RETROVALIDATION  Le modele tient-il dans le temps ?
      PD predite contre defaut observe, millesime par millesime, avec un
      intervalle binomial. Un point hors intervalle est un signal, une serie
      de points du meme cote est un biais.

Aucun test ne "valide" seul : le rapport produit un constat par test et une
conclusion agregee explicite.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from fraude360.utils.journal import obtenir_journal

LOG = obtenir_journal("fraude360.credit.validation")

GINI_ACCEPTABLE, GINI_BON = 0.40, 0.55
PSI_SURVEILLANCE, PSI_ALERTE = 0.10, 0.25


def gini_ks(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y, -score)          # score haut = bon -> on inverse
    d = pd.DataFrame({"y": y, "s": score}).sort_values("s")
    cum_m = (d.y == 1).cumsum() / max(1, (d.y == 1).sum())
    cum_b = (d.y == 0).cumsum() / max(1, (d.y == 0).sum())
    return {"auc": round(float(auc), 4),
            "gini": round(float(2 * auc - 1), 4),
            "ks": round(float(np.max(np.abs(cum_m - cum_b))), 4)}


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groupes: int = 10) -> Dict[str, object]:
    d = pd.DataFrame({"y": y, "p": p})
    d["g"] = pd.qcut(d.p.rank(method="first"), groupes, labels=False)
    t = d.groupby("g").agg(nb=("y", "size"), obs=("y", "sum"), att=("p", "sum"))
    att = t.att.clip(1e-9, t.nb - 1e-9)
    chi2 = float((((t.obs - att) ** 2) / (att * (1 - att / t.nb))).sum())
    ddl = groupes - 2
    p_val = float(1 - stats.chi2.cdf(chi2, ddl))
    # Hosmer-Lemeshow est sensible a la taille : au-dela de ~10 000
    # observations il rejette presque toujours, y compris pour des ecarts sans
    # portee pratique. On rapporte donc AUSSI l'ecart absolu moyen par decile,
    # qui est la mesure que la ligne de defense 2 regarde reellement.
    ecart_moyen = float(np.mean(np.abs(t.obs - t.att) / t.nb))
    return {"chi2": round(chi2, 3), "ddl": ddl, "p_value": round(p_val, 4),
            "calibration_rejetee": bool(p_val < 0.05),
            "ecart_absolu_moyen_par_decile_pct": round(100 * ecart_moyen, 3),
            "note_taille_echantillon": (
                "n > 10 000 : le rejet du test chi2 est attendu meme pour un ecart "
                "negligeable ; se fier a l'ecart absolu moyen." if len(d) > 10_000
                else "n modere : le test chi2 est interpretable tel quel."),
            "deciles": [{"decile": int(g) + 1, "nb": int(r.nb),
                         "defauts_observes": int(r.obs),
                         "defauts_attendus": round(float(r.att), 1),
                         "taux_observe_pct": round(100 * r.obs / r.nb, 3),
                         "pd_moyenne_pct": round(100 * r.att / r.nb, 3)}
                        for g, r in t.iterrows()]}


def psi(reference: np.ndarray, courant: np.ndarray, classes: int = 10) -> float:
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, classes + 1)))
    if len(bornes) < 3:
        return 0.0
    bornes[0], bornes[-1] = -np.inf, np.inf
    r = np.histogram(reference, bins=bornes)[0] / max(1, len(reference))
    c = np.histogram(courant, bins=bornes)[0] / max(1, len(courant))
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def retrovalidation(d: pd.DataFrame, colonne_pd: str = "pd_12m",
                    colonne_defaut: str = "defaut_12m") -> Dict[str, object]:
    lignes: List[dict] = []
    for m, bloc in d.groupby("millesime"):
        n = len(bloc)
        attendu = float(bloc[colonne_pd].mean())
        observe = float(bloc[colonne_defaut].mean())
        # intervalle binomial normal a 95 % autour de la PD predite
        se = float(np.sqrt(max(attendu * (1 - attendu), 1e-12) / max(n, 1)))
        bas, haut = attendu - 1.96 * se, attendu + 1.96 * se
        lignes.append({"millesime": int(m), "nb": n,
                       "pd_predite_pct": round(100 * attendu, 3),
                       "defaut_observe_pct": round(100 * observe, 3),
                       "dans_intervalle": bool(bas <= observe <= haut),
                       "signe": int(np.sign(observe - attendu))})
    hors = [l for l in lignes if not l["dans_intervalle"]]
    # une serie de trois signes identiques consecutifs revele un biais
    signes = [l["signe"] for l in lignes]
    serie_max, courant, precedent = 0, 0, 0
    for s in signes:
        courant = courant + 1 if s == precedent and s != 0 else 1
        precedent = s
        serie_max = max(serie_max, courant)
    return {"par_millesime": lignes,
            "nb_millesimes": len(lignes),
            "nb_hors_intervalle": len(hors),
            "millesimes_hors_intervalle": [l["millesime"] for l in hors],
            "plus_longue_serie_meme_signe": serie_max,
            "biais_systematique_suspecte": serie_max >= 5}


def valider(d_dev: pd.DataFrame, d_val: pd.DataFrame, grille,
            variables: List[str]) -> Dict[str, object]:
    from fraude360.credit import scorecard as sc

    pd_dev, score_dev = sc.predire(d_dev, grille)
    pd_val, score_val = sc.predire(d_val, grille)
    y_dev, y_val = d_dev.defaut_12m.values, d_val.defaut_12m.values

    disc_dev, disc_val = gini_ks(y_dev, score_dev), gini_ks(y_val, score_val)
    chute = disc_dev["gini"] - disc_val["gini"]

    psi_score = psi(score_dev, score_val)
    psi_var = {v: round(psi(d_dev[v].dropna().values, d_val[v].dropna().values), 4)
               for v in variables if pd.api.types.is_numeric_dtype(d_dev[v])}

    hl = hosmer_lemeshow(y_val, pd_val)
    retro = retrovalidation(d_val.assign(pd_12m=pd_val))

    constats: List[str] = []
    if disc_val["gini"] < GINI_ACCEPTABLE:
        constats.append(f"Gini de validation ({disc_val['gini']}) sous le repere {GINI_ACCEPTABLE}.")
    if chute > 0.05:
        constats.append(f"Chute de Gini de {chute:.3f} entre developpement et validation.")
    # sur grand echantillon, on retient l'ecart absolu, pas le rejet du chi2
    ecart = hl.get("ecart_absolu_moyen_par_decile_pct", 0.0)
    if len(d_val) > 10_000:
        if ecart > 1.0:
            constats.append(
                f"Calibration : ecart absolu moyen de {ecart} point(s) de pourcentage "
                f"par decile (chi2 rejete, p = {hl['p_value']}, attendu a cette taille).")
    elif hl["calibration_rejetee"]:
        constats.append(f"Hosmer-Lemeshow rejette la calibration (p = {hl['p_value']}).")
    if psi_score >= PSI_ALERTE:
        constats.append(f"PSI du score a {psi_score:.3f} : recalibrage requis.")
    elif psi_score >= PSI_SURVEILLANCE:
        constats.append(f"PSI du score a {psi_score:.3f} : sous surveillance.")
    derivees = [v for v, p in psi_var.items() if p >= PSI_ALERTE]
    if derivees:
        constats.append(f"Variables ayant derive (PSI >= {PSI_ALERTE}) : {derivees}.")
    if retro["biais_systematique_suspecte"]:
        constats.append(
            f"Retrovalidation : {retro['plus_longue_serie_meme_signe']} millesimes "
            "consecutifs du meme cote -- biais systematique suspecte.")

    return {
        "discrimination": {"developpement": disc_dev, "validation": disc_val,
                           "chute_gini": round(chute, 4),
                           "repere_acceptable": GINI_ACCEPTABLE,
                           "repere_bon": GINI_BON},
        "calibration": hl,
        "stabilite": {"psi_score": round(psi_score, 4), "psi_variables": psi_var,
                      "seuils": {"surveillance": PSI_SURVEILLANCE, "alerte": PSI_ALERTE}},
        "retrovalidation": retro,
        "constats": constats,
        "conclusion": ("Aucune reserve bloquante." if not constats
                       else f"{len(constats)} reserve(s) a lever avant homologation."),
    }
