"""Generation d'un portefeuille de prets aux particuliers, avec defauts observes.

Le portefeuille est genere sur 36 millesimes mensuels. Chaque pret porte ses
caracteristiques a l'octroi, un comportement observe, et un indicateur de defaut
a 12 mois (90 jours de retard ou plus). La generation introduit volontairement :

  - une relation MONOTONE entre les variables structurantes et le risque,
    pour que la discretisation WOE ait un sens a verifier ;
  - une DERIVE de population a partir du millesime 24 (relachement des criteres
    d'octroi), pour que le PSI ait quelque chose a detecter ;
  - une degradation macroeconomique sur les douze derniers millesimes, pour que
    la retrovalidation montre un ecart entre PD predite et defaut observe.

Sans ces trois defauts deliberes, la validation n'aurait rien a trouver et ne
demontrerait rien.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.utils.journal import chrono, obtenir_journal, titre

LOG = obtenir_journal("fraude360.credit.portefeuille")

PRODUITS = ["pret_personnel", "marge_credit", "auto", "renovation"]
REGIONS = ["montreal", "quebec", "estrie", "outaouais", "saguenay"]


def generer(cfg: Config | None = None, n: int = 60_000,
            millesimes: int = 36) -> pd.DataFrame:
    cfg = cfg or charger_config()
    rng = np.random.default_rng(cfg.seed)

    mois = rng.integers(0, millesimes, n)
    # --- caracteristiques a l'octroi ---------------------------------------
    age = np.clip(rng.normal(43, 13, n), 18, 85).round().astype(int)
    anciennete_emploi = np.clip(rng.gamma(2.0, 3.0, n), 0, 40).round(1)
    revenu = np.clip(rng.lognormal(11.0, 0.45, n), 18_000, 400_000).round(-2)
    montant = np.clip(rng.lognormal(9.6, 0.7, n), 1_000, 150_000).round(-2)
    duree_mois = rng.choice([12, 24, 36, 48, 60, 72, 84], n,
                            p=[.05, .12, .25, .25, .20, .08, .05])
    produit = rng.choice(PRODUITS, n, p=[.40, .25, .25, .10])
    region = rng.choice(REGIONS, n, p=[.45, .20, .15, .12, .08])

    # score de bureau : correle au revenu et a l'anciennete, avec du bruit
    bureau = np.clip(
        620 + 0.00035 * revenu + 3.2 * anciennete_emploi + rng.normal(0, 55, n),
        300, 900).round().astype(int)

    # ratio d'endettement : dette totale / revenu
    autres_dettes = np.clip(rng.lognormal(9.3, 0.9, n), 0, 300_000)
    ratio_endettement = np.clip((montant + autres_dettes) / revenu, 0.01, 6.0).round(3)

    # ratio pret/valeur, pertinent pour auto et renovation
    valeur_bien = np.where(np.isin(produit, ["auto", "renovation"]),
                           montant * rng.uniform(1.0, 1.9, n), np.nan)
    ratio_pret_valeur = np.where(np.isnan(valeur_bien), np.nan,
                                 np.clip(montant / valeur_bien, 0.2, 1.2))

    nb_retards_12m = rng.poisson(
        np.clip(0.9 - 0.0009 * (bureau - 600), 0.03, 2.2), n)
    utilisation_marge = np.clip(rng.beta(2, 3, n) + 0.15 * ratio_endettement, 0, 1.5).round(3)

    # --- relachement des criteres a partir du millesime 24 ------------------
    relachement = (mois >= 24)
    bureau = np.where(relachement,
                      np.clip(bureau - rng.integers(0, 45, n), 300, 900), bureau)
    ratio_endettement = np.where(relachement,
                                 np.clip(ratio_endettement * 1.18, 0.01, 6.0),
                                 ratio_endettement)

    # --- probabilite de defaut, monotone par construction ------------------
    logit = (-4.35
             - 0.0072 * (bureau - 650)
             + 1.28 * (ratio_endettement - 1.0)
             + 0.34 * nb_retards_12m
             - 0.031 * anciennete_emploi
             + 0.85 * np.nan_to_num(ratio_pret_valeur, nan=0.55)
             + 0.55 * utilisation_marge
             + 0.0055 * (duree_mois - 48)
             - 0.011 * (age - 43))
    # degradation macro sur les douze derniers millesimes
    logit += np.where(mois >= millesimes - 12, 0.42, 0.0)
    logit += rng.normal(0, 0.32, n)
    pd_reelle = 1.0 / (1.0 + np.exp(-logit))
    defaut = (rng.uniform(size=n) < pd_reelle).astype(int)

    # exposition et perte en cas de defaut
    exposition = (montant * np.clip(rng.beta(6, 2, n) + 0.10, 0, 1.0)).round(2)
    garanti = np.isin(produit, ["auto", "renovation"])
    lgd = np.clip(np.where(garanti, rng.normal(0.38, 0.11, n),
                           rng.normal(0.68, 0.13, n)), 0.05, 0.95).round(3)

    d = pd.DataFrame({
        "pret_id": [f"PR{i:07d}" for i in range(n)],
        "millesime": mois,
        "produit": produit,
        "region": region,
        "age": age,
        "anciennete_emploi": anciennete_emploi,
        "revenu": revenu,
        "montant": montant,
        "duree_mois": duree_mois,
        "score_bureau": bureau,
        "ratio_endettement": ratio_endettement,
        "ratio_pret_valeur": np.round(ratio_pret_valeur, 3),
        "nb_retards_12m": nb_retards_12m,
        "utilisation_marge": utilisation_marge,
        "exposition": exposition,
        "lgd_observee": lgd,
        "defaut_12m": defaut,
    })
    return d.sort_values("millesime").reset_index(drop=True)


def executer(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or charger_config()
    titre("CREDIT 00 - Generation du portefeuille de prets")
    c = (cfg.get("credit") or {})
    with chrono(LOG, "Generation du portefeuille"):
        d = generer(cfg, n=int(c.get("nb_prets", 60_000)),
                    millesimes=int(c.get("nb_millesimes", 36)))
    dossier = cfg.chemin("gold") / "credit"
    dossier.mkdir(parents=True, exist_ok=True)
    d.to_parquet(dossier / "portefeuille.parquet", index=False)
    LOG.info("%d prets, taux de defaut global %.2f %%, %d millesimes",
             len(d), 100 * d.defaut_12m.mean(), d.millesime.nunique())
    return d
