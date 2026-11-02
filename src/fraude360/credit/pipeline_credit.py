"""Orchestrateur du volet CREDIT.

    python -m fraude360.credit.pipeline_credit

Enchaine : portefeuille -> grille de score -> IFRS 9 -> validation independante,
puis produit les artefacts et les exports Power BI.

Le decoupage developpement / validation est TEMPOREL, comme cote fraude : les
millesimes 0 a 23 servent au developpement, 24 a 35 a la validation. C'est ce
decoupage qui rend visibles le relachement des criteres d'octroi et la
degradation macroeconomique -- un tirage aleatoire les aurait dilues dans les
deux echantillons et la validation n'aurait rien trouve.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.credit import ifrs9, portefeuille, scorecard, validation
from fraude360.utils.journal import chrono, ecrire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.credit.pipeline")

VAR_NUM = ["score_bureau", "ratio_endettement", "nb_retards_12m",
           "anciennete_emploi", "utilisation_marge", "ratio_pret_valeur",
           "duree_mois", "age", "revenu", "montant"]
VAR_CAT = ["produit", "region"]


def _figures(dossier, d_val, pd_val, score_val, grille, rap):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # 1. distribution du score par issue
    for v, lib, c in [(0, "sains", "#2e7d32"), (1, "defauts", "#c62828")]:
        ax[0, 0].hist(score_val[d_val.defaut_12m.values == v], bins=40, alpha=0.6,
                      label=lib, color=c, density=True)
    ax[0, 0].set_title("Distribution du score (validation)")
    ax[0, 0].set_xlabel("points"); ax[0, 0].legend()

    # 2. calibration par decile
    dec = rap["calibration"]["deciles"]
    x = [d["pd_moyenne_pct"] for d in dec]
    y = [d["taux_observe_pct"] for d in dec]
    ax[0, 1].plot([0, max(x + y)], [0, max(x + y)], "--", c="grey", lw=1)
    ax[0, 1].scatter(x, y, s=45, c="#1f4e79")
    ax[0, 1].set_xlabel("PD moyenne predite (%)")
    ax[0, 1].set_ylabel("taux de defaut observe (%)")
    ax[0, 1].set_title("Calibration par decile (Hosmer-Lemeshow)")

    # 3. retrovalidation
    r = pd.DataFrame(rap["retrovalidation"]["par_millesime"])
    ax[1, 0].plot(r.millesime, r.pd_predite_pct, "-o", ms=4, label="PD predite")
    ax[1, 0].plot(r.millesime, r.defaut_observe_pct, "-s", ms=4, label="defaut observe")
    ax[1, 0].set_title("Retrovalidation par millesime")
    ax[1, 0].set_xlabel("millesime"); ax[1, 0].legend()

    # 4. valeur d'information des variables
    iv = pd.Series(grille.iv).sort_values()
    ax[1, 1].barh(iv.index, iv.values, color="#b8860b")
    ax[1, 1].axvline(0.02, ls="--", c="crimson", lw=1)
    ax[1, 1].set_title("Valeur d'information par variable")

    fig.tight_layout()
    fig.savefig(dossier / "40_credit_validation.png", dpi=130)
    plt.close(fig)


def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("VOLET CREDIT - Grille de score, IFRS 9 et validation independante")
    dossier = cfg.chemin("artefacts") / "credit"
    dossier.mkdir(parents=True, exist_ok=True)

    d = portefeuille.executer(cfg)
    coupure = int(d.millesime.max()) - 11
    dev, val = d[d.millesime < coupure], d[d.millesime >= coupure]
    LOG.info("Developpement : %d prets (millesimes < %d) | Validation : %d",
             len(dev), coupure, len(val))

    with chrono(LOG, "Ajustement de la grille de score (WOE + regression logistique)"):
        grille = scorecard.ajuster(dev, dev.defaut_12m.values, VAR_NUM, VAR_CAT)
    LOG.info("Variables retenues : %s", list(grille.decoupages))

    with chrono(LOG, "Validation independante"):
        rap = validation.valider(dev, val, grille, VAR_NUM)

    # --- IFRS 9 sur le portefeuille de validation --------------------------
    pd_val, score_val = scorecard.predire(val, grille)
    # PD a l'octroi : la grille appliquee aux caracteristiques d'origine, ici
    # approchee en neutralisant la degradation comportementale observee depuis.
    val_octroi = val.assign(nb_retards_12m=0,
                            utilisation_marge=val.utilisation_marge * 0.6)
    pd_octroi, _ = scorecard.predire(val_octroi, grille)
    portefeuille_ifrs9 = ifrs9.calculer(val, pd_val, pd_octroi)
    synth = ifrs9.synthese(portefeuille_ifrs9)

    _figures(dossier, val, pd_val, score_val, grille, rap)

    # --- artefacts ----------------------------------------------------------
    tables_woe = pd.concat(
        [dec.table.assign(variable=v) for v, dec in grille.decoupages.items()],
        ignore_index=True)
    tables_woe.to_csv(dossier / "tables_woe.csv", index=False, encoding="utf-8-sig")
    portefeuille_ifrs9.to_parquet(dossier / "portefeuille_ifrs9.parquet", index=False)

    grille_export = {
        "intercept": grille.intercept, "facteur": grille.facteur,
        "decalage": grille.decalage, "coefficients": grille.coefficients,
        "valeur_information": grille.iv, "points": grille.points,
        "bornes": {v: dec.bornes for v, dec in grille.decoupages.items()},
        "monotonie": {v: dec.monotone for v, dec in grille.decoupages.items()},
    }
    ecrire_json(dossier / "grille_score.json", grille_export)

    rapport = {"decoupage": {"coupure_millesime": coupure,
                             "nb_developpement": len(dev), "nb_validation": len(val),
                             "taux_defaut_dev_pct": round(100 * dev.defaut_12m.mean(), 3),
                             "taux_defaut_val_pct": round(100 * val.defaut_12m.mean(), 3)},
               "validation": rap, "ifrs9": synth}
    ecrire_json(dossier / "rapport_credit.json", rapport)

    # export Power BI
    bi = portefeuille_ifrs9[["pret_id", "millesime", "produit", "region",
                             "exposition", "pd_12m", "pd_vie", "lgd_observee",
                             "etape_ifrs9", "ecl", "defaut_12m"]]
    bi.to_csv(cfg.chemin("exports_bi") / "fait_credit_ifrs9.csv",
              index=False, encoding="utf-8-sig")

    LOG.info("Gini validation %.4f | KS %.4f | PSI score %.4f",
             rap["discrimination"]["validation"]["gini"],
             rap["discrimination"]["validation"]["ks"],
             rap["stabilite"]["psi_score"])
    LOG.info("ECL totale %.2f $ sur %.2f $ d'exposition (%.3f %%)",
             synth["ecl_totale"], synth["exposition_totale"],
             synth["taux_couverture_global_pct"])
    LOG.info("Conclusion de validation : %s", rap["conclusion"])
    for c in rap["constats"]:
        LOG.info("   - %s", c)
    return rapport


if __name__ == "__main__":
    executer()
