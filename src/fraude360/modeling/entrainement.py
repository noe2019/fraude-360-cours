"""ETAPE 05 - Entrainement, selection, calibration et enregistrement du modele.

Demarche :
  1. decoupage temporel (entrainement / validation / test) ;
  2. pipeline scikit-learn unique : imputation + encodage + modele, pour que
     l'objet servi en production soit exactement celui qui a ete evalue ;
  3. trois familles de modeles comparees sur la PR-AUC de validation ;
  4. calibration des probabilites (isotonique) : sans elle, un score de 0,8 ne
     veut pas dire 80 % de risque, et le seuil economique n'a plus de sens ;
  5. choix du seuil de decision par MINIMISATION DU COUT EN DOLLARS, sous
     contrainte du budget quotidien d'alertes de l'equipe d'enquete ;
  6. evaluation finale sur le jeu de test, jamais utilise auparavant ;
  7. tracage complet dans MLflow et enregistrement au registre (alias champion).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraude360.config import Config, charger_config
from fraude360.modeling import donnees, evaluation, registre
from fraude360.utils.journal import chrono, ecrire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.entrainement")


# =============================================================================
#  Pipelines
# =============================================================================
def _preparation(num: list[str], cat: list[str], mise_a_echelle: bool) -> ColumnTransformer:
    etapes_num = [("imputation", SimpleImputer(strategy="median"))]
    if mise_a_echelle:
        etapes_num.append(("echelle", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(etapes_num), num),
        ("cat", Pipeline([
            ("imputation", SimpleImputer(strategy="most_frequent")),
            ("encodage", OneHotEncoder(handle_unknown="ignore", min_frequency=30,
                                       sparse_output=False)),
        ]), cat),
    ], remainder="drop", verbose_feature_names_out=False)


def construire_modeles(num: list[str], cat: list[str], seed: int) -> dict[str, Pipeline]:
    return {
        # Reference interpretable, exigee par la gouvernance des modeles
        "regression_logistique": Pipeline([
            ("preparation", _preparation(num, cat, True)),
            ("modele", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          C=0.5, n_jobs=None, random_state=seed)),
        ]),
        # Robuste, capture les interactions, peu sensible aux valeurs extremes
        "foret_aleatoire": Pipeline([
            ("preparation", _preparation(num, cat, False)),
            ("modele", RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=15,
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
        ]),
        # Etat de l'art sur donnees tabulaires desequilibrees
        "gradient_boosting": Pipeline([
            ("preparation", _preparation(num, cat, False)),
            ("modele", HistGradientBoostingClassifier(
                max_iter=350, learning_rate=0.07, max_leaf_nodes=48,
                min_samples_leaf=40, l2_regularization=1.0,
                class_weight="balanced", early_stopping=True,
                validation_fraction=0.12, random_state=seed)),
        ]),
    }


def _calibrer(modele, X, y, methode: str):
    """Calibration sur le jeu de validation, modele de base fige."""
    try:
        from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(modele), method=methode)
    except Exception:  # pragma: no cover - anciennes versions
        cal = CalibratedClassifierCV(modele, method=methode, cv="prefit")
    cal.fit(X, y)
    return cal


# =============================================================================
#  Figures
# =============================================================================
def _figures(dossier: Path, y_te, s_te, tab_couts, seuil, y_va, s_va,
             importances: pd.DataFrame | None) -> list[str]:
    dossier.mkdir(parents=True, exist_ok=True)
    noms = []
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})

    p, r, _ = precision_recall_curve(y_te, s_te)
    fpr, tpr, _ = roc_curve(y_te, s_te)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(r, p, color="#d1495b")
    ax[0].axhline(np.mean(y_te), ls="--", c="#999",
                  label=f"hasard ({np.mean(y_te):.4f})")
    ax[0].set_xlabel("Rappel"); ax[0].set_ylabel("Precision"); ax[0].legend()
    ax[0].set_title(f"Courbe precision-rappel (PR-AUC={average_precision_score(y_te,s_te):.3f})")
    ax[1].plot(fpr, tpr, color="#3a7ca5"); ax[1].plot([0, 1], [0, 1], "--", c="#999")
    ax[1].set_xlabel("Taux de faux positifs"); ax[1].set_ylabel("Taux de vrais positifs")
    ax[1].set_title(f"Courbe ROC (AUC={roc_auc_score(y_te,s_te):.3f})")
    plt.tight_layout(); plt.savefig(dossier / "10_courbes.png", bbox_inches="tight"); plt.close()
    noms.append("10_courbes.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(tab_couts.seuil, tab_couts.cout_total, color="#2b3a55")
    ax.axvline(seuil, color="#d1495b", ls="--", label=f"seuil retenu = {seuil:.4f}")
    ax.set_xlabel("Seuil de decision"); ax.set_ylabel("Cout attendu ($)")
    ax.set_title("Le seuil est choisi en dollars, pas en F1")
    ax.legend()
    plt.tight_layout(); plt.savefig(dossier / "11_cout_seuil.png", bbox_inches="tight"); plt.close()
    noms.append("11_cout_seuil.png")

    frac_pos, moy_pred = calibration_curve(y_va, s_va, n_bins=12, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(moy_pred, frac_pos, "o-", color="#3a7ca5")
    ax.plot([0, max(moy_pred)], [0, max(moy_pred)], "--", c="#999")
    ax.set_xlabel("Probabilite predite"); ax.set_ylabel("Frequence observee")
    ax.set_title("Calibration du modele")
    plt.tight_layout(); plt.savefig(dossier / "12_calibration.png", bbox_inches="tight"); plt.close()
    noms.append("12_calibration.png")

    if importances is not None and len(importances):
        top = importances.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(top.variable, top.importance, color="#3a7ca5")
        ax.set_xlabel("Chute de PR-AUC quand la variable est permutee")
        ax.set_title("Importance des variables (permutation, jeu de test)")
        plt.tight_layout(); plt.savefig(dossier / "13_importances.png", bbox_inches="tight")
        plt.close()
        noms.append("13_importances.png")
    return noms


# =============================================================================
#  Entrainement
# =============================================================================
def executer(cfg: Config | None = None, etiquette: str = "v1",
             enregistrer_champion: bool = True,
             alias: str | None = None, pdf: pd.DataFrame | None = None) -> dict:
    cfg = cfg or charger_config()
    titre(f"ETAPE 05 - Entrainement du modele ({etiquette})")
    m = cfg["modelisation"]
    eco = m["economie"]
    dossier = cfg.chemin("artefacts") / "modele"
    dossier.mkdir(parents=True, exist_ok=True)

    jeux = donnees.decouper(cfg, pdf=pdf)
    num, cat = jeux.variables_numeriques, jeux.variables_categorielles
    cible = jeux.cible

    Xtr, ytr = jeux.entrainement[num + cat], jeux.entrainement[cible].values
    Xva, yva = jeux.validation[num + cat], jeux.validation[cible].values
    Xte, yte = jeux.test[num + cat], jeux.test[cible].values
    mva = jeux.validation["montant"].values
    mte = jeux.test["montant"].values
    jours_valid = max(jeux.validation["jour_index"].nunique(), 1)
    jours_test = max(jeux.test["jour_index"].nunique(), 1)

    # Sur un poste de travail, la memoire est la ressource limitante : on libere
    # la table complete des que les trois sous-ensembles sont extraits.
    del jeux
    gc.collect()

    registre.configurer(cfg)
    resultats = {}
    modeles = construire_modeles(num, cat, cfg.seed)

    with mlflow.start_run(run_name=f"entrainement_{etiquette}") as run_parent:
        mlflow.log_params({
            "jeu_variables": etiquette,
            "nb_variables": len(num) + len(cat),
            "nb_lignes_entrainement": len(Xtr),
            "taux_fraude_entrainement": round(float(ytr.mean()), 6),
            "decoupage": "temporel",
            "calibration": m["calibration"],
        })

        for nom, pipe in modeles.items():
            if nom not in m["modeles"]:
                continue
            with mlflow.start_run(run_name=nom, nested=True):
                t0 = time.time()
                with chrono(LOG, f"Entrainement : {nom}"):
                    pipe.fit(Xtr, ytr)
                s_va = pipe.predict_proba(Xva)[:, 1]
                pr = float(average_precision_score(yva, s_va))
                roc = float(roc_auc_score(yva, s_va))
                r50 = evaluation.rappel_a_precision(yva, s_va, 0.5)
                resultats[nom] = dict(pr_auc=pr, roc_auc=roc, rappel_a_precision_50=r50,
                                      duree_s=round(time.time() - t0, 1))
                mlflow.log_metrics({"pr_auc_validation": pr, "roc_auc_validation": roc,
                                    "rappel_a_precision_50": r50})
                LOG.info("   %-24s PR-AUC=%.4f  ROC-AUC=%.4f  rappel@prec50=%.3f",
                         nom, pr, roc, r50)

        meilleur = max(resultats, key=lambda k: resultats[k]["pr_auc"])
        LOG.info("Modele retenu : %s (PR-AUC validation = %.4f)",
                 meilleur, resultats[meilleur]["pr_auc"])
        mlflow.log_param("modele_retenu", meilleur)

        with chrono(LOG, f"Calibration {m['calibration']} du modele retenu"):
            modele = _calibrer(modeles[meilleur], Xva, yva, m["calibration"])

        s_va_cal = modele.predict_proba(Xva)[:, 1]
        with chrono(LOG, "Recherche du seuil de decision optimal (cout en dollars)"):
            seuil, tab_couts = evaluation.seuil_optimal(
                yva, s_va_cal, mva, eco,
                budget_alertes=m["budget_alertes_par_jour"], nb_jours=jours_valid)

        s_te = modele.predict_proba(Xte)[:, 1]
        mt = evaluation.evaluer(yte, s_te, mte, seuil, eco)
        mv = evaluation.evaluer(yva, s_va_cal, mva, seuil, eco)

        # Reference : sans modele, toute la fraude passe et on perd son montant.
        cout_sans_modele = float(mte[yte == 1].sum() * eco["cout_faux_negatif_ratio"])
        economie_realisee = cout_sans_modele - mt.cout_total

        LOG.info("--- Performance sur le jeu de TEST (jamais vu) ---")
        for k, v in mt.to_dict().items():
            LOG.info("   %-24s : %s", k, round(v, 4) if isinstance(v, float) else v)
        LOG.info("   %-24s : %.2f $", "economie_vs_sans_modele", economie_realisee)

        with chrono(LOG, "Importance des variables par permutation"):
            importances = _importances(modele, Xte, yte, cfg.seed)

        figures = _figures(dossier, yte, s_te, tab_couts, seuil, yva, s_va_cal, importances)
        for f in figures:
            mlflow.log_artifact(str(dossier / f), artifact_path="figures")

        mlflow.log_metrics({f"test_{k}": v for k, v in mt.to_dict().items()
                            if isinstance(v, (int, float))})
        mlflow.log_metric("economie_dollars_vs_sans_modele", economie_realisee)
        mlflow.log_param("seuil_decision", seuil)

        signature = mlflow.models.infer_signature(Xte.head(50), s_te[:50])
        info_modele = mlflow.sklearn.log_model(
            modele, name="modele", signature=signature, input_example=Xte.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        uri = info_modele.model_uri

        alias_final = alias or (registre.ALIAS_CHAMPION if enregistrer_champion else None)
        version = registre.enregistrer(
            uri, cfg, alias=alias_final,
            description=(f"{meilleur} calibre ({m['calibration']}), variables {etiquette}, "
                         f"seuil={seuil:.4f}, PR-AUC test={mt.pr_auc:.4f}"))

        resume = {
            "etiquette": etiquette,
            "run_id": run_parent.info.run_id,
            "version_registre": version,
            "alias": alias_final,
            "modele_retenu": meilleur,
            "comparaison_validation": resultats,
            "seuil_decision": seuil,
            "metriques_validation": mv.to_dict(),
            "metriques_test": mt.to_dict(),
            "economie_vs_sans_modele": round(economie_realisee, 2),
            "variables_numeriques": num,
            "variables_categorielles": cat,
            "nb_variables": len(num) + len(cat),
            "top_variables": (importances.head(15).to_dict("records")
                              if importances is not None else []),
            "figures": figures,
        }
        ecrire_json(dossier / f"resume_{etiquette}.json", resume)
        # `resume_courant.json` decrit le modele SERVI : il n'est mis a jour que
        # lorsque cette version devient effectivement le champion.
        if alias_final == registre.ALIAS_CHAMPION:
            ecrire_json(dossier / "resume_courant.json", resume)
        tab_couts.to_csv(dossier / f"courbe_cout_{etiquette}.csv", index=False)
        if importances is not None:
            importances.to_csv(dossier / f"importances_{etiquette}.csv", index=False)

    return resume


def _importances(modele, X, y, seed: int, n: int = 8000) -> pd.DataFrame | None:
    try:
        ech = X.sample(min(n, len(X)), random_state=seed)
        yech = pd.Series(y, index=X.index).loc[ech.index].values
        r = permutation_importance(modele, ech, yech, n_repeats=3,
                                   scoring="average_precision",
                                   random_state=seed, n_jobs=1)
        return (pd.DataFrame({"variable": X.columns, "importance": r.importances_mean})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    except Exception as exc:  # pragma: no cover
        LOG.warning("Importances non calculees : %s", exc)
        return None


if __name__ == "__main__":
    executer()
