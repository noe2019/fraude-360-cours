"""ETAPE 09 - Surveillance continue des schemas de fraude et du modele.

Un modele de fraude se degrade toujours : les fraudeurs s'adaptent. La question
n'est pas de savoir SI la performance va baisser, mais de le detecter avant que
le cout ne devienne visible dans les etats financiers.

Trois familles de signaux sont suivies quotidiennement :

  1. DERIVE DES DONNEES (disponible immediatement)
     Indice de stabilite de population (PSI) entre la periode de reference
     (entrainement) et chaque journee de production.
        PSI < 0,10  stable   |  0,10-0,25  a surveiller  |  > 0,25  alerte

  2. DERIVE OPERATIONNELLE (disponible immediatement)
     Taux d'alerte, taux de blocage, score moyen. Une hausse brutale sature
     l'equipe d'enquete meme si le modele est bon.

  3. DERIVE DE PERFORMANCE (disponible avec le delai de labellisation)
     Rappel, precision et dollars non evites, calcules uniquement sur les
     journees dont les etiquettes sont matures. C'est le signal le plus fiable,
     mais aussi le plus tardif : d'ou l'interet des deux premiers.

Sortie : table gold/surveillance, alertes JSON, figures, et une recommandation
explicite de reentrainement.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fraude360.config import Config, charger_config
from fraude360.modeling import donnees
from fraude360.utils.journal import chrono, ecrire_json, lire_json, obtenir_journal, titre
from fraude360.utils.spark import ecrire_table, lire_table, obtenir_spark

LOG = obtenir_journal("fraude360.surveillance")


# =============================================================================
#  Indicateurs de derive
# =============================================================================
def psi(reference: np.ndarray, courant: np.ndarray, n_classes: int = 10) -> float:
    """Indice de stabilite de population entre deux distributions."""
    reference = np.asarray(reference, dtype=float)
    courant = np.asarray(courant, dtype=float)
    reference = reference[np.isfinite(reference)]
    courant = courant[np.isfinite(courant)]
    if len(reference) < 50 or len(courant) < 20:
        return 0.0
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, n_classes + 1)))
    if len(bornes) < 3:
        return 0.0
    bornes[0], bornes[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference, bins=bornes)[0] / len(reference)
    p_cur = np.histogram(courant, bins=bornes)[0] / len(courant)
    p_ref = np.clip(p_ref, 1e-6, None)
    p_cur = np.clip(p_cur, 1e-6, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def ks(reference: np.ndarray, courant: np.ndarray) -> tuple[float, float]:
    reference = np.asarray(reference, dtype=float)
    courant = np.asarray(courant, dtype=float)
    reference = reference[np.isfinite(reference)]
    courant = courant[np.isfinite(courant)]
    if len(reference) < 50 or len(courant) < 20:
        return 0.0, 1.0
    r = stats.ks_2samp(reference, courant)
    return float(r.statistic), float(r.pvalue)


def niveau(valeur_psi: float, cfg: Config) -> str:
    s = cfg["surveillance"]
    if valeur_psi >= s["psi_alerte"]:
        return "ALERTE"
    if valeur_psi >= s["psi_avertissement"]:
        return "AVERTISSEMENT"
    return "STABLE"


# =============================================================================
#  Execution
# =============================================================================
def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("ETAPE 09 - Surveillance des donnees, des alertes et de la performance")
    s = cfg["surveillance"]
    dossier = cfg.chemin("artefacts") / "surveillance"
    dossier.mkdir(parents=True, exist_ok=True)

    spark = obtenir_spark(cfg)
    pdf = donnees.charger_gold(cfg)
    decisions = lire_table(spark, cfg.chemin_table("gold", "decisions"), cfg).toPandas()
    decisions["horodatage"] = pd.to_datetime(decisions["horodatage"])
    decisions["date"] = decisions["horodatage"].dt.date

    d = cfg["decoupage"]
    fin_reference = d["jours_entrainement"]
    reference = pdf[pdf.jour_index < fin_reference]
    production = pdf[pdf.jour_index >= fin_reference + d["jours_validation"]]

    # --- 1. derive des donnees ----------------------------------------------
    variables = [v for v in s["variables_suivies"] if v in pdf.columns]
    lignes_derive: List[dict] = []
    with chrono(LOG, "Calcul du PSI et du test de Kolmogorov-Smirnov par journee"):
        for jour, bloc in production.groupby(production.horodatage.dt.date):
            for v in variables:
                valeur = psi(reference[v].values, bloc[v].values)
                stat_ks, p_ks = ks(reference[v].values, bloc[v].values)
                lignes_derive.append(dict(date=jour, variable=v, psi=round(valeur, 4),
                                          ks=round(stat_ks, 4), valeur_p_ks=round(p_ks, 6),
                                          niveau=niveau(valeur, cfg)))
    derive = pd.DataFrame(lignes_derive)

    # --- 2. derive operationnelle -------------------------------------------
    ops = (decisions.groupby("date")
           .agg(transactions=("transaction_id", "count"),
                alertes=("alerte", "sum"),
                blocages=("bloquee", "sum"),
                score_moyen=("score_modele", "mean"),
                montant_total=("montant", "sum"))
           .reset_index())
    ops["taux_alerte_pct"] = 100 * ops.alertes / ops.transactions
    ops["taux_blocage_pct"] = 100 * ops.blocages / ops.transactions

    base_alerte = ops.taux_alerte_pct.head(5).mean()
    ops["ecart_taux_alerte"] = ops.taux_alerte_pct / max(base_alerte, 1e-9) - 1

    # --- 3. derive de performance (etiquettes matures uniquement) -----------
    delai = int(cfg["generation"]["delai_labellisation_jours"])
    date_max = decisions["horodatage"].max()
    limite_maturite = (date_max - timedelta(days=delai)).date()
    mures = decisions[decisions["date"] <= limite_maturite]

    perf = []
    for jour, bloc in mures.groupby("date"):
        n_fraudes = int(bloc.est_fraude.sum())
        if n_fraudes == 0:
            continue
        vp = int(bloc.loc[bloc.est_fraude == 1, "alerte"].sum())
        alertes = int(bloc.alerte.sum())
        perf.append(dict(
            date=jour, transactions=len(bloc), fraudes=n_fraudes,
            fraudes_detectees=vp,
            rappel=round(vp / n_fraudes, 4),
            precision=round(vp / max(alertes, 1), 4),
            montant_fraude=round(float(bloc.loc[bloc.est_fraude == 1, "montant"].sum()), 2),
            montant_non_detecte=round(float(bloc.loc[(bloc.est_fraude == 1)
                                                     & (bloc.alerte == 0), "montant"].sum()), 2)))
    performance = pd.DataFrame(perf)

    resume_modele = lire_json(cfg.chemin("artefacts") / "modele" / "resume_courant.json")
    rappel_reference = resume_modele["metriques_test"]["rappel"]

    # --- alertes -------------------------------------------------------------
    alertes: List[dict] = []
    if len(derive):
        pires = (derive.groupby("variable")["psi"].max()
                 .sort_values(ascending=False))
        for v, valeur in pires.items():
            if valeur >= s["psi_alerte"]:
                alertes.append(dict(type="derive_donnees", gravite="ALERTE", variable=v,
                                    valeur=round(float(valeur), 4),
                                    message=f"PSI de {v} a {valeur:.3f} (seuil "
                                            f"{s['psi_alerte']}) : la distribution a change."))
            elif valeur >= s["psi_avertissement"]:
                alertes.append(dict(type="derive_donnees", gravite="AVERTISSEMENT", variable=v,
                                    valeur=round(float(valeur), 4),
                                    message=f"PSI de {v} a {valeur:.3f} : a surveiller."))

    if len(performance):
        rappel_recent = float(performance.tail(7).rappel.mean())
        chute = rappel_reference - rappel_recent
        if chute >= s["chute_rappel_alerte"]:
            alertes.append(dict(
                type="derive_performance", gravite="ALERTE", variable="rappel",
                valeur=round(chute, 4),
                message=f"Rappel tombe a {rappel_recent:.1%} contre {rappel_reference:.1%} "
                        f"a la mise en production (-{100*chute:.1f} points)."))
    else:
        rappel_recent = None

    hausse = float(ops.ecart_taux_alerte.tail(5).max()) if len(ops) else 0.0
    if hausse >= s["hausse_taux_alerte"]:
        alertes.append(dict(type="derive_operationnelle", gravite="AVERTISSEMENT",
                            variable="taux_alerte", valeur=round(hausse, 4),
                            message=f"Volume d'alertes en hausse de {100*hausse:.0f} % : "
                                    f"verifier la capacite de l'equipe d'enquete."))

    # --- typologies non couvertes -------------------------------------------
    non_detectees = (mures[(mures.est_fraude == 1) & (mures.alerte == 0)]
                     .groupby("typologie_fraude")
                     .agg(cas=("transaction_id", "count"),
                          montant=("montant", "sum")).reset_index()
                     .sort_values("montant", ascending=False))
    # "contestation_non_fondee" est du bruit d'etiquetage, pas un mode operatoire.
    connues = set(cfg["generation"]["typologies"]) | {"contestation_non_fondee", "", "aucune"}
    for _, r in non_detectees.iterrows():
        if r.typologie_fraude not in connues and r.cas >= 5:
            alertes.append(dict(
                type="nouvelle_typologie", gravite="CRITIQUE", variable=r.typologie_fraude,
                valeur=int(r.cas),
                message=f"Typologie inconnue du modele : '{r.typologie_fraude}' - "
                        f"{int(r.cas)} cas non detectes, {r.montant:,.0f} $ de pertes."))

    declencher = any(a["gravite"] in ("ALERTE", "CRITIQUE") for a in alertes)

    rapport = {
        "horodatage": datetime.now().isoformat(),
        "periode_reference": f"J0 a J{fin_reference-1}",
        "periode_surveillee": f"{production.horodatage.min()} -> {production.horodatage.max()}",
        "psi_max_par_variable": (derive.groupby("variable")["psi"].max()
                                 .round(4).sort_values(ascending=False).to_dict()
                                 if len(derive) else {}),
        "rappel_reference": round(rappel_reference, 4),
        "rappel_recent": round(rappel_recent, 4) if rappel_recent is not None else None,
        "taux_alerte_moyen_pct": round(float(ops.taux_alerte_pct.mean()), 3) if len(ops) else None,
        "montant_non_detecte": round(float(performance.montant_non_detecte.sum()), 2)
                               if len(performance) else 0.0,
        "nb_alertes": len(alertes),
        "alertes": alertes,
        "reentrainement_recommande": declencher,
        "typologies_non_detectees": non_detectees.to_dict("records"),
    }

    # --- persistance ---------------------------------------------------------
    if len(derive):
        ecrire_table(spark.createDataFrame(derive.astype({"date": "str"})),
                     cfg.chemin_table("gold", "surveillance_derive"), cfg)
        derive.to_csv(dossier / "derive_donnees.csv", index=False)
    if len(ops):
        ops.to_csv(dossier / "suivi_operationnel.csv", index=False)
        ecrire_table(spark.createDataFrame(ops.astype({"date": "str"})),
                     cfg.chemin_table("gold", "surveillance_operations"), cfg)
    if len(performance):
        performance.to_csv(dossier / "suivi_performance.csv", index=False)
        ecrire_table(spark.createDataFrame(performance.astype({"date": "str"})),
                     cfg.chemin_table("gold", "surveillance_performance"), cfg)
    ecrire_json(dossier / "rapport_surveillance.json", rapport)
    _figures(dossier, derive, ops, performance, rappel_reference, cfg)

    LOG.info("PSI maximal par variable :")
    for v, val in list(rapport["psi_max_par_variable"].items())[:8]:
        LOG.info("   %-32s %.4f  %s", v, val, niveau(val, cfg))
    if rappel_recent is not None:
        LOG.info("Rappel : reference %.1f %% -> recent %.1f %%",
                 100 * rappel_reference, 100 * rappel_recent)
    LOG.info("%d alerte(s) levee(s).", len(alertes))
    for a in alertes:
        LOG.info("   [%s] %s", a["gravite"], a["message"])
    LOG.info("Reentrainement recommande : %s", "OUI" if declencher else "non")
    return rapport


def _figures(dossier: Path, derive, ops, performance, rappel_reference, cfg):
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})
    if len(derive):
        pivot = derive.pivot_table(index="date", columns="variable", values="psi")
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in pivot.columns:
            ax.plot(pivot.index, pivot[c], label=c, lw=1.3)
        ax.axhline(cfg["surveillance"]["psi_avertissement"], ls="--", c="#e0a458",
                   label="avertissement")
        ax.axhline(cfg["surveillance"]["psi_alerte"], ls="--", c="#d1495b", label="alerte")
        ax.set_ylabel("PSI"); ax.set_title("Derive des variables dans le temps")
        ax.legend(fontsize=7, ncol=3)
        plt.tight_layout(); plt.savefig(dossier / "20_derive_psi.png", bbox_inches="tight")
        plt.close()

    if len(ops):
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.plot(ops.date, ops.taux_alerte_pct, color="#3a7ca5", label="taux d'alerte")
        ax.plot(ops.date, ops.taux_blocage_pct, color="#d1495b", label="taux de blocage")
        ax.set_ylabel("%"); ax.set_title("Charge operationnelle quotidienne"); ax.legend()
        plt.tight_layout(); plt.savefig(dossier / "21_charge_alertes.png", bbox_inches="tight")
        plt.close()

    if len(performance):
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax[0].plot(performance.date, 100 * performance.rappel, color="#d1495b", marker="o", ms=3)
        ax[0].axhline(100 * rappel_reference, ls="--", c="#999",
                      label=f"reference {100*rappel_reference:.0f} %")
        ax[0].set_ylabel("Rappel (%)"); ax[0].legend()
        ax[0].set_title("Performance en production (etiquettes matures)")
        ax[1].bar(performance.date, performance.montant_non_detecte, color="#8b1e3f")
        ax[1].set_ylabel("Fraude non detectee ($)")
        plt.tight_layout(); plt.savefig(dossier / "22_performance.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    executer()
