"""ETAPE 04 - Analyse exploratoire (EDA).

Objectif : comprendre les caracteristiques des phenomenes frauduleux actifs
avant de modeliser. Produit huit figures et un rapport HTML autonome.

Questions traitees :
  - quel est le poids reel de la fraude, en volume et en dollars ?
  - comment se repartissent les typologies ?
  - quelles variables separent le mieux fraude et non-fraude ?
  - la fraude a-t-elle une signature horaire, geographique, sectorielle ?
  - quelles variables sont redondantes (correlation) ?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.utils.journal import chrono, ecrire_json, obtenir_journal, titre
from fraude360.utils.spark import lire_table, obtenir_spark

LOG = obtenir_journal("fraude360.eda")

COULEUR_LEG = "#3a7ca5"
COULEUR_FRA = "#d1495b"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def charger_gold_pandas(cfg: Config) -> pd.DataFrame:
    spark = obtenir_spark(cfg)
    df = lire_table(spark, cfg.chemin_table("gold", "caracteristiques"), cfg)
    return df.toPandas()


def _fig(dossier: Path, nom: str) -> str:
    chemin = dossier / nom
    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    return nom


def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("ETAPE 04 - Analyse exploratoire des donnees")
    dossier = cfg.chemin("artefacts") / "eda"
    dossier.mkdir(parents=True, exist_ok=True)

    with chrono(LOG, "Chargement de la table GOLD"):
        df = charger_gold_pandas(cfg)
    df["date"] = pd.to_datetime(df["horodatage"]).dt.date
    fraude = df[df.est_fraude == 1]
    legit = df[df.est_fraude == 0]

    figures = []

    # 1. Volume et fraude dans le temps -------------------------------------
    quotidien = df.groupby("date").agg(
        transactions=("transaction_id", "count"),
        fraudes=("est_fraude", "sum"),
        montant_fraude=("montant", lambda s: 0)).reset_index()
    quotidien["montant_fraude"] = (df[df.est_fraude == 1].groupby("date")["montant"]
                                   .sum().reindex(quotidien["date"]).fillna(0).values)
    quotidien["taux_fraude_pct"] = 100 * quotidien.fraudes / quotidien.transactions

    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].plot(quotidien.date, quotidien.transactions, color=COULEUR_LEG)
    ax[0].set_ylabel("Transactions / jour")
    ax[0].set_title("Volume transactionnel et intensite de la fraude")
    ax[1].plot(quotidien.date, quotidien.taux_fraude_pct, color=COULEUR_FRA)
    ax[1].set_ylabel("Taux de fraude (%)")
    ax[1].set_xlabel("Date")
    figures.append(_fig(dossier, "01_serie_temporelle.png"))

    # 2. Repartition des typologies -----------------------------------------
    rep = fraude["typologie_fraude"].value_counts()
    mnt = fraude.groupby("typologie_fraude")["montant"].sum().reindex(rep.index)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].barh(rep.index, rep.values, color=COULEUR_FRA)
    ax[0].set_title("Nombre de transactions frauduleuses")
    ax[1].barh(mnt.index, mnt.values, color="#8b1e3f")
    ax[1].set_title("Pertes ($) par typologie")
    figures.append(_fig(dossier, "02_typologies.png"))

    # 3. Distribution des montants ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.logspace(0, 4, 60)
    ax.hist(legit.montant.clip(1, 10000), bins=bins, alpha=0.65,
            label="Legitime", color=COULEUR_LEG, density=True)
    ax.hist(fraude.montant.clip(1, 10000), bins=bins, alpha=0.65,
            label="Fraude", color=COULEUR_FRA, density=True)
    ax.set_xscale("log"); ax.set_xlabel("Montant ($, echelle log)"); ax.legend()
    ax.set_title("Distribution des montants : la fraude occupe les deux extremes")
    figures.append(_fig(dossier, "03_montants.png"))

    # 4. Signature horaire ---------------------------------------------------
    par_heure = df.groupby("heure")["est_fraude"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(par_heure.index, 100 * par_heure["mean"], color=COULEUR_FRA)
    ax.set_xlabel("Heure de la journee"); ax.set_ylabel("Taux de fraude (%)")
    ax.set_title("La fraude est sur-representee la nuit")
    figures.append(_fig(dossier, "04_heures.png"))

    # 5. Canal et categorie de marchand -------------------------------------
    canal = df.groupby("canal")["est_fraude"].mean().sort_values() * 100
    cat = (df.groupby("categorie_marchand")["est_fraude"].mean()
           .sort_values(ascending=False).head(10) * 100)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].barh(canal.index, canal.values, color=COULEUR_LEG)
    ax[0].set_title("Taux de fraude par canal (%)")
    ax[1].barh(cat.index[::-1], cat.values[::-1], color=COULEUR_FRA)
    ax[1].set_title("10 categories de marchand les plus touchees (%)")
    figures.append(_fig(dossier, "05_canal_categorie.png"))

    # 6. Pouvoir discriminant des variables (difference standardisee) --------
    num = [c for c in df.columns
           if df[c].dtype.kind in "if" and c not in ("est_fraude",)]
    lignes = []
    for c in num:
        a, b = fraude[c].astype(float), legit[c].astype(float)
        s = np.sqrt((a.var() + b.var()) / 2) + 1e-9
        lignes.append((c, (a.mean() - b.mean()) / s))
    disc = pd.DataFrame(lignes, columns=["variable", "d_cohen"]).set_index("variable")
    disc["abs"] = disc.d_cohen.abs()
    top = disc.sort_values("abs", ascending=False).head(18).sort_values("d_cohen")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    couleurs = [COULEUR_FRA if v > 0 else COULEUR_LEG for v in top.d_cohen]
    ax.barh(top.index, top.d_cohen, color=couleurs)
    ax.set_xlabel("Difference standardisee (d de Cohen) fraude - legitime")
    ax.set_title("Variables les plus discriminantes")
    figures.append(_fig(dossier, "06_pouvoir_discriminant.png"))

    # 7. Correlations entre les variables les plus fortes -------------------
    sel = list(top.index[-12:])
    corr = df[sel].astype(float).corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=7)
    ax.grid(False); fig.colorbar(im, shrink=0.8)
    ax.set_title("Correlations (detection de la redondance)")
    figures.append(_fig(dossier, "07_correlations.png"))

    # 8. Velocite : signature du test de carte -------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(np.log10(legit.secondes_depuis_tx_precedente.clip(1, 1e6)), bins=50,
               density=True, alpha=0.65, color=COULEUR_LEG, label="Legitime")
    ax[0].hist(np.log10(fraude.secondes_depuis_tx_precedente.clip(1, 1e6)), bins=50,
               density=True, alpha=0.65, color=COULEUR_FRA, label="Fraude")
    ax[0].set_xlabel("log10(secondes depuis la transaction precedente)"); ax[0].legend()
    ax[1].hist(legit.nb_tx_carte_1h.clip(0, 25), bins=25, density=True, alpha=0.65,
               color=COULEUR_LEG, label="Legitime")
    ax[1].hist(fraude.nb_tx_carte_1h.clip(0, 25), bins=25, density=True, alpha=0.65,
               color=COULEUR_FRA, label="Fraude")
    ax[1].set_xlabel("Nombre de transactions de la carte dans l'heure"); ax[1].legend()
    figures.append(_fig(dossier, "08_velocite.png"))

    # --- synthese chiffree --------------------------------------------------
    synthese = {
        "periode": f"{df.horodatage.min()} -> {df.horodatage.max()}",
        "nb_transactions": int(len(df)),
        "nb_fraudes": int(fraude.shape[0]),
        "taux_fraude_pct": round(100 * len(fraude) / len(df), 4),
        "montant_total": round(float(df.montant.sum()), 2),
        "pertes_fraude": round(float(fraude.montant.sum()), 2),
        "part_pertes_pct": round(100 * fraude.montant.sum() / df.montant.sum(), 3),
        "montant_median_legitime": round(float(legit.montant.median()), 2),
        "montant_median_fraude": round(float(fraude.montant.median()), 2),
        "nb_cartes_touchees": int(fraude.carte_id.nunique()),
        "typologies": rep.to_dict(),
        "top_variables_discriminantes": disc.sort_values("abs", ascending=False)
                                            .head(12)["d_cohen"].round(3).to_dict(),
        "desequilibre_classes": f"1 fraude pour {int(len(legit)/max(len(fraude),1))} legitimes",
    }
    ecrire_json(dossier / "synthese_eda.json", synthese)

    _rapport_html(dossier, synthese, figures, cfg)
    LOG.info("EDA terminee : %d figures + rapport HTML dans %s", len(figures), dossier)
    for k in ("nb_transactions", "nb_fraudes", "taux_fraude_pct", "pertes_fraude",
              "desequilibre_classes"):
        LOG.info("   %-26s : %s", k, synthese[k])
    return synthese


def _rapport_html(dossier: Path, s: dict, figures: list[str], cfg: Config) -> None:
    cartes = [
        ("Transactions analysees", f"{s['nb_transactions']:,}".replace(",", " ")),
        ("Transactions frauduleuses", f"{s['nb_fraudes']:,}".replace(",", " ")),
        ("Taux de fraude", f"{s['taux_fraude_pct']:.3f} %"),
        ("Pertes attribuables a la fraude", f"{s['pertes_fraude']:,.0f} $".replace(",", " ")),
        ("Part des pertes dans le volume", f"{s['part_pertes_pct']:.3f} %"),
        ("Desequilibre des classes", s["desequilibre_classes"]),
    ]
    html = [f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Fraude-360 - Analyse exploratoire</title><style>
body{{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1c2430}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 24px 60px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:34px 0 10px;
border-left:4px solid #d1495b;padding-left:10px}}
.sub{{color:#5b6775;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #e3e7ec;border-radius:10px;padding:14px 16px}}
.card .lab{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#6b7684}}
.card .val{{font-size:22px;font-weight:600;margin-top:4px}}
img{{width:100%;background:#fff;border:1px solid #e3e7ec;border-radius:10px;margin-top:10px}}
table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden}}
td,th{{padding:7px 10px;border-bottom:1px solid #edf0f3;font-size:13px;text-align:left}}
</style></head><body><div class="wrap">
<h1>Fraude-360 &mdash; analyse exploratoire</h1>
<div class="sub">Periode : {s['periode']}</div><div class="grid">"""]
    for lab, val in cartes:
        html.append(f'<div class="card"><div class="lab">{lab}</div>'
                    f'<div class="val">{val}</div></div>')
    html.append("</div>")
    titres = ["Volume et intensite de la fraude", "Typologies de fraude",
              "Distribution des montants", "Signature horaire",
              "Canaux et secteurs marchands", "Pouvoir discriminant des variables",
              "Redondance entre variables", "Velocite transactionnelle"]
    for t, f in zip(titres, figures):
        html.append(f"<h2>{t}</h2><img src='{f}' alt='{t}'>")
    html.append("<h2>Variables les plus discriminantes</h2><table>"
                "<tr><th>Variable</th><th>d de Cohen</th></tr>")
    for k, v in s["top_variables_discriminantes"].items():
        html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    html.append("</table></div></body></html>")
    (dossier / "rapport_eda.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    executer()
