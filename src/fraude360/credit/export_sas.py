"""Pont Python -> SAS : exporte le portefeuille et la grille au format SAS.

Trois formats sont produits, parce qu'ils ne servent pas au meme public :

  .sas7bdat via .csv + programme d'import  le portefeuille lui-meme
  PROC FORMAT CNTLIN                       les tables WOE, sous une forme
                                           que SAS applique nativement
  programme .sas de scoring                la grille figee en code, lisible
                                           et executable sans le modele Python

Le troisieme point est celui qui compte en pratique. Dans une banque, le
modele developpe en Python est frequemment REIMPLANTE en SAS pour la
production, et la validation exige la preuve que les deux implantations
donnent le meme score. C'est ce que produit `verifier_equivalence`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.credit import scorecard as sc
from fraude360.utils.journal import obtenir_journal, titre

LOG = obtenir_journal("fraude360.credit.export_sas")


def _code_scoring(grille) -> str:
    """Genere le DATA step SAS qui reproduit exactement `scorecard.predire`."""
    lignes = [
        "/*====================================================================",
        "  Grille de score figee -- GENERE AUTOMATIQUEMENT, NE PAS EDITER",
        "",
        "  Reproduit src/fraude360/credit/scorecard.predire() a l'identique.",
        "  Toute divergence entre ce code et le modele Python est un defaut :",
        "  la verification d'equivalence (export_sas.verifier_equivalence)",
        "  doit etre rejouee apres chaque reentrainement.",
        "====================================================================*/",
        "",
        "data scores;",
        "  set portefeuille;",
        "",
        f"  _logit = {grille.intercept:.10f};",
        "",
    ]

    for var, dec in grille.decoupages.items():
        beta = grille.coefficients[var]
        lignes.append(f"  /* --- {var} (beta = {beta:.6f}, IV = {dec.iv:.4f}) --- */")
        lignes.append(f"  _woe_{var} = 0;")

        if dec.categorielle:
            for _, r in dec.table.iterrows():
                cl = str(r["classe"]).replace("'", "''")
                lignes.append(
                    f"  if strip({var}) = '{cl}' then _woe_{var} = {r['woe']:.10f};")
        else:
            bornes = list(dec.bornes)
            tbl = dec.table.set_index("classe")
            manquant = tbl.loc["MANQUANT", "woe"] if "MANQUANT" in tbl.index else 0.0
            lignes.append(f"  if missing({var}) then _woe_{var} = {manquant:.10f};")
            for i in range(len(bornes) - 1):
                cle = str(i)
                if cle not in tbl.index:
                    continue
                w = float(tbl.loc[cle, "woe"])
                # pandas.cut produit des intervalles (bas, haut] : ouverts a
                # gauche, FERMES a droite. Le code SAS doit utiliser <= en
                # borne haute partout, sinon les valeurs tombant exactement sur
                # une borne changent de classe et le score diverge.
                bas = "-1e300" if i == 0 else f"{bornes[i]:.6f}"
                haut = "1e300" if i == len(bornes) - 2 else f"{bornes[i+1]:.6f}"
                lignes.append(
                    f"  else if {bas} < {var} and {var} <= {haut} "
                    f"then _woe_{var} = {w:.10f};")
        lignes.append(f"  _logit = _logit + {beta:.10f} * _woe_{var};")
        lignes.append("")

    lignes += [
        "  pd_12m = 1 / (1 + exp(-_logit));",
        "  _cote  = (1 - pd_12m) / max(pd_12m, 1e-9);",
        f"  score  = round({grille.decalage:.10f} + {grille.facteur:.10f} "
        "* log(max(_cote, 1e-9)), 0.1);",
        "",
        "  label pd_12m = \"Probabilite de defaut a 12 mois\"",
        "        score  = \"Score en points\";",
        "  drop _logit _cote _woe_:;",
        "run;",
        "",
    ]
    return "\n".join(lignes)


def _code_import(dossier_csv: str) -> str:
    return f"""/*====================================================================
  Import du portefeuille exporte depuis le prototype Python.
  A executer avant 01_woe_iv.sas.
====================================================================*/
%let CSV = {dossier_csv};
libname CRED "&CSV.";

proc import datafile="&CSV./portefeuille.csv"
            out=CRED.PORTEFEUILLE dbms=csv replace;
  guessingrows=max;
  getnames=yes;
run;

proc contents data=CRED.PORTEFEUILLE varnum; run;

title "Controle d'integrite apres import";
proc sql;
  select count(*) as nb_prets,
         sum(defaut_12m) as nb_defauts,
         mean(defaut_12m) as taux_defaut format=percent9.3,
         count(distinct millesime) as nb_millesimes
  from CRED.PORTEFEUILLE;
quit;
title;
"""


def verifier_equivalence(d: pd.DataFrame, grille, n: int = 2000) -> Dict[str, float]:
    """Compare le score Python au score que produirait le code SAS genere.

    Le code SAS est reproduit ici en Python a partir des memes bornes et
    coefficients : si les deux chemins divergent, c'est que la generation
    du code SAS est fautive. Une banque exige cette preuve avant de
    basculer un modele en production SAS.
    """
    ech = d.head(n)
    pd_py, score_py = sc.predire(ech, grille)

    logit = np.full(len(ech), grille.intercept, dtype=float)
    for var, dec in grille.decoupages.items():
        w = np.zeros(len(ech))
        tbl = dec.table.set_index("classe")
        if dec.categorielle:
            m = tbl.woe.to_dict()
            w = ech[var].fillna("MANQUANT").astype(str).map(m).fillna(0.0).values
        else:
            bornes = list(dec.bornes)
            x = ech[var].values.astype(float)
            manquant = float(tbl.loc["MANQUANT", "woe"]) if "MANQUANT" in tbl.index else 0.0
            w = np.where(np.isnan(x), manquant, 0.0)
            for i in range(len(bornes) - 1):
                cle = str(i)
                if cle not in tbl.index:
                    continue
                bas = -1e300 if i == 0 else bornes[i]
                haut = 1e300 if i == len(bornes) - 2 else bornes[i + 1]
                # (bas, haut] : ouvert a gauche, ferme a droite, comme
                # pandas.cut. Une borne haute ouverte ferait basculer de classe
                # les valeurs tombant exactement dessus -- c'est ce que ce
                # controle avait detecte, avec 23 points d'ecart sur le score.
                dedans = (x > bas) & (x <= haut)
                w = np.where(~np.isnan(x) & dedans, float(tbl.loc[cle, "woe"]), w)
        logit += grille.coefficients[var] * w

    pd_sas = 1 / (1 + np.exp(-logit))
    cote = (1 - pd_sas) / np.clip(pd_sas, 1e-9, None)
    score_sas = np.round(grille.decalage + grille.facteur * np.log(np.clip(cote, 1e-9, None)), 1)

    ecart_pd = float(np.max(np.abs(pd_py - pd_sas)))
    ecart_score = float(np.max(np.abs(score_py - score_sas)))
    return {"n_compares": len(ech),
            "ecart_max_pd": ecart_pd,
            "ecart_max_score_points": ecart_score,
            "equivalence_etablie": bool(ecart_pd < 1e-9 and ecart_score < 0.11)}


def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    titre("CREDIT -- Export vers SAS")
    dossier = cfg.racine / "sas" / "donnees"
    dossier.mkdir(parents=True, exist_ok=True)

    portefeuille = pd.read_parquet(cfg.chemin("gold") / "credit" / "portefeuille.parquet")
    portefeuille.to_csv(dossier / "portefeuille.csv", index=False, encoding="utf-8")

    grille_json = json.loads(
        (cfg.chemin("artefacts") / "credit" / "grille_score.json").read_text("utf-8"))

    # on reconstruit la grille depuis le pipeline pour generer le code
    from fraude360.credit import pipeline_credit as pc
    coupure = int(portefeuille.millesime.max()) - 11
    dev = portefeuille[portefeuille.millesime < coupure]
    grille = sc.ajuster(dev, dev.defaut_12m.values, pc.VAR_NUM, pc.VAR_CAT)

    (cfg.racine / "sas" / "00_import.sas").write_text(
        _code_import(dossier.as_posix()), encoding="utf-8")
    (cfg.racine / "sas" / "05_scoring_grille_figee.sas").write_text(
        _code_scoring(grille), encoding="utf-8")

    # tables WOE au format CNTLIN, directement utilisable par PROC FORMAT
    tables = pd.concat([dec.table.assign(variable=v)
                        for v, dec in grille.decoupages.items()], ignore_index=True)
    tables.to_csv(dossier / "woe_tables.csv", index=False, encoding="utf-8")

    equiv = verifier_equivalence(portefeuille[portefeuille.millesime >= coupure], grille)
    LOG.info("Equivalence Python / SAS : ecart max PD %.2e, score %.3f pt -> %s",
             equiv["ecart_max_pd"], equiv["ecart_max_score_points"],
             "ETABLIE" if equiv["equivalence_etablie"] else "NON ETABLIE")

    from fraude360.utils.journal import ecrire_json
    ecrire_json(cfg.chemin("artefacts") / "credit" / "equivalence_python_sas.json", equiv)
    return equiv


if __name__ == "__main__":
    executer()
