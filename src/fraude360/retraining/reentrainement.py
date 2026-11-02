"""ETAPE 11 - Reentrainement et promotion conditionnelle.

Le reentrainement n'est pas un rituel de calendrier : il est DECLENCHE par un
signal de surveillance (derive, chute de rappel, nouvelle typologie) et il doit
FAIRE SES PREUVES avant d'atteindre la production.

Enchainement :
  1. verification du declencheur (rapport de surveillance de l'etape 09) ;
  2. fenetre d'entrainement glissante : on reapprend sur l'historique recent,
     jeu de variables v2, en conservant un jeu de test posterieur jamais vu ;
  3. le candidat est enregistre avec l'alias `challenger`, jamais `champion` ;
  4. test A/B champion contre challenger sur la periode de test, les deux bras
     conservant le moteur de regles deploye ;
  5. promotion uniquement si TROIS conditions sont reunies :
        - le test A/B conclut en faveur du challenger ;
        - le gain de PR-AUC depasse le seuil minimal configure ;
        - le rappel ne regresse pas au-dela du garde-fou.
     Sinon le champion reste en place et l'ecart est documente.
"""
from __future__ import annotations

import gc
import shutil
from datetime import datetime

import numpy as np

from fraude360.abtest import experience
from fraude360.config import Config, charger_config
from fraude360.modeling import donnees, entrainement, registre
from fraude360.serving.regles import MoteurDeRegles, regle_nouvelle_typologie
from fraude360.utils.journal import chrono, ecrire_json, lire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.reentrainement")


def declencheur(cfg: Config) -> dict:
    chemin = cfg.chemin("artefacts") / "surveillance" / "rapport_surveillance.json"
    if not chemin.exists():
        return {"declenche": False, "motif": "aucun rapport de surveillance disponible"}
    r = lire_json(chemin)
    motifs = [a["message"] for a in r["alertes"] if a["gravite"] in ("ALERTE", "CRITIQUE")]
    return {"declenche": bool(r.get("reentrainement_recommande")),
            "motif": motifs[0] if motifs else "aucune alerte bloquante",
            "nb_alertes": len(r.get("alertes", []))}


def _fixer_version_variables(cfg: Config, version: int) -> None:
    """Rend la bascule v1 -> v2 durable dans conf/config.yaml.

    Sans cela, une execution ulterieure de l'etape 03 reconstruirait la couche
    GOLD avec l'ancien jeu de variables et le champion v2 deviendrait inservable.
    """
    import re
    fichier = cfg.fichier or (cfg.racine / "conf" / "config.yaml")
    texte = fichier.read_text(encoding="utf-8")
    nouveau = re.sub(r"(variables:\s*\n\s*version:\s*)\d+", rf"\g<1>{version}", texte)
    if nouveau != texte:
        fichier.write_text(nouveau, encoding="utf-8")
        LOG.info("%s : variables.version fixe a %d.", fichier.name, version)


def executer(cfg: Config | None = None, forcer: bool = False) -> dict:
    cfg = cfg or charger_config()
    titre("ETAPE 11 - Reentrainement du modele et promotion conditionnelle")
    rt = cfg["reentrainement"]

    d = declencheur(cfg)
    LOG.info("Declencheur : %s (%s)", "OUI" if d["declenche"] else "non", d["motif"])
    if not d["declenche"] and not forcer:
        LOG.info("Aucun reentrainement necessaire.")
        return {"statut": "non_declenche", **d}

    # --- 1. fenetre glissante ------------------------------------------------
    pdf = donnees.charger_gold(cfg)
    jour_max = int(pdf.jour_index.max())
    jours_valid, jours_test = 16, 20
    decoupage_initial = dict(cfg["decoupage"])
    cfg.brut["decoupage"] = {
        "jours_entrainement": jour_max + 1 - jours_valid - jours_test,
        "jours_validation": jours_valid,
        "jours_test": jours_test,
    }
    LOG.info("Fenetre glissante : entrainement J0-J%d, validation %d j, test %d j "
             "(historique total : %d jours).",
             cfg.brut["decoupage"]["jours_entrainement"] - 1, jours_valid, jours_test,
             jour_max + 1)

    # --- 2. entrainement du candidat ----------------------------------------
    resume_v1 = lire_json(cfg.chemin("artefacts") / "modele" / "resume_v1.json")
    version_champion = registre.version_par_alias(registre.ALIAS_CHAMPION, cfg)

    with chrono(LOG, "Entrainement du challenger sur le jeu de variables v2"):
        resume_v2 = entrainement.executer(cfg, etiquette="v2",
                                          enregistrer_champion=False,
                                          alias=registre.ALIAS_CHALLENGER,
                                          pdf=pdf)

    # --- 3. comparaison sur UN MEME jeu de test ------------------------------
    # Le PR-AUC du champion doit etre recalcule sur la fenetre de test courante :
    # comparer deux modeles evalues sur deux periodes differentes n'a aucun sens.
    jeux = donnees.decouper(cfg, pdf=pdf)
    df_test = jeux.test.copy()
    del jeux
    gc.collect()

    modele_champion = registre.charger_par_alias(registre.ALIAS_CHAMPION, cfg)
    modele_challenger = registre.charger_par_alias(registre.ALIAS_CHALLENGER, cfg)

    var_v1 = resume_v1["variables_numeriques"] + resume_v1["variables_categorielles"]
    var_v2 = resume_v2["variables_numeriques"] + resume_v2["variables_categorielles"]
    seuil_v1 = resume_v1["seuil_decision"]
    seuil_v2 = resume_v2["seuil_decision"]

    from sklearn.metrics import average_precision_score
    y_test = df_test[cfg["modelisation"]["cible"]].values
    pr_auc_champion = float(average_precision_score(
        y_test, modele_champion.predict_proba(df_test[var_v1])[:, 1]))
    pr_auc_challenger = float(average_precision_score(
        y_test, modele_challenger.predict_proba(df_test[var_v2])[:, 1]))
    gain_pr_auc = pr_auc_challenger - pr_auc_champion
    LOG.info("PR-AUC sur la meme fenetre de test : champion v1 = %.4f -> "
             "challenger v2 = %.4f (%+0.4f)",
             pr_auc_champion, pr_auc_challenger, gain_pr_auc)

    # Les deux bras conservent la regle de mitigation deja deployee : on isole
    # ainsi l'apport propre du nouveau modele.
    moteur = MoteurDeRegles()
    moteur.ajouter(regle_nouvelle_typologie())

    def _decision(modele, variables, seuil):
        def _f(bloc):
            s = modele.predict_proba(bloc[variables])[:, 1]
            r = moteur.evaluer(bloc)["score_regles"].values
            return ((s >= seuil) | (r >= 0.5)).astype(int)
        return _f

    champion = experience.Strategie(
        nom=f"champion v1 ({resume_v1['modele_retenu']}) + regles",
        decider=_decision(modele_champion, var_v1, seuil_v1))
    challenger = experience.Strategie(
        nom=f"challenger v2 ({resume_v2['modele_retenu']}) + regles",
        decider=_decision(modele_challenger, var_v2, seuil_v2))

    rapport_ab = experience.comparer(df_test, champion, challenger, cfg,
                                     "champion_v1_vs_challenger_v2")

    # --- 4. decision de promotion -------------------------------------------
    rappel_v1 = rapport_ab["bras"]["A_champion"]["taux_detection"]
    rappel_v2 = rapport_ab["bras"]["B_challenger"]["taux_detection"]
    conditions = {
        "test_ab_favorable": rapport_ab["decision"] == "DEPLOYER_B",
        "gain_pr_auc_suffisant": bool(gain_pr_auc >= rt["gain_minimal_pr_auc"]),
        "pas_de_regression_rappel": bool(rappel_v2 >= rappel_v1 - rt["garde_fou_baisse_rappel"]),
    }
    promouvoir = all(conditions.values())

    LOG.info("--- Grille de decision ---")
    for k, v in conditions.items():
        LOG.info("   %-28s : %s", k, "OK" if v else "NON")

    version_challenger = resume_v2["version_registre"]
    if promouvoir:
        registre.promouvoir(version_challenger, registre.ALIAS_CHAMPION, cfg)
        dossier = cfg.chemin("artefacts") / "modele"
        shutil.copy(dossier / "resume_v2.json", dossier / "resume_courant.json")
        cfg.brut["variables"]["version"] = 2
        _fixer_version_variables(cfg, 2)
        LOG.info("PROMOTION : la version %d devient le champion servi en production.",
                 version_challenger)
        LOG.info("Le service de scoring doit etre recharge : POST /recharger")
    else:
        LOG.info("PAS DE PROMOTION : le champion v%s reste en place. "
                 "Le challenger est conserve au registre pour analyse.",
                 version_champion)

    cfg.brut["decoupage"] = decoupage_initial

    rapport = {
        "horodatage": datetime.now().isoformat(),
        "declencheur": d,
        "version_champion_avant": version_champion,
        "version_challenger": version_challenger,
        "pr_auc_champion_meme_test": round(pr_auc_champion, 5),
        "pr_auc_challenger_meme_test": round(pr_auc_challenger, 5),
        "gain_pr_auc": round(gain_pr_auc, 5),
        "taux_detection_champion": rappel_v1,
        "taux_detection_challenger": rappel_v2,
        "cout_par_1000tx_champion": rapport_ab["bras"]["A_champion"]["cout_par_1000_tx"],
        "cout_par_1000tx_challenger": rapport_ab["bras"]["B_challenger"]["cout_par_1000_tx"],
        "conditions_de_promotion": conditions,
        "promotion": promouvoir,
        "version_champion_apres": version_challenger if promouvoir else version_champion,
        "rapport_ab": rapport_ab,
        "nouvelles_variables_utilisees": [v for v in var_v2 if v not in var_v1],
    }
    ecrire_json(cfg.chemin("artefacts") / "modele" / "rapport_reentrainement.json", rapport)
    return rapport


if __name__ == "__main__":
    executer(forcer=True)
