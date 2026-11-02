"""Orchestrateur du prototype Fraude-360.

Usage :
    python -m fraude360 tout              # enchaine les 13 etapes
    python -m fraude360 <etape>           # execute une seule etape
    python -m fraude360 --liste           # affiche les etapes disponibles

Sur Databricks, chaque etape correspond a une tache du job decrit dans
infra/databricks/databricks.yml : le meme code, orchestre par Workflows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Tuple

from fraude360.config import charger_config
from fraude360.utils.journal import ecrire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.pipeline")


def _etapes() -> List[Tuple[str, str, Callable]]:
    from fraude360.abtest import experience
    from fraude360.bi import export
    from fraude360.eda import exploration
    from fraude360.features import gold
    from fraude360.genai import evaluation as genai_evaluation
    from fraude360.gouvernance import carte_modele, equite
    from fraude360.generation import simulateur
    from fraude360.ingestion import bronze
    from fraude360.modeling import entrainement
    from fraude360.monitoring import surveillance
    from fraude360.newfraud import nouvelle_typologie
    from fraude360.quality import silver
    from fraude360.retraining import reentrainement
    from fraude360.serving import lot

    return [
        ("generer", "00 - Creation des donnees transactionnelles",
         lambda cfg: simulateur.executer(cfg)),
        ("ingerer", "01 - Ingestion incrementale vers Bronze",
         lambda cfg: bronze.executer(cfg, complet=True)),
        ("nettoyer", "02 - Nettoyage et controles qualite vers Silver",
         lambda cfg: silver.executer(cfg)),
        ("variables", "03 - Ingenierie des variables vers Gold",
         lambda cfg: gold.executer(cfg)),
        ("eda", "04 - Analyse exploratoire",
         lambda cfg: exploration.executer(cfg)),
        ("entrainer", "05 - Entrainement, calibration et enregistrement",
         lambda cfg: entrainement.executer(cfg, etiquette="v1")),
        ("abtest", "07 - Test A/B : regles expertes contre modele",
         lambda cfg: experience.executer(cfg)),
        ("scorer", "08 - Scoring par lots (deploiement)",
         lambda cfg: lot.executer(cfg)),
        ("surveiller", "09 - Surveillance des donnees et de la performance",
         lambda cfg: surveillance.executer(cfg)),
        ("nouvelle-fraude", "10 - Prise en charge d'une nouvelle typologie",
         lambda cfg: nouvelle_typologie.executer(cfg)),
        ("reentrainer", "11 - Reentrainement et promotion conditionnelle",
         lambda cfg: reentrainement.executer(cfg, forcer=True)),
        ("rescorer", "08b - Rescoring de la production avec le champion courant",
         lambda cfg: lot.executer(cfg)),
        ("triage-genai", "13 - Triage des alertes par modele de langage (RAG + juge)",
         lambda cfg: genai_evaluation.executer(cfg)),
        ("equite", "14 - Analyse d'impact disparate",
         lambda cfg: equite.executer(cfg)),
        ("carte-modele", "15 - Fiche de modele (gouvernance)",
         lambda cfg: carte_modele.executer(cfg)),
        ("exporter-bi", "12 - Exports pour Power BI",
         lambda cfg: export.executer(cfg)),
    ]


def executer_tout(cfg=None, depuis: str | None = None) -> dict:
    cfg = cfg or charger_config()
    etapes = _etapes()
    if depuis:
        noms = [e[0] for e in etapes]
        if depuis not in noms:
            raise SystemExit(f"Etape inconnue : {depuis}")
        etapes = etapes[noms.index(depuis):]

    debut = time.time()
    journal = []
    titre("FRAUDE-360 - EXECUTION COMPLETE DU PIPELINE")
    for i, (nom, libelle, fonction) in enumerate(etapes, 1):
        t0 = time.time()
        LOG.info("")
        LOG.info(">>> [%d/%d] %s", i, len(etapes), libelle)
        try:
            fonction(cfg)
            statut = "SUCCES"
            erreur = None
        except Exception as exc:  # pragma: no cover
            statut = "ECHEC"
            erreur = str(exc)
            LOG.exception("Echec de l'etape %s", nom)
        duree = round(time.time() - t0, 1)
        journal.append({"etape": nom, "libelle": libelle, "statut": statut,
                        "duree_s": duree, "erreur": erreur})
        if statut == "ECHEC":
            break

    total = round(time.time() - debut, 1)
    resume = {"horodatage": datetime.now().isoformat(), "duree_totale_s": total,
              "etapes": journal,
              "statut_global": "SUCCES" if all(e["statut"] == "SUCCES" for e in journal)
                               else "ECHEC"}
    ecrire_json(cfg.chemin("artefacts") / "execution_pipeline.json", resume)

    titre(f"BILAN DE L'EXECUTION ({total} s)")
    for e in journal:
        LOG.info("   %-8s %-52s %6.1f s", e["statut"], e["libelle"], e["duree_s"])
    return resume


def principal(argv: List[str] | None = None) -> int:
    etapes = {nom: (lib, f) for nom, lib, f in _etapes()}
    parseur = argparse.ArgumentParser(
        prog="fraude360", description="Prototype de detection de la fraude transactionnelle")
    parseur.add_argument("etape", nargs="?", default="tout",
                         help="nom de l'etape, ou 'tout'")
    parseur.add_argument("--depuis", help="reprendre l'execution complete a partir de cette etape")
    parseur.add_argument("--liste", action="store_true", help="lister les etapes")
    parseur.add_argument("--config", help="chemin d'un fichier de configuration alternatif")
    args = parseur.parse_args(argv)

    if args.liste:
        print("Etapes disponibles :")
        for nom, (lib, _) in etapes.items():
            print(f"  {nom:<18} {lib}")
        print(f"  {'tout':<18} enchaine toutes les etapes")
        return 0

    cfg = charger_config(args.config, forcer=bool(args.config))

    if args.etape == "tout":
        resume = executer_tout(cfg, depuis=args.depuis)
        return 0 if resume["statut_global"] == "SUCCES" else 1

    if args.etape not in etapes:
        print(f"Etape inconnue : {args.etape}. Utiliser --liste.", file=sys.stderr)
        return 2
    etapes[args.etape][1](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
