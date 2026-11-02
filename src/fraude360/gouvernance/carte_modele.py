"""Fiche de modele (model card), generee depuis les artefacts, jamais saisie.

Une fiche recopiee a la main diverge du modele des la version suivante. Celle-ci
est reconstruite a chaque entrainement depuis `resume_courant.json`, le registre
MLflow et les rapports de surveillance et d'equite : elle ne peut pas mentir sur
les metriques, seulement etre incomplete si une etape n'a pas tourne.

Structure alignee sur ce qu'attend une ligne de defense 2 (BSIF E-23) :
identification et lignee, usage prevu et usages proscrits, donnees, methode,
performance, limites connues, surveillance, et conditions de retrait.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fraude360.config import Config, charger_config
from fraude360.utils.journal import lire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.gouvernance.carte")

USAGE_PREVU = """Classer les transactions de paiement d'une cooperative financiere selon
leur risque de fraude, afin d'alimenter une file d'alertes traitee par des
enqueteurs humains, et de bloquer automatiquement les transactions au-dessus
d'un seuil calibre economiquement."""

USAGES_PROSCRITS = [
    "Toute decision d'octroi de credit ou de tarification.",
    "Toute decision affectant durablement la relation avec le membre "
    "(fermeture de compte, signalement externe) sans revue humaine.",
    "Le score ne doit pas etre reutilise comme variable d'entree d'un autre "
    "modele sans nouvelle validation : sa distribution depend du seuil courant.",
    "Aucun usage sur une population ou un canal absents des donnees "
    "d'entrainement sans revalidation prealable.",
]

LIMITES = [
    "Les etiquettes de fraude arrivent avec un delai de confirmation : la "
    "performance recente est toujours mesuree sur une population partielle.",
    "Le modele apprend des fraudes DETECTEES. Une typologie jamais detectee "
    "reste invisible pour lui -- c'est le role des regles expertes et de "
    "l'etape de prise en charge d'une nouvelle typologie.",
    "Les variables de velocite supposent un historique de la carte. Une carte "
    "neuve est systematiquement moins bien couverte.",
    "Le seuil est optimise sous une contrainte de capacite d'enquete. Si "
    "l'effectif de l'equipe change, le seuil doit etre recalcule.",
]


def _charger(chemin: Path) -> Dict[str, Any] | None:
    try:
        return lire_json(chemin)
    except Exception:
        return None


def construire(cfg: Config | None = None) -> str:
    cfg = cfg or charger_config()
    art = cfg.chemin("artefacts")
    resume = _charger(art / "modele" / "resume_courant.json")
    if resume is None:
        raise FileNotFoundError(
            "resume_courant.json absent : executer `python -m fraude360 entrainer`.")
    surv = _charger(art / "surveillance" / "rapport_surveillance.json") or {}
    equite = _charger(art / "gouvernance" / "rapport_equite.json") or {}
    qualite = _charger(art / "qualite" / "rapport_qualite.json") or {}
    ab = _charger(art / "abtest" / "rapport_regles_vs_modele.json") or {}

    mv, mt = resume.get("metriques_validation", {}), resume.get("metriques_test", {})
    comp = resume.get("comparaison_validation", {})

    def ligne(cle, unite=""):
        v = mt.get(cle)
        return f"{v:.4f}{unite}" if isinstance(v, (int, float)) else "n.d."

    top = resume.get("top_variables", [])[:10]
    tv = "\n".join(f"| {i+1} | `{v.get('variable','?')}` | {v.get('importance',0):.5f} |"
                   for i, v in enumerate(top)) or "| — | _non disponible_ | — |"

    proscrits = "\n".join(f"- {u}" for u in USAGES_PROSCRITS)
    limites = "\n".join(f"- {l}" for l in LIMITES)
    candidats = "\n".join(
        f"| {n} | {m.get('pr_auc',0):.4f} | {m.get('roc_auc',0):.4f} | "
        f"{m.get('rappel_a_precision_50',0):.4f} |" for n, m in comp.items())

    eq = equite.get("sous_groupes_a_justifier", {})
    eq_txt = ("Aucun ecart de faux positifs a justifier."
              if equite and not eq else
              (f"Ecarts a justifier : {json.dumps(eq, ensure_ascii=False)}" if eq
               else "_Analyse d'equite non executee._"))

    return f"""# Fiche de modele — {resume.get('modele_retenu', 'n.d.')} (version {resume.get('version_registre','n.d.')})

> Document genere automatiquement le {datetime.now():%Y-%m-%d %H:%M} depuis les
> artefacts d'execution. Ne pas editer a la main : toute correction doit passer
> par le code qui produit les artefacts.

## 1. Identification et lignee

| | |
|---|---|
| Modele | `{resume.get('modele_retenu','n.d.')}`, calibre |
| Version au registre | {resume.get('version_registre','n.d.')} |
| Alias | `{resume.get('alias','n.d.')}` |
| Execution MLflow | `{resume.get('run_id','n.d.')}` |
| Jeu de variables | `{resume.get('etiquette','n.d.')}` — {resume.get('nb_variables','n.d.')} variables |
| Seuil de decision | {resume.get('seuil_decision', 0):.4f} |

## 2. Usage prevu

{USAGE_PREVU}

## 3. Usages proscrits

{proscrits}

## 4. Donnees

| | |
|---|---|
| Controles qualite | {qualite.get('nb_controles', 'n.d.')} |
| Lignes mises en quarantaine | {qualite.get('nb_quarantaine', 'n.d.')} |
| Decoupage | temporel strict — aucun melange aleatoire |
| Fuite temporelle | variables calculees en fenetres fermees a t-1 seconde |

## 5. Methode et modeles candidats

| Candidat | PR-AUC (validation) | ROC-AUC | Rappel a precision 50 % |
|---|---|---|---|
{candidats or "| _non disponible_ | — | — | — |"}

Le seuil n'est pas choisi sur le F1 mais en minimisant le cout attendu en
dollars, sous contrainte du budget quotidien de l'equipe d'enquete.

## 6. Performance sur l'echantillon de test

| Indicateur | Valeur |
|---|---|
| PR-AUC | {ligne('pr_auc')} |
| ROC-AUC | {ligne('roc_auc')} |
| Score de Brier (calibration) | {ligne('brier')} |
| Precision | {ligne('precision')} |
| Rappel | {ligne('rappel')} |
| Rappel en dollars | {mt.get('rappel_en_dollars', mv.get('rappel_en_dollars', 'n.d.'))} |
| Taux d'alerte | {mt.get('taux_alerte_pct','n.d.')} % |

Comparaison aux regles expertes seules : {ab.get('conclusion', '_test A/B non execute_')}

### Variables les plus influentes (importance par permutation)

| Rang | Variable | Importance |
|---|---|---|
{tv}

## 7. Equite

{eq_txt}

## 8. Limites connues

{limites}

## 9. Surveillance

| | |
|---|---|
| Derive des donnees | PSI et Kolmogorov-Smirnov, par journee |
| Derive operationnelle | taux d'alerte, taux de blocage, score moyen |
| Derive de performance | sur etiquettes matures uniquement |
| Recommandation courante | {surv.get('recommandation', '_surveillance non executee_')} |

## 10. Conditions de retrait ou de reentrainement

Le modele est retire ou reentraine si l'une de ces conditions est reunie :

- PSI superieur a 0,25 sur une variable structurante, deux journees consecutives.
- Chute du rappel superieure a 10 points sur etiquettes matures.
- Apparition d'une typologie de fraude non couverte (voir etape 10).
- Changement de la capacite d'enquete invalidant le seuil economique.

La promotion d'un remplacant exige trois conditions simultanees : test A/B
favorable, gain de PR-AUC suffisant, et aucune regression du rappel.
"""


def executer(cfg: Config | None = None) -> Path:
    cfg = cfg or charger_config()
    titre("GOUVERNANCE - Generation de la fiche de modele")
    texte = construire(cfg)
    dossier = cfg.chemin("artefacts") / "gouvernance"
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / "carte_modele.md"
    chemin.write_text(texte, encoding="utf-8")
    LOG.info("Fiche de modele ecrite : %s (%d caracteres)", chemin, len(texte))
    return chemin
