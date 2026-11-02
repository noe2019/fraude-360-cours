# Cours — Science des données appliquée : détection de la fraude

*Applied Data Science for Fraud Detection: from Pipeline to Production* · 3 crédits (45 h) · Noé Carème Fouotsa Manfouo, Ph. D.

Ce dossier transforme le prototype **Fraude-360** en cours universitaire. Chaque séance de 3 h (1 h 30 de cours + 1 h 30 de
laboratoire) ouvre un module du projet, l'exécute sur un jeu réduit, puis fait **réécrire sa fonction clé** et la compare à
l'implémentation de référence grâce à des cellules de vérification ✅.

## Démarrer

- **Google Colab (recommandé)** : cliquez sur le lien d'un laboratoire ci-dessous. La première cellule clone le dépôt, installe
  PySpark et MLflow et vérifie Java. Aucune installation sur votre poste.
- **Poste local** : `pip install -r requirements.txt httpx`, Java 17, puis ouvrez `cours/laboratoires/` dans Jupyter.

Chaque laboratoire appelle `preparer("<étape>")` (voir [`outils_cours.py`](outils_cours.py)) : les étapes du pipeline nécessaires
sont exécutées une seule fois avec la configuration [`conf/config_cours.yaml`](../conf/config_cours.yaml) (≈ 60 000 transactions,
environ 4 minutes pour tout le pipeline). Vous pouvez donc ouvrir n'importe quel laboratoire directement.
`from outils_cours import reinitialiser; reinitialiser()` remet le cours à zéro.

## Calendrier et parcours de formation

● séance complète · ○ démonstration en classe (laboratoire facultatif) · — hors parcours

| S. | Thème | Laboratoire (Colab) | Gestion | Informatique | Cégep / FC |
|---|---|---|:-:|:-:|:-:|
| 1 | La fraude transactionnelle et ses coûts ; données simulées | [`01_contexte_simulation`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/01_contexte_simulation.ipynb) | ● | ● | ● |
| 2 | Ingestion incrémentale (Bronze), PySpark | [`02_ingestion_bronze`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/02_ingestion_bronze.ipynb) | ○ | ● | — |
| 3 | Qualité des données et quarantaine (Silver) | [`03_qualite_silver`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/03_qualite_silver.ipynb) | ● | ● | ● |
| 4 | Variables comportementales sans fuite temporelle (Gold) | [`04_variables_gold`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/04_variables_gold.ipynb) | ○ | ● | ○ |
| 5 | Analyse exploratoire orientée décision | [`05_exploration`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/05_exploration.ipynb) | ● | ● | ● |
| 6 | Découpage temporel, déséquilibre, calibration | [`06_modelisation_calibration`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/06_modelisation_calibration.ipynb) | ● | ● | ● |
| 7 | Seuil choisi en dollars, capacité d'enquête | [`07_seuil_economique`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/07_seuil_economique.ipynb) | ● | ● | ● |
| 8 | Registre MLflow, champion/challenger · examen | [`08_mlflow_registre`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/08_mlflow_registre.ipynb) | ● | ● | — |
| 9 | Test A/B : randomisation par carte, bootstrap | [`09_test_ab`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/09_test_ab.ipynb) | ● | ● | ○ |
| 10 | Déploiement : API de scoring en temps réel | [`10_deploiement_api`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/10_deploiement_api.ipynb) | ○ | ● | ● |
| 11 | Surveillance : PSI, KS, dérive de performance | [`11_surveillance_derive`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/11_surveillance_derive.ipynb) | ● | ● | ○ |
| 12 | Nouvelle typologie et réentraînement conditionnel | [`12_nouvelle_fraude_reentrainement`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/12_nouvelle_fraude_reentrainement.ipynb) | ○ | ● | — |
| 13 | Gouvernance, équité et risque de crédit (IFRS 9) | [`13_gouvernance_credit`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/13_gouvernance_credit.ipynb) | ● | ○ | — |
| 14 | Power BI et triage des alertes par IA générative | [`14_bi_ia_generative`](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/14_bi_ia_generative.ipynb) | ● | ● | ● |
| 15 | Présentations des projets devant un « comité de risque » | — | ● | ● | ● |

| Parcours | Programmes visés | Durée |
|---|---|---|
| **Gestion** | UdeS École de gestion (SIMQG) : intelligence d'affaires et analytique ; Bishop's Williams School of Business | 45 h |
| **Informatique** | UdeS Département d'informatique : maîtrise (IA et science des données), baccalauréat | 45 h |
| **Cégep / FC** | Cégep de Sherbrooke (formation continue, AEC), Techniques de l'informatique ; UdeS formation continue | 30 h, extensible à 45 h |

Dans chaque notebook, les exercices sont étiquetés : 🟢 tous les parcours · 🟣 gestion + informatique · 🔷 informatique ·
🔶 gestion. Les exercices 🔷/🔶 sont facultatifs pour les autres parcours.

## Évaluation

Carnets de laboratoire (4 remises) 20 % · quiz (3) 15 % · examen pratique (S8) 20 % · projet d'équipe 45 % (proposition,
rapport d'étape, présentation à un « comité de risque », rapport final + dépôt GitHub + fiche de modèle).
Le plan de cours détaillé est disponible auprès du chargé de cours.

## Licence

Contenu pédagogique : CC BY 4.0 · code : MIT. Données entièrement synthétiques.
