# Fraude-360

**Prototype de bout en bout de détection de la fraude transactionnelle**
pour une coopérative financière — de la création des données au tableau de bord
Power BI, en passant par la mise en production, la surveillance, la prise en
charge d'une nouvelle typologie de fraude et le réentraînement.

Conçu pour être exécutable en une commande sur un poste Windows, et transposable
tel quel sur **Azure Databricks** (le même code, orchestré par Workflows).

---

## Ce que le prototype démontre

| # | Étape | Module | Compétence démontrée |
|---|-------|--------|----------------------|
| 00 | Création des données | `generation/simulateur.py` | Modélisation d'un flux transactionnel réaliste, 5 typologies de fraude |
| 01 | Ingestion / chargement | `ingestion/bronze.py` | Ingestion incrémentale, schéma contractuel, idempotence, Auto Loader |
| 02 | Nettoyage et qualité | `quality/silver.py` | 7 contrôles, quarantaine tracée, dédoublonnage, seuil de rejet bloquant |
| 03 | Ingénierie des variables | `features/gold.py` | 44 variables PySpark en fenêtres glissantes, sans fuite temporelle |
| 04 | Analyse exploratoire | `eda/exploration.py` | Compréhension des phénomènes frauduleux actifs, 8 figures + rapport |
| 05 | Modélisation et entraînement | `modeling/entrainement.py` | Découpage temporel, 3 modèles, calibration, **seuil choisi en dollars** |
| 06 | Enregistrement | `modeling/registre.py` | MLflow : suivi, registre, alias champion / challenger |
| 07 | Test A/B | `abtest/experience.py` | Randomisation par carte, contrôle A/A, test z, bootstrap par grappes |
| 08 | Déploiement | `serving/api.py`, `serving/lot.py` | API temps réel < 100 ms + scoring par lots, moteur de règles hybride |
| 09 | Surveillance | `monitoring/surveillance.py` | PSI, Kolmogorov-Smirnov, dérive de performance, alertes |
| 10 | Nouvelle typologie de fraude | `newfraud/nouvelle_typologie.py` | Diagnostic d'angle mort, mitigation par règle, nouvelles variables |
| 11 | Réentraînement | `retraining/reentrainement.py` | Déclenchement conditionnel, promotion sous 3 conditions |
| 12 | Tableau de bord | `bi/export.py`, `scripts/generer_pbip.py` | Modèle en étoile, 31 mesures DAX, projet Power BI `.pbip` |
| 13 | Triage par modèle de langage | `genai/triage.py`, `genai/evaluation.py` | RAG sur les politiques internes, garde-fous anti-invention, LLM-as-judge, mesure contre vérité terrain |

---

## 🎓 Cours universitaire de 45 h construit sur ce projet

Le dossier [`cours/`](cours/) transforme Fraude-360 en cours de 3 crédits :
**Science des données appliquée : détection de la fraude — du pipeline à la production**
(Noé Carème Fouotsa Manfouo, Ph. D.). 14 laboratoires Google Colab, un par séance, avec rétroaction automatique,
et trois parcours de formation (gestion, informatique, cégep/formation continue).
Voir [`cours/README.md`](cours/README.md) et la page du cours : <https://noe2019.github.io/cours-fraude360.html>.

[![Ouvrir la séance 1 dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noe2019/fraude-360-cours/blob/main/cours/laboratoires/01_contexte_simulation.ipynb)

---

## Démarrage rapide (poste local)

### Prérequis

- **Python 3.11** — <https://www.python.org/downloads/>
- **Java 17 ou 21** (requis par PySpark) — `winget install Microsoft.OpenJDK.17`
- **Power BI Desktop** (gratuit) pour la dernière étape

### Installation

```powershell
cd fraude-360-cours
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Exécution complète

```powershell
$env:PYTHONPATH = "src"
python -m fraude360 tout
```

Durée : **15 à 30 minutes** selon la machine. Le bilan des 13 étapes s'affiche à
la fin et est écrit dans `artifacts/execution_pipeline.json`.

Pour une démonstration en 3 minutes, utiliser le jeu réduit :

```powershell
python -m fraude360 tout --config conf/config_rapide.yaml
```

### Étapes individuelles

```powershell
python -m fraude360 --liste          # liste les étapes
python -m fraude360 entrainer        # n'exécute qu'une étape
python -m fraude360 tout --depuis surveiller
```

### Service de scoring temps réel

```powershell
python scripts/demo_api.py           # démarre l'API et envoie 20 transactions
python -m fraude360.serving.api      # démarre l'API seule -> http://127.0.0.1:8077/docs
```

### Tableau de bord Power BI

```powershell
python scripts/generer_pbip.py
```

### Triage des alertes par modèle de langage (étape 13)

```powershell
python -m fraude360 triage-genai
```

Fonctionne **hors ligne** : sans Ollama, un client de repli déterministe prend le
relais et l'évaluation s'applique de la même façon. Pour le vrai modèle :

```powershell
ollama pull llama3.1:8b     # rien ne sort du poste
```

puis ouvrir `powerbi/Fraude360.pbip` avec Power BI Desktop et cliquer
**Actualiser**.

---

## Ce qui se passe pendant l'exécution

```
data/00_landing/    650 000 transactions en 150 fichiers journaliers (systèmes sources)
      ↓ ingestion incrémentale, schéma déclaré, registre des fichiers traités
data/01_bronze/     copie fidèle + métadonnées techniques (fichier, horodatage, lot)
      ↓ normalisation, dédoublonnage, 7 contrôles de qualité
data/02_silver/     données conformes            data/99_quarantaine/  lignes rejetées + motif
      ↓ 44 variables en fenêtres glissantes PySpark
data/03_gold/       table de variables + décisions + tables de surveillance
      ↓ découpage temporel, 3 modèles, calibration, seuil économique
mlruns/             expériences MLflow, registre, alias champion / challenger
      ↓
artifacts/          rapports JSON, 15 figures, courbes, rapports d'incident
powerbi/data/       9 tables en étoile prêtes pour Power BI
```

---

## Choix techniques et pourquoi

**Le seuil de décision est choisi en dollars, pas en F1.** Une fraude ratée coûte
le montant de la transaction ; un faux positif coûte une revue manuelle et de la
friction pour le membre. Le seuil retenu minimise le coût attendu, sous
contrainte du budget quotidien de l'équipe d'enquête. Un modèle qu'on ne peut pas
traiter opérationnellement n'a aucune valeur.

**Le découpage est temporel, jamais aléatoire.** Un mélange aléatoire ferait fuir
de l'information du futur vers le passé : une carte compromise apparaîtrait des
deux côtés et les performances seraient irréalistes.

**Les probabilités sont calibrées.** Sans calibration, un score de 0,8 ne veut pas
dire 80 % de risque, et l'optimisation économique du seuil perd son sens.

**Le test A/B randomise par carte, pas par transaction.** Sinon les variables de
vélocité d'une même carte fuient d'un bras à l'autre et l'expérience est biaisée.
Le bootstrap est fait par grappes pour la même raison.

**Le modèle et les règles cohabitent.** Une règle experte se déploie en quelques
heures quand une nouvelle fraude apparaît ; un modèle demande un cycle de
réentraînement complet. L'étape 10 montre les deux temps de réponse.

**Le modèle de langage ne décide jamais.** L'étape 13 produit une note de triage
pour l'enquêteur ; la décision de blocage reste celle du modèle calibré et des
règles. Un LLM n'est ni calibré ni stable, et on ne peut pas lui attribuer un coût
d'erreur — il n'a donc pas sa place dans une boucle de décision sur de l'argent.
Il est évalué à trois niveaux : garde-fous déterministes (JSON valide, action
autorisée, politique citée réellement récupérée), LLM-as-judge, et surtout **taux
de fraude réel par action recommandée**, seule mesure indépendante du format et
du jugement d'un second modèle.

**La promotion d'un modèle est conditionnelle.** Trois conditions doivent être
réunies : test A/B favorable, gain de PR-AUC suffisant, aucune régression du
rappel. Sinon le champion reste en place et l'écart est documenté.

---

## Transposition sur Azure Databricks

| Local | Azure Databricks |
|---|---|
| `data/00_landing/` | Volume Unity Catalog, alimenté par ADLS Gen2 |
| Lecture de répertoire | **Auto Loader** (`cloudFiles`), point de reprise |
| Parquet | **Delta Lake** + Unity Catalog (`fraude360.bronze/silver/gold`) |
| Spark `local[*]` | Cluster Databricks à mise à l'échelle automatique |
| MLflow sur SQLite | MLflow du workspace + registre Unity Catalog |
| API FastAPI | **Databricks Model Serving** (alias `champion`) |
| `pipeline.py` | **Databricks Workflows** (`infra/databricks/databricks.yml`) |
| CSV → Power BI | Tables Delta `gold` en **Direct Lake** |

Une seule bascule dans `conf/config.yaml` :

```yaml
environnement:
  cible: databricks
  format_table: delta
```

Ressources Azure : `infra/azure/01_creer_ressources.ps1` (et
`99_supprimer_ressources.ps1` pour tout effacer après la démonstration).

---

## Structure du dépôt

```
conf/config.yaml              configuration unique du projet, tout est paramétrable
src/fraude360/                bibliothèque Python (13 modules, un par étape)
notebooks/                    7 notebooks Databricks (Auto Loader, Delta, MLflow, Serving)
infra/azure/                  scripts Azure CLI de création et de suppression
infra/databricks/             Asset Bundle : job multi-tâches, planification, serving
powerbi/                      projet Power BI .pbip (modèle sémantique + rapport)
scripts/                      démonstration de l'API, générateur du projet Power BI
tests/                        tests de non-régression (fuite temporelle, A/B, coûts, triage)
docs/                         guide d'implémentation pas à pas (.docx)
```

---

## Tests

```powershell
pytest -q
```

Les tests verrouillent les erreurs invisibles : fuite temporelle dans les
variables, instabilité de l'affectation A/B, erreur de calcul du coût métier.

---

## Licence

Code : licence MIT (voir `LICENSE`). Contenu pédagogique du dossier `cours/` : CC BY 4.0.
Toutes les données sont **synthétiques** : aucune donnée réelle de membre ou de transaction.
