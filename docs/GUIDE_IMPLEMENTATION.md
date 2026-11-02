# Guide d'implémentation — Fraude-360

Guide technique et pratique. Il suppose que vous partez d'un poste vierge et
que vous voulez, dans l'ordre : **faire tourner**, **comprendre**, **modifier**,
puis **déployer sur Azure Databricks**.

---

## Partie 1 — Faire tourner

### 1.1 Prérequis

| Composant | Version | Vérification |
|---|---|---|
| Python | 3.11 recommandé (3.10–3.12) | `python --version` |
| Java | 17 ou 21, requis par PySpark | `java -version` |
| Power BI Desktop | gratuit, dernière étape seulement | — |
| Ollama | optionnel, étape 13 | `ollama --version` |

```powershell
winget install Microsoft.OpenJDK.17
```

Si `java` répond mais que PySpark échoue, `JAVA_HOME` n'est pas défini :

```powershell
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.13.11-hotspot"
```

### 1.2 Installation

```powershell
cd fraude-360
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1.3 Première exécution

Commencez par le jeu réduit — trois minutes, et il valide toute la chaîne :

```powershell
$env:PYTHONPATH = "src"
python -m fraude360 tout --config conf/config_rapide.yaml
```

Puis l'exécution complète, 15 à 30 minutes :

```powershell
python -m fraude360 tout
```

Le bilan des étapes s'affiche à la fin et s'écrit dans
`artifacts/execution_pipeline.json`.

### 1.4 Volet crédit

Indépendant du volet fraude, il a son propre orchestrateur :

```powershell
python -m fraude360.credit.pipeline_credit
python -m fraude360.credit.export_sas
```

### 1.5 Avec Docker, si vous préférez ne rien installer

```bash
docker build -t fraude360 .
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" fraude360
```

---

## Partie 2 — Comprendre l'architecture

### 2.1 Le flux des données

```
data/00_landing/    650 000 transactions, 150 fichiers journaliers
      ↓             ingestion incrémentale, schéma déclaré, registre des fichiers
data/01_bronze/     copie fidèle + métadonnées techniques
      ↓             normalisation, dédoublonnage, 7 contrôles
data/02_silver/     conforme        data/99_quarantaine/  rejets + motif
      ↓             44 variables en fenêtres glissantes PySpark
data/03_gold/       variables + décisions + tables de surveillance
      ↓
mlruns/             MLflow : expériences, registre, alias
artifacts/          rapports JSON, figures, fiche de modèle
powerbi/data/       tables en étoile
```

### 2.2 Les quatre décisions structurantes

Ce sont celles à savoir défendre. Le reste est de l'exécution.

**Le seuil est choisi en dollars, pas en F1.** Une fraude ratée coûte le
montant de la transaction ; un faux positif coûte une revue manuelle et de la
friction. Le seuil retenu minimise le coût attendu **sous contrainte du budget
quotidien de l'équipe d'enquête**. Un modèle qu'on ne peut pas traiter
opérationnellement n'a aucune valeur.

**Le découpage est temporel, jamais aléatoire.** Un mélange ferait fuir de
l'information du futur vers le passé : une carte compromise apparaîtrait des
deux côtés.

**Les probabilités sont calibrées.** Sans calibration, un score de 0,8 ne veut
pas dire 80 % de risque et l'optimisation économique perd son sens.

**L'A/B randomise par carte, pas par transaction.** Sinon les variables de
vélocité d'une même carte fuient d'un bras à l'autre. Le bootstrap est fait par
grappes pour la même raison.

### 2.3 Carte des modules

| Module | Rôle | À lire en premier si… |
|---|---|---|
| `config.py` | source unique de configuration | vous changez un paramètre |
| `generation/simulateur.py` | flux transactionnel, 5 typologies | vous voulez d'autres fraudes |
| `ingestion/bronze.py` | ingestion idempotente | vous branchez une vraie source |
| `quality/silver.py` | 7 contrôles, quarantaine | vous ajoutez une règle qualité |
| `features/gold.py` | 44 variables, fenêtres fermées à t-1 | **toute modification de variable** |
| `modeling/entrainement.py` | 3 modèles, calibration, seuil | vous changez d'algorithme |
| `abtest/experience.py` | randomisation par grappes | vous concevez une expérience |
| `serving/api.py` | API temps réel | vous exposez le modèle |
| `monitoring/surveillance.py` | PSI, KS, performance | vous ajoutez un signal |
| `genai/` | triage LLM + RAG + évaluation | vous touchez au volet IA générative |
| `gouvernance/` | fiche de modèle, équité | vous préparez une revue |
| `credit/` | grille de score, IFRS 9, validation | volet crédit |

---

## Partie 3 — Modifier

### 3.1 Ajouter une variable

Trois endroits, dans cet ordre. En sauter un casse silencieusement le service.

1. **`features/gold.py`** — la calculer. **Toujours dans une fenêtre fermée
   avant la transaction courante** :

```python
w = Window.partitionBy("carte_id").orderBy("ts_unix").rangeBetween(-J7, -1)
d = d.withColumn("ma_variable", F.count("*").over(w))
```

   Le `-1` n'est pas décoratif : sans lui, la transaction se voit elle-même et
   la performance devient irréaliste.

2. **`conf/config.yaml`**, section `variables` — la déclarer.

3. **`serving/api.py`** — l'ajouter au schéma d'entrée, sinon le service
   rejettera la requête à cause de la signature MLflow.

Puis relancer :

```powershell
python -m fraude360 tout --depuis variables
```

### 3.2 Ajouter une règle experte

Dans `serving/regles.py`, une entrée de `regles_de_base()` :

```python
Regle("R12", "Libellé lisible par un enquêteur",
      lambda d: (d["montant"] > 1000) & (d["est_nuit"] == 1),
      score=0.45, gravite="elevee", typologie_visee="usurpation_de_compte")
```

Une règle se déploie en heures ; un modèle demande un cycle de réentraînement
complet. C'est pourquoi les deux cohabitent.

### 3.3 Changer le coût métier

`conf/config.yaml`, section `modelisation`. Le seuil est recalculé
automatiquement. Le test `test_seuil_optimal_minimise_le_cout` verrouille le
calcul.

### 3.4 Brancher un vrai modèle de langage

```powershell
ollama pull llama3.1:8b
python -m fraude360 triage-genai
```

Sans Ollama, un client de repli déterministe prend le relais et l'évaluation
s'applique de la même façon. Le rapport indique toujours **quel client a servi**.

### 3.5 Avant chaque livraison

```powershell
ruff check src/ tests/
pytest -q
```

Les 19 tests verrouillent les erreurs invisibles : fuite temporelle,
instabilité de l'affectation A/B, erreur de coût, invention de politique par le
LLM, non-monotonie d'un découpage crédit, équivalence Python/SAS.

---

## Partie 4 — Déployer sur Azure Databricks

### 4.1 Créer les ressources

```powershell
.\infra\azure\01_creer_ressources.ps1
.\infra\azure\02_televerser_donnees.ps1
# et, après la démonstration :
.\infra\azure\99_supprimer_ressources.ps1
```

### 4.2 Basculer la configuration

Une seule modification dans `conf/config.yaml` :

```yaml
environnement:
  cible: databricks
  format_table: delta
```

### 4.3 Correspondance local → Databricks

| Local | Azure Databricks |
|---|---|
| `data/00_landing/` | Volume Unity Catalog alimenté par ADLS Gen2 |
| lecture de répertoire | **Auto Loader** (`cloudFiles`), point de reprise |
| Parquet | **Delta Lake** + Unity Catalog |
| Spark `local[*]` | cluster à mise à l'échelle automatique |
| MLflow sur SQLite | MLflow du workspace + registre Unity Catalog |
| API FastAPI | **Databricks Model Serving**, alias `champion` |
| `pipeline.py` | **Databricks Workflows** |
| CSV → Power BI | tables Delta `gold` en **Direct Lake** |

### 4.4 Déployer le job

```bash
databricks bundle deploy -t dev
databricks bundle run pipeline_fraude -t dev
```

L'orchestration est versionnée dans `infra/databricks/databricks.yml` : le job,
ses dépendances entre tâches, ses clusters et sa planification sont dans Git,
pas cliqués dans une interface.

---

## Partie 5 — Le volet crédit et SAS

### 5.1 Chaîne Python

```powershell
python -m fraude360.credit.pipeline_credit
```

Produit `artifacts/credit/` : grille de score, tables WOE, rapport de
validation, portefeuille IFRS 9, figure.

### 5.2 Portage SAS

```powershell
python -m fraude360.credit.export_sas
```

Génère `sas/donnees/portefeuille.csv`, `sas/00_import.sas` et
`sas/05_scoring_grille_figee.sas` — la grille **figée en code SAS**.

Puis, dans SAS 9.4 ou Viya, dans l'ordre :

| Programme | Contenu |
|---|---|
| `00_import.sas` | import + contrôle d'intégrité |
| `01_woe_iv.sas` | discrétisation WOE, IV, contrôle de monotonie |
| `02_scorecard.sas` | `PROC LOGISTIC` sur WOE, mise en points |
| `03_validation.sas` | Gini, KS, Hosmer-Lemeshow, PSI, rétrovalidation |
| `04_ifrs9_ecl.sas` | étapes IFRS 9, ECL, sensibilité |
| `05_scoring_grille_figee.sas` | scoring sans dépendance Python |

Adapter `%let CHEMIN = ...` en tête de chaque programme.

### 5.3 La preuve d'équivalence

Dans une banque, un modèle développé en Python est fréquemment réimplanté en
SAS pour la production, et la validation **exige la preuve que les deux
implantations donnent le même score**.

`export_sas.verifier_equivalence` la produit. Elle a détecté un écart réel de
**23 points** : bornes de classe ouvertes à droite côté SAS, fermées côté
`pandas.cut`. Écart désormais nul, verrouillé par un test.

Rejouer cette vérification **après chaque réentraînement**.

---

## Partie 6 — Incidents fréquents

| Symptôme | Cause | Correction |
|---|---|---|
| `JAVA_HOME is not set` | Java absent ou non déclaré | installer le JDK 17, définir `JAVA_HOME` |
| `Python worker exited unexpectedly` | versions Python driver / worker différentes | `utils/spark.py` force `PYSPARK_PYTHON` ; vérifier que le venv est actif |
| `ModuleNotFoundError: fraude360` | `PYTHONPATH` absent | `$env:PYTHONPATH = "src"` |
| Étape 13 en repli déterministe | Ollama ne répond pas | normal hors ligne ; `ollama pull llama3.1:8b` pour le vrai modèle |
| `decisions.jsonl` absent | étape 08 non exécutée | `python -m fraude360 scorer` |
| Power BI n'actualise pas | chemins relatifs | ouvrir `powerbi/Fraude360.pbip` depuis la racine du dépôt |
| Équivalence SAS non établie | grille réentraînée sans régénérer le code | relancer `export_sas` |

---

## Partie 7 — Ce que le prototype ne fait pas

À dire d'emblée plutôt qu'à faire découvrir.

- **Données synthétiques.** Les étiquettes de fraude n'ont pas le délai de
  confirmation, les rétrofacturations ni les reclassements tardifs du réel.
- **Identifiants en clair.** Acceptable sur des données simulées ;
  pseudonymisation requise avant toute donnée réelle.
- **Pas d'explicabilité locale.** Les règles donnent un motif, le modèle non.
  Pour le traitement des contestations, il faudrait des valeurs de Shapley par
  alerte.
- **Pas de magasin de variables.** Un seul consommateur ne le justifie pas ;
  un second modèle réutilisant ces variables l'exigerait.
- **Volet crédit sur portefeuille simulé.** La mécanique IFRS 9 et la
  validation sont justes ; les niveaux de PD et de LGD ne sont pas des
  références de marché.
