# Audit — écart entre le prototype et la pratique industrielle

Revue du code au 4 septembre 2026. Chaque constat est étayé par une lecture du
code ou par un artefact d'exécution, jamais par une grille générique.

**Verdict.** Le noyau méthodologique est au niveau de la pratique bancaire.
Les écarts qui restaient portaient sur la **gouvernance** et l'**outillage**,
pas sur la modélisation. Ils ont été comblés pendant cet audit.

---

## 1. Ce qui était déjà au niveau industriel

| Pratique | Preuve dans le code |
|---|---|
| **Correction temporelle des variables** | `features/gold.py:44` — fenêtres fermées à `-1` seconde, `rangeBetween` sur `ts_unix`. La transaction courante ne se voit jamais elle-même. |
| **Découpage temporel strict** | `modeling/donnees.py` — aucun mélange aléatoire, test verrouillé par `test_la_reputation_marchand_ignore_la_transaction_courante`. |
| **Calibration des probabilités** | `CalibratedClassifierCV`, score de Brier suivi (0,0024 en test). Sans cela, l'optimisation économique du seuil n'a pas de sens. |
| **Seuil choisi en dollars** | `modeling/entrainement.py` — minimisation du coût attendu sous contrainte de capacité d'enquête, pas maximisation du F1. |
| **Signature de modèle MLflow** | `entrainement.py:257` — `infer_signature` + `input_example`. C'est ce qui fait échouer proprement un appel de service au schéma dérivé. |
| **Champion / challenger** | `modeling/registre.py` — alias MLflow, promotion sous trois conditions simultanées. |
| **A/B méthodologiquement correct** | `abtest/experience.py` — randomisation **par carte**, bootstrap **par grappes**, contrôle A/A. Randomiser par transaction aurait fait fuir les variables de vélocité d'un bras à l'autre. |
| **Surveillance en trois couches** | PSI et Kolmogorov-Smirnov, dérive opérationnelle, dérive de performance sur étiquettes matures seulement. |
| **Ingestion idempotente** | `ingestion/bronze.py` — registre des fichiers traités, schéma déclaré, quarantaine tracée avec motif. |
| **Orchestration versionnée** | `infra/databricks/databricks.yml` — Asset Bundle : le job, ses dépendances et sa planification sont dans Git, pas cliqués dans une interface. |
| **Secrets** | `infra/azure/01_creer_ressources.ps1` — Key Vault, aucun secret en clair dans le dépôt. |

---

## 2. Écarts constatés, et ce qui a été fait

### 2.1 Aucune fiche de modèle — **comblé**

Un dossier de modèle est exigé par la ligne directrice **E-23 du BSIF** et par
toute deuxième ligne de défense. Le prototype n'en produisait aucun.

Ajouté : `gouvernance/carte_modele.py`, étape `carte-modele`. La fiche est
**générée depuis les artefacts**, jamais saisie — une fiche recopiée à la main
diverge du modèle dès la version suivante. Elle couvre l'identification et la
lignée, l'usage prévu, **les usages proscrits**, les données, la méthode, la
performance, les limites connues, la surveillance et les conditions de retrait.

### 2.2 Aucune analyse d'impact disparate — **comblé**

Le modèle ne reçoit ni âge, ni sexe, ni origine : pas de discrimination directe
possible. Cela ne suffit pas — des variables neutres en apparence peuvent être
corrélées à un attribut protégé.

Ajouté : `gouvernance/equite.py`, étape `equite`. Deux indicateurs : ratio
d'impact (règle des quatre cinquièmes) et **écart de taux de faux positifs**.
Le second est celui qui compte : un blocage plus fréquent est légitime si le
groupe fraude réellement davantage ; ce qui ne l'est pas, c'est de bloquer plus
souvent des transactions **légitimes**.

Résultat sur les données courantes — écarts réels, à justifier :

| Sous-groupe | Faux positifs vs référence |
|---|---|
| Canal INTERAC | **1,55×** |
| Marchands électronique | 1,67× |
| Montants 500–2000 $ | **2,96×** |

Ces écarts sont explicables, mais ils sont désormais **mesurés et documentés**
plutôt qu'ignorés.

### 2.3 Ni conteneur, ni intégration continue, ni analyse statique — **comblé**

Ajoutés : `Dockerfile` (utilisateur non privilégié — un conteneur qui tourne en
root est un constat d'audit), `.github/workflows/ci.yml` (ruff + pytest +
**pipeline complet sur le jeu réduit**), `ruff.toml`.

### 2.4 Précision à corriger dans le README — **constat maintenu**

Le README annonçait « API temps réel < 100 ms ». Mesure réelle sur
`artifacts/service/decisions.jsonl` :

```
n = 22   médiane 45,5 ms   p95 68,6 ms   max 411,4 ms
20 / 22 sous 100 ms
```

La promesse tient **au p95**, pas au maximum : le premier appel paie le
démarrage à froid (411 ms). Une affirmation de latence sans percentile ni
mention du démarrage à froid ne survit pas à une revue d'architecture.

---

## 3. Écarts assumés, non comblés

Ces choix sont défendables pour un prototype, mais doivent être nommés
plutôt que passés sous silence.

| Écart | Pourquoi il est assumé | Ce qu'exigerait la production |
|---|---|---|
| **Identifiants en clair** | données synthétiques, aucun renseignement réel | pseudonymisation des `carte_id` et `client_id` dans les couches d'analyse, jetons réversibles en coffre |
| **Pas d'explicabilité locale** | les règles fournissent déjà un motif lisible par alerte | valeurs de Shapley par alerte, ou codes de motif normalisés, pour le traitement des contestations |
| **Pas de magasin de variables** | un seul consommateur | magasin partagé avec correction point-in-time garantie dès qu'un second modèle réutilise ces variables |
| **Pas de reprise sur incident** | exécution locale | reprise de tâche Databricks, politique de nouvelle tentative, alerte sur échec |
| **Étiquettes de fraude simulées** | pas d'accès à des données réelles | délai de confirmation réel, gestion des rétrofacturations et des reclassements tardifs |

---

## 4. Ce que l'audit a fait ressortir de plus utile

Deux défauts **réels** ont été trouvés par les contrôles eux-mêmes, pas par
lecture :

**Le triage GenAI ne discriminait pas.** L'évaluation contre vérité terrain a
montré 58 alertes sur 60 classées `BLOQUER` : une règle grave (`R02`, `R11`)
court-circuitait le score. Corrigé — la règle relève désormais un **plancher**
(POL-06), elle ne force plus la décision. Après correction :
`REVUE_MANUELLE` 3,1 % de fraude réelle contre `BLOQUER` 77,8 %.

**Le portage SAS divergeait de 23 points.** Le contrôle d'équivalence
Python/SAS a détecté que les bornes de classe étaient ouvertes à droite côté
SAS et fermées côté `pandas.cut`. Écart maintenant nul, et verrouillé par
`test_equivalence_python_sas_du_scoring`.

C'est l'argument central à tenir en entrevue : **la valeur d'un dispositif de
contrôle se démontre par ce qu'il attrape**, pas par sa présence au
diagramme.

---

## 5. État après audit

| Dimension | Avant | Après |
|---|---|---|
| Étapes du pipeline | 13 | 15 (+ volet crédit distinct) |
| Tests de non-régression | 9 | 19 |
| Fiche de modèle | absente | générée, 10 sections |
| Équité | absente | 3 axes, écarts chiffrés |
| Conteneur et CI | absents | Dockerfile + GitHub Actions + ruff |
| Preuve de portage SAS | absente | équivalence établie, écart nul |
