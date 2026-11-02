"""Generation du projet Power BI (.pbip) a partir des exports de l'etape 12.

Un fichier .pbix est un binaire : impossible a versionner, a relire ou a
reviser. Le format .pbip (Power BI Project) est du texte : le modele semantique
(model.bim, TMSL) et le rapport (report.json) sont lisibles, comparables dans
Git et generables par programme. C'est exactement ce que fait ce script.

    python scripts/generer_pbip.py

Il produit :
    powerbi/Fraude360.pbip
    powerbi/Fraude360.SemanticModel/   tables, relations, 24 mesures DAX
    powerbi/Fraude360.Report/          3 pages, cartes, courbes, tableaux

Ouvrir ensuite `powerbi/Fraude360.pbip` avec Power BI Desktop.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

DOSSIER_PBI = RACINE / "powerbi"
DOSSIER_DONNEES = DOSSIER_PBI / "data"
NOM = "Fraude360"
THEME_FICHIER = "FraudeBanque.json"

# Theme professionnel - secteur bancaire / lutte contre la fraude.
# Bleu nuit institutionnel, accents sarcelle, semantique rouge/ambre/vert.
THEME = {
    "name": "Fraude 360 - Banque",
    "dataColors": ["#1F3A5F", "#2C8C99", "#C9A227", "#8C4A3C", "#5B6B8C",
                   "#3E7CB1", "#6BA292", "#B45F5F", "#7A5C8C", "#9AA5B1"],
    "background": "#FFFFFF", "foreground": "#1F2933", "tableAccent": "#1F3A5F",
    "good": "#1B7F4B", "neutral": "#B7791F", "bad": "#B42318",
    "maximum": "#1F3A5F", "center": "#6BA292", "minimum": "#C0D3E0", "null": "#EEF1F4",
    "textClasses": {
        "title": {"fontFace": "Segoe UI Semibold", "fontSize": 12, "color": "#1F2933"},
        "header": {"fontFace": "Segoe UI Semibold", "fontSize": 11, "color": "#1F3A5F"},
        "label": {"fontFace": "Segoe UI", "fontSize": 10, "color": "#3E4C59"},
        "callout": {"fontFace": "Segoe UI Semibold", "fontSize": 32, "color": "#1F3A5F"},
    },
    "visualStyles": {
        "*": {"*": {
            "background": [{"show": True, "color": {"solid": {"color": "#FFFFFF"}},
                           "transparency": 0}],
            "border": [{"show": True, "color": {"solid": {"color": "#E1E5EA"}},
                        "radius": 6}],
            "visualHeader": [{"show": True, "background": {"solid": {"color": "#FFFFFF"}}}],
            "title": [{"show": True, "fontColor": {"solid": {"color": "#1F2933"}},
                       "fontSize": 11, "fontFamily": "Segoe UI Semibold",
                       "alignment": "left"}],
        }},
        "card": {"*": {
            "labels": [{"color": {"solid": {"color": "#1F3A5F"}}, "fontSize": 28,
                        "fontFamily": "Segoe UI Semibold"}],
            "categoryLabels": [{"color": {"solid": {"color": "#6B7684"}}, "fontSize": 9}],
        }},
        "tableEx": {"*": {
            "grid": [{"gridVerticalWeight": 1, "gridHorizontalWeight": 1,
                      "outlineColor": {"solid": {"color": "#E1E5EA"}}}],
            "columnHeaders": [{"fontColor": {"solid": {"color": "#FFFFFF"}},
                               "backColor": {"solid": {"color": "#1F3A5F"}},
                               "fontFamily": "Segoe UI Semibold"}],
            "values": [{"fontColorPrimary": {"solid": {"color": "#1F2933"}},
                        "backColorPrimary": {"solid": {"color": "#FFFFFF"}},
                        "backColorSecondary": {"solid": {"color": "#F4F6F8"}}}],
        }},
    },
}

# --- correspondance des types pandas -> types tabulaires ---------------------
TYPES = {"int64": ("int64", "Int64.Type"), "float64": ("double", "type number"),
         "bool": ("boolean", "type logical"), "datetime64[ns]": ("dateTime", "type datetime"),
         "object": ("string", "type text")}

COLONNES_DATE = {"date"}
COLONNES_ENTIERES = {"annee", "mois", "semaine", "numero_jour", "est_fin_semaine",
                     "transactions", "alertes", "blocages", "fraudes",
                     "fraudes_detectees", "faux_positifs", "cas", "detectes",
                     "est_fraude", "version_modele", "connue_du_modele_v1",
                     "nb_variables", "transactions_touchees", "fraudes_captees",
                     "vrais_positifs", "faux_negatifs"}


# =============================================================================
#  Modele semantique (TMSL)
# =============================================================================
def _type_colonne(nom: str, serie: pd.Series) -> tuple[str, str]:
    if nom in COLONNES_DATE:
        return "dateTime", "type date"
    if nom in COLONNES_ENTIERES:
        return "int64", "Int64.Type"
    return TYPES.get(str(serie.dtype), ("string", "type text"))


def _format(nom: str, type_tabulaire: str) -> str | None:
    if type_tabulaire == "dateTime":
        return "yyyy-mm-dd"
    if type_tabulaire == "int64":
        return "#,0"
    if type_tabulaire == "double":
        if "montant" in nom or "cout" in nom or "economie" in nom:
            return '#,0.00 "$"'
        if "taux" in nom or "pct" in nom or "precision" in nom or "rappel" in nom:
            return "0.00"
        return "#,0.0000"
    return None


def table_depuis_csv(fichier: Path) -> dict:
    df = pd.read_csv(fichier, nrows=500)
    nom_table = fichier.stem
    colonnes, transformations = [], []
    for c in df.columns:
        type_tab, type_m = _type_colonne(c, df[c])
        col = {"name": c, "dataType": type_tab, "sourceColumn": c,
               "summarizeBy": "none" if type_tab in ("string", "dateTime") else "sum",
               "annotations": [{"name": "SummarizationSetBy", "value": "Automatic"}]}
        fmt = _format(c, type_tab)
        if fmt:
            col["formatString"] = fmt
        colonnes.append(col)
        transformations.append(f'{{"{c}", {type_m}}}')

    expression = [
        "let",
        f'    Source = Csv.Document(File.Contents(CheminDonnees & "{fichier.name}"),'
        "[Delimiter=\",\", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),",
        '    Entetes = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),',
        "    Types = Table.TransformColumnTypes(Entetes,{"
        + ", ".join(transformations) + '}, "en-US")',
        "in",
        "    Types",
    ]
    table = {
        "name": nom_table,
        "columns": colonnes,
        "partitions": [{"name": nom_table, "mode": "import",
                        "source": {"type": "m", "expression": expression}}],
        "annotations": [{"name": "PBI_ResultType", "value": "Table"}],
    }
    if nom_table == "dim_date":
        table["dataCategory"] = "Time"
        for col in table["columns"]:
            if col["name"] == "date":
                col["isKey"] = True
    return table


MESURES = {
    "fait_kpi_quotidien": [
        ("Nb transactions", "SUM(fait_kpi_quotidien[transactions])", "#,0",
         "Volume transactionnel traite sur la periode"),
        ("Nb fraudes", "SUM(fait_kpi_quotidien[fraudes])", "#,0",
         "Transactions frauduleuses confirmees"),
        ("Nb fraudes detectees", "SUM(fait_kpi_quotidien[fraudes_detectees])", "#,0",
         "Fraudes captees par le dispositif (modele ou regles)"),
        ("Nb alertes", "SUM(fait_kpi_quotidien[alertes])", "#,0",
         "Transactions envoyees en revue ou bloquees"),
        ("Nb blocages", "SUM(fait_kpi_quotidien[blocages])", "#,0", "Transactions refusees"),
        ("Taux de fraude", "DIVIDE([Nb fraudes], [Nb transactions])", "0.000 %",
         "Part des transactions frauduleuses"),
        ("Taux de detection",
         "DIVIDE([Nb fraudes detectees], [Nb fraudes])", "0.0 %",
         "Rappel : part des fraudes captees"),
        ("Taux d'alerte", "DIVIDE([Nb alertes], [Nb transactions])", "0.00 %",
         "Charge de travail imposee a l'equipe d'enquete"),
        ("Precision des alertes",
         "DIVIDE([Nb fraudes detectees], [Nb alertes])", "0.0 %",
         "Part des alertes qui sont de vraies fraudes"),
        ("Montant traite", "SUM(fait_kpi_quotidien[montant_total])", '#,0 "$"',
         "Volume monetaire des transactions"),
        ("Pertes evitees", "SUM(fait_kpi_quotidien[montant_fraude_evite])", '#,0 "$"',
         "Dollars de fraude interceptes"),
        ("Pertes subies", "SUM(fait_kpi_quotidien[montant_fraude_subi])", '#,0 "$"',
         "Dollars de fraude passes au travers"),
        ("Pertes totales potentielles", "[Pertes evitees] + [Pertes subies]", '#,0 "$"',
         "Exposition brute a la fraude"),
        ("Rappel en dollars",
         "DIVIDE([Pertes evitees], [Pertes totales potentielles])", "0.0 %",
         "Part des dollars de fraude effectivement protegee"),
        ("Cout des revues", "SUM(fait_kpi_quotidien[cout_revues_manuelles])", '#,0 "$"',
         "Cout operationnel du traitement des alertes"),
        ("Cout total", "SUM(fait_kpi_quotidien[cout_total])", '#,0 "$"',
         "Pertes subies + cout des revues manuelles"),
        ("Cout pour 1000 transactions",
         "DIVIDE([Cout total], [Nb transactions]) * 1000", '#,0.00 "$"',
         "Indicateur de pilotage normalise par le volume"),
        ("Pertes evitees 7 jours",
         "CALCULATE([Pertes evitees], DATESINPERIOD(dim_date[date], "
         "MAX(dim_date[date]), -7, DAY))", '#,0 "$"',
         "Dollars proteges sur les sept derniers jours"),
        ("Taux de detection 7 jours",
         "CALCULATE([Taux de detection], DATESINPERIOD(dim_date[date], "
         "MAX(dim_date[date]), -7, DAY))", "0.0 %",
         "Rappel glissant : le signal a surveiller au quotidien"),
        ("Variation du taux d'alerte",
         "VAR Courant = [Taux d'alerte]\n"
         "VAR Precedent = CALCULATE([Taux d'alerte], "
         "DATEADD(dim_date[date], -7, DAY))\n"
         "RETURN DIVIDE(Courant - Precedent, Precedent)", "+0.0 %;-0.0 %;0 %",
         "Evolution de la charge d'alertes par rapport a la semaine precedente"),
    ],
    "fait_alertes": [
        ("Alertes en attente", "COUNTROWS(fait_alertes)", "#,0",
         "Transactions dans la file d'enquete"),
        ("Montant sous enquete", "SUM(fait_alertes[montant])", '#,0 "$"',
         "Dollars concernes par les alertes"),
        ("Score moyen des alertes", "AVERAGE(fait_alertes[score_modele])", "0.000",
         "Niveau de risque moyen de la file"),
        ("Alertes critiques",
         "CALCULATE(COUNTROWS(fait_alertes), fait_alertes[priorite] = \"1-critique\")",
         "#,0", "Alertes a traiter en priorite absolue"),
    ],
    "fait_typologie_quotidien": [
        ("Cas par typologie", "SUM(fait_typologie_quotidien[cas])", "#,0",
         "Nombre de fraudes par typologie"),
        ("Pertes par typologie", "SUM(fait_typologie_quotidien[montant])", '#,0 "$"',
         "Montant en jeu par typologie"),
        ("Detection par typologie",
         "DIVIDE(SUM(fait_typologie_quotidien[detectes]), "
         "SUM(fait_typologie_quotidien[cas]))", "0.0 %",
         "Rappel decompose par mode operatoire"),
    ],
    "fait_derive": [
        ("PSI maximal", "MAX(fait_derive[psi])", "0.000",
         "Derive la plus forte observee ; alerte au-dela de 0,25"),
        ("Variables en alerte",
         "CALCULATE(DISTINCTCOUNT(fait_derive[variable]), fait_derive[psi] >= 0.25)",
         "#,0", "Nombre de variables dont la distribution a change"),
    ],
    "fait_performance_modele": [
        ("PR-AUC", "MAX(fait_performance_modele[pr_auc])", "0.000",
         "Aire sous la courbe precision-rappel du modele"),
        ("Economie annoncee", "SUM(fait_performance_modele[economie_vs_sans_modele])",
         '#,0 "$"', "Gain estime du modele par rapport a l'absence de dispositif"),
    ],
}


def construire_modele(fichiers: list[Path], chemin_donnees: str) -> dict:
    tables = [table_depuis_csv(f) for f in fichiers]
    presentes = {t["name"] for t in tables}

    for table in tables:
        for nom, dax, fmt, description in MESURES.get(table["name"], []):
            table.setdefault("measures", []).append({
                "name": nom, "expression": dax, "formatString": fmt,
                "description": description,
                "annotations": [{"name": "PBI_FormatHint",
                                 "value": "{\"isGeneralNumber\":false}"}]})

    relations = []
    def _relier(table_fait, colonne_fait, table_dim, colonne_dim):
        if table_fait in presentes and table_dim in presentes:
            relations.append({
                "name": str(uuid.uuid4()),
                "fromTable": table_fait, "fromColumn": colonne_fait,
                "toTable": table_dim, "toColumn": colonne_dim,
                "crossFilteringBehavior": "oneDirection"})

    for fait in ("fait_kpi_quotidien", "fait_alertes", "fait_typologie_quotidien",
                 "fait_derive"):
        _relier(fait, "date", "dim_date", "date")
    _relier("fait_typologie_quotidien", "typologie_fraude", "dim_typologie", "typologie")
    _relier("fait_alertes", "typologie_fraude", "dim_typologie", "typologie")

    parametre = {
        "name": "CheminDonnees",
        "kind": "m",
        "expression": [f'"{chemin_donnees}" meta [IsParameterQuery=true, '
                       'Type="Text", IsParameterQueryRequired=true]'],
    }

    return {
        "name": NOM,
        "compatibilityLevel": 1567,
        "model": {
            "culture": "fr-CA",
            "dataAccessOptions": {"legacyRedirects": True,
                                  "returnErrorValuesAsNull": True},
            "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "sourceQueryCulture": "fr-CA",
            "expressions": [parametre],
            "tables": tables,
            "relationships": relations,
            "annotations": [{"name": "PBI_QueryOrder",
                             "value": json.dumps([t["name"] for t in tables])},
                            {"name": "__PBI_TimeIntelligenceEnabled", "value": "0"}],
        },
    }


# =============================================================================
#  Rapport (report.json, format historique lisible par Power BI Desktop)
# =============================================================================
_compteur = [0]


def _identifiant() -> str:
    _compteur[0] += 1
    return f"visuel{_compteur[0]:03d}"


def _mesure(table: str, mesure: str, alias: str | None = None) -> dict:
    return {"Measure": {"Expression": {"SourceRef": {"Entity": table}},
                        "Property": mesure},
            "Name": f"{table}.{mesure}", "NativeReferenceName": alias or mesure}


def _colonne(table: str, colonne: str, alias: str | None = None) -> dict:
    return {"Column": {"Expression": {"SourceRef": {"Entity": table}},
                       "Property": colonne},
            "Name": f"{table}.{colonne}", "NativeReferenceName": alias or colonne}


def _agregat(table: str, colonne: str, fonction: int = 0,
             alias: str | None = None) -> dict:
    noms = {0: "Sum", 1: "Avg", 2: "Min", 3: "Max", 4: "Count"}
    return {"Aggregation": {"Expression": {"Column": {
                "Expression": {"SourceRef": {"Entity": table}}, "Property": colonne}},
            "Function": fonction},
            "Name": f"{noms[fonction]}({table}.{colonne})",
            "NativeReferenceName": alias or colonne}


def _visuel(type_visuel: str, x: int, y: int, largeur: int, hauteur: int,
            entites: list[str], projections: dict[str, list[dict]],
            titre: str | None = None, tri: dict | None = None) -> dict:
    nom = _identifiant()
    selections = [s for liste in projections.values() for s in liste]
    prototype = {
        "Version": 2,
        "From": [{"Name": e[0].lower() + str(i), "Entity": e, "Type": 0}
                 for i, e in enumerate(entites)],
        "Select": selections,
    }
    # Les references SourceRef par entite sont converties en alias de la clause From
    alias = {e: f"{e[0].lower()}{i}" for i, e in enumerate(entites)}
    def _remplacer(obj):
        if isinstance(obj, dict):
            if "SourceRef" in obj and "Entity" in obj["SourceRef"]:
                return {"SourceRef": {"Source": alias[obj["SourceRef"]["Entity"]]}}
            return {k: _remplacer(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_remplacer(v) for v in obj]
        return obj
    prototype = _remplacer(prototype)
    if tri:
        prototype["OrderBy"] = [tri]

    config = {
        "name": nom,
        "layouts": [{"id": 0, "position": {"x": x, "y": y, "z": _compteur[0],
                                           "width": largeur, "height": hauteur}}],
        "singleVisual": {
            "visualType": type_visuel,
            "projections": {r: [{"queryRef": s["Name"]} for s in liste]
                            for r, liste in projections.items()},
            "prototypeQuery": prototype,
            "drillFilterOtherVisuals": True,
            "objects": {},
        },
    }
    if titre:
        config["singleVisual"]["vcObjects"] = {"title": [{"properties": {
            "show": {"expr": {"Literal": {"Value": "true"}}},
            "text": {"expr": {"Literal": {"Value": f"'{titre}'"}}},
            "fontSize": {"expr": {"Literal": {"Value": "11D"}}},
        }}]}
    return {"x": x, "y": y, "z": _compteur[0], "width": largeur, "height": hauteur,
            "config": json.dumps(config, ensure_ascii=False), "filters": "[]"}


def construire_rapport(tables: set[str]) -> dict:
    K = "fait_kpi_quotidien"
    sections = []

    # ---------------- page 1 : pilotage ------------------------------------
    v = []
    cartes = [("Nb transactions", "Transactions"), ("Nb fraudes", "Fraudes confirmees"),
              ("Taux de detection", "Taux de detection"), ("Pertes evitees", "Pertes evitees"),
              ("Pertes subies", "Pertes subies"),
              ("Cout pour 1000 transactions", "Cout / 1000 tx")]
    for i, (mesure, libelle) in enumerate(cartes):
        v.append(_visuel("card", 16 + i * 208, 12, 196, 96, [K],
                         {"Values": [_mesure(K, mesure)]}, libelle))
    v.append(_visuel("lineChart", 16, 120, 820, 260, [K, "dim_date"],
                     {"Category": [_colonne("dim_date", "date")],
                      "Y": [_mesure(K, "Pertes evitees"), _mesure(K, "Pertes subies")]},
                     "Dollars de fraude evites et subis, par jour"))
    v.append(_visuel("lineChart", 844, 120, 420, 260, [K, "dim_date"],
                     {"Category": [_colonne("dim_date", "date")],
                      "Y": [_mesure(K, "Taux de detection"), _mesure(K, "Taux d'alerte")]},
                     "Taux de detection et charge d'alertes"))
    if "fait_typologie_quotidien" in tables:
        T = "fait_typologie_quotidien"
        v.append(_visuel("donutChart", 16, 392, 400, 288, [T],
                         {"Category": [_colonne(T, "typologie_fraude")],
                          "Y": [_mesure(T, "Pertes par typologie")]},
                         "Repartition des pertes par typologie"))
        v.append(_visuel("clusteredBarChart", 424, 392, 412, 288, [T],
                         {"Category": [_colonne(T, "typologie_fraude")],
                          "Y": [_mesure(T, "Detection par typologie")]},
                         "Taux de detection par typologie"))
        v.append(_visuel("tableEx", 844, 392, 420, 288, [T],
                         {"Values": [_colonne(T, "typologie_fraude"),
                                     _mesure(T, "Cas par typologie"),
                                     _mesure(T, "Pertes par typologie")]},
                         "Detail par typologie"))
    sections.append({"name": "page_pilotage", "displayName": "1 - Pilotage",
                     "filters": "[]", "ordinal": 0, "visualContainers": v,
                     "config": "{}", "displayOption": 1, "width": 1280, "height": 720})

    # ---------------- page 2 : file d'enquete -------------------------------
    v = []
    if "fait_alertes" in tables:
        A = "fait_alertes"
        for i, (mesure, libelle) in enumerate([
                ("Alertes en attente", "Alertes"), ("Alertes critiques", "Critiques"),
                ("Montant sous enquete", "Montant en jeu"),
                ("Score moyen des alertes", "Score moyen")]):
            v.append(_visuel("card", 16 + i * 312, 12, 296, 96, [A],
                             {"Values": [_mesure(A, mesure)]}, libelle))
        v.append(_visuel("clusteredColumnChart", 16, 120, 620, 250, [A],
                         {"Category": [_colonne(A, "priorite")],
                          "Y": [_mesure(A, "Alertes en attente")]},
                         "File d'enquete par niveau de priorite"))
        v.append(_visuel("clusteredBarChart", 644, 120, 620, 250, [A],
                         {"Category": [_colonne(A, "categorie_marchand")],
                          "Y": [_mesure(A, "Montant sous enquete")]},
                         "Montant sous enquete par secteur marchand"))
        v.append(_visuel("tableEx", 16, 382, 1248, 300, [A],
                         {"Values": [_colonne(A, "transaction_id"),
                                     _colonne(A, "date"),
                                     _colonne(A, "canal"),
                                     _colonne(A, "categorie_marchand"),
                                     _agregat(A, "montant", 0),
                                     _agregat(A, "score_modele", 3, "score"),
                                     _colonne(A, "regles_declenchees"),
                                     _colonne(A, "decision"),
                                     _colonne(A, "statut_enquete")]},
                         "Transactions a examiner, les plus risquees en premier"))
    sections.append({"name": "page_enquete", "displayName": "2 - File d'enquete",
                     "filters": "[]", "ordinal": 1, "visualContainers": v,
                     "config": "{}", "displayOption": 1, "width": 1280, "height": 720})

    # ---------------- page 3 : modele et surveillance -----------------------
    v = []
    if "fait_performance_modele" in tables:
        P = "fait_performance_modele"
        v.append(_visuel("tableEx", 16, 12, 620, 220, [P],
                         {"Values": [_colonne(P, "jeu_variables"),
                                     _colonne(P, "algorithme"),
                                     _agregat(P, "pr_auc", 3, "PR-AUC"),
                                     _agregat(P, "rappel", 3, "rappel"),
                                     _agregat(P, "precision", 3, "precision"),
                                     _agregat(P, "cout_par_1000_tx", 3, "cout/1000tx")]},
                         "Comparaison des versions du modele"))
    if "fait_ab_test" in tables:
        B = "fait_ab_test"
        v.append(_visuel("clusteredColumnChart", 644, 12, 620, 220, [B],
                         {"Category": [_colonne(B, "strategie")],
                          "Y": [_agregat(B, "cout_par_1000_tx", 1, "cout/1000tx")]},
                         "Test A/B : cout par 1 000 transactions selon la strategie"))
    if "fait_derive" in tables:
        D = "fait_derive"
        v.append(_visuel("lineChart", 16, 244, 820, 250, [D, "dim_date"],
                         {"Category": [_colonne("dim_date", "date")],
                          "Y": [_mesure(D, "PSI maximal")],
                          "Series": [_colonne(D, "variable")]},
                         "Derive des variables (PSI) - alerte au-dela de 0,25"))
        v.append(_visuel("tableEx", 844, 244, 420, 250, [D],
                         {"Values": [_colonne(D, "variable"), _mesure(D, "PSI maximal")]},
                         "Variables les plus instables"))
    if "fait_performance_regles" in tables:
        R = "fait_performance_regles"
        v.append(_visuel("tableEx", 16, 506, 1248, 190, [R],
                         {"Values": [_colonne(R, "identifiant"), _colonne(R, "libelle"),
                                     _agregat(R, "fraudes_captees", 0),
                                     _agregat(R, "precision_pct", 3, "precision %"),
                                     _agregat(R, "couverture_fraude_pct", 3,
                                              "couverture %")]},
                         "Performance individuelle des regles expertes"))
    sections.append({"name": "page_modele", "displayName": "3 - Modele et surveillance",
                     "filters": "[]", "ordinal": 2, "visualContainers": v,
                     "config": "{}", "displayOption": 1, "width": 1280, "height": 720})

    return {
        "config": json.dumps({"version": "5.55", "activeSectionIndex": 0,
                              "defaultDrillFilterOtherVisuals": True,
                              "themeCollection": {"baseTheme": {"name": "CY24SU10"},
                                                  "customTheme": {"name": THEME_FICHIER}},
                              "settings": {"useStylableVisualContainerHeader": True}}),
        "layoutOptimization": 0,
        "resourcePackages": [{
            "resourcePackage": {
                "name": "RegisteredResources", "type": "RegisteredResources",
                "items": [{"type": "CustomTheme", "name": THEME_FICHIER,
                           "path": THEME_FICHIER}]}}],
        "sections": sections,
    }


# =============================================================================
#  Ecriture du projet
# =============================================================================
def _platform(type_element: str, nom: str) -> dict:
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/"
                   "platformProperties/2.0.0/schema.json",
        "metadata": {"type": type_element, "displayName": nom},
        "config": {"version": "2.0", "logicalId": str(uuid.uuid4())},
    }


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    texte = (json.dumps(contenu, indent=2, ensure_ascii=False)
             if not isinstance(contenu, str) else contenu)
    chemin.write_text(texte, encoding="utf-8")


def executer(chemin_donnees: str | None = None) -> None:
    fichiers = sorted(DOSSIER_DONNEES.glob("*.csv"))
    if not fichiers:
        raise SystemExit(
            f"Aucun fichier CSV dans {DOSSIER_DONNEES}.\n"
            "Executer d'abord :  python -m fraude360 exporter-bi")

    print(f"Tables detectees : {len(fichiers)}")
    for f in fichiers:
        print("   -", f.name)

    # Chemin injecte dans le parametre Power Query. Par defaut, le dossier reel
    # de cette machine ; --chemin permet d'en imposer un autre (autre poste).
    brut = chemin_donnees or str(DOSSIER_DONNEES)
    chemin_m = brut.replace("\\", "\\\\")
    if not chemin_m.endswith(("\\\\", "/")):
        chemin_m += "\\\\" if "\\\\" in chemin_m else "/"

    modele = construire_modele(fichiers, chemin_m)
    rapport = construire_rapport({f.stem for f in fichiers})

    dossier_modele = DOSSIER_PBI / f"{NOM}.SemanticModel"
    dossier_rapport = DOSSIER_PBI / f"{NOM}.Report"

    _ecrire(dossier_modele / ".platform", _platform("SemanticModel", NOM))
    _ecrire(dossier_modele / "definition.pbism", {"version": "4.2", "settings": {}})
    _ecrire(dossier_modele / "model.bim", modele)

    _ecrire(dossier_rapport / ".platform", _platform("Report", NOM))
    _ecrire(dossier_rapport / "definition.pbir",
            {"version": "4.0",
             "datasetReference": {"byPath": {"path": f"../{NOM}.SemanticModel"}}})
    _ecrire(dossier_rapport / "report.json", rapport)
    _ecrire(dossier_rapport / "StaticResources" / "RegisteredResources" / THEME_FICHIER,
            THEME)

    _ecrire(DOSSIER_PBI / f"{NOM}.pbip", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/"
                   "pbipProperties/1.0.0/schema.json",
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{NOM}.Report"}}],
        "settings": {"enableAutoRecovery": True}})

    n_mesures = sum(len(m) for m in MESURES.values())
    print(f"\nProjet Power BI genere :")
    print(f"   {DOSSIER_PBI / (NOM + '.pbip')}")
    print(f"   {len(modele['model']['tables'])} tables, "
          f"{len(modele['model']['relationships'])} relations, {n_mesures} mesures DAX,"
          f" {len(rapport['sections'])} pages")
    print("\nOuvrir Fraude360.pbip avec Power BI Desktop, puis Actualiser.")


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser(description="Genere le projet Power BI .pbip")
    a.add_argument("--chemin", help="chemin du dossier powerbi/data a inscrire dans "
                                    "le parametre CheminDonnees du modele")
    executer(a.parse_args().chemin)
