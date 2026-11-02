"""Tests de non-regression sur les points ou une erreur serait invisible.

En detection de fraude, les bogues dangereux ne font pas planter le programme :
ils gonflent les performances. Ces tests verrouillent les trois plus courants.

    pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraude360.abtest.experience import affecter, comparer_deux_proportions
from fraude360.modeling.evaluation import cout_metier, evaluer, seuil_optimal
from fraude360.monitoring.surveillance import psi
from fraude360.serving.regles import MoteurDeRegles

ECONOMIE = {"cout_faux_negatif_ratio": 1.0, "cout_faux_positif": 12.0,
            "cout_vrai_positif": 4.0, "gain_vrai_negatif": 0.0}


# --- 1. economie de la decision ---------------------------------------------
def test_cout_metier_est_exact():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])          # 1 VP, 1 FN, 1 FP, 1 VN
    montants = np.array([100.0, 500.0, 50.0, 20.0])
    c = cout_metier(y, pred, montants, ECONOMIE)
    assert (c["vp"], c["fn"], c["fp"], c["vn"]) == (1, 1, 1, 1)
    assert c["montant_subi"] == 500.0        # la fraude ratee coute son montant
    assert c["cout_total"] == 500.0 + 12.0 + 4.0


def test_seuil_optimal_minimise_le_cout():
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.01).astype(int)
    scores = np.clip(rng.beta(1, 20, n) + y * 0.6, 0, 1)
    montants = rng.uniform(10, 400, n)
    seuil, tab = seuil_optimal(y, scores, montants, ECONOMIE)
    assert tab.loc[tab.seuil == seuil, "cout_total"].iloc[0] == tab.cout_total.min()


# --- 2. absence de fuite temporelle ------------------------------------------
def test_la_reputation_marchand_ignore_la_transaction_courante():
    """Si la variable voyait la ligne courante, elle vaudrait 1 pour toute fraude
    isolee chez un marchand : ce serait une fuite parfaite."""
    from fraude360.features.gold import variables_de_base
    pytest.importorskip("pyspark")
    from fraude360.utils.spark import obtenir_spark

    spark = obtenir_spark()
    base = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "horodatage": pd.to_datetime(["2026-01-01 10:00:00", "2026-01-01 11:00:00"]),
        "horodatage_label": pd.to_datetime(["2026-01-05", "2026-01-05"]),
        "carte_id": ["C1", "C1"], "client_id": ["M1", "M1"],
        "marchand_id": ["X", "X"], "categorie_marchand": ["epicerie", "epicerie"],
        "canal": ["POS", "POS"], "mode_saisie": ["puce", "puce"],
        "montant": [10.0, 20.0], "pays": ["CA", "CA"],
        "latitude": [45.5, 45.5], "longitude": [-73.5, -73.5],
        "authentification_3ds": [True, True], "appareil_id": ["A1", "A1"],
        "adresse_ip": ["1.1.1.1", "1.1.1.1"], "duree_session_s": [0, 0],
        "est_fraude": [1, 1], "typologie_fraude": ["test_de_carte"] * 2,
        "latitude_domicile": [45.5, 45.5], "longitude_domicile": [-73.5, -73.5],
        "age": [40, 40], "anciennete_compte_jours": [1000, 1000],
        "province": ["QC", "QC"], "segment": ["particulier"] * 2,
        "type_carte": ["debit", "debit"], "reseau": ["visa", "visa"],
        "limite": [5000.0, 5000.0],
    })
    res = (variables_de_base(spark.createDataFrame(base))
           .select("transaction_id", "taux_fraude_marchand_30j").toPandas()
           .set_index("transaction_id"))
    # La premiere transaction n'a aucun passe : la variable doit valoir 0.
    assert res.loc["T1", "taux_fraude_marchand_30j"] == 0.0


# --- 3. integrite du test A/B ------------------------------------------------
def test_affectation_stable_par_carte():
    cles = pd.Series(["CAR001", "CAR001", "CAR002", "CAR001"])
    bras = affecter(cles, 0.5)
    assert bras.iloc[0] == bras.iloc[1] == bras.iloc[3]


def test_affectation_equilibree():
    cles = pd.Series([f"CAR{i:06d}" for i in range(20000)])
    part = (affecter(cles, 0.5) == "B_challenger").mean()
    assert 0.48 < part < 0.52


def test_test_de_proportions_detecte_une_vraie_difference():
    r = comparer_deux_proportions(500, 1000, 600, 1000)
    assert r["valeur_p"] < 0.001
    assert r["ic95_bas"] > 0


# --- 4. surveillance ---------------------------------------------------------
def test_psi_nul_sur_distributions_identiques():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    assert psi(x, rng.normal(size=5000)) < 0.05


def test_psi_detecte_un_deplacement():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=5000), rng.normal(loc=2.0, size=5000)) > 0.25


# --- 5. moteur de regles -----------------------------------------------------
def test_regles_declenchent_sur_le_test_de_carte():
    d = pd.DataFrame({
        "nb_tx_carte_1h": [1, 8], "nb_micro_tx_carte_1h": [0, 6],
        "vitesse_kmh": [10.0, 20.0], "canal": ["POS", "ECOM"],
        "authentification_3ds": [True, False], "montant": [50.0, 2.0],
        "est_nuit": [0, 0], "ratio_montant_moyenne_carte": [1.0, 0.1],
        "est_nouvel_appareil_client": [0, 1], "montant_somme_carte_24h": [100.0, 30.0],
        "taux_fraude_marchand_30j": [0.0, 0.05], "nb_pays_distincts_carte_24h": [1, 1],
        "secondes_depuis_tx_precedente": [4000.0, 20.0],
        "est_nouveau_marchand_carte": [0, 1],
    })
    res = MoteurDeRegles().evaluer(d)
    assert res.score_regles.iloc[0] < res.score_regles.iloc[1]
    assert "R02" in res.regles_declenchees.iloc[1]


# --------------------------------------------------------------------------
# Etape 13 - triage par modele de langage
# --------------------------------------------------------------------------
def test_rag_retrouve_la_politique_du_test_de_carte():
    """Une requete decrivant des micro-montants doit ramener POL-01."""
    from fraude360.genai.politiques import Index
    trouves = Index().rechercher(
        "micro-montants repetes sur la carte en une heure test de carte", k=3)
    assert "POL-01" in [p.identifiant for p, _ in trouves]


def test_le_repli_ne_cite_jamais_une_politique_inexistante():
    from fraude360.genai.client import ClientReplique
    from fraude360.genai.politiques import IDENTIFIANTS
    invite = ("[POL-05] Arnaque au virement\nscore du modele : 0.81\n"
              "seuil de blocage : 0.33\nmontant : 640.0\nregles : R11\n"
              "Typologie visee : arnaque_virement_instantane")
    import json
    d = json.loads(ClientReplique().completer(invite))
    assert set(d["politiques_citees"]) <= IDENTIFIANTS


def test_une_regle_grave_ne_court_circuite_pas_le_score():
    """R02 releve un plancher (POL-06), il ne force pas le blocage.

    Sans cette distinction, toute alerte portant R02 partait en BLOQUER et le
    triage perdait tout pouvoir de discrimination.
    """
    import json
    from fraude360.genai.client import ClientReplique
    c = ClientReplique()
    faible = json.loads(c.completer(
        "score du modele : 0.10\nseuil de blocage : 0.33\nmontant : 12.0\nR02"))
    fort = json.loads(c.completer(
        "score du modele : 0.90\nseuil de blocage : 0.33\nmontant : 900.0\nR02"))
    assert faible["action"] == "REVUE_MANUELLE"   # plancher, pas blocage
    assert fort["action"] == "BLOQUER"


def test_ordonnancement_ignore_les_strates_trop_petites():
    import pandas as pd
    from fraude360.genai.evaluation import contre_verite_terrain, NB_MIN_PAR_STRATE
    n = NB_MIN_PAR_STRATE
    notes = pd.DataFrame({
        "action": ["MISE_EN_ATTENTE"] + ["REVUE_MANUELLE"] * n + ["BLOQUER"] * n,
        "est_fraude": [1] + [0] * n + [1] * n,
        "montant": [100.0] * (1 + 2 * n),
    })
    r = contre_verite_terrain(notes)
    assert r["actions_ignorees_effectif_insuffisant"] == ["MISE_EN_ATTENTE"]
    assert r["ordonnancement_respecte"] is True


# --------------------------------------------------------------------------
# Volet CREDIT
# --------------------------------------------------------------------------
def _portefeuille_test(n=6000):
    from fraude360.config import charger_config
    from fraude360.credit import portefeuille
    return portefeuille.generer(charger_config(), n=n, millesimes=24)


def test_woe_monotone_sur_ratio_endettement():
    """Le risque doit croitre avec l'endettement, classe apres classe."""
    from fraude360.credit import scorecard as sc
    d = _portefeuille_test()
    dec = sc.discretiser_numerique(d.ratio_endettement, d.defaut_12m.values,
                                   "ratio_endettement")
    reelles = dec.table[dec.table.classe != "MANQUANT"].sort_values("classe")
    taux = reelles.taux_defaut.values
    assert (taux[1:] >= taux[:-1] - 1e-9).all(), "decoupage non monotone"


def test_valeurs_manquantes_forment_leur_propre_classe():
    """Une valeur manquante n'est jamais imputee : elle porte de l'information."""
    from fraude360.credit import scorecard as sc
    d = _portefeuille_test()
    assert d.ratio_pret_valeur.isna().any()
    dec = sc.discretiser_numerique(d.ratio_pret_valeur, d.defaut_12m.values,
                                   "ratio_pret_valeur")
    assert "MANQUANT" in set(dec.table.classe)


def test_ecl_croit_avec_l_etape_ifrs9():
    """A exposition egale, une etape 2 doit provisionner plus qu'une etape 1."""
    import numpy as np
    from fraude360.credit import ifrs9
    pd_12m = np.array([0.02, 0.02, 0.02])
    lgd = np.array([0.5, 0.5, 0.5])
    ead = np.array([10_000.0] * 3)
    duree = np.array([60, 60, 60])
    etape = np.array([1, 2, 3])
    e = ifrs9.ecl(pd_12m, lgd, ead, duree, etape)
    assert e[0] < e[1] < e[2]


def test_pd_a_vie_superieure_a_pd_12m():
    import numpy as np
    from fraude360.credit import ifrs9
    p12 = np.array([0.01, 0.05, 0.20])
    vie = ifrs9.pd_a_vie(p12, np.array([60, 60, 60]))
    assert (vie > p12).all()


def test_asrc_exige_les_deux_criteres():
    """Doubler une PD minuscule ne doit pas basculer un pret en etape 2."""
    import numpy as np
    from fraude360.credit import ifrs9
    # PD passe de 0,05 % a 0,11 % : doublee, mais hausse absolue negligeable
    etape = ifrs9.classer_etapes(
        np.array([0.0011]), np.array([0.0005]), np.array([36]),
        np.array([0]), np.array([0]))
    assert etape[0] == 1


def test_equivalence_python_sas_du_scoring():
    """Le code SAS genere doit reproduire le score Python au point pres.

    Ce test avait detecte un ecart reel de 23 points : les bornes de classe
    etaient ouvertes a droite cote SAS et fermees cote pandas.
    """
    from fraude360.credit import export_sas, pipeline_credit as pc, scorecard as sc
    d = _portefeuille_test()
    coupure = int(d.millesime.max()) - 5
    dev, val = d[d.millesime < coupure], d[d.millesime >= coupure]
    g = sc.ajuster(dev, dev.defaut_12m.values, pc.VAR_NUM, pc.VAR_CAT)
    r = export_sas.verifier_equivalence(val, g, n=1000)
    assert r["equivalence_etablie"], r
