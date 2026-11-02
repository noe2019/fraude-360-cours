"""ETAPE 07 - Test A/B champion / challenger.

Un modele meilleur en laboratoire ne garantit rien en production. On le prouve
par une experience controlee avant de basculer 100 % du trafic.

Protocole applique ici :
  - unite de randomisation = la CARTE, pas la transaction. Deux transactions de
    la meme carte doivent recevoir la meme strategie, sinon les variables de
    velocite fuient d'un bras a l'autre et l'experience est biaisee ;
  - affectation deterministe par hachage (md5) : reproductible, sans etat, et
    identique en local, sur Databricks et dans l'API ;
  - controle d'equilibre (A/A) avant lecture des resultats : si les deux bras
    n'ont pas le meme profil de trafic, la comparaison ne vaut rien ;
  - lecture sur trois metriques : dollars de fraude evites (metrique primaire),
    taux de detection, charge d'alertes (garde-fou operationnel) ;
  - significativite : test z sur deux proportions pour la detection, et
    intervalle de confiance bootstrap sur l'ecart de cout par 1 000 transactions.

L'experience par defaut oppose le dispositif existant (moteur de regles) au
modele. La meme fonction sert ensuite a comparer deux versions du modele
(etape 11).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.modeling import donnees, evaluation, registre
from fraude360.serving.regles import MoteurDeRegles
from fraude360.utils.journal import chrono, ecrire_json, obtenir_journal, titre

LOG = obtenir_journal("fraude360.abtest")


# =============================================================================
#  Affectation
# =============================================================================
def affecter(cles: pd.Series, part_challenger: float, sel: str = "fraude360") -> pd.Series:
    """Affectation deterministe et stable d'une cle a un bras d'experience."""
    seuil = int(part_challenger * 10000)
    def _bras(v: str) -> str:
        h = hashlib.md5(f"{sel}:{v}".encode()).hexdigest()
        return "B_challenger" if int(h[:8], 16) % 10000 < seuil else "A_champion"
    return cles.astype(str).map(_bras)


def controle_equilibre(df: pd.DataFrame, colonnes: list[str]) -> Dict[str, float]:
    """Test A/A : les deux bras doivent etre comparables avant lecture."""
    res = {}
    a = df[df.bras == "A_champion"]
    b = df[df.bras == "B_challenger"]
    res["part_trafic_B_pct"] = round(100 * len(b) / max(len(df), 1), 2)
    for c in colonnes:
        if c in df.columns and df[c].dtype.kind in "if":
            ma, mb = a[c].mean(), b[c].mean()
            ecart_type = np.sqrt((a[c].var() + b[c].var()) / 2) + 1e-9
            res[f"ecart_standardise_{c}"] = round(float((mb - ma) / ecart_type), 4)
    res["desequilibre_max"] = round(max(
        [abs(v) for k, v in res.items() if k.startswith("ecart_standardise_")] or [0]), 4)
    res["equilibre_acceptable"] = bool(res["desequilibre_max"] < 0.1
                                       and 40 <= res["part_trafic_B_pct"] <= 60)
    return res


# =============================================================================
#  Tests statistiques
# =============================================================================
def comparer_deux_proportions(succes_a: int, n_a: int, succes_b: int, n_b: int) -> Dict[str, float]:
    from scipy import stats
    p_a = succes_a / max(n_a, 1)
    p_b = succes_b / max(n_b, 1)
    p = (succes_a + succes_b) / max(n_a + n_b, 1)
    se = np.sqrt(p * (1 - p) * (1 / max(n_a, 1) + 1 / max(n_b, 1))) + 1e-12
    z = (p_b - p_a) / se
    valeur_p = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p_a * (1 - p_a) / max(n_a, 1) + p_b * (1 - p_b) / max(n_b, 1)) + 1e-12
    return {"taux_A": round(p_a, 4), "taux_B": round(p_b, 4),
            "difference": round(p_b - p_a, 4),
            "ic95_bas": round(p_b - p_a - 1.96 * se_diff, 4),
            "ic95_haut": round(p_b - p_a + 1.96 * se_diff, 4),
            "z": round(float(z), 3), "valeur_p": round(float(valeur_p), 5)}


def bootstrap_ecart_cout(df: pd.DataFrame, economie: dict, n_iter: int,
                         seed: int) -> Dict[str, float]:
    """Intervalle de confiance sur l'ecart de cout par 1 000 transactions.

    On re-echantillonne par CARTE (grappe) et non par transaction, pour respecter
    la dependance intra-carte.
    """
    rng = np.random.default_rng(seed)

    # Le cout est additif ligne a ligne : on l'agrege une seule fois par carte,
    # puis on re-echantillonne des cartes. Le bootstrap devient instantane.
    y = df.est_fraude.values.astype(int)
    p = df.decision.values.astype(int)
    m = df.montant.values.astype(float)
    cout_ligne = (((p == 0) & (y == 1)) * m * economie["cout_faux_negatif_ratio"]
                  + ((p == 1) & (y == 0)) * economie["cout_faux_positif"]
                  + ((p == 1) & (y == 1)) * economie["cout_vrai_positif"])
    agr = (pd.DataFrame({"carte_id": df.carte_id.values, "bras": df.bras.values,
                         "cout": cout_ligne, "n": 1})
           .groupby(["bras", "carte_id"], observed=True)[["cout", "n"]].sum())

    tirages = {}
    for bras in ("A_champion", "B_challenger"):
        bloc = agr.loc[bras]
        couts, effectifs = bloc["cout"].values, bloc["n"].values
        morceaux, restant = [], n_iter
        while restant > 0:                      # par blocs, pour borner la memoire
            k = min(100, restant)
            idx = rng.integers(0, len(couts), size=(k, len(couts)))
            morceaux.append(1000 * couts[idx].sum(axis=1)
                            / np.maximum(effectifs[idx].sum(axis=1), 1))
            restant -= k
        tirages[bras] = np.concatenate(morceaux)
    ecarts = tirages["B_challenger"] - tirages["A_champion"]
    return {"ecart_moyen": round(float(ecarts.mean()), 2),
            "ic95_bas": round(float(np.percentile(ecarts, 2.5)), 2),
            "ic95_haut": round(float(np.percentile(ecarts, 97.5)), 2),
            "probabilite_B_meilleur": round(float((ecarts < 0).mean()), 4)}


# =============================================================================
#  Experience
# =============================================================================
@dataclass
class Strategie:
    nom: str
    decider: Callable[[pd.DataFrame], np.ndarray]   # -> 0/1 par transaction
    scorer: Callable[[pd.DataFrame], np.ndarray] | None = None


def comparer(df: pd.DataFrame, champion: Strategie, challenger: Strategie,
             cfg: Config, nom_experience: str = "regles_vs_modele") -> dict:
    ab = cfg["ab_test"]
    eco = cfg["modelisation"]["economie"]
    titre(f"ETAPE 07 - Test A/B : {champion.nom} (A) contre {challenger.nom} (B)")

    df = df.copy()
    df["bras"] = affecter(df[ab["cle_affectation"]], ab["part_challenger"], nom_experience)

    equilibre = controle_equilibre(df, ["montant", "nb_tx_carte_24h", "heure",
                                        "distance_domicile_km"])
    LOG.info("Controle A/A : %.1f %% du trafic en B, desequilibre max = %.3f -> %s",
             equilibre["part_trafic_B_pct"], equilibre["desequilibre_max"],
             "acceptable" if equilibre["equilibre_acceptable"] else "A VERIFIER")

    with chrono(LOG, "Application des deux strategies sur leur bras respectif"):
        df["decision"] = 0
        ma = df.bras == "A_champion"
        mb = df.bras == "B_challenger"
        df.loc[ma, "decision"] = champion.decider(df[ma]).astype(int)
        df.loc[mb, "decision"] = challenger.decider(df[mb]).astype(int)

    resultats = {}
    for bras, nom in (("A_champion", champion.nom), ("B_challenger", challenger.nom)):
        b = df[df.bras == bras]
        c = evaluation.cout_metier(b.est_fraude.values, b.decision.values,
                                   b.montant.values, eco)
        n_fraudes = int(b.est_fraude.sum())
        alertes = c["vp"] + c["fp"]
        resultats[bras] = {
            "strategie": nom,
            "transactions": int(len(b)),
            "fraudes_reelles": n_fraudes,
            "fraudes_detectees": c["vp"],
            "taux_detection": round(c["vp"] / max(n_fraudes, 1), 4),
            "alertes": alertes,
            "taux_alerte_pct": round(100 * alertes / max(len(b), 1), 3),
            "precision": round(c["vp"] / max(alertes, 1), 4),
            "montant_fraude_evite": round(c["montant_evite"], 2),
            "montant_fraude_subi": round(c["montant_subi"], 2),
            "cout_total": round(c["cout_total"], 2),
            "cout_par_1000_tx": round(1000 * c["cout_total"] / max(len(b), 1), 2),
        }

    a, b = resultats["A_champion"], resultats["B_challenger"]
    test_detection = comparer_deux_proportions(a["fraudes_detectees"], a["fraudes_reelles"],
                                           b["fraudes_detectees"], b["fraudes_reelles"])
    with chrono(LOG, f"Bootstrap par grappes ({ab['nb_bootstrap']} tirages)"):
        boot = bootstrap_ecart_cout(df, eco, ab["nb_bootstrap"], cfg.seed)

    # --- regle de decision, fixee AVANT de regarder les resultats -----------
    # Metrique primaire : le taux de detection. C'est la seule sur laquelle on
    # exige une significativite statistique. Le cout en dollars est une
    # distribution a queue lourde (quelques fraudes de tres gros montant pesent
    # enormement) : exiger un intervalle de confiance strictement negatif sur
    # cette metrique reviendrait a ne jamais rien deployer. Il sert donc de
    # garde-fou directionnel, accompagne de la probabilite que B soit meilleur.
    significatif = test_detection["valeur_p"] < ab["alpha"]
    gain_suffisant = test_detection["difference"] >= ab["gain_minimal_detectable"]
    cout_non_degrade = boot["ecart_moyen"] <= 0
    # Garde-fou operationnel : la charge d'alertes du bras gagnant doit rester
    # absorbable par l'equipe d'enquete. On la compare au plafond configure, et
    # non au bras A : un dispositif A trop laxiste ne doit pas empecher un
    # meilleur modele de passer.
    plafond = float(ab.get("taux_alerte_max_pct", 1.5))
    charge_maitrisee = b["taux_alerte_pct"] <= plafond

    decision = ("DEPLOYER_B" if (significatif and gain_suffisant and cout_non_degrade
                                 and charge_maitrisee)
                else "CONSERVER_A")
    justification = []
    justification.append(
        f"detection {'+' if test_detection['difference']>=0 else ''}"
        f"{100*test_detection['difference']:.1f} pt (p={test_detection['valeur_p']:.4f})"
        f" -> {'significatif' if significatif else 'non significatif'}")
    justification.append(
        f"gain {'suffisant' if gain_suffisant else 'insuffisant'} "
        f"(seuil {100*ab['gain_minimal_detectable']:.0f} pt)")
    justification.append(
        f"cout/1000 tx : {boot['ecart_moyen']:+.2f} $ "
        f"[{boot['ic95_bas']:+.2f} ; {boot['ic95_haut']:+.2f}], "
        f"probabilite que B soit moins cher = {100*boot['probabilite_B_meilleur']:.0f} % "
        f"-> {'garde-fou respecte' if cout_non_degrade else 'cout degrade, B rejete'}")
    justification.append(
        f"charge d'alertes B = {b['taux_alerte_pct']:.2f} % (A : "
        f"{a['taux_alerte_pct']:.2f} %) pour un plafond operationnel de "
        f"{plafond:.2f} % -> {'absorbable' if charge_maitrisee else 'trop elevee'}")

    rapport = {
        "experience": nom_experience,
        "periode": f"{df.horodatage.min()} -> {df.horodatage.max()}",
        "unite_randomisation": ab["cle_affectation"],
        "controle_equilibre": equilibre,
        "bras": resultats,
        "test_taux_detection": test_detection,
        "bootstrap_cout": boot,
        "metrique_primaire": "taux_de_detection",
        "regle_de_decision": {
            "significativite_metrique_primaire": bool(significatif),
            "gain_minimal_atteint": bool(gain_suffisant),
            "garde_fou_cout": bool(cout_non_degrade),
            "garde_fou_charge_alertes": bool(charge_maitrisee),
            "plafond_taux_alerte_pct": plafond,
        },
        "decision": decision,
        "justification": justification,
        "economie_annualisee_estimee": round(
            (a["cout_par_1000_tx"] - b["cout_par_1000_tx"]) * len(df) / 1000
            * (365 / max(df.horodatage.dt.date.nunique(), 1)), 2),
    }

    LOG.info("--- Resultats du test A/B ---")
    for bras in ("A_champion", "B_challenger"):
        r = resultats[bras]
        LOG.info("   %-14s %-22s detection=%.1f %% | precision=%.1f %% | "
                 "alertes=%.2f %% | cout/1000tx=%.2f $",
                 bras, r["strategie"], 100 * r["taux_detection"],
                 100 * r["precision"], r["taux_alerte_pct"], r["cout_par_1000_tx"])
    for j in justification:
        LOG.info("   %s", j)
    LOG.info("   DECISION : %s", decision)

    dossier = cfg.chemin("artefacts") / "abtest"
    ecrire_json(dossier / f"rapport_{nom_experience}.json", rapport)
    _figure(dossier, nom_experience, resultats, boot)
    df[["transaction_id", "carte_id", "horodatage", "bras", "decision",
        "est_fraude", "montant"]].to_csv(dossier / f"trafic_{nom_experience}.csv", index=False)
    return rapport


def _figure(dossier, nom, resultats, boot):
    a, b = resultats["A_champion"], resultats["B_challenger"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for i, (cle, lib, fmt) in enumerate([
            ("taux_detection", "Taux de detection", 100),
            ("taux_alerte_pct", "Charge d'alertes (%)", 1),
            ("cout_par_1000_tx", "Cout par 1 000 tx ($)", 1)]):
        vals = [a[cle] * fmt, b[cle] * fmt]
        ax[i].bar(["A - champion", "B - challenger"], vals, color=["#8d99ae", "#3a7ca5"])
        ax[i].set_title(lib)
        for j, v in enumerate(vals):
            ax[i].text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(dossier / f"figure_{nom}.png", bbox_inches="tight")
    plt.close()


# =============================================================================
#  Experience par defaut : moteur de regles contre modele
# =============================================================================
def executer(cfg: Config | None = None) -> dict:
    cfg = cfg or charger_config()
    from fraude360.utils.journal import lire_json

    resume = lire_json(cfg.chemin("artefacts") / "modele" / "resume_courant.json")
    seuil = resume["seuil_decision"]
    variables = resume["variables_numeriques"] + resume["variables_categorielles"]

    jeux = donnees.decouper(cfg)
    df = jeux.test.copy()

    modele = registre.charger_par_alias(registre.ALIAS_CHAMPION, cfg)
    moteur = MoteurDeRegles()

    # Le seuil du moteur de regles est cale pour produire une charge d'alertes
    # comparable a celle du modele : sans cela on comparerait deux points de
    # fonctionnement differents et non deux strategies.
    scores_regles = moteur.evaluer(df)["score_regles"]
    seuil_regles = float(np.quantile(scores_regles, 1 - 0.006)) if len(df) else 0.4
    seuil_regles = max(seuil_regles, 0.30)

    champion = Strategie(
        nom="moteur de regles expertes",
        decider=lambda d: (moteur.evaluer(d)["score_regles"] >= seuil_regles).values)
    challenger = Strategie(
        nom=f"modele {resume['modele_retenu']} calibre",
        decider=lambda d: (modele.predict_proba(d[variables])[:, 1] >= seuil).astype(int))

    rapport = comparer(df, champion, challenger, cfg, "regles_vs_modele")

    stats = moteur.statistiques(df)
    stats.to_csv(cfg.chemin("artefacts") / "abtest" / "performance_des_regles.csv", index=False)
    LOG.info("Performance individuelle des regles ecrite (%d regles).", len(stats))
    return rapport


if __name__ == "__main__":
    executer()
