"""Moteur de regles expertes.

Deux usages dans le projet :
  1. il represente le dispositif EXISTANT avant l'arrivee du modele : c'est lui
     le champion initial du test A/B (etape 07) ;
  2. en production il reste actif EN COMPLEMENT du modele : une regle se deploie
     en quelques heures quand une nouvelle fraude apparait, la ou un modele
     demande un cycle de reentrainement complet (etape 10).

Chaque regle est declarative : identifiant, libelle metier, condition
vectorisee, score de risque et niveau de gravite. Elles sont evaluables sur un
DataFrame complet (traitement par lots) comme sur une transaction unique (API).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import pandas as pd


@dataclass
class Regle:
    identifiant: str
    libelle: str
    condition: Callable[[pd.DataFrame], pd.Series]
    score: float                      # contribution au score de risque (0-1)
    gravite: str = "moyenne"          # faible | moyenne | elevee
    active: bool = True
    typologie_visee: str = ""
    date_activation: str = ""


def regles_de_base() -> List[Regle]:
    """Regles issues de l'expertise des enqueteurs, en place avant le modele."""
    return [
        Regle("R01", "Rafale : 5 transactions ou plus sur la carte en une heure",
              lambda d: d["nb_tx_carte_1h"] >= 5, 0.45, "elevee", "carte_perdue_volee"),
        Regle("R02", "Test de carte : 4 micro-montants ou plus en une heure",
              lambda d: d["nb_micro_tx_carte_1h"] >= 4, 0.55, "elevee", "test_de_carte"),
        Regle("R03", "Deplacement impossible : plus de 700 km/h entre deux achats",
              lambda d: d["vitesse_kmh"] > 700, 0.50, "elevee", "carte_perdue_volee"),
        Regle("R04", "Achat en ligne eleve sans authentification forte",
              lambda d: (d["canal"] == "ECOM") & (~d["authentification_3ds"].astype(bool))
                        & (d["montant"] > 400), 0.40, "moyenne", "achat_en_ligne_cnp"),
        Regle("R05", "Montant nocturne tres superieur aux habitudes de la carte",
              lambda d: (d["est_nuit"] == 1) & (d["ratio_montant_moyenne_carte"] > 8), 0.40,
              "moyenne", "usurpation_de_compte"),
        Regle("R06", "Appareil inconnu du membre sur une transaction elevee",
              lambda d: (d["est_nouvel_appareil_client"] == 1) & (d["montant"] > 500), 0.35,
              "moyenne", "usurpation_de_compte"),
        Regle("R07", "Cumul 24 h de la carte superieur a 3 000 $",
              lambda d: d["montant_somme_carte_24h"] > 3000, 0.30, "moyenne", ""),
        Regle("R08", "Marchand a taux de fraude eleve sur 30 jours",
              lambda d: d["taux_fraude_marchand_30j"] > 0.02, 0.30, "faible", ""),
        Regle("R09", "Carte utilisee dans 2 pays ou plus en 24 h",
              lambda d: d["nb_pays_distincts_carte_24h"] >= 2, 0.35, "moyenne", ""),
        Regle("R10", "Transactions separees de moins de 60 secondes chez des marchands differents",
              lambda d: (d["secondes_depuis_tx_precedente"] < 60)
                        & (d["est_nouveau_marchand_carte"] == 1), 0.30, "moyenne", ""),
    ]


def regle_nouvelle_typologie() -> Regle:
    """Regle ajoutee a chaud a l'etape 10 (arnaque au virement instantane).

    Mitigation immediate : elle couvre le risque pendant les jours nécessaires
    au reentrainement du modele.
    """
    return Regle(
        "R11",
        "Virement Interac vers un nouveau beneficiaire, montant eleve, cumul 24 h anormal",
        lambda d: ((d.get("est_nouveau_beneficiaire", pd.Series(0, index=d.index)) == 1)
                   & (d["canal"] == "INTERAC")
                   & (d["montant"] > 250)
                   & (d.get("nb_virements_client_24h", pd.Series(0, index=d.index)) >= 2)),
        0.60, "elevee", "arnaque_virement_instantane", date_activation="J+1")


class MoteurDeRegles:
    def __init__(self, regles: List[Regle] | None = None):
        self.regles = regles or regles_de_base()

    def ajouter(self, regle: Regle) -> None:
        self.regles = [r for r in self.regles if r.identifiant != regle.identifiant]
        self.regles.append(regle)

    # --- traitement par lots -------------------------------------------------
    def evaluer(self, d: pd.DataFrame) -> pd.DataFrame:
        """Retourne le score de risque et les regles declenchees pour chaque ligne."""
        score = pd.Series(0.0, index=d.index)
        declenchees = pd.Series([[] for _ in range(len(d))], index=d.index)
        detail = {}
        for r in self.regles:
            if not r.active:
                continue
            try:
                m = r.condition(d).fillna(False).astype(bool)
            except KeyError:
                continue
            detail[r.identifiant] = m
            score = score + m.astype(float) * r.score
            declenchees[m] = declenchees[m].map(lambda lst: lst + [r.identifiant])
        score = pd.Series(np.clip(score, 0, 1), index=d.index)
        return pd.DataFrame({"score_regles": score,
                             "regles_declenchees": declenchees.map(lambda l: ",".join(l))})

    def statistiques(self, d: pd.DataFrame, cible: str = "est_fraude") -> pd.DataFrame:
        """Performance individuelle de chaque regle : couverture et precision."""
        lignes = []
        for r in self.regles:
            if not r.active:
                continue
            try:
                m = r.condition(d).fillna(False).astype(bool)
            except KeyError:
                continue
            n = int(m.sum())
            vp = int(d.loc[m, cible].sum()) if n else 0
            lignes.append(dict(
                identifiant=r.identifiant, libelle=r.libelle, gravite=r.gravite,
                typologie_visee=r.typologie_visee,
                transactions_touchees=n,
                taux_declenchement_pct=round(100 * n / max(len(d), 1), 4),
                fraudes_captees=vp,
                precision_pct=round(100 * vp / n, 2) if n else 0.0,
                couverture_fraude_pct=round(100 * vp / max(int(d[cible].sum()), 1), 2)))
        return pd.DataFrame(lignes).sort_values("fraudes_captees", ascending=False)

    # --- transaction unique (API) -------------------------------------------
    def evaluer_une(self, transaction: dict) -> dict:
        d = pd.DataFrame([transaction])
        res = self.evaluer(d).iloc[0]
        return {"score_regles": float(res.score_regles),
                "regles_declenchees": [x for x in res.regles_declenchees.split(",") if x]}
