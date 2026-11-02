"""ETAPE 00 - Creation des donnees.

Simulateur de transactions de paiement d'une cooperative financiere quebecoise.
Produit :
  - trois referentiels (clients, cartes, marchands) ;
  - un flux transactionnel depose en fichiers journaliers dans data/00_landing/ ;
  - des defauts de qualite volontaires (doublons, nuls, formats mixtes) pour que
    les etapes d'ingestion et de nettoyage aient une raison d'exister ;
  - quatre typologies de fraude realistes, plus une cinquieme injectable
    apres la mise en production (voir fraude360.newfraud).

Le simulateur est deterministe (seed dans conf/config.yaml).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from fraude360.config import Config, charger_config
from fraude360.utils.journal import chrono, obtenir_journal, titre

LOG = obtenir_journal("fraude360.generation")

# --- Geographie : principales villes desservies -------------------------------
VILLES = [
    ("Montreal", "QC", 45.508, -73.561, 0.34),
    ("Quebec", "QC", 46.813, -71.208, 0.14),
    ("Laval", "QC", 45.606, -73.712, 0.09),
    ("Gatineau", "QC", 45.477, -75.701, 0.07),
    ("Longueuil", "QC", 45.531, -73.518, 0.07),
    ("Sherbrooke", "QC", 45.400, -71.899, 0.05),
    ("Trois-Rivieres", "QC", 46.343, -72.542, 0.04),
    ("Saguenay", "QC", 48.428, -71.068, 0.04),
    ("Levis", "QC", 46.803, -71.177, 0.04),
    ("Terrebonne", "QC", 45.700, -73.646, 0.03),
    ("Ottawa", "ON", 45.421, -75.697, 0.03),
    ("Toronto", "ON", 43.653, -79.383, 0.03),
    ("Rimouski", "QC", 48.449, -68.523, 0.02),
    ("Rouyn-Noranda", "QC", 48.236, -79.023, 0.01),
]

# --- Categories de marchands (inspirees des codes MCC) ------------------------
CATEGORIES = {
    "epicerie":            dict(poids=0.20, mu=3.6, sigma=0.55, risque=0.4, ecom=0.05),
    "restaurant":          dict(poids=0.15, mu=3.1, sigma=0.60, risque=0.5, ecom=0.15),
    "essence":             dict(poids=0.11, mu=3.9, sigma=0.40, risque=0.9, ecom=0.02),
    "pharmacie":           dict(poids=0.07, mu=3.2, sigma=0.60, risque=0.3, ecom=0.10),
    "depanneur":           dict(poids=0.07, mu=2.6, sigma=0.55, risque=1.1, ecom=0.02),
    "vetement":            dict(poids=0.07, mu=4.1, sigma=0.70, risque=1.0, ecom=0.35),
    "electronique":        dict(poids=0.05, mu=5.2, sigma=0.85, risque=2.6, ecom=0.55),
    "ameublement":         dict(poids=0.04, mu=5.3, sigma=0.80, risque=1.3, ecom=0.30),
    "voyage":              dict(poids=0.04, mu=5.9, sigma=0.75, risque=2.4, ecom=0.85),
    "jeux_en_ligne":       dict(poids=0.03, mu=3.7, sigma=0.90, risque=3.4, ecom=0.98),
    "transport":           dict(poids=0.05, mu=3.0, sigma=0.60, risque=0.6, ecom=0.45),
    "sante":               dict(poids=0.04, mu=4.4, sigma=0.70, risque=0.3, ecom=0.15),
    "loisirs":             dict(poids=0.04, mu=3.8, sigma=0.75, risque=0.8, ecom=0.40),
    "services_financiers": dict(poids=0.02, mu=5.0, sigma=0.90, risque=1.8, ecom=0.60),
    "telecom":             dict(poids=0.02, mu=4.3, sigma=0.45, risque=0.5, ecom=0.70),
}

CANAUX = ["POS", "ECOM", "GAB", "INTERAC"]
MODES_SAISIE = ["puce", "sans_contact", "bande", "manuel", "ecom"]

TYPOLOGIES = ["carte_perdue_volee", "achat_en_ligne_cnp", "usurpation_de_compte", "test_de_carte"]


# =============================================================================
#  Outils
# =============================================================================
def _distance_km(lat1, lon1, lat2, lon2):
    """Distance grand-cercle vectorisee (km)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class Referentiels:
    clients: pd.DataFrame
    cartes: pd.DataFrame
    marchands: pd.DataFrame


# =============================================================================
#  Referentiels
# =============================================================================
def construire_referentiels(cfg: Config, rng: np.random.Generator) -> Referentiels:
    g = cfg["generation"]
    n_cli, n_car, n_mar = g["nb_clients"], g["nb_cartes"], g["nb_marchands"]
    date_debut = pd.Timestamp(g["date_debut"])

    # ---- clients ------------------------------------------------------------
    poids_villes = np.array([v[4] for v in VILLES], dtype=float)
    poids_villes /= poids_villes.sum()
    idx_ville = rng.choice(len(VILLES), size=n_cli, p=poids_villes)
    clients = pd.DataFrame({
        "client_id": [f"CLI{i:07d}" for i in range(n_cli)],
        "age": np.clip(rng.normal(45, 16, n_cli), 18, 92).astype(int),
        "ville": [VILLES[i][0] for i in idx_ville],
        "province": [VILLES[i][1] for i in idx_ville],
        "latitude_domicile": [VILLES[i][2] for i in idx_ville] + rng.normal(0, 0.05, n_cli),
        "longitude_domicile": [VILLES[i][3] for i in idx_ville] + rng.normal(0, 0.05, n_cli),
        "anciennete_compte_jours": rng.integers(20, 9000, n_cli),
        "segment": rng.choice(["particulier", "jeunesse", "aine", "affaires"],
                              n_cli, p=[0.62, 0.14, 0.16, 0.08]),
        "revenu_annuel": np.round(np.clip(rng.lognormal(11.0, 0.45, n_cli), 15000, 400000), -2),
    })
    clients["date_ouverture_compte"] = date_debut - pd.to_timedelta(
        clients["anciennete_compte_jours"], unit="D")

    # ---- cartes -------------------------------------------------------------
    proprietaires = rng.choice(clients["client_id"].to_numpy(), size=n_car)
    cartes = pd.DataFrame({
        "carte_id": [f"CAR{i:07d}" for i in range(n_car)],
        "client_id": proprietaires,
        "type_carte": rng.choice(["debit", "credit"], n_car, p=[0.55, 0.45]),
        "reseau": rng.choice(["visa", "mastercard", "interac"], n_car, p=[0.45, 0.35, 0.20]),
        "limite": np.round(np.clip(rng.lognormal(8.6, 0.55, n_car), 500, 50000), -2),
        "intensite": np.clip(rng.gamma(2.4, 0.55, n_car), 0.15, 9.0),  # tx/jour moyennes
    })

    # ---- marchands ----------------------------------------------------------
    cats = list(CATEGORIES)
    poids_cats = np.array([CATEGORIES[c]["poids"] for c in cats], dtype=float)
    poids_cats /= poids_cats.sum()
    cat_mar = rng.choice(cats, size=n_mar, p=poids_cats)
    idx_ville_m = rng.choice(len(VILLES), size=n_mar, p=poids_villes)
    marchands = pd.DataFrame({
        "marchand_id": [f"MAR{i:06d}" for i in range(n_mar)],
        "categorie": cat_mar,
        "ville": [VILLES[i][0] for i in idx_ville_m],
        "pays": np.where(rng.random(n_mar) < 0.06, "US", "CA"),
        "latitude": [VILLES[i][2] for i in idx_ville_m] + rng.normal(0, 0.08, n_mar),
        "longitude": [VILLES[i][3] for i in idx_ville_m] + rng.normal(0, 0.08, n_mar),
        # popularite tres inegale : quelques grandes enseignes, beaucoup de petits
        "popularite": rng.pareto(1.4, n_mar) + 0.15,
    })
    marchands["indice_risque"] = [CATEGORIES[c]["risque"] for c in cat_mar]
    return Referentiels(clients, cartes, marchands)


# =============================================================================
#  Transactions legitimes
# =============================================================================
def generer_legitimes(cfg: Config, ref: Referentiels, rng: np.random.Generator,
                      n_tx: int, jour_debut: int, nb_jours: int) -> pd.DataFrame:
    g = cfg["generation"]
    t0 = pd.Timestamp(g["date_debut"]) + pd.Timedelta(days=jour_debut)

    cartes = ref.cartes
    p_carte = cartes["intensite"].to_numpy()
    p_carte = p_carte / p_carte.sum()
    idx_carte = rng.choice(len(cartes), size=n_tx, p=p_carte)

    # --- horodatage : saisonnalite horaire + hebdomadaire --------------------
    jours = rng.random(n_tx) * nb_jours
    # profil horaire realiste (creux la nuit, pointes midi et 17-19 h)
    profil = np.array([0.6, 0.35, 0.22, 0.18, 0.2, 0.35, 0.9, 1.8, 2.4, 2.6, 2.9, 3.6,
                       4.1, 3.7, 3.4, 3.5, 4.0, 4.6, 4.2, 3.3, 2.6, 2.0, 1.4, 0.9])
    profil = profil / profil.sum()
    heures = rng.choice(24, size=n_tx, p=profil)
    minutes = rng.integers(0, 60, n_tx)
    secondes = rng.integers(0, 60, n_tx)
    horodatage = (t0 + pd.to_timedelta(jours.astype(int), unit="D")
                  + pd.to_timedelta(heures, unit="h")
                  + pd.to_timedelta(minutes, unit="m")
                  + pd.to_timedelta(secondes, unit="s"))

    # --- marchand : melange "habitudes" (marchands proches) et hasard --------
    p_mar = ref.marchands["popularite"].to_numpy()
    p_mar = p_mar / p_mar.sum()
    idx_mar = rng.choice(len(ref.marchands), size=n_tx, p=p_mar)

    cat = ref.marchands["categorie"].to_numpy()[idx_mar]
    mu = np.array([CATEGORIES[c]["mu"] for c in cat])
    sg = np.array([CATEGORIES[c]["sigma"] for c in cat])
    p_ecom = np.array([CATEGORIES[c]["ecom"] for c in cat])

    montant = np.round(np.exp(rng.normal(mu, sg)), 2)
    montant = np.clip(montant, 1.0, 9000.0)

    est_ecom = rng.random(n_tx) < p_ecom
    canal = np.where(est_ecom, "ECOM", "POS")
    # une part de retraits au guichet et de virements Interac
    tirage = rng.random(n_tx)
    canal = np.where((~est_ecom) & (tirage < 0.05), "GAB", canal)
    canal = np.where((~est_ecom) & (tirage >= 0.05) & (tirage < 0.11), "INTERAC", canal)

    mode = np.where(canal == "ECOM", "ecom",
                    np.where(rng.random(n_tx) < 0.62, "sans_contact",
                             np.where(rng.random(n_tx) < 0.9, "puce", "bande")))

    client_id = cartes["client_id"].to_numpy()[idx_carte]
    cli_idx = ref.clients.set_index("client_id").index.get_indexer(client_id)
    lat_dom = ref.clients["latitude_domicile"].to_numpy()[cli_idx]
    lon_dom = ref.clients["longitude_domicile"].to_numpy()[cli_idx]

    lat = np.where(est_ecom, lat_dom, ref.marchands["latitude"].to_numpy()[idx_mar])
    lon = np.where(est_ecom, lon_dom, ref.marchands["longitude"].to_numpy()[idx_mar])
    # 92 % des achats physiques se font pres du domicile : on rapproche le marchand
    proche = rng.random(n_tx) < 0.92
    lat = np.where(proche & (~est_ecom), lat_dom + rng.normal(0, 0.06, n_tx), lat)
    lon = np.where(proche & (~est_ecom), lon_dom + rng.normal(0, 0.06, n_tx), lon)

    df = pd.DataFrame({
        "horodatage": horodatage,
        "carte_id": cartes["carte_id"].to_numpy()[idx_carte],
        "client_id": client_id,
        "marchand_id": ref.marchands["marchand_id"].to_numpy()[idx_mar],
        "categorie_marchand": cat,
        "canal": canal,
        "mode_saisie": mode,
        "montant": montant,
        "devise": "CAD",
        "pays": ref.marchands["pays"].to_numpy()[idx_mar],
        "latitude": np.round(lat, 5),
        "longitude": np.round(lon, 5),
        "authentification_3ds": np.where(canal == "ECOM", rng.random(n_tx) < 0.72, True),
        "appareil_id": [f"APP{v:07d}" for v in (idx_carte * 7 % 900000)],
        "adresse_ip": np.where(canal == "ECOM",
                               [f"24.{a}.{b}.{c}" for a, b, c in
                                zip(rng.integers(0, 255, n_tx), rng.integers(0, 255, n_tx),
                                    rng.integers(0, 255, n_tx))], ""),
        "duree_session_s": np.where(canal == "ECOM", rng.integers(30, 1800, n_tx), 0),
        "est_fraude": 0,
        "typologie_fraude": "",
    })
    return df


# =============================================================================
#  Episodes de fraude
# =============================================================================
def generer_fraudes(cfg: Config, ref: Referentiels, rng: np.random.Generator,
                    n_fraudes: int, jour_debut: int, nb_jours: int,
                    poids_typologies: Dict[str, float]) -> pd.DataFrame:
    g = cfg["generation"]
    t0 = pd.Timestamp(g["date_debut"]) + pd.Timedelta(days=jour_debut)

    types = list(poids_typologies)
    poids = np.array([poids_typologies[t] for t in types], dtype=float)
    poids /= poids.sum()

    marchands = ref.marchands
    clients_idx = ref.clients.set_index("client_id")
    lignes: List[dict] = []
    restant = n_fraudes
    garde = 0

    while restant > 0 and garde < n_fraudes * 5:
        garde += 1
        typologie = rng.choice(types, p=poids)
        i_carte = rng.integers(0, len(ref.cartes))
        carte = ref.cartes.iloc[i_carte]
        cli = clients_idx.loc[carte["client_id"]]
        depart = t0 + pd.Timedelta(seconds=int(rng.random() * nb_jours * 86400))

        if typologie == "carte_perdue_volee":
            n = int(rng.integers(3, 9))
            cats_cibles = ["essence", "depanneur", "electronique", "vetement", "epicerie"]
            sel = marchands[marchands["categorie"].isin(cats_cibles)]
            choix = sel.sample(n, replace=True, random_state=int(rng.integers(1e9)))
            deltas = np.cumsum(rng.integers(120, 1500, n))
            montants = np.round(np.clip(np.exp(rng.normal(4.2, 0.7, n)), 15, 900), 2)
            for k in range(n):
                m = choix.iloc[k]
                lignes.append(dict(
                    horodatage=depart + pd.Timedelta(seconds=int(deltas[k])),
                    carte_id=carte["carte_id"], client_id=carte["client_id"],
                    marchand_id=m["marchand_id"], categorie_marchand=m["categorie"],
                    canal="POS",
                    mode_saisie=str(rng.choice(["sans_contact", "puce", "bande"],
                                               p=[0.55, 0.35, 0.10])),
                    montant=float(montants[k]), devise="CAD", pays=m["pays"],
                    latitude=round(float(m["latitude"]), 5),
                    longitude=round(float(m["longitude"]), 5),
                    authentification_3ds=True, appareil_id="",
                    adresse_ip="", duree_session_s=0,
                    est_fraude=1, typologie_fraude=typologie))

        elif typologie == "achat_en_ligne_cnp":
            n = int(rng.integers(1, 5))
            sel = marchands[marchands["categorie"].isin(
                ["electronique", "jeux_en_ligne", "voyage", "vetement", "services_financiers"])]
            choix = sel.sample(n, replace=True, random_state=int(rng.integers(1e9)))
            deltas = np.cumsum(rng.integers(300, 40000, n))
            montants = np.round(np.clip(np.exp(rng.normal(5.4, 0.8, n)), 40, 3000), 2)
            ip = f"185.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(0,255)}"
            # Recouvrement volontaire avec le comportement legitime : dans une
            # partie des cas le fraudeur passe l'authentification forte et opere
            # depuis un appareil deja connu (session detournee). Sans ce
            # recouvrement, la fraude serait separable de facon irrealiste.
            appareil = (f"APP{(int(carte['carte_id'][3:]) * 7 % 900000):07d}"
                        if rng.random() < 0.30 else f"APP{rng.integers(900000, 999999)}")
            trois_ds = bool(rng.random() < 0.35)
            for k in range(n):
                m = choix.iloc[k]
                lignes.append(dict(
                    horodatage=depart + pd.Timedelta(seconds=int(deltas[k])),
                    carte_id=carte["carte_id"], client_id=carte["client_id"],
                    marchand_id=m["marchand_id"], categorie_marchand=m["categorie"],
                    canal="ECOM", mode_saisie="ecom", montant=float(montants[k]),
                    devise="CAD", pays=m["pays"],
                    latitude=round(float(cli["latitude_domicile"]), 5),
                    longitude=round(float(cli["longitude_domicile"]), 5),
                    authentification_3ds=trois_ds, appareil_id=appareil,
                    adresse_ip=ip, duree_session_s=int(rng.integers(30, 900)),
                    est_fraude=1, typologie_fraude=typologie))

        elif typologie == "usurpation_de_compte":
            n = int(rng.integers(1, 4))
            deltas = np.cumsum(rng.integers(600, 9000, n))
            montants = np.round(np.clip(rng.normal(1400, 700, n), 300, 3000), 2)
            ip = f"91.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(0,255)}"
            appareil = (f"APP{(int(carte['carte_id'][3:]) * 7 % 900000):07d}"
                        if rng.random() < 0.25 else f"APP{rng.integers(900000, 999999)}")
            # Une usurpation sur trois se produit en pleine journee.
            heure = int(rng.integers(1, 5)) if rng.random() < 0.66 else int(rng.integers(9, 21))
            nuit = depart.normalize() + pd.Timedelta(hours=heure)
            # Le beneficiaire est un compte ordinaire : aucune valeur "marqueur"
            # ne doit trahir la fraude, sinon le modele apprendrait un artefact.
            benefs = marchands.sample(n, replace=True, random_state=int(rng.integers(1e9)))
            for k in range(n):
                b = benefs.iloc[k]
                lignes.append(dict(
                    horodatage=nuit + pd.Timedelta(seconds=int(deltas[k])),
                    carte_id=carte["carte_id"], client_id=carte["client_id"],
                    marchand_id=b["marchand_id"], categorie_marchand=b["categorie"],
                    canal="INTERAC", mode_saisie="ecom", montant=float(montants[k]),
                    devise="CAD", pays="CA",
                    latitude=round(float(cli["latitude_domicile"]) + 0.9, 5),
                    longitude=round(float(cli["longitude_domicile"]) - 1.4, 5),
                    authentification_3ds=False, appareil_id=appareil, adresse_ip=ip,
                    duree_session_s=int(rng.integers(60, 600)),
                    est_fraude=1, typologie_fraude=typologie))

        else:  # test_de_carte
            n = int(rng.integers(6, 20))
            m = marchands[marchands["categorie"].isin(["jeux_en_ligne", "services_financiers"])]\
                .sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            deltas = np.cumsum(rng.integers(8, 90, n))
            montants = np.round(rng.uniform(0.5, 5.0, n), 2)
            ip = f"45.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(0,255)}"
            for k in range(n):
                lignes.append(dict(
                    horodatage=depart + pd.Timedelta(seconds=int(deltas[k])),
                    carte_id=carte["carte_id"], client_id=carte["client_id"],
                    marchand_id=m["marchand_id"], categorie_marchand=m["categorie"],
                    canal="ECOM", mode_saisie="ecom", montant=float(montants[k]),
                    devise="CAD", pays=m["pays"],
                    latitude=round(float(m["latitude"]), 5),
                    longitude=round(float(m["longitude"]), 5),
                    authentification_3ds=False,
                    appareil_id=f"APP{rng.integers(900000, 999999)}", adresse_ip=ip,
                    duree_session_s=int(rng.integers(5, 60)),
                    est_fraude=1, typologie_fraude=typologie))

        restant = n_fraudes - len(lignes)

    df = pd.DataFrame(lignes[:max(n_fraudes, 0)])
    return df


# =============================================================================
#  Bruit d'etiquetage : la verite terrain n'est jamais parfaite
# =============================================================================
def bruiter_etiquettes(df: pd.DataFrame, rng: np.random.Generator,
                       taux_faux_signalements: float = 0.0008,
                       taux_fraudes_non_signalees: float = 0.06) -> pd.DataFrame:
    """Reproduit les deux imperfections reelles de la verite terrain.

    1. Des membres contestent des transactions parfaitement legitimes (achat
       oublie, achat d'un conjoint) : elles arrivent etiquetees comme fraude.
    2. Une partie des fraudes n'est jamais contestee - petits montants, membres
       inattentifs - et reste donc etiquetee comme legitime.

    Sans ce bruit, un modele entraine sur des donnees simulees atteint des
    performances qu'aucun systeme reel n'obtient, et la demonstration perd toute
    credibilite.
    """
    df = df.copy()
    legit = df.index[df.est_fraude == 0]
    faux = rng.choice(legit, size=int(len(legit) * taux_faux_signalements), replace=False)
    df.loc[faux, "est_fraude"] = 1
    df.loc[faux, "typologie_fraude"] = "contestation_non_fondee"

    fraudes = df.index[(df.est_fraude == 1) & (df.typologie_fraude != "contestation_non_fondee")]
    manquees = rng.choice(fraudes, size=int(len(fraudes) * taux_fraudes_non_signalees),
                          replace=False)
    df.loc[manquees, "est_fraude"] = 0
    df.loc[manquees, "typologie_fraude"] = ""
    return df


# =============================================================================
#  Degradation volontaire de la qualite (pour justifier l'etape de nettoyage)
# =============================================================================
def degrader(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df = df.copy()

    # 1) doublons techniques (rejeu de fichier par le systeme source)
    n_dup = int(n * 0.003)
    if n_dup:
        df = pd.concat([df, df.sample(n_dup, random_state=1)], ignore_index=True)

    # 2) categorie de marchand manquante
    m = rng.random(len(df)) < 0.012
    df.loc[m, "categorie_marchand"] = None

    # 3) montants ecrits avec une virgule decimale (systeme source francophone)
    m = rng.random(len(df)) < 0.02
    df["montant"] = df["montant"].astype(object)
    df.loc[m, "montant"] = df.loc[m, "montant"].map(lambda v: str(v).replace(".", ","))

    # 4) quelques montants aberrants et devises inconnues
    m = rng.random(len(df)) < 0.0015
    df.loc[m, "montant"] = -1 * rng.uniform(1, 500, m.sum()).round(2)
    m = rng.random(len(df)) < 0.001
    df.loc[m, "devise"] = "XXX"

    # 5) espaces parasites dans les identifiants
    m = rng.random(len(df)) < 0.01
    df.loc[m, "carte_id"] = df.loc[m, "carte_id"].map(lambda v: f"  {v} ")

    return df


# =============================================================================
#  Point d'entree
# =============================================================================
def executer(cfg: Config | None = None, jour_debut: int = 0, nb_jours: int | None = None,
             n_tx: int | None = None, n_fraudes: int | None = None,
             poids_typologies: Dict[str, float] | None = None,
             reinitialiser: bool = True) -> Path:
    cfg = cfg or charger_config()
    g = cfg["generation"]
    rng = np.random.default_rng(cfg.seed + jour_debut)

    nb_jours = nb_jours or g["nb_jours"]
    n_tx = n_tx or g["nb_transactions"]
    n_fraudes = n_fraudes if n_fraudes is not None else int(n_tx * g["taux_fraude_cible"])
    poids_typologies = poids_typologies or g["typologies"]

    titre(f"ETAPE 00 - Creation des donnees (jours {jour_debut} a {jour_debut + nb_jours - 1})")
    dossier = cfg.chemin("atterrissage")
    ref_dir = cfg.chemin("racine_donnees") / "referentiels"
    ref_dir.mkdir(parents=True, exist_ok=True)

    if reinitialiser:
        for f in dossier.glob("*.csv"):
            f.unlink()

    with chrono(LOG, "Construction des referentiels"):
        ref = construire_referentiels(cfg, rng)
        ref.clients.to_csv(ref_dir / "clients.csv", index=False)
        ref.cartes.to_csv(ref_dir / "cartes.csv", index=False)
        ref.marchands.to_csv(ref_dir / "marchands.csv", index=False)
    LOG.info("Referentiels : %d clients, %d cartes, %d marchands",
             len(ref.clients), len(ref.cartes), len(ref.marchands))

    with chrono(LOG, f"Generation de {n_tx:,} transactions legitimes"):
        legit = generer_legitimes(cfg, ref, rng, n_tx - n_fraudes, jour_debut, nb_jours)

    with chrono(LOG, f"Generation de ~{n_fraudes:,} transactions frauduleuses"):
        fraude = generer_fraudes(cfg, ref, rng, n_fraudes, jour_debut, nb_jours,
                                 poids_typologies)

    df = pd.concat([legit, fraude], ignore_index=True)
    df = df.sort_values("horodatage").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TX{jour_debut:03d}{i:09d}" for i in range(len(df))])

    # Delai de labellisation : la fraude n'est confirmee qu'apres contestation
    delai = int(g["delai_labellisation_jours"])
    df["horodatage_label"] = pd.NaT
    m = df["est_fraude"] == 1
    df.loc[m, "horodatage_label"] = df.loc[m, "horodatage"] + pd.to_timedelta(
        rng.integers(1, delai * 2, m.sum()), unit="D")

    df = bruiter_etiquettes(df, rng)
    df = degrader(df, rng)
    df = df.sort_values("horodatage").reset_index(drop=True)

    with chrono(LOG, "Ecriture des fichiers journaliers dans la zone d'atterrissage"):
        df["_jour"] = pd.to_datetime(df["horodatage"]).dt.date
        for jour, bloc in df.groupby("_jour"):
            bloc.drop(columns=["_jour"]).to_csv(
                dossier / f"transactions_{jour}.csv", index=False, encoding="utf-8")

    n_fichiers = len(list(dossier.glob("*.csv")))
    LOG.info("%d transactions ecrites dans %d fichiers journaliers -> %s",
             len(df), n_fichiers, dossier)
    LOG.info("Taux de fraude reel : %.3f %%", 100 * df["est_fraude"].mean())
    LOG.info("Repartition des typologies :\n%s",
             df.loc[df.est_fraude == 1, "typologie_fraude"].value_counts().to_string())
    return dossier


if __name__ == "__main__":
    executer()
