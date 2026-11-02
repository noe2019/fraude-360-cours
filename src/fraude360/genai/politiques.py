"""Corpus des politiques internes et recherche par similarite (le R de RAG).

Pourquoi un corpus et pas un prompt unique : un enqueteur ne juge pas une alerte
dans l'abstrait, il l'evalue contre une politique ecrite. Le modele doit faire
pareil, et sa note doit CITER la politique appliquee. C'est ce qui rend la sortie
verifiable : on peut controler qu'une citation existe, et qu'elle est pertinente.

La recherche utilise TF-IDF et la similarite cosinus (scikit-learn, deja une
dependance du projet). Un index vectoriel dense serait plus fin sur un corpus de
milliers de documents ; sur douze passages, il n'apporterait rien et ajouterait
une dependance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Passage:
    identifiant: str
    titre: str
    texte: str


# --- corpus -----------------------------------------------------------------
# Politiques fictives mais coherentes avec les regles de serving/regles.py :
# chaque passage couvre une typologie ou une obligation operationnelle reelle.
CORPUS: List[Passage] = [
    Passage(
        "POL-01", "Test de carte (card testing)",
        "Une serie de micro-montants sur une meme carte en moins d'une heure indique "
        "une validation de numeros voles. Seuil d'intervention : 4 transactions de "
        "moins de 5 $ en 60 minutes. Action : blocage immediat de la carte et "
        "reemission. Ne pas contacter le membre avant blocage : le fraudeur teste "
        "souvent en parallele. Regle applicable : R02."),
    Passage(
        "POL-02", "Carte perdue ou volee",
        "Une rafale d'achats de proximite ou un deplacement geographique impossible "
        "entre deux transactions (vitesse implicite superieure a 700 km/h) signale une "
        "carte physique compromise. Action : blocage et appel sortant au membre dans "
        "l'heure. Regles applicables : R01, R03."),
    Passage(
        "POL-03", "Achat en ligne sans authentification forte",
        "Une transaction de commerce electronique superieure a 400 $ sans "
        "authentification 3-D Secure fait porter la responsabilite de la perte au "
        "commercant, pas a l'emetteur. L'exposition financiere de la cooperative est "
        "donc limitee. Action : mise en attente pour revue manuelle, pas de blocage "
        "automatique. Regle applicable : R04."),
    Passage(
        "POL-04", "Usurpation de compte",
        "Un appareil inconnu combine a un montant tres superieur aux habitudes de la "
        "carte, en particulier la nuit, indique une prise de controle du compte. "
        "Action : blocage, revocation des sessions actives et reinitialisation des "
        "identifiants. Regles applicables : R05, R06."),
    Passage(
        "POL-05", "Arnaque au virement instantane",
        "Un virement Interac vers un nouveau beneficiaire, d'un montant superieur a "
        "250 $, avec au moins deux virements dans les 24 heures, correspond au schema "
        "de fraude par manipulation du membre. Le membre autorise lui-meme la "
        "transaction : le blocage doit etre accompagne d'un appel de sensibilisation. "
        "Ces cas ne sont pas couverts par la garantie zero responsabilite. "
        "Regle applicable : R11."),
    Passage(
        "POL-06", "Budget quotidien de l'equipe d'enquete",
        "L'equipe traite au maximum 150 alertes par jour. Au-dela, les alertes de "
        "gravite faible sont mises en file et traitees le lendemain. Une alerte de "
        "gravite elevee ne peut jamais etre mise en file. Le seuil de decision du "
        "modele est calibre sous cette contrainte."),
    Passage(
        "POL-07", "Marchand a risque",
        "Un marchand dont le taux de fraude sur 30 jours depasse 2 % declenche une "
        "surveillance renforcee, mais ne justifie pas a lui seul un blocage : le "
        "signal est trop peu specifique. Il augmente la gravite d'une alerte deja "
        "declenchee par un autre motif. Regle applicable : R08."),
    Passage(
        "POL-08", "Faux positifs et friction membre",
        "Un blocage injustifie coute en moyenne 35 $ : appel entrant, reemission de "
        "carte et risque d'attrition. Un blocage doit etre proportionne. En cas de "
        "doute sur une transaction inferieure a 100 $ chez un marchand habituel du "
        "membre, privilegier la mise en attente."),
    Passage(
        "POL-09", "Cumul et velocite",
        "Un cumul superieur a 3 000 $ sur 24 heures, ou l'utilisation de la carte dans "
        "deux pays ou plus dans la meme journee, constitue un signal de velocite. "
        "Seul, il justifie une revue ; combine a un autre signal, il justifie un "
        "blocage. Regles applicables : R07, R09."),
    Passage(
        "POL-10", "Obligation de tracabilite",
        "Toute decision de blocage doit etre documentee : motif, regles declenchees, "
        "score du modele et version du modele. Cette trace est exigee pour le "
        "traitement des contestations et pour les controles de deuxieme ligne."),
    Passage(
        "POL-11", "Escalade a la conformite",
        "Une alerte est escaladee a l'equipe de lutte contre le blanchiment lorsqu'elle "
        "combine un virement sortant, un nouveau beneficiaire et un cumul anormal. Le "
        "soupcon de blanchiment se traite separement de la fraude par carte."),
    Passage(
        "POL-12", "Traitement des scores intermediaires",
        "Une alerte dont le score du modele se situe a moins de 0,05 au-dessus du "
        "seuil est consideree comme limite. Elle est traitee en revue manuelle "
        "prioritaire plutot qu'en blocage automatique, afin de limiter les faux "
        "positifs sur la zone la moins fiable du modele."),
]

IDENTIFIANTS = {p.identifiant for p in CORPUS}


class Index:
    """Index TF-IDF sur le corpus de politiques."""

    def __init__(self, corpus: List[Passage] | None = None):
        self.corpus = corpus or CORPUS
        self._vect = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        self._matrice = self._vect.fit_transform(
            [f"{p.titre}. {p.texte}" for p in self.corpus])

    def rechercher(self, requete: str, k: int = 3) -> List[Tuple[Passage, float]]:
        """Les k passages les plus proches de la requete, avec leur similarite."""
        v = self._vect.transform([requete])
        sims = cosine_similarity(v, self._matrice)[0]
        ordre = sims.argsort()[::-1][:k]
        return [(self.corpus[i], float(sims[i])) for i in ordre if sims[i] > 0.0]
