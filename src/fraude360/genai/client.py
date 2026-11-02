"""Acces au modele de langage, avec repli deterministe.

Deux implementations derriere la meme interface :

  ClientOllama    appelle un modele local via l'API HTTP d'Ollama (127.0.0.1:11434).
                  Aucune donnee ne sort du poste : c'est la contrainte usuelle
                  quand on manipule des transactions de membres.

  ClientReplique  ne fait aucun appel reseau. Il produit une reponse structuree
                  a partir des passages recuperes et des signaux de l'alerte.

Le repli n'est pas un bouchon de test : c'est ce qui rend l'etape 13 executable
sur n'importe quel poste, comme les douze autres. Le rapport indique toujours
quel client a servi, et l'evaluation s'applique aux deux de la meme maniere.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

import requests

from fraude360.utils.journal import obtenir_journal

LOG = obtenir_journal("fraude360.genai.client")

DELAI_S = 60


class ClientLLM(Protocol):
    nom: str

    def completer(self, invite: str) -> str: ...


class ClientOllama:
    """Modele local servi par Ollama."""

    def __init__(self, modele: str = "llama3.1:8b", hote: str = "http://127.0.0.1:11434"):
        self.modele = modele
        self.hote = hote.rstrip("/")
        self.nom = f"ollama:{modele}"

    @classmethod
    def disponible(cls, hote: str = "http://127.0.0.1:11434", modele: str = "") -> bool:
        try:
            r = requests.get(f"{hote.rstrip('/')}/api/tags", timeout=3)
            if r.status_code != 200:
                return False
            if not modele:
                return True
            noms = [m.get("name", "") for m in r.json().get("models", [])]
            return any(n.split(":")[0] == modele.split(":")[0] for n in noms)
        except Exception:
            return False

    def completer(self, invite: str) -> str:
        r = requests.post(
            f"{self.hote}/api/generate",
            json={"model": self.modele, "prompt": invite, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 400}},
            timeout=DELAI_S)
        r.raise_for_status()
        return r.json().get("response", "")


class ClientReplique:
    """Repli deterministe : compose la reponse a partir du contexte fourni.

    Il lit dans l'invite les elements deja structures (identifiants de politique,
    regles declenchees, score) et les recombine au format JSON attendu. Il ne
    devine rien qui ne soit pas dans l'invite -- ce qui, accessoirement, en fait
    une reference utile pour mesurer ce que le vrai modele apporte en plus.
    """

    nom = "repli-deterministe"

    def completer(self, invite: str) -> str:
        if "Note deux criteres" in invite:
            return self._juger(invite)
        return self._trier(invite)

    @staticmethod
    def _juger(invite: str) -> str:
        """Proxy deterministe du juge : ancrage mesure, utilite mesuree.

        Ce n'est pas un jugement semantique -- c'est une borne inferieure
        verifiable. L'ancrage vaut 5 si toute politique citee figure parmi les
        passages fournis, 1 si aucune ; l'utilite depend de la presence d'une
        action et d'une justification substantielle.
        """
        m_dispo = re.search(r"EXTRAITS DE POLITIQUE DISPONIBLES\s*:\s*([^\n]*)", invite)
        m_cite = re.search(r"politiques citees\s*:\s*([^\n]*)", invite)
        m_just = re.search(r"justification\s*:\s*([^\n]*)", invite)
        m_act = re.search(r"action\s*:\s*([A-Z_]+)", invite)
        dispo = set(re.findall(r"POL-\d{2}", m_dispo.group(1) if m_dispo else ""))
        cites = set(re.findall(r"POL-\d{2}", m_cite.group(1) if m_cite else ""))
        if not cites:
            ancrage = 1
        elif cites <= dispo:
            ancrage = 5
        else:
            ancrage = 2
        just = (m_just.group(1) if m_just else "").strip()
        utilite = 1 + (2 if m_act else 0) + (2 if len(just) > 80 else 1 if just else 0)
        return json.dumps({"ancrage": ancrage, "utilite": min(utilite, 5)})

    @staticmethod
    def _trier(invite: str) -> str:
        politiques = re.findall(r"\b(POL-\d{2})\b", invite)
        regles = re.findall(r"\b(R\d{2})\b", invite)
        m_score = re.search(r"score du modele\s*:\s*([0-9.]+)", invite)
        m_seuil = re.search(r"seuil de blocage\s*:\s*([0-9.]+)", invite)
        m_montant = re.search(r"montant\s*:\s*([0-9.]+)", invite)
        m_typo = re.search(r"Typologie visee\s*:\s*([a-z_]+)", invite)
        score = float(m_score.group(1)) if m_score else 0.0
        seuil = float(m_seuil.group(1)) if m_seuil else 0.5
        montant = float(m_montant.group(1)) if m_montant else 0.0

        principale = politiques[0] if politiques else "POL-10"
        typologie = m_typo.group(1) if m_typo else "indeterminee"
        marge = score - seuil
        # Le score pilote l'action ; POL-12 met en revue la zone limite
        # (moins de 0,05 au-dessus du seuil).
        if marge < 0:
            action, confiance = "MISE_EN_ATTENTE", "faible"
        elif marge < 0.05:
            action, confiance = "REVUE_MANUELLE", "moyenne"
        else:
            action, confiance = "BLOQUER", "elevee" if marge > 0.25 else "moyenne"

        # POL-06 : une alerte de gravite elevee ne peut jamais etre mise en file.
        # Une regle grave releve donc un PLANCHER, elle ne court-circuite pas le
        # score -- sans quoi toute alerte portant R02 ou R11 serait bloquee, ce
        # que l'evaluation contre la verite terrain rend immediatement visible.
        if ("R02" in regles or "R11" in regles) and action == "MISE_EN_ATTENTE":
            action, confiance = "REVUE_MANUELLE", "moyenne"

        justification = (
            f"Score du modele a {score:.2f} et regles {', '.join(regles) or 'aucune'} "
            f"declenchees sur un montant de {montant:.2f} $. La politique {principale} "
            f"couvre ce schema et prescrit l'action retenue.")
        return json.dumps({"typologie": typologie, "action": action,
                           "confiance": confiance, "justification": justification,
                           "politiques_citees": politiques[:2] or [principale]},
                          ensure_ascii=False)


def obtenir_client(cfg_genai: dict) -> ClientLLM:
    """Choisit Ollama s'il repond, sinon le repli. Ne leve jamais."""
    if cfg_genai.get("activer_ollama", True):
        modele = cfg_genai.get("modele", "llama3.1:8b")
        hote = cfg_genai.get("hote_ollama", "http://127.0.0.1:11434")
        if ClientOllama.disponible(hote, modele):
            LOG.info("Modele de langage : Ollama (%s)", modele)
            return ClientOllama(modele, hote)
        LOG.warning("Ollama indisponible sur %s -> repli deterministe", hote)
    return ClientReplique()
