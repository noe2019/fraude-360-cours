"""ETAPE 13 - Triage des alertes de fraude par modele de langage.

Sous-modules :
    politiques  corpus des politiques internes + recherche par similarite (RAG)
    client      acces au modele de langage : Ollama local, sinon repli deterministe
    triage      generation de la note de triage pour chaque alerte bloquee
    evaluation  garde-fous deterministes + LLM-as-judge
"""
