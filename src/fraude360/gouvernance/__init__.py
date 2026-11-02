"""Gouvernance de modele : fiche de modele, equite, pseudonymisation.

Ce que l'industrie exige et qu'un prototype oublie presque toujours :

  carte_modele  document de modele (lignee, donnees, metriques, limites,
                usage prevu et usages proscrits) -- attendu par la ligne de
                defense 2 et par la ligne directrice E-23 du BSIF.
  equite        analyse d'impact disparate par sous-groupe : une decision qui
                bloque la transaction d'un membre doit pouvoir etre defendue.
  pseudonymes   les identifiants de carte et de membre n'ont aucune raison de
                circuler en clair dans les artefacts d'analyse.
"""
