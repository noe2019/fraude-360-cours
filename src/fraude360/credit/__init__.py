"""Variante CREDIT du prototype : grille de score, IFRS 9, validation.

Le volet fraude repond a la question « cette transaction est-elle frauduleuse
maintenant ? ». Le volet credit repond a « quelle est la probabilite que ce
pret se degrade dans les douze prochains mois, et combien dois-je provisionner
aujourd'hui ? ». Ce ne sont pas les memes metiers, ni les memes contraintes.

Ce que la pratique bancaire impose ici et qui n'existe pas cote fraude :

  GRILLE DE SCORE INTERPRETABLE  Regression logistique sur variables
      discretisees en WOE. Non pas par gout du simple, mais parce qu'un modele
      de credit doit etre explicable a un regulateur, a un verificateur et au
      membre a qui on refuse un pret. Le gradient boosting ferait mieux de
      quelques points de Gini et ne serait pas homologable.

  MONOTONIE  Un decoupage doit avoir un sens metier : plus le ratio
      d'endettement monte, plus le risque monte. Une classe qui casse la
      monotonie est un artefact d'echantillon.

  IFRS 9  Provisionner la perte de credit ATTENDUE, pas la perte subie :
      12 mois pour les prets sains (etape 1), a vie des qu'il y a
      augmentation significative du risque (etape 2), et en defaut (etape 3).

  VALIDATION INDEPENDANTE  Discrimination (Gini, KS), calibration
      (Hosmer-Lemeshow), stabilite (PSI), et retrovalidation dans le temps.

Sous-modules :
    portefeuille  generation d'un portefeuille de prets avec defauts observes
    scorecard     discretisation WOE/IV, regression logistique, mise en points
    ifrs9         classement par etape, PD a 12 mois et a vie, ECL
    validation    Gini, KS, Hosmer-Lemeshow, PSI, retrovalidation
"""
