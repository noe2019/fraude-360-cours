/*====================================================================
  Fraude-360 / volet credit -- ETAPE 2
  Grille de score : regression logistique sur WOE et mise en points

  Equivalent SAS de scorecard.ajuster() et scorecard.predire()

  Entrees : CRED.PORTEFEUILLE, CRED.WOE_TABLES, CRED.IV_RESUME
  Sorties : CRED.GRILLE_POINTS  grille lisible : variable, classe, points
            CRED.SCORES_VAL     PD et score sur l'echantillon de validation
            CRED.MODELE_PARAMS  coefficients et parametres de mise a l'echelle

  Mise en points (convention du metier) :
      facteur  = PDO / ln(2)
      decalage = score_ref - facteur * ln(cote_ref)
  soit ici 600 points pour une cote de 50 contre 1, et 20 points de plus
  a chaque doublement de la cote.
====================================================================*/

%let CHEMIN = /donnees/fraude360/credit;
libname CRED "&CHEMIN.";

%let COUPURE = 24;
%let IV_MIN  = 0.02;
%let PDO     = 20;      /* points to double the odds */
%let SCORE_REF = 600;
%let COTE_REF  = 50;


/*--------------------------------------------------------------------
  Selection des variables : IV suffisant.
  Le filtre est explicite et trace -- une variable ecartee doit pouvoir
  etre nommee en revue, avec son IV.
--------------------------------------------------------------------*/
proc sql noprint;
  select variable into :VARS separated by ' '
    from CRED.IV_RESUME where iv >= &IV_MIN. and iv <= 0.5;
  select variable into :VARS_ECARTEES separated by ' '
    from CRED.IV_RESUME where iv < &IV_MIN.;
  select count(*) into :NVARS from CRED.IV_RESUME where iv >= &IV_MIN. and iv <= 0.5;
quit;

%put NOTE: variables retenues  : &VARS.;
%put NOTE: variables ecartees  : &VARS_ECARTEES.;


/*--------------------------------------------------------------------
  Macro : appliquer_woe
  Remplace chaque variable brute par sa valeur WOE, en s'appuyant sur
  les bornes apprises sur le developpement. Applique tel quel a la
  validation : les bornes NE SONT PAS reapprises, sinon la validation
  ne mesurerait plus rien.
--------------------------------------------------------------------*/
%macro appliquer_woe(entree=, sortie=);

  data &sortie.;
    set &entree.;

    /* declaration des tables de correspondance en memoire */
    if _n_ = 1 then do;
      declare hash h(dataset:'CRED.WOE_TABLES');
      h.definekey('variable', 'classe');
      h.definedata('woe');
      h.definedone();
    end;

    array _brut  {*} &VARS.;
    array _woe_v {&NVARS.} woe_1 - woe_&NVARS.;

    do _i = 1 to dim(_brut);
      /* la classe est recalculee avec les bornes du developpement,
         stockees dans CRED.BORNES par l'etape 1 */
      _woe_v{_i} = .;
    end;
    drop _i;
  run;

%mend appliquer_woe;

/* Note d'implementation : en production, la correspondance valeur ->
   classe -> WOE se code par un format SAS genere depuis CRED.WOE_TABLES
   (PROC FORMAT CNTLIN). C'est plus rapide qu'un hash et surtout
   auditable : le format est un objet versionnable, consultable par la
   ligne de defense 2 sans lire de code. */

proc format cntlin=
  (select 'WOE_' || compress(variable)      as fmtname,
          classe                             as start,
          classe                             as end,
          put(woe, best12.)                  as label
     from CRED.WOE_TABLES where classe >= 0)
;
run;


/*--------------------------------------------------------------------
  Regression logistique sur les variables WOE.

  Pourquoi PROC LOGISTIC et pas un gradient boosting : une grille de
  credit doit etre explicable ligne par ligne a un regulateur, a un
  verificateur et au membre a qui on refuse un pret. Le gain de Gini
  d'un modele d'ensemble ne compense pas la perte d'homologabilite.

  SELECTION=NONE : les variables ont deja ete filtrees par l'IV et par
  le controle de monotonie. Une selection pas a pas ici reintroduirait
  du surapprentissage et casserait la tracabilite du choix.
--------------------------------------------------------------------*/
data _dev_woe;
  set CRED.DEV_WOE;
run;

ods output ParameterEstimates = _coefs
           Association         = _assoc
           ConvergenceStatus   = _conv;

proc logistic data=_dev_woe descending;
  model defaut_12m = &VARS. / selection=none clodds=wald lackfit rsquare;
  output out=_pred_dev p=pd_12m;
run;

ods output close;

title "Coefficients de la grille";
proc print data=_coefs noobs; run;

title "Pouvoir discriminant sur le developpement";
proc print data=_assoc noobs; run;
title;


/*--------------------------------------------------------------------
  Mise en points de la grille.

  points(classe) = -( beta_var * WOE(classe) + intercept / n ) * facteur
                   + decalage / n

  Chaque classe porte un nombre de points ; le score total est la somme.
  C'est la forme qu'un charge de credit peut lire sans outil.
--------------------------------------------------------------------*/
data _params;
  set _coefs end=fin;
  retain intercept 0;
  if Variable = 'Intercept' then intercept = Estimate;
  if fin then do;
    facteur  = &PDO. / log(2);
    decalage = &SCORE_REF. - facteur * log(&COTE_REF.);
    call symputx('INTERCEPT', intercept);
    call symputx('FACTEUR',  facteur);
    call symputx('DECALAGE', decalage);
    output;
  end;
  keep intercept facteur decalage;
run;

data CRED.MODELE_PARAMS; set _params; run;

proc sql;
  create table CRED.GRILLE_POINTS as
  select w.variable,
         w.classe,
         w.nb,
         w.taux_defaut,
         w.woe,
         c.Estimate as beta,
         round( -( c.Estimate * w.woe + &INTERCEPT. / &NVARS. ) * &FACTEUR.
                + &DECALAGE. / &NVARS. , 0.01) as points
  from CRED.WOE_TABLES as w
       inner join _coefs as c
         on upcase(compress(w.variable)) = upcase(compress(c.Variable))
  order by w.variable, w.classe;
quit;

title "Grille de score en points";
proc print data=CRED.GRILLE_POINTS noobs label;
  label variable='Variable' classe='Classe' nb='Effectif'
        taux_defaut='Taux de defaut' woe='WOE' points='Points';
  format taux_defaut percent8.2 woe 8.4 points 8.2;
run;
title;


/*--------------------------------------------------------------------
  Application a l'echantillon de VALIDATION.
  Decoupage TEMPOREL : millesimes >= COUPURE. Jamais un tirage
  aleatoire -- il diluerait la derive de population et la degradation
  macroeconomique dans les deux echantillons, et la validation ne
  trouverait plus rien.
--------------------------------------------------------------------*/
data _val;
  set CRED.PORTEFEUILLE;
  where millesime >= &COUPURE.;
run;

%appliquer_woe(entree=_val, sortie=_val_woe);

proc logistic inmodel=_modele_store;
  score data=_val_woe out=CRED.SCORES_VAL(rename=(P_1=pd_12m));
run;

data CRED.SCORES_VAL;
  set CRED.SCORES_VAL;
  cote  = (1 - pd_12m) / max(pd_12m, 1e-9);
  score = round(&DECALAGE. + &FACTEUR. * log(max(cote, 1e-9)), 0.1);
  label pd_12m = "Probabilite de defaut a 12 mois"
        score  = "Score en points";
run;

title "Distribution du score sur la validation";
proc means data=CRED.SCORES_VAL n mean std min p5 p25 median p75 p95 max maxdec=1;
  var score pd_12m;
run;
title;
