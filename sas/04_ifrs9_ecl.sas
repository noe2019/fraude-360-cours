/*====================================================================
  Fraude-360 / volet credit -- ETAPE 4
  IFRS 9 : classement par etape et perte de credit attendue (ECL)

  Equivalent SAS de src/fraude360/credit/ifrs9.py

  Entrees : CRED.SCORES_VAL (PD courante), CRED.SCORES_OCTROI (PD a l'octroi)
  Sorties : CRED.IFRS9_PRETS    un enregistrement par pret, avec etape et ECL
            CRED.IFRS9_SYNTHESE agregat par etape

  Rappel de la norme :
     Etape 1  sain depuis l'octroi                  -> ECL a 12 mois
     Etape 2  augmentation significative du risque  -> ECL a VIE
     Etape 3  en defaut                             -> ECL a vie

  Le critere d'augmentation significative du risque de credit (ASRC)
  n'est PAS chiffre par la norme : chaque institution le definit et doit
  le justifier. C'est le premier point que regarde la validation.
====================================================================*/

%let CHEMIN = /donnees/fraude360/credit;
libname CRED "&CHEMIN.";

%let TAUX_ACTUALISATION = 0.065;   /* taux d'interet effectif moyen */
%let ASRC_MULTIPLE      = 2.0;     /* PD a vie multipliee par ... */
%let ASRC_ABSOLU        = 0.005;   /* ET hausse d'au moins 50 pb */
%let ASRC_CLASSES       = 3;       /* ou degradation de 3 classes internes */
%let HORIZON_MAX        = 10;      /* annees */


/*--------------------------------------------------------------------
  1. PD a vie depuis la PD a 12 mois

  Survie a hasard constant :  PD_vie = 1 - (1 - PD_12m) ^ (duree / 12)

  C'est une APPROXIMATION, a declarer comme telle dans le dossier de
  modele. Un modele multi-horizon (PD par annee) serait preferable ;
  en son absence, cette forme est l'usage du marche et se justifie tant
  que la duree residuelle reste modeste.
--------------------------------------------------------------------*/
data _pd;
  merge CRED.SCORES_VAL   (in=a keep=pret_id millesime pd_12m score
                                  duree_mois exposition lgd_observee
                                  nb_retards_12m defaut_12m)
        CRED.SCORES_OCTROI(in=b keep=pret_id rename=(pd_12m=pd_octroi));
  by pret_id;
  if a;

  annees_residuelles = max(min(duree_mois / 12, &HORIZON_MAX.), 1/12);

  pd_vie_courante = 1 - (1 - min(max(pd_12m,   1e-9), 0.999)) ** annees_residuelles;
  pd_vie_octroi   = 1 - (1 - min(max(pd_octroi,1e-9), 0.999)) ** annees_residuelles;

  /* proxy d'arrieres : 2 retards ou plus sur 12 mois -> 30 jours et plus */
  jours_retard = ifn(nb_retards_12m >= 2, 45, 0);
run;


/*--------------------------------------------------------------------
  2. Classement par etape

  Le double critere quantitatif (multiple ET hausse absolue) evite de
  basculer en etape 2 des prets tres sains dont la PD passe de 0,05 %
  a 0,11 % : mathematiquement un doublement, economiquement rien.
--------------------------------------------------------------------*/
data CRED.IFRS9_PRETS;
  set _pd;

  asrc_quantitatif = ( pd_vie_courante > &ASRC_MULTIPLE. * pd_vie_octroi )
                     and ( (pd_vie_courante - pd_vie_octroi) >= &ASRC_ABSOLU. );
  asrc_arrieres    = ( jours_retard >= 30 );

  if      defaut_12m = 1                            then etape_ifrs9 = 3;
  else if asrc_quantitatif or asrc_arrieres         then etape_ifrs9 = 2;
  else                                                   etape_ifrs9 = 1;

  length motif_etape $60;
  if      etape_ifrs9 = 3 then motif_etape = 'defaut avere';
  else if asrc_arrieres   then motif_etape = 'arrieres de 30 jours ou plus';
  else if asrc_quantitatif then motif_etape = 'PD a vie plus que doublee et hausse >= 50 pb';
  else                          motif_etape = 'sain';

  /*------------------------------------------------------------------
    3. ECL = somme sur les periodes de
             PD_marginale(t) * LGD * EAD * facteur d'actualisation(t)

    PD marginale en annee t = (1 - PD_12m)^(t-1) * PD_12m
    (defaut en t, ayant survecu les t-1 annees precedentes).

    Etape 1 : une seule periode. Etapes 2 et 3 : duree residuelle.
    Etape 3 : la perte n'est plus probabiliste, elle est encourue.
  ------------------------------------------------------------------*/
  horizon = ifn(etape_ifrs9 = 1, 1, max(1, ceil(annees_residuelles)));

  ecl = 0;
  survie = 1;
  do t = 1 to horizon;
    pd_marginale = survie * pd_12m;
    ecl = ecl + pd_marginale * lgd_observee * exposition
                / ((1 + &TAUX_ACTUALISATION.) ** t);
    survie = survie * (1 - pd_12m);
  end;

  if etape_ifrs9 = 3 then ecl = lgd_observee * exposition;

  ecl = round(ecl, 0.01);
  taux_couverture = ecl / max(exposition, 0.01);

  label pd_12m          = "PD a 12 mois"
        pd_vie_courante = "PD a vie (courante)"
        etape_ifrs9     = "Etape IFRS 9"
        ecl             = "Perte de credit attendue"
        motif_etape     = "Motif du classement";

  drop t survie pd_marginale;
run;


/*--------------------------------------------------------------------
  4. Synthese par etape
--------------------------------------------------------------------*/
proc sql;
  create table CRED.IFRS9_SYNTHESE as
  select etape_ifrs9,
         count(*)            as nb_prets,
         sum(exposition)     as exposition,
         sum(ecl)            as ecl,
         mean(pd_12m)        as pd_moyenne,
         sum(ecl) / sum(exposition) as taux_couverture
  from CRED.IFRS9_PRETS
  group by etape_ifrs9
  order by etape_ifrs9;
quit;

title "IFRS 9 -- synthese par etape";
proc print data=CRED.IFRS9_SYNTHESE noobs label;
  label etape_ifrs9='Etape' nb_prets='Prets' exposition='Exposition'
        ecl='ECL' pd_moyenne='PD moyenne' taux_couverture='Couverture';
  format exposition ecl dollar18.2 pd_moyenne taux_couverture percent9.3;
run;

title "IFRS 9 -- motifs de classement en etape 2";
proc freq data=CRED.IFRS9_PRETS;
  where etape_ifrs9 = 2;
  tables motif_etape / nocum;
run;
title;


/*--------------------------------------------------------------------
  5. Analyse de sensibilite

  La provision depend d'hypotheses non observables. Une provision
  presentee sans son intervalle de sensibilite n'est pas defendable
  devant un verificateur : c'est le premier tableau qu'il demande.
--------------------------------------------------------------------*/
%macro sensibilite(facteur_pd=, libelle=);
  proc sql;
    create table _sens_&libelle. as
    select "&libelle." as scenario length=32,
           sum( case when etape_ifrs9 = 3 then lgd_observee * exposition
                     else ecl * &facteur_pd. end ) as ecl_totale
    from CRED.IFRS9_PRETS;
  quit;
  proc append base=_SENSIBILITE data=_sens_&libelle. force; run;
%mend;

proc datasets lib=work nolist; delete _SENSIBILITE; quit;
%sensibilite(facteur_pd=0.8, libelle=PD_moins_20pct);
%sensibilite(facteur_pd=1.0, libelle=Central);
%sensibilite(facteur_pd=1.2, libelle=PD_plus_20pct);
%sensibilite(facteur_pd=1.5, libelle=Scenario_adverse);

title "IFRS 9 -- sensibilite de la provision aux hypotheses de PD";
proc print data=_SENSIBILITE noobs label;
  label scenario='Scenario' ecl_totale='ECL totale';
  format ecl_totale dollar18.2;
run;
title;
