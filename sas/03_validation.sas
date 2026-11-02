/*====================================================================
  Fraude-360 / volet credit -- ETAPE 3
  Validation independante : discrimination, calibration, stabilite,
  retrovalidation.

  Equivalent SAS de src/fraude360/credit/validation.py

  Entrees : CRED.SCORES_VAL, CRED.PORTEFEUILLE
  Sorties : CRED.VAL_DISCRIMINATION, CRED.VAL_CALIBRATION,
            CRED.VAL_PSI, CRED.VAL_RETRO, CRED.VAL_CONSTATS

  Ordre volontaire : un modele qui ne discrimine pas ne merite pas
  qu'on regarde sa calibration.
====================================================================*/

%let CHEMIN = /donnees/fraude360/credit;
libname CRED "&CHEMIN.";

%let COUPURE          = 24;
%let GINI_ACCEPTABLE  = 0.40;
%let GINI_BON         = 0.55;
%let PSI_SURVEILLANCE = 0.10;
%let PSI_ALERTE       = 0.25;


/*====================================================================
  1. DISCRIMINATION -- Gini et Kolmogorov-Smirnov
====================================================================*/
ods output Association=_assoc_val;
proc logistic data=CRED.SCORES_VAL descending;
  model defaut_12m = score;
run;
ods output close;

data CRED.VAL_DISCRIMINATION;
  set _assoc_val;
  where Label2 = 'c';
  auc  = nValue2;
  gini = 2 * auc - 1;
  length verdict $40;
  if      gini >= &GINI_BON.        then verdict = 'bon';
  else if gini >= &GINI_ACCEPTABLE. then verdict = 'acceptable';
  else                                   verdict = 'INSUFFISANT';
  keep auc gini verdict;
run;

/* KS : ecart maximal entre les fonctions de repartition cumulees des
   sains et des defauts. PROC NPAR1WAY le donne directement. */
ods output KS2Stats=_ks;
proc npar1way data=CRED.SCORES_VAL edf;
  class defaut_12m;
  var score;
run;
ods output close;

data _ks_val;
  set _ks;
  where Name1 = '_D_';
  ks = nValue1;
  keep ks;
run;

data CRED.VAL_DISCRIMINATION;
  merge CRED.VAL_DISCRIMINATION _ks_val;
run;

title "1. Discrimination sur l'echantillon de validation";
proc print data=CRED.VAL_DISCRIMINATION noobs label;
  label auc='AUC' gini='Gini' ks='KS' verdict='Verdict';
  format auc gini ks 8.4;
run;
title;


/*====================================================================
  2. CALIBRATION -- Hosmer-Lemeshow par decile

  Reserve methodologique importante : au-dela d'environ 10 000
  observations, le test du chi2 rejette presque toujours, y compris
  pour des ecarts sans portee pratique. On rapporte donc AUSSI l'ecart
  absolu moyen par decile, qui est la mesure sur laquelle la ligne de
  defense 2 se prononce reellement.
====================================================================*/
proc rank data=CRED.SCORES_VAL out=_rangs groups=10;
  var pd_12m;
  ranks decile;
run;

proc sql;
  create table CRED.VAL_CALIBRATION as
  select decile + 1            as decile,
         count(*)              as nb,
         sum(defaut_12m)       as defauts_observes,
         sum(pd_12m)           as defauts_attendus,
         mean(defaut_12m)      as taux_observe,
         mean(pd_12m)          as pd_moyenne,
         abs(mean(defaut_12m) - mean(pd_12m)) as ecart_absolu
  from _rangs
  group by decile
  order by decile;
quit;

proc sql noprint;
  select sum( (defauts_observes - defauts_attendus)**2
              / (defauts_attendus * (1 - defauts_attendus / nb)) ),
         mean(ecart_absolu),
         sum(nb)
    into :HL_CHI2, :ECART_MOYEN, :N_VAL
    from CRED.VAL_CALIBRATION;
quit;

data _hl;
  chi2    = &HL_CHI2.;
  ddl     = 8;                       /* 10 groupes - 2 */
  p_value = 1 - probchi(chi2, ddl);
  ecart_absolu_moyen_pct = 100 * &ECART_MOYEN.;
  n = &N_VAL.;
  length note $140;
  if n > 10000 then
    note = "n > 10 000 : rejet du chi2 attendu meme pour un ecart negligeable ; se fier a l'ecart absolu moyen.";
  else
    note = "n modere : le test chi2 est interpretable tel quel.";
run;

title "2. Calibration -- Hosmer-Lemeshow";
proc print data=_hl noobs; run;
proc print data=CRED.VAL_CALIBRATION noobs label;
  label decile='Decile' nb='Effectif' taux_observe='Taux observe'
        pd_moyenne='PD moyenne' ecart_absolu='Ecart';
  format taux_observe pd_moyenne ecart_absolu percent9.3;
run;
title;


/*====================================================================
  3. STABILITE -- indice de stabilite de population (PSI)

      PSI = somme sur les classes de (part_courant - part_ref)
                                     * ln(part_courant / part_ref)

  Lecture : < 0,10 stable | 0,10-0,25 a surveiller | > 0,25 recalibrage.
====================================================================*/
%macro psi(var=, ref_where=, cur_where=, sortie=);

  proc rank data=CRED.PORTEFEUILLE(where=(&ref_where.)) out=_r groups=10;
    var &var.; ranks _cl;
  run;

  proc sql;
    create table _bornes as
      select _cl, min(&var.) as bmin, max(&var.) as bmax
      from _r group by _cl;

    create table _ref as
      select _cl, count(*) / (select count(*) from _r) as part_ref
      from _r group by _cl;
  quit;

  /* les bornes du DEVELOPPEMENT sont appliquees au courant :
     recalculer les quantiles sur le courant masquerait la derive */
  proc sql;
    create table _cur as
      select b._cl,
             count(*) as n
      from CRED.PORTEFEUILLE as p, _bornes as b
      where &cur_where. and p.&var. >= b.bmin and p.&var. <= b.bmax
      group by b._cl;

    create table _cur2 as
      select _cl, n / (select sum(n) from _cur) as part_cur from _cur;

    create table &sortie. as
      select "&var." as variable length=32,
             sum( (coalesce(c.part_cur, 1e-6) - coalesce(r.part_ref, 1e-6))
                  * log( coalesce(c.part_cur, 1e-6) / coalesce(r.part_ref, 1e-6) ) )
                as psi
      from _ref as r left join _cur2 as c on r._cl = c._cl;
  quit;

%mend psi;

proc datasets lib=work nolist; delete PSI_TOUT; quit;

%macro psi_toutes;
  %local i v;
  %let liste = score_bureau ratio_endettement nb_retards_12m anciennete_emploi
               utilisation_marge revenu montant;
  %do i = 1 %to %sysfunc(countw(&liste.));
    %let v = %scan(&liste., &i.);
    %psi(var=&v., ref_where=millesime < &COUPURE.,
         cur_where=p.millesime >= &COUPURE., sortie=_psi_&i.);
    proc append base=PSI_TOUT data=_psi_&i. force; run;
  %end;
%mend psi_toutes;
%psi_toutes;

data CRED.VAL_PSI;
  set PSI_TOUT;
  length lecture $22;
  if      psi >= &PSI_ALERTE.         then lecture = 'recalibrage requis';
  else if psi >= &PSI_SURVEILLANCE.   then lecture = 'a surveiller';
  else                                     lecture = 'stable';
run;

title "3. Stabilite de population (PSI)";
proc print data=CRED.VAL_PSI noobs label;
  label variable='Variable' psi='PSI' lecture='Lecture';
  format psi 8.4;
run;
title;


/*====================================================================
  4. RETROVALIDATION -- PD predite contre defaut observe, par millesime

  Un point hors intervalle est un signal. Une SERIE de points du meme
  cote est un biais systematique : c'est ce second cas qui condamne un
  modele, pas le premier.
====================================================================*/
proc sql;
  create table CRED.VAL_RETRO as
  select millesime,
         count(*)          as nb,
         mean(pd_12m)      as pd_predite,
         mean(defaut_12m)  as defaut_observe,
         sqrt( max(mean(pd_12m) * (1 - mean(pd_12m)), 1e-12) / count(*) ) as se,
         sign(mean(defaut_12m) - mean(pd_12m)) as signe
  from CRED.SCORES_VAL
  group by millesime
  order by millesime;
quit;

data CRED.VAL_RETRO;
  set CRED.VAL_RETRO;
  borne_basse = pd_predite - 1.96 * se;
  borne_haute = pd_predite + 1.96 * se;
  dans_intervalle = (borne_basse <= defaut_observe <= borne_haute);
run;

/* plus longue serie de millesimes consecutifs du meme cote */
data _series;
  set CRED.VAL_RETRO;
  retain serie 0 serie_max 0 precedent 0;
  if signe = precedent and signe ne 0 then serie + 1;
  else serie = 1;
  serie_max = max(serie_max, serie);
  precedent = signe;
  if last then call symputx('SERIE_MAX', serie_max);
run;

proc sql noprint;
  select count(*) into :N_HORS from CRED.VAL_RETRO where dans_intervalle = 0;
  select max(serie_max) into :SERIE_MAX from _series;
quit;

title "4. Retrovalidation par millesime";
proc print data=CRED.VAL_RETRO noobs label;
  label millesime='Millesime' nb='Effectif' pd_predite='PD predite'
        defaut_observe='Defaut observe' dans_intervalle='Dans IC 95 %';
  format pd_predite defaut_observe borne_basse borne_haute percent9.3;
run;
title;


/*====================================================================
  5. CONSTATS DE VALIDATION
  Aucun test ne conclut seul. Le rapport agrege les reserves et les
  nomme ; l'homologation reste une decision humaine.
====================================================================*/
data CRED.VAL_CONSTATS;
  length constat $200;
  set CRED.VAL_DISCRIMINATION;

  if gini < &GINI_ACCEPTABLE. then do;
    constat = catx(' ', 'Gini de validation', put(gini, 8.4),
                   "sous le repere &GINI_ACCEPTABLE..");
    output;
  end;

  if &ECART_MOYEN. * 100 > 1.0 then do;
    constat = catx(' ', 'Calibration : ecart absolu moyen de',
                   put(&ECART_MOYEN. * 100, 8.3),
                   'point(s) de pourcentage par decile.');
    output;
  end;

  if &SERIE_MAX. >= 5 then do;
    constat = catx(' ', 'Retrovalidation :', put(&SERIE_MAX., 3.),
                   'millesimes consecutifs du meme cote -- biais systematique suspecte.');
    output;
  end;

  keep constat;
run;

proc sql noprint; select count(*) into :N_CONSTATS from CRED.VAL_CONSTATS; quit;

title "5. Constats de validation";
%if &N_CONSTATS. = 0 %then %do;
  data _null_; put "Aucune reserve bloquante."; run;
%end;
%else %do;
  proc print data=CRED.VAL_CONSTATS noobs label;
    label constat="&N_CONSTATS. reserve(s) a lever avant homologation";
  run;
%end;
title;
