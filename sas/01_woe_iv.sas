/*====================================================================
  Fraude-360 / volet credit -- ETAPE 1
  Discretisation en poids de la preuve (WOE) et valeur d'information (IV)

  Equivalent SAS de src/fraude360/credit/scorecard.py

  Entree  : CRED.PORTEFEUILLE  (export du portefeuille de prets)
  Sorties : CRED.WOE_TABLES    une ligne par variable x classe
            CRED.IV_RESUME     une ligne par variable
            CRED.DEV_WOE       developpement, variables remplacees par leur WOE

  Contexte : SAS 9.4 ou SAS Viya. Aucune procedure sous licence
  SAS/STAT n'est utilisee ici hors PROC RANK et PROC SQL.
====================================================================*/

%let CHEMIN = /donnees/fraude360/credit;   /* a adapter */
libname CRED "&CHEMIN.";

/* correction de Laplace : evite un WOE infini sur une classe sans mauvais */
%let EPS = 0.5;
/* millesime de coupure developpement / validation */
%let COUPURE = 24;
/* seuil d'IV sous lequel une variable est ecartee */
%let IV_MIN = 0.02;


/*--------------------------------------------------------------------
  Macro : woe_numerique
  Discretise une variable numerique en NCLASSES quantiles, calcule le
  WOE et l'IV, et verifie la MONOTONIE du taux de defaut.

  Si la monotonie n'est pas atteinte, on reduit le nombre de classes et
  on recommence. Un decoupage non monotone sur une variable ordonnee
  capte du bruit d'echantillon, pas du signal : il ne passerait pas la
  revue de la ligne de defense 2.
--------------------------------------------------------------------*/
%macro woe_numerique(table=, var=, cible=defaut_12m, nclasses=6, croissant=1);

  %local k ok;
  %let k = &nclasses.;
  %let ok = 0;

  %do %while (&k. >= 2 and &ok. = 0);

    /* --- decoupage en quantiles ----------------------------------- */
    proc rank data=&table. out=_rangs groups=&k. ties=low;
      var &var.;
      ranks _classe;
      where &var. is not null;
    run;

    /* les valeurs manquantes forment leur PROPRE classe : jamais
       imputees. Une valeur manquante porte souvent de l'information
       (pas de bien en garantie, pas d'historique) et l'imputer la
       detruit. */
    data _avec_manq;
      set &table.;
      if missing(&var.) then _classe = -1;
      else _classe = .;
    run;

    proc sql;
      create table _base as
      select coalesce(r._classe, m._classe) as classe,
             t.&cible. as y
      from &table. as t
           left join _rangs   as r on t.pret_id = r.pret_id
           left join _avec_manq as m on t.pret_id = m.pret_id
      ;
    quit;

    /* --- comptages par classe -------------------------------------- */
    proc sql;
      create table _cnt as
      select classe,
             count(*)                as nb,
             sum(y)                  as nb_mauvais,
             count(*) - sum(y)       as nb_bons,
             mean(y)                 as taux_defaut
      from _base
      group by classe
      order by classe;

      select sum(nb_bons), sum(nb_mauvais), count(*)
        into :tot_b, :tot_m, :n_cl
        from _cnt;
    quit;

    /* --- WOE et IV -------------------------------------------------- */
    data _woe;
      set _cnt;
      length variable $32;
      variable    = "&var.";
      part_bons    = (nb_bons    + &EPS.) / (&tot_b. + &EPS. * &n_cl.);
      part_mauvais = (nb_mauvais + &EPS.) / (&tot_m. + &EPS. * &n_cl.);
      woe        = log(part_bons / part_mauvais);
      iv_partiel = (part_bons - part_mauvais) * woe;
    run;

    /* --- controle de monotonie (classes reelles seulement) ---------- */
    proc sql noprint;
      select count(*) into :n_rupture
      from (select classe, taux_defaut,
                   lag(taux_defaut) over (order by classe) as precedent
            from _woe where classe >= 0)
      where precedent is not null
        and ( (&croissant. = 1 and taux_defaut < precedent)
           or (&croissant. = 0 and taux_defaut > precedent) );
    quit;

    %if &n_rupture. = 0 %then %let ok = 1;
    %else %let k = %eval(&k. - 1);

  %end;

  %if &ok. = 0 %then
    %put ATTENTION: &var. -- monotonie non atteinte, decoupage a &k. classes conserve.;

  proc append base=CRED.WOE_TABLES data=_woe force; run;

%mend woe_numerique;


/*--------------------------------------------------------------------
  Macro : woe_categorielle
  Une modalite par classe. Pas de regroupement automatique : le
  regroupement de modalites doit etre un choix metier documente.
--------------------------------------------------------------------*/
%macro woe_categorielle(table=, var=, cible=defaut_12m);

  proc sql;
    create table _cnt as
    select &var. as classe_c length=32,
           count(*)          as nb,
           sum(&cible.)      as nb_mauvais,
           count(*) - sum(&cible.) as nb_bons,
           mean(&cible.)     as taux_defaut
    from &table.
    group by &var.;

    select sum(nb_bons), sum(nb_mauvais), count(*)
      into :tot_b, :tot_m, :n_cl from _cnt;
  quit;

  data _woe;
    set _cnt;
    length variable $32;
    variable     = "&var.";
    part_bons    = (nb_bons    + &EPS.) / (&tot_b. + &EPS. * &n_cl.);
    part_mauvais = (nb_mauvais + &EPS.) / (&tot_m. + &EPS. * &n_cl.);
    woe          = log(part_bons / part_mauvais);
    iv_partiel   = (part_bons - part_mauvais) * woe;
  run;

  proc append base=CRED.WOE_TABLES data=_woe force; run;

%mend woe_categorielle;


/*====================  EXECUTION  ==================================*/

proc datasets lib=CRED nolist; delete WOE_TABLES IV_RESUME; quit;

/* echantillon de developpement : millesimes anterieurs a la coupure */
data _dev;
  set CRED.PORTEFEUILLE;
  where millesime < &COUPURE.;
run;

/* Le sens attendu est declare variable par variable : c'est une
   hypothese metier, pas une observation. On la teste, on ne la subit pas. */
%woe_numerique(table=_dev, var=score_bureau,      croissant=0);
%woe_numerique(table=_dev, var=ratio_endettement, croissant=1);
%woe_numerique(table=_dev, var=nb_retards_12m,    croissant=1, nclasses=4);
%woe_numerique(table=_dev, var=anciennete_emploi, croissant=0);
%woe_numerique(table=_dev, var=utilisation_marge, croissant=1);
%woe_numerique(table=_dev, var=ratio_pret_valeur, croissant=1);
%woe_numerique(table=_dev, var=duree_mois,        croissant=1, nclasses=5);
%woe_numerique(table=_dev, var=age,               croissant=0);
%woe_numerique(table=_dev, var=revenu,            croissant=0);
%woe_numerique(table=_dev, var=montant,           croissant=1);

%woe_categorielle(table=_dev, var=produit);
%woe_categorielle(table=_dev, var=region);


/* --- valeur d'information par variable ----------------------------- */
proc sql;
  create table CRED.IV_RESUME as
  select variable,
         sum(iv_partiel) as iv,
         count(*)        as nb_classes,
         case when calculated iv <  0.02 then 'inutile'
              when calculated iv <  0.10 then 'faible'
              when calculated iv <  0.30 then 'moyen'
              when calculated iv <  0.50 then 'fort'
              else 'suspect - verifier la fuite'
         end as force length=28
  from CRED.WOE_TABLES
  group by variable
  order by iv desc;
quit;

title "Valeur d'information par variable -- echantillon de developpement";
proc print data=CRED.IV_RESUME noobs label;
  label variable='Variable' iv='IV' nb_classes='Classes' force='Lecture';
  format iv 8.4;
run;
title;

/* Un IV superieur a 0,5 n'est presque jamais un bon signe sur un
   modele de credit : c'est le plus souvent une variable qui contient
   l'information du defaut (fuite). A verifier avant toute suite. */
proc sql noprint;
  select count(*) into :n_suspect from CRED.IV_RESUME where iv > 0.5;
quit;
%if &n_suspect. > 0 %then
  %put ATTENTION: &n_suspect. variable(s) a IV > 0.5 -- verifier une fuite d information.;
