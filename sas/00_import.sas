/*====================================================================
  Import du portefeuille exporte depuis le prototype Python.
  A executer avant 01_woe_iv.sas.
====================================================================*/
%let CSV = C:/fraude-360/sas/donnees;   /* a adapter : dossier produit par python -m fraude360.credit */
libname CRED "&CSV.";

proc import datafile="&CSV./portefeuille.csv"
            out=CRED.PORTEFEUILLE dbms=csv replace;
  guessingrows=max;
  getnames=yes;
run;

proc contents data=CRED.PORTEFEUILLE varnum; run;

title "Controle d'integrite apres import";
proc sql;
  select count(*) as nb_prets,
         sum(defaut_12m) as nb_defauts,
         mean(defaut_12m) as taux_defaut format=percent9.3,
         count(distinct millesime) as nb_millesimes
  from CRED.PORTEFEUILLE;
quit;
title;
