/*====================================================================
  Grille de score figee -- GENERE AUTOMATIQUEMENT, NE PAS EDITER

  Reproduit src/fraude360/credit/scorecard.predire() a l'identique.
  Toute divergence entre ce code et le modele Python est un defaut :
  la verification d'equivalence (export_sas.verifier_equivalence)
  doit etre rejouee apres chaque reentrainement.
====================================================================*/

data scores;
  set portefeuille;

  _logit = -3.4308420636;

  /* --- score_bureau (beta = -0.794014, IV = 0.2277) --- */
  _woe_score_bureau = 0;
  if missing(score_bureau) then _woe_score_bureau = 0.0000000000;
  else if -1e300 < score_bureau and score_bureau <= 606.000000 then _woe_score_bureau = -0.6159342626;
  else if 606.000000 < score_bureau and score_bureau <= 637.000000 then _woe_score_bureau = -0.2282610909;
  else if 637.000000 < score_bureau and score_bureau <= 662.000000 then _woe_score_bureau = -0.0028977072;
  else if 662.000000 < score_bureau and score_bureau <= 687.000000 then _woe_score_bureau = 0.1373386211;
  else if 687.000000 < score_bureau and score_bureau <= 717.000000 then _woe_score_bureau = 0.4103419047;
  else if 717.000000 < score_bureau and score_bureau <= 1e300 then _woe_score_bureau = 0.9981874837;
  _logit = _logit + -0.7940135159 * _woe_score_bureau;

  /* --- ratio_endettement (beta = -0.764766, IV = 0.7846) --- */
  _woe_ratio_endettement = 0;
  if missing(ratio_endettement) then _woe_ratio_endettement = 0.0000000000;
  else if -1e300 < ratio_endettement and ratio_endettement <= 0.243000 then _woe_ratio_endettement = 1.1055780909;
  else if 0.243000 < ratio_endettement and ratio_endettement <= 0.356000 then _woe_ratio_endettement = 0.8323186352;
  else if 0.356000 < ratio_endettement and ratio_endettement <= 0.489000 then _woe_ratio_endettement = 0.7146988244;
  else if 0.489000 < ratio_endettement and ratio_endettement <= 0.672000 then _woe_ratio_endettement = 0.5056634125;
  else if 0.672000 < ratio_endettement and ratio_endettement <= 0.996000 then _woe_ratio_endettement = 0.1310082044;
  else if 0.996000 < ratio_endettement and ratio_endettement <= 1e300 then _woe_ratio_endettement = -1.2498555233;
  _logit = _logit + -0.7647664296 * _woe_ratio_endettement;

  /* --- nb_retards_12m (beta = -0.982439, IV = 0.0874) --- */
  _woe_nb_retards_12m = 0;
  if missing(nb_retards_12m) then _woe_nb_retards_12m = 0.0000000000;
  else if -1e300 < nb_retards_12m and nb_retards_12m <= 1.000000 then _woe_nb_retards_12m = 0.1639255450;
  else if 1.000000 < nb_retards_12m and nb_retards_12m <= 2.000000 then _woe_nb_retards_12m = -0.3168888425;
  else if 2.000000 < nb_retards_12m and nb_retards_12m <= 1e300 then _woe_nb_retards_12m = -0.7833505776;
  _logit = _logit + -0.9824387759 * _woe_nb_retards_12m;

  /* --- anciennete_emploi (beta = -0.499473, IV = 0.0275) --- */
  _woe_anciennete_emploi = 0;
  if missing(anciennete_emploi) then _woe_anciennete_emploi = 0.0000000000;
  else if -1e300 < anciennete_emploi and anciennete_emploi <= 2.900000 then _woe_anciennete_emploi = -0.1429660434;
  else if 2.900000 < anciennete_emploi and anciennete_emploi <= 5.000000 then _woe_anciennete_emploi = -0.1139473423;
  else if 5.000000 < anciennete_emploi and anciennete_emploi <= 8.100000 then _woe_anciennete_emploi = 0.0250159719;
  else if 8.100000 < anciennete_emploi and anciennete_emploi <= 1e300 then _woe_anciennete_emploi = 0.2921491117;
  _logit = _logit + -0.4994729826 * _woe_anciennete_emploi;

  /* --- utilisation_marge (beta = -0.589108, IV = 0.2924) --- */
  _woe_utilisation_marge = 0;
  if missing(utilisation_marge) then _woe_utilisation_marge = 0.0000000000;
  else if -1e300 < utilisation_marge and utilisation_marge <= 0.331000 then _woe_utilisation_marge = 0.6467530180;
  else if 0.331000 < utilisation_marge and utilisation_marge <= 0.480000 then _woe_utilisation_marge = 0.3818196464;
  else if 0.480000 < utilisation_marge and utilisation_marge <= 0.646000 then _woe_utilisation_marge = 0.1999826042;
  else if 0.646000 < utilisation_marge and utilisation_marge <= 1e300 then _woe_utilisation_marge = -0.7054474900;
  _logit = _logit + -0.5891084574 * _woe_utilisation_marge;

  /* --- revenu (beta = -0.280019, IV = 0.3621) --- */
  _woe_revenu = 0;
  if missing(revenu) then _woe_revenu = 0.0000000000;
  else if -1e300 < revenu and revenu <= 38800.000000 then _woe_revenu = -0.9142965977;
  else if 38800.000000 < revenu and revenu <= 49400.000000 then _woe_revenu = -0.0681204678;
  else if 49400.000000 < revenu and revenu <= 59900.000000 then _woe_revenu = 0.1974226669;
  else if 59900.000000 < revenu and revenu <= 72500.000000 then _woe_revenu = 0.4213822853;
  else if 72500.000000 < revenu and revenu <= 92200.000000 then _woe_revenu = 0.5303575498;
  else if 92200.000000 < revenu and revenu <= 1e300 then _woe_revenu = 0.8062857837;
  _logit = _logit + -0.2800187669 * _woe_revenu;

  /* --- montant (beta = -0.165600, IV = 0.1376) --- */
  _woe_montant = 0;
  if missing(montant) then _woe_montant = 0.0000000000;
  else if -1e300 < montant and montant <= 10900.000000 then _woe_montant = 0.4197220823;
  else if 10900.000000 < montant and montant <= 19900.000000 then _woe_montant = 0.1978952417;
  else if 19900.000000 < montant and montant <= 1e300 then _woe_montant = -0.4336232396;
  _logit = _logit + -0.1656003013 * _woe_montant;

  pd_12m = 1 / (1 + exp(-_logit));
  _cote  = (1 - pd_12m) / max(pd_12m, 1e-9);
  score  = round(487.1228762045 + 28.8539008178 * log(max(_cote, 1e-9)), 0.1);

  label pd_12m = "Probabilite de defaut a 12 mois"
        score  = "Score en points";
  drop _logit _cote _woe_:;
run;
