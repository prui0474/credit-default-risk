# Credit Default Risk - Model Report

*Data source:* `freddie` &nbsp;|&nbsp; *Rows:* 331,122 &nbsp;|&nbsp; *Defaults:* 2.82% &nbsp;|&nbsp; *Generated:* 2026-10-15 22:23

## 1. Target and sample

- **Target definition:** `ever_90dpd_within_36m` - a loan is a default if it reaches 90+ days past due within 36 months of origination.
- **Model inputs:** origination-time attributes only. Payment, recovery and delinquency fields are excluded to prevent target leakage.
- **Split:** out-of-time by origination vintage, so the test cohort is booked in a different rate environment than the training cohort.

| split | rows | vintages | default_rate |
|---|---|---|---|
| train | 183668 | 2015-2018 | 0.0264 |
| valid | 48321 | 2019-2019 | 0.0515 |
| oot | 99133 | 2020-2021 | 0.0202 |

## 2. Feature screening (WOE / IV)

Information value on the training cohort: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious (often a leakage smell).

| feature | iv | strength |
|---|---|---|
| fico | 0.5690 | suspicious |
| orig_rate | 0.4510 | strong |
| dti | 0.2278 | medium |
| cltv | 0.1381 | medium |
| ltv | 0.1341 | medium |
| state | 0.1242 | medium |
| term_months | 0.0637 | weak |
| loan_purpose | 0.0273 | weak |
| mi_pct | 0.0152 | useless |
| channel | 0.0144 | useless |
| orig_upb | 0.0101 | useless |
| occupancy | 0.0083 | useless |
| num_borrowers | 0.0053 | useless |
| property_type | 0.0022 | useless |

## 3. Discrimination

| model | split | n | bad_rate | auc | gini | ks |
|---|---|---|---|---|---|---|
| scorecard | train | 183668 | 0.0264 | 0.7805 | 0.5611 | 0.4187 |
| scorecard | valid | 48321 | 0.0515 | 0.7050 | 0.4100 | 0.3025 |
| scorecard | oot | 99133 | 0.0202 | 0.7462 | 0.4924 | 0.3754 |
| gbm | train | 183668 | 0.0264 | 0.7993 | 0.5985 | 0.4514 |
| gbm | valid | 48321 | 0.0515 | 0.7040 | 0.4080 | 0.3021 |
| gbm | oot | 99133 | 0.0202 | 0.7381 | 0.4762 | 0.3542 |

Gini = 2*AUC-1. KS is the maximum gap between the cumulative default and non-default score distributions.

## 4. Stability over time

A model that holds up in-sample but decays across vintages is not deployable. PSI below 0.10 is stable, 0.10-0.25 needs monitoring, above 0.25 means the population has moved.

| vintage_year | n | scorecard_auc | gbm_auc | scorecard_gini | gbm_gini | scorecard_ks | gbm_ks | gbm_psi |
|---|---|---|---|---|---|---|---|---|
| 2015 | 39629 | 0.7913 | 0.8119 | 0.5826 | 0.6238 | 0.4645 | 0.4925 | 0.0128 |
| 2016 | 47753 | 0.8055 | 0.8266 | 0.6111 | 0.6532 | 0.4630 | 0.5089 | 0.0217 |
| 2017 | 48777 | 0.7561 | 0.7723 | 0.5122 | 0.5445 | 0.3808 | 0.4088 | 0.0436 |
| 2018 | 47509 | 0.7236 | 0.7467 | 0.4471 | 0.4934 | 0.3347 | 0.3664 | 0.3726 |
| 2019 | 48321 | 0.7050 | 0.7040 | 0.4100 | 0.4080 | 0.3025 | 0.3021 | 0.1670 |
| 2020 | 49337 | 0.7503 | 0.7406 | 0.5005 | 0.4813 | 0.3778 | 0.3544 | 0.2283 |
| 2021 | 49796 | 0.7282 | 0.7276 | 0.4565 | 0.4552 | 0.3700 | 0.3511 | 0.3283 |

Feature-level PSI (train vs OOT):

| feature | psi | verdict |
|---|---|---|
| orig_rate | 2.5973 | shifted |
| dti | 0.3162 | shifted |
| loan_purpose | 0.1612 | monitor |
| cltv | 0.0949 | stable |
| ltv | 0.0844 | stable |
| orig_upb | 0.0521 | stable |
| fico | 0.0255 | stable |
| term_months | 0.0044 | stable |
| channel | 0.0025 | stable |
| state | 0.0023 | stable |
| property_type | 0.0010 | stable |
| occupancy | 0.0001 | stable |

## 5. Calibration

Ranking power alone is not enough: an uncalibrated PD mis-prices the loan and mis-states expected loss.

| n | predicted | actual | gap |
|---|---|---|---|
| 9914 | 0.0015 | 0.0026 | 0.0012 |
| 9913 | 0.0023 | 0.0055 | 0.0032 |
| 9913 | 0.0033 | 0.0065 | 0.0032 |
| 9913 | 0.0042 | 0.0089 | 0.0046 |
| 9914 | 0.0055 | 0.0125 | 0.0070 |
| 9913 | 0.0071 | 0.0157 | 0.0086 |
| 9913 | 0.0096 | 0.0218 | 0.0122 |
| 9913 | 0.0138 | 0.0240 | 0.0102 |
| 9913 | 0.0218 | 0.0417 | 0.0198 |
| 9914 | 0.0528 | 0.0632 | 0.0105 |

## 6. Decision layer

Cutoff scenarios on the out-of-time cohort (approve every applicant below the PD cutoff):

| cutoff_pd | approval_rate | approved_bad_rate | rejected_bad_rate | bad_capture_of_rejected |
|---|---|---|---|---|
| 0.0500 | 0.9623 | 0.0179 | 0.0804 | 0.1500 |
| 0.1000 | 0.9932 | 0.0197 | 0.0962 | 0.0324 |
| 0.1500 | 0.9987 | 0.0200 | 0.1840 | 0.0115 |
| 0.2000 | 0.9997 | 0.0202 | 0.2000 | 0.0030 |
| 0.3000 | 1.0000 | 0.0202 |  | 0.0000 |

## 7. Risk ranking

| band | n | bad | min_score | max_score | bad_rate | lift | cum_bad_capture |
|---|---|---|---|---|---|---|---|
| D1 | 9914 | 627.0000 | 0.0283 | 0.2790 | 0.0632 | 3.1238 | 0.3124 |
| D2 | 9913 | 413.0000 | 0.0167 | 0.0283 | 0.0417 | 2.0579 | 0.5182 |
| D3 | 9913 | 238.0000 | 0.0113 | 0.0167 | 0.0240 | 1.1859 | 0.6368 |
| D4 | 9913 | 216.0000 | 0.0081 | 0.0113 | 0.0218 | 1.0763 | 0.7444 |
| D5 | 9913 | 156.0000 | 0.0062 | 0.0081 | 0.0157 | 0.7773 | 0.8221 |
| D6 | 9914 | 124.0000 | 0.0048 | 0.0062 | 0.0125 | 0.6178 | 0.8839 |
| D7 | 9913 | 88.0000 | 0.0037 | 0.0048 | 0.0089 | 0.4385 | 0.9278 |
| D8 | 9913 | 64.0000 | 0.0028 | 0.0037 | 0.0065 | 0.3189 | 0.9596 |
| D9 | 9913 | 55.0000 | 0.0019 | 0.0028 | 0.0055 | 0.2740 | 0.9870 |
| D10 | 9914 | 26.0000 | 0.0004 | 0.0019 | 0.0026 | 0.1295 | 1.0000 |

## 8. Scorecard coefficients

| feature | woe_coefficient | odds_multiplier |
|---|---|---|
| orig_upb | 1.0786 | 2.9407 |
| state | 0.8156 | 2.2606 |
| channel | 0.8145 | 2.2580 |
| fico | 0.7997 | 2.2248 |
| orig_rate | 0.6846 | 1.9829 |
| dti | 0.6748 | 1.9636 |
| occupancy | 0.4300 | 1.5373 |
| ltv | 0.3828 | 1.4663 |
| term_months | -0.3519 | 0.7033 |
| cltv | 0.3311 | 1.3925 |
| loan_purpose | 0.3132 | 1.3678 |
| num_borrowers | 0.2617 | 1.2992 |
| property_type | 0.2178 | 1.2433 |
| mi_pct | -0.2011 | 0.8178 |

## 9. Limitations

- AUC/KS measure ranking, not profit. Origination-level economics need rate, loss-given-default and prepayment assumptions to be judged properly.
- The target is a binary 90+ DPD flag within a fixed window; it ignores time-to-default, which a survival model would exploit.
- No reject inference: the training book only contains approved loans, so the model is blind to the applicants that were previously declined.
- Protected attributes are not in the data. Fairness testing therefore needs a proxy analysis (e.g. geography) which is indicative, not conclusive.
- The note rate is risk-priced: it is partly a function of the credit score and LTV, so feeding it to the model can double-count information that is already there. Check whether the scorecard holds up without it.

## Figures

![01_roc_oot.png](figures/01_roc_oot.png)
![02_score_separation.png](figures/02_score_separation.png)
![03_auc_by_vintage.png](figures/03_auc_by_vintage.png)
![04_calibration.png](figures/04_calibration.png)
![05_feature_psi.png](figures/05_feature_psi.png)
![06_risk_bands.png](figures/06_risk_bands.png)
