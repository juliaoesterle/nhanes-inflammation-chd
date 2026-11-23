# Lasso and Ridge Logistic Regression — Results

## 1. Objective

This analysis evaluates whether adding inflammatory biomarkers (neutrophil-to-lymphocyte ratio and high-sensitivity C-reactive protein) improves the prediction of self-reported coronary heart disease or myocardial infarction (CHD/MI) among US adults.

Two regularised logistic regression approaches were compared:

- **Lasso (L1 regularisation):** encourages sparse models by shrinking some coefficients to zero.
- **Ridge (L2 regularisation):** shrinks coefficients towards zero without generally eliminating predictors.

## 2. Data and modelling approach

The analysis used the shared project dataset and train/test split, imported through `prepare_data.py`.

- Training set: 3,635 observations
- Test set: 909 observations
- CHD/MI cases in training set: 270

Both models were evaluated using four progressively expanded feature sets:

| Feature set | Predictors |
|---|---|
| M0 | Age, sex |
| M1 | M0 + BMI, race/ethnicity, income |
| M2 | M1 + smoking, diabetes, hypertension, total cholesterol, HDL cholesterol, eGFR |
| M3 | M2 + log-transformed NLR and hs-CRP |

Hyperparameters were tuned using stratified 5-fold cross-validation on the training set. The held-out test set was used for final evaluation.

Performance was assessed using ROC-AUC, PR-AUC and Brier score.

## 3. Cross-validation results

| Model | Feature set | Best C | Mean CV ROC-AUC |
|---|---:|---:|---:|
| Lasso | M0 | 1.0 | 0.7977 |
| Lasso | M1 | 0.1 | 0.8107 |
| Lasso | M2 | 0.1 | 0.8587 |
| Lasso | M3 | 0.1 | 0.8594 |
| Ridge | M0 | 0.1 | 0.7980 |
| Ridge | M1 | 0.1 | 0.8066 |
| Ridge | M2 | 0.1 | 0.8583 |
| Ridge | M3 | 0.1 | 0.8580 |

## 4. Test-set performance

| Model | Feature set | ROC-AUC | PR-AUC | Brier score |
|---|---|---:|---:|---:|
| Lasso | M0 | 0.8094 | 0.1956 | 0.0628 |
| Lasso | M1 | 0.8272 | 0.2557 | 0.0614 |
| Lasso | M2 | 0.8723 | 0.4308 | 0.0538 |
| Lasso | M3 | 0.8751 | 0.4488 | 0.0531 |
| Ridge | M0 | 0.8131 | 0.1990 | 0.0626 |
| Ridge | M1 | 0.8339 | 0.2846 | 0.0607 |
| Ridge | M2 | 0.8808 | 0.4723 | 0.0524 |
| Ridge | M3 | 0.8842 | 0.4908 | 0.0515 |

## 5. Interpretation

Predictive performance improved substantially when established cardiovascular risk factors were added (M0 to M2).

Adding the inflammatory biomarkers in M3 produced only small further improvements in test-set ROC-AUC:

- Lasso: increase of 0.0027.
- Ridge: increase of 0.0034.

Bootstrap confidence intervals for these paired AUC differences included zero:

- Lasso: 95% CI −0.0021 to 0.0080.
- Ridge: 95% CI −0.0015 to 0.0084.

Therefore, these results do not provide clear evidence that adding NLR and hs-CRP improves discrimination beyond the established risk factors.

## 6. Coefficient interpretation

In the Lasso M3 model, log-NLR had a positive coefficient (+0.1069), while log-CRP was shrunk to zero.

In the Ridge M3 model, log-NLR also had a positive coefficient (+0.1197), whereas log-CRP had a small negative coefficient (−0.0156).

These are regularised, standardised model coefficients. They are not odds ratios, statistical significance tests or definitive evidence of independent associations.

## 7. Calibration

Calibration was examined for the Ridge M3 model.

- Brier score: 0.0515
- Calibration intercept: 0.4275
- Calibration slope: 1.2931

The calibration slope above one suggests that predicted risks were not sufficiently spread out. In the highest predicted-risk group, the model predicted approximately 35.2% risk, compared with an observed rate of approximately 40.7%.

## 8. Overall conclusion

Both Lasso and Ridge models showed improved predictive performance as established cardiovascular risk factors were introduced.

Adding log-NLR and log-hs-CRP produced only small additional improvements. Bootstrap confidence intervals for the changes in ROC-AUC included zero, so the incremental predictive benefit remains uncertain.

This predictive modelling exercise does not establish whether NLR and hs-CRP are independently associated with CHD/MI. That research question requires a separate inferential analysis.

## 9. Limitations

- The outcome is based on self-reported CHD/MI history.
- The analysis uses a complete-case analytic sample.
- The number of CHD/MI cases is relatively small compared with non-cases.
- Predictive performance does not establish causality or independent association.
- Calibration was examined for Ridge M3, rather than systematically for every model.

