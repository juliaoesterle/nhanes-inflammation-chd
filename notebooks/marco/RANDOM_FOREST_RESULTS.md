# Random Forest Analysis Results (`notebooks/marco/random_forest_model.py`)

**Author:** Marco  
**Project:** NHANES 2021–2023 — Incremental Predictive Value of Systemic Inflammatory Markers (NLR & hs-CRP) for Prevalent CHD / MI  
**Baseline Dataset & Split:** Shared `prepare_data.py` pipeline (`n = 4,544` complete cases, `337` CHD/MI cases = `7.42%`; stratified 80/20 split, `random_state = 42` -> Training `n = 3,635` [`270` cases, `7.4%`], Test `n = 909` [`67` cases, `7.4%`]).

---

## 1. Methodology & Harmonization Choices

To ensure an apples-to-apples comparison with k-NN (Julia), Lasso/Ridge (Petya), and Logistic Regression (Jannis):

- **Shared Pipeline (`prepare_data.py`):** Uses the exact same analytic dataset (`n = 4,544`), feature definitions (`M0` through `M3` plus sensitivity sets `M2 + NLR`, `M2 + CRP`, `M2 no chol`, `M3 no chol`), and stratified 80/20 train/test split (`random_state = 42`).
- **Leakage-Free Preprocessing (`ColumnTransformer` inside CV):** Numeric predictors remain untransformed (decision trees are invariant to monotonic scaling and log transforms); `race_eth` is one-hot encoded (`OneHotEncoder(handle_unknown="ignore")`) exclusively inside each CV fold and final training pipeline.
- **Exhaustive Hyperparameter Tuning (`GridSearchCV`):** 5-fold stratified CV (`scoring="roc_auc"`) over 40 combinations (`200` fits per feature set): `n_estimators=300`, `max_depth ∈ {4, 6, 8, 12, None}`, `min_samples_leaf ∈ {5, 10, 20, 50}`, `max_features ∈ {"sqrt", 0.5}`, and uniform `class_weight=None`.
- **Uniform Unweighted Probabilities (`class_weight=None`):** All 8 feature sets use `class_weight=None` so predicted probabilities are directly calibrated against the true `7.4%` prevalence, making Brier scores (`null Brier = 0.0683`) and probability thresholds directly comparable across `M0–M3` and across all four team models. A separate check in `results/class_weight_sensitivity.csv` evaluates `balanced_subsample` on `M2` and `M3`.
- **Thresholding & Uncertainty:** The operating threshold on `M3` is chosen via **training-set out-of-fold (OOF) Youden's J** (`0.0883`) to avoid test-set leakage. Test-set 95% CIs and ΔAUC one-sided bootstrap p-values (`P(ΔAUC <= 0)`) use `1,000` stratified bootstrap resamples.

---

## 2. Hyperparameter Tuning & Training CV Performance

| Feature Set | Best Hyperparameters (`class_weight=None`) | 5-Fold CV ROC-AUC (Mean ± SD) | Training OOF ΔAUC vs Reference |
| :--- | :--- | :---: | :---: |
| **M0** (Age, Sex) | `max_depth=4`, `min_samples_leaf=50`, `max_features=sqrt` | `0.7921 ± 0.0098` | — |
| **M1** (+ BMI, Race/Eth, PIR) | `max_depth=4`, `min_samples_leaf=50`, `max_features=0.5` | `0.8043 ± 0.0078` | — |
| **M2** (+ Clinical Risk Factors) | `max_depth=12`, `min_samples_leaf=50`, `max_features=sqrt` | `0.8584 ± 0.0116` | Reference (`OOF = 0.8574`) |
| **M2 + NLR** | `max_depth=4`, `min_samples_leaf=5`, `max_features=sqrt` | `0.8612 ± 0.0122` | `+0.0023` (`0.8596 vs 0.8574`) |
| **M2 + CRP** | `max_depth=12`, `min_samples_leaf=50`, `max_features=sqrt` | `0.8585 ± 0.0123` | `+0.0003` (`0.8577 vs 0.8574`) |
| **M3** (M2 + log NLR + log hs-CRP) | `max_depth=None`, `min_samples_leaf=50`, `max_features=sqrt` | `0.8607 ± 0.0135` | `+0.0028` (`0.8602 vs 0.8574`) |
| **M2 no chol** | `max_depth=6`, `min_samples_leaf=50`, `max_features=sqrt` | `0.8365 ± 0.0216` | Reference (`OOF = 0.8355`) |
| **M3 no chol** | `max_depth=4`, `min_samples_leaf=20`, `max_features=sqrt` | `0.8370 ± 0.0207` | `+0.0004` (`0.8359 vs 0.8355`) |

---

## 3. Held-Out Test Set Performance (`n = 909`, `67` CHD/MI Cases)

*(Null Brier score predicting constant `7.4%` prevalence = `0.0683`; random PR-AUC baseline = `0.074`)*

| Feature Set | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test Brier Score (95% CI) |
| :--- | :---: | :---: | :---: |
| **M0** | `0.8195 (0.7797–0.8573)` | `0.1987 (0.1497–0.2671)` | `0.0622 (0.0503–0.0746)` |
| **M1** | `0.8270 (0.7870–0.8634)` | `0.2162 (0.1619–0.2977)` | `0.0617 (0.0493–0.0742)` |
| **M2** (Baseline Clinical) | `0.8824 (0.8425–0.9161)` | `0.4134 (0.3098–0.5461)` | `0.0563 (0.0450–0.0672)` |
| **M2 + NLR** | `0.8854 (0.8462–0.9219)` | `0.4191 (0.3082–0.5404)` | `0.0563 (0.0448–0.0675)` |
| **M2 + CRP** | `0.8811 (0.8406–0.9159)` | `0.4076 (0.3020–0.5370)` | `0.0565 (0.0451–0.0678)` |
| **M3** (Full Model) | `0.8868 (0.8472–0.9207)` | `0.4330 (0.3250–0.5516)` | `0.0563 (0.0449–0.0675)` |
| **M2 no chol** | `0.8593 (0.8191–0.8967)` | `0.3431 (0.2484–0.4694)` | `0.0587 (0.0471–0.0700)` |
| **M3 no chol** | `0.8578 (0.8156–0.8975)` | `0.3672 (0.2661–0.4980)` | `0.0586 (0.0467–0.0703)` |

---

## 4. Incremental Marker Gain (Paired Bootstrap ΔAUC)

| Comparison | ΔAUC on Test Set | 95% Bootstrap CI | `p (ΔAUC <= 0)` | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **M3 vs M2** (Primary RQ) | `+0.0044` | `-0.0051 to +0.0141` | `0.149` | Small positive point estimate, not statistically significant |
| **M2 + NLR vs M2** | `+0.0030` | `-0.0065 to +0.0132` | `0.278` | NLR accounts for the modest signal |
| **M2 + CRP vs M2** | `-0.0013` | `-0.0044 to +0.0017` | `0.809` | hs-CRP adds no incremental discrimination |
| **M3 no chol vs M2 no chol** | `-0.0015` | `-0.0114 to +0.0081` | `0.598` | Even without cholesterol, RF sees no AUC gain from markers |
| **M2 vs M2 no chol** | `+0.0231` | `+0.0003 to +0.0443` | `0.024` | Statin-confounded cholesterol adds significant discrimination (`p = 0.024`) |

### `M3` Clinical Operating Point (Training OOF Youden Threshold = `0.0883`)

- **Sensitivity (Recall):** `86.6%` (`0.8657`, 95% CI: `78.2%–94.5%`)
- **Specificity:** `74.2%` (`0.7423`, 95% CI: `71.0%–76.9%`)
- **PPV (Precision):** `21.1%` (`0.2109`, 95% CI: `16.5%–26.1%`)
- **NPV:** `98.6%` (`0.9858`, 95% CI: `97.6%–99.4%`)

### Class-Weight Sensitivity (`results/class_weight_sensitivity.csv`)

| Model | `class_weight` | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Brier |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `M2` | `None` | `0.8584` | `0.8824` | `0.4134` | `0.0563` |
| `M2` | `balanced_subsample` | `0.8587` | `0.8916` | `0.4159` | `0.1067` |
| `M3` | `None` | `0.8607` | `0.8868` | `0.4330` | `0.0563` |
| `M3` | `balanced_subsample` | `0.8613` | `0.8921` | `0.4231` | `0.0814` |

Using `balanced_subsample` distorts calibration (`Brier = 0.0814–0.1067`, worse than the null model `0.0683`) without changing the substantive conclusion (`M3 vs M2 ΔAUC = +0.0005` under `balanced_subsample`).

---

## 5. Model Explainability (`M3` Held-Out Test Set)

### Permutation Importance (`30` Repeats, Mean ROC-AUC Drop ± SD)

1. `total_chol`: `0.0390 ± 0.0079` *(strongest signal, driven by reverse causation / statin therapy: 79.8% of CHD cases vs 25.4% of controls take lipid-lowering medication)*
2. `age`: `0.0346 ± 0.0113`
3. `egfr`: `0.0174 ± 0.0063`
4. `hypertension`: `0.0143 ± 0.0066`
5. `diabetes`: `0.0117 ± 0.0047`
6. `smoking`: `0.0076 ± 0.0051`
7. `hdl_chol`: `0.0054 ± 0.0019`
8. **`log_nlr`:** `0.0044 ± 0.0022` *(small positive contribution)*
9. `male`: `0.0042 ± 0.0013`
10. `income_pir`: `0.0032 ± 0.0011`
11. `race_eth`: `0.0005 ± 0.0006`
12. **`log_crp`:** `0.0001 ± 0.0007` *(essentially zero)*
13. `bmi`: `-0.0002 ± 0.0012`

### Partial Dependence (`5th–95th` Percentiles)

- **`NLR` (`0.80` to `5.19`):** Average predicted CHD/MI probability increases monotonically from `6.78%` (`NLR = 0.80`) / `6.77%` (`NLR = 1.01`) to `8.92%` (`NLR = 5.19`), a modest `+2.1` percentage-point gradient.
- **`hs-CRP` (`0.22` to `19.15 mg/L`):** Average predicted probability remains flat between `7.37%` and `7.55%` across the entire clinical distribution.

---

## 6. Cross-Model Team Synthesis (`comparison/results_summary.csv` & `delta_auc.csv`)

All four prediction models are now harmonized on `class_weight=None` over the exact same `n = 909` test split:

| Model (Owner) | M0 ROC | M1 ROC | M2 ROC | M3 ROC | M3 PR-AUC | M3 Brier | ΔAUC (`M3 vs M2`, 95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **k-NN** (Julia) | `0.8123` | `0.8392` | `0.8802` | `0.8837` | `0.4418` | `0.0583` | `+0.0035 (-0.0054 to +0.0125)` |
| **Random Forest** (Marco) | `0.8195` | `0.8270` | `0.8824` | **`0.8868`** | `0.4330` | `0.0563` | `+0.0044 (-0.0051 to +0.0141)` |
| **Lasso** (Petya) | `0.8094` | `0.8272` | `0.8723` | `0.8751` | `0.4488` | `0.0531` | `+0.0027 (-0.0021 to +0.0080)` |
| **Ridge** (Petya) | `0.8131` | `0.8339` | `0.8808` | `0.8842` | **`0.4908`** | **`0.0515`** | `+0.0034 (-0.0015 to +0.0084)` |

**Conclusion across all paradigms:** Adding `log NLR` and `log hs-CRP` to standard clinical risk factors yields only a tiny, statistically non-significant discrimination gain (`ΔAUC = +0.0027` to `+0.0044`, every 95% CI crosses zero). Even though `log NLR` shows an independent statistical association in logistic regression (`OR = 1.18, p = 0.011`, Jannis), neither linear (`Lasso`, `Ridge`), local non-parametric (`k-NN`), nor tree-ensemble (`Random Forest`) models extract meaningful extra predictive accuracy beyond `M2`.

---

## 7. Generated Output Files (`notebooks/marco/results/`)

- `cv_results.csv`, `cv_delta_auc.csv`
- `test_performance.csv`, `delta_auc.csv`
- `threshold_metrics_m3.csv`, `class_weight_sensitivity.csv`
- `permutation_importance_m3.csv`, `partial_dependence_m3.csv`
- `figures/auc_by_feature_set.png`
- `figures/roc_pr_m2_m3.png`
- `figures/calibration_m3.png`
- `figures/permutation_importance_m3.png`
- `figures/partial_dependence_markers_m3.png`
