# Technical documentation: NLR, hs-CRP and coronary heart disease in NHANES 2021–2023

*Technical reference for the pipeline in `NewMarco/`. The step-by-step explanation in plain language (German) is [ANLEITUNG_SCHRITT_FUER_SCHRITT.md](ANLEITUNG_SCHRITT_FUER_SCHRITT.md); the report draft is [../report/REPORT_DRAFT.md](../report/REPORT_DRAFT.md).*

## Contents

0. [Summary](#0-summary)
1. [What was wrong with the first draft](#1-what-was-wrong-with-the-first-draft)
2. [Methods and the course](#2-methods-and-the-course)
3. [Data, cleaning and sample](#3-data-cleaning-and-sample)
4. [Decision records](#4-decision-records)
5. [Pipeline](#5-pipeline)
6. [Results](#6-results)
7. [Limitations](#7-limitations)
8. [Reproducibility](#8-reproducibility)
9. [References](#9-references)

---

## 0. Summary

**Question (group README).** Among US adults, are higher NLR and hs-CRP associated with a self-reported history of coronary heart disease or myocardial infarction (CHD/MI), independent of established risk factors (H1a NLR, H1b hs-CRP)? Secondary (H1c): do they improve discrimination beyond established risk factors, and does that depend on the algorithm?

**Answer.**
- **H1a supported:** NLR OR 1.20 per SD of log NLR (95% CI 1.07–1.34, p = 0.002; likelihood-ratio p = 0.002).
- **H1b not supported:** hs-CRP OR 0.96 (0.84–1.08, p = 0.48). The crude hs-CRP association is accounted for by BMI.
- **H1c not supported:** logistic regression AUC 0.861 with established risk factors, +0.0013 (−0.0009 to +0.0035) with both markers. No significant gain for any of the six algorithms, and none beats logistic regression.
- **Robustness:** all conclusions hold in 13 sensitivity analyses.

**Approach.** Every decision compares three (or, where the course taught more, all) options against a rule written in `config.py` before the corresponding experiment. All methods come from the course (W1–W9, tutorials) except for a few small, explained additions (Section 2).

---

## 1. What was wrong with the first draft

| # | Problem | Fix |
|---|---|---|
| 1 | Feature sets not justified | Reference set from validated risk scores (D5); E2 shows what naive selection does |
| 2 | Complete-case analysis dropped 975 people (70 cases) | Median imputation + missing indicator, compared with the other W2 options (E1) |
| 3 | One 80/20 split; test set with 67 cases looked at repeatedly | Repeated nested CV, 15 test folds (D7, E4) |
| 4 | Tuning on AUC, partly with class weights → broken calibration | Tuning on log-loss (D10); class correction only by rule (E3) |
| 5 | Model comparison not tested | Paired t-test over folds with overlap correction, Bonferroni (D8) |
| 6 | Association only complete-case, no survey design | Three missing-data options and a survey-weighted analysis (D3, D13) |
| 7 | No sample-size consideration | Events per parameter (D18) |
| 8 | Inconsistent methods between team members | One pipeline, identical folds for all algorithms |
| 9 | Outcome definition not discussed | Three outcome variants as sensitivity analyses (D2) |
| 10 | Statin problem noticed but not handled | Model 2 primary, Model 3 / untreated / no-cholesterol analyses (D4, D17) |
| 11 | Accuracy reported at threshold 0.5 | Threshold chosen in the training part; accuracy paradox shown (E3, D8) |
| 12 | Physical-activity units lost in cleaning | Units read separately (`01_prepare_data.py`) |
| 13 | Selection-bias analysis unchecked | Recomputed with SMDs (`02_descriptives.py`) |

---

## 2. Methods and the course

| Step | Method | Course |
|---|---|---|
| Research question, hypotheses | focused / testable / feasible / meaningful | W1 |
| Missing data | MCAR/MAR/MNAR; listwise deletion, dropping a column, median imputation; "missing data can hold information" (indicator) | W2 |
| Outliers | IQR rule | W2 |
| Distributions | skewness, Q-Q plot, log transform; normality tests not used as criterion at large n | W3 |
| Group comparison | Mann-Whitney U, chi-square, Bonferroni | W3, tutorial 4 |
| Sampling | representativeness, survey weights; sample-size justification (checklist) | W3 |
| ML design | train/test, CV, overfitting, simpler models first, scaling (checklist) | W4 |
| Regularisation, feature selection | ridge/lasso/elastic net; filter / wrapper / embedded, RFE | W4–W6 |
| Tuning | GridSearchCV, nested CV | W5, tutorials 7–8 |
| Classifiers | logistic regression, decision tree, kNN, random forest, SVM | W6–W9 |
| Logistic regression inference | odds ratio, Wald test, likelihood, VIF, interaction | W7 |
| Metrics | confusion matrix, sensitivity, specificity, threshold, ROC/AUC, PR curve, calibration | W5, W6, W9 |
| Imbalance | accuracy paradox, class weights, over-/undersampling, SMOTE | W9, tutorial 8 |
| Model comparison | paired t-test over the same folds | W3, tutorial 4 |

**Small additions beyond the course (each explained in the guide):**
- the variance correction of the paired t-test for overlapping training sets (Nadeau & Bengio 2003);
- Platt scaling (a logistic regression on SVM scores);
- the likelihood-ratio test (difference of log-likelihoods);
- design-based standard errors for the survey-weighted sensitivity analysis;
- permutation importance and partial dependence for interpretation.

---

## 3. Data, cleaning and sample

- **Source:** NHANES 2021–2023 course export (11,933 participants × 763 columns), cleaned by the team's `prepare_data.py` (`clean_raw`). `01_prepare_data.py` builds the analysis variables.
- **Inclusion:** age ≥ 20 → CHD/MI questions answered → MEC exam attended → NLR and hs-CRP measured: **5,538 adults, 412 cases (7.4%)**. The old complete-case sample had 4,563 adults (342 cases).
- **Outcome:** MCQ160C (CHD) or MCQ160E (MI). Variants: MI only, CHD only, including angina (MCQ160D).
- **Markers:** NLR = neutrophils / lymphocytes (percentages; absolute counts as sensitivity); hs-CRP in mg/L; both log-transformed.
- **Covariates:**
  - age, sex, race/ethnicity (6 groups), income-to-poverty ratio, BMI;
  - smoking (never/former/current), diabetes;
  - hypertension (diagnosed or mean BP ≥ 140/90 mmHg), total and HDL cholesterol;
  - eGFR (CKD-EPI 2021 without race), lipid-lowering and BP medication.
- **Knowledge-screened pool (E2, 39 variables):** M3 plus education, waist, BP, pulse, lipids, glucose, uric acid, albumin, liver enzymes, albuminuria, cotinine, blood count, vitamin D, sleep, sedentary time, physical activity, alcohol and PHQ-9.
- **Selection:** people without a blood draw are poorer (SMD for income −0.36) and more often non-Hispanic Black (`fig06`).

---

## 4. Decision records

Each record: options → choice → evidence. The empirical rules are written in `config.py`.

| ID | Decision | Options | Choice | Evidence |
|---|---|---|---|---|
| D1 | Question / estimand | prediction of future events · detection as main question · **association primary, detection secondary** | association primary | Cross-sectional data cannot predict; detecting known diagnoses is not meaningful on its own (W1, W4 checklist); aligned with the group README |
| D2 | Outcome | **CHD or MI** · MI only · incl. angina | CHD or MI | Group definition; variants as sensitivity |
| D3 | Missing covariates (E1) | listwise deletion · drop income column · **median + missing indicator** | median + indicator | Same test persons: AUC 0.866 / 0.867 / 0.867; keeps 975 people (70 cases); missingness unrelated to outcome (OR 1.03) |
| D4 | Treatment variables | ignore · **adjust (Model 3) as sensitivity** · untreated only | sensitivity analyses | Treatment is a consequence of the diagnosis (78% vs. 25%) |
| D5 | Reference feature set (E2 checks it) | **literature (risk scores)** · all variables + lasso · knowledge pool + filter / wrapper / lasso | literature | Fixed by the research question; naive lasso AUC 0.923 through leakage (35% of the top 20 variables); hybrid +0.008, less stable (Jaccard 0.69–0.86) |
| D6 | Marker scale | raw · **log** · categories | log (+ quartiles shown) | Skewness 4.66 / 7.78 → 0.15 / 0.23; squared term p = 0.67 / 0.79 |
| D7 | Evaluation (E4) | single split · CV · **repeated nested CV** | repeated nested CV | Same model on 200 splits: AUC 0.80–0.92; ΔAUC < 0 in 29.5% |
| D8 | Model comparison | by eye · plain paired t-test · **paired t-test with overlap correction** | corrected paired t-test + Bonferroni | Folds share 75% of their training data |
| D9 | Class imbalance (E3) | **none** · class weights · oversampling · undersampling · SMOTE | none | AUC 0.856–0.863 for all; Brier 0.058 vs. 0.094–0.167; mean predicted risk 22–38% vs. 7.4%. *Post hoc exception:* SVM with balanced weights (below) |
| D10 | Tuning objective | AUC · **log-loss** · Brier | log-loss | Negative log-likelihood rewards ranking and calibration |
| D11 | Algorithms | **six course classifiers** (LR, penalised LR, tree, kNN, SVM, RF) | all six | W4 checklist: simple before complex |
| D12 | Final model | **LR unless another is significantly better** | LR | SVM +0.004 (−0.002 to +0.010, Bonferroni p = 0.74); tree −0.058 (p = 0.002) |
| D13 | Survey weights | unweighted · weighted everywhere · **weighted sensitivity** | unweighted primary | Results describe the sample; survey-weighted ORs agree (NLR 1.18, hs-CRP 0.91) |
| D14 | Outliers | **keep** · IQR rule · hs-CRP > 10 mg/L | keep | Plausible values; both exclusions as sensitivity |
| D15 | Interpretation | impurity importance · LR coefficients · **permutation importance (+ coefficients)** | permutation importance | Works for every algorithm, measured on test folds |
| D16 | Hypertension | **diagnosed or ≥ 140/90** · continuous BP · 130/80 | diagnosed or ≥ 140/90 | Captures treated hypertension without a treatment variable |
| D17 | Primary association model | Model 1 · **Model 2** · Model 3 (+ lipid drugs) | Model 2 | Avoids adjusting for a consequence of the outcome; Model 3 as sensitivity |
| D18 | Sample size | none · **events per parameter ≥ 10** · formal calculation | EPV | 19 coefficients: 21.7 cases per coefficient (17.3 per outer training fold) |

**SVM exception (post hoc, D9).** The first run without class weights gave AUC 0.757 (M2) / 0.765 (M3), and the inner CV picked the most flexible grid corner (C = 10, γ = 0.08) in 57/60 folds. The hinge loss at 7.4% prevalence collapses towards "no disease". A check on one split confirmed it: AUC 0.71–0.78 unweighted vs. 0.83–0.89 balanced. Platt scaling restores calibration, so the E3 argument does not apply. New grid C ∈ {0.03, 0.1, 0.3, 1}, γ ∈ {0.002, 0.005, 0.02}; AUC 0.867. The first run is archived in `results/cv/*_previous_SVM.*`. It had shown the largest marker gain (+0.008), an artefact of its weak baseline.

**E2 selection frequency** (`e2_selection_frequency.csv`): NLR is selected in every fold by filter, wrapper and lasso; hs-CRP in 0% (filter), 90% (wrapper) and 100% (lasso).

---

## 5. Pipeline

| Script | Content | Outputs |
|---|---|---|
| `config.py` | feature sets, options, decision rules, labels | – |
| `common.py` | pipelines, model zoo and grids, metrics, corrected t-test, Bonferroni, SMOTE, survey logit, figure style | – |
| `01_prepare_data.py` | cleaning, variables, sample flow | `data/*.parquet`, `flowchart.csv`, `missingness.csv` |
| `02_descriptives.py` | Table 1 (Mann-Whitney / chi-square, Bonferroni, SMD), selection bias, missingness model and patterns, distributions, EPV | `table1.csv`, `sample_size_epv.csv`, `fig01`–`fig07` |
| `03_design_experiments.py` | E1–E4 with decision rules | `e1`–`e4` tables, `decisions/*.json`, `fig08`–`fig11` |
| `04_nested_cv.py` | 6 algorithms × {M2, M3, M2+NLR, M2+CRP} (+ LR M0, M1, M4, NLR alone, hs-CRP alone), 15 outer folds, inner grid search; `python 04_nested_cv.py SVM` re-runs single algorithms | `cv/fold_metrics.csv`, `cv/oof_predictions.parquet`, permutation importance |
| `05_evaluate_models.py` | corrected CIs and tests, Bonferroni, final-model rule, thresholds, calibration, ROC/PR | `main_*.csv`, `fig12`–`fig16` |
| `06_association.py` | logistic regression (Models 1–3), three missing-data options, survey weights, LR test, squared term, quartiles, interactions, VIF, decomposition, full model, sensitivity analyses | `assoc_*.csv`, `sensitivity_*.csv`, `fig18`–`fig20` |
| `07_interpretation.py` | permutation importance across algorithms, partial dependence (LR, RF), depth-3 tree, learning curves | `fig21`–`fig24` |

**Leakage protection.** Imputation, missing indicator, scaling and one-hot encoding are pipeline steps, refitted in every training fold (including inner folds). Over-/undersampling and SMOTE are applied to the training part only; SMOTE runs after imputation and scaling. Thresholds are chosen on out-of-fold predictions within the training part.

**Corrected paired t-test.** For J = 15 per-fold differences d: t = d̄ / √((1/J + n_test/n_train) · s²), with n_test/n_train = 1/4 and J − 1 df.

**Association models.** statsmodels `Logit`; OR = exp(β) per 1 SD of the log marker; Wald p; Bonferroni α = 0.025 per marker. Likelihood-ratio test: LR = 2(ℓ₁ − ℓ₀) ~ χ²(df = number of added markers). Survey-weighted: weighted pseudo-likelihood with Taylor-linearised variance (15 strata, 30 PSUs, 15 df).

---

## 6. Results

### 6.1 Sample

Cases are older (median 70 vs. 55 years), more often male (65% vs. 44%), diabetic (37% vs. 12%) and hypertensive (78% vs. 42%). NLR is higher in cases (2.39 vs. 1.96; SMD 0.42); hs-CRP hardly differs (1.97 vs. 1.81 mg/L; SMD 0.02; Bonferroni p = 1.00). Total cholesterol is lower in cases (152 vs. 187 mg/dL) because 78% take lipid-lowering drugs (statin paradox, `fig03`). Spearman ρ(NLR, hs-CRP) = 0.15.

### 6.2 Association (H1a, H1b)

| Model / analysis | NLR OR (95% CI), p | hs-CRP OR (95% CI), p |
|---|---|---|
| Model 1 | 1.30 (1.17–1.44), < 0.001 | 1.11 (0.99–1.24), 0.066 |
| **Model 2 (primary)** | **1.20 (1.07–1.34), 0.002** | **0.96 (0.84–1.08), 0.48** |
| Model 3 (+ lipid drugs) | 1.21 (1.08–1.36), 0.001 | 0.99 (0.87–1.13), 0.92 |
| Model 2, complete cases (n = 4,544) | 1.18 (1.04–1.34), 0.010 | 0.96 (0.83–1.11), 0.58 |
| Model 2, income dropped | 1.20 (1.08–1.35), 0.001 | 0.97 (0.85–1.10), 0.59 |
| Model 2, survey-weighted | 1.18 (1.05–1.33), 0.008 | 0.91 (0.81–1.02), 0.10 |

- **Likelihood-ratio test vs. Model 2:** NLR χ² = 9.83 (p = 0.002); hs-CRP 0.50 (p = 0.48); both 11.52 (2 df, p = 0.003).
- **Shape:** squared term p = 0.67 (NLR), 0.79 (hs-CRP). Quartiles Q2–Q4 vs. Q1: NLR 1.08 / 1.38 / 1.33; hs-CRP 0.83 / 0.90 / 0.85. Clinical hs-CRP categories 1–3 and > 3 mg/L: 1.01 and 0.94.
- **Effect modification:** interaction p: NLR × lipid drugs 0.95, NLR × sex 0.22, hs-CRP × lipid drugs 0.64, hs-CRP × sex 0.70. Maximum VIF 2.16.
- **Full model** (`assoc_full_model.csv`): age 2.55 per SD, current smoking 2.70, hypertension 2.05, diabetes 1.79, male 1.74, total cholesterol 0.58 per SD (treatment effect).
- **Decomposition (exploratory, post hoc):** Model 1 + one covariate group. hs-CRP: 1.11 → BMI 1.00 (0.88–1.13), HDL 1.02, other single groups 1.06–1.12, lipid drugs 1.15 (1.02–1.29). NLR: 1.30 → single groups 1.26–1.32 → all 1.20.

### 6.3 Discrimination (H1c)

- **LR feature ladder:** M0 0.801 → M1 0.815 → M2 0.861 → M3 0.863.
- **Markers alone (reference models suggested by the TA, LR only):** M4 (NLR + hs-CRP) 0.631 (0.600–0.662), NLR alone 0.633, hs-CRP alone 0.506 (chance). NLR has a weak standalone signal that is largely shared with the risk factors; hs-CRP has none.

| Algorithm | AUC M2 | AUC M3 | ΔAUC (95% CI) | Bonferroni p | vs. LR (M3), Bonferroni p |
|---|---|---|---|---|---|
| Logistic regression | 0.861 | 0.863 | +0.0013 (−0.0009, +0.0035) | 1.00 | – |
| Penalised LR | 0.862 | 0.863 | +0.0015 (−0.0008, +0.0038) | 1.00 | 1.00 |
| Decision tree | 0.808 | 0.805 | −0.0030 (−0.0123, +0.0064) | 1.00 | 0.002 (worse) |
| kNN | 0.859 | 0.863 | +0.0039 (−0.0067, +0.0144) | 1.00 | 1.00 |
| SVM (RBF) | 0.867 | 0.867 | +0.0003 (−0.0025, +0.0030) | 1.00 | 0.74 |
| Random forest | 0.856 | 0.859 | +0.0022 (−0.0055, +0.0099) | 1.00 | 1.00 |

- **Marker gains:** NLR alone −0.002 to +0.004; hs-CRP alone −0.0004 to +0.003. ΔBrier < 0.001.
- **Train–test AUC gap:** LR 0.008, penalised LR 0.007, SVM 0.019, tree 0.033, RF 0.093, kNN 0.137 (artefact of distance weighting).
- **Calibration:** good for LR, penalised LR, SVM; RF and kNN too moderate, tree too extreme.
- **Threshold (LR M3):** sensitivity 0.827, specificity 0.750, PPV 0.211, NPV 0.982; "always no CHD": accuracy 0.926, sensitivity 0.
- **Learning curves:** flat beyond about 2,500 training persons.
- **Importance:** age ≫ total cholesterol > hypertension, smoking, diabetes, eGFR; NLR rank 6–9; hs-CRP last for five of six algorithms. The depth-3 tree uses age, diabetes, cholesterol and eGFR.

### 6.4 Sensitivity

- **Association (13 variants):** NLR OR 1.14–1.28, CI excludes 1 in 12 (untreated: 1.21, 0.99–1.49, 96 cases); hs-CRP 0.86–1.00, never significant.
- **Prediction (10 variants):** no significant gain. Without cholesterol, the reference AUC drops from 0.861 to 0.849 (treatment-distorted cholesterol carries 0.013), and the marker gain stays +0.002.

---

## 7. Limitations

| Threat | Consequence | Handling |
|---|---|---|
| Cross-sectional design | reverse causation (infarct raises NLR; statins lower CRP) | framed as association; Model 3, untreated-only |
| Self-reported outcome | misclassification, undiagnosed disease missing | three outcome variants |
| Treatment distorts predictors | cholesterol inverse (0.58 per SD), CRP lowered | Model 2 primary; no-cholesterol and treatment-as-predictor variants |
| Single hs-CRP measurement | regression dilution | exclusion of hs-CRP > 10 mg/L |
| Exclusion without blood draw | limited generalisability | documented with SMDs |
| Unweighted primary analysis | describes the sample | survey-weighted sensitivity |
| Single median imputation | understates uncertainty slightly | complete-case and column-dropping analyses agree |
| Small grids | optimum possibly outside (SVM γ at upper edge, tree at simple edge) | single-split check for SVM; simple-edge choices are benign |
| Post hoc steps | researcher degrees of freedom | SVM weights and decomposition labelled; first SVM run archived |
| Internal validation only | transportability unknown | – |

---

## 8. Reproducibility

- **Environment:** Python 3.11.16; exact versions in `NewMarco/requirements.txt` (scikit-learn 1.9.1, statsmodels 0.15.0, pandas 3.1.0, NumPy 2.4.6, SciPy 1.17.1, matplotlib 3.11.2, pyarrow 26.0.0, patsy 1.0.3). Older scikit-learn (< 1.8) silently changes `LogisticRegression(l1_ratio=1)` from lasso to ridge.
- **Seeds:** `config.SEED = 42`; all folds come from the seed and are identical for every algorithm and feature set.
- **Data:** participant-level data are not committed (`data/` is git-ignored).
- **Run:**

```bash
cd NewMarco
pip install -r requirements.txt
python run_all.py              # all steps
python run_all.py 05 06        # selected steps
python 04_nested_cv.py SVM     # re-run one algorithm and merge into the CV outputs
```

| Step | Run time (20 cores) |
|---|---|
| 01–02 | ~1 min |
| 03 E1–E4 | ~3 min |
| 04 nested CV | ~60 min (SVM ~25 min) |
| 05 | < 1 min |
| 06 | < 1 min |
| 07 | ~1 min |

Every empirical decision writes a JSON file with rule, numbers and decision to `results/decisions/`; downstream scripts read these files.

---

## 9. References

*Clinical background*
- Goff DC Jr, et al. 2013 ACC/AHA guideline on the assessment of cardiovascular risk. *Circulation* 2014;129(Suppl 2):S49–S73.
- Khan SS, et al. Development and validation of the American Heart Association's PREVENT equations. *Circulation* 2024;149:430–449.
- SCORE2 working group and ESC Cardiovascular Risk Collaboration. SCORE2 risk prediction algorithms. *Eur Heart J* 2021;42:2439–2454.
- Hippisley-Cox J, Coupland C, Brindle P. Development and validation of QRISK3 risk prediction algorithms. *BMJ* 2017;357:j2099.
- Yusuf S, et al. Effect of potentially modifiable risk factors associated with myocardial infarction in 52 countries (INTERHEART). *Lancet* 2004;364:937–952.
- Inker LA, et al. New creatinine- and cystatin C-based equations to estimate GFR without race. *N Engl J Med* 2021;385:1737–1749.
- World Health Organization. The top 10 causes of death. Fact sheet, 2024.

*Inflammation markers*
- Ridker PM, et al. Antiinflammatory therapy with canakinumab for atherosclerotic disease (CANTOS). *N Engl J Med* 2017;377:1119–1131.
- Adamstein NH, et al. The neutrophil–lymphocyte ratio and incident atherosclerotic events. *Eur Heart J* 2021;42:896–903.
- Emerging Risk Factors Collaboration. C-reactive protein concentration and risk of coronary heart disease, stroke, and mortality. *Lancet* 2010;375:132–140.
- Emerging Risk Factors Collaboration. C-reactive protein, fibrinogen, and cardiovascular disease prediction. *N Engl J Med* 2012;367:1310–1320.
- C Reactive Protein Coronary Heart Disease Genetics Collaboration. Mendelian randomisation analysis. *BMJ* 2011;342:d548.
- Pearson TA, et al. Markers of inflammation and cardiovascular disease (CDC/AHA statement). *Circulation* 2003;107:499–511.
- Forget P, et al. What is the normal value of the neutrophil-to-lymphocyte ratio? *BMC Res Notes* 2017;10:12.
- Okura Y, et al. Agreement between self-report questionnaires and medical record data. *J Clin Epidemiol* 2004;57:1096–1103.

*Methods*
- Nadeau C, Bengio Y. Inference for the generalization error. *Mach Learn* 2003;52:239–281.
- Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 2006;7:91.
- Chawla NV, et al. SMOTE: synthetic minority over-sampling technique. *J Artif Intell Res* 2002;16:321–357.
- Akbani R, Kwek S, Japkowicz N. Applying support vector machines to imbalanced datasets. *ECML* 2004, LNCS 3201:39–50.
- Platt J. Probabilistic outputs for support vector machines. In: *Advances in Large Margin Classifiers*. MIT Press, 1999.
- van den Goorbergh R, et al. The harm of class imbalance corrections for risk prediction models. *J Am Med Inform Assoc* 2022;29:1525–1534.
- Pepe MS, et al. Limitations of the odds ratio in gauging the performance of a diagnostic, prognostic, or screening marker. *Am J Epidemiol* 2004;159:882–890.
- Christodoulou E, et al. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *J Clin Epidemiol* 2019;110:12–22.
- Strobl C, et al. Bias in random forest variable importance measures. *BMC Bioinformatics* 2007;8:25.
- Groenwold RHH, et al. Missing covariate data in clinical research: when and when not to use the missing-indicator method. *CMAJ* 2012;184:1265–1269.
- Binder DA. On the variances of asymptotically normal estimators from complex surveys. *Int Stat Rev* 1983;51:279–292.

*Check bibliographic details against the original sources before reuse.*
