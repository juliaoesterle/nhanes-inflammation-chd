# Inflammatory Blood Markers and Coronary Heart Disease — NHANES 2021–2023

Group project for **Foundations of Data Science**, ETH Zürich (Dataset 5: Health and Nutrition).

> **Research question.** Among US adults, are a higher neutrophil-to-lymphocyte ratio (NLR) and
> higher hs-CRP levels associated with a self-reported history of coronary heart disease or
> myocardial infarction (CHD/MI), independent of established cardiovascular risk factors?
>
> *Secondary:* Do NLR and hs-CRP improve discrimination beyond established risk factors?

---

## Quick start

```bash
git clone https://github.com/juliaoesterle/nhanes-inflammation-chd.git
cd nhanes-inflammation-chd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put the two course files here (they are NOT in git):
#   data/raw/nhanes_raw.csv
#   data/raw/nhanes_metadata.csv

python prepare_data.py
```

Then, in any notebook or script:

```python
from prepare_data import load_data, REQUIRED_COVARIATES

df           = load_data()                       # complete-case analytic sample
train, test  = load_data(split=True)             # fixed 80/20 split; touch `test` ONCE, at the end
eligible     = load_data("eligible")             # incl. missing covariates -> imputation / selection bias
clean        = load_data("clean")                # all participants, all 763 cleaned raw columns
```

## Repository layout

```
prepare_data.py        shared cleaning + cohort script (the only place the cohort is defined)
requirements.txt
data/raw/              course CSVs (git-ignored)
data/processed/        parquet outputs of prepare_data.py (git-ignored)
reports/               flowchart.csv, cleaning_log.csv, missingness.csv (generated)
notebooks/<name>/      everyone's own experiments, e.g. notebooks/julia/01_eda.ipynb
```

**Rule of thumb:** experiment freely in your own `notebooks/<name>/` folder, but never redefine
the cohort or recode variables there. If you need a different choice, change `CONFIG` in
`prepare_data.py` in a branch / pull request and add it to the decision log (B6 on Overleaf).

---

## What `prepare_data.py` does

The course file is a merged NHANES export (11,933 participants × 763 columns) with **deliberately
injected dirt**. The script cleans every column against the documented codes in
`nhanes_metadata.csv`, so nothing is hard-coded per variable:

| Problem found in the raw file | Example | Handling |
|---|---|---|
| Random strings / timestamps in numeric columns (~125k cells, 67 columns) | `MCQ160E = 'xvUXYNA'`, `'2019-03-18T07:53:54'` | → `NaN` (`non_numeric`) |
| SAS-export artefact for exact zero | `5.397605346934028e-79` | → `0` |
| Non-response codes | `7/9`, `77/99`, `777/999` = Refused / Don't know | → `NaN` (`non_response`) |
| Undocumented sentinel values | `123456789`, `-1000000`, `10000000`, `999999` | value is neither a documented code nor inside the documented range → `NaN` (`out_of_range`) |

Every replaced cell is counted per column in `reports/cleaning_log.csv`.

Then:

1. **Analysis variables** (row-wise only, so no leakage): composite outcome `chd_mi`
   (MCQ160C or MCQ160E; angina optional), `nlr` = neutrophil % / lymphocyte %, `log_nlr`, `log_crp`,
   `crp_below_lod`, `crp_above_10`, smoking (never/former/current), diabetes, hypertension
   (self-report or measured BP), cholesterol, HDL, eGFR (CKD-EPI 2021), lipid/BP medication,
   survey weight (`weight_mec` = WTMEC2YR — note: `WTPH2YR` in this file only covers females 12–49, see script)/strata/PSU.
2. **Cohort selection** with a STROBE flowchart: all → age ≥ 20 → outcome available →
   NLR + hs-CRP available (= *eligible sample*) → complete covariates (= *analytic sample*).
3. **Fixed split**: stratified 80/20 on outcome × complete-case status, `random_state=42`, stored as
   a `split` column so everyone uses the same test participants.

Current flowchart with default `CONFIG`:

| Step | n | CHD/MI cases | Sample prevalence |
|---|---:|---:|---:|
| All NHANES 2021–2023 participants | 11,933 | 559 | 4.7% |
| Age ≥ 20 years | 7,809 | 559 | 7.2% |
| Outcome (CHD/MI) available | 7,773 | 559 | 7.2% |
| NLR and hs-CRP available → **eligible sample** | 5,538 | 412 | 7.4% |
| Complete covariates → **analytic sample** | 4,544 | 337 | 7.4% |

Most of the complete-case loss comes from the income-to-poverty ratio (12.7% missing in the eligible
sample), which makes imputation (or dropping income) a worthwhile sensitivity analysis.

All analysis choices (age cut-off, angina, BP thresholds, borderline diabetes, CRP > 10 exclusion,
seed) live in the `CONFIG` dict at the top of the file.

**Not** done in this script, on purpose: scaling, imputation, one-hot encoding, outlier removal,
resampling. These must be fitted on the training data only, inside a scikit-learn `Pipeline`.

---

## Methods we can explore

A menu to pick from — the primary analysis (logistic regression, nested adjustment) is fixed in the
report outline; everything else is a playground. Grab one, put your name next to it in the task
split (B7), and work in `notebooks/<name>/`.

### 1. Exploratory data analysis
- Distributions of `crp` and `nlr` on raw vs. log scale; Q-Q plots; how many `crp_below_lod`.
- Marker levels by age, sex, race/ethnicity, smoking, BMI category — and by `lipid_med`
  (statins lower CRP: do treated cases have *lower* CRP than untreated controls?).
- Correlation heatmap of all candidate covariates (Spearman), pairwise scatterplots.
- **Table 1** by CHD/MI status (mean (SD) / median [IQR] / n (%), with tests).
- Data-quality view: `reports/cleaning_log.csv` and `reports/missingness.csv` as figures.

### 2. Missing data and selection bias
- Missingness patterns (`missingno`-style matrix, co-missingness): is bloodwork missing at random?
- **Included vs. excluded** comparison (age, sex, outcome rate) using `eligible["complete_case"]`.
- Logistic model *predicting* missingness (MAR plausibility).
- **Multiple imputation** (`IterativeImputer` with `sample_posterior=True`, or `statsmodels.imputation.mice`)
  vs. complete-case — do the ORs change?

### 3. Inferential models (answer H1a / H1b)
- Nested logistic regression M0 → M3, OR per SD of `log_nlr` / `log_crp` with 95% CI (statsmodels).
- Alternative effect scales: per doubling, per quartile (Q4 vs. Q1), clinical CRP categories (<1, 1–3, >3 mg/L).
- **Restricted cubic splines** / GAM for the dose–response shape (Figure 2).
- **Interactions / effect modification**: marker × `lipid_med`, × sex, × age group.
- **Survey-weighted** logistic regression (`weight_mec`, `strata`, `psu`) vs. unweighted.
- Multiple-testing handling (Bonferroni for two co-primary markers).
- Diagnostics: VIF, influential points, linearity in the logit, Hosmer–Lemeshow / calibration.

### 4. Confounding and causal thinking
- Draw a **DAG** (e.g. dagitty) for inflammation → CHD; justify the adjustment set.
- **Propensity-score** approaches for "high NLR" (top tertile) vs. rest: matching or IPW, balance plots.
- **E-value** for unmeasured confounding.
- Discuss reverse causation: are markers in people diagnosed long ago different? (We can't time it, but
  we can stratify by `lipid_med` as a proxy for "managed disease".)

### 5. Predictive modelling (answer H1c)
Same nested feature sets M0–M3 for all models, preprocessing inside `Pipeline`, stratified 5-fold CV
on the training set only. **No medication variables as features** (leakage).
- Baselines: `DummyClassifier`, logistic regression on age + sex.
- Penalised LR (L1 / L2 / elastic net) — robustness and feature selection.
- Random forest, `HistGradientBoostingClassifier` (handles NaN natively → compare with the eligible
  sample without imputation), optionally XGBoost / LightGBM.
- Optional: SVM, k-NN, small MLP — probably won't beat boosting, but a nice comparison.
- **Class imbalance** (~7% cases): `class_weight="balanced"` vs. none vs. SMOTE *inside* folds;
  check what it does to calibration.
- Hyperparameter tuning: `GridSearchCV` / `RandomizedSearchCV`, nested CV when comparing many models.

### 6. Evaluation
- ROC-AUC (primary), **PR-AUC** (baseline ≈ prevalence), Brier score, calibration curves
  (`CalibrationDisplay`), possibly Platt / isotonic recalibration.
- **ΔAUC (M3 − M2)** with ≥ 1,000 bootstrap resamples; DeLong test.
- Threshold-based metrics at a pre-chosen threshold; **decision-curve analysis** (net benefit).
- Learning curves: do we have enough cases?

### 7. Interpretation
- Permutation importance on the test set (not impurity importance — biased toward continuous features).
- **SHAP** summary and dependence plots for NLR and CRP in the tree models.
- Partial dependence / ICE plots — compare with the spline curves from the inferential model.

### 8. Sensitivity analyses
- Exclude hs-CRP > 10 mg/L (`CONFIG["exclude_crp_above_10"] = True`).
- Untreated participants only (`lipid_med == 0`; underpowered → report CIs).
- Outcome variants: CHD only, MI only, + angina (`CONFIG["include_angina"]`).
- Hypertension threshold 130/80 vs. 140/90; borderline diabetes as yes.
- NLR from absolute counts (`LBDNENO / LBDLYMNO`) vs. percentages.
- Additional markers from the CBC: WBC, monocyte-to-lymphocyte ratio, platelet-to-lymphocyte ratio, SII.

### 9. Unsupervised / exploratory extras
- PCA / UMAP of the CBC + metabolic panel, coloured by CHD/MI.
- Clustering (k-means, Gaussian mixtures) into "inflammatory phenotypes", then compare CHD prevalence.

---

## Reproducibility checklist
- Raw data is read-only; all figures are generated by code.
- Fixed seeds (`CONFIG["random_state"]`), pinned `requirements.txt`.
- "Restart & Run All" before pushing a notebook; clear large outputs.
- Test set evaluated **once**, at the end.
- Reporting follows STROBE (observational part) and TRIPOD+AI (prediction part).

## Data
NHANES August 2021 – August 2023, CDC / National Center for Health Statistics,
<https://www.cdc.gov/nchs/nhanes/>. The course-provided CSVs are not redistributed in this repository.
