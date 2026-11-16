# Inflammatory Biomarkers and Coronary Heart Disease in NHANES 2021–2023

**ETH Zürich — Foundations of Data Science (Course Project, Dataset 5)**  
*Project Grade Weight: 50% (6 ECTS)*

---

## Executive Summary

This repository contains a modular, scientifically rigorous, and fully reproducible machine learning analysis evaluating whether systemic inflammatory blood markers—specifically the **Neutrophil-to-Lymphocyte Ratio (NLR)** and **high-sensitivity C-reactive Protein (hs-CRP)**—are associated with prevalent coronary heart disease (CHD / myocardial infarction) and whether they improve risk discrimination beyond established cardiovascular risk factors.

Every analytical choice follows an **explicit decision protocol** comparing three candidate options against pre-specified criteria strictly aligned with lectures W1–W9 of the ETH curriculum.

### Core Scientific Findings

| Research Question / Hypothesis | Statistical & ML Finding | Verdict & Scientific Interpretation |
|---|---|---|
| **H1a (Primary):** NLR association beyond risk factors | Adjusted Odds Ratio = **1.20 per SD** (95% CI: 1.07–1.34, p = 0.002 < 0.025) | **Supported.** Higher systemic neutrophil activity is independently associated with CHD/MI. |
| **H1b (Primary):** hs-CRP association beyond risk factors | Adjusted Odds Ratio = **0.96 per SD** (95% CI: 0.84–1.08, p = 0.48) | **Not Supported.** Crude association disappears once adjusting for BMI (hs-CRP reflects adiposity). |
| **H1c (Secondary):** Added discrimination beyond established risk factors | Logistic Regression $\Delta$AUC = **+0.0013** (95% CI: −0.0009 to +0.0035, p = 1.00) | **Not Supported.** Markers fail to improve individual classification beyond established risk factors. |
| **Algorithm Comparison:** Do complex ML models outperform Logistic Regression? | SVM $\Delta$AUC = +0.0041 (p = 0.74); RF $\Delta$AUC = −0.0041; Tree $\Delta$AUC = −0.0575 (p = 0.002) | **No complex ML model outperforms Logistic Regression.** Simple, interpretable models remain optimal. |
| **Sensitivity & Robustness:** Stability across modeling choices | Confirmed across **13 pre-specified sensitivity analyses** | Findings are invariant to missing data strategies, sampling weights, or diagnostic variations. |

---

## Repository Structure

```
newagy/
├── README.md                      # Project overview and quickstart guide
├── requirements.txt               # Pinned library dependencies
├── config.py                      # Central configuration, feature sets, paths, and decision rules
├── run_all.py                     # Master execution pipeline
│
├── modules/                       # Reusable, modular Python library
│   ├── __init__.py
│   ├── data_processing.py         # Codebook cleaning, feature calculation, cohort selection
│   ├── pipeline_factory.py        # Leakage-free ColumnTransformers, Imputation + Indicator
│   ├── models.py                  # 6 ML algorithms (LR, EN, DT, KNN, SVM, RF) & tuning grids
│   ├── evaluation.py              # Discrimination metrics, Nadeau-Bengio paired t-tests, Bonferroni
│   └── stats_analysis.py          # Multivariable logistic regression, ORs, LR-tests, VIF, decomposition
│
├── scripts/                       # Sequential workflow scripts
│   ├── 01_prepare_cohort.py       # Raw cleaning & cohort derivation (n = 5,538)
│   ├── 02_exploratory_analysis.py # Table 1, Mann-Whitney U, Chi-square, SMD, EPV sample size
│   ├── 03_design_experiments.py   # 4 design experiments (E1 missing data, E2 leakage, E3 imbalance, E4 split)
│   ├── 04_run_nested_cv.py        # 15-fold repeated nested cross-validation (5x3 outer, 5 inner)
│   ├── 05_model_evaluations.py    # Discrimination metrics, marker gain, statistical model comparison
│   ├── 06_association_analysis.py # Multivariable logistic regression (M1, M2, M3), LR tests, sensitivity
│   └── 07_interpretability.py     # Permutation feature importance, partial dependence, decision tree
│
├── data/                          # Analysis cohort parquet files (git-ignored)
├── results/                       # Generated outputs
│   ├── tables/                    # CSV outputs (Table 1, flowchart, performance, sensitivity)
│   ├── figures/                   # Publication-quality figures (PDF & PNG)
│   ├── cv/                        # Nested CV metrics per fold and out-of-fold predictions
│   └── logs/                      # Step execution logs
│
├── docs/                          # In-depth didactic documentation
│   ├── 01_SCHRITT_FUER_SCHRITT_ANLEITUNG.md # Student-friendly German guide & exam defense questions
│   └── 02_METHODEN_UND_ENTSCHEIDUNGEN.md   # Rigorous technical methodology & decision matrix
│
└── report/
    └── FINAL_REPORT.md            # Submission-ready final academic report (strictly following template)
```

---

## Quickstart & Execution

### 1. Environment Setup

Ensure Python 3.11+ is available (e.g. within your conda environment):

```bash
cd newagy
pip install -r requirements.txt
```

### 2. Running the Pipeline

You can run the entire pipeline from end to end:

```bash
python run_all.py
```

Or execute specific stages selectively:

```bash
# Run only data preparation and descriptive Table 1
python run_all.py 01 02

# Run model evaluation and epidemiological association analysis
python run_all.py 05 06
```

---

## Key Methodological Principles (Course Alignment)

1. **Strict Leakage Prevention (W4):** All imputation parameters (medians), scaling metrics (mean, SD), and categorical encodings are fitted strictly on training folds within cross-validation.
2. **Missing Data Handling (W2):** Median imputation with dedicated missingness indicators preserves the full cohort ($n = 5,538$, including 412 cases) without discarding 975 individuals (70 cases) that listwise deletion drops.
3. **Repeated Nested Cross-Validation (W4, W5):** Avoids the "lottery" of a single 80/20 train/test split (which showed AUC fluctuations between 0.80 and 0.92 across 200 random splits).
4. **Resampled Paired t-Test (W3, W9):** Uses the Nadeau & Bengio (2003) correction to account for non-independent, overlapping training sets across cross-validation folds.
5. **Class Imbalance Realism (W9):** Retains natural prevalence ($7.4\%$) to prevent probability distortion caused by oversampling or SMOTE.
