# NLR, hs-CRP and coronary heart disease in NHANES 2021–2023

Redesigned analysis for the ETH *Foundations of Data Science* group project (Dataset 5). Every analytical decision compares three options (or all options taught in the course) against a rule written down beforehand, and the methods stay close to lectures W1–W9.

**Research question (group README).** Among US adults, are higher NLR and hs-CRP associated with a self-reported history of coronary heart disease or myocardial infarction (CHD/MI), independent of established risk factors?
- **H1a:** NLR is associated with CHD/MI after adjustment.
- **H1b:** hs-CRP is associated with CHD/MI after adjustment.
- **H1c (secondary):** NLR and hs-CRP improve discrimination beyond established risk factors.

## Results at a glance

| | |
|---|---|
| Sample | 5,538 adults, 412 with CHD/MI (7.4%). Median imputation + indicator keeps 975 people (70 cases) that listwise deletion would drop |
| **H1a supported** | NLR: OR 1.20 per SD (95% CI 1.07–1.34, p = 0.002 < 0.025) |
| **H1b not supported** | hs-CRP: OR 0.96 (0.84–1.08, p = 0.48); its crude association is explained by BMI |
| Markers alone | AUC 0.631 (NLR alone 0.633, hs-CRP alone 0.506 = chance) vs. 0.801 for age + sex |
| **H1c not supported** | Logistic regression AUC 0.861 → +0.0013 (−0.0009 to +0.0035) with both markers; no significant gain for any of six algorithms |
| Algorithms | None significantly better than logistic regression (SVM highest, 0.867, Bonferroni p = 0.74) → logistic regression recommended |
| Robustness | Same conclusions in 13 sensitivity analyses |

## Documents

| File | For whom |
|---|---|
| [docs/ANLEITUNG_SCHRITT_FUER_SCHRITT.md](docs/ANLEITUNG_SCHRITT_FUER_SCHRITT.md) | **Start here.** The whole process step by step (German): every decision with its options, the course lecture, the result, how to build it yourself, and check questions |
| [docs/LAIEN_ERKLAERUNG.md](docs/LAIEN_ERKLAERUNG.md) | Plain-language summary for non-experts (German) |
| [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | Technical reference (English): decision records D1–D18, pipeline, results, limitations, references |
| [report/REPORT_DRAFT.md](report/REPORT_DRAFT.md) | Report draft following the course template |
| [WEITERMACHEN.md](WEITERMACHEN.md) | Handover notes (German) |

## Pipeline

| Script | What it does |
|---|---|
| `config.py` | All decisions, feature sets and decision rules |
| `common.py` | Leakage-free pipelines, the six classifiers and their grids, metrics, corrected paired t-test, Bonferroni, SMOTE, figure style |
| `01_prepare_data.py` | Cleaning, variables, sample flow |
| `02_descriptives.py` | Table 1, missingness, selection bias, distributions, events per parameter (`fig01`–`fig07`) |
| `03_design_experiments.py` | E1 missing data, E2 feature selection, E3 class imbalance, E4 split variability (`fig08`–`fig11`) |
| `04_nested_cv.py` | Repeated nested CV: 6 algorithms × feature sets, 15 outer / 5 inner folds |
| `05_evaluate_models.py` | Marker gain, algorithm comparison, thresholds, calibration, ROC/PR (`fig12`–`fig16`) |
| `06_association.py` | Primary question: logistic regression, missing-data options, survey weights, LR test, shape, interactions, sensitivity analyses (`fig18`–`fig20`) |
| `07_interpretation.py` | Permutation importance, partial dependence, illustrative tree, learning curves (`fig21`–`fig24`) |

## How to run

```bash
cd NewMarco
pip install -r requirements.txt   # exact versions; older scikit-learn changes the models
python run_all.py                 # everything (about 1.2 h on 20 cores, mostly step 04)
python run_all.py 05 06           # selected steps only
```

Participant-level data are not committed (`data/` is git-ignored); `01_prepare_data.py` rebuilds them from the course files.
