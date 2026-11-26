# Final: our code for the hand-in

Foundations of Data Science, group project (NHANES 2021-2023, dataset 5)

## Notebooks

| Notebook | Content | Status |
|---|---|---|
| `nhanes_chd_analysis.ipynb` | Part 1: data cleaning with the codebook, variables, study sample (flowchart, selection bias), log transform of the markers, missing data (E1), feature selection (E2) | done |
| (next) | Part 2: Table 1, class imbalance (E3), evaluation design (E4), nested CV with the six models, M2 vs M3 | open |
| (next) | Part 3: association analysis (logistic regression, Models 1-3) and the five sensitivity analyses | open |

The template for parts 2 and 3 is `newagy/scripts/` (explained in `newagy/docs/01_SCHRITT_FUER_SCHRITT_ANLEITUNG.md`).
## How to run

Put the two course files in `data/raw/` (`nhanes_raw.csv`, `nhanes_metadata.csv`) and run part 1 from this folder. It takes about one minute and saves the analysis sample to `Final/data/analysis_sample.csv`, which the next parts can load. The `data/` folder is not pushed (course data).

## Results of part 1

- 11,933 participants -> 5,538 adults with NLR and hs-CRP, 412 cases (7.4 %)
- NLR and hs-CRP on the log scale (skewness 4.66 / 7.78 -> 0.15 / 0.23)
- E1: listwise deletion, dropping income and median + missing indicator give the same AUC on the same people (LR 0.867), so we use median + indicator and keep 975 people (70 cases)
- E2: the literature set M3 (AUC 0.862). Rule: keep it unless another strategy has no post-diagnosis variables and is clearly better (more than 0.005 AUC, 95% CI above 0). A lasso on all 342 raw columns reaches 0.923 only through label leakage (aspirin advice, heart failure, stents, lipid-lowering medication); the hybrid pool gives +0.007 (95% CI -0.007 to +0.020)

Planned sensitivity analyses (part 3): complete cases, + lipid-lowering medication (Model 3), excluding hs-CRP > 10 mg/L, survey weights, outcome including angina.
