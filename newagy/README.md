# newagy: modular reference analysis

Inflammation markers (NLR, hs-CRP) and coronary heart disease in NHANES 2021-2023
(Foundations of Data Science, group project, dataset 5).

This folder is the template for our own notebooks in `Final/`. Every number in
`report/FINAL_REPORT.md` and in the docs is computed by the scripts below; nothing is typed in
by hand. The methods are the same as in `NewMarco/`, and the results agree with it.

## Results

| Question | Result |
|---|---|
| H1a: NLR associated beyond the risk factors? | yes: OR 1.20 per SD of log NLR (1.07-1.34), p = 0.002 |
| H1b: hs-CRP associated beyond the risk factors? | no: OR 0.96 (0.84-1.08), p = 0.48; the crude association disappears after adjusting for BMI |
| H1c: better discrimination with the markers? | no: logistic regression dAUC +0.0013 (-0.0009 to +0.0035) |
| Does any algorithm beat logistic regression? | no significant difference (Bonferroni) |
| Sensitivity analyses | same conclusions in all five (complete cases, + lipid-lowering medication, hs-CRP > 10 mg/L excluded, survey weights, outcome incl. angina) |

## Scripts

| Script | What it computes | Runtime |
|---|---|---|
| `scripts/01_prepare_cohort.py` | cleaning with the codebook, variables, flowchart, missingness | < 1 min |
| `scripts/02_exploratory_analysis.py` | Table 1, marker skewness, selection bias (SMD), why income is missing, events per parameter | < 1 min |
| `scripts/03_design_experiments.py` | E1 missing data, E2 feature selection, E3 class imbalance, E4 single-split variability | ~ 1 min |
| `scripts/04_run_nested_cv.py` | repeated nested CV (15 outer folds), six algorithms, permutation importance | 30-60 min |
| `scripts/05_model_evaluations.py` | performance, marker gain M3 - M2, algorithms vs LR, Youden threshold, final model | < 1 min |
| `scripts/06_association_analysis.py` | logistic regression Models 1-3, LR test, VIF, squared term, interactions, decomposition, five sensitivity analyses | < 1 min |
| `scripts/07_interpretability.py` | permutation importance ranks, partial dependence, depth-3 tree, learning curves | ~ 2 min |

`modules/` holds the shared code (cleaning, preprocessing pipeline, models, metrics, logistic
regression). All settings and the decision rules are in `config.py`. The output is written to
`results/tables`, `results/cv` and `results/decisions`; participant-level data goes to `data/`
(not pushed). newagy has no plotting code; the figures are in `NewMarco/results/figures/`.

## Run

Put `nhanes_raw.csv` and `nhanes_metadata.csv` in `../data/raw/`, then (conda env `nhanes`):

```bash
cd newagy
python run_all.py            # everything
python run_all.py 05 06      # single steps
```

## Docs

- `docs/01_SCHRITT_FUER_SCHRITT_ANLEITUNG.md`: every step with the options we compared and why we chose one (German)
- `docs/02_METHODEN_UND_ENTSCHEIDUNGEN.md`: formulas and the decision table
- `report/FINAL_REPORT.md`: orientation for our report (we write our own text)
