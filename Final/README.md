# Data Cleaning, Cohort Selection & Preprocessing Pipeline

**Foundations of Data Science — Group Project (Dataset 5: NHANES 2021–2023)**  
*Lead Author: Marco*

---

## Executive Overview for the Team

This folder contains the complete, scientifically validated foundation for our group project's **Data Cleaning**, **Participant Selection**, **Missing Data Strategy**, and **Feature Selection**:

1. **Cohort & Participant Flow (`01_cohort_selection.py`):**
   * We start with **11,933** total NHANES participants.
   * We end with **5,538** participants in our final analytic sample (**412 cases of CHD/MI, 7.4% prevalence**).
   * Exclusions follow 4 transparent protocol steps (Adults $\ge 20$, non-missing outcome, attended MEC, both NLR and hs-CRP measured).
   * A selection bias check via Standardized Mean Difference (SMD) confirms excluded participants without blood draws had lower income ($\text{SMD} = -0.36$), which we document as an explicit limitation.

2. **Missing Data Strategy (`02_missing_data_experiment.py`):**
   * We evaluated **3 competing strategies** for handling missing income (12.7% missing):
     * *Option A:* Listwise deletion (complete cases).
     * *Option B:* Dropping the income column.
     * *Option C:* Median imputation with a dedicated missingness indicator (`income_pir_missing`).
   * *Experiment E1 Result:* Test AUC on identical complete cases is virtually identical across all 3 options ($\text{AUC} \approx 0.865$).
   * *Decision:* We choose **Option C**. It prevents dropping **909 participants and 70 real heart disease cases (17% of all cases!)**, and allows our models to generate predictions for patients with incomplete survey data.

3. **Feature Selection & Leakage Prevention (`03_feature_selection_experiment.py`):**
   * We evaluated **3 feature selection approaches**:
     * *Strategy 1:* Literature / Guideline risk factor set (M2/M3: Framingham / PCE / SCORE2 / PREVENT).
     * *Strategy 2:* Naive Lasso across all 342 raw survey variables.
     * *Strategy 3:* Hybrid domain-screened pool (39 candidate variables) + regularized selection.
   * *Experiment E2 Result:* Naive Lasso reaches an apparent AUC of 0.923, but inspectable coefficients expose **severe label leakage**: it selects post-diagnostic consequences ("metal objects in body" [stents], "advised to take aspirin", "heart failure diagnosed", "statin prescriptions").
   * *Decision:* We choose **Strategy 1 (Literature Set)**. It directly answers our research question (*"do inflammatory markers add value beyond established clinical risk scores?"*) while completely avoiding circular post-treatment bias.

4. **Shared Preprocessing Module (`preprocessing.py`):**
   * Ready-to-import module for everyone in the team:
     ```python
     from preprocessing import load_cohort, get_preprocessor, make_pipeline
     df = load_cohort() # returns the clean 5,538 cohort
     ```

---

## Quickstart & Reproducibility

To run the experiments and inspect the generated tables:

```bash
# 1. Run cohort selection & flowchart
python 01_cohort_selection.py

# 2. Run missing data experiment (E1)
python 02_missing_data_experiment.py

# 3. Run feature selection & leakage check (E2)
python 03_feature_selection_experiment.py
```

All generated tables are stored in `results/tables/`, and publication-quality figures are stored in `results/figures/`.
