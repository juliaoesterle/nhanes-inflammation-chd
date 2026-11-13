"""
Central configuration of the NewMarco analysis.

Every analysis choice lives here and points to its decision record
(D1 ... D16) in docs/TECHNICAL_DOCUMENTATION.md. Each record lists three
candidate options, the criteria, the evidence (literature and/or an
experiment in 03_design_experiments.py) and the chosen option.

Decision RULES for the empirically tested choices are fixed here BEFORE the
experiments are run, so that the outcome of an experiment cannot be
re-interpreted after the fact ("garden of forking paths").
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_DATA = ROOT / "data" / "raw" / "nhanes_raw.csv"
METADATA = ROOT / "data" / "raw" / "nhanes_metadata.csv"

DATA = HERE / "data"            # participant-level files -> git-ignored
RESULTS = HERE / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"
DECISIONS = RESULTS / "decisions"   # machine-written outcomes of decision rules
CV_DIR = RESULTS / "cv"             # fold-level results and out-of-fold predictions

for _p in (DATA, TABLES, FIGURES, LOGS, DECISIONS, CV_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
N_JOBS = 18          # 20 logical cores on the development machine; keep 2 free

# ----------------------------------------------------------------------------
# D2 Outcome: self-reported, doctor-diagnosed CHD or MI (angina excluded)
# ----------------------------------------------------------------------------
OUTCOME = "chd_mi"
SENSITIVITY_OUTCOMES = {
    "chd_mi_angina": "CHD, MI or angina (broader)",
    "mi_strict": "MI only (controls: neither CHD nor MI)",
    "chd_strict": "CHD only (controls: neither CHD nor MI)",
}

# ----------------------------------------------------------------------------
# D3 Study population
# ----------------------------------------------------------------------------
MIN_AGE = 20    # MCQ160 heart questions are only asked from age 20 -> not a choice

# ----------------------------------------------------------------------------
# D4/D5 Predictors. S1 = knowledge-driven sets (primary analysis).
# M2 = established risk factors of the PCE / PREVENT / SCORE2 / QRISK3 family,
#      minus treatment indicators (see D4), see docs for the mapping table.
# ----------------------------------------------------------------------------
M0 = ["age", "male"]
M1 = M0 + ["race_eth", "income_pir", "bmi"]
M2 = M1 + ["smoking", "diabetes", "hypertension", "total_chol", "hdl_chol", "egfr"]
MARKERS = ["log_nlr", "log_crp"]
M3 = M2 + MARKERS

FEATURE_SETS = {
    "M0": M0,
    "M1": M1,
    "M2": M2,
    "M2+NLR": M2 + ["log_nlr"],
    "M2+CRP": M2 + ["log_crp"],
    "M3": M3,
    # Reference models suggested by the TA: the markers ALONE (no risk factors).
    # They show how much NLR / hs-CRP can discriminate on their own; the research
    # question itself ("beyond established risk factors") is answered by M3 vs M2.
    "M4": MARKERS,
    "M4-NLR": ["log_nlr"],
    "M4-CRP": ["log_crp"],
}

# S3 = hybrid: knowledge-screened candidate pool, then data-driven selection.
# Inclusion criteria (D5): (a) belongs to a recognised cardiovascular risk domain
# (AHA Life's Essential 8, INTERHEART, PREVENT, emerging biomarkers),
# (b) measured in everyone examined (<= 15 % missing in the analysis sample),
# (c) NOT a treatment, a health-care-use variable, another heart diagnosis or
#     any other direct consequence of having CHD (label-leakage screen).
# Criterion (b) follows the course guideline (W2): < 5 % missing = safe,
# 5-20 % = acceptable with caution; all pool variables are <= 13.6 % missing.
S3_EXTRA = [
    "education", "waist", "sbp", "dbp", "pulse", "told_high_chol",
    "log_tg", "glucose", "uric_acid", "albumin", "log_ggt", "log_alt",
    "log_uacr", "log_cotinine", "wbc", "hemoglobin", "platelets", "rdw",
    "monocyte_pct", "eosinophil_pct", "vitamin_d",
    "sleep_hours", "sedentary_min", "log_activity", "log_drinks_wk", "phq9",
]
S3_POOL = M3 + S3_EXTRA

# Variable types (for preprocessing)
CATEGORICAL = ["race_eth", "smoking"]
BINARY = ["male", "diabetes", "hypertension", "told_high_chol", "lipid_med", "bp_med"]
# every other feature is treated as continuous

# D3 missing data: income non-response is plausibly informative (refusal),
# so it gets a missing-indicator. Lab values are missing for logistic reasons
# (sample handling) -> no indicator.
INDICATOR_VARS = ["income_pir"]

# Treatment variables: never used as ML predictors in the primary analysis (D4)
TREATMENT_VARS = ["lipid_med", "bp_med"]

# ----------------------------------------------------------------------------
# D7 Resampling: repeated nested cross-validation instead of one 80/20 split
# ----------------------------------------------------------------------------
OUTER_SPLITS = 5
OUTER_REPEATS = 3          # 15 outer folds for the main comparison
INNER_SPLITS = 5
DESIGN_REPEATS = 2         # 10 outer folds for the (cheaper) design experiments
TEST_TRAIN_RATIO = 1 / (OUTER_SPLITS - 1)   # n_test / n_train, Nadeau-Bengio correction

# ----------------------------------------------------------------------------
# D10 Tuning objective: log-loss (strictly proper scoring rule)
# ----------------------------------------------------------------------------
TUNING_SCORING = "neg_log_loss"

# ----------------------------------------------------------------------------
# Pre-specified decision rules (fixed before running 03_design_experiments.py)
# ----------------------------------------------------------------------------
# D3 (missing data; course W2 "deletion and imputation methods"): three options
#   complete_case : listwise deletion, drop every person with any gap
#   drop_income   : drop the column with most gaps (income, 12.7 % missing)
#   median        : median imputation + a missing-indicator for income
#                   (W2: "missing data can hold information")
#   Rule: keep everybody and every variable (median) unless one of the other two
#   options is more than RULE_IMPUTATION_TOLERANCE better in AUC on the SAME test
#   people (the complete cases, the only people all three options can score).
MISSING_OPTIONS = ["complete_case", "drop_income", "median"]
RULE_IMPUTATION_TOLERANCE = 0.005

# D9 (class imbalance): use a correction (class weights, random over- or
#   undersampling) only if it raises the mean AUC by more than
#   RULE_WEIGHT_MIN_AUC_GAIN AND does not worsen the Brier score; otherwise keep
#   the natural prevalence and move the decision threshold instead.
RULE_WEIGHT_MIN_AUC_GAIN = 0.005

# D12 (final model): recommend the simplest model (unpenalised logistic
#   regression) unless a more complex algorithm is significantly better
#   (Bonferroni-adjusted paired t-test over the CV folds, p < 0.05).
ALPHA = 0.05
# Two co-primary hypotheses (H1a NLR, H1b hs-CRP) -> Bonferroni: each tested at 0.025
ALPHA_PRIMARY = ALPHA / 2

# ----------------------------------------------------------------------------
# Labels for tables and figures
# ----------------------------------------------------------------------------
LABELS = {
    "age": "Age", "male": "Male sex", "race_eth": "Race/ethnicity",
    "income_pir": "Income-to-poverty ratio", "bmi": "BMI", "smoking": "Smoking",
    "diabetes": "Diabetes", "hypertension": "Hypertension",
    "total_chol": "Total cholesterol", "hdl_chol": "HDL cholesterol", "egfr": "eGFR",
    "log_nlr": "NLR (log)", "log_crp": "hs-CRP (log)", "education": "Education",
    "waist": "Waist circumference", "sbp": "Systolic BP", "dbp": "Diastolic BP",
    "pulse": "Resting pulse", "told_high_chol": "Ever told high cholesterol",
    "log_tg": "Triglycerides (log)", "glucose": "Serum glucose", "uric_acid": "Uric acid",
    "albumin": "Serum albumin", "log_ggt": "GGT (log)", "log_alt": "ALT (log)",
    "log_uacr": "Urine albumin/creatinine (log)", "log_cotinine": "Cotinine (log)",
    "wbc": "White blood cells", "hemoglobin": "Haemoglobin", "platelets": "Platelets",
    "rdw": "Red cell distribution width", "monocyte_pct": "Monocytes %",
    "eosinophil_pct": "Eosinophils %", "vitamin_d": "Vitamin D (25-OH)",
    "sleep_hours": "Sleep hours", "sedentary_min": "Sedentary minutes/day",
    "log_activity": "Leisure activity (log)", "log_drinks_wk": "Alcohol drinks/week (log)",
    "phq9": "Depression score (PHQ-9)", "lipid_med": "Lipid-lowering medication",
    "bp_med": "BP medication",
}

# D11 model families: every classifier taught in the course (W5-W9)
ALGORITHMS = ["LR", "EN", "DT", "kNN", "SVM", "RF"]
ALGO_LABELS = {
    "LR": "Logistic regression",
    "EN": "Ridge/Lasso/elastic-net LR",
    "DT": "Decision tree",
    "kNN": "k-nearest neighbours",
    "SVM": "SVM (RBF kernel)",
    "RF": "Random forest",
}
COURSE_LECTURE = {"LR": "W6-W7", "EN": "W5-W6", "DT": "W7-W8", "kNN": "W6/W8",
                  "SVM": "W9", "RF": "W8"}

# D9 class imbalance options compared in experiment E3 (course W9)
IMBALANCE_OPTIONS = ["none", "class_weight", "oversample", "undersample", "smote"]
# Post-hoc correction (documented in D9, found when inspecting the first run of 04):
# an RBF-SVM fitted on 7 % prevalence without class weights collapses towards the
# constant "everybody healthy" solution for small C; the inner CV then always picked the
# grid corner (C = 10, gamma = 0.08) and the test AUC was only 0.76. For the SVM, class
# weights change only the ranking function; its probabilities come from Platt scaling on
# out-of-fold decision values, which restores calibration (unlike LR/RF in E3).
SVM_CLASS_WEIGHT = "balanced"

# Threshold for confusion-matrix metrics (course W9): chosen on out-of-fold
# predictions of the TRAINING part only (never on the test fold), by Youden's J.
THRESHOLD_RULE = "youden"

# D14 outliers (course W2): keep (log scale) in the primary analysis;
# sensitivity analyses: exclude hs-CRP > 10 mg/L, and exclude IQR-rule outliers
# of the log markers.
IQR_FACTOR = 1.5
