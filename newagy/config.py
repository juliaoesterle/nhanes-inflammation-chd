"""
Settings of the newagy analysis: paths, feature sets and the decision rules of the
design experiments E1-E3 and of the final model choice.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nhanes_raw.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "nhanes_metadata.csv"

DATA_DIR = HERE / "data"                 # participant-level files, git-ignored
RESULTS_DIR = HERE / "results"
TABLES_DIR = RESULTS_DIR / "tables"
CV_DIR = RESULTS_DIR / "cv"
DECISIONS_DIR = RESULTS_DIR / "decisions"

for directory in (DATA_DIR, TABLES_DIR, CV_DIR, DECISIONS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ANALYSIS_SAMPLE_PARQUET = DATA_DIR / "analysis_sample.parquet"
ADULTS_OUTCOME_PARQUET = DATA_DIR / "adults_with_outcome.parquet"
CLEAN_RAW_PARQUET = DATA_DIR / "clean_raw_analysis_sample.parquet"

# -----------------------------------------------------------------------------
# Reproducibility and cross-validation
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
N_JOBS = 18

N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 3          # 15 outer folds for the main model comparison
N_INNER_SPLITS = 5
N_DESIGN_REPEATS = 2         # 10 outer folds for the cheaper design experiments E1-E3
TEST_TRAIN_RATIO = 1 / (N_OUTER_SPLITS - 1)   # n_test / n_train for the Nadeau-Bengio correction
TUNING_SCORING = "neg_log_loss"

# -----------------------------------------------------------------------------
# Outcome, markers and feature sets
# -----------------------------------------------------------------------------
MIN_AGE = 20                 # the MCQ160 heart questions are only asked from age 20
PRIMARY_OUTCOME = "chd_mi"
PRIMARY_EXPOSURES = ["log_nlr", "log_crp"]

M0_FEATURES = ["age", "male"]
M1_FEATURES = M0_FEATURES + ["race_eth", "income_pir", "bmi"]
# established risk factors of PCE / SCORE2 / PREVENT, without treatment variables
M2_FEATURES = M1_FEATURES + ["smoking", "diabetes", "hypertension", "total_chol", "hdl_chol", "egfr"]
M3_FEATURES = M2_FEATURES + PRIMARY_EXPOSURES

FEATURE_SETS = {
    "M0": M0_FEATURES,
    "M1": M1_FEATURES,
    "M2": M2_FEATURES,
    "M2+NLR": M2_FEATURES + ["log_nlr"],
    "M2+CRP": M2_FEATURES + ["log_crp"],
    "M3": M3_FEATURES,
    "M4": PRIMARY_EXPOSURES,          # markers alone, no risk factors
    "M4-NLR": ["log_nlr"],
    "M4-CRP": ["log_crp"],
}

# hybrid pool (S3 in E2): cardiovascular risk domains, <= 15 % missing,
# no treatment, health-care use or other heart diagnosis (leakage screen)
S3_CANDIDATE_POOL = M3_FEATURES + [
    "education", "waist", "sbp", "dbp", "pulse", "told_high_chol",
    "log_tg", "glucose", "uric_acid", "albumin", "log_ggt", "log_alt",
    "log_uacr", "log_cotinine", "wbc", "hemoglobin", "platelets", "rdw",
    "monocyte_pct", "eosinophil_pct", "vitamin_d",
    "sleep_hours", "sedentary_min", "log_activity", "log_drinks_wk", "phq9",
]

CATEGORICAL_VARS = ["race_eth", "smoking"]
BINARY_VARS = ["male", "diabetes", "hypertension", "told_high_chol", "lipid_med"]
INDICATOR_VARS = ["income_pir"]       # gets a 0/1 "missing" column

ALGORITHMS = ["LR", "EN", "DT", "kNN", "SVM", "RF"]

# -----------------------------------------------------------------------------
# Hypotheses and decision rules
# -----------------------------------------------------------------------------
# H1a / H1b: NLR / hs-CRP associated with CHD/MI after adjustment (Bonferroni: 0.05 / 2)
# H1c: NLR + hs-CRP improve the discrimination beyond M2 (relevant gain: dAUC >= 0.01)
BONFERRONI_ALPHA_PRIMARY = 0.05 / 2.0
CLINICALLY_RELEVANT_DELTA_AUC = 0.010

# E1: keep median + indicator unless another option is > 0.005 AUC better on the same people
MISSING_OPTIONS = ["complete_case", "drop_income", "median"]
DELTA_AUC_EXPERIMENT_RULE = 0.005

# E2: keep the literature set S1 unless another strategy has no post-diagnosis variables (no leakage)
# and is more than 0.005 AUC better than S1 with a corrected 95% CI of the paired difference above 0
E2_MIN_AUC_GAIN = 0.005

# E3: use a correction only if it raises the AUC by > 0.005 and the Brier score is not worse
IMBALANCE_OPTIONS = ["none", "class_weight", "oversample", "undersample", "smote"]
RULE_WEIGHT_MIN_AUC_GAIN = 0.005
# the RBF-SVM collapses to "everybody healthy" without class weights; its probabilities
# come from Platt scaling afterwards, so the weights do not distort the calibration
SVM_CLASS_WEIGHT = "balanced"

# final model: the simplest one (LR) unless another algorithm is significantly better
ALPHA = 0.05

# five sensitivity analyses (06), each tied to one of our choices
SENSITIVITY_ANALYSES = [
    "Complete cases (listwise deletion)",
    "Model 3 (+ lipid-lowering drugs)",
    "Exclude hs-CRP > 10 mg/L",
    "Survey-weighted (US population)",
    "Outcome incl. angina",
]
