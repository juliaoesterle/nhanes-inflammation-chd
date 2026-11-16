"""
Central Configuration for the NHANES CHD & Inflammation Project (newagy).

This module defines all experimental settings, feature subsets, paths,
and pre-specified decision rules before any modeling is conducted.
Following the ETH Foundations of Data Science curriculum (W1-W9), all criteria
are written down beforehand to avoid data snooping ('garden of forking paths').
"""

from __future__ import annotations
from pathlib import Path

# =============================================================================
# 1. Project Directory Structure & Paths
# =============================================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# Raw data sources (course dataset)
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nhanes_raw.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "nhanes_metadata.csv"

# Local directories
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
CV_DIR = RESULTS_DIR / "cv"
LOGS_DIR = RESULTS_DIR / "logs"
DECISIONS_DIR = RESULTS_DIR / "decisions"

for directory in (DATA_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, CV_DIR, LOGS_DIR, DECISIONS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Parquet file paths
ANALYSIS_SAMPLE_PARQUET = DATA_DIR / "analysis_sample.parquet"
ADULTS_OUTCOME_PARQUET = DATA_DIR / "adults_with_outcome.parquet"
CLEAN_RAW_PARQUET = DATA_DIR / "clean_raw_analysis_sample.parquet"

# =============================================================================
# 2. Computational Parameters & Reproducibility
# =============================================================================
RANDOM_SEED = 42
N_JOBS = 18

# Nested Cross-Validation structure
N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 3
N_OUTER_FOLDS = N_OUTER_SPLITS * N_OUTER_REPEATS  # 15 folds total
N_INNER_SPLITS = 5

# =============================================================================
# 3. Cohort & Variable Definitions
# =============================================================================
MIN_AGE = 20  # MCQ160 questions administered only to adults >= 20 years

# Primary outcome: Self-reported doctor-diagnosed CHD or MI
PRIMARY_OUTCOME = "chd_mi"

# Alternative outcome definitions for sensitivity checks
SENSITIVITY_OUTCOMES = {
    "chd_mi_angina": "CHD, MI, or Angina Pectoris (broader composite)",
    "mi_strict": "Myocardial Infarction only (isolated)",
    "chd_strict": "Coronary Heart Disease only (isolated)",
}

# Key Exposure Markers
PRIMARY_EXPOSURES = ["log_nlr", "log_crp"]

# Core Risk Factor Sets (Nested Modeling Strategy)
# M0: Basic Demographics
M0_FEATURES = ["age", "male"]

# M1: Sociodemographics & Body Mass
M1_FEATURES = M0_FEATURES + ["race_eth", "income_pir", "bmi"]

# M2: Established Clinical Risk Factors (PCE / SCORE2 / PREVENT / QRISK3 guideline factors)
# Note: Statins and anti-hypertensive drugs excluded from baseline to avoid post-treatment bias
M2_FEATURES = M1_FEATURES + [
    "smoking", "diabetes", "hypertension", "total_chol", "hdl_chol", "egfr"
]

# M3: Established Clinical Risk Factors + Both Inflammatory Markers
M3_FEATURES = M2_FEATURES + PRIMARY_EXPOSURES

# Predefined Feature Subsets for Comparative ML Experiments
FEATURE_SETS = {
    "M0": M0_FEATURES,
    "M1": M1_FEATURES,
    "M2": M2_FEATURES,
    "M2+NLR": M2_FEATURES + ["log_nlr"],
    "M2+CRP": M2_FEATURES + ["log_crp"],
    "M3": M3_FEATURES,
    # Marker-only benchmarks (evaluates standalone signal without covariates)
    "M4": PRIMARY_EXPOSURES,
    "M4-NLR": ["log_nlr"],
    "M4-CRP": ["log_crp"],
}

# Candidate pool for Hybrid Feature Selection (S3)
# Selected based on domain literature (Life's Essential 8, INTERHEART, laboratory markers)
# strictly excluding treatment and diagnostic consequences (leakage prevention).
S3_CANDIDATE_POOL = M3_FEATURES + [
    "education", "waist", "sbp", "dbp", "pulse", "told_high_chol",
    "log_tg", "glucose", "uric_acid", "albumin", "log_ggt", "log_alt",
    "log_uacr", "log_cotinine", "wbc", "hemoglobin", "platelets", "rdw",
    "monocyte_pct", "eosinophil_pct", "vitamin_d",
    "sleep_hours", "sedentary_min", "log_activity", "log_drinks_wk", "phq9",
]

# Variable Type Specifications for Scikit-Learn Preprocessing Pipelines
CATEGORICAL_VARS = ["race_eth", "smoking"]
BINARY_VARS = ["male", "diabetes", "hypertension", "told_high_chol"]
INDICATOR_VARS = ["income_pir"]  # Covariates with dedicated missingness indicators

# Continuous variables derived dynamically
EXCLUDE_FROM_CONTINUOUS = set(CATEGORICAL_VARS + BINARY_VARS + [PRIMARY_OUTCOME, "SEQN"])

# =============================================================================
# 4. Pre-specified Decision Rules & Hypotheses
# =============================================================================
# Primary Hypotheses:
# H1a: NLR is associated with CHD/MI after adjustment (alpha = 0.025, Bonferroni)
# H1b: hs-CRP is associated with CHD/MI after adjustment (alpha = 0.025, Bonferroni)
# H1c: NLR + hs-CRP improve discrimination beyond M2 (Delta-AUC >= 0.01 threshold)

BONFERRONI_ALPHA_PRIMARY = 0.05 / 2.0  # 0.025 for two co-primary exposure tests

# Minimum threshold for meaningful empirical difference in design experiments
DELTA_AUC_EXPERIMENT_RULE = 0.005

# Minimum clinically meaningful improvement in discrimination
CLINICALLY_RELEVANT_DELTA_AUC = 0.010
