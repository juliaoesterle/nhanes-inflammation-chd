"""
Configuration and Variable Definitions for the Preprocessing & Feature Selection Stage.

Shared configuration for the team's data cleaning, cohort definition, missing data handling,
and feature selection experiments.
"""

from pathlib import Path

# Paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_DATA = ROOT / "data" / "raw" / "nhanes_raw.csv"
METADATA = ROOT / "data" / "raw" / "nhanes_metadata.csv"

RESULTS_DIR = HERE / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Cohort Criteria
MIN_AGE = 20
PRIMARY_OUTCOME = "chd_mi"

# Core Feature Sets
# M0: Basic Demographics
M0_FEATURES = ["age", "male"]

# M1: Sociodemographics & Body Mass
M1_FEATURES = M0_FEATURES + ["race_eth", "income_pir", "bmi"]

# M2: Established Clinical Risk Factors (Framingham / PCE / SCORE2 / PREVENT)
# Note: Post-diagnostic medications (statins, antihypertensives) are excluded to prevent collider/treatment leakage.
M2_FEATURES = M1_FEATURES + [
    "smoking", "diabetes", "hypertension", "total_chol", "hdl_chol", "egfr"
]

# Primary Inflammatory Biomarkers
EXPOSURES = ["log_nlr", "log_crp"]

# M3: Full Primary Model (Established Risk Factors + Inflammatory Markers)
M3_FEATURES = M2_FEATURES + EXPOSURES

# Variable Types for Pipelines
CATEGORICAL_VARS = ["race_eth", "smoking"]
BINARY_VARS = ["male", "diabetes", "hypertension"]
INDICATOR_VARS = ["income_pir"]  # Variables receiving a missingness indicator flag
