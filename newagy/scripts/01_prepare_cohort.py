"""
Step 01: Cohort Extraction & Data Cleaning.

Executes codebook-based raw data cleaning, derives engineered clinical
covariates, applies eligibility criteria, and records the participant flow.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Setup import path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from data_processing import (
    build_analysis_variables,
    clean_raw_data,
    generate_cohort_and_flowchart,
)


def run_cohort_preparation():
    start_time = time.time()
    print("=" * 70)
    print("STEP 01: COHORT PREPARATION & DATA CLEANING")
    print("=" * 70)

    print(f"Reading raw data from: {cfg.RAW_DATA_PATH}")
    raw_df = pd.read_csv(cfg.RAW_DATA_PATH, low_memory=False)
    print(f"Raw dataset shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    print("Cleaning raw variables against metadata codebook...")
    clean_df = clean_raw_data(cfg.RAW_DATA_PATH, cfg.METADATA_PATH)

    print("Engineering clinical variables and inflammatory markers...")
    engineered_df = build_analysis_variables(clean_df, raw_df)

    print("Filtering study cohort according to protocol criteria...")
    analysis_sample, adults_outcome, flowchart_df = generate_cohort_and_flowchart(engineered_df)

    # Save outputs
    print(f"Saving analysis cohort (n = {len(analysis_sample)}) to {cfg.ANALYSIS_SAMPLE_PARQUET}")
    analysis_sample.to_parquet(cfg.ANALYSIS_SAMPLE_PARQUET, index=False)
    adults_outcome.to_parquet(cfg.ADULTS_OUTCOME_PARQUET, index=False)

    flowchart_path = cfg.TABLES_DIR / "flowchart.csv"
    flowchart_df.to_csv(flowchart_path, index=False)
    print(f"Flowchart saved to: {flowchart_path}")
    print(flowchart_df.to_string(index=False))

    # Missingness summary
    key_vars = cfg.M3_FEATURES + [cfg.PRIMARY_OUTCOME]
    missing_data = []
    for var in key_vars:
        if var in analysis_sample.columns:
            n_miss = int(analysis_sample[var].isna().sum())
            pct_miss = 100.0 * n_miss / len(analysis_sample)
            missing_data.append({"variable": var, "n_missing": n_miss, "percent_missing": pct_miss})
    missing_df = pd.DataFrame(missing_data).sort_values(by="percent_missing", ascending=False)
    missing_path = cfg.TABLES_DIR / "missingness.csv"
    missing_df.to_csv(missing_path, index=False)
    print(f"Missingness table saved to: {missing_path}")

    elapsed = time.time() - start_time
    print(f"Step 01 completed successfully in {elapsed:.1f} seconds.\n")


if __name__ == "__main__":
    run_cohort_preparation()
