"""
Step 01: cleaning with the codebook, analysis variables, study sample (flowchart).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from data_processing import build_analysis_variables, clean_raw_data, select_cohort


def main():
    raw = pd.read_csv(cfg.RAW_DATA_PATH, dtype=str, low_memory=False)
    meta = pd.read_csv(cfg.METADATA_PATH)
    print(f"raw data: {raw.shape[0]} rows, {raw.shape[1]} columns")

    clean, counts = clean_raw_data(raw, meta)
    print("cells set to NaN / 0:", counts)
    pd.DataFrame({"reason": list(counts), "cells": list(counts.values())}).to_csv(
        cfg.TABLES_DIR / "cleaning_summary.csv", index=False)

    variables = build_analysis_variables(clean, raw)
    sample, adults, flow = select_cohort(variables, cfg.M2_FEATURES)
    print(flow.to_string(index=False))
    flow.to_csv(cfg.TABLES_DIR / "flowchart.csv", index=False)

    cc = sample["complete_case"]
    print(f"complete on all M2 covariates: {cc.sum()} people, {int(sample.loc[cc, 'chd_mi'].sum())} cases")

    missing = sample[cfg.S3_CANDIDATE_POOL + ["lipid_med"]].isna().agg(["sum", "mean"]).T
    missing.columns = ["n_missing", "pct_missing"]
    missing["pct_missing"] = (100 * missing["pct_missing"]).round(1)
    missing.sort_values("n_missing", ascending=False).to_csv(cfg.TABLES_DIR / "missingness.csv")
    print("missing in M3 (%):\n" + missing.loc[cfg.M3_FEATURES, "pct_missing"].sort_values(ascending=False).to_string())

    sample.to_parquet(cfg.ANALYSIS_SAMPLE_PARQUET, index=False)
    adults.to_parquet(cfg.ADULTS_OUTCOME_PARQUET, index=False)
    clean[clean["SEQN"].isin(sample["SEQN"])].to_parquet(cfg.CLEAN_RAW_PARQUET, index=False)


if __name__ == "__main__":
    main()
