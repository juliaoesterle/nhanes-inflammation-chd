"""
Preprocessing & Data Pipeline Module for the Group Project.

This module provides the team with a single, consistent, leakage-free entry point
for loading and preprocessing the NHANES analytic cohort.

Key functions:
    load_cohort()               -> pd.DataFrame (n = 5,538; 412 cases)
    get_preprocessor()          -> ColumnTransformer (median imputation, scaling, one-hot, indicator)
    make_pipeline(model, feats) -> Pipeline (preprocessor + estimator)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg

# Regular expression for NHANES non-response entries
NON_RESPONSE_REGEX = re.compile(
    r"^(refused|don.?t know|don\W*t know|missing|sp refusal|participant refusal)$",
    re.IGNORECASE,
)
SAS_ZERO_EPSILON = 1e-70


# =============================================================================
# 1. Custom Transformer: Missingness Indicator
# =============================================================================
class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Constructs a binary indicator (0.0 / 1.0) flagging whether values in specified
    columns were missing prior to imputation (Course W2: 'Missingness can hold information').
    """

    def __init__(self, target_cols: List[str] | None = None):
        self.target_cols = target_cols or cfg.INDICATOR_VARS

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_present_ = [c for c in self.target_cols if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame):
        df_out = pd.DataFrame(index=X.index)
        for col in self.cols_present_:
            df_out[f"{col}_missing"] = X[col].isna().astype(float)
        return df_out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_missing" for c in self.cols_present_], dtype=object)


# =============================================================================
# 2. Raw Data Cleaning & Feature Calculation
# =============================================================================
def parse_metadata_codes(spec: str | float) -> Tuple[List[Tuple[float, float]], dict]:
    """Parse metadata value_codes for valid ranges and non-response sentinels."""
    ranges, codes = [], {}
    if not isinstance(spec, str):
        return ranges, codes
    for part in spec.split(" | "):
        part = part.strip()
        if not part or "=" not in part:
            continue
        code_str, label = part.split("=", 1)
        code_str, label = code_str.strip(), label.strip()

        range_match = re.match(r"^(-?[\d\.]+)\s+to\s+(-?[\d\.]+)$", code_str, re.IGNORECASE)
        if range_match:
            try:
                lo, hi = float(range_match.group(1)), float(range_match.group(2))
                ranges.append((min(lo, hi), max(lo, hi)))
            except ValueError:
                pass
            continue

        try:
            val = float(code_str)
            codes[val] = label
        except ValueError:
            pass
    return ranges, codes


def clean_raw_columns(raw_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Replaces SAS zero artifacts, non-response codes, and undocumented sentinels with NaN."""
    cleaned = raw_df.copy()
    col_name_key = "column_name" if "column_name" in meta_df.columns else "variable_name"
    meta_indexed = meta_df.set_index(col_name_key)

    for col in cleaned.columns:
        if col not in meta_indexed.index or col == "SEQN":
            continue

        spec = meta_indexed.loc[col].get("value_codes", "")
        ranges, codes = parse_metadata_codes(spec)
        non_resp = {c for c, lbl in codes.items() if NON_RESPONSE_REGEX.match(lbl)}
        valid_discrete = set(codes.keys()) - non_resp

        series = pd.to_numeric(cleaned[col], errors="coerce")
        series = series.mask(series.abs() < SAS_ZERO_EPSILON, 0.0)

        if non_resp:
            series = series.mask(series.isin(non_resp), np.nan)

        if ranges or valid_discrete:
            in_range = pd.Series(False, index=series.index)
            for lo, hi in ranges:
                in_range |= (series >= lo) & (series <= hi)
            is_valid = in_range | series.isin(valid_discrete) | series.isna()
            series = series.mask(~is_valid, np.nan)

        cleaned[col] = series
    return cleaned


def calculate_ckd_epi_egfr(serum_creatinine: pd.Series, age: pd.Series, male: pd.Series) -> pd.Series:
    """Official 2021 race-free CKD-EPI equation (Inker et al., NEJM 2021)."""
    kappa = np.where(male == 1.0, 0.9, 0.7)
    alpha = np.where(male == 1.0, -0.302, -0.241)
    gender_factor = np.where(male == 1.0, 1.0, 1.012)
    scr_k = serum_creatinine / kappa
    egfr = 142.0 * np.minimum(scr_k, 1.0)**alpha * np.maximum(scr_k, 1.0)**(-1.200) * (0.9938**age) * gender_factor
    return pd.Series(egfr, index=serum_creatinine.index)


def load_cohort() -> pd.DataFrame:
    """
    Main data loader: cleans raw NHANES data, derives all engineered variables,
    and returns the final analysis cohort (n = 5,538 participants, 412 cases).
    """
    raw_df = pd.read_csv(cfg.RAW_DATA, low_memory=False)
    meta_df = pd.read_csv(cfg.METADATA)
    clean_df = clean_raw_columns(raw_df, meta_df)

    df = pd.DataFrame({"SEQN": clean_df["SEQN"].astype("int64")})

    # Outcomes: Doctor-diagnosed CHD (MCQ160C) or Heart Attack / MI (MCQ160E)
    chd = clean_df["MCQ160C"].map({1.0: 1.0, 2.0: 0.0})
    mi = clean_df["MCQ160E"].map({1.0: 1.0, 2.0: 0.0})
    df["chd_mi"] = np.select([(chd == 1.0) | (mi == 1.0), (chd == 0.0) & (mi == 0.0)], [1.0, 0.0], np.nan)

    # Inflammatory Biomarkers (Natural log transformed to correct extreme skewness)
    df["crp"] = clean_df["LBXHSCRP"]
    df["log_crp"] = np.log(clean_df["LBXHSCRP"].where(clean_df["LBXHSCRP"] > 0))

    lymph_pct = clean_df["LBXLYPCT"].where(clean_df["LBXLYPCT"] > 0)
    df["nlr"] = clean_df["LBXNEPCT"] / lymph_pct
    df["log_nlr"] = np.log(df["nlr"].where(df["nlr"] > 0))

    # Demographics & SES
    df["age"] = clean_df["RIDAGEYR"]
    df["male"] = clean_df["RIAGENDR"].map({1.0: 1.0, 2.0: 0.0})
    df["race_eth"] = clean_df["RIDRETH3"].map({
        1.0: "mexican_american", 2.0: "other_hispanic", 3.0: "nh_white",
        4.0: "nh_black", 6.0: "nh_asian", 7.0: "other_multi",
    })
    df["income_pir"] = clean_df["INDFMPIR"]

    # Clinical Risk Factors
    df["bmi"] = clean_df["BMXBMI"]

    smq020, smq040 = clean_df["SMQ020"], clean_df["SMQ040"]
    df["smoking"] = np.select(
        [smq020 == 2.0, (smq020 == 1.0) & (smq040 == 3.0), (smq020 == 1.0) & smq040.isin([1.0, 2.0])],
        ["never", "former", "current"], default=None,
    )
    df["diabetes"] = clean_df["DIQ010"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0})

    bp_told = clean_df["BPQ020"].map({1.0: 1.0, 2.0: 0.0})
    bp_meas = ((clean_df["BPXOSY1"] >= 140.0) | (clean_df["BPXODI1"] >= 90.0)).astype(float)
    df["hypertension"] = np.select([(bp_told == 1.0) | (bp_meas == 1.0), (bp_told == 0.0) & (bp_meas == 0.0)], [1.0, 0.0], np.nan)

    df["total_chol"] = clean_df["LBXTC"]
    df["hdl_chol"] = clean_df["LBDHDD"]
    df["egfr"] = calculate_ckd_epi_egfr(clean_df["LBXSCR"], df["age"], df["male"])

    # Medications (Consequence of diagnosis -> Stored for sensitivity analysis, never used as baseline predictor)
    df["lipid_med"] = clean_df.get("BPQ101D", clean_df.get("BPQ100D", pd.Series(np.nan))).map({1.0: 1.0, 2.0: 0.0})

    # Survey Weights
    df["wtmec2yr"] = clean_df["WTMEC2YR"]

    # Cohort Exclusions
    is_adult = df["age"] >= cfg.MIN_AGE
    has_outcome = df["chd_mi"].notna()
    attended_mec = df["wtmec2yr"] > 0
    has_markers = df["nlr"].notna() & df["crp"].notna()

    cohort = df[is_adult & has_outcome & attended_mec & has_markers].copy()
    return cohort


# =============================================================================
# 3. Leakage-Free Preprocessing Pipelines
# =============================================================================
def get_preprocessor(
    feature_list: List[str],
    imputation_strategy: str = "median",
    include_indicators: bool = True,
    scale_continuous: bool = True,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that guarantees zero test-set leakage.
    Imputation statistics and scaling factors are fitted strictly on training folds.
    """
    cat_cols = [f for f in feature_list if f in cfg.CATEGORICAL_VARS]
    bin_cols = [f for f in feature_list if f in cfg.BINARY_VARS]
    ind_cols = [f for f in feature_list if f in cfg.INDICATOR_VARS]
    cont_cols = [f for f in feature_list if f not in cat_cols and f not in bin_cols]

    transformers = []

    # 1. Continuous Features: Imputation + Standardization
    if cont_cols:
        steps = [("imputer", SimpleImputer(strategy=imputation_strategy))]
        if scale_continuous:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("cont", Pipeline(steps), cont_cols))

    # 2. Binary Features: Imputation (Mode)
    if bin_cols:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), bin_cols))

    # 3. Categorical Features: Imputation (Mode) + One-Hot Encoding
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    # 4. Missingness Indicator Flags
    if include_indicators and ind_cols:
        transformers.append(("ind", MissingIndicatorTransformer(target_cols=ind_cols), ind_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_pipeline(
    classifier: BaseEstimator,
    feature_list: List[str],
    imputation_strategy: str = "median",
    include_indicators: bool = True,
    scale_continuous: bool = True,
) -> Pipeline:
    """Helper to assemble a full scikit-learn Pipeline ready for cross-validation."""
    preprocessor = get_preprocessor(
        feature_list=feature_list,
        imputation_strategy=imputation_strategy,
        include_indicators=include_indicators,
        scale_continuous=scale_continuous,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
