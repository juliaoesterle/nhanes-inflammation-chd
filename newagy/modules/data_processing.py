"""
Data Processing & Cohort Engineering Module.

Handles codebook-based raw data cleaning, calculation of established clinical
biomarkers (including race-free CKD-EPI 2021 eGFR), composite outcome definitions,
log-transformation of skewed inflammatory markers, and cohort derivation with
transparent participant flow tracking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Regular expression matching NHANES non-response value labels
NON_RESPONSE_REGEX = re.compile(
    r"^(refused|don.?t know|don\W*t know|missing|sp refusal|participant refusal)$",
    re.IGNORECASE,
)

SAS_ZERO_EPSILON = 1e-70


def parse_metadata_value_codes(spec: str | float) -> Tuple[List[Tuple[float, float]], Dict[float, str]]:
    """Parse a metadata `value_codes` entry into numeric ranges and discrete codes."""
    ranges, codes = [], {}
    if not isinstance(spec, str):
        return ranges, codes
    for part in spec.split(" | "):
        part = part.strip()
        if not part or "=" not in part:
            continue
        code_str, label = part.split("=", 1)
        code_str = code_str.strip()
        label = label.strip()

        range_match = re.match(r"^(-?[\d\.]+)\s+to\s+(-?[\d\.]+)$", code_str, re.IGNORECASE)
        if range_match:
            try:
                lo = float(range_match.group(1))
                hi = float(range_match.group(2))
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


def clean_raw_data(raw_csv_path: Path, metadata_csv_path: Path) -> pd.DataFrame:
    """
    Clean the course raw CSV dataset against documented metadata codes.
    
    Replaces:
    - SAS zero artifacts (5.39e-79 -> 0.0)
    - Documented non-response codes (refused, don't know -> NaN)
    - Undocumented extreme sentinel values -> NaN
    - Non-numeric garbage strings -> NaN
    """
    raw_df = pd.read_csv(raw_csv_path, low_memory=False)
    col_name = "column_name" if "column_name" in pd.read_csv(metadata_csv_path, nrows=1).columns else "variable_name"
    meta_df = pd.read_csv(metadata_csv_path).set_index(col_name)

    cleaned_df = raw_df.copy()

    for col in cleaned_df.columns:
        if col not in meta_df.index:
            continue

        meta_row = meta_df.loc[col]
        spec = meta_row.get("value_codes", "")
        ranges, codes = parse_metadata_value_codes(spec)

        non_resp_codes = {c for c, lbl in codes.items() if NON_RESPONSE_REGEX.match(lbl)}
        valid_discrete_codes = set(codes.keys()) - non_resp_codes

        series = pd.to_numeric(cleaned_df[col], errors="coerce")

        # Map SAS zero representation
        zero_mask = series.abs() < SAS_ZERO_EPSILON
        series = series.mask(zero_mask, 0.0)

        # Replace non-response codes
        if non_resp_codes:
            series = series.mask(series.isin(non_resp_codes), np.nan)

        # Sentinel checks: if documented ranges or valid discrete codes exist
        if ranges or valid_discrete_codes:
            in_range = pd.Series(False, index=series.index)
            for lo, hi in ranges:
                in_range |= (series >= lo) & (series <= hi)
            in_discrete = series.isin(valid_discrete_codes)
            is_valid = in_range | in_discrete | series.isna()
            series = series.mask(~is_valid, np.nan)

        cleaned_df[col] = series

    return cleaned_df


def calculate_ckd_epi_2021(serum_creatinine: pd.Series, age: pd.Series, is_male: pd.Series) -> pd.Series:
    """
    Calculate estimated Glomerular Filtration Rate (eGFR) using the official
    2021 race-free CKD-EPI creatinine equation (Inker et al., NEJM 2021).
    
    Formula:
    eGFR = 142 * min(Scr/kappa, 1)^alpha * max(Scr/kappa, 1)^(-1.200) * 0.9938^Age * (1.012 if female)
    where:
      male:   kappa = 0.9, alpha = -0.302
      female: kappa = 0.7, alpha = -0.241
    """
    scr = serum_creatinine
    kappa = np.where(is_male == 1.0, 0.9, 0.7)
    alpha = np.where(is_male == 1.0, -0.302, -0.241)
    gender_factor = np.where(is_male == 1.0, 1.0, 1.012)

    scr_k = scr / kappa
    egfr = (
        142.0
        * np.minimum(scr_k, 1.0) ** alpha
        * np.maximum(scr_k, 1.0) ** (-1.200)
        * (0.9938 ** age)
        * gender_factor
    )
    return pd.Series(egfr, index=serum_creatinine.index)


def binary_yes_no(series: pd.Series) -> pd.Series:
    """Maps NHANES binary response: 1.0 -> 1.0 (Yes), 2.0 -> 0.0 (No), otherwise NaN."""
    return series.map({1.0: 1.0, 2.0: 0.0})


def calculate_weekly_activity(freq: pd.Series, unit: pd.Series, duration_min: pd.Series) -> pd.Series:
    """Converts activity frequency, unit (D/W/M/Y), and minutes into weekly total minutes."""
    unit_multiplier = unit.map({"D": 7.0, "W": 1.0, "M": 7.0 / 30.44, "Y": 7.0 / 365.25})
    total = freq * unit_multiplier * duration_min
    return total.where(freq != 0, 0.0)


def build_analysis_variables(clean_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform cleaned NHANES survey items into standardized epidemiological variables.
    """
    df = pd.DataFrame({"SEQN": clean_df["SEQN"].astype("int64")})

    # --- Outcomes (Doctor-Diagnosed CHD / MI / Angina) ---
    chd = binary_yes_no(clean_df["MCQ160C"])
    mi = binary_yes_no(clean_df["MCQ160E"])
    angina = binary_yes_no(clean_df["MCQ160D"])

    df["chd"] = chd
    df["mi"] = mi
    df["angina"] = angina
    # Primary composite outcome: CHD or MI
    df["chd_mi"] = np.select([(chd == 1.0) | (mi == 1.0), (chd == 0.0) & (mi == 0.0)], [1.0, 0.0], np.nan)
    # Sensitivity outcomes
    df["chd_mi_angina"] = np.select(
        [(chd == 1.0) | (mi == 1.0) | (angina == 1.0), (chd == 0.0) & (mi == 0.0) & (angina == 0.0)],
        [1.0, 0.0],
        np.nan,
    )
    df["mi_strict"] = np.select([mi == 1.0, (mi == 0.0) & (chd == 0.0)], [1.0, 0.0], np.nan)
    df["chd_strict"] = np.select([chd == 1.0, (chd == 0.0) & (mi == 0.0)], [1.0, 0.0], np.nan)

    # --- Inflammatory Biomarkers (Exposures) ---
    df["crp"] = clean_df["LBXHSCRP"]
    df["log_crp"] = np.log(clean_df["LBXHSCRP"])
    df["crp_above_10"] = (clean_df["LBXHSCRP"] > 10.0).astype(float).where(clean_df["LBXHSCRP"].notna())

    lymph_pct = clean_df["LBXLYPCT"].where(clean_df["LBXLYPCT"] > 0)
    df["nlr"] = clean_df["LBXNEPCT"] / lymph_pct
    df["log_nlr"] = np.log(df["nlr"])

    # Alternative NLR based on absolute leukocyte counts
    lymph_cnt = clean_df["LBDLYMNO"].where(clean_df["LBDLYMNO"] > 0)
    df["log_nlr_counts"] = np.log(clean_df["LBDNENO"] / lymph_cnt)

    # --- Demographics & Socio-Economic Status ---
    df["age"] = clean_df["RIDAGEYR"]
    df["male"] = clean_df["RIAGENDR"].map({1.0: 1.0, 2.0: 0.0})
    df["race_eth"] = clean_df["RIDRETH3"].map({
        1.0: "mexican_american",
        2.0: "other_hispanic",
        3.0: "nh_white",
        4.0: "nh_black",
        6.0: "nh_asian",
        7.0: "other_multi",
    })
    df["income_pir"] = clean_df["INDFMPIR"]
    df["education"] = clean_df["DMDEDUC2"]

    # --- Established Clinical Risk Factors ---
    df["bmi"] = clean_df["BMXBMI"]

    # Smoking status: never (smq020==2), former (smq020==1 & smq040==3), current (smq020==1 & smq040 in [1,2])
    smq020, smq040 = clean_df["SMQ020"], clean_df["SMQ040"]
    df["smoking"] = np.select(
        [smq020 == 2.0, (smq020 == 1.0) & (smq040 == 3.0), (smq020 == 1.0) & smq040.isin([1.0, 2.0])],
        ["never", "former", "current"],
        default=None,
    )
    df["smoking"] = df["smoking"].where(df["smoking"].notna(), None)

    # Diabetes: diagnosed by doctor (DIQ010 == 1)
    df["diabetes"] = binary_yes_no(clean_df["DIQ010"])

    # Hypertension: diagnosed by doctor OR measured SBP >= 140 OR measured DBP >= 90
    bp_diag = binary_yes_no(clean_df["BPQ020"])
    measured_htn = (clean_df["BPXOSY1"] >= 140.0) | (clean_df["BPXODI1"] >= 90.0)
    df["hypertension"] = np.select(
        [(bp_diag == 1.0) | (measured_htn == True), (bp_diag == 0.0) & (measured_htn == False)],
        [1.0, 0.0],
        np.nan,
    )

    df["total_chol"] = clean_df["LBXTC"]
    df["hdl_chol"] = clean_df["LBDHDD"]
    df["egfr"] = calculate_ckd_epi_2021(clean_df["LBXSCR"], df["age"], df["male"])

    # --- Medications (Post-treatment / Confounding indicators) ---
    told_htn = binary_yes_no(clean_df["BPQ020"])
    df["lipid_med"] = binary_yes_no(clean_df["BPQ101D"]) if "BPQ101D" in clean_df else binary_yes_no(clean_df["BPQ100D"])
    df["bp_med"] = np.where(told_htn == 0, 0.0, binary_yes_no(clean_df["BPQ150"])) if "BPQ150" in clean_df else np.nan

    # --- Survey Weights & Study Design Variables ---
    df["wtmec2yr"] = clean_df["WTMEC2YR"]
    df["sdmvpsu"] = clean_df["SDMVPSU"]
    df["sdmvstra"] = clean_df["SDMVSTRA"]

    # --- Auxiliary S3 Candidate Pool Features ---
    df["waist"] = clean_df["BMXWAIST"]
    df["sbp"] = clean_df["BPXOSY1"]
    df["dbp"] = clean_df["BPXODI1"]
    df["pulse"] = clean_df["BPXOPLS1"]
    df["told_high_chol"] = binary_yes_no(clean_df["BPQ080"])
    df["log_tg"] = np.log(clean_df["LBXSTR"].where(clean_df["LBXSTR"] > 0))
    df["glucose"] = clean_df["LBXSGL"] if "LBXSGL" in clean_df else clean_df.get("LBXGLU", np.nan)
    df["uric_acid"] = clean_df["LBXSUA"]
    df["albumin"] = clean_df["LBXSAL"]
    df["log_ggt"] = np.log(clean_df["LBXSGTSI"].where(clean_df["LBXSGTSI"] > 0))
    df["log_alt"] = np.log(clean_df["LBXSATSI"].where(clean_df["LBXSATSI"] > 0)) if "LBXSATSI" in clean_df else np.nan
    df["log_uacr"] = np.log1p(clean_df["URDACT"].where(clean_df["URDACT"] > 0))
    df["log_cotinine"] = np.log(clean_df["LBXCOT"].where(clean_df["LBXCOT"] > 0))
    df["wbc"] = clean_df["LBXWBCSI"].where(clean_df["LBXWBCSI"] < 100)
    df["hemoglobin"] = clean_df["LBXHGB"]
    df["platelets"] = clean_df["LBXPLTSI"]
    df["rdw"] = clean_df["LBXRDW"]
    df["monocyte_pct"] = clean_df["LBXMOPCT"]
    df["eosinophil_pct"] = clean_df["LBXEOPCT"]
    df["vitamin_d"] = clean_df["LBXVIDMS"]
    df["sleep_hours"] = clean_df["SLD012"]
    df["sedentary_min"] = clean_df["PAD680"]

    # Activity units from raw text columns
    if "PAD790Q" in clean_df and "PAD800" in clean_df and "PAD790U" in raw_df:
        moderate = calculate_weekly_activity(clean_df["PAD790Q"], raw_df["PAD790U"], clean_df["PAD800"])
        vigorous = calculate_weekly_activity(clean_df["PAD810Q"], raw_df["PAD810U"], clean_df["PAD820"]) if "PAD810Q" in clean_df and "PAD820" in clean_df and "PAD810U" in raw_df else 0.0
        df["activity_min_wk"] = moderate + 2.0 * vigorous
        df["log_activity"] = np.log1p(df["activity_min_wk"])
    else:
        df["log_activity"] = np.nan

    # Alcohol frequency per week
    alq_map = {0: 0.0, 1: 7.0, 2: 5.5, 3: 3.5, 4: 2.0, 5: 1.0, 6: 2.5 * 12 / 52, 7: 12 / 52, 8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52}
    drinks_wk = clean_df["ALQ121"].map(alq_map) * clean_df["ALQ130"].fillna(0)
    df["log_drinks_wk"] = np.log1p(drinks_wk)

    # PHQ-9 Depression Screener Score
    phq_cols = [f"DPQ0{i}0" for i in range(1, 10)]
    valid_phq = clean_df[phq_cols].isin([0.0, 1.0, 2.0, 3.0])
    df["phq9"] = clean_df[phq_cols].where(valid_phq).sum(axis=1, min_count=7)

    return df


def generate_cohort_and_flowchart(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply sequential eligibility criteria and construct the study sample flowchart.
    
    Inclusion criteria:
    1. Total enrolled participants (11,933)
    2. Adults aged >= 20 years (7,809)
    3. Responded to CHD / MI medical conditions questions (7,773)
    4. Attended Mobile Examination Center [MEC] examination (6,037)
    5. Complete measurement of both NLR and hs-CRP inflammatory biomarkers (5,538)
    """
    total_n = len(df)
    total_cases = int(df["chd_mi"].sum()) if "chd_mi" in df else np.nan

    # Step 1: Age >= 20
    adults = df[df["age"] >= 20].copy()
    adults_n = len(adults)
    adults_cases = int(adults["chd_mi"].sum())

    # Step 2: Answered outcome questions
    answered_outcome = adults[adults["chd_mi"].notna()].copy()
    answered_n = len(answered_outcome)
    answered_cases = int(answered_outcome["chd_mi"].sum())

    # Step 3: Attended MEC
    mec_attended = answered_outcome[answered_outcome["wtmec2yr"] > 0].copy()
    mec_n = len(mec_attended)
    mec_cases = int(mec_attended["chd_mi"].sum())

    # Step 4: Valid measurements of both markers
    has_markers = (mec_attended["nlr"].notna()) & (mec_attended["crp"].notna())
    analysis_sample = mec_attended[has_markers].copy()
    analysis_n = len(analysis_sample)
    analysis_cases = int(analysis_sample["chd_mi"].sum())

    flowchart_data = [
        {"step": "Total enrolled participants", "n": total_n, "cases": adults_cases},
        {"step": "Adults (age >= 20 years)", "n": adults_n, "cases": adults_cases},
        {"step": "Answered CHD/MI medical question", "n": answered_n, "cases": answered_cases},
        {"step": "Examined in Mobile Examination Center (MEC)", "n": mec_n, "cases": mec_cases},
        {"step": "Both NLR and hs-CRP measured (Analysis Sample)", "n": analysis_n, "cases": analysis_cases},
    ]
    flowchart_df = pd.DataFrame(flowchart_data)

    return analysis_sample, answered_outcome, flowchart_df
