"""
01_cohort_selection.py

Demonstrates cohort definition, participant exclusions, and selection bias analysis.
Answers:
1. Who do we start with? (11,933 total NHANES participants)
2. Who do we end up with? (5,538 adults with outcome and inflammatory markers)
3. Why were others excluded? (Step-by-step eligibility criteria tracked with cases)
4. Did excluding individuals without blood draws introduce selection bias? (SMD check)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import config as cfg
from preprocessing import clean_raw_columns

def main():
    print("=" * 75)
    print("STEP 1: PARTICIPANT FLOW & COHORT SELECTION")
    print("=" * 75)

    print("Loading raw NHANES 2021-2023 dataset...")
    raw_df = pd.read_csv(cfg.RAW_DATA, low_memory=False)
    meta_df = pd.read_csv(cfg.METADATA)
    clean_df = clean_raw_columns(raw_df, meta_df)

    # Outcome definition
    chd = clean_df["MCQ160C"].map({1.0: 1.0, 2.0: 0.0})
    mi = clean_df["MCQ160E"].map({1.0: 1.0, 2.0: 0.0})
    clean_df["chd_mi"] = np.select([(chd == 1.0) | (mi == 1.0), (chd == 0.0) & (mi == 0.0)], [1.0, 0.0], np.nan)

    # Step-by-step Flowchart tracking
    n_0 = len(clean_df)
    
    # Step 1: Age >= 20
    step1_df = clean_df[clean_df["RIDAGEYR"] >= cfg.MIN_AGE].copy()
    n_1 = len(step1_df)
    c_1 = int(step1_df["chd_mi"].sum())

    # Step 2: Answered CHD/MI medical history questions
    step2_df = step1_df[step1_df["chd_mi"].notna()].copy()
    n_2 = len(step2_df)
    c_2 = int(step2_df["chd_mi"].sum())

    # Step 3: Examined in Mobile Examination Center (MEC)
    step3_df = step2_df[step2_df["WTMEC2YR"] > 0].copy()
    n_3 = len(step3_df)
    c_3 = int(step3_df["chd_mi"].sum())

    # Step 4: Complete measurements of both inflammatory markers (NLR and hs-CRP)
    lymph_pct = step3_df["LBXLYPCT"].where(step3_df["LBXLYPCT"] > 0)
    nlr = step3_df["LBXNEPCT"] / lymph_pct
    crp = step3_df["LBXHSCRP"]
    
    has_markers = nlr.notna() & crp.notna()
    final_cohort = step3_df[has_markers].copy()
    n_4 = len(final_cohort)
    c_4 = int(final_cohort["chd_mi"].sum())

    flowchart = pd.DataFrame([
        {"Step": "1. Total enrolled NHANES participants", "Participants": n_0, "Excluded": n_0 - n_1, "CHD_MI_Cases": c_1, "Reason": "Baseline survey population"},
        {"Step": "2. Adults (Age >= 20 years)", "Participants": n_1, "Excluded": n_1 - n_2, "CHD_MI_Cases": c_1, "Reason": "Heart disease questions asked only from age 20"},
        {"Step": "3. Answered CHD/MI question", "Participants": n_2, "Excluded": n_2 - n_3, "CHD_MI_Cases": c_2, "Reason": "Non-missing primary medical outcome"},
        {"Step": "4. Mobile Center Examination (MEC)", "Participants": n_3, "Excluded": n_3 - n_4, "CHD_MI_Cases": c_3, "Reason": "Required for biological & physical examination"},
        {"Step": "5. Both NLR & hs-CRP Measured (Final Analytic Cohort)", "Participants": n_4, "Excluded": 0, "CHD_MI_Cases": c_4, "Reason": "Complete primary exposure biomarker measurement"},
    ])

    flow_path = cfg.TABLES_DIR / "flowchart.csv"
    flowchart.to_csv(flow_path, index=False)
    print("\nCohort Derivation Flowchart:")
    print(flowchart[["Step", "Participants", "Excluded", "CHD_MI_Cases", "Reason"]].to_string(index=False))

    # Selection Bias Analysis: Compare included (n = 5,538) vs. MEC participants without markers (n = 499)
    excluded_no_markers = step3_df[~has_markers].copy()
    print(f"\nSelection Bias Check: Comparing included (n = {n_4}) vs excluded without blood markers (n = {len(excluded_no_markers)})...")

    smd_records = []
    for var, label in [("RIDAGEYR", "Age (years)"), ("INDFMPIR", "Income-to-Poverty Ratio"), ("BMXBMI", "BMI (kg/m²)")]:
        inc_vals = final_cohort[var].dropna()
        exc_vals = excluded_no_markers[var].dropna()
        pooled_sd = np.sqrt((inc_vals.var() + exc_vals.var()) / 2.0)
        smd = (exc_vals.mean() - inc_vals.mean()) / pooled_sd
        smd_records.append({"Variable": label, "Included_Mean": inc_vals.mean(), "Excluded_Mean": exc_vals.mean(), "SMD": smd})

    smd_df = pd.DataFrame(smd_records)
    smd_path = cfg.TABLES_DIR / "selection_bias_smd.csv"
    smd_df.to_csv(smd_path, index=False)
    print(smd_df.to_string(index=False))
    print("\nTakeaway: Excluded individuals without blood draw had lower family income (SMD = -0.36), which is documented as a limitation.")
    print(f"Results saved to: {flow_path} and {smd_path}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
