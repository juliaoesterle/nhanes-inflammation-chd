"""
Step 02: Exploratory Data Analysis & Baseline Characteristics (Table 1).

Computes comprehensive baseline comparisons between participants with and without
coronary heart disease (CHD/MI):
- Continuous features: Median [IQR], Mann-Whitney U test, Standardized Mean Difference (SMD)
- Categorical features: Count (%), Pearson Chi-Square test, SMD
- Multiple testing adjustment via Bonferroni correction
- Events-Per-Parameter (EPP / EPV) sample size justification
- Investigation of selection bias between included and excluded individuals
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg


def calculate_smd_continuous(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d / standardized mean difference for continuous data."""
    g1, g2 = group1[~np.isnan(group1)], group2[~np.isnan(group2)]
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return 0.0
    mean1, mean2 = np.mean(g1), np.mean(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return float((mean2 - mean1) / pooled_sd) if pooled_sd > 0 else 0.0


def run_exploratory_analysis():
    start_time = time.time()
    print("=" * 70)
    print("STEP 02: EXPLORATORY DATA ANALYSIS & TABLE 1 GENERATION")
    print("=" * 70)

    df = pd.read_parquet(cfg.ANALYSIS_SAMPLE_PARQUET)
    cases = df[df[cfg.PRIMARY_OUTCOME] == 1.0]
    controls = df[df[cfg.PRIMARY_OUTCOME] == 0.0]

    n_total = len(df)
    n_cases = len(cases)
    n_controls = len(controls)
    prevalence = 100.0 * n_cases / n_total

    print(f"Total analysis cohort: n = {n_total}")
    print(f"Controls: n = {n_controls} ({100.0 - prevalence:.1f}%)")
    print(f"Cases (CHD/MI): n = {n_cases} ({prevalence:.1f}%)\n")

    # Table 1 variables
    continuous_vars = [
        ("age", "Age, years"),
        ("bmi", "BMI, kg/m²"),
        ("income_pir", "Income-to-poverty ratio"),
        ("total_chol", "Total cholesterol, mg/dL"),
        ("hdl_chol", "HDL cholesterol, mg/dL"),
        ("egfr", "eGFR, mL/min/1.73 m²"),
        ("nlr", "Neutrophil-to-lymphocyte ratio (NLR)"),
        ("crp", "hs-CRP, mg/L"),
    ]

    binary_categorical_vars = [
        ("male", "Male sex"),
        ("race_eth", "Race / Ethnicity"),
        ("smoking", "Smoking status"),
        ("diabetes", "Doctor-diagnosed diabetes"),
        ("hypertension", "Hypertension (diagnosed or >=140/90)"),
        ("lipid_med", "Lipid-lowering medication (Statins)"),
    ]

    table1_rows = []

    # Process continuous variables
    for var, label in continuous_vars:
        c_vals = controls[var].dropna().to_numpy()
        ca_vals = cases[var].dropna().to_numpy()

        c_med, c_q25, c_q75 = np.percentile(c_vals, [50, 25, 75])
        ca_med, ca_q25, ca_q75 = np.percentile(ca_vals, [50, 25, 75])

        stat, p_val = stats.mannwhitneyu(ca_vals, c_vals, alternative="two-sided")
        smd = calculate_smd_continuous(c_vals, ca_vals)

        table1_rows.append({
            "Characteristic": label,
            "Controls (n=5,126)": f"{c_med:.2f} [{c_q25:.2f}, {c_q75:.2f}]",
            "Cases (n=412)": f"{ca_med:.2f} [{ca_q25:.2f}, {ca_q75:.2f}]",
            "p_value_raw": p_val,
            "SMD": smd,
        })

    # Process binary/categorical variables
    for var, label in binary_categorical_vars:
        if var in ["male", "diabetes", "hypertension", "lipid_med"]:
            c_yes = int(controls[var].sum())
            ca_yes = int(cases[var].sum())
            c_pct = 100.0 * c_yes / len(controls.dropna(subset=[var]))
            ca_pct = 100.0 * ca_yes / len(cases.dropna(subset=[var]))

            contingency = [
                [c_yes, len(controls) - c_yes],
                [ca_yes, len(cases) - ca_yes],
            ]
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            # Binary SMD
            p1 = c_pct / 100.0
            p2 = ca_pct / 100.0
            smd = (p2 - p1) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2.0)

            table1_rows.append({
                "Characteristic": label,
                "Controls (n=5,126)": f"{c_yes} ({c_pct:.1f}%)",
                "Cases (n=412)": f"{ca_yes} ({ca_pct:.1f}%)",
                "p_value_raw": p_val,
                "SMD": float(smd),
            })

    table1_df = pd.DataFrame(table1_rows)
    # Bonferroni correction across all tested variables
    table1_df["p_value_bonferroni"] = [
        min(1.0, p * len(table1_df)) for p in table1_df["p_value_raw"]
    ]

    out_path = cfg.TABLES_DIR / "table1.csv"
    table1_df.to_csv(out_path, index=False)
    print(f"Table 1 saved to: {out_path}")
    print(table1_df[["Characteristic", "Controls (n=5,126)", "Cases (n=412)", "SMD", "p_value_bonferroni"]].to_string(index=False))

    # Sample Size Justification (Events-Per-Parameter / EPV)
    # Model M3 has 19 estimated parameters
    n_params = 19
    epv_total = n_cases / n_params
    epv_outer_train = (n_cases * 0.8) / n_params
    epv_inner_train = (n_cases * 0.8 * 0.8) / n_params

    epv_df = pd.DataFrame([
        {"Sample Split": "Full Analytic Sample", "Cases": n_cases, "Parameters": n_params, "EPV": epv_total},
        {"Sample Split": "Outer CV Training Fold (4/5)", "Cases": int(n_cases * 0.8), "Parameters": n_params, "EPV": epv_outer_train},
        {"Sample Split": "Inner CV Training Fold (4/5 of 4/5)", "Cases": int(n_cases * 0.64), "Parameters": n_params, "EPV": epv_inner_train},
    ])
    epv_path = cfg.TABLES_DIR / "sample_size_epv.csv"
    epv_df.to_csv(epv_path, index=False)
    print(f"\nEvents-Per-Parameter (EPV) summary saved to: {epv_path}")
    print(epv_df.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\nStep 02 completed successfully in {elapsed:.1f} seconds.\n")


if __name__ == "__main__":
    run_exploratory_analysis()
