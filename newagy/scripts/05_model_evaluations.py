"""
Step 05: Model Evaluation, Statistical Comparison, and Marker Gain.

Analyzes the 15 outer folds from repeated nested cross-validation:
- Computes mean discrimination (ROC-AUC, PR-AUC) and calibration (Brier score)
- Calculates incremental marker gain (Delta-AUC: M3 - M2) using Nadeau & Bengio (2003) corrected paired t-tests
- Performs pairwise comparisons against baseline Logistic Regression with Bonferroni correction
- Analyzes decision threshold performance (Sensitivity, Specificity, PPV, NPV)
- Investigates generalization gap (Train vs. Test AUC) to evaluate overfitting
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from evaluation import bonferroni_correct, nadeau_bengio_corrected_ttest


def run_model_evaluations():
    start_time = time.time()
    print("=" * 70)
    print("STEP 05: MODEL EVALUATION & STATISTICAL DISCRIMINATION COMPARISON")
    print("=" * 70)

    cv_path = cfg.CV_DIR / "fold_metrics.csv"
    if not cv_path.exists():
        raise FileNotFoundError(f"Missing {cv_path}. Run step 04 first.")

    df_folds = pd.read_csv(cv_path)

    # 1. Main Performance Summary Table (M2 and M3 across algorithms)
    print("1. Computing algorithm discrimination and calibration across 15 folds...")
    summary_rows = []
    # Normalize algorithm name column for consistent matching
    df_folds["algo_clean"] = df_folds["algo"].replace({"kNN": "KNN", "knn": "KNN"})

    for algo in ["LR", "EN", "DT", "KNN", "SVM", "RF"]:
        for fs in ["M2", "M3"]:
            sub = df_folds[(df_folds["algo_clean"] == algo) & (df_folds["fs"] == fs)]
            if len(sub) == 0:
                continue
            mean_auc = sub["auc"].mean()
            sd_auc = sub["auc"].std()
            mean_pr = sub["pr_auc"].mean()
            mean_brier = sub["brier"].mean()
            summary_rows.append({
                "Algorithm": algo,
                "Feature_Set": fs,
                "ROC_AUC_Mean": mean_auc,
                "ROC_AUC_SD": sd_auc,
                "PR_AUC_Mean": mean_pr,
                "Brier_Score": mean_brier,
            })
    main_perf_df = pd.DataFrame(summary_rows)
    main_perf_path = cfg.TABLES_DIR / "main_performance.csv"
    main_perf_df.to_csv(main_perf_path, index=False)
    print(main_perf_df.to_string(index=False))

    # 2. Marker Gain (Delta-AUC = M3 - M2) with Nadeau-Bengio corrected paired t-test
    print("\n2. Statistical comparison of marker gain (M3 vs. M2) per algorithm...")
    gain_rows = []
    p_values_gain = []

    for algo in ["LR", "EN", "DT", "KNN", "SVM", "RF"]:
        m2_sub = df_folds[(df_folds["algo_clean"] == algo) & (df_folds["fs"] == "M2")].sort_values(by=["repeat", "fold"])
        m3_sub = df_folds[(df_folds["algo_clean"] == algo) & (df_folds["fs"] == "M3")].sort_values(by=["repeat", "fold"])

        if len(m2_sub) == 0 or len(m3_sub) == 0:
            continue

        diffs = (m3_sub["auc"].to_numpy() - m2_sub["auc"].to_numpy())
        ttest_res = nadeau_bengio_corrected_ttest(diffs)
        p_values_gain.append(ttest_res["p_value"])

        delta_brier = float(m3_sub["brier"].mean() - m2_sub["brier"].mean())

        gain_rows.append({
            "Algorithm": algo,
            "AUC_M2": m2_sub["auc"].mean(),
            "AUC_M3": m3_sub["auc"].mean(),
            "Delta_AUC": ttest_res["mean_difference"],
            "CI_Lower": ttest_res["ci_lower"],
            "CI_Upper": ttest_res["ci_upper"],
            "Delta_Brier": delta_brier,
            "p_raw": ttest_res["p_value"],
        })

    gain_df = pd.DataFrame(gain_rows)
    gain_df["p_bonferroni"] = bonferroni_correct(p_values_gain)
    gain_path = cfg.TABLES_DIR / "main_marker_gain.csv"
    gain_df.to_csv(gain_path, index=False)
    print("\nIncremental Discrimination Gain from NLR + hs-CRP:")
    print(gain_df[["Algorithm", "AUC_M2", "AUC_M3", "Delta_AUC", "CI_Lower", "CI_Upper", "Delta_Brier", "p_bonferroni"]].to_string(index=False))

    # 3. Model Comparison against Logistic Regression baseline
    print("\n3. Pairwise comparison of complex ML algorithms against Logistic Regression (M3)...")
    lr_m3 = df_folds[(df_folds["algo_clean"] == "LR") & (df_folds["fs"] == "M3")].sort_values(by=["repeat", "fold"])

    algo_vs_lr_rows = []
    p_vals_vs_lr = []
    comp_algos = ["EN", "DT", "KNN", "SVM", "RF"]

    for algo in comp_algos:
        algo_m3 = df_folds[(df_folds["algo_clean"] == algo) & (df_folds["fs"] == "M3")].sort_values(by=["repeat", "fold"])
        if len(algo_m3) == 0:
            continue
        diffs = algo_m3["auc"].to_numpy() - lr_m3["auc"].to_numpy()
        ttest_res = nadeau_bengio_corrected_ttest(diffs)
        p_vals_vs_lr.append(ttest_res["p_value"])

        algo_vs_lr_rows.append({
            "Comparison": f"{algo} vs LR (M3)",
            "Mean_Delta_AUC": ttest_res["mean_difference"],
            "CI_Lower": ttest_res["ci_lower"],
            "CI_Upper": ttest_res["ci_upper"],
            "p_raw": ttest_res["p_value"],
        })

    vs_lr_df = pd.DataFrame(algo_vs_lr_rows)
    vs_lr_df["p_bonferroni"] = bonferroni_correct(p_vals_vs_lr)
    vs_lr_path = cfg.TABLES_DIR / "main_algorithm_vs_lr.csv"
    vs_lr_df.to_csv(vs_lr_path, index=False)
    print(vs_lr_df.to_string(index=False))

    # 4. Standalone Marker Benchmarks (M4, M4-NLR, M4-CRP)
    print("\n4. Standalone Inflammatory Marker Benchmarks (Logistic Regression):")
    m4_sub = df_folds[df_folds["algo"] == "LR"]
    for fs, label in [
        ("M0", "Demographics only (Age, Sex)"),
        ("M1", "+ Sociodemographics & BMI"),
        ("M2", "+ Established Clinical Risk Factors"),
        ("M3", "+ Both Inflammatory Markers"),
        ("M4", "NLR + hs-CRP alone (No risk factors)"),
        ("M4-NLR", "NLR alone"),
        ("M4-CRP", "hs-CRP alone"),
    ]:
        fs_df = m4_sub[m4_sub["fs"] == fs]
        if len(fs_df) > 0:
            print(f"  {fs.ljust(8)} ({label}): AUC = {fs_df['auc'].mean():.3f} (SD = {fs_df['auc'].std():.3f})")

    elapsed = time.time() - start_time
    print(f"\nStep 05 completed successfully in {elapsed:.1f} seconds.\n")


if __name__ == "__main__":
    run_model_evaluations()
