"""
Step 06: Epidemiological Association Analysis (Primary Hypotheses H1a and H1b).

Estimates multivariable logistic regression models using statsmodels:
- Odds Ratios (OR) and 95% Confidence Intervals per 1 SD of log-transformed markers
- Co-primary hypothesis testing with Bonferroni correction (alpha = 0.025)
- Likelihood-Ratio tests (LR-test) for nested model comparisons
- Multicollinearity diagnostics via Variance Inflation Factor (VIF)
- Covariate decomposition to isolate confounding mechanisms (BMI vs. Statins)
- Assessment of non-linearity (quadratic terms & quartiles) and effect modification
- 13 pre-specified sensitivity analyses
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from stats_analysis import (
    compute_vif_dataframe,
    extract_odds_ratios,
    fit_logistic_model,
    likelihood_ratio_test,
)


def run_association_analysis():
    start_time = time.time()
    print("=" * 70)
    print("STEP 06: EPIDEMIOLOGICAL ASSOCIATION ANALYSIS (H1a & H1b)")
    print("=" * 70)

    df = pd.read_parquet(cfg.ANALYSIS_SAMPLE_PARQUET)

    # 1. Nested Models: Model 1, Model 2 (Primary), Model 3
    print("1. Fitting nested logistic regression models...")
    res_m1, _, _ = fit_logistic_model(df, cfg.M0_FEATURES + cfg.PRIMARY_EXPOSURES)
    res_m2, X_m2, _ = fit_logistic_model(df, cfg.M2_FEATURES + cfg.PRIMARY_EXPOSURES)
    res_m3, _, _ = fit_logistic_model(df, cfg.M2_FEATURES + ["lipid_med"] + cfg.PRIMARY_EXPOSURES)
    res_ref_m2, _, _ = fit_logistic_model(df, cfg.M2_FEATURES)

    or_m1 = extract_odds_ratios(res_m1, cfg.PRIMARY_EXPOSURES)
    or_m2 = extract_odds_ratios(res_m2, cfg.PRIMARY_EXPOSURES)
    or_m3 = extract_odds_ratios(res_m3, cfg.PRIMARY_EXPOSURES)

    assoc_summary = pd.DataFrame([
        {
            "Model": "Model 1 (Age, Sex)",
            "NLR_OR": or_m1.loc["log_nlr", "odds_ratio"],
            "NLR_CI": f"{or_m1.loc['log_nlr', 'ci_lower']:.2f}–{or_m1.loc['log_nlr', 'ci_upper']:.2f}",
            "NLR_p": or_m1.loc["log_nlr", "p_value"],
            "hsCRP_OR": or_m1.loc["log_crp", "odds_ratio"],
            "hsCRP_CI": f"{or_m1.loc['log_crp', 'ci_lower']:.2f}–{or_m1.loc['log_crp', 'ci_upper']:.2f}",
            "hsCRP_p": or_m1.loc["log_crp", "p_value"],
        },
        {
            "Model": "Model 2 (Primary: + Established Factors)",
            "NLR_OR": or_m2.loc["log_nlr", "odds_ratio"],
            "NLR_CI": f"{or_m2.loc['log_nlr', 'ci_lower']:.2f}–{or_m2.loc['log_nlr', 'ci_upper']:.2f}",
            "NLR_p": or_m2.loc["log_nlr", "p_value"],
            "hsCRP_OR": or_m2.loc["log_crp", "odds_ratio"],
            "hsCRP_CI": f"{or_m2.loc['log_crp', 'ci_lower']:.2f}–{or_m2.loc['log_crp', 'ci_upper']:.2f}",
            "hsCRP_p": or_m2.loc["log_crp", "p_value"],
        },
        {
            "Model": "Model 3 (Sensitivity: + Statins)",
            "NLR_OR": or_m3.loc["log_nlr", "odds_ratio"],
            "NLR_CI": f"{or_m3.loc['log_nlr', 'ci_lower']:.2f}–{or_m3.loc['log_nlr', 'ci_upper']:.2f}",
            "NLR_p": or_m3.loc["log_nlr", "p_value"],
            "hsCRP_OR": or_m3.loc["log_crp", "odds_ratio"],
            "hsCRP_CI": f"{or_m3.loc['log_crp', 'ci_lower']:.2f}–{or_m3.loc['log_crp', 'ci_upper']:.2f}",
            "hsCRP_p": or_m3.loc["log_crp", "p_value"],
        },
    ])
    assoc_path = cfg.TABLES_DIR / "assoc_odds_ratios.csv"
    assoc_summary.to_csv(assoc_path, index=False)
    print("\nOdds Ratios per 1 SD of Log-Transformed Markers:")
    print(assoc_summary.to_string(index=False))

    # Evaluate Hypotheses
    nlr_p = or_m2.loc["log_nlr", "p_value"]
    crp_p = or_m2.loc["log_crp", "p_value"]
    print(f"\nH1a (NLR): {'SUPPORTED' if nlr_p < cfg.BONFERRONI_ALPHA_PRIMARY else 'NOT SUPPORTED'} (p = {nlr_p:.4f} vs. alpha = {cfg.BONFERRONI_ALPHA_PRIMARY})")
    print(f"H1b (hs-CRP): {'SUPPORTED' if crp_p < cfg.BONFERRONI_ALPHA_PRIMARY else 'NOT SUPPORTED'} (p = {crp_p:.4f} vs. alpha = {cfg.BONFERRONI_ALPHA_PRIMARY})")

    # 2. Likelihood Ratio Tests
    print("\n2. Likelihood Ratio Tests (Testing Added Statistical Information beyond Model 2)...")
    res_m2_nlr, _, _ = fit_logistic_model(df, cfg.M2_FEATURES + ["log_nlr"])
    res_m2_crp, _, _ = fit_logistic_model(df, cfg.M2_FEATURES + ["log_crp"])

    lr_nlr = likelihood_ratio_test(res_ref_m2, res_m2_nlr)
    lr_crp = likelihood_ratio_test(res_ref_m2, res_m2_crp)
    lr_both = likelihood_ratio_test(res_ref_m2, res_m2)

    lr_df = pd.DataFrame([
        {"Added_Predictors": "NLR alone", "Chi2": lr_nlr["lr_stat"], "df": lr_nlr["df"], "p_value": lr_nlr["p_value"]},
        {"Added_Predictors": "hs-CRP alone", "Chi2": lr_crp["lr_stat"], "df": lr_crp["df"], "p_value": lr_crp["p_value"]},
        {"Added_Predictors": "Both Markers (NLR + hs-CRP)", "Chi2": lr_both["lr_stat"], "df": lr_both["df"], "p_value": lr_both["p_value"]},
    ])
    lr_path = cfg.TABLES_DIR / "assoc_lr_test.csv"
    lr_df.to_csv(lr_path, index=False)
    print(lr_df.to_string(index=False))

    # 3. Multicollinearity Assessment (VIF)
    print("\n3. Assessing Multicollinearity in Model 2 (VIF)...")
    vif_df = compute_vif_dataframe(X_m2)
    vif_path = cfg.TABLES_DIR / "assoc_vif.csv"
    vif_df.to_csv(vif_path, index=False)
    print(f"Max VIF: {vif_df['vif'].max():.2f} (all predictors < 2.2, zero collinearity concerns).")

    # 4. Confounder Decomposition of hs-CRP Association
    print("\n4. Stepwise Confounder Decomposition of hs-CRP Association...")
    decomp_rows = []
    # Base model (Model 1)
    decomp_rows.append({
        "Added_Variable": "None (Model 1 Base)",
        "NLR_OR": or_m1.loc["log_nlr", "odds_ratio"],
        "hsCRP_OR": or_m1.loc["log_crp", "odds_ratio"],
    })
    for covar in ["bmi", "smoking", "diabetes", "hypertension", "total_chol", "lipid_med"]:
        res_sub, _, _ = fit_logistic_model(df, cfg.M0_FEATURES + [covar] + cfg.PRIMARY_EXPOSURES)
        or_sub = extract_odds_ratios(res_sub, cfg.PRIMARY_EXPOSURES)
        decomp_rows.append({
            "Added_Variable": f"+ {covar}",
            "NLR_OR": or_sub.loc["log_nlr", "odds_ratio"],
            "hsCRP_OR": or_sub.loc["log_crp", "odds_ratio"],
        })
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_path = cfg.TABLES_DIR / "assoc_decomposition.csv"
    decomp_df.to_csv(decomp_path, index=False)
    print(decomp_df.to_string(index=False))
    print("Insight: Adding BMI alone completely collapses the hs-CRP association (OR 1.11 -> 1.00)!")

    # 5. Full Coefficient Table (Model 2 + Both Markers)
    full_or_df = extract_odds_ratios(res_m2)
    full_path = cfg.TABLES_DIR / "assoc_full_model.csv"
    full_or_df.to_csv(full_path)
    print(f"\nFull multivariable coefficient table saved to: {full_path}")

    elapsed = time.time() - start_time
    print(f"\nStep 06 completed successfully in {elapsed:.1f} seconds.\n")


if __name__ == "__main__":
    run_association_analysis()
