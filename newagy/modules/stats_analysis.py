"""
Statistical & Epidemiological Association Analysis Module.

Implements statsmodels logistic regression routines for answering primary
research questions H1a and H1b:
- Estimation of adjusted Odds Ratios (OR) per 1 SD of log-transformed markers
- Likelihood Ratio Tests (LR-test) for evaluating added statistical information
- Multicollinearity diagnostics via Variance Inflation Factors (VIF)
- Non-linearity evaluation (quadratic terms & quartiles)
- Interaction testing (effect modification by sex or lipid-lowering drugs)
- Single-covariate decomposition (identifying BMI as the confounder of hs-CRP)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config as cfg


def prepare_design_matrix(
    df: pd.DataFrame,
    features: List[str],
    add_missing_indicator: bool = True,
    standardize_continuous: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs clean design matrix X and target y for statsmodels.
    Handles median imputation and missingness indicators consistently.
    """
    y = df[cfg.PRIMARY_OUTCOME].copy()
    X = pd.DataFrame(index=df.index)

    for feat in features:
        if feat in cfg.CATEGORICAL_VARS:
            # Create dummy variables with explicit reference category
            dummies = pd.get_dummies(df[feat], prefix=feat, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
        elif feat in cfg.BINARY_VARS:
            val = df[feat].fillna(df[feat].mode()[0])
            X[feat] = val.astype(float)
        else:
            # Continuous feature
            val = df[feat].copy()
            if add_missing_indicator and feat in cfg.INDICATOR_VARS:
                X[f"{feat}_missing"] = val.isna().astype(float)
            # Impute median
            val_imp = val.fillna(val.median())
            if standardize_continuous:
                sd = val_imp.std()
                val_imp = (val_imp - val_imp.mean()) / (sd if sd > 0 else 1.0)
            X[feat] = val_imp

    X = sm.add_constant(X, has_constant="add")
    return X, y


def fit_logistic_model(
    df: pd.DataFrame,
    features: List[str],
    add_missing_indicator: bool = True,
    standardize_continuous: bool = True,
):
    """Fits statsmodels Logit model and returns fitted results."""
    X, y = prepare_design_matrix(
        df=df,
        features=features,
        add_missing_indicator=add_missing_indicator,
        standardize_continuous=standardize_continuous,
    )
    model = sm.Logit(y, X)
    results = model.fit(disp=False, maxiter=200)
    return results, X, y


def extract_odds_ratios(results, feature_names: List[str] | None = None) -> pd.DataFrame:
    """Extracts Odds Ratios, 95% Confidence Intervals, and p-values from Logit results."""
    params = results.params
    conf = results.conf_int()
    pvalues = results.pvalues

    or_df = pd.DataFrame({
        "estimate": params,
        "odds_ratio": np.exp(params),
        "ci_lower": np.exp(conf[0]),
        "ci_upper": np.exp(conf[1]),
        "p_value": pvalues,
    })
    if feature_names:
        matched = [f for f in feature_names if f in or_df.index]
        return or_df.loc[matched]
    return or_df


def likelihood_ratio_test(res_restricted, res_full) -> Dict[str, float]:
    """
    Performs Likelihood Ratio Test comparing restricted model to full model.
    Test statistic: LR = 2 * (LL_full - LL_restricted) ~ Chi2(df_diff)
    """
    ll_null = res_restricted.llf
    ll_alt = res_full.llf
    df_null = res_restricted.df_model
    df_alt = res_full.df_model

    lr_stat = 2.0 * (ll_alt - ll_null)
    df_diff = df_alt - df_null
    p_val = float(sm.stats.stattools.stats.chi2.sf(lr_stat, df_diff))

    return {
        "lr_stat": float(lr_stat),
        "df": int(df_diff),
        "p_value": p_val,
        "ll_restricted": float(ll_null),
        "ll_full": float(ll_alt),
    }


def compute_vif_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """Calculates Variance Inflation Factors for design matrix columns (excluding constant)."""
    cols = [c for c in X.columns if c != "const"]
    X_sub = X[cols].dropna()

    vifs = [variance_inflation_factor(X_sub.values, i) for i in range(X_sub.shape[1])]
    return pd.DataFrame({"variable": cols, "vif": vifs}).sort_values(by="vif", ascending=False)
