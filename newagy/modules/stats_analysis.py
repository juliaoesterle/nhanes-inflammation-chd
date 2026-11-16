"""
Logistic regression (statsmodels) for the association question H1a / H1b:
odds ratio per 1 SD of the log marker, 95% CI, Wald p-value; survey-weighted version;
standardized mean difference for the selection-bias check.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

RACES = ["mexican_american", "other_hispanic", "nh_black", "nh_asian", "other_multi"]  # ref: nh_white
MODEL1 = ["age", "male"]
RISK = ([f"race_{r}" for r in RACES]
        + ["income_pir", "income_missing", "bmi", "smk_former", "smk_current", "diabetes",
           "hypertension", "total_chol", "hdl_chol", "egfr"])
MODEL2 = MODEL1 + RISK
MODEL3 = MODEL2 + ["lipid_med"]
MODELS = {"Model 1": MODEL1, "Model 2": MODEL2, "Model 3": MODEL3}
CONT = ["age", "income_pir", "bmi", "total_chol", "hdl_chol", "egfr"]
BIN = (["male", "diabetes", "hypertension", "lipid_med", "smk_former", "smk_current"]
       + [f"race_{r}" for r in RACES])
MISSING = {"median": "Median + indicator (primary)", "complete_case": "Complete cases",
           "drop_income": "Income column dropped"}


def prepare(df):
    """Dummies for race/ethnicity and smoking; markers per 1 SD of their log."""
    d = df.copy().reset_index(drop=True)
    for r in RACES:
        d[f"race_{r}"] = (d["race_eth"] == r).astype(float)
    for level in ("former", "current"):
        d[f"smk_{level}"] = np.where(d["smoking"].isna(), np.nan, (d["smoking"] == level).astype(float))
    d["income_missing"] = d["income_pir"].isna().astype(float)
    for m, col in (("nlr", "log_nlr"), ("crp", "log_crp")):
        d[f"{m}_sd"] = (d[col] - d[col].mean()) / d[col].std()
    return d


def impute(d, how):
    """complete_case: only people without a gap in the Model-2 covariates;
    otherwise median (continuous) / most frequent value (0/1) for the gaps."""
    d = d.copy()
    if how == "complete_case":
        model2_cols = [c for c in CONT + BIN if c != "lipid_med"]
        return d[d[model2_cols].notna().all(axis=1)].copy()
    for c in CONT:
        d[c] = d[c].fillna(d[c].median())
    for c in BIN:
        d[c] = d[c].fillna(d[c].mode().iloc[0])
    return d


def terms_for(model, how="median"):
    t = list(MODELS[model])
    if how != "median":
        t = [x for x in t if x != "income_missing"]
    if how == "drop_income":
        t = [x for x in t if x != "income_pir"]
    return t


def used_rows(d, y, terms):
    return d.dropna(subset=[y] + list(terms))


def svy_logit(y, X, w, strata, psu):
    """Survey-weighted logistic regression: weighted ML estimate and design-based variance
    (Taylor linearisation, PSUs nested in strata, like R's survey::svyglm)."""
    y = np.asarray(y, float)
    Xm = np.asarray(X, float)
    w = np.asarray(w, float)
    w = w / w.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(y, Xm, family=sm.families.Binomial(), freq_weights=w).fit()
    beta = np.asarray(res.params)
    p = 1 / (1 + np.exp(-Xm @ beta))
    Hinv = np.linalg.inv((Xm * (w * p * (1 - p))[:, None]).T @ Xm)
    scores = pd.DataFrame(Xm * (w * (y - p))[:, None])
    scores["h"] = np.asarray(strata)
    scores["i"] = np.asarray(psu)
    meat = np.zeros((Xm.shape[1], Xm.shape[1]))
    n_psu, n_strata = 0, 0
    for _, g in scores.groupby("h"):
        z = g.groupby("i").sum().drop(columns="h", errors="ignore").to_numpy()
        nh = z.shape[0]
        n_psu += nh
        n_strata += 1
        if nh < 2:
            continue
        zc = z - z.mean(axis=0)
        meat += nh / (nh - 1) * zc.T @ zc
    return beta, Hinv @ meat @ Hinv, n_psu - n_strata


def fit(d, y, terms, weighted=False):
    dd = used_rows(d, y, terms)
    X = sm.add_constant(dd[terms].astype(float), has_constant="add")
    if weighted:
        beta, cov, dfree = svy_logit(dd[y], X, dd["weight_mec"], dd["strata"], dd["psu"])
        return (pd.Series(beta, index=X.columns), pd.Series(np.sqrt(np.diag(cov)), index=X.columns),
                dfree, None)
    res = sm.Logit(dd[y].astype(float), X).fit(disp=0, maxiter=200)
    return res.params, res.bse, np.inf, res


def odds_ratio(d, y, terms, coef, weighted=False):
    b, se, dfree, _ = fit(d, y, terms, weighted)
    q = stats.t.ppf(0.975, dfree) if np.isfinite(dfree) else stats.norm.ppf(0.975)
    z = b[coef] / se[coef]
    p = 2 * (stats.t.sf(abs(z), dfree) if np.isfinite(dfree) else stats.norm.sf(abs(z)))
    return {"or": np.exp(b[coef]), "lo": np.exp(b[coef] - q * se[coef]),
            "hi": np.exp(b[coef] + q * se[coef]), "p": p}


def smd(a, b):
    """Standardized mean difference; |SMD| > 0.1 = relevant difference."""
    a, b = pd.Series(a).dropna().astype(float), pd.Series(b).dropna().astype(float)
    den = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / den if den > 0 else 0.0
