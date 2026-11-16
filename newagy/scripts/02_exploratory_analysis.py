"""
Step 02: description of the sample.
- Table 1: cases vs controls (median [IQR] / n (%), Mann-Whitney U / chi-square, SMD, Bonferroni)
- skewness of the two markers, raw and log
- selection bias: excluded adults vs the analysis sample (SMD)
- why is income missing? (logistic regression, MCAR / MAR)
- events per parameter of the main model M3
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from pipeline_factory import make_preprocessor
from sklearn.pipeline import Pipeline
from stats_analysis import smd

CONTINUOUS = [("age", "Age, years"), ("bmi", "BMI, kg/m2"), ("income_pir", "Income-to-poverty ratio"),
              ("total_chol", "Total cholesterol, mg/dL"), ("hdl_chol", "HDL cholesterol, mg/dL"),
              ("egfr", "eGFR, mL/min/1.73 m2"), ("nlr", "NLR"), ("crp", "hs-CRP, mg/L")]
BINARY = [("male", "Male sex"), ("nh_white", "Non-Hispanic White"), ("current_smoker", "Current smoker"),
          ("diabetes", "Diabetes"), ("hypertension", "Hypertension"),
          ("lipid_med", "Lipid-lowering medication")]


def add_dummies(d):
    d = d.copy()
    d["nh_white"] = (d["race_eth"] == "nh_white").astype(float)
    d["nh_black"] = (d["race_eth"] == "nh_black").astype(float)
    d["current_smoker"] = (d["smoking"] == "current").astype(float).where(d["smoking"].notna())
    return d


def table1(df):
    cases, controls = df[df[cfg.PRIMARY_OUTCOME] == 1], df[df[cfg.PRIMARY_OUTCOME] == 0]
    rows = []
    for var, label in CONTINUOUS:
        a, b = controls[var].dropna(), cases[var].dropna()
        q0, q1 = np.percentile(a, [25, 50, 75]), np.percentile(b, [25, 50, 75])
        rows.append({"characteristic": label,
                     f"controls (n={len(controls)})": f"{q0[1]:.2f} [{q0[0]:.2f}, {q0[2]:.2f}]",
                     f"cases (n={len(cases)})": f"{q1[1]:.2f} [{q1[0]:.2f}, {q1[2]:.2f}]",
                     "smd": smd(b, a), "p_raw": stats.mannwhitneyu(b, a).pvalue})
    for var, label in BINARY:
        a, b = controls[var].dropna(), cases[var].dropna()
        table = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
        rows.append({"characteristic": label,
                     f"controls (n={len(controls)})": f"{int(a.sum())} ({100 * a.mean():.1f}%)",
                     f"cases (n={len(cases)})": f"{int(b.sum())} ({100 * b.mean():.1f}%)",
                     "smd": smd(b, a), "p_raw": stats.chi2_contingency(table)[1]})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = np.minimum(1, out["p_raw"] * len(out))
    return out


def marker_shape(df):
    rows = []
    for col, name in (("nlr", "NLR"), ("crp", "hs-CRP")):
        rows.append({"marker": name, "skewness_raw": stats.skew(df[col]),
                     "skewness_log": stats.skew(np.log(df[col])),
                     "factor_per_sd_of_log": np.exp(np.log(df[col]).std())})
    return pd.DataFrame(rows)


def selection_bias(adults):
    variables = ["age", "male", "income_pir", "education", "nh_white", "nh_black", "current_smoker",
                 "diabetes", "told_htn", "lipid_med", "chd_mi"]
    ref = adults[adults["group"] == "Analysis sample"]
    rows = []
    for var in variables:
        row = {"variable": var}
        for group in ("Excluded: no examination", "Excluded: no blood markers"):
            row[group] = smd(adults.loc[adults["group"] == group, var], ref[var])
        rows.append(row)
    return pd.DataFrame(rows)


def income_missingness_model(df):
    d = df.assign(income_missing=df["income_pir"].isna().astype(int), age10=df["age"] / 10)
    formula = ("income_missing ~ age10 + male + C(race_eth, Treatment('nh_white')) + education"
               " + C(smoking, Treatment('never')) + diabetes + hypertension + chd_mi + log_nlr + log_crp")
    m = smf.logit(formula, data=d).fit(disp=0)
    ci = np.exp(m.conf_int())
    return pd.DataFrame({"OR": np.exp(m.params), "lo": ci[0], "hi": ci[1], "p": m.pvalues}).drop("Intercept")


def events_per_parameter(df):
    """Number of coefficients of M3 after one-hot (drop first) and the income indicator."""
    pre = Pipeline(make_preprocessor(cfg.M3_FEATURES, drop_first=True))
    n_params = pre.fit_transform(df[cfg.M3_FEATURES]).shape[1]
    cases = int(df[cfg.PRIMARY_OUTCOME].sum())
    return pd.DataFrame([
        {"sample": "full analysis sample", "cases": cases, "parameters": n_params, "epv": cases / n_params},
        {"sample": "outer training fold (4/5)", "cases": 0.8 * cases, "parameters": n_params,
         "epv": 0.8 * cases / n_params},
        {"sample": "inner training fold (4/5 of 4/5)", "cases": 0.64 * cases, "parameters": n_params,
         "epv": 0.64 * cases / n_params}])


def main():
    df = add_dummies(pd.read_parquet(cfg.ANALYSIS_SAMPLE_PARQUET))
    adults = add_dummies(pd.read_parquet(cfg.ADULTS_OUTCOME_PARQUET))

    for name, table in (("table1", table1(df)), ("marker_shape", marker_shape(df)),
                        ("selection_bias_smd", selection_bias(adults)),
                        ("income_missingness_model", income_missingness_model(df)),
                        ("sample_size_epv", events_per_parameter(df))):
        table.to_csv(cfg.TABLES_DIR / f"{name}.csv", index=(name == "income_missingness_model"))
        print(f"\n{name}:\n" + table.round(3).to_string())


if __name__ == "__main__":
    main()
