"""
Step 06: main question H1a / H1b with logistic regression (statsmodels).

Model 1 = age + sex, Model 2 = + established risk factors (primary),
Model 3 = + lipid-lowering drugs (sensitivity only, the drug is a consequence of the diagnosis).
OR per 1 SD of the log marker, 95% CI, Wald p; Bonferroni: each marker at alpha = 0.025.
Also: likelihood-ratio test, VIF, squared term, interactions, confounder decomposition and
five sensitivity analyses (plus the same checks for the dAUC of logistic regression).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from evaluation import corrected_ttest, outer_folds
from pipeline_factory import make_preprocessor
from stats_analysis import (MISSING, MODEL1, MODELS, RACES, RISK, fit, impute, odds_ratio, prepare,
                            terms_for, used_rows)

MARKERS = {"nlr_sd": "NLR", "crp_sd": "hs-CRP"}


def main_models(data):
    rows = []
    for marker in ("nlr", "crp", "both"):
        exposures = ["nlr_sd", "crp_sd"] if marker == "both" else [f"{marker}_sd"]
        for model in MODELS:
            for coef in exposures:
                base = {"marker": MARKERS[coef], "adjusted_for_other_marker": marker == "both", "model": model}
                for how, label in MISSING.items():
                    terms = exposures + terms_for(model, how)
                    rows.append({**base, "analysis": label, "n": len(used_rows(data[how], "chd_mi", terms)),
                                 **odds_ratio(data[how], "chd_mi", terms, coef)})
                rows.append({**base, "analysis": "Survey-weighted", "n": len(data["median"]),
                             **odds_ratio(data["median"], "chd_mi", exposures + terms_for(model), coef,
                                          weighted=True)})
    out = pd.DataFrame(rows)
    out["significant_bonferroni"] = out["p"] < cfg.BONFERRONI_ALPHA_PRIMARY
    return out


def full_model(d):
    """All coefficients of Model 2 + both markers; continuous covariates per 1 SD."""
    cont = ["age", "income_pir", "bmi", "total_chol", "hdl_chol", "egfr"]
    dd = d.copy()
    for c in cont:
        dd[c] = (dd[c] - dd[c].mean()) / dd[c].std()
    terms = ["nlr_sd", "crp_sd"] + terms_for("Model 2")
    res = fit(dd, "chd_mi", terms)[3]
    ci = np.exp(res.conf_int())
    return pd.DataFrame({"variable": terms, "or": np.exp(res.params[terms]).to_numpy(),
                         "lo": ci.loc[terms, 0].to_numpy(), "hi": ci.loc[terms, 1].to_numpy(),
                         "p": res.pvalues[terms].to_numpy()})


def lr_tests(d):
    """LR = 2 x (log-likelihood with markers - without), chi-square with df = number of markers."""
    base = terms_for("Model 2")
    r0 = fit(d, "chd_mi", base)[3]
    rows = []
    for label, add in (("NLR", ["nlr_sd"]), ("hs-CRP", ["crp_sd"]), ("NLR + hs-CRP", ["nlr_sd", "crp_sd"])):
        r1 = fit(d, "chd_mi", add + base)[3]
        lr = 2 * (r1.llf - r0.llf)
        rows.append({"added_to_model_2": label, "df": len(add), "lr_statistic": lr,
                     "p": stats.chi2.sf(lr, len(add))})
    return pd.DataFrame(rows)


def nonlinearity(d):
    base = terms_for("Model 2")
    rows = []
    for m, label in (("nlr", "NLR"), ("crp", "hs-CRP")):
        r = odds_ratio(d.assign(sq=d[f"{m}_sd"] ** 2), "chd_mi", [f"{m}_sd", "sq"] + base, "sq")
        rows.append({"marker": label, "or_squared_term": r["or"], "p_squared_term": r["p"]})
    return pd.DataFrame(rows)


def interactions(d):
    base = terms_for("Model 2")
    rows = []
    for m, label in (("nlr", "NLR"), ("crp", "hs-CRP")):
        for mod in ("lipid_med", "male"):
            extra = ["lipid_med"] if mod == "lipid_med" else []
            r = odds_ratio(d.assign(ix=d[f"{m}_sd"] * d[mod]), "chd_mi", [f"{m}_sd", "ix"] + base + extra, "ix")
            rows.append({"marker": label, "modifier": mod, "interaction_or": r["or"], "p": r["p"]})
    return pd.DataFrame(rows)


def vif_table(d):
    X = sm.add_constant(d[["nlr_sd", "crp_sd"] + terms_for("Model 2")].astype(float))
    return pd.DataFrame({"variable": X.columns[1:],
                         "vif": [variance_inflation_factor(X.to_numpy(), i) for i in range(1, X.shape[1])]})


DECOMPOSITION = [   # exploratory: which covariate explains the crude association?
    ("Model 1 (age, sex)", []), ("+ race/ethnicity", [f"race_{r}" for r in RACES]),
    ("+ income", ["income_pir", "income_missing"]), ("+ BMI", ["bmi"]),
    ("+ smoking", ["smk_former", "smk_current"]), ("+ diabetes", ["diabetes"]),
    ("+ hypertension", ["hypertension"]), ("+ total cholesterol", ["total_chol"]),
    ("+ HDL cholesterol", ["hdl_chol"]), ("+ eGFR", ["egfr"]),
    ("+ lipid-lowering drugs", ["lipid_med"]), ("Model 2 (all risk factors)", RISK),
]


def decomposition(d):
    rows = []
    for label, extra in DECOMPOSITION:
        for coef, name in MARKERS.items():
            rows.append({"adjustment": label, "marker": name,
                         **odds_ratio(d, "chd_mi", [coef] + MODEL1 + extra, coef)})
    return pd.DataFrame(rows)


def sensitivity_association(data):
    d = data["median"]
    base = terms_for("Model 2")
    variants = [
        ("Primary (Model 2, median + indicator)", d, "chd_mi", base, False),
        ("Complete cases (listwise deletion)", data["complete_case"], "chd_mi",
         terms_for("Model 2", "complete_case"), False),
        ("Model 3 (+ lipid-lowering drugs)", d, "chd_mi", terms_for("Model 3"), False),
        ("Exclude hs-CRP > 10 mg/L", d[d["crp"] <= 10], "chd_mi", base, False),
        ("Survey-weighted (US population)", d, "chd_mi", base, True),
        ("Outcome incl. angina", d, "chd_mi_angina", base, False),
    ]
    rows = []
    for label, dd, y, terms, weighted in variants:
        dd = used_rows(dd, y, terms)
        row = {"analysis": label, "n": len(dd), "cases": int(dd[y].sum())}
        for coef, name in MARKERS.items():
            r = odds_ratio(dd, y, [coef] + terms, coef, weighted=weighted)
            row.update({f"{name}_or": r["or"], f"{name}_lo": r["lo"], f"{name}_hi": r["hi"], f"{name}_p": r["p"]})
        rows.append(row)
    return pd.DataFrame(rows)


def cv_delta(df, y_col, m2, m3, mask=None):
    """Unpenalized LR in repeated CV (15 folds): mean AUC of M2 and M3 and the corrected dAUC."""
    d = df[df[y_col].notna()] if mask is None else df[mask & df[y_col].notna()]
    d = d.reset_index(drop=True)
    y = d[y_col].astype(int).to_numpy()
    auc2, auc3 = [], []
    for _, _, tr, te in outer_folds(y, cfg.N_OUTER_REPEATS):
        for feats, store in ((m2, auc2), (m3, auc3)):
            pipe = Pipeline(make_preprocessor(feats, drop_first=True)
                            + [("clf", LogisticRegression(C=np.inf, max_iter=10_000))])
            pipe.fit(d.iloc[tr][feats], y[tr])
            store.append(roc_auc_score(y[te], pipe.predict_proba(d.iloc[te][feats])[:, 1]))
    t = corrected_ttest(np.array(auc3) - np.array(auc2))
    return {"n": len(d), "cases": int(y.sum()), "auc_m2": np.mean(auc2), "auc_m3": np.mean(auc3),
            "delta_auc": t["mean"], "lo": t["lo"], "hi": t["hi"], "p": t["p"]}


def sensitivity_prediction(df):
    m2, m3 = cfg.M2_FEATURES, cfg.M3_FEATURES
    variants = [
        ("Primary (CHD or MI)", "chd_mi", None, m2, m3),
        ("Complete cases (listwise deletion)", "chd_mi", df["complete_case"], m2, m3),
        ("+ lipid-lowering drugs as predictor", "chd_mi", None, m2 + ["lipid_med"], m3 + ["lipid_med"]),
        ("Exclude hs-CRP > 10 mg/L", "chd_mi", df["crp"] <= 10, m2, m3),
        ("Outcome incl. angina", "chd_mi_angina", None, m2, m3),
    ]
    return pd.DataFrame([{"analysis": label, **cv_delta(df, y, a, b, mask)}
                         for label, y, mask, a, b in variants])


def main():
    df = pd.read_parquet(cfg.ANALYSIS_SAMPLE_PARQUET)
    d0 = prepare(df)
    data = {how: impute(d0, how) for how in MISSING}

    res = main_models(data)
    res.to_csv(cfg.TABLES_DIR / "assoc_odds_ratios.csv", index=False)
    show = res[(~res["adjusted_for_other_marker"])]
    print(show[["marker", "model", "analysis", "n", "or", "lo", "hi", "p"]].round(3).to_string(index=False))

    d = data["median"]
    for name, table in (("assoc_full_model", full_model(d)), ("assoc_lr_test", lr_tests(d)),
                        ("assoc_nonlinearity", nonlinearity(d)), ("assoc_vif", vif_table(d)),
                        ("assoc_decomposition", decomposition(d)), ("assoc_interactions", interactions(d)),
                        ("sensitivity_association", sensitivity_association(data)),
                        ("sensitivity_prediction", sensitivity_prediction(df))):
        table.to_csv(cfg.TABLES_DIR / f"{name}.csv", index=False)
        print(f"\n{name}:\n" + table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
