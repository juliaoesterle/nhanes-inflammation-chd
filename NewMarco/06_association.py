"""
06 - Primary question (H1a NLR, H1b hs-CRP): are the markers associated with CHD/MI
independently of established cardiovascular risk factors?

Method (course W7): logistic regression (statsmodels). Result per marker: odds ratio
(OR) per 1 SD of the log marker, 95% CI and Wald p-value. Two co-primary hypotheses
-> Bonferroni (course): each marker is tested at alpha = 0.05 / 2 = 0.025.

  Models (nested)      Model 1 = age + sex
                       Model 2 = + all established risk factors          <- PRIMARY
                       Model 3 = + lipid-lowering drugs (sensitivity only: the drug is
                                 largely a CONSEQUENCE of the diagnosis; adjusting for
                                 a consequence of the outcome can distort the result)
  Missing covariates   the three options of course W2, side by side:
                       median + missing indicator (primary, keeps all 5,538 people),
                       listwise deletion (complete cases), dropping the income column
  Added information    likelihood-ratio test: does Model 2 fit better with the markers?
  Shape                squared term of the log marker + quartiles (dose-response)
  Effect modification  marker x lipid-lowering drugs, marker x sex
  Collinearity         VIF (course W7)
  Decomposition        Model 1 + one covariate group at a time (exploratory, post hoc)
  Sensitivity          outcome definitions, outlier rules (course W2), untreated only,
                       age 40-79, NLR from absolute counts, survey weights (course W3:
                       representativeness)
  Prediction check     logistic-regression dAUC M2 -> M3 for the same variants
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config as cfg
from common import (legend_below, AQUA, AXIS, BLUE, INK, INK2, ORANGE, SURFACE, YELLOW, Logger,
                    corrected_ttest, fmt_p, load_analysis_sample, make_preprocessor, outer_folds,
                    save_fig, set_style, svy_logit)

log = Logger("06_association")
set_style()
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

RACES = ["mexican_american", "other_hispanic", "nh_black", "nh_asian", "other_multi"]  # ref: nh_white
MODEL1 = ["age", "male"]
RISK = ([f"race_{r}" for r in RACES]
        + ["income_pir", "income_missing", "bmi", "smk_former", "smk_current", "diabetes",
           "hypertension", "total_chol", "hdl_chol", "egfr"])
MODEL2 = MODEL1 + RISK
MODEL3 = MODEL2 + ["lipid_med"]
MODELS = {"Model 1": MODEL1, "Model 2": MODEL2, "Model 3": MODEL3}
MISSING = {"median": "Median + indicator (primary)", "complete_case": "Complete cases",
           "drop_income": "Income column dropped"}
CONT = ["age", "income_pir", "bmi", "total_chol", "hdl_chol", "egfr"]
BIN = (["male", "diabetes", "hypertension", "lipid_med", "smk_former", "smk_current"]
       + [f"race_{r}" for r in RACES])


# =============================================================================
# Data: covariates as numbers, three ways to handle the gaps (course W2)
# =============================================================================
def prepare(df):
    """Dummies for race/ethnicity and smoking; markers per 1 SD of their log."""
    d = df.copy().reset_index(drop=True)
    for r in RACES:
        d[f"race_{r}"] = (d.race_eth == r).astype(float)
    for level in ("former", "current"):
        d[f"smk_{level}"] = np.where(d.smoking.isna(), np.nan, (d.smoking == level).astype(float))
    d["income_missing"] = d.income_pir.isna().astype(float)
    for m, col in (("nlr", "log_nlr"), ("crp", "log_crp"), ("nlr_counts", "log_nlr_counts")):
        d[f"{m}_sd"] = (d[col] - d[col].mean()) / d[col].std()
    return d


def impute(d, how):
    """complete_case: drop people with any gap; otherwise fill gaps with the median
    (continuous) or the most frequent value (0/1). 'median' keeps income and its
    missing indicator; 'drop_income' simply does not use the income column."""
    d = d.copy()
    if how == "complete_case":
        return d[d[CONT + BIN].notna().all(axis=1)].copy()
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


# =============================================================================
# Fitting: OR per SD with 95% CI and Wald p (statsmodels summary, course W7)
# =============================================================================
def fit(d, y, terms, weighted=False):
    dd = d[d[y].notna()]
    X = sm.add_constant(dd[terms].astype(float), has_constant="add")
    if weighted:   # NHANES weights + design (strata, PSUs) -> wider, honest CIs
        beta, cov, dfree = svy_logit(dd[y], X, dd.weight_mec, dd.strata, dd.psu)
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
            "hi": np.exp(b[coef] + q * se[coef]), "p": p, "beta": b[coef], "se": se[coef]}


def main_models(data):
    rows = []
    for marker in ("nlr", "crp", "both"):
        exp = ["nlr_sd", "crp_sd"] if marker == "both" else [f"{marker}_sd"]
        for mname in MODELS:
            for coef in exp:
                base = {"marker": {"nlr_sd": "NLR", "crp_sd": "hs-CRP"}[coef],
                        "adjusted_for_other_marker": marker == "both", "model": mname}
                for how, lab in MISSING.items():
                    d = data[how]
                    rows.append({**base, "analysis": lab, "n": int(d.chd_mi.notna().sum()),
                                 **odds_ratio(d, "chd_mi", exp + terms_for(mname, how), coef)})
                rows.append({**base, "analysis": "Survey-weighted", "n": len(data["median"]),
                             **odds_ratio(data["median"], "chd_mi", exp + terms_for(mname), coef,
                                          weighted=True)})
    out = pd.DataFrame(rows)
    out["significant_bonferroni"] = out.p < cfg.ALPHA_PRIMARY
    return out


def full_model(d):
    """Every coefficient of the primary model (Model 2 + both markers), as in the
    statsmodels summary of course W7. Continuous covariates per 1 SD so that the ORs
    are comparable; 0/1 variables yes vs no; race/smoking vs their reference group."""
    cont = ["age", "income_pir", "bmi", "total_chol", "hdl_chol", "egfr"]
    dd = d.copy()
    sds = {c: dd[c].std() for c in cont}
    for c in cont:
        dd[c] = (dd[c] - dd[c].mean()) / sds[c]
    terms = ["nlr_sd", "crp_sd"] + terms_for("Model 2")
    res = fit(dd, "chd_mi", terms)[3]
    ci = np.exp(res.conf_int())
    scale = ["per SD of log" if t in ("nlr_sd", "crp_sd") else f"per SD ({sds[t]:.1f} units)" if t in cont
             else "yes vs no (or vs reference)" for t in terms]
    return pd.DataFrame({"variable": terms, "scale": scale, "or": np.exp(res.params[terms]).to_numpy(),
                         "lo": ci.loc[terms, 0].to_numpy(), "hi": ci.loc[terms, 1].to_numpy(),
                         "p": res.pvalues[terms].to_numpy()})


def lr_tests(d):
    """Likelihood-ratio test (course W7: logistic regression is fitted by maximum
    likelihood). LR = 2 x (log-likelihood with markers - log-likelihood without),
    compared with a chi-square distribution with df = number of added markers."""
    base = terms_for("Model 2")
    r0 = fit(d, "chd_mi", base)[3]
    rows = []
    for lab, add in (("NLR", ["nlr_sd"]), ("hs-CRP", ["crp_sd"]), ("NLR + hs-CRP", ["nlr_sd", "crp_sd"])):
        r1 = fit(d, "chd_mi", add + base)[3]
        lr = 2 * (r1.llf - r0.llf)
        rows.append({"added_to_model_2": lab, "df": len(add), "loglik_model2": r0.llf,
                     "loglik_with_markers": r1.llf, "lr_statistic": lr, "p": stats.chi2.sf(lr, len(add))})
    return pd.DataFrame(rows)


def nonlinearity(d):
    """Is 'log-linear' enough? Add the squared (log) marker; a small p would mean a curve."""
    base = terms_for("Model 2")
    rows = []
    for m, lab in (("nlr", "NLR"), ("crp", "hs-CRP")):
        r = odds_ratio(d.assign(sq=d[f"{m}_sd"] ** 2), "chd_mi", [f"{m}_sd", "sq"] + base, "sq")
        rows.append({"marker": lab, "or_squared_term": r["or"], "p_squared_term": r["p"]})
    return pd.DataFrame(rows)


def quartiles(d):
    """OR of quartiles 2-4 vs quartile 1 (Model 2) and the clinical hs-CRP categories."""
    base = terms_for("Model 2")
    rows = []
    for m, raw, lab in (("nlr", "nlr", "NLR"), ("crp", "crp", "hs-CRP")):
        q = pd.qcut(d[raw], 4, labels=False)
        info = d.groupby(q)[raw].agg(["min", "max", "median"])
        dd = d.assign(**{f"q{k + 1}": (q == k).astype(float) for k in (1, 2, 3)})
        rows.append({"marker": lab, "group": "Q1 (reference)", "median_value": info.loc[0, "median"],
                     "range": f"{info.loc[0, 'min']:.2f}-{info.loc[0, 'max']:.2f}",
                     "or": 1.0, "lo": 1.0, "hi": 1.0, "p": np.nan})
        for k in (1, 2, 3):
            rows.append({"marker": lab, "group": f"Q{k + 1}", "median_value": info.loc[k, "median"],
                         "range": f"{info.loc[k, 'min']:.2f}-{info.loc[k, 'max']:.2f}",
                         **odds_ratio(dd, "chd_mi", ["q2", "q3", "q4"] + base, f"q{k + 1}")})
    dd = d.assign(crp_1_3=((d.crp >= 1) & (d.crp <= 3)).astype(float), crp_gt3=(d.crp > 3).astype(float))
    for c, lab in (("crp_1_3", "1-3 mg/L"), ("crp_gt3", "> 3 mg/L")):
        rows.append({"marker": "hs-CRP clinical (ref < 1 mg/L)", "group": lab,
                     **odds_ratio(dd, "chd_mi", ["crp_1_3", "crp_gt3"] + base, c)})
    return pd.DataFrame(rows)


def interactions(d):
    base = terms_for("Model 2")
    rows = []
    for m, lab in (("nlr", "NLR"), ("crp", "hs-CRP")):
        for mod in ("lipid_med", "male"):
            extra = ["lipid_med"] if mod == "lipid_med" else []
            r = odds_ratio(d.assign(ix=d[f"{m}_sd"] * d[mod]), "chd_mi",
                           [f"{m}_sd", "ix"] + base + extra, "ix")
            rows.append({"marker": lab, "modifier": mod, "interaction_or": r["or"],
                         "lo": r["lo"], "hi": r["hi"], "p": r["p"]})
            for level in (0, 1):
                sub = d[d[mod] == level]
                s = odds_ratio(sub, "chd_mi", [f"{m}_sd"] + [t for t in base if t != mod], f"{m}_sd")
                rows.append({"marker": lab, "modifier": f"{mod} = {level}", "stratum_or": s["or"],
                             "stratum_lo": s["lo"], "stratum_hi": s["hi"],
                             "stratum_cases": int(sub.chd_mi.sum())})
    return pd.DataFrame(rows)


def vif_table(d):
    X = sm.add_constant(d[["nlr_sd", "crp_sd"] + terms_for("Model 2")].astype(float))
    out = pd.DataFrame({"variable": X.columns[1:],
                        "vif": [variance_inflation_factor(X.to_numpy(), i) for i in range(1, X.shape[1])]})
    return out.sort_values("vif", ascending=False)


DECOMPOSITION = [   # exploratory: which covariate explains the crude association?
    ("Model 1 (age, sex)", []),
    ("+ race/ethnicity", [f"race_{r}" for r in RACES]),
    ("+ income", ["income_pir", "income_missing"]),
    ("+ BMI", ["bmi"]),
    ("+ smoking", ["smk_former", "smk_current"]),
    ("+ diabetes", ["diabetes"]),
    ("+ hypertension", ["hypertension"]),
    ("+ total cholesterol", ["total_chol"]),
    ("+ HDL cholesterol", ["hdl_chol"]),
    ("+ eGFR", ["egfr"]),
    ("+ lipid-lowering drugs", ["lipid_med"]),
    ("Model 2 (all risk factors)", RISK),
]


def decomposition(d):
    rows = []
    for lab, extra in DECOMPOSITION:
        for coef, name in (("nlr_sd", "NLR"), ("crp_sd", "hs-CRP")):
            rows.append({"adjustment": lab, "marker": name,
                         **odds_ratio(d, "chd_mi", [coef] + MODEL1 + extra, coef)})
    return pd.DataFrame(rows)


# =============================================================================
# Sensitivity analyses: do the conclusions depend on our choices?
# =============================================================================
def iqr_ok(v):
    """Course W2 outlier rule: inside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]."""
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return (v >= q1 - cfg.IQR_FACTOR * (q3 - q1)) & (v <= q3 + cfg.IQR_FACTOR * (q3 - q1))


def sensitivity_association(data):
    d = data["median"]
    base = terms_for("Model 2")
    variants = [
        ("Primary (Model 2, median + indicator)", d, "chd_mi", base, "nlr_sd", False),
        ("Complete cases (listwise deletion)", data["complete_case"], "chd_mi",
         terms_for("Model 2", "complete_case"), "nlr_sd", False),
        ("Income column dropped", data["drop_income"], "chd_mi",
         terms_for("Model 2", "drop_income"), "nlr_sd", False),
        ("Model 3 (+ lipid-lowering drugs)", d, "chd_mi", terms_for("Model 3"), "nlr_sd", False),
        ("Survey-weighted (US population)", d, "chd_mi", base, "nlr_sd", True),
        ("Outcome incl. angina", d, "chd_mi_angina", base, "nlr_sd", False),
        ("Outcome MI only", d, "mi_strict", base, "nlr_sd", False),
        ("Outcome CHD only", d, "chd_strict", base, "nlr_sd", False),
        ("Exclude hs-CRP > 10 mg/L", d[d.crp <= 10], "chd_mi", base, "nlr_sd", False),
        ("Exclude IQR outliers of log markers", d[iqr_ok(d.log_nlr) & iqr_ok(d.log_crp)],
         "chd_mi", base, "nlr_sd", False),
        ("Untreated only (no lipid-lowering drug)", d[d.lipid_med == 0], "chd_mi", base, "nlr_sd", False),
        ("Age 40-79 (risk-score range)", d[(d.age >= 40) & (d.age <= 79)], "chd_mi", base,
         "nlr_sd", False),
        ("NLR from absolute counts", d, "chd_mi", base, "nlr_counts_sd", False),
    ]
    rows = []
    for lab, dd, y, terms, nlr_coef, weighted in variants:
        dd = dd[dd[y].notna()]
        row = {"analysis": lab, "n": len(dd), "cases": int(dd[y].sum())}
        for coef, name in ((nlr_coef, "NLR"), ("crp_sd", "hs-CRP")):
            r = odds_ratio(dd, y, [coef] + terms, coef, weighted=weighted)
            row.update({f"{name}_or": r["or"], f"{name}_lo": r["lo"], f"{name}_hi": r["hi"],
                        f"{name}_p": r["p"]})
        rows.append(row)
    return pd.DataFrame(rows)


def cv_delta(df, y_col, m2, m3, rows_mask=None):
    """Logistic regression (no tuning) in repeated CV: mean AUC of M2 and M3 and dAUC."""
    d = df[df[y_col].notna()] if rows_mask is None else df[rows_mask & df[y_col].notna()]
    d = d.reset_index(drop=True)
    y = d[y_col].astype(int).to_numpy()
    diffs, auc2, auc3 = [], [], []
    for _, _, tr, te in outer_folds(y, cfg.OUTER_REPEATS):
        aucs = []
        for feats in (m2, m3):
            pipe = Pipeline(make_preprocessor(feats, "median", drop_first=True)
                            + [("clf", LogisticRegression(C=np.inf, max_iter=10_000))])
            pipe.fit(d.iloc[tr][feats], y[tr])
            aucs.append(roc_auc_score(y[te], pipe.predict_proba(d.iloc[te][feats])[:, 1]))
        auc2.append(aucs[0])
        auc3.append(aucs[1])
        diffs.append(aucs[1] - aucs[0])
    t = corrected_ttest(diffs)
    return {"n": len(d), "cases": int(y.sum()), "auc_m2": float(np.mean(auc2)),
            "auc_m3": float(np.mean(auc3)), "delta_auc": t["mean"], "lo": t["lo"], "hi": t["hi"],
            "p": t["p"]}


def sensitivity_prediction(df):
    m2, m3 = cfg.M2, cfg.M3
    lq = (lambda v: iqr_ok(v.to_numpy()))
    no_chol = [f for f in m2 if f not in ("total_chol", "hdl_chol")]
    variants = [
        ("Primary (CHD or MI)", "chd_mi", None, m2, m3),
        ("Outcome incl. angina", "chd_mi_angina", None, m2, m3),
        ("Outcome MI only", "mi_strict", None, m2, m3),
        ("Outcome CHD only", "chd_strict", None, m2, m3),
        ("Exclude hs-CRP > 10 mg/L", "chd_mi", df.crp <= 10, m2, m3),
        ("Exclude IQR outliers of log markers", "chd_mi", lq(df.log_nlr) & lq(df.log_crp), m2, m3),
        ("Untreated only", "chd_mi", df.lipid_med == 0, m2, m3),
        ("Age 40-79", "chd_mi", (df.age >= 40) & (df.age <= 79), m2, m3),
        ("Treatment as predictor (+ lipid & BP drugs)", "chd_mi", None,
         m2 + cfg.TREATMENT_VARS, m3 + cfg.TREATMENT_VARS),
        ("Without cholesterol (treatment-distorted)", "chd_mi", None, no_chol, no_chol + cfg.MARKERS),
    ]
    return pd.DataFrame([{"analysis": lab, **cv_delta(df, y, a, b, mask)}
                         for lab, y, mask, a, b in variants])


# =============================================================================
# Figures
# =============================================================================
def fig_or(res):
    sub = res[~res.adjusted_for_other_marker]
    rows = [(mk, md) for mk in ("NLR", "hs-CRP") for md in MODELS]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    y0 = np.arange(len(rows))[::-1]
    ax.axvline(1, color=AXIS, lw=1)
    analyses = [(MISSING["median"], BLUE), (MISSING["complete_case"], ORANGE),
                (MISSING["drop_income"], AQUA), ("Survey-weighted", YELLOW)]
    for k, (an, col) in enumerate(analyses):
        s = sub[sub.analysis == an].set_index(["marker", "model"]).loc[rows]
        yy = y0 + 0.27 - 0.18 * k
        ax.hlines(yy, s.lo, s.hi, color=col, lw=2)
        ax.scatter(s["or"], yy, s=34, color=col, edgecolors=SURFACE, linewidths=1.3, zorder=3, label=an)
    ax.set_yticks(y0, [f"{mk}, {md}" for mk, md in rows])
    ax.set_xscale("log")
    ax.set_xticks([0.8, 1, 1.25, 1.5], ["0.8", "1", "1.25", "1.5"])
    ax.minorticks_off()
    ax.set_xlabel("Odds ratio per 1 SD of the log marker (95% CI), log scale")
    ax.set_title("Association with CHD/MI: NLR persists, hs-CRP disappears after adjustment")
    legend_below(ax, ncol=2, y=-0.13)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "fig18_odds_ratios")


def fig_quartiles(q, nl):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
    pq = nl.set_index("marker").p_squared_term
    for ax, mk, unit in [(axes[0], "NLR", ""), (axes[1], "hs-CRP", " mg/L")]:
        c = q[q.marker == mk].reset_index(drop=True)
        x = np.arange(len(c))
        ax.axhline(1, color=AXIS, lw=1)
        ax.vlines(x, c.lo, c.hi, color=BLUE, lw=2)
        ax.scatter(x, c["or"], s=46, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_xticks(x, [f"{g.split(' ')[0]}\n(median {v:.2f}{unit})" for g, v in zip(c.group, c.median_value)],
                      fontsize=8)
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1, 1.4, 2], ["0.5", "0.7", "1", "1.4", "2"])
        ax.minorticks_off()
        ax.set_title(f"{mk}: squared-term p = {pq[mk]:.2f}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Odds ratio vs. lowest quartile (Model 2)")
    fig.suptitle("Dose-response by quartile: is the relation a straight line on the log scale?",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, "fig19_dose_response")


def fig_sensitivity(sa, sp):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), gridspec_kw={"width_ratios": [1.15, 0.85]})
    ax = axes[0]
    y = np.arange(len(sa))[::-1]
    ax.axvline(1, color=AXIS, lw=1)
    for k, (mk, col) in enumerate([("NLR", BLUE), ("hs-CRP", ORANGE)]):
        yy = y + 0.12 - 0.24 * k
        ax.hlines(yy, sa[f"{mk}_lo"], sa[f"{mk}_hi"], color=col, lw=2)
        ax.scatter(sa[f"{mk}_or"], yy, s=32, color=col, edgecolors=SURFACE, linewidths=1.2, zorder=3, label=mk)
    ax.set_yticks(y, [f"{a} (cases {c})" for a, c in zip(sa.analysis, sa.cases)], fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1, 1.4], ["0.7", "1", "1.4"])
    ax.minorticks_off()
    ax.set_xlabel("OR per SD (95% CI)")
    ax.set_title("A  Association")
    legend_below(ax, ncol=2, y=-0.1)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    y = np.arange(len(sp))[::-1]
    ax.axvline(0, color=AXIS, lw=1)
    ax.hlines(y, sp.lo, sp.hi, color=BLUE, lw=2)
    ax.scatter(sp.delta_auc, y, s=32, color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.set_yticks(y, sp.analysis, fontsize=8)
    ax.set_xlabel("ΔAUC M3 − M2, logistic regression (corrected 95% CI)")
    ax.set_title("B  Gain in discrimination")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Sensitivity analyses: do the conclusions depend on our choices?",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, "fig20_sensitivity")


def main():
    df = load_analysis_sample()
    log(f"1 SD of log NLR = factor {np.exp(df.log_nlr.std()):.2f}; "
        f"1 SD of log CRP = factor {np.exp(df.log_crp.std()):.2f}")
    d0 = prepare(df)
    data = {how: impute(d0, how) for how in MISSING}
    for how, d in data.items():
        log(f"{MISSING[how]}: n = {len(d):,}, cases = {int(d.chd_mi.sum())}")

    res = main_models(data)
    res.to_csv(cfg.TABLES / "assoc_odds_ratios.csv", index=False)
    show = res.assign(OR=[f"{r['or']:.2f} ({r.lo:.2f}-{r.hi:.2f})" for _, r in res.iterrows()],
                      P=[fmt_p(p) for p in res.p])
    log("Odds ratios per SD:\n" + show[["marker", "adjusted_for_other_marker", "model", "analysis",
                                        "n", "OR", "P", "significant_bonferroni"]].to_string(index=False))

    d = data["median"]
    for name, table in (("assoc_full_model", full_model(d)), ("assoc_lr_test", lr_tests(d)),
                        ("assoc_nonlinearity", nonlinearity(d)),
                        ("assoc_categories", quartiles(d)), ("assoc_vif", vif_table(d)),
                        ("assoc_decomposition", decomposition(d)), ("assoc_interactions", interactions(d))):
        table.to_csv(cfg.TABLES / f"{name}.csv", index=False)
        log(f"{name}:\n" + table.round(4).to_string(index=False))

    sa = sensitivity_association(data)
    sa.to_csv(cfg.TABLES / "sensitivity_association.csv", index=False)
    log("Sensitivity (association):\n" + sa.round(3).to_string(index=False))
    sp = sensitivity_prediction(df)
    sp.to_csv(cfg.TABLES / "sensitivity_prediction.csv", index=False)
    log("Sensitivity (prediction, LR dAUC):\n" + sp.round(4).to_string(index=False))

    fig_or(res)
    fig_quartiles(pd.read_csv(cfg.TABLES / "assoc_categories.csv"),
                  pd.read_csv(cfg.TABLES / "assoc_nonlinearity.csv"))
    fig_sensitivity(sa, sp)


if __name__ == "__main__":
    main()
