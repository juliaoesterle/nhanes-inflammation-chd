"""
03 - Design experiments: test the empirically testable decisions BEFORE the
main analysis, each with a decision rule fixed in config.py beforehand.

  E1  Missing data (D3, course W2): the three W2 options listwise deletion
      (complete cases) vs dropping the income column vs median imputation +
      missing indicator, with logistic regression and a random forest. All options
      are scored on the SAME people (complete cases of the test folds); the two
      options that keep everybody are additionally scored on the people with gaps,
      whom a complete-case model cannot score at all.
  E2  Feature selection (D5, course W4/W6). The reference set is fixed by the
      research question (established risk factors); E2 CHECKS it: literature set
      (S1) vs a naive "let the machine pick from everything" pool (S2, lasso) vs a
      knowledge-screened pool (S3) with the three course families filter
      (univariate F-test), wrapper (RFECV) and embedded (lasso). Reported: AUC,
      number of variables, stability across folds, leakage, selection frequency.
  E3  Class imbalance (D9, course W9): none vs class weights vs random
      oversampling vs random undersampling vs SMOTE.
  E4  Evaluation design (D7, course W4): how much does a single 80/20 split
      vary? 200 random splits of the same data.

Outputs: results/tables/e*_*.csv, results/decisions/*.json, figures fig08-fig11.
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from common import (legend_below, AQUA, AXIS, BLUE, DEEMPH, GRID, INK, INK2, MUTED, ORANGE, SURFACE, Logger,
                    all_metrics, corrected_ttest, fit_tuned, load_analysis_sample,
                    make_preprocessor, outer_folds, read_json, resample_training, save_fig,
                    set_style, smote, threshold_metrics, write_json)

log = Logger("03_design_experiments")
set_style()
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def lr_unpenalised(class_weight=None):
    return LogisticRegression(C=np.inf, max_iter=10_000, class_weight=class_weight)


def lasso(c=0.1):
    return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c, max_iter=5_000,
                              random_state=C.SEED)


def rf_fixed(class_weight=None):
    """Random forest with fixed, sensible settings (no tuning inside the experiments)."""
    return RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=20,
                                  class_weight=class_weight, n_jobs=1, random_state=C.SEED)


def to_original(col: str) -> str:
    if col.startswith("race_eth_"):
        return "race_eth"
    if col.startswith("smoking_"):
        return "smoking"
    if col.endswith("_missing"):
        return col[: -len("_missing")]
    return col


def summarise(res: pd.DataFrame, by, metric):
    rows = []
    for key, g in res.groupby(by):
        vals = g[metric].dropna()
        ci = corrected_ttest(vals) if len(vals) >= 2 else {"mean": np.nan, "lo": np.nan, "hi": np.nan}
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by if isinstance(by, list) else [by], key)),
                     f"{metric}_mean": ci["mean"], f"{metric}_lo": ci["lo"], f"{metric}_hi": ci["hi"]})
    return pd.DataFrame(rows)


# =============================================================================
# E1 Missing data
# =============================================================================
def e1_task(df, rep, fold, tr, te, learner, strategy):
    """One outer fold of E1. The three options of course W2:
    complete_case = train only on people without gaps; drop_income = remove the income
    column, keep everybody; median = keep everybody and everything (median + indicator)."""
    feats = [f for f in C.M3 if not (strategy == "drop_income" and f == "income_pir")]
    y = df[C.OUTCOME].astype(int).to_numpy()
    cc = df["complete_case"].to_numpy()
    tr_rows = tr[cc[tr]] if strategy == "complete_case" else tr
    clf = lr_unpenalised() if learner == "LR" else rf_fixed()
    pre = make_preprocessor(feats, "median", drop_first=(learner == "LR"),
                            indicators=(strategy == "median"))
    pipe = Pipeline(pre + [("clf", clf)]).fit(df.iloc[tr_rows][feats], y[tr_rows])
    p = pipe.predict_proba(df.iloc[te][feats])[:, 1]
    yt, tcc = y[te], cc[te]
    out = {"repeat": rep, "fold": fold, "learner": learner, "strategy": strategy,
           "n_train": len(tr_rows), "cases_train": int(y[tr_rows].sum()),
           "auc_common": roc_auc_score(yt[tcc], p[tcc]),
           "brier_common": brier_score_loss(yt[tcc], p[tcc]),
           "auc_incomplete": np.nan, "auc_all": np.nan}
    if strategy != "complete_case":
        yi = yt[~tcc]
        if 0 < yi.sum() < len(yi):
            out["auc_incomplete"] = roc_auc_score(yi, p[~tcc])
        out["auc_all"] = roc_auc_score(yt, p)
    return out


def run_e1(df):
    log("E1 missing data ...")
    strat = df[C.OUTCOME].astype(int).astype(str) + "_" + df["complete_case"].astype(str)
    folds = outer_folds(strat, C.DESIGN_REPEATS)
    tasks = [(rep, fold, tr, te, lrn, s) for rep, fold, tr, te in folds
             for lrn in ("LR", "RF") for s in C.MISSING_OPTIONS]
    res = pd.DataFrame(Parallel(n_jobs=C.N_JOBS)(delayed(e1_task)(df, *t) for t in tasks))
    res.to_csv(C.TABLES / "e1_missing_data_folds.csv", index=False)

    summ = summarise(res, ["learner", "strategy"], "auc_common")
    for m in ("auc_incomplete", "auc_all", "brier_common"):
        summ = summ.merge(summarise(res, ["learner", "strategy"], m), on=["learner", "strategy"])
    summ["n_train_mean"] = summ.apply(lambda r: res[(res.learner == r.learner) & (res.strategy == r.strategy)].n_train.mean(), axis=1)
    summ.to_csv(C.TABLES / "e1_missing_data_summary.csv", index=False)
    log("E1 summary:\n" + summ.round(4).to_string(index=False))

    # paired differences vs complete case (same folds, same test people)
    diffs = []
    for lrn in ("LR", "RF"):
        base = res[(res.learner == lrn) & (res.strategy == "complete_case")].set_index(["repeat", "fold"]).auc_common
        for s in ("drop_income", "median"):
            sub = res[(res.learner == lrn) & (res.strategy == s)]
            if sub.empty:
                continue
            d = sub.set_index(["repeat", "fold"]).auc_common - base
            t = corrected_ttest(d.to_numpy())
            diffs.append({"learner": lrn, "strategy": s, "delta_auc_vs_cc": t["mean"],
                          "lo": t["lo"], "hi": t["hi"], "p": t["p"]})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(C.TABLES / "e1_missing_data_vs_cc.csv", index=False)
    log("E1 differences vs complete case:\n" + diffs.round(4).to_string(index=False))

    # ---- pre-specified decision rule (config.py) --------------------------
    m = res.groupby("strategy").auc_common.mean()          # both learners, same test people
    others = [s for s in C.MISSING_OPTIONS if s != "median"]
    best_other = max(others, key=lambda s: m[s])
    final = best_other if m[best_other] > m["median"] + C.RULE_IMPUTATION_TOLERANCE else "median"
    decision = {"rule": f"median + indicator (keeps everybody) unless another option is more than "
                        f"{C.RULE_IMPUTATION_TOLERANCE} AUC better on the same test people",
                "mean_auc_common_subset": m.to_dict(), "decision": final}
    write_json(decision, C.DECISIONS / "imputation.json")
    log("E1 decision:", decision)
    return res, summ


# =============================================================================
# E2 Feature selection
# =============================================================================
LEAKAGE = {
    "treatment": ["BPQ101D", "BPQ150", "RXQ510", "RXQ515", "RXQ520", "DIQ050", "DIQ070",
                  "RXQ033", "RXQ050"],
    "other heart diagnosis": ["MCQ160B", "MCQ160D"],
    "health-care use / status": ["HUQ010", "HUQ030", "HUQ042", "HUQ055", "HUQ090", "HIQ011",
                                 "HIQ032A", "HIQ032B", "HIQ032C", "HIQ032D", "HIQ210", "OCQ383",
                                 "OCD150"],
}


def leakage_category(col):
    for cat, cols in LEAKAGE.items():
        if col in cols:
            return cat
    if col.startswith("MCQ") or col.startswith("KIQ") or col.startswith("OSQ"):
        return "other diagnosis / condition"
    return "risk factor / other"


def naive_pool(df):
    raw = pd.read_parquet(C.DATA / "clean_raw_analysis_sample.parquet").set_index("SEQN").loc[df.SEQN]
    drop = {"SEQN", "MCQ160C", "MCQ160E", "WTINT2YR", "WTMEC2YR", "WTPH2YR", "WTSAF2YR",
            "WTSPF2YR", "WTSVOC2Y", "SDMVSTRA", "SDMVPSU", "SDDSRVYR", "RIDSTATR"}
    cols = [c for c in raw.columns if c not in drop]
    raw = raw[cols]
    raw = raw.loc[:, raw.isna().mean() <= 0.40]          # course W2: > 40 % = often too much
    raw = raw.loc[:, raw.nunique() > 1]
    return raw.reset_index(drop=True)


def e2_pipeline(strategy):
    if strategy == "S1 literature (M3)":
        return Pipeline(make_preprocessor(C.M3, "median", drop_first=True) + [("clf", lr_unpenalised())]), {}
    if strategy == "S2 all variables + lasso":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("clf", lasso())])
        pipe.set_output(transform="pandas")
        return pipe, {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3]}
    if strategy == "S3 pool + filter":
        return (Pipeline(make_preprocessor(C.S3_POOL, "median", drop_first=True)
                         + [("select", SelectKBest(f_classif)), ("clf", lr_unpenalised())]),
                {"select__k": [5, 10, 15, 20, 30]})
    if strategy == "S3 pool + wrapper (RFECV)":
        rfe = RFECV(LogisticRegression(C=1.0, max_iter=5_000), step=1, min_features_to_select=3,
                    cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED), scoring="neg_log_loss")
        return Pipeline(make_preprocessor(C.S3_POOL, "median", drop_first=True)
                        + [("select", rfe), ("clf", lr_unpenalised())]), {}
    if strategy == "S3 pool + embedded (lasso)":
        return (Pipeline(make_preprocessor(C.S3_POOL, "median", drop_first=False) + [("clf", lasso())]),
                {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3]})
    raise ValueError(strategy)


E2_STRATEGIES = ["S1 literature (M3)", "S2 all variables + lasso", "S3 pool + filter",
                 "S3 pool + wrapper (RFECV)", "S3 pool + embedded (lasso)"]


def e2_task(X, y, rep, fold, tr, te, strategy):
    pipe, grid = e2_pipeline(strategy)
    est, best = fit_tuned(pipe, grid, X.iloc[tr], y[tr], seed=C.SEED + 10 * rep + fold)
    p = est.predict_proba(X.iloc[te])[:, 1]
    if strategy.startswith("S1"):
        cols = list(C.M3)
        coefs = {}
    elif "lasso" in strategy:
        clf = est.named_steps["clf"]
        mask = clf.coef_.ravel() != 0
        cols = list(np.asarray(clf.feature_names_in_)[mask])
        coefs = dict(zip(clf.feature_names_in_, clf.coef_.ravel()))
    else:
        sel = est.named_steps["select"]
        cols = list(np.asarray(sel.feature_names_in_)[sel.get_support()])
        coefs = {}
    selected = sorted({to_original(c) for c in cols})
    return {"repeat": rep, "fold": fold, "strategy": strategy, "auc": roc_auc_score(y[te], p),
            "brier": brier_score_loss(y[te], p), "n_selected": len(selected),
            "selected": "|".join(selected), "best_params": str(best),
            "coefs": coefs if strategy.startswith("S2") else {}}



def run_e2(df):
    log("E2 feature selection ...")
    y = df[C.OUTCOME].astype(int).to_numpy()
    raw = naive_pool(df)
    log(f"  naive pool S2: {raw.shape[1]} raw columns (<= 40 % missing); "
        f"S3 pool: {len(C.S3_POOL)} variables")
    folds = outer_folds(df[C.OUTCOME].astype(int), C.DESIGN_REPEATS)
    tasks = []
    for rep, fold, tr, te in folds:
        for s in E2_STRATEGIES:
            X = raw if s.startswith("S2") else df
            tasks.append((X, y, rep, fold, tr, te, s))
    res = pd.DataFrame(Parallel(n_jobs=C.N_JOBS)(delayed(e2_task)(*t) for t in tasks))

    # S2: which raw variables does the lasso pick, and are they leakage?
    s2 = res[res.strategy.str.startswith("S2")]
    sel_counts = pd.Series([v for s in s2.selected for v in s.split("|") if v]).value_counts()
    coef = pd.DataFrame(list(s2.coefs)).abs().mean()
    meta = pd.read_csv(C.METADATA).set_index("column_name")
    top = pd.DataFrame({"selection_frequency": sel_counts / len(s2)})
    top["mean_abs_coef"] = coef.reindex(top.index)
    top["label"] = [meta.sas_label.get(v, v) for v in top.index]
    top["category"] = [leakage_category(v) for v in top.index]
    top = top.sort_values(["selection_frequency", "mean_abs_coef"], ascending=False)
    top.to_csv(C.TABLES / "e2_naive_selected_variables.csv")
    res.drop(columns="coefs").to_csv(C.TABLES / "e2_feature_selection_folds.csv", index=False)

    rows = []
    for s, g in res.groupby("strategy", sort=False):
        ci = corrected_ttest(g.auc)
        sets = [set(x.split("|")) for x in g.selected]
        jac = np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])
        freq = pd.Series([v for x in sets for v in x]).value_counts() / len(sets)
        rows.append({"strategy": s, "auc_mean": ci["mean"], "auc_lo": ci["lo"], "auc_hi": ci["hi"],
                     "brier_mean": g.brier.mean(), "n_selected_mean": g.n_selected.mean(),
                     "jaccard_stability": jac,
                     "freq_log_nlr": freq.get("log_nlr", np.nan if s.startswith("S2") else 0.0),
                     "freq_log_crp": freq.get("log_crp", np.nan if s.startswith("S2") else 0.0)})
    summ = pd.DataFrame(rows)
    top_leak = top.head(20)
    summ.loc[summ.strategy.str.startswith("S2"), "share_leakage_in_top20"] = \
        (top_leak.category != "risk factor / other").mean()
    summ.to_csv(C.TABLES / "e2_feature_selection_summary.csv", index=False)
    log("E2 summary:\n" + summ.round(4).to_string(index=False))
    log("E2 naive pool, top 15 selected raw variables:\n"
        + top.head(15)[["selection_frequency", "mean_abs_coef", "category", "label"]].round(3).to_string())

    freq = selection_frequency(res)
    freq.to_csv(C.TABLES / "e2_selection_frequency.csv")
    log("E2 hybrid pool, share of folds in which each variable is selected:\n"
        + freq.round(2).to_string())
    return summ, top, freq


def selection_frequency(res):
    """Share of the 10 outer folds in which each pool variable is selected, per hybrid strategy."""
    hybrid = [s for s in E2_STRATEGIES if s.startswith("S3")]
    out = {}
    for s in hybrid:
        sets = res[res.strategy == s].selected
        counts = pd.Series([v for x in sets for v in x.split("|") if v]).value_counts()
        out[s.replace("S3 pool + ", "")] = counts / len(sets)
    freq = pd.DataFrame(out).reindex(sorted(set(C.S3_POOL))).fillna(0.0)
    return freq.loc[freq.mean(axis=1).sort_values(ascending=False).index]


# =============================================================================
# E3 Class imbalance
# =============================================================================
def e3_task(df, rep, fold, tr, te, learner, option):
    feats = C.M3
    y = df[C.OUTCOME].astype(int).to_numpy()
    Xtr, ytr = df.iloc[tr][feats], y[tr]
    cw = None
    if option == "class_weight":
        cw = "balanced" if learner == "LR" else "balanced_subsample"
    if option in ("oversample", "undersample"):
        Xtr, ytr = resample_training(Xtr, ytr, option, seed=C.SEED + 100 * rep + fold)
    clf = lr_unpenalised(cw) if learner == "LR" else rf_fixed(cw)
    pre = Pipeline(make_preprocessor(feats, "median", drop_first=(learner == "LR")))
    if option == "smote":
        # SMOTE needs numbers: impute + scale + one-hot first (fitted on the training part),
        # then create synthetic cases, then train; the test part is only transformed
        Xt = np.asarray(pre.fit_transform(Xtr, ytr), dtype=float)
        Xs, ys = smote(Xt, ytr, k=5, seed=C.SEED + 100 * rep + fold)
        clf.fit(Xs, ys)
        p = clf.predict_proba(np.asarray(pre.transform(df.iloc[te][feats]), dtype=float))[:, 1]
    else:
        pipe = Pipeline(pre.steps + [("clf", clf)]).fit(Xtr, ytr)
        p = pipe.predict_proba(df.iloc[te][feats])[:, 1]
    prev = y[tr].mean()
    m = all_metrics(y[te], p, base_rate=prev)
    t05 = threshold_metrics(y[te], p, 0.5)
    tpv = threshold_metrics(y[te], p, prev)
    return {"repeat": rep, "fold": fold, "learner": learner, "option": option, **m,
            "sens_at_0.5": t05["sensitivity"], "spec_at_0.5": t05["specificity"],
            "f1_at_0.5": t05["f1"], "accuracy_at_0.5": t05["accuracy"],
            "sens_at_prev": tpv["sensitivity"], "spec_at_prev": tpv["specificity"],
            "f1_at_prev": tpv["f1"]}


def run_e3(df):
    log("E3 class imbalance ...")
    folds = outer_folds(df[C.OUTCOME].astype(int), C.DESIGN_REPEATS)
    tasks = [(rep, fold, tr, te, lrn, o) for rep, fold, tr, te in folds
             for lrn in ("LR", "RF") for o in C.IMBALANCE_OPTIONS]
    res = pd.DataFrame(Parallel(n_jobs=C.N_JOBS)(delayed(e3_task)(df, *t) for t in tasks))
    res.to_csv(C.TABLES / "e3_imbalance_folds.csv", index=False)
    cols = ["auc", "pr_auc", "brier", "scaled_brier", "cal_intercept", "cal_slope", "mean_pred",
            "sens_at_0.5", "spec_at_0.5", "f1_at_0.5", "accuracy_at_0.5", "sens_at_prev",
            "spec_at_prev", "f1_at_prev"]
    summ = res.groupby(["learner", "option"], sort=False)[cols].mean().reset_index()
    summ.to_csv(C.TABLES / "e3_imbalance_summary.csv", index=False)
    log("E3 summary:\n" + summ.round(3).to_string(index=False))

    m = res.groupby("option")[["auc", "brier"]].mean()
    gain = m.auc - m.loc["none", "auc"]
    ok = (gain > C.RULE_WEIGHT_MIN_AUC_GAIN) & (m.brier <= m.loc["none", "brier"])
    chosen = gain[ok].idxmax() if ok.any() else "none"
    decision = {"rule": f"correction only if AUC gain > {C.RULE_WEIGHT_MIN_AUC_GAIN} and Brier not worse",
                "mean_auc": m.auc.to_dict(), "mean_brier": m.brier.to_dict(), "decision": chosen}
    write_json(decision, C.DECISIONS / "imbalance.json")
    log("E3 decision:", decision)
    return summ


# =============================================================================
# E4 Variability of a single 80/20 split
# =============================================================================
def e4_task(df, seed, imputation):
    y = df[C.OUTCOME].astype(int).to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    out = {"seed": seed, "test_cases": int(y[te].sum())}
    for fs in ("M2", "M3"):
        feats = C.FEATURE_SETS[fs]
        pipe = Pipeline(make_preprocessor(feats, imputation, drop_first=True) + [("clf", lr_unpenalised())])
        pipe.fit(df.iloc[tr][feats], y[tr])
        out[f"auc_{fs}"] = roc_auc_score(y[te], pipe.predict_proba(df.iloc[te][feats])[:, 1])
    out["delta_auc"] = out["auc_M3"] - out["auc_M2"]
    return out


def run_e4(df, imputation):
    log("E4 single-split variability ...")
    res = pd.DataFrame(Parallel(n_jobs=C.N_JOBS)(delayed(e4_task)(df, s, imputation) for s in range(200)))
    res.to_csv(C.TABLES / "e4_split_variability.csv", index=False)
    q = res[["auc_M3", "delta_auc"]].quantile([0.025, 0.5, 0.975])
    info = {"auc_M3_range_95": [q.auc_M3[0.025], q.auc_M3[0.975]],
            "auc_M3_min_max": [res.auc_M3.min(), res.auc_M3.max()],
            "delta_auc_range_95": [q.delta_auc[0.025], q.delta_auc[0.975]],
            "share_splits_delta_negative": float((res.delta_auc < 0).mean()),
            "share_splits_delta_above_0.01": float((res.delta_auc > 0.01).mean())}
    write_json(info, C.DECISIONS / "split_variability_info.json")
    log("E4:", info)
    return res


# =============================================================================
# Figures
# =============================================================================
def dot_ci(ax, labels, est, lo, hi, color, offset=0.0, label=None):
    y = np.arange(len(labels))[::-1] + offset
    ax.hlines(y, lo, hi, color=color, lw=2)
    ax.scatter(est, y, s=46, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)
    return y


def fig_e1(summ):
    order = C.MISSING_OPTIONS
    nice = {"complete_case": "Listwise deletion (complete cases)", "drop_income": "Drop the income column",
            "median": "Median + missing indicator"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    for ax, metric, title in [(axes[0], "auc_common", "Same test people (complete cases)"),
                              (axes[1], "auc_incomplete", "People with gaps (deletion cannot score them)")]:
        for k, (lrn, col) in enumerate([("LR", BLUE), ("RF", ORANGE)]):
            s = summ[summ.learner == lrn].set_index("strategy").reindex(order)
            dot_ci(ax, order, s[f"{metric}_mean"], s[f"{metric}_lo"], s[f"{metric}_hi"], col,
                   offset=0.12 - 0.24 * k, label={"LR": "Logistic regression", "RF": "Random forest"}[lrn])
        ax.set_title(title)
        ax.set_xlabel("ROC-AUC (mean of 10 outer folds, corrected 95% CI)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order))[::-1], [nice[o] for o in order])
    fig.suptitle("E1: three ways to handle missing values (course W2)",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    legend_below(axes[0], ncol=2, fig=fig)
    fig.subplots_adjust(bottom=0.24)
    save_fig(fig, "fig08_e1_missing_data")


def fig_e2(summ, top, freq):
    fig = plt.figure(figsize=(11.2, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], width_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    s = summ.set_index("strategy").loc[E2_STRATEGIES]
    yy = dot_ci(ax, E2_STRATEGIES, s.auc_mean, s.auc_lo, s.auc_hi, BLUE)
    for y_, n, j in zip(yy, s.n_selected_mean, s.jaccard_stability):
        ax.text(s.auc_hi.max() + 0.004, y_,
                f"{n:.0f} variables · stability {j:.2f}", va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(yy, E2_STRATEGIES)
    ax.set_xlim(s.auc_lo.min() - 0.01, s.auc_hi.max() + 0.06)
    ax.set_xlabel("ROC-AUC (10 outer folds, corrected 95% CI)")
    ax.set_title("A  Discrimination by selection strategy")
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(gs[0, 1])
    t = top.head(12).iloc[::-1]
    leak = t.category != "risk factor / other"
    ax.barh(range(len(t)), t.selection_frequency, height=0.55, color=np.where(leak, ORANGE, BLUE))
    ax.set_yticks(range(len(t)), [f"{i}: {str(l)[:34]}" for i, l in zip(t.index, t.label)], fontsize=7.4)
    ax.set_xlabel("Selected in share of outer folds")
    ax.set_title("B  Naive pool: what the lasso picks")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="treatment / diagnosis / health-care use"),
                       Patch(color=BLUE, label="risk factor / other")], loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=7.6)

    # C: how often does each pool variable get selected by the three course methods?
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(freq))
    for k, (col, mk, name) in enumerate(zip([BLUE, ORANGE, AQUA], ["o", "s", "D"], freq.columns)):
        ax.scatter(x + (k - 1) * 0.24, freq[name], s=26, color=col, marker=mk, edgecolors=SURFACE,
                   linewidths=1.0, zorder=3, label=name)
    ax.set_xticks(x, [C.LABELS.get(v, v) for v in freq.index], rotation=60, ha="right", fontsize=7.6)
    for lab, v in zip(ax.get_xticklabels(), freq.index):
        if v in C.MARKERS:
            lab.set_fontweight("bold")
    ax.set_ylabel("Selected in share of the 10 folds")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("C  Knowledge-screened pool: how often each variable is selected (markers in bold)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.62), fontsize=8, frameon=False, ncol=3)
    fig.suptitle("E2: feature-selection strategies", x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, "fig09_e2_feature_selection")


def fig_e3(summ):
    order = C.IMBALANCE_OPTIONS
    nice = {"none": "No correction", "class_weight": "Class weights", "oversample": "Random oversampling",
            "undersample": "Random undersampling", "smote": "SMOTE (synthetic cases)"}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), sharey=True)
    for ax, metric, title in [(axes[0], "auc", "ROC-AUC (ranking)"),
                              (axes[1], "brier", "Brier score (lower = better)"),
                              (axes[2], "mean_pred", "Mean predicted risk (true: 7.4%)")]:
        for k, (lrn, col) in enumerate([("LR", BLUE), ("RF", ORANGE)]):
            s = summ[summ.learner == lrn].set_index("option").reindex(order)
            y = np.arange(len(order))[::-1] + 0.12 - 0.24 * k
            ax.scatter(s[metric], y, s=46, color=col, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                       label={"LR": "Logistic regression", "RF": "Random forest"}[lrn])
        if metric == "mean_pred":
            ax.axvline(PREVALENCE, color=INK2, lw=1)
        if metric == "auc":   # honest scale: differences of 0.005 should look small
            mid = summ.auc.mean()
            ax.set_xlim(min(summ.auc.min(), mid - 0.015) - 0.002, max(summ.auc.max(), mid + 0.015) + 0.002)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order))[::-1], [nice[o] for o in order])
    fig.suptitle("E3: resampling and weighting leave the ranking unchanged but destroy the probabilities",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    legend_below(axes[0], ncol=2, fig=fig)
    fig.subplots_adjust(bottom=0.2)
    save_fig(fig, "fig10_e3_imbalance")


def fig_e4(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    ax = axes[0]
    ax.hist(res.auc_M3, bins=24, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("Test ROC-AUC of the same model (M3, logistic regression)")
    ax.set_ylabel("Number of random 80/20 splits")
    ax.set_title(f"A  One model, 200 splits: AUC {res.auc_M3.min():.3f} to {res.auc_M3.max():.3f}")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.hist(res.delta_auc, bins=24, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Test ΔAUC (M3 − M2) on the split")
    ax.set_title(f"B  Marker gain is negative in {100 * (res.delta_auc < 0).mean():.0f}% of splits")
    ax.grid(axis="x", visible=False)
    fig.suptitle("E4: a single train/test split is a lottery ticket",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, "fig11_e4_split_variability")


PREVALENCE = 0.0744


def main():
    global PREVALENCE
    df = load_analysis_sample()
    PREVALENCE = float(df[C.OUTCOME].mean())
    log(f"n = {len(df):,}, cases = {int(df[C.OUTCOME].sum())}")
    _, e1s = run_e1(df)
    fig_e1(e1s)
    imputation = read_json(C.DECISIONS / "imputation.json")["decision"]
    imputation = "median" if imputation == "complete_case" else imputation
    e2s, top, freq = run_e2(df)
    fig_e2(e2s, top, freq)
    e3s = run_e3(df)
    fig_e3(e3s)
    e4 = run_e4(df, "median")
    fig_e4(e4)
    log("done")


if __name__ == "__main__":
    main()
