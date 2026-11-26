"""
Step 03: the four design experiments, each decided with a rule from config.py.

E1 missing data: listwise deletion vs drop the income column vs median + missing indicator,
   with LR and RF, all scored on the same test people (complete cases of the test folds)
E2 feature selection: S1 literature set (M3) vs S2 lasso on all raw columns vs
   S3 lasso on a knowledge-screened pool of 39 variables; S1 is kept unless another
   strategy is leakage-free and clearly better (config.E2_MIN_AUC_GAIN)
E3 class imbalance: none vs class weights vs oversampling vs undersampling vs SMOTE
E4 evaluation design: how much does one 80/20 split vary? (200 random splits)
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from evaluation import (all_metrics, corrected_ttest, outer_folds, resample_training, smote,
                        threshold_metrics)
from models import fit_tuned, make_model
from pipeline_factory import make_preprocessor

warnings.filterwarnings("ignore")
Y = cfg.PRIMARY_OUTCOME


def lr_unpenalized(class_weight=None):
    return LogisticRegression(C=np.inf, max_iter=10_000, class_weight=class_weight)


def rf_fixed(class_weight=None):
    """Random forest with fixed settings (no tuning inside the experiments)."""
    return RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=20,
                                  class_weight=class_weight, n_jobs=1, random_state=cfg.RANDOM_SEED)


def save_json(obj, name):
    with open(cfg.DECISIONS_DIR / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# -----------------------------------------------------------------------------
# E1 missing data
# -----------------------------------------------------------------------------
def e1_task(df, rep, fold, tr, te, learner, strategy):
    feats = [f for f in cfg.M3_FEATURES if not (strategy == "drop_income" and f == "income_pir")]
    y = df[Y].astype(int).to_numpy()
    cc = df["complete_case"].to_numpy()
    tr_rows = tr[cc[tr]] if strategy == "complete_case" else tr
    clf = lr_unpenalized() if learner == "LR" else rf_fixed()
    pre = make_preprocessor(feats, drop_first=(learner == "LR"), indicators=(strategy == "median"))
    pipe = Pipeline(pre + [("clf", clf)]).fit(df.iloc[tr_rows][feats], y[tr_rows])
    p = pipe.predict_proba(df.iloc[te][feats])[:, 1]
    yt, tcc = y[te], cc[te]
    out = {"repeat": rep, "fold": fold, "learner": learner, "strategy": strategy,
           "n_train": len(tr_rows), "auc_common": roc_auc_score(yt[tcc], p[tcc]),
           "brier_common": brier_score_loss(yt[tcc], p[tcc]), "auc_incomplete": np.nan}
    if strategy != "complete_case":
        out["auc_incomplete"] = roc_auc_score(yt[~tcc], p[~tcc])
    return out


def run_e1(df):
    strat = df[Y].astype(int).astype(str) + "_" + df["complete_case"].astype(str)
    folds = outer_folds(strat, cfg.N_DESIGN_REPEATS)
    tasks = [(rep, fold, tr, te, lrn, s) for rep, fold, tr, te in folds
             for lrn in ("LR", "RF") for s in cfg.MISSING_OPTIONS]
    res = pd.DataFrame(Parallel(n_jobs=cfg.N_JOBS)(delayed(e1_task)(df, *t) for t in tasks))
    res.to_csv(cfg.TABLES_DIR / "e1_missing_data_folds.csv", index=False)
    summ = res.groupby(["learner", "strategy"])[["n_train", "auc_common", "auc_incomplete", "brier_common"]].mean()
    summ.to_csv(cfg.TABLES_DIR / "e1_missing_data_summary.csv")
    print("E1:\n" + summ.round(4).to_string())

    mean_auc = res.groupby("strategy")["auc_common"].mean()   # both learners, same test people
    best_other = max(["complete_case", "drop_income"], key=lambda s: mean_auc[s])
    better = mean_auc[best_other] > mean_auc["median"] + cfg.DELTA_AUC_EXPERIMENT_RULE
    decision = best_other if better else "median"
    save_json({"rule": "median + indicator unless another option is more than 0.005 AUC better "
                       "on the same test people", "mean_auc_common": mean_auc.to_dict(),
               "decision": decision}, "imputation.json")
    print("E1 decision:", decision)


# -----------------------------------------------------------------------------
# E2 feature selection
# -----------------------------------------------------------------------------
LEAKAGE = {  # consequences or treatment of a known heart disease
    "RXQ510": "treatment", "RXQ515": "treatment", "RXQ520": "treatment", "RXQ050": "treatment",
    "RXQ033": "treatment", "BPQ101D": "treatment", "BPQ150": "treatment",
    "DIQ050": "treatment", "DIQ070": "treatment",
    "MCQ160B": "other heart diagnosis", "MCQ160D": "other heart diagnosis",
    "OSQ230": "metal in the body (stents)", "HUQ010": "health status", "HUQ030": "health-care use",
    "HUQ042": "health-care use", "HUQ055": "health-care use", "HUQ090": "health-care use",
}


def naive_pool(df):
    raw = pd.read_parquet(cfg.CLEAN_RAW_PARQUET).set_index("SEQN").loc[df["SEQN"]]
    drop = ["MCQ160C", "MCQ160E", "WTINT2YR", "WTMEC2YR", "WTPH2YR", "WTSAF2YR", "WTSPF2YR",
            "WTSVOC2Y", "SDMVSTRA", "SDMVPSU", "SDDSRVYR", "RIDSTATR"]
    raw = raw.drop(columns=[c for c in drop if c in raw.columns])
    raw = raw.loc[:, raw.isna().mean() <= 0.40]      # course W2: > 40 % missing is too much
    raw = raw.loc[:, raw.nunique() > 1]
    return raw.reset_index(drop=True)


def original_name(col):
    for v in cfg.CATEGORICAL_VARS:
        if col.startswith(v + "_"):
            return v
    return col[: -len("_missing")] if col.endswith("_missing") else col


def e2_task(X, y, rep, fold, tr, te, strategy):
    lasso, grid = make_model("L1")
    if strategy == "S1 literature (M3)":
        pipe, grid = Pipeline(make_preprocessor(cfg.M3_FEATURES, drop_first=True)
                              + [("clf", lr_unpenalized())]), {}
    elif strategy == "S2 all raw columns + lasso":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("clf", lasso)])
        pipe.set_output(transform="pandas")
    else:
        pipe = Pipeline(make_preprocessor(cfg.S3_CANDIDATE_POOL) + [("clf", lasso)])
    est, _ = fit_tuned(pipe, grid, X.iloc[tr], y[tr], seed=cfg.RANDOM_SEED + 10 * rep + fold)
    p = est.predict_proba(X.iloc[te])[:, 1]
    coefs = {}
    if strategy.startswith("S1"):
        selected = list(cfg.M3_FEATURES)
    else:
        clf = est.named_steps["clf"]
        coefs = dict(zip(clf.feature_names_in_, clf.coef_.ravel()))
        selected = sorted({original_name(c) for c, b in coefs.items() if b != 0})
    return {"repeat": rep, "fold": fold, "strategy": strategy, "auc": roc_auc_score(y[te], p),
            "brier": brier_score_loss(y[te], p), "n_selected": len(selected),
            "selected": "|".join(selected), "coefs": coefs if strategy.startswith("S2") else {}}


def run_e2(df):
    y = df[Y].astype(int).to_numpy()
    raw = naive_pool(df)
    print(f"E2: S2 pool = {raw.shape[1]} raw columns, S3 pool = {len(cfg.S3_CANDIDATE_POOL)} variables")
    strategies = ["S1 literature (M3)", "S2 all raw columns + lasso", "S3 hybrid pool + lasso"]
    tasks = [(raw if s.startswith("S2") else df, y, rep, fold, tr, te, s)
             for rep, fold, tr, te in outer_folds(df[Y].astype(int), cfg.N_DESIGN_REPEATS)
             for s in strategies]
    res = pd.DataFrame(Parallel(n_jobs=cfg.N_JOBS)(delayed(e2_task)(*t) for t in tasks))
    res.drop(columns="coefs").to_csv(cfg.TABLES_DIR / "e2_feature_selection_folds.csv", index=False)

    # which raw variables does the naive lasso choose?
    s2 = res[res["strategy"].str.startswith("S2")]
    coefs = pd.DataFrame(list(s2["coefs"]))
    meta = pd.read_csv(cfg.METADATA_PATH).set_index("column_name")
    top = pd.DataFrame({"selection_frequency": (coefs != 0).mean(), "mean_abs_coef": coefs.abs().mean()})
    top = top[top["selection_frequency"] > 0].sort_values(["selection_frequency", "mean_abs_coef"],
                                                          ascending=False)
    top["label"] = [meta["sas_label"].get(v, v) for v in top.index]
    top["leakage"] = [LEAKAGE.get(v, "") for v in top.index]
    top.to_csv(cfg.TABLES_DIR / "e2_naive_selected_variables.csv")

    rows = []
    for s, g in res.groupby("strategy", sort=False):
        ci = corrected_ttest(g["auc"])
        sets = [set(x.split("|")) for x in g["selected"]]
        rows.append({"strategy": s, "auc_mean": ci["mean"], "auc_lo": ci["lo"], "auc_hi": ci["hi"],
                     "brier_mean": g["brier"].mean(), "n_selected_mean": g["n_selected"].mean(),
                     "jaccard_stability": np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])})
    summ = pd.DataFrame(rows)
    summ["share_leakage_in_top20"] = np.where(summ["strategy"].str.startswith("S2"),
                                              (top.head(20)["leakage"] != "").mean(), np.nan)
    summ.to_csv(cfg.TABLES_DIR / "e2_feature_selection_summary.csv", index=False)
    print("E2:\n" + summ.round(4).to_string(index=False))
    print("S2 top 10:\n" + top.head(10)[["selection_frequency", "mean_abs_coef", "leakage", "label"]]
          .round(3).to_string())

    # decision rule: compare every strategy with S1 on the same folds
    auc = res.pivot_table(index=["repeat", "fold"], columns="strategy", values="auc")
    s1 = auc["S1 literature (M3)"]
    leakage_top20 = (top.head(20)["leakage"] != "").mean()
    checks = {}
    for s in ["S2 all raw columns + lasso", "S3 hybrid pool + lasso"]:
        t = corrected_ttest(auc[s] - s1)
        leak_free = s.startswith("S3")          # S3 pool screened for leakage, S2 is not
        better = t["mean"] > cfg.E2_MIN_AUC_GAIN and t["lo"] > 0
        checks[s] = {"delta_auc_vs_s1": t["mean"], "lo": t["lo"], "hi": t["hi"], "p": t["p"],
                     "leakage_free": leak_free, "clearly_better": bool(better)}
    winners = [s for s, c in checks.items() if c["leakage_free"] and c["clearly_better"]]
    decision = max(winners, key=lambda s: checks[s]["delta_auc_vs_s1"]) if winners else "S1 literature (M3)"
    save_json({"rule": "keep S1 unless another strategy has no post-diagnosis variables and is more than "
                       "0.005 AUC better than S1 with a corrected 95% CI above 0",
               "share_leakage_in_s2_top20": leakage_top20, "comparisons_vs_s1": checks,
               "decision": decision}, "feature_selection.json")
    print("E2 decision:", decision, {s: round(c["delta_auc_vs_s1"], 4) for s, c in checks.items()})

    s3 = res[res["strategy"].str.startswith("S3")]["selected"]
    freq = pd.Series([v for x in s3 for v in x.split("|")]).value_counts() / len(s3)
    freq = freq.reindex(cfg.S3_CANDIDATE_POOL).fillna(0).sort_values(ascending=False)
    freq.rename("selection_frequency").to_csv(cfg.TABLES_DIR / "e2_selection_frequency.csv")


# -----------------------------------------------------------------------------
# E3 class imbalance
# -----------------------------------------------------------------------------
def e3_task(df, rep, fold, tr, te, learner, option):
    feats = cfg.M3_FEATURES
    y = df[Y].astype(int).to_numpy()
    Xtr, ytr = df.iloc[tr][feats], y[tr]
    cw = None
    if option == "class_weight":
        cw = "balanced" if learner == "LR" else "balanced_subsample"
    if option in ("oversample", "undersample"):
        Xtr, ytr = resample_training(Xtr, ytr, option, seed=cfg.RANDOM_SEED + 100 * rep + fold)
    clf = lr_unpenalized(cw) if learner == "LR" else rf_fixed(cw)
    pre = Pipeline(make_preprocessor(feats, drop_first=(learner == "LR")))
    if option == "smote":   # impute + scale + one-hot on the training part, then synthetic cases
        Xt = np.asarray(pre.fit_transform(Xtr, ytr), dtype=float)
        Xs, ys = smote(Xt, ytr, k=5, seed=cfg.RANDOM_SEED + 100 * rep + fold)
        clf.fit(Xs, ys)
        p = clf.predict_proba(np.asarray(pre.transform(df.iloc[te][feats]), dtype=float))[:, 1]
    else:
        p = Pipeline(pre.steps + [("clf", clf)]).fit(Xtr, ytr).predict_proba(df.iloc[te][feats])[:, 1]
    t05 = threshold_metrics(y[te], p, 0.5)
    return {"repeat": rep, "fold": fold, "learner": learner, "option": option,
            **all_metrics(y[te], p, base_rate=y[tr].mean()),
            "sens_at_0.5": t05["sensitivity"], "spec_at_0.5": t05["specificity"],
            "accuracy_at_0.5": t05["accuracy"]}


def run_e3(df):
    folds = outer_folds(df[Y].astype(int), cfg.N_DESIGN_REPEATS)
    tasks = [(rep, fold, tr, te, lrn, o) for rep, fold, tr, te in folds
             for lrn in ("LR", "RF") for o in cfg.IMBALANCE_OPTIONS]
    res = pd.DataFrame(Parallel(n_jobs=cfg.N_JOBS)(delayed(e3_task)(df, *t) for t in tasks))
    res.to_csv(cfg.TABLES_DIR / "e3_imbalance_folds.csv", index=False)
    cols = ["auc", "pr_auc", "brier", "cal_slope", "mean_pred", "sens_at_0.5", "spec_at_0.5", "accuracy_at_0.5"]
    summ = res.groupby(["learner", "option"], sort=False)[cols].mean().reset_index()
    summ.to_csv(cfg.TABLES_DIR / "e3_imbalance_summary.csv", index=False)
    print("E3:\n" + summ.round(3).to_string(index=False))

    m = res.groupby("option")[["auc", "brier"]].mean()
    gain = m["auc"] - m.loc["none", "auc"]
    ok = (gain > cfg.RULE_WEIGHT_MIN_AUC_GAIN) & (m["brier"] <= m.loc["none", "brier"])
    decision = gain[ok].idxmax() if ok.any() else "none"
    save_json({"rule": "correction only if AUC gain > 0.005 and Brier not worse",
               "mean_auc": m["auc"].to_dict(), "mean_brier": m["brier"].to_dict(), "decision": decision},
              "imbalance.json")
    print("E3 decision:", decision)


# -----------------------------------------------------------------------------
# E4 variability of a single 80/20 split
# -----------------------------------------------------------------------------
def e4_task(df, seed):
    y = df[Y].astype(int).to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    out = {"seed": seed}
    for fs in ("M2", "M3"):
        feats = cfg.FEATURE_SETS[fs]
        pipe = Pipeline(make_preprocessor(feats, drop_first=True) + [("clf", lr_unpenalized())])
        pipe.fit(df.iloc[tr][feats], y[tr])
        out[f"auc_{fs}"] = roc_auc_score(y[te], pipe.predict_proba(df.iloc[te][feats])[:, 1])
    out["delta_auc"] = out["auc_M3"] - out["auc_M2"]
    return out


def run_e4(df):
    res = pd.DataFrame(Parallel(n_jobs=cfg.N_JOBS)(delayed(e4_task)(df, s) for s in range(200)))
    res.to_csv(cfg.TABLES_DIR / "e4_split_variability.csv", index=False)
    info = {"n_splits": 200, "auc_M3_min": res["auc_M3"].min(), "auc_M3_max": res["auc_M3"].max(),
            "auc_M3_mean": res["auc_M3"].mean(),
            "auc_M3_middle_95_percent": list(res["auc_M3"].quantile([0.025, 0.975])),
            "share_splits_delta_negative": float((res["delta_auc"] < 0).mean())}
    save_json(info, "split_variability_info.json")
    print("E4:", info)


def main():
    df = pd.read_parquet(cfg.ANALYSIS_SAMPLE_PARQUET)
    run_e1(df)
    run_e2(df)
    run_e3(df)
    run_e4(df)


if __name__ == "__main__":
    main()
