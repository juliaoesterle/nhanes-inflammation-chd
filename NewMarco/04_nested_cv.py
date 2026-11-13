"""
04 - Main analysis: repeated nested cross-validation (D7) of six algorithms
(D11: every classifier of the course, W5-W9)
on the knowledge-driven feature sets (D5).

  outer loop  5-fold stratified CV x 3 repeats = 15 test folds (performance)
  inner loop  5-fold GridSearchCV on the training part only, log-loss (D10)
  preprocessing (imputation, scaling, one-hot) inside the pipeline -> fitted on
  training folds only (no leakage); imputation strategy and class weighting are
  read from the decision files written by 03_design_experiments.py.

Stored per outer fold x algorithm x feature set:
  * test-fold metrics (AUC, PR-AUC, Brier, log-loss, calibration)
  * training ("apparent") AUC -> overfitting check (course W4)
  * best hyperparameters
  * out-of-fold predicted probabilities (for ROC/PR curves and calibration)
  * M2/M3: Youden threshold chosen on out-of-fold predictions INSIDE the training
    part, then confusion-matrix metrics on the test fold (course W9)
  * M3: permutation importance on the test fold
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config as C
from common import (Logger, all_metrics, fit_tuned, load_analysis_sample, make_pipeline,
                    outer_folds, read_json, threshold_metrics, youden_threshold)

log = Logger("04_nested_cv")

# relative cost, used to dispatch the expensive tasks first (better load balance)
COST = {"RF": 6, "SVM": 4, "kNN": 2, "EN": 1, "DT": 1, "LR": 0}


def task(df, rep, fold, tr, te, algo, fs, imputation, class_weight):
    warnings.filterwarnings("ignore")
    feats = C.FEATURE_SETS[fs]
    y = df[C.OUTCOME].astype(int).to_numpy()
    X = df[feats]
    seed = C.SEED + 100 * rep + fold
    t0 = time.time()
    pipe, grid = make_pipeline(feats, algo, imputation=imputation, class_weight=class_weight)
    est, best = fit_tuned(pipe, grid, X.iloc[tr], y[tr], seed=seed)
    p_te = est.predict_proba(X.iloc[te])[:, 1]
    p_tr = est.predict_proba(X.iloc[tr])[:, 1]
    out = {"repeat": rep, "fold": fold, "algo": algo, "fs": fs,
           **all_metrics(y[te], p_te, base_rate=y[tr].mean()),
           "auc_train": roc_auc_score(y[tr], p_tr),
           "best_params": json.dumps({k.replace("clf__", ""): v for k, v in best.items()}, default=str)}
    if fs in ("M2", "M3"):
        inner = StratifiedKFold(C.INNER_SPLITS, shuffle=True, random_state=seed)
        oof = cross_val_predict(clone(est), X.iloc[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
        thr = youden_threshold(y[tr], oof)
        out.update({f"thr_{k}": v for k, v in threshold_metrics(y[te], p_te, thr).items()})
    if fs == "M3":
        pi = permutation_importance(est, X.iloc[te], y[te], scoring="roc_auc", n_repeats=10,
                                    random_state=seed, n_jobs=1)
        out["perm"] = dict(zip(feats, pi.importances_mean))
    out["fit_seconds"] = time.time() - t0
    out["test_idx"] = te
    out["p_test"] = p_te
    return out


def main():
    df = load_analysis_sample()
    imputation = read_json(C.DECISIONS / "imputation.json")["decision"]
    if imputation == "complete_case":
        raise SystemExit("Decision rule chose complete-case analysis: restrict df before running.")
    imb = read_json(C.DECISIONS / "imbalance.json")["decision"]
    if imb in ("oversample", "undersample", "smote"):
        raise SystemExit("Resampling inside the inner loop needs an imbalanced-learn pipeline.")
    class_weight = "balanced" if imb == "class_weight" else None
    log(f"n = {len(df):,}, cases = {int(df[C.OUTCOME].sum())}; imputation = {imputation}; "
        f"class_weight = {class_weight}")

    folds = outer_folds(df[C.OUTCOME].astype(int), C.OUTER_REPEATS)
    combos = [(a, fs) for a in C.ALGORITHMS for fs in ("M2", "M3", "M2+NLR", "M2+CRP")]
    combos += [("LR", "M0"), ("LR", "M1"), ("LR", "M4"), ("LR", "M4-NLR"), ("LR", "M4-CRP")]
    only = [a for a in sys.argv[1:] if a in C.ALGORITHMS]   # e.g. python 04_nested_cv.py SVM
    if only:
        combos = [(a, fs) for a, fs in combos if a in only]
        log(f"re-running only {only}; the other algorithms are kept from the previous run")
    tasks = [(rep, fold, tr, te, a, fs) for a, fs in combos for rep, fold, tr, te in folds]
    tasks.sort(key=lambda t: -COST[t[4]])
    log(f"{len(tasks)} tasks ({len(folds)} outer folds x {len(combos)} algorithm/feature-set combinations)")

    t0 = time.time()
    res = Parallel(n_jobs=C.N_JOBS, verbose=5)(
        delayed(task)(df, *t, imputation, class_weight) for t in tasks)
    log(f"nested CV finished in {(time.time() - t0) / 60:.1f} min")

    folds_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("test_idx", "p_test", "perm")}
                             for r in res])
    oof = []
    for r in res:
        oof.append(pd.DataFrame({"repeat": r["repeat"], "algo": r["algo"], "fs": r["fs"],
                                 "row": r["test_idx"], "p": r["p_test"]}))
    oof = pd.concat(oof, ignore_index=True)
    oof["SEQN"] = df.SEQN.to_numpy()[oof.row]
    oof["y"] = df[C.OUTCOME].astype(int).to_numpy()[oof.row]
    perm = pd.DataFrame([{"repeat": r["repeat"], "fold": r["fold"], "algo": r["algo"], **r["perm"]}
                         for r in res if "perm" in r])

    if only:   # merge: keep the other algorithms, archive the replaced rows
        tag = "_".join(only)
        merged = []
        for new, name, reader in ((folds_df, "fold_metrics.csv", pd.read_csv),
                                  (oof, "oof_predictions.parquet", pd.read_parquet),
                                  (perm, "permutation_importance_folds.csv", pd.read_csv)):
            old = reader(C.CV_DIR / name)
            stem, ext = name.rsplit(".", 1)
            archived = old[old.algo.isin(only)]
            (archived.to_csv if ext == "csv" else archived.to_parquet)(
                C.CV_DIR / f"{stem}_previous_{tag}.{ext}", index=False)
            merged.append(pd.concat([old[~old.algo.isin(only)], new], ignore_index=True))
        folds_df, oof, perm = merged

    folds_df.to_csv(C.CV_DIR / "fold_metrics.csv", index=False)
    oof.to_parquet(C.CV_DIR / "oof_predictions.parquet", index=False)
    perm.to_csv(C.CV_DIR / "permutation_importance_folds.csv", index=False)

    summ = folds_df.groupby(["algo", "fs"])[["auc", "pr_auc", "brier", "auc_train", "fit_seconds"]].mean()
    log("Mean over outer folds:\n" + summ.round(4).to_string())


if __name__ == "__main__":
    main()
