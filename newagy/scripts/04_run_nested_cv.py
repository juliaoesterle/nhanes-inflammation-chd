"""
Step 04: repeated nested cross-validation (5 folds x 3 repeats outside, 5 folds inside).

Six algorithms x feature sets M2, M3, M2+NLR, M2+CRP (+ M0, M1, M4 for LR).
Inner loop: GridSearchCV on log-loss. Outer test fold: AUC, PR-AUC, Brier, calibration,
metrics at the Youden threshold (chosen on the training part), and for M3 the
permutation importance on the test fold. Takes about 30-60 minutes.

    python scripts/04_run_nested_cv.py          # everything
    python scripts/04_run_nested_cv.py SVM      # only one algorithm, keep the others
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from evaluation import all_metrics, outer_folds, threshold_metrics, youden_threshold
from models import fit_tuned
from pipeline_factory import make_pipeline

COST = {"RF": 6, "SVM": 4, "kNN": 2, "EN": 1, "DT": 1, "LR": 0}   # start the slow ones first


def task(df, rep, fold, tr, te, algo, fs, class_weight):
    warnings.filterwarnings("ignore")
    feats = cfg.FEATURE_SETS[fs]
    y = df[cfg.PRIMARY_OUTCOME].astype(int).to_numpy()
    X = df[feats]
    seed = cfg.RANDOM_SEED + 100 * rep + fold
    t0 = time.time()
    pipe, grid = make_pipeline(feats, algo, class_weight=class_weight)
    est, best = fit_tuned(pipe, grid, X.iloc[tr], y[tr], seed=seed)
    p_te = est.predict_proba(X.iloc[te])[:, 1]
    p_tr = est.predict_proba(X.iloc[tr])[:, 1]
    out = {"repeat": rep, "fold": fold, "algo": algo, "fs": fs,
           **all_metrics(y[te], p_te, base_rate=y[tr].mean()),
           "auc_train": roc_auc_score(y[tr], p_tr),
           "best_params": json.dumps({k.replace("clf__", ""): v for k, v in best.items()}, default=str)}
    if fs in ("M2", "M3"):
        inner = StratifiedKFold(cfg.N_INNER_SPLITS, shuffle=True, random_state=seed)
        oof = cross_val_predict(clone(est), X.iloc[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
        thr = youden_threshold(y[tr], oof)
        out.update({f"thr_{k}": v for k, v in threshold_metrics(y[te], p_te, thr).items()})
    if fs == "M3":
        pi = permutation_importance(est, X.iloc[te], y[te], scoring="roc_auc", n_repeats=10,
                                    random_state=seed, n_jobs=1)
        out["perm"] = dict(zip(feats, pi.importances_mean))
    out["fit_seconds"] = time.time() - t0
    return out


def main():
    df = pd.read_parquet(cfg.ANALYSIS_SAMPLE_PARQUET)
    imbalance = json.load(open(cfg.DECISIONS_DIR / "imbalance.json"))["decision"]
    class_weight = "balanced" if imbalance == "class_weight" else None

    folds = outer_folds(df[cfg.PRIMARY_OUTCOME].astype(int), cfg.N_OUTER_REPEATS)
    combos = [(a, fs) for a in cfg.ALGORITHMS for fs in ("M2", "M3", "M2+NLR", "M2+CRP")]
    combos += [("LR", fs) for fs in ("M0", "M1", "M4", "M4-NLR", "M4-CRP")]
    only = [a for a in sys.argv[1:] if a in cfg.ALGORITHMS]
    if only:
        combos = [(a, fs) for a, fs in combos if a in only]
    tasks = [(rep, fold, tr, te, a, fs) for a, fs in combos for rep, fold, tr, te in folds]
    tasks.sort(key=lambda t: -COST[t[4]])
    print(f"{len(tasks)} tasks, class_weight = {class_weight}")

    t0 = time.time()
    res = Parallel(n_jobs=cfg.N_JOBS, verbose=5)(delayed(task)(df, *t, class_weight) for t in tasks)
    print(f"nested CV finished in {(time.time() - t0) / 60:.1f} min")

    fold_metrics = pd.DataFrame([{k: v for k, v in r.items() if k != "perm"} for r in res])
    perm = pd.DataFrame([{"repeat": r["repeat"], "fold": r["fold"], "algo": r["algo"], **r["perm"]}
                         for r in res if "perm" in r])
    if only:   # keep the rows of the other algorithms
        old_fm = pd.read_csv(cfg.CV_DIR / "fold_metrics.csv")
        old_perm = pd.read_csv(cfg.CV_DIR / "permutation_importance_folds.csv")
        fold_metrics = pd.concat([old_fm[~old_fm["algo"].isin(only)], fold_metrics], ignore_index=True)
        perm = pd.concat([old_perm[~old_perm["algo"].isin(only)], perm], ignore_index=True)
    fold_metrics.to_csv(cfg.CV_DIR / "fold_metrics.csv", index=False)
    perm.to_csv(cfg.CV_DIR / "permutation_importance_folds.csv", index=False)

    summary = fold_metrics.groupby(["algo", "fs"])[["auc", "pr_auc", "brier", "auc_train"]].mean()
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
