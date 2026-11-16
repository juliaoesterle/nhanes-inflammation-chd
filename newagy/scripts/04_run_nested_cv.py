"""
Step 04: Repeated Nested Cross-Validation (The Main Modeling Pipeline).

Executes a 5-fold Stratified Cross-Validation repeated 3 times (15 outer test folds)
with nested 5-fold inner CV for hyperparameter tuning on log-loss.
Evaluates 6 machine learning algorithms across nested feature subsets (M0, M1, M2, M3, M4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from evaluation import (
    compute_discrimination_metrics,
    compute_threshold_metrics,
    find_optimal_youden_threshold,
)
from models import ALL_ALGORITHMS, get_base_estimator, get_hyperparameter_grid
from pipeline_factory import assemble_full_pipeline


def execute_single_fold_task(df, rep, fold, train_idx, test_idx, algo, fs):
    """Executes inner tuning and outer evaluation for a single fold and model."""
    warnings.filterwarnings("ignore")
    features = cfg.FEATURE_SETS[fs]
    y = df[cfg.PRIMARY_OUTCOME].astype(int).to_numpy()
    X = df[features]
    fold_seed = cfg.RANDOM_SEED + 100 * rep + fold

    t0 = time.time()
    estimator = get_base_estimator(algo, random_state=fold_seed)
    pipe = assemble_full_pipeline(
        estimator=estimator,
        feature_list=features,
        scale_continuous=True,
        imputation_strategy="median",
        include_indicators=True,
    )

    grid = get_hyperparameter_grid(algo)
    if grid:
        inner_cv = StratifiedKFold(n_splits=cfg.N_INNER_SPLITS, shuffle=True, random_state=fold_seed)
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="neg_log_loss",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X.iloc[train_idx], y[train_idx])
        best_model = search.best_estimator_
        best_params = search.best_params_
    else:
        pipe.fit(X.iloc[train_idx], y[train_idx])
        best_model = pipe
        best_params = {}

    p_test = best_model.predict_proba(X.iloc[test_idx])[:, 1]
    p_train = best_model.predict_proba(X.iloc[train_idx])[:, 1]

    metrics = compute_discrimination_metrics(y[test_idx], p_test)
    metrics["auc_train"] = float(roc_auc_score(y[train_idx], p_train))
    metrics["best_params"] = json.dumps(best_params)
    metrics["repeat"] = rep
    metrics["fold"] = fold
    metrics["algo"] = algo
    metrics["fs"] = fs
    metrics["fit_seconds"] = time.time() - t0

    # Decision threshold metrics for M2 and M3
    if fs in ("M2", "M3"):
        inner_cv = StratifiedKFold(n_splits=cfg.N_INNER_SPLITS, shuffle=True, random_state=fold_seed)
        oof_train_probs = cross_val_predict(
            clone(best_model), X.iloc[train_idx], y[train_idx], cv=inner_cv, method="predict_proba"
        )[:, 1]
        thr = find_optimal_youden_threshold(y[train_idx], oof_train_probs)
        thr_res = compute_threshold_metrics(y[test_idx], p_test, thr)
        for k, v in thr_res.items():
            metrics[f"thr_{k}"] = v

    return metrics


def run_nested_cv(recompute=False, target_algo=None):
    start_time = time.time()
    print("=" * 70)
    print("STEP 04: REPEATED NESTED CROSS-VALIDATION")
    print("=" * 70)

    cv_metrics_path = cfg.CV_DIR / "fold_metrics.csv"

    if cv_metrics_path.exists() and not recompute and target_algo is None:
        print(f"Verified pre-computed CV fold metrics at: {cv_metrics_path}")
        df_metrics = pd.read_csv(cv_metrics_path)
        print(f"Total evaluated fold-configurations: {len(df_metrics)}")
        # Print summary table of AUC across algorithms for M2 and M3
        m2_m3 = df_metrics[df_metrics["fs"].isin(["M2", "M3"])]
        summary = m2_m3.groupby(["algo", "fs"])["auc"].agg(["mean", "std"]).reset_index()
        print("\nSummary Cross-Validation Performance (Mean AUC ± SD across 15 folds):")
        print(summary.to_string(index=False))
        print(f"\nStep 04 verified in {time.time() - start_time:.1f} seconds. (Use --recompute to re-run from scratch)\n")
        return

    df = pd.read_parquet(cfg.ANALYSIS_SAMPLE_PARQUET)
    y = df[cfg.PRIMARY_OUTCOME].astype(int)

    rskf = RepeatedStratifiedKFold(
        n_splits=cfg.N_OUTER_SPLITS,
        n_repeats=cfg.N_OUTER_REPEATS,
        random_state=cfg.RANDOM_SEED,
    )
    folds = []
    rep = 0
    fold = 0
    for train_idx, test_idx in rskf.split(df, y):
        folds.append((rep, fold, train_idx, test_idx))
        fold += 1
        if fold == cfg.N_OUTER_SPLITS:
            fold = 0
            rep += 1

    algos = [target_algo] if target_algo else ALL_ALGORITHMS
    feature_subsets = ["M2", "M3", "M2+NLR", "M2+CRP"]
    combos = [(a, fs) for a in algos for fs in feature_subsets]
    if not target_algo or target_algo == "LR":
        combos += [("LR", "M0"), ("LR", "M1"), ("LR", "M4"), ("LR", "M4-NLR"), ("LR", "M4-CRP")]

    tasks = [(rep, fold, tr, te, a, fs) for a, fs in combos for rep, fold, tr, te in folds]
    print(f"Dispatching {len(tasks)} nested cross-validation tasks across {cfg.N_JOBS} worker processes...")

    results = Parallel(n_jobs=cfg.N_JOBS, verbose=5)(
        delayed(execute_single_fold_task)(df, rep, fold, tr, te, a, fs)
        for rep, fold, tr, te, a, fs in tasks
    )

    df_new = pd.DataFrame(results)
    if cv_metrics_path.exists() and target_algo:
        df_old = pd.read_csv(cv_metrics_path)
        df_combined = pd.concat([df_old[df_old["algo"] != target_algo], df_new], ignore_index=True)
        df_combined.to_csv(cv_metrics_path, index=False)
    else:
        df_new.to_csv(cv_metrics_path, index=False)

    print(f"Saved nested CV fold metrics to: {cv_metrics_path}")
    print(f"Step 04 completed in {time.time() - start_time:.1f} seconds.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run repeated nested CV.")
    parser.add_argument("--recompute", action="store_true", help="Recompute all fold evaluations from scratch.")
    parser.add_argument("--algo", type=str, default=None, choices=ALL_ALGORITHMS, help="Recompute specific algorithm.")
    args = parser.parse_args()

    run_nested_cv(recompute=args.recompute, target_algo=args.algo)
