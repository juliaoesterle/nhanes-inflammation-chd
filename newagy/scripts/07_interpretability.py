"""
Step 07: interpretation.
- permutation importance on the test folds of step 04 (M3), mean and rank per algorithm
- partial dependence: predicted risk across the percentiles of NLR and hs-CRP (LR and RF, all data)
- an illustrative depth-3 decision tree: which variables does it use?
- learning curves: would more data help?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from models import fit_tuned
from pipeline_factory import make_pipeline, make_preprocessor

FEATS = cfg.M3_FEATURES


def importance():
    perm = pd.read_csv(cfg.CV_DIR / "permutation_importance_folds.csv")
    mean = perm.groupby("algo")[FEATS].mean().T[cfg.ALGORITHMS]
    ranks = mean.rank(ascending=False).astype(int)
    return mean, ranks


def partial_dependence(df, y):
    rows = []
    sample = df.sample(1000, random_state=cfg.RANDOM_SEED)[FEATS]
    for algo in ("LR", "RF"):
        pipe, grid = make_pipeline(FEATS, algo)
        est, _ = fit_tuned(pipe, grid, df[FEATS], y, seed=cfg.RANDOM_SEED)
        for col, name in (("log_nlr", "NLR"), ("log_crp", "hs-CRP")):
            for q in np.linspace(0.05, 0.95, 19):
                X = sample.copy()
                X[col] = df[col].quantile(q)
                rows.append({"algo": algo, "marker": name, "percentile": round(100 * q),
                             "value": float(np.exp(df[col].quantile(q))),
                             "mean_risk": float(est.predict_proba(X)[:, 1].mean())})
    return pd.DataFrame(rows)


def small_tree(df, y):
    """Depth-3 tree on imputed, unscaled data, so the split values are readable."""
    X = Pipeline(make_preprocessor(FEATS, drop_first=True)[:1]).fit_transform(df[FEATS], y)
    X = pd.get_dummies(X, columns=cfg.CATEGORICAL_VARS, drop_first=True, dtype=float)
    X["nlr"], X["crp"] = np.exp(X.pop("log_nlr")), np.exp(X.pop("log_crp"))
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=cfg.RANDOM_SEED).fit(X, y)
    used = sorted({X.columns[i] for i in tree.tree_.feature if i >= 0})
    return export_text(tree, feature_names=list(X.columns)), used


def learning_curves(df, y):
    rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_SEED)
    for algo, params in (("LR", {}), ("RF", {"clf__max_features": "sqrt", "clf__min_samples_leaf": 30})):
        pipe, _ = make_pipeline(FEATS, algo)
        pipe.set_params(**params)
        sizes, tr, te = learning_curve(pipe, df[FEATS], y, cv=cv, scoring="roc_auc",
                                       train_sizes=np.linspace(0.1, 1.0, 7), n_jobs=cfg.N_JOBS,
                                       shuffle=True, random_state=cfg.RANDOM_SEED)
        for n, a, b in zip(sizes, tr.mean(axis=1), te.mean(axis=1)):
            rows.append({"algo": algo, "n_train": int(n), "auc_train": a, "auc_cv": b})
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(cfg.ANALYSIS_SAMPLE_PARQUET)
    y = df[cfg.PRIMARY_OUTCOME].astype(int).to_numpy()

    mean, ranks = importance()
    mean.to_csv(cfg.TABLES_DIR / "permutation_importance_mean.csv")
    ranks.to_csv(cfg.TABLES_DIR / "permutation_importance_ranks.csv")
    print("permutation importance (drop in test AUC):\n" + mean.round(4).to_string())
    print("\nrank (1 = most important):\n" + ranks.to_string())

    pdp = partial_dependence(df, y)
    pdp.to_csv(cfg.TABLES_DIR / "partial_dependence.csv", index=False)
    print("\npredicted risk from 5th to 95th percentile (%):\n"
          + pdp.groupby(["algo", "marker"])["mean_risk"].agg(["first", "last", "min", "max"]).mul(100).round(1).to_string())

    text, used = small_tree(df, y)
    (cfg.TABLES_DIR / "decision_tree_depth3.txt").write_text(text)
    print("\ndepth-3 tree uses:", used)

    lc = learning_curves(df, y)
    lc.to_csv(cfg.TABLES_DIR / "learning_curves.csv", index=False)
    print("\nlearning curves:\n" + lc.round(4).to_string(index=False))

    final = json.load(open(cfg.DECISIONS_DIR / "final_model.json"))["decision"]
    print("\nfinal model (step 05):", final)


if __name__ == "__main__":
    main()
