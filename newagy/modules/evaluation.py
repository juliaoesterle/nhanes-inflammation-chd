"""
Metrics, folds and tests for the model comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors

import config as cfg


def outer_folds(strat, n_repeats):
    """Repeated stratified 5-fold CV; list of (repeat, fold, train_idx, test_idx)."""
    rskf = RepeatedStratifiedKFold(n_splits=cfg.N_OUTER_SPLITS, n_repeats=n_repeats,
                                   random_state=cfg.RANDOM_SEED)
    out = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(strat)), strat)):
        out.append((k // cfg.N_OUTER_SPLITS, k % cfg.N_OUTER_SPLITS, tr, te))
    return out


def calibration_intercept_slope(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(intercept), float(slope)


def all_metrics(y, p, base_rate):
    """AUC, PR-AUC, Brier (and scaled Brier vs. predicting the training prevalence), calibration."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    brier = brier_score_loss(y, p)
    brier_null = brier_score_loss(y, np.full(len(y), base_rate))
    cal_int, cal_slope = calibration_intercept_slope(y.astype(float), p)
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier,
            "scaled_brier": 1 - brier / brier_null,
            "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
            "cal_intercept": cal_int, "cal_slope": cal_slope,
            "mean_pred": float(p.mean()), "observed": float(y.mean())}


def youden_threshold(y, p):
    """Threshold with the largest sensitivity + specificity - 1 (only on training data!)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_metrics(y, p, threshold):
    y = np.asarray(y, int)
    pred = np.asarray(p, float) >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))

    def ratio(a, b):
        return a / b if b else np.nan
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "precision": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "accuracy": ratio(tp + tn, len(y))}


def corrected_ttest(values, ratio=cfg.TEST_TRAIN_RATIO, alpha=0.05):
    """Nadeau & Bengio (2003): the training sets of the CV folds overlap, so the naive
    variance s^2/J is too small; it is inflated to (1/J + n_test/n_train) * s^2."""
    d = np.asarray(values, float)
    J = d.size
    mean, sd = d.mean(), d.std(ddof=1)
    se = np.sqrt(1.0 / J + ratio) * sd
    t = mean / se if se > 0 else np.inf * np.sign(mean)
    p = 2 * stats.t.sf(abs(t), J - 1)
    q = stats.t.ppf(1 - alpha / 2, J - 1)
    return {"mean": mean, "se": se, "lo": mean - q * se, "hi": mean + q * se, "t": t, "p": p}


def bonferroni(pvals):
    p = np.asarray(pvals, float)
    return np.minimum(1.0, p * p.size)


def resample_training(X, y, how, seed):
    """Random over- or undersampling of the training fold only (course W9)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if how == "oversample":
        idx = np.concatenate([neg, pos, rng.choice(pos, len(neg) - len(pos), replace=True)])
    else:
        idx = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


def smote(X, y, k=5, seed=0):
    """SMOTE (course W9): new case = A + u * (B - A), with A a random case and B one of
    its k nearest case neighbours, until both classes are equally large. Training data only."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    minority = X[y == 1]
    n_new = int((y == 0).sum() - (y == 1).sum())
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority)
    neighbours = nn.kneighbors(minority, return_distance=False)[:, 1:]
    a = rng.integers(0, len(minority), n_new)
    b = neighbours[a, rng.integers(0, k, n_new)]
    u = rng.random((n_new, 1))
    synthetic = minority[a] + u * (minority[b] - minority[a])
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
