"""
Model Evaluation & Statistical Comparison Module.

Implements rigorous evaluation metrics for imbalanced clinical classification:
- Discrimination: ROC-AUC, Precision-Recall AUC (PR-AUC)
- Calibration: Brier Score, calibration slope & intercept
- Decision Metrics: Sensitivity, Specificity, PPV, NPV, F1 at optimal Youden threshold
- Resampled Paired t-Test: Nadeau & Bengio (2003) variance correction for overlapping CV folds
- Multiplicity Adjustment: Family-wise Bonferroni correction
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def compute_discrimination_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Calculate threshold-free discrimination and overall calibration metrics."""
    return {
        "auc": float(roc_auc_score(y_true, y_pred_prob)),
        "pr_auc": float(average_precision_score(y_true, y_pred_prob)),
        "brier": float(brier_score_loss(y_true, y_pred_prob)),
        "log_loss": float(log_loss(y_true, y_pred_prob, eps=1e-15)),
        "mean_predicted_risk": float(np.mean(y_pred_prob)),
        "observed_prevalence": float(np.mean(y_true)),
    }


def find_optimal_youden_threshold(y_train: np.ndarray, prob_train: np.ndarray) -> float:
    """
    Determine classification threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1)
    computed strictly on the TRAINING fold to avoid optimistic leakage.
    """
    fpr, tpr, thresholds = roc_curve(y_train, prob_train)
    youden_idx = np.argmax(tpr - fpr)
    return float(thresholds[youden_idx])


def compute_threshold_metrics(
    y_test: np.ndarray, prob_test: np.ndarray, threshold: float
) -> Dict[str, float]:
    """
    Evaluate confusion matrix metrics at the pre-specified threshold on TEST data.
    """
    pred_binary = (prob_test >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(y_test)

    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def nadeau_bengio_corrected_ttest(
    diffs: np.ndarray | List[float],
    n_train_ratio: float = 4.0,  # n_train / n_test for 5-fold CV is 4.0 (i.e. test/train ratio = 0.25)
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Nadeau & Bengio (2003) corrected paired t-test for repeated cross-validation folds.
    
    Standard paired t-tests assume independent samples; overlapping training sets in
    cross-validation induce positive covariance between fold estimates, causing
    anti-conservative p-values and overly narrow confidence intervals.
    
    Correction factor:
        Var(d_bar) = s^2 * (1 / J + (n_test / n_train))
    where J = total folds (e.g. 15), n_test / n_train = 1 / 4 = 0.25 for 5-fold CV.
    """
    diff_arr = np.asarray(diffs, dtype=float)
    n_folds = len(diff_arr)
    mean_diff = float(np.mean(diff_arr))
    sample_var = float(np.var(diff_arr, ddof=1))

    # Variance correction factor
    correction = (1.0 / n_folds) + (1.0 / n_train_ratio)
    corrected_var = sample_var * correction
    corrected_se = float(np.sqrt(corrected_var))

    df = n_folds - 1
    t_stat = mean_diff / corrected_se if corrected_se > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df)))

    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    ci_lower = mean_diff - t_crit * corrected_se
    ci_upper = mean_diff + t_crit * corrected_se

    return {
        "mean_difference": mean_diff,
        "sample_sd": float(np.sqrt(sample_var)),
        "corrected_se": corrected_se,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def bonferroni_correct(p_values: List[float] | np.ndarray) -> List[float]:
    """Applies Bonferroni adjustment (p * m, capped at 1.0)."""
    m = len(p_values)
    return [float(min(1.0, p * m)) for p in p_values]
