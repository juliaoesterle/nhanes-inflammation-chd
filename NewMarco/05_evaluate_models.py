"""
05 - Statistical evaluation of the nested-CV results (D8, D12).

  * performance per algorithm x feature set: mean over the 15 outer folds with a
    corrected resampled 95% CI (Nadeau & Bengio 2003)
  * marker gain dAUC (M3 - M2, M2+NLR - M2, M2+CRP - M2): paired t-test over the
    15 outer folds (course W3), with the correction for overlapping training sets
    (Nadeau & Bengio 2003), Bonferroni across the six algorithms (course)
  * algorithm comparison on M3: every algorithm vs. logistic regression
    (same paired test, Bonferroni) -> pre-specified rule D12 picks the final model
  * confusion-matrix metrics at the threshold with the highest sensitivity +
    specificity on the training part (course W9), with the "accuracy paradox"
    reference (always predict 'no CHD')
  * figures: discrimination, marker gain, overfitting, calibration, ROC/PR
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

import config as C
from common import (legend_below, AQUA, AXIS, BLUE, DEEMPH, GRID, INK, INK2, MUTED, ORANGE, SURFACE, Logger,
                    bonferroni, corrected_ttest, fmt_ci, fmt_p, save_fig,
                    set_style, write_json)

log = Logger("05_evaluate_models")
set_style()
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ["auc", "pr_auc", "brier", "scaled_brier", "log_loss", "cal_intercept", "cal_slope",
           "auc_train"]
COMPARISONS = [("M3", "M2"), ("M2+NLR", "M2"), ("M2+CRP", "M2")]


def perf_table(fm):
    rows = []
    for (algo, fs), g in fm.groupby(["algo", "fs"]):
        row = {"algo": algo, "fs": fs}
        for m in METRICS:
            ci = corrected_ttest(g[m])
            row[f"{m}"], row[f"{m}_lo"], row[f"{m}_hi"] = ci["mean"], ci["lo"], ci["hi"]
        row["overfit_gap"] = (g.auc_train - g.auc).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired(fm, algo, a, b, metric="auc"):
    ga = fm[(fm.algo == algo) & (fm.fs == a)].set_index(["repeat", "fold"])[metric]
    gb = fm[(fm.algo == algo) & (fm.fs == b)].set_index(["repeat", "fold"])[metric]
    return (ga - gb).dropna()


def marker_gain(fm):
    """dAUC per algorithm with a paired t-test over the 15 folds (corrected for the
    overlap of the training sets) and Bonferroni across the algorithms. dBrier and
    dlog-loss show whether the probabilities themselves get better (lower = better)."""
    rows = []
    for a, b in COMPARISONS:
        for algo in C.ALGORITHMS:
            d = paired(fm, algo, a, b)
            t = corrected_ttest(d.to_numpy())
            rows.append({"comparison": f"{a} - {b}", "algo": algo, "delta_auc": t["mean"],
                         "lo": t["lo"], "hi": t["hi"], "p_corrected_t": t["p"],
                         "delta_brier": paired(fm, algo, a, b, "brier").mean(),
                         "delta_log_loss": paired(fm, algo, a, b, "log_loss").mean(),
                         "folds_with_gain": int((d > 0).sum()), "n_folds": len(d)})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = np.nan
    for comp, g in out.groupby("comparison"):
        out.loc[g.index, "p_bonferroni"] = bonferroni(g.p_corrected_t.to_numpy())
    return out


def algorithm_comparison(fm, fs="M3"):
    rows = []
    for algo in C.ALGORITHMS:
        if algo == "LR":
            continue
        a = fm[(fm.algo == algo) & (fm.fs == fs)].set_index(["repeat", "fold"]).auc
        b = fm[(fm.algo == "LR") & (fm.fs == fs)].set_index(["repeat", "fold"]).auc
        t = corrected_ttest((a - b).dropna().to_numpy())
        rows.append({"algo": algo, "delta_auc_vs_LR": t["mean"], "lo": t["lo"], "hi": t["hi"],
                     "p": t["p"]})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = bonferroni(out.p.to_numpy())
    return out


def threshold_table(fm, prevalence):
    cols = ["thr_threshold", "thr_sensitivity", "thr_specificity", "thr_precision", "thr_npv",
            "thr_f1", "thr_accuracy"]
    t = fm[fm.fs.isin(["M2", "M3"])].groupby(["algo", "fs"])[cols].mean().reset_index()
    t.columns = [c.replace("thr_", "") for c in t.columns]
    ref = pd.DataFrame([
        {"algo": "Always 'no CHD/MI'", "fs": "-", "threshold": np.nan, "sensitivity": 0.0,
         "specificity": 1.0, "precision": np.nan, "npv": 1 - prevalence, "f1": 0.0,
         "accuracy": 1 - prevalence},
        {"algo": "Always 'CHD/MI'", "fs": "-", "threshold": np.nan, "sensitivity": 1.0,
         "specificity": 0.0, "precision": prevalence, "npv": np.nan,
         "f1": 2 * prevalence / (1 + prevalence), "accuracy": prevalence}])
    return pd.concat([t, ref], ignore_index=True)


# =============================================================================
# Figures
# =============================================================================
def fig_discrimination(perf):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [0.95, 1.05]})
    ax = axes[0]
    lr = perf[perf.algo == "LR"].set_index("fs")
    ladder, alone = ["M0", "M1", "M2", "M3"], ["M4", "M4-NLR", "M4-CRP"]   # alone = TA reference models
    names = {"M0": "M0  age, sex", "M1": "M1  + race, income, BMI", "M2": "M2  + clinical risk factors",
             "M3": "M3  + NLR, hs-CRP", "M4": "M4  NLR + hs-CRP alone", "M4-NLR": "NLR alone",
             "M4-CRP": "hs-CRP alone"}
    rows = ladder + alone
    y = np.arange(len(rows))[::-1].astype(float)
    y[len(ladder):] -= 0.8                                   # visual gap between the two groups
    for group, col in ((ladder, BLUE), (alone, AQUA)):
        yy = y[[rows.index(f) for f in group]]
        s = lr.loc[group]
        ax.hlines(yy, s.auc_lo, s.auc_hi, color=col, lw=2)
        ax.scatter(s.auc, yy, s=46, color=col, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        for y_, v, hi in zip(yy, s.auc, s.auc_hi):
            ax.text(hi + 0.006, y_, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0.5, color=AXIS, lw=1, ls=(0, (4, 3)))
    ax.text(0.505, y.min() - 0.55, "chance", fontsize=7.6, color=INK2, va="center")
    ax.set_xlim(0.45, lr.loc[rows].auc_hi.max() + 0.05)
    ax.set_ylim(y.min() - 0.8, y.max() + 0.6)
    ax.set_yticks(y, [names[r] for r in rows])
    ax.set_xlabel("ROC-AUC (15 outer folds, corrected 95% CI)")
    ax.set_title("A  Where does discrimination come from? (LR)")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    algos = C.ALGORITHMS
    y = np.arange(len(algos))[::-1]
    for k, (fs, col, lab) in enumerate([("M2", BLUE, "M2 established risk factors"),
                                        ("M3", ORANGE, "M3 = M2 + NLR + hs-CRP")]):
        s = perf[perf.fs == fs].set_index("algo").loc[algos]
        yy = y + 0.14 - 0.28 * k
        ax.hlines(yy, s.auc_lo, s.auc_hi, color=col, lw=2)
        ax.scatter(s.auc, yy, s=46, color=col, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=lab)
    ax.set_yticks(y, [C.ALGO_LABELS[a] for a in algos])
    ax.set_xlabel("ROC-AUC (15 outer folds, corrected 95% CI)")
    ax.set_title("B  Six algorithms, with and without the markers")
    legend_below(ax, ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, "fig12_discrimination")


def fig_marker_gain(gain):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    algos = C.ALGORITHMS
    y = np.arange(len(algos))[::-1]
    ax.axvline(0, color=AXIS, lw=1)
    ax.axvline(0.01, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.0102, y.max() + 0.45, "0.01 = smallest gain usually\nconsidered clinically relevant",
            fontsize=7.8, color=INK2, va="top")
    for k, (comp, col, lab) in enumerate([("M3 - M2", BLUE, "Both markers (M3 − M2)"),
                                          ("M2+NLR - M2", ORANGE, "NLR only"),
                                          ("M2+CRP - M2", AQUA, "hs-CRP only")]):
        s = gain[gain.comparison == comp].set_index("algo").loc[algos]
        yy = y + 0.2 - 0.2 * k
        ax.hlines(yy, s.lo, s.hi, color=col, lw=2)
        ax.scatter(s.delta_auc, yy, s=40, color=col, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=lab)
    ax.set_yticks(y, [C.ALGO_LABELS[a] for a in algos])
    ax.set_xlabel("Change in ROC-AUC vs. M2 (corrected 95% CI)")
    ax.set_title("Marker gain is tiny for every algorithm")
    legend_below(ax, ncol=3, y=-0.14)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "fig13_marker_gain")


def fig_overfitting(perf):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    algos = C.ALGORITHMS
    s = perf[perf.fs == "M3"].set_index("algo").loc[algos]
    y = np.arange(len(algos))[::-1]
    ax.hlines(y, s.auc, s.auc_train, color=GRID, lw=3)
    ax.scatter(s.auc_train, y, s=46, color=DEEMPH, edgecolors=SURFACE, linewidths=1.5, zorder=3,
               label="Training data (seen)")
    ax.scatter(s.auc, y, s=46, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3,
               label="Test folds (unseen)")
    for yy, a, b, algo in zip(y, s.auc, s.auc_train, algos):
        note = ("\n(distance weights: each training point is its own\nnearest neighbour, "
                "so training AUC = 1 by construction)") if algo == "kNN" and b > 0.999 else ""
        ax.text(max(a, b) + 0.006, yy, f"gap {b - a:+.3f}{note}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [C.ALGO_LABELS[a] for a in algos])
    ax.set_xlabel("ROC-AUC, M3")
    ax.set_xlim(None, 1.04)
    ax.set_title("Overfitting check: seen vs. unseen data (course W4)")
    legend_below(ax, ncol=2, y=-0.16)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "fig14_overfitting")


def fig_calibration(oof, perf):
    algos = C.ALGORITHMS
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0), sharex=True, sharey=True)
    curves = {}
    for algo in algos:
        o = oof[(oof.algo == algo) & (oof.fs == "M3") & (oof.repeat == 0)]
        curves[algo] = calibration_curve(o.y, o.p, n_bins=10, strategy="quantile")
    lim = 1.08 * max(max(f.max(), m.max()) for f, m in curves.values())
    for ax, algo in zip(axes.ravel(), algos):
        frac, mean = curves[algo]
        ax.plot([0, lim], [0, lim], color=AXIS, lw=1)
        ax.plot(mean, frac, "-o", color=BLUE, ms=5, mec=SURFACE, mew=1.2)
        r = perf[(perf.algo == algo) & (perf.fs == "M3")].iloc[0]
        ax.set_title(f"{C.ALGO_LABELS[algo]}", fontsize=9)
        ax.text(0.04 * lim, 0.95 * lim, f"Brier {r.brier:.4f}", fontsize=7.8, color=INK2, va="top")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    for ax in axes[1, :]:
        ax.set_xlabel("Predicted risk")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed share with CHD/MI")
    fig.suptitle("Calibration of M3 (out-of-fold predictions, deciles of predicted risk)",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, "fig15_calibration")


def fig_roc_pr(oof, prevalence, best):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    series = [("LR", "M2", BLUE, "LR, M2"), ("LR", "M3", ORANGE, "LR, M3")]
    if best != "LR":
        series.append((best, "M3", AQUA, f"{best}, M3"))
    axes[0].plot([0, 1], [0, 1], color=AXIS, lw=1)
    axes[1].axhline(prevalence, color=AXIS, lw=1)
    axes[1].text(0.06, prevalence + 0.015, f"no skill = prevalence {prevalence:.3f}", ha="left",
                 fontsize=8, color=INK2)
    for algo, fs, col, lab in series:
        o = oof[(oof.algo == algo) & (oof.fs == fs) & (oof.repeat == 0)]
        fpr, tpr, _ = roc_curve(o.y, o.p)
        pr, rc, _ = precision_recall_curve(o.y, o.p)
        axes[0].plot(fpr, tpr, color=col, lw=2, label=lab)
        axes[1].plot(rc, pr, color=col, lw=2, label=lab)
    axes[0].set(xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)",
                title="ROC curve (pooled out-of-fold, repeat 1)")
    axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)",
                title="Precision-recall curve", ylim=(0, 1))
    fig.tight_layout()
    legend_below(axes[0], ncol=3, fig=fig)
    save_fig(fig, "fig16_roc_pr")



def main():
    fm = pd.read_csv(C.CV_DIR / "fold_metrics.csv")
    oof = pd.read_parquet(C.CV_DIR / "oof_predictions.parquet")
    fm = fm[fm.algo.isin(C.ALGORITHMS)]        # the course algorithms (older runs also had HGB)
    oof = oof[oof.algo.isin(C.ALGORITHMS)]
    prevalence = oof[(oof.algo == "LR") & (oof.fs == "M3") & (oof.repeat == 0)].y.mean()

    perf = perf_table(fm)
    perf.to_csv(C.TABLES / "main_performance.csv", index=False)
    show = perf.copy()
    show["AUC (95% CI)"] = [fmt_ci(r.auc, r.auc_lo, r.auc_hi) for r in perf.itertuples()]
    log("Performance (15 outer folds):\n" + show[["algo", "fs", "AUC (95% CI)", "pr_auc", "brier",
                                                  "scaled_brier", "cal_slope", "overfit_gap"]]
        .round(4).to_string(index=False))

    gain = marker_gain(fm)
    gain.to_csv(C.TABLES / "main_marker_gain.csv", index=False)
    log("Marker gain:\n" + gain.round(4).to_string(index=False))

    comp = algorithm_comparison(fm)
    comp.to_csv(C.TABLES / "main_algorithm_vs_lr.csv", index=False)
    log("Algorithms vs logistic regression (M3):\n" + comp.round(4).to_string(index=False))
    better = comp[(comp.p_bonferroni < C.ALPHA) & (comp.delta_auc_vs_LR > 0)]
    best = better.sort_values("delta_auc_vs_LR").algo.iloc[-1] if len(better) else "LR"
    m3 = perf[perf.fs == "M3"].set_index("algo")
    top_auc = m3.auc.idxmax()
    decision = {"rule": "LR unless another algorithm is significantly better (Bonferroni p < 0.05)",
                "highest_mean_auc": top_auc, "decision": best,
                "significantly_better": better.algo.tolist()}
    write_json(decision, C.DECISIONS / "final_model.json")
    log("Final-model decision:", decision)

    thr = threshold_table(fm, prevalence)
    thr.to_csv(C.TABLES / "main_threshold_metrics.csv", index=False)
    log("Confusion-matrix metrics at the training-OOF Youden threshold:\n" + thr.round(3).to_string(index=False))

    fig_discrimination(perf)
    fig_marker_gain(gain)
    fig_overfitting(perf)
    fig_calibration(oof, perf)
    flex = top_auc if top_auc != "LR" else m3.drop("LR").auc.idxmax()
    fig_roc_pr(oof, prevalence, flex)

    key = {"prevalence": prevalence,
           "lr_auc_m2": perf[(perf.algo == "LR") & (perf.fs == "M2")].auc.iloc[0],
           "lr_auc_m3": m3.loc["LR", "auc"],
           "lr_delta_m3_m2": gain[(gain.algo == "LR") & (gain.comparison == "M3 - M2")].delta_auc.iloc[0],
           "lr_delta_p": gain[(gain.algo == "LR") & (gain.comparison == "M3 - M2")].p_corrected_t.iloc[0],
           "best_algorithm_auc_m3": top_auc, "best_auc_m3": m3.auc.max(), "final_model": best}
    pd.Series(key).to_csv(C.TABLES / "key_numbers_models.csv", header=["value"])
    log("Key numbers:", key)


if __name__ == "__main__":
    main()
