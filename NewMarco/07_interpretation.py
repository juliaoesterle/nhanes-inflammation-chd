"""
07 - Interpretation (D15) and learning curves.

  * permutation importance: computed on the unseen test folds of the nested CV
    (04), averaged over 15 folds, for all six algorithms -> is the ranking of
    predictors consistent across model families?  (Impurity-based importance is
    biased towards continuous variables, Strobl et al. 2007.)
  * partial dependence ("what-if" curves) of the predicted risk on NLR and hs-CRP
    for logistic regression and the random forest, refitted on all data
  * an illustrative depth-3 decision tree (course W7/W8: trees are easy to read)
  * learning curves (course W4: would more training data help?)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

import config as C
from common import (legend_below, AQUA, BLUE, BLUE_RAMP, DEEMPH, INK, INK2, ORANGE, SURFACE, Logger, fit_tuned,
                    load_analysis_sample, make_pipeline, make_preprocessor, read_json, save_fig,
                    set_style)

log = Logger("07_interpretation")
set_style()
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


def importance(perm):
    feats = [c for c in perm.columns if c in C.M3]
    mean = perm.groupby("algo")[feats].mean().T
    sd = perm.groupby("algo")[feats].std().T
    return mean[C.ALGORITHMS], sd[C.ALGORITHMS]


def fig_importance(mean, sd, final):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [0.9, 1.1]})
    ax = axes[0]
    s = mean[final].sort_values()
    e = sd[final].reindex(s.index)
    col = [BLUE if f in C.MARKERS else DEEMPH for f in s.index]
    ax.barh(range(len(s)), s.values, height=0.55, color=col, xerr=e.values,
            error_kw={"ecolor": INK2, "elinewidth": 0.8})
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(s)), [C.LABELS.get(f, f) for f in s.index])
    ax.set_xlabel("Drop in test AUC when the variable is shuffled")
    ax.set_title(f"A  {C.ALGO_LABELS[final]} (blue = markers)")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ranks = mean.rank(ascending=False)
    order = ranks.mean(axis=1).sort_values().index
    from matplotlib.colors import LinearSegmentedColormap    # continuous: ranks 1 and 2 must differ
    cmap = LinearSegmentedColormap.from_list("rank", BLUE_RAMP[::-1])
    im = ax.imshow(ranks.loc[order].to_numpy(), cmap=cmap, aspect="auto", vmin=1, vmax=len(order))
    ax.set_xticks(range(len(C.ALGORITHMS)), C.ALGORITHMS)
    ax.set_yticks(range(len(order)), [C.LABELS.get(f, f) for f in order])
    for i, f in enumerate(order):
        for j, a in enumerate(C.ALGORITHMS):
            if f in C.MARKERS:
                ax.text(j, i, f"{int(ranks.loc[f, a])}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if ranks.loc[f, a] <= 6 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Importance rank (1 = most important)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("B  Rank across all algorithms (markers labelled)")
    fig.tight_layout()
    save_fig(fig, "fig21_importance")


def partial_dependence(df, y):
    feats = C.M3
    out = []
    sample = df.sample(1000, random_state=C.SEED)[feats]
    for algo in ("LR", "RF"):
        pipe, grid = make_pipeline(feats, algo)
        est, best = fit_tuned(pipe, grid, df[feats], y, seed=C.SEED)
        log(f"final {algo} on all data: {best}")
        for m, raw in (("log_nlr", "NLR"), ("log_crp", "hs-CRP")):
            for q in np.linspace(0.05, 0.95, 19):
                v = df[m].quantile(q)
                X = sample.copy()
                X[m] = v
                out.append({"algo": algo, "marker": raw, "value": float(np.exp(v)),
                            "mean_risk": float(est.predict_proba(X)[:, 1].mean())})
    return pd.DataFrame(out)


def fig_pdp(pdp):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
    for ax, mk, ticks in [(axes[0], "NLR", [1, 2, 3, 5]), (axes[1], "hs-CRP", [0.3, 1, 3, 10])]:
        for algo, col in (("LR", BLUE), ("RF", ORANGE)):
            c = pdp[(pdp.algo == algo) & (pdp.marker == mk)]
            ax.plot(c.value, 100 * c.mean_risk, color=col, lw=2, label=C.ALGO_LABELS[algo])
        ax.set_xscale("log")
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(f"{mk} (5th-95th percentile, log scale)")
        ax.set_title(mk)
    axes[0].set_ylabel("Mean predicted risk of CHD/MI, %")
    axes[0].set_ylim(0, None)
    # verified against partial_dependence.csv: NLR +2.9 to +3.4 percentage points (5th -> 95th
    # percentile) in both models; hs-CRP -0.5 to -1.5 points, U-shaped in the forest
    fig.suptitle("Partial dependence: predicted risk rises with NLR in both models; "
                 "flat to slightly falling with hs-CRP",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    legend_below(axes[0], ncol=2, fig=fig)
    save_fig(fig, "fig22_partial_dependence")


def fig_tree(df, y):
    feats = C.M3
    # imputation only, no scaling: trees are scale-invariant, and raw units (years, mg/dL, NLR)
    # make the split thresholds readable
    pre = Pipeline(make_preprocessor(feats, "median", drop_first=True)[:1])
    X = pre.fit_transform(df[feats], y)
    X = pd.get_dummies(X, columns=[c for c in C.CATEGORICAL if c in X.columns], drop_first=True,
                       dtype=float)
    X["log_nlr"], X["log_crp"] = np.exp(X["log_nlr"]), np.exp(X["log_crp"])
    X = X.rename(columns={"log_nlr": "NLR", "log_crp": "hs-CRP (mg/L)"})
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=C.SEED).fit(X, y)
    names = [C.LABELS.get(c, c) for c in X.columns]
    used = sorted({names[i] for i in tree.tree_.feature if i >= 0})
    markers = [m for m in ("NLR", "hs-CRP (mg/L)") if m in used]
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    plot_tree(tree, feature_names=names, class_names=["no CHD/MI", "CHD/MI"], proportion=True,
              filled=False, impurity=False, rounded=True, precision=2, fontsize=8, ax=ax)
    ax.set_title(f"Illustrative decision tree (depth 3, M3, all data): first split on "
                 f"{names[tree.tree_.feature[0]]}; "
                 + (f"markers used: {', '.join(markers)}" if markers else "neither marker is used"))
    save_fig(fig, "fig23_decision_tree")
    return used


def fig_learning(df, y):
    rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    for algo, params in (("LR", {}), ("RF", {"clf__max_features": "sqrt", "clf__min_samples_leaf": 30})):
        pipe, _ = make_pipeline(C.M3, algo)
        pipe.set_params(**params)
        sizes, tr, te = learning_curve(pipe, df[C.M3], y, cv=cv, scoring="roc_auc",
                                       train_sizes=np.linspace(0.1, 1.0, 7), n_jobs=C.N_JOBS,
                                       shuffle=True, random_state=C.SEED)
        for n, a, b in zip(sizes, tr.mean(axis=1), te.mean(axis=1)):
            rows.append({"algo": algo, "n_train": int(n), "auc_train": a, "auc_cv": b})
    lc = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    for ax, algo in zip(axes, ("LR", "RF")):
        c = lc[lc.algo == algo]
        ax.plot(c.n_train, c.auc_train, "-o", color=DEEMPH, ms=5, mec=SURFACE, label="Training data")
        ax.plot(c.n_train, c.auc_cv, "-o", color=BLUE, ms=5, mec=SURFACE, label="Cross-validation")
        ax.set_title(C.ALGO_LABELS[algo])
        ax.set_xlabel("Number of training participants")
    axes[0].set_ylabel("ROC-AUC (M3)")
    legend_below(axes[0], ncol=2, fig=fig)

    def state(algo):      # data-driven verdict: change over the last two training sizes
        c = lc[lc.algo == algo]
        g = c.auc_cv.iloc[-1] - c.auc_cv.iloc[-3]
        verdict = "levelled off" if g < 0.005 else "still rising"
        return f"{algo} {verdict} ({g:+.3f} from n = {c.n_train.iloc[-3]:,} to {c.n_train.iloc[-1]:,})"
    fig.suptitle("Learning curves, validation AUC: " + "; ".join(state(a) for a in ("LR", "RF")),
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    save_fig(fig, "fig24_learning_curves")
    return lc


def main():
    df = load_analysis_sample()
    y = df[C.OUTCOME].astype(int).to_numpy()
    final = read_json(C.DECISIONS / "final_model.json")["decision"]
    perm = pd.read_csv(C.CV_DIR / "permutation_importance_folds.csv")
    mean, sd = importance(perm)
    mean.to_csv(C.TABLES / "permutation_importance_mean.csv")
    log("Permutation importance (mean AUC drop):\n" + mean.round(4).to_string())
    fig_importance(mean, sd, final)

    pdp = partial_dependence(df, y)
    pdp.to_csv(C.TABLES / "partial_dependence.csv", index=False)
    fig_pdp(pdp)
    log("Partial dependence range (risk %):\n"
        + pdp.groupby(["algo", "marker"]).mean_risk.agg(["min", "max"]).mul(100).round(2).to_string())

    used = fig_tree(df, y)
    log("Variables used by the illustrative tree:", used)
    lc = fig_learning(df, y)
    lc.to_csv(C.TABLES / "learning_curves.csv", index=False)
    log("Learning curves:\n" + lc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
