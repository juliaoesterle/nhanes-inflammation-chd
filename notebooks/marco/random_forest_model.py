# %% [markdown]
# # Random Forest for CHD/MI Prediction (Marco's Assignment)
#
# Secondary question (H1c): do NLR and hs-CRP add discriminative information
# beyond established cardiovascular risk factors?
#
# * Model: RandomForestClassifier (bagging of 300 de-correlated decision trees)
# * Feature sets M0-M3 (+ M2+NLR, M2+CRP) and cholesterol sensitivity sets
# * No medication variables as predictors (avoids target/treatment leakage)
# * Preprocessing inside Pipeline (OneHotEncoder for categoricals; trees need no scaling)
# * Unweighted probabilities (class_weight=None) across all feature sets so Brier
#   scores stay calibrated and M3 vs M2 compares identical model formulations
#   (balanced_subsample checked separately in class_weight_sensitivity.csv)
# * Tuning: exhaustive GridSearchCV, stratified 5-fold CV on TRAINING set only
# * Operating threshold: Youden's J on out-of-fold training predictions
# * Test set: evaluated once at the end with 1,000 bootstrap resamples
#
# Run from repo root:
#   python notebooks/marco/random_forest_model.py

import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prepare_data import CONFIG, load_data  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = CONFIG["random_state"]
N_ESTIMATORS = 300
N_BOOT = 1000       # bootstrap resamples for test-set 95% CIs
N_PERM = 30         # permutation-importance repeats
N_PDP = 500         # training rows averaged per partial-dependence point

M0 = ["age", "male"]
M1 = M0 + ["bmi", "race_eth", "income_pir"]
M2 = M1 + ["smoking", "diabetes", "hypertension", "total_chol", "hdl_chol", "egfr"]
CHOLESTEROL = ["total_chol", "hdl_chol"]
M2_NO_CHOL = [f for f in M2 if f not in CHOLESTEROL]

FEATURE_SETS = {
    "M0": M0,
    "M1": M1,
    "M2": M2,
    "M2 + NLR": M2 + ["log_nlr"],
    "M2 + CRP": M2 + ["log_crp"],
    "M3": M2 + ["log_nlr", "log_crp"],
    "M2 no chol": M2_NO_CHOL,
    "M3 no chol": M2_NO_CHOL + ["log_nlr", "log_crp"],
}
CATEGORICAL = ["race_eth", "smoking"]
OUTCOME = "chd_mi"

PARAM_GRID = {
    "model__max_depth": [4, 6, 8, 12, None],
    "model__min_samples_leaf": [5, 10, 20, 50],
    "model__max_features": ["sqrt", 0.5],
}


def make_pipeline(features, class_weight=None):
    cats = [c for c in features if c in CATEGORICAL]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=class_weight,
        random_state=SEED,
        n_jobs=1,  # parallelism handled by GridSearchCV
    )
    return Pipeline([("pre", pre), ("model", rf)])


def ci(values):
    return np.percentile(values, [2.5, 97.5])


def threshold_metrics(y, pred):
    tp = np.sum((y == 1) & pred)
    fp = np.sum((y == 0) & pred)
    fn = np.sum((y == 1) & ~pred)
    tn = np.sum((y == 0) & ~pred)
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
    }


def main():
    t_start = time.time()
    train, test = load_data(split=True)
    y_train, y_test = train[OUTCOME].astype(int), test[OUTCOME].astype(int)
    print(f"train n={len(train)} cases={y_train.sum()} ({y_train.mean():.1%}) | "
          f"test n={len(test)} cases={y_test.sum()} ({y_test.mean():.1%})", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fitted, cv_rows, grid_rows = {}, [], []

    print("\n--- Exhaustive 5-Fold Stratified CV Tuning (class_weight=None) ---", flush=True)
    for fs_name, features in FEATURE_SETS.items():
        t0 = time.time()
        search = GridSearchCV(
            make_pipeline(features, class_weight=None),
            PARAM_GRID,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(train[features], y_train)
        fitted[fs_name] = search.best_estimator_
        i = search.best_index_
        best = search.best_params_
        cv_rows.append({
            "features": fs_name,
            "max_depth": str(best["model__max_depth"]),
            "min_samples_leaf": best["model__min_samples_leaf"],
            "max_features": str(best["model__max_features"]),
            "class_weight": "None",
            "cv_auc_mean": search.cv_results_["mean_test_score"][i],
            "cv_auc_sd": search.cv_results_["std_test_score"][i],
        })
        g = pd.DataFrame(search.cv_results_)[[
            "param_model__max_depth", "param_model__min_samples_leaf",
            "param_model__max_features", "mean_test_score", "std_test_score",
        ]].copy()
        g.columns = ["max_depth", "min_samples_leaf", "max_features", "cv_auc_mean", "cv_auc_sd"]
        g.insert(0, "features", fs_name)
        grid_rows.append(g)

        print(f"[{time.time()-t_start:5.1f}s] {fs_name:11s} depth={str(best['model__max_depth']):4s} "
              f"leaf={best['model__min_samples_leaf']:<2d} feat={str(best['model__max_features']):4s} "
              f"CV AUC {cv_rows[-1]['cv_auc_mean']:.3f} ± {cv_rows[-1]['cv_auc_sd']:.3f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(OUT / "cv_results.csv", index=False)
    pd.concat(grid_rows, ignore_index=True).to_csv(OUT / "cv_grid_all.csv", index=False)

    # --- Out-of-fold predictions on training set (for CV delta & Youden threshold)
    print("\n--- Computing Training Out-of-Fold Predictions & Youden Threshold ---", flush=True)
    oof_preds = {}
    for name in ["M2", "M2 + NLR", "M2 + CRP", "M3", "M2 no chol", "M3 no chol"]:
        oof_preds[name] = cross_val_predict(
            clone(fitted[name]),
            train[FEATURE_SETS[name]],
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

    cv_delta_rows = []
    for a, b in [("M2 + NLR", "M2"), ("M2 + CRP", "M2"), ("M3", "M2"), ("M3 no chol", "M2 no chol")]:
        auc_a = roc_auc_score(y_train, oof_preds[a])
        auc_b = roc_auc_score(y_train, oof_preds[b])
        cv_delta_rows.append({
            "comparison": f"{a} vs {b}",
            "auc_a_oof": auc_a,
            "auc_b_oof": auc_b,
            "delta_auc_oof": auc_a - auc_b,
        })
    cv_delta_table = pd.DataFrame(cv_delta_rows)
    cv_delta_table.to_csv(OUT / "cv_delta_auc.csv", index=False)
    print(cv_delta_table.round(4).to_string(index=False), flush=True)

    fpr_tr, tpr_tr, thr_tr = roc_curve(y_train, oof_preds["M3"])
    threshold = float(thr_tr[np.argmax(tpr_tr - fpr_tr)])
    print(f"Youden threshold (training OOF, M3): {threshold:.4f}", flush=True)

    # --- Held-out test set evaluation (used once) -------------------------
    print("\n--- Evaluating on Held-Out Test Set (1,000 Bootstrap Resamples) ---", flush=True)
    yt = y_test.values
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, len(yt), len(yt)) for _ in range(N_BOOT)]
    boot_idx = [b for b in boot_idx if yt[b].min() != yt[b].max()]
    proba = {name: est.predict_proba(test[FEATURE_SETS[name]])[:, 1] for name, est in fitted.items()}

    perf_rows = []
    for name, p in proba.items():
        row = {"features": name}
        row["roc_auc"] = roc_auc_score(yt, p)
        row["pr_auc"] = average_precision_score(yt, p)
        row["brier"] = brier_score_loss(yt, p)

        b_rocs = [roc_auc_score(yt[b], p[b]) for b in boot_idx]
        b_prs = [average_precision_score(yt[b], p[b]) for b in boot_idx]
        b_briers = [np.mean((yt[b] - p[b]) ** 2) for b in boot_idx]

        row["roc_auc_lo"], row["roc_auc_hi"] = ci(b_rocs)
        row["pr_auc_lo"], row["pr_auc_hi"] = ci(b_prs)
        row["brier_lo"], row["brier_hi"] = ci(b_briers)
        perf_rows.append(row)

    perf = pd.DataFrame(perf_rows)
    perf["brier_null"] = brier_score_loss(yt, np.full(len(yt), y_train.mean()))
    perf.to_csv(OUT / "test_performance.csv", index=False)

    delta_rows = []
    for a, b in [("M2 + NLR", "M2"), ("M2 + CRP", "M2"), ("M3", "M2"),
                 ("M3 no chol", "M2 no chol"), ("M2", "M2 no chol")]:
        deltas = [roc_auc_score(yt[i], proba[a][i]) - roc_auc_score(yt[i], proba[b][i]) for i in boot_idx]
        lo, hi = ci(deltas)
        delta_rows.append({
            "comparison": f"{a} vs {b}",
            "delta_auc": roc_auc_score(yt, proba[a]) - roc_auc_score(yt, proba[b]),
            "lo": lo,
            "hi": hi,
            "p_boot_le_0": float(np.mean(np.array(deltas) <= 0)),
        })
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(OUT / "delta_auc.csv", index=False)

    pred_m3 = proba["M3"] >= threshold
    point = threshold_metrics(yt, pred_m3)
    boots = pd.DataFrame([threshold_metrics(yt[b], pred_m3[b]) for b in boot_idx])
    threshold_table = pd.DataFrame([
        {"metric": m, "value": v, "lo": np.nanpercentile(boots[m], 2.5), "hi": np.nanpercentile(boots[m], 97.5)}
        for m, v in point.items()
    ])
    threshold_table.insert(0, "threshold", threshold)
    threshold_table.to_csv(OUT / "threshold_metrics_m3.csv", index=False)

    print(f"\nPrevalence in test set (PR-AUC baseline): {yt.mean():.3f}", flush=True)
    print("\nTest Performance (unweighted Random Forest):", flush=True)
    print(perf[["features", "roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "brier", "brier_null"]]
          .round(4).to_string(index=False), flush=True)
    print("\nDelta AUC (Added Value):", flush=True)
    print(delta.round(4).to_string(index=False), flush=True)
    print("\nThreshold Metrics (M3):", flush=True)
    print(threshold_table.round(3).to_string(index=False), flush=True)

    # --- Sensitivity check: balanced_subsample vs None on M2 & M3 ----------
    print("\n--- Class Weight Sensitivity Check (M2 & M3) ---", flush=True)
    cw_rows = []
    for fs_name in ["M2", "M3"]:
        features = FEATURE_SETS[fs_name]
        s_bal = GridSearchCV(
            make_pipeline(features, class_weight="balanced_subsample"),
            PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
        )
        s_bal.fit(train[features], y_train)
        p_bal = s_bal.best_estimator_.predict_proba(test[features])[:, 1]
        p_unw = proba[fs_name]
        cw_rows.extend([
            {"features": fs_name, "class_weight": "None",
             "cv_auc": cv_table.set_index("features").loc[fs_name, "cv_auc_mean"],
             "test_roc_auc": roc_auc_score(yt, p_unw),
             "test_pr_auc": average_precision_score(yt, p_unw),
             "test_brier": brier_score_loss(yt, p_unw)},
            {"features": fs_name, "class_weight": "balanced_subsample",
             "cv_auc": s_bal.best_score_,
             "test_roc_auc": roc_auc_score(yt, p_bal),
             "test_pr_auc": average_precision_score(yt, p_bal),
             "test_brier": brier_score_loss(yt, p_bal)},
        ])
    cw_df = pd.DataFrame(cw_rows)
    cw_df.to_csv(OUT / "class_weight_sensitivity.csv", index=False)
    print(cw_df.round(4).to_string(index=False), flush=True)

    # --- Permutation importance on test set (M3) --------------------------
    print("\n--- Computing Permutation Feature Importance (M3) ---", flush=True)
    res = permutation_importance(
        fitted["M3"],
        test[FEATURE_SETS["M3"]],
        yt,
        scoring="roc_auc",
        n_repeats=N_PERM,
        random_state=SEED,
        n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": FEATURE_SETS["M3"],
        "auc_drop_mean": res.importances_mean,
        "auc_drop_sd": res.importances_std,
    }).sort_values("auc_drop_mean")
    imp.to_csv(OUT / "permutation_importance_m3.csv", index=False)
    print(imp.round(4).to_string(index=False), flush=True)

    # --- Partial dependence on training sample (M3) -----------------------
    print("\n--- Computing Partial Dependence Curves (M3) ---", flush=True)
    pdp_rows = []
    sample = train.sample(N_PDP, random_state=SEED)[FEATURE_SETS["M3"]]
    for col, raw in [("log_nlr", "nlr"), ("log_crp", "crp")]:
        for q in np.linspace(0.02, 0.98, 25):
            val = train[col].quantile(q)
            X = sample.copy()
            X[col] = val
            pdp_rows.append({
                "marker": raw,
                "value": float(np.exp(val)),
                "mean_predicted_risk": fitted["M3"].predict_proba(X)[:, 1].mean(),
            })
    pdp = pd.DataFrame(pdp_rows)
    pdp.to_csv(OUT / "partial_dependence_m3.csv", index=False)

    # --- Figures (matching Julia's palette & typography) ------------------
    print("\n--- Generating Publication Figures ---", flush=True)
    INK, INK2, MUTED, GRID_COLOR, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "legend.fontsize": 8.5, "lines.linewidth": 2, "font.size": 9.5,
    })

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 1. Test AUC by feature set
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    sub = perf.set_index("features").loc[list(FEATURE_SETS)]
    ypos = np.arange(len(sub))
    ax.errorbar(
        sub.roc_auc, ypos,
        xerr=[sub.roc_auc - sub.roc_auc_lo, sub.roc_auc_hi - sub.roc_auc],
        fmt="o", ms=6, color=BLUE, ecolor=BLUE, elinewidth=1.5,
        markeredgecolor=SURFACE, markeredgewidth=1.5,
    )
    ax.set_yticks(ypos, sub.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Test ROC-AUC (95% bootstrap CI)", title="Random Forest discrimination by feature set")
    save(fig, "auc_by_feature_set")

    # 2. ROC & PR, M2 vs M3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
    ax1.text(0.62, 0.55, "chance", color=MUTED, fontsize=8, rotation=38)
    ax2.axhline(yt.mean(), ls="--", lw=1, color=MUTED)
    ax2.text(0.98, yt.mean() + 0.015, f"prevalence {yt.mean():.3f}", color=MUTED, fontsize=8, ha="right")

    for name, color in [("M2", BLUE), ("M3", ORANGE)]:
        r = perf.set_index("features").loc[name]
        fpr, tpr, _ = roc_curve(yt, proba[name])
        prec, rec, _ = precision_recall_curve(yt, proba[name])
        ax1.plot(fpr, tpr, color=color, label=f"{name}  AUC {r.roc_auc:.3f}")
        ax2.plot(rec, prec, color=color, label=f"{name}  AP {r.pr_auc:.3f}")

    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)",
            xlim=(0, 1), ylim=(0, 1.01))
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve (test set)",
            xlim=(0, 1), ylim=(0, 1.01))
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    save(fig, "roc_pr_m2_m3")

    # 3. Calibration, M3
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    frac, mean_pred = calibration_curve(yt, proba["M3"], n_bins=8, strategy="quantile")
    lim = min(1, max(frac.max(), mean_pred.max()) * 1.15)
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color=MUTED)
    ax.plot(mean_pred, frac, marker="o", ms=6, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted probability", ylabel="Observed fraction with CHD/MI",
           title=f"Calibration (M3, test set), Brier {perf.set_index('features').loc['M3'].brier:.4f}")
    save(fig, "calibration_m3")

    # 4. Permutation importance, M3
    LABELS = {
        "age": "Age", "male": "Male sex", "bmi": "BMI", "race_eth": "Race/ethnicity",
        "income_pir": "Income-to-poverty ratio", "smoking": "Smoking", "diabetes": "Diabetes",
        "hypertension": "Hypertension", "total_chol": "Total cholesterol", "hdl_chol": "HDL cholesterol",
        "egfr": "eGFR", "log_nlr": "log NLR", "log_crp": "log hs-CRP",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.barh(
        range(len(imp)), imp.auc_drop_mean, height=0.6, color=BLUE, xerr=imp.auc_drop_sd,
        error_kw={"ecolor": INK2, "elinewidth": 0.8},
    )
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(len(imp)), [LABELS.get(f, f) for f in imp.feature])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=f"Drop in test ROC-AUC when permuted (mean ± SD, {N_PERM} repeats)",
           title="Permutation importance, Random Forest (M3)")
    save(fig, "permutation_importance_m3")

    # 5. Partial dependence, M3
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, raw, label in [(axes[0], "nlr", "NLR"), (axes[1], "crp", "hs-CRP (mg/L)")]:
        c = pdp[pdp.marker == raw]
        ax.plot(c.value, c.mean_predicted_risk, color=BLUE)
        ax.set_xscale("log")
        ticks = [1, 2, 3, 5] if raw == "nlr" else [0.3, 1, 3, 10]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set(xlabel=f"{label}, log scale", title=f"Partial dependence: {label}")
    axes[0].set_ylabel("Mean predicted risk of CHD/MI")
    save(fig, "partial_dependence_markers_m3")

    versions = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    pd.Series(versions, name="version").to_csv(OUT / "versions.csv", header=True)
    print("\nEnvironment:", versions, flush=True)
    print(f"\nAll Random Forest outputs saved in {time.time()-t_start:.1f}s to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
