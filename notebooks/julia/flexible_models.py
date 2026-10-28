# %% [markdown]
# # Package D: flexible prediction models (random forest, gradient boosting)
#
# Secondary question (H1c): do NLR and hs-CRP add discriminative information
# beyond established risk factors?
#
# * Models: RandomForestClassifier, HistGradientBoostingClassifier
# * Feature sets M0-M3 (+ M2+NLR, M2+CRP); no medication variables (leakage)
# * Tuning: randomized search, stratified 5-fold CV on the TRAINING set only
# * Test set: used once, in the final evaluation cell
# * Interpretation: permutation importance (test set), SHAP for boosting
# * Extra: boosting on the eligible sample with native NaN handling
#
# Run from the repo root:  python notebooks/julia/flexible_models.py
# (cells also run interactively in VS Code / Jupyter via "# %%").

# %%
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import loguniform
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prepare_data import CONFIG, load_data  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = CONFIG["random_state"]
N_ITER = 25          # random-search candidates per model x feature set
N_BOOT = 1000        # bootstrap resamples for test-set CIs
N_PERM = 30          # permutation-importance repeats

# %% [markdown]
# ## Feature sets (shared with package C)

# %%
M0 = ["age", "male"]
M1 = M0 + ["bmi", "race_eth", "income_pir"]
M2 = M1 + ["smoking", "diabetes", "hypertension", "total_chol", "hdl_chol", "egfr"]
FEATURE_SETS = {
    "M0": M0,
    "M1": M1,
    "M2": M2,
    "M2 + NLR": M2 + ["log_nlr"],
    "M2 + CRP": M2 + ["log_crp"],
    "M3": M2 + ["log_nlr", "log_crp"],
}
CATEGORICAL = ["race_eth", "smoking"]
OUTCOME = "chd_mi"

train, test = load_data(split=True)
y_train, y_test = train[OUTCOME].astype(int), test[OUTCOME].astype(int)
print(f"train n={len(train)} cases={y_train.sum()} | test n={len(test)} cases={y_test.sum()}")

# %% [markdown]
# ## Models and search spaces
#
# Preprocessing lives inside the pipeline (one-hot for the two categorical
# variables, numeric passthrough; trees need no scaling). `class_weight` is a
# tuned hyperparameter: reweighting can help ranking but hurts calibration,
# which we check below.

# %%
def make_pipeline(model, features):
    cats = [c for c in features if c in CATEGORICAL]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("model", model)])


MODELS = {
    "Random forest": (
        RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
        {
            "model__max_depth": [None, 4, 6, 8, 12],
            "model__min_samples_leaf": [1, 5, 10, 20, 50],
            "model__max_features": ["sqrt", 0.3, 0.5],
            "model__class_weight": [None, "balanced_subsample"],
        },
    ),
    "Gradient boosting": (
        HistGradientBoostingClassifier(early_stopping=False, random_state=SEED),
        {
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_iter": [100, 200, 400],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": loguniform(1e-3, 10),
            "model__class_weight": [None, "balanced"],
        },
    ),
}

# %% [markdown]
# ## Tuning on the training set (stratified 5-fold CV, ROC-AUC)

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fitted, cv_rows = {}, []
for model_name, (estimator, space) in MODELS.items():
    for fs_name, features in FEATURE_SETS.items():
        t0 = time.time()
        search = RandomizedSearchCV(
            make_pipeline(estimator, features), space, n_iter=N_ITER, scoring="roc_auc",
            cv=cv, random_state=SEED, n_jobs=-1, refit=True,
        )
        search.fit(train[features], y_train)
        fitted[(model_name, fs_name)] = search.best_estimator_
        i = search.best_index_
        cv_rows.append({
            "model": model_name, "features": fs_name,
            "cv_auc_mean": search.cv_results_["mean_test_score"][i],
            "cv_auc_sd": search.cv_results_["std_test_score"][i],
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        })
        print(f"{model_name:18s} {fs_name:9s} CV AUC {cv_rows[-1]['cv_auc_mean']:.3f} "
              f"({time.time() - t0:.0f}s)")

cv_table = pd.DataFrame(cv_rows)
cv_table.to_csv(OUT / "cv_results.csv", index=False)

# %% [markdown]
# ## Final evaluation on the test set (used once)
#
# Point estimates plus 95% bootstrap CIs; paired bootstrap for the AUC gain of
# adding the markers to M2 (same resampled participants for both models).

# %%
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(y_test), len(y_test)) for _ in range(N_BOOT)]
boot_idx = [b for b in boot_idx if y_test.values[b].min() != y_test.values[b].max()]

proba = {k: est.predict_proba(test[FEATURE_SETS[k[1]]])[:, 1] for k, est in fitted.items()}
yt = y_test.values


def ci(values):
    return np.percentile(values, [2.5, 97.5])


perf_rows = []
for (model_name, fs_name), p in proba.items():
    row = {"model": model_name, "features": fs_name}
    for metric, f in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score),
                      ("brier", brier_score_loss)]:
        row[metric] = f(yt, p)
        row[f"{metric}_lo"], row[f"{metric}_hi"] = ci([f(yt[b], p[b]) for b in boot_idx])
    perf_rows.append(row)
perf = pd.DataFrame(perf_rows)
perf.to_csv(OUT / "test_performance.csv", index=False)

delta_rows = []
for model_name in MODELS:
    base = proba[(model_name, "M2")]
    for fs_name in ["M2 + NLR", "M2 + CRP", "M3"]:
        p = proba[(model_name, fs_name)]
        deltas = [roc_auc_score(yt[b], p[b]) - roc_auc_score(yt[b], base[b]) for b in boot_idx]
        lo, hi = ci(deltas)
        delta_rows.append({
            "model": model_name, "comparison": f"{fs_name} vs M2",
            "delta_auc": roc_auc_score(yt, p) - roc_auc_score(yt, base),
            "lo": lo, "hi": hi, "p_boot_le_0": float(np.mean(np.array(deltas) <= 0)),
        })
delta = pd.DataFrame(delta_rows)
delta.to_csv(OUT / "delta_auc.csv", index=False)

print(f"\nPrevalence in test set (PR-AUC baseline): {yt.mean():.3f}")
print(perf[["model", "features", "roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "brier"]]
      .round(3).to_string(index=False))
print("\n" + delta.round(3).to_string(index=False))

# %% [markdown]
# ## Figures
#
# Palette: blue = random forest, orange = gradient boosting (validated for
# colour-vision deficiency); every series is also direct-labelled.

# %%
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"Random forest": "#2a78d6", "Gradient boosting": "#eb6834"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 8.5, "lines.linewidth": 2, "font.size": 9.5,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- ROC and PR curves, M3 --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
ax1.text(0.62, 0.55, "chance", color=MUTED, fontsize=8, rotation=38)
ax2.axhline(yt.mean(), ls="--", lw=1, color=MUTED)
ax2.text(0.98, yt.mean() + 0.015, f"prevalence {yt.mean():.3f}", color=MUTED, fontsize=8, ha="right")
for model_name, color in COLORS.items():
    p = proba[(model_name, "M3")]
    r = perf.set_index(["model", "features"]).loc[(model_name, "M3")]
    fpr, tpr, _ = roc_curve(yt, p)
    prec, rec, _ = precision_recall_curve(yt, p)
    ax1.plot(fpr, tpr, color=color, label=f"{model_name}  AUC {r.roc_auc:.3f}")
    ax2.plot(rec, prec, color=color, label=f"{model_name}  AP {r.pr_auc:.3f}")
ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (M3, test set)",
        xlim=(0, 1), ylim=(0, 1.01))
ax2.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve (M3, test set)",
        xlim=(0, 1), ylim=(0, 1.01))
ax1.legend(loc="lower right")
ax2.legend(loc="upper right")
save(fig, "roc_pr_m3")

# --- Test AUC by feature set (dot + 95% CI) ---------------------------------
fs_order = list(FEATURE_SETS)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
offsets = {"Random forest": -0.14, "Gradient boosting": 0.14}
for model_name, color in COLORS.items():
    sub = perf[perf.model == model_name].set_index("features").loc[fs_order]
    ypos = np.arange(len(fs_order)) + offsets[model_name]
    ax.errorbar(sub.roc_auc, ypos, xerr=[sub.roc_auc - sub.roc_auc_lo, sub.roc_auc_hi - sub.roc_auc],
                fmt="o", ms=6, color=color, ecolor=color, elinewidth=1.5, capsize=0, label=model_name,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.set_yticks(range(len(fs_order)), fs_order)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set(xlabel="Test ROC-AUC (95% bootstrap CI)", title="Discrimination by feature set")
ax.legend(loc="lower left")
save(fig, "auc_by_feature_set")

# --- Calibration, M3 --------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
ax.text(0.33, 0.37, "perfect calibration", color=MUTED, fontsize=8, rotation=45)
top = 0
for model_name, color in COLORS.items():
    p = proba[(model_name, "M3")]
    frac, mean_pred = calibration_curve(yt, p, n_bins=8, strategy="quantile")
    top = max(top, frac.max(), mean_pred.max())
    r = perf.set_index(["model", "features"]).loc[(model_name, "M3")]
    ax.plot(mean_pred, frac, marker="o", ms=6, color=color, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=f"{model_name}  Brier {r.brier:.4f}")
lim = min(1, top * 1.15)
ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted probability",
       ylabel="Observed fraction with CHD/MI", title="Calibration (M3, test set, octiles)")
ax.legend(loc="upper left")
save(fig, "calibration_m3")

# %% [markdown]
# ## Permutation importance on the test set (M3)
#
# Permutes the *original* columns (before one-hot encoding), scoring ROC-AUC.
# Unlike impurity importance this is not biased toward continuous features,
# but correlated features (e.g. the two markers, BMI and CRP) share credit.

# %%
LABELS = {"age": "Age", "male": "Male sex", "bmi": "BMI", "race_eth": "Race/ethnicity",
          "income_pir": "Income-to-poverty ratio", "smoking": "Smoking", "diabetes": "Diabetes",
          "hypertension": "Hypertension", "total_chol": "Total cholesterol", "hdl_chol": "HDL cholesterol",
          "egfr": "eGFR", "log_nlr": "log NLR", "log_crp": "log hs-CRP"}

imp_rows = []
for model_name in MODELS:
    res = permutation_importance(fitted[(model_name, "M3")], test[FEATURE_SETS["M3"]], yt,
                                 scoring="roc_auc", n_repeats=N_PERM, random_state=SEED, n_jobs=-1)
    for f, m, s in zip(FEATURE_SETS["M3"], res.importances_mean, res.importances_std):
        imp_rows.append({"model": model_name, "feature": f, "auc_drop_mean": m, "auc_drop_sd": s})
imp = pd.DataFrame(imp_rows)
imp.to_csv(OUT / "permutation_importance.csv", index=False)

order = imp.groupby("feature").auc_drop_mean.mean().sort_values().index
fig, ax = plt.subplots(figsize=(6.5, 4.6))
h = 0.38
for j, (model_name, color) in enumerate(COLORS.items()):
    sub = imp[imp.model == model_name].set_index("feature").loc[order]
    ypos = np.arange(len(order)) + (j - 0.5) * (h + 0.04)
    ax.barh(ypos, sub.auc_drop_mean, height=h, color=color, label=model_name,
            xerr=sub.auc_drop_sd, error_kw={"ecolor": INK2, "elinewidth": 0.8})
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(range(len(order)), [LABELS[f] for f in order])
ax.grid(axis="y", visible=False)
ax.set(xlabel="Drop in test ROC-AUC when permuted (mean ± SD, 30 repeats)",
       title="Permutation importance (M3)")
ax.legend(loc="lower right")
save(fig, "permutation_importance_m3")

# %% [markdown]
# ## SHAP for gradient boosting (M3)
#
# SHAP values are in log-odds. One-hot columns are summed back to their
# original variable for the importance ranking.

# %%
hgb = fitted[("Gradient boosting", "M3")]
X_test_t = hgb.named_steps["pre"].transform(test[FEATURE_SETS["M3"]])
names_t = hgb.named_steps["pre"].get_feature_names_out()
sv = shap.TreeExplainer(hgb.named_steps["model"]).shap_values(X_test_t)
sv = sv[1] if isinstance(sv, list) else sv
sv = pd.DataFrame(sv, columns=names_t)


def original(name):
    return next((c for c in CATEGORICAL if name.startswith(c + "_")), name)


mean_abs = sv.abs().T.groupby(original).sum().T.mean().sort_values()
mean_abs.rename("mean_abs_shap").to_csv(OUT / "shap_mean_abs_gb_m3.csv")

fig, ax = plt.subplots(figsize=(6, 4.4))
ax.barh(range(len(mean_abs)), mean_abs.values, height=0.6, color=COLORS["Gradient boosting"])
ax.set_yticks(range(len(mean_abs)), [LABELS[f] for f in mean_abs.index])
ax.grid(axis="y", visible=False)
ax.set(xlabel="Mean |SHAP value| (log-odds), test set", title="SHAP importance, gradient boosting (M3)")
save(fig, "shap_importance_gb_m3")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, col, raw, label in [(axes[0], "log_nlr", "nlr", "NLR"), (axes[1], "log_crp", "crp", "hs-CRP (mg/L)")]:
    ax.scatter(test[raw], sv[col], s=10, alpha=0.45, color=COLORS["Gradient boosting"], linewidths=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xscale("log")
    ax.set(xlabel=f"{label}, log scale", title=f"SHAP dependence: {label}")
axes[0].set_ylabel("SHAP value (log-odds of CHD/MI)")
save(fig, "shap_dependence_markers_gb_m3")

# %% [markdown]
# ## Extra: gradient boosting on the eligible sample with native NaN handling
#
# The eligible sample keeps participants with missing covariates (mostly
# income). HistGradientBoosting handles NaN natively, so no imputation and no
# complete-case loss. Same hyperparameters as the complete-case M3 model; the
# split column is shared, so the complete-case test participants are a subset.

# %%
elig_train, elig_test = load_data("eligible", split=True)
params = hgb.named_steps["model"].get_params()
nan_rows = []
for fs_name in ["M2", "M3"]:
    feats = FEATURE_SETS[fs_name]
    est = make_pipeline(HistGradientBoostingClassifier(**params), feats)
    est.fit(elig_train[feats], elig_train[OUTCOME].astype(int))
    p = est.predict_proba(elig_test[feats])[:, 1]
    ye = elig_test[OUTCOME].astype(int).values
    cc = elig_test["complete_case"].values
    nan_rows.append({
        "features": fs_name, "train_n": len(elig_train), "test_n": len(ye), "test_cases": int(ye.sum()),
        "roc_auc_all_eligible": roc_auc_score(ye, p),
        "roc_auc_complete_case_subset": roc_auc_score(ye[cc], p[cc]),
        "roc_auc_incomplete_subset": roc_auc_score(ye[~cc], p[~cc]) if ye[~cc].min() != ye[~cc].max() else np.nan,
        "incomplete_test_n": int((~cc).sum()), "incomplete_test_cases": int(ye[~cc].sum()),
    })
nan_table = pd.DataFrame(nan_rows)
nan_table.to_csv(OUT / "eligible_native_nan_gb.csv", index=False)
print("\n" + nan_table.round(3).to_string(index=False))
