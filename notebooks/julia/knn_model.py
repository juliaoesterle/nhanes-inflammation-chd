# %% [markdown]
# # k-nearest neighbours (k-NN) for CHD/MI
#
# Secondary question (H1c): do NLR and hs-CRP add discriminative information
# beyond established risk factors?
#
# * Feature sets M0-M3 (+ M2+NLR, M2+CRP); no medication variables (leakage)
# * Sensitivity sets without total/HDL cholesterol (treatment proxy)
# * Preprocessing inside the pipeline, so scaling is fitted on training folds only
# * Tuning: exhaustive grid, stratified 5-fold CV on the TRAINING set only
# * Stability of the marker gain: nested CV on the training set
# * Test set: only used in the final evaluation cell
#
# Run from the repo root:  python notebooks/julia/knn_model.py
# (cells also run interactively in VS Code / Jupyter via "# %%").

# %%
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prepare_data import CONFIG, load_data  # noqa: E402

OUT = Path(__file__).resolve().parent / "results_knn"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = CONFIG["random_state"]
N_BOOT = 1000       # bootstrap resamples for test-set CIs
N_PERM = 30         # permutation-importance repeats
N_PDP = 500         # training rows averaged over per partial-dependence point

K_GRID = [5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 300, 400, 500]

# %% [markdown]
# ## Feature sets (shared with the other models)

# %%
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
BINARY = ["male", "diabetes", "hypertension"]
OUTCOME = "chd_mi"

train, test = load_data(split=True)
y_train, y_test = train[OUTCOME].astype(int), test[OUTCOME].astype(int)
print(f"train n={len(train)} cases={y_train.sum()} | test n={len(test)} cases={y_test.sum()}")

# %% [markdown]
# ## Pipeline
#
# k-NN compares raw distances, so preprocessing defines what "similar" means:
#
# * continuous variables: z-scores, so 1 mg/dL of cholesterol and 1 year of age
#   do not count the same by accident
# * binary and one-hot variables: kept at 0/1, so a category difference has a
#   fixed size. Standardising them would inflate the distance for rare categories.
#
# `n_neighbors`, `weights` and `metric` are tuned. A k-NN has no class weights;
# its probability is simply the (weighted) share of cases among the neighbours.

# %%
def make_pipeline(features):
    cats = [c for c in features if c in CATEGORICAL]
    binary = [c for c in features if c in BINARY]
    continuous = [c for c in features if c not in CATEGORICAL + BINARY]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), continuous),
            ("bin", "passthrough", binary),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("model", KNeighborsClassifier())])


GRID = {
    "model__n_neighbors": K_GRID,
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"],
}

# %% [markdown]
# ## Tuning on the training set (stratified 5-fold CV, ROC-AUC)

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fitted, cv_rows, cv_curves = {}, [], []
for fs_name, features in FEATURE_SETS.items():
    t0 = time.time()
    search = GridSearchCV(make_pipeline(features), GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(train[features], y_train)
    fitted[fs_name] = search.best_estimator_
    i = search.best_index_
    best = search.best_params_
    cv_rows.append({
        "features": fs_name,
        "n_neighbors": best["model__n_neighbors"],
        "weights": best["model__weights"],
        "metric": best["model__metric"],
        "cv_auc_mean": search.cv_results_["mean_test_score"][i],
        "cv_auc_sd": search.cv_results_["std_test_score"][i],
    })
    curve = pd.DataFrame(search.cv_results_)
    curve = curve[["param_model__n_neighbors", "param_model__weights", "param_model__metric",
                   "mean_test_score", "std_test_score"]].copy()
    curve.columns = ["n_neighbors", "weights", "metric", "cv_auc_mean", "cv_auc_sd"]
    curve.insert(0, "features", fs_name)
    cv_curves.append(curve)
    print(f"{fs_name:11s} k={best['model__n_neighbors']:<4d} {best['model__weights']:9s} "
          f"{best['model__metric']:10s} CV AUC {cv_rows[-1]['cv_auc_mean']:.3f} ({time.time() - t0:.0f}s)")

cv_table = pd.DataFrame(cv_rows)
cv_table.to_csv(OUT / "cv_results.csv", index=False)
cv_curves = pd.concat(cv_curves, ignore_index=True)
cv_curves.to_csv(OUT / "cv_grid_all.csv", index=False)

# %% [markdown]
# ## Grid boundary check on the training set
#
# Most feature sets select the largest k in the grid, so the curve might still
# be rising. This repeats the CV above k = 500 (training set only). Decision rule
# fixed before looking at the result: keep K_GRID unless a larger k improves the
# CV AUC by more than 0.005, which is well below the SD across folds.

# %%
BOUNDARY_K = [500, 750, 1000, 1500, 2000]
boundary_rows = []
for name in ["M2", "M3", "M2 no chol", "M3 no chol"]:
    features = FEATURE_SETS[name]
    s = GridSearchCV(make_pipeline(features), {**GRID, "model__n_neighbors": BOUNDARY_K},
                     scoring="roc_auc", cv=cv, n_jobs=-1)
    s.fit(train[features], y_train)
    r = pd.DataFrame(s.cv_results_)
    best_per_k = r.loc[r.groupby("param_model__n_neighbors")["mean_test_score"].idxmax()]
    for _, row in best_per_k.iterrows():
        boundary_rows.append({"features": name, "n_neighbors": row["param_model__n_neighbors"],
                              "weights": row["param_model__weights"], "metric": row["param_model__metric"],
                              "cv_auc_mean": row["mean_test_score"], "cv_auc_sd": row["std_test_score"]})
boundary = pd.DataFrame(boundary_rows)
boundary.to_csv(OUT / "k_boundary_check.csv", index=False)
print(boundary.round(4).to_string(index=False))

# %% [markdown]
# ## Nested CV on the training set: is the marker gain stable?
#
# The test set has only 67 cases, so one split is noisy. Here k-NN is re-tuned
# inside every outer fold, and each feature set sees the same outer folds, so
# the AUC differences are paired. The outer folds overlap, so min/max show the
# spread across folds and are not a confidence interval.

# %%
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
NESTED_SETS = ["M2", "M2 + NLR", "M2 + CRP", "M3", "M2 no chol", "M3 no chol"]
folds = list(outer.split(train, y_train))
nested = {name: [] for name in NESTED_SETS}
t0 = time.time()
for name in NESTED_SETS:
    features = FEATURE_SETS[name]
    for tr, va in folds:
        s = GridSearchCV(make_pipeline(features), GRID, scoring="roc_auc", cv=inner, n_jobs=-1)
        s.fit(train.iloc[tr][features], y_train.iloc[tr])
        nested[name].append(roc_auc_score(y_train.iloc[va], s.predict_proba(train.iloc[va][features])[:, 1]))
print(f"nested CV finished in {time.time() - t0:.0f}s")
nested = pd.DataFrame(nested)

nested_rows = []
for a, b in [("M2 + NLR", "M2"), ("M2 + CRP", "M2"), ("M3", "M2"), ("M3 no chol", "M2 no chol")]:
    d = nested[a] - nested[b]
    nested_rows.append({
        "comparison": f"{a} vs {b}", "auc_a": nested[a].mean(), "auc_b": nested[b].mean(),
        "delta_auc_mean": d.mean(), "delta_auc_min": d.min(), "delta_auc_max": d.max(),
        "folds_with_gain": int((d > 0).sum()), "n_folds": len(d),
    })
nested_table = pd.DataFrame(nested_rows)
nested_table.to_csv(OUT / "nested_cv_delta_auc.csv", index=False)
print(nested_table.round(4).to_string(index=False))

# %% [markdown]
# ## Operating threshold, chosen before touching the test set
#
# Youden's J on out-of-fold training predictions of the tuned M3 model.

# %%
oof = cross_val_predict(clone(fitted["M3"]), train[FEATURE_SETS["M3"]], y_train, cv=cv,
                        method="predict_proba", n_jobs=-1)[:, 1]
fpr_tr, tpr_tr, thr_tr = roc_curve(y_train, oof)
THRESHOLD = thr_tr[np.argmax(tpr_tr - fpr_tr)]
print(f"Youden threshold from training OOF predictions: {THRESHOLD:.4f}")

# %% [markdown]
# ## Final evaluation on the test set
#
# Point estimates plus 95% bootstrap CIs; paired bootstrap for the AUC gain
# (same resampled participants for both models).

# %%
yt = y_test.values
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(yt), len(yt)) for _ in range(N_BOOT)]
boot_idx = [b for b in boot_idx if yt[b].min() != yt[b].max()]
proba = {name: est.predict_proba(test[FEATURE_SETS[name]])[:, 1] for name, est in fitted.items()}


def ci(values):
    return np.percentile(values, [2.5, 97.5])


perf_rows = []
for name, p in proba.items():
    row = {"features": name}
    for metric, f in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score),
                      ("brier", brier_score_loss)]:
        row[metric] = f(yt, p)
        row[f"{metric}_lo"], row[f"{metric}_hi"] = ci([f(yt[b], p[b]) for b in boot_idx])
    perf_rows.append(row)
perf = pd.DataFrame(perf_rows)
perf["brier_null"] = brier_score_loss(yt, np.full(len(yt), y_train.mean()))
perf.to_csv(OUT / "test_performance.csv", index=False)

delta_rows = []
for a, b in [("M2 + NLR", "M2"), ("M2 + CRP", "M2"), ("M3", "M2"), ("M3 no chol", "M2 no chol"),
             ("M2", "M2 no chol")]:
    deltas = [roc_auc_score(yt[i], proba[a][i]) - roc_auc_score(yt[i], proba[b][i]) for i in boot_idx]
    lo, hi = ci(deltas)
    delta_rows.append({"comparison": f"{a} vs {b}",
                       "delta_auc": roc_auc_score(yt, proba[a]) - roc_auc_score(yt, proba[b]),
                       "lo": lo, "hi": hi})
delta = pd.DataFrame(delta_rows)
delta.to_csv(OUT / "delta_auc.csv", index=False)


def threshold_metrics(y, pred):
    tp = np.sum((y == 1) & pred)
    fp = np.sum((y == 0) & pred)
    fn = np.sum((y == 1) & ~pred)
    tn = np.sum((y == 0) & ~pred)
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan}


pred_m3 = proba["M3"] >= THRESHOLD
point = threshold_metrics(yt, pred_m3)
boots = pd.DataFrame([threshold_metrics(yt[b], pred_m3[b]) for b in boot_idx])
threshold_table = pd.DataFrame([
    {"metric": m, "value": v, "lo": np.nanpercentile(boots[m], 2.5), "hi": np.nanpercentile(boots[m], 97.5)}
    for m, v in point.items()
])
threshold_table.insert(0, "threshold", THRESHOLD)
threshold_table.to_csv(OUT / "threshold_metrics_m3.csv", index=False)

print(f"\nPrevalence in test set (PR-AUC baseline): {yt.mean():.3f}")
print(perf[["features", "roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "brier", "brier_null"]]
      .round(4).to_string(index=False))
print("\n" + delta.round(4).to_string(index=False))
print("\n" + threshold_table.round(3).to_string(index=False))

# %% [markdown]
# ## Permutation importance on the test set (M3)
#
# Permutes the original columns, scoring ROC-AUC. Correlated features share
# credit, so a small value does not prove a variable is unrelated to CHD/MI.

# %%
res = permutation_importance(fitted["M3"], test[FEATURE_SETS["M3"]], yt, scoring="roc_auc",
                             n_repeats=N_PERM, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"feature": FEATURE_SETS["M3"], "auc_drop_mean": res.importances_mean,
                    "auc_drop_sd": res.importances_std}).sort_values("auc_drop_mean")
imp.to_csv(OUT / "permutation_importance_m3.csv", index=False)
print("\n" + imp.round(4).to_string(index=False))

# %% [markdown]
# ## Partial dependence of predicted risk on the markers (M3, training data)
#
# Each marker is set to a grid value for a random sample of training rows, and
# the predictions are averaged. This shows what the model learned; it is not a
# causal dose-response curve.

# %%
pdp_rows = []
sample = train.sample(N_PDP, random_state=SEED)[FEATURE_SETS["M3"]]
for col, raw in [("log_nlr", "nlr"), ("log_crp", "crp")]:
    for q in np.linspace(0.02, 0.98, 25):
        value = train[col].quantile(q)
        X = sample.copy()
        X[col] = value
        pdp_rows.append({"marker": raw, "value": float(np.exp(value)),
                         "mean_predicted_risk": fitted["M3"].predict_proba(X)[:, 1].mean()})
pdp = pd.DataFrame(pdp_rows)
pdp.to_csv(OUT / "partial_dependence_m3.csv", index=False)

# %% [markdown]
# ## Figures

# %%
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


# --- CV AUC against k -------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for name, color in [("M2", BLUE), ("M3", ORANGE)]:
    best = cv_table.set_index("features").loc[name]
    c = cv_curves[(cv_curves.features == name) & (cv_curves.weights == best.weights)
                  & (cv_curves.metric == best.metric)].astype({"n_neighbors": int}).sort_values("n_neighbors")
    ax.plot(c.n_neighbors, c.cv_auc_mean, marker="o", ms=5, color=color, markeredgecolor=SURFACE,
            markeredgewidth=1, label=f"{name} ({best.weights}, {best.metric})")
ax.set_xscale("log")
ax.set_xticks(K_GRID[::2], [str(k) for k in K_GRID[::2]])
ax.minorticks_off()
ax.set(xlabel="Number of neighbours k (log scale)", ylabel="Mean 5-fold CV ROC-AUC",
       title="Tuning curve on the training set")
ax.legend(loc="lower right")
save(fig, "cv_auc_vs_k")

# --- Test AUC by feature set ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.9))
sub = perf.set_index("features").loc[list(FEATURE_SETS)]
ypos = np.arange(len(sub))
ax.errorbar(sub.roc_auc, ypos, xerr=[sub.roc_auc - sub.roc_auc_lo, sub.roc_auc_hi - sub.roc_auc], fmt="o", ms=6,
            color=BLUE, ecolor=BLUE, elinewidth=1.5, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.set_yticks(ypos, sub.index)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set(xlabel="Test ROC-AUC (95% bootstrap CI)", title="k-NN discrimination by feature set")
save(fig, "auc_by_feature_set")

# --- ROC and PR, M2 vs M3 ---------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
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
ax2.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve (test set)", xlim=(0, 1), ylim=(0, 1.01))
ax1.legend(loc="lower right")
ax2.legend(loc="upper right")
save(fig, "roc_pr_m2_m3")

# --- Calibration, M3 --------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 4.2))
frac, mean_pred = calibration_curve(yt, proba["M3"], n_bins=8, strategy="quantile")
lim = min(1, max(frac.max(), mean_pred.max()) * 1.15)
ax.plot([0, lim], [0, lim], ls="--", lw=1, color=MUTED)
ax.plot(mean_pred, frac, marker="o", ms=6, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted probability", ylabel="Observed fraction with CHD/MI",
       title=f"Calibration (M3, test set), Brier {perf.set_index('features').loc['M3'].brier:.4f}")
save(fig, "calibration_m3")

# --- Permutation importance -------------------------------------------------
LABELS = {"age": "Age", "male": "Male sex", "bmi": "BMI", "race_eth": "Race/ethnicity",
          "income_pir": "Income-to-poverty ratio", "smoking": "Smoking", "diabetes": "Diabetes",
          "hypertension": "Hypertension", "total_chol": "Total cholesterol", "hdl_chol": "HDL cholesterol",
          "egfr": "eGFR", "log_nlr": "log NLR", "log_crp": "log hs-CRP"}
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.barh(range(len(imp)), imp.auc_drop_mean, height=0.6, color=BLUE, xerr=imp.auc_drop_sd,
        error_kw={"ecolor": INK2, "elinewidth": 0.8})
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(range(len(imp)), [LABELS[f] for f in imp.feature])
ax.grid(axis="y", visible=False)
ax.set(xlabel=f"Drop in test ROC-AUC when permuted (mean ± SD, {N_PERM} repeats)",
       title="Permutation importance, k-NN (M3)")
save(fig, "permutation_importance_m3")

# --- Partial dependence -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
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

# %%
versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}
pd.Series(versions, name="version").to_csv(OUT / "versions.csv", header=True)
print("\n", versions)
