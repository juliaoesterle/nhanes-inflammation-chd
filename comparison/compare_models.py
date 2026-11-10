"""
Collect the results of all group members and plot them side by side.

Sources (read only, nothing is re-fitted):
  k-NN           notebooks/julia/results_knn/*.csv
  Random forest  notebooks/marco/results/*.csv
  Lasso, Ridge   outputs saved in notebooks/Peti/01_lasso_ridge.ipynb
  Logistic reg.  outputs saved in notebooks/jannis/04_logistic_regression.ipynb

Run from the repo root after everyone has pushed:  python comparison/compare_models.py
If a notebook was re-run with different output, the parser fails loudly
instead of silently plotting old numbers.

Outputs in comparison/:
  results_summary.csv          prediction metrics, one row per model x feature set
  delta_auc.csv                AUC gain of M3 over M2 per model
  odds_ratios.csv              adjusted odds ratios from the logistic regression
  figures/*.png, *.pdf
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FEATURE_SETS = ["M0", "M1", "M2", "M3"]
# fixed order = fixed colour; identity is also carried by marker and label
MODELS = {
    "k-NN": {"owner": "Julia", "color": "#2a78d6", "marker": "o"},
    "Random forest": {"owner": "Marco", "color": "#eb6834", "marker": "s"},
    "Lasso": {"owner": "Petya", "color": "#1baf7a", "marker": "^"},
    "Ridge": {"owner": "Petya", "color": "#eda100", "marker": "D"},
}


# =====================================================================
# 1. Collect
# =====================================================================
def notebook_outputs(path):
    """All text outputs of a notebook, one string per output."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    texts = []
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain") or ""
            texts.append("".join(t) if isinstance(t, list) else str(t))
    return texts


def from_csv_model(name, folder):
    perf = pd.read_csv(ROOT / folder / "test_performance.csv").set_index("features")
    cv = pd.read_csv(ROOT / folder / "cv_results.csv").set_index("features")
    rows = []
    for fs in FEATURE_SETS:
        r = perf.loc[fs]
        rows.append({"model": name, "feature_set": fs, "cv_auc": cv.loc[fs, "cv_auc_mean"],
                     "roc_auc": r.roc_auc, "roc_auc_lo": r.roc_auc_lo, "roc_auc_hi": r.roc_auc_hi,
                     "pr_auc": r.pr_auc, "brier": r.brier,
                     "class_weighted": str(cv.loc[fs].get("class_weight", "None")) not in ("None", "nan")})
    d = pd.read_csv(ROOT / folder / "delta_auc.csv").set_index("comparison").loc["M3 vs M2"]
    return rows, {"model": name, "delta_auc": d.delta_auc, "lo": d.lo, "hi": d.hi}


def from_peti():
    texts = notebook_outputs(ROOT / "notebooks/Peti/01_lasso_ridge.ipynb")
    num = r"(-?\d+\.\d+)"
    test = [m.groups() for t in texts if "Brier score" in t
            for m in re.finditer(rf"^\s*\d+\s+(Lasso|Ridge)\s+(M\d)\s+{num}\s+{num}\s+{num}\s*$", t, re.M)]
    cv = [m.groups() for t in texts if "Mean CV ROC-AUC" in t
          for m in re.finditer(rf"^\s*\d+\s+(Lasso|Ridge)\s+(M\d)\s+{num}\s+{num}\s*$", t, re.M)]
    delta = [m.groups() for t in texts
             for m in re.finditer(rf"(Lasso|Ridge)\s*\nChange in ROC-AUC \(M3 - M2\): {num}\s*\n"
                                  rf"95% bootstrap CI: \({num}, {num}\)", t)]
    if len(test) != 8 or len(cv) != 8 or len(delta) != 2:
        raise RuntimeError(f"Petya's notebook output changed (found {len(test)} test rows, "
                           f"{len(cv)} CV rows, {len(delta)} delta blocks); update the parser.")
    cv = {(m, fs): float(a) for m, fs, _, a in cv}
    rows = [{"model": m, "feature_set": fs, "cv_auc": cv[(m, fs)], "roc_auc": float(roc),
             "roc_auc_lo": np.nan, "roc_auc_hi": np.nan, "pr_auc": float(pr), "brier": float(br),
             "class_weighted": False} for m, fs, roc, pr, br in test]
    deltas = [{"model": m, "delta_auc": float(d), "lo": float(lo), "hi": float(hi)} for m, d, lo, hi in delta]
    return rows, deltas


def from_jannis():
    texts = notebook_outputs(ROOT / "notebooks/jannis/04_logistic_regression.ipynb")
    ci = r"(\d+\.\d+) \((\d+\.\d+)[–-](\d+\.\d+)\)"
    table = next(t for t in texts if "Model 1" in t and "hs-CRP" in t and "(" in t)
    pvals = next(t for t in texts if "Model 1" in t and "hs-CRP" in t and "(" not in t)
    rows = []
    for marker in ["NLR", "hs-CRP"]:
        line = next(l for l in table.splitlines() if l.strip().startswith(marker + " "))
        p_line = next(l for l in pvals.splitlines() if l.strip().startswith(marker + " "))
        ors, ps = re.findall(ci, line), p_line.split()[1:]
        if len(ors) != 3 or len(ps) != 3:
            raise RuntimeError(f"Jannis' table for {marker} changed; update the parser.")
        for i, ((o, lo, hi), p) in enumerate(zip(ors, ps), start=1):
            rows.append({"marker": marker, "model": f"Model {i}", "or": float(o),
                         "lo": float(lo), "hi": float(hi), "p": p})
    return pd.DataFrame(rows)


rows, deltas = [], []
for name, folder in [("k-NN", "notebooks/julia/results_knn"), ("Random forest", "notebooks/marco/results")]:
    r, d = from_csv_model(name, folder)
    rows += r
    deltas.append(d)
r, d = from_peti()
rows += r
deltas += d
summary = pd.DataFrame(rows)
summary.insert(1, "owner", summary.model.map(lambda m: MODELS[m]["owner"]))
delta = pd.DataFrame(deltas).set_index("model").loc[list(MODELS)].reset_index()
ors = from_jannis()

summary.round(4).to_csv(OUT / "results_summary.csv", index=False)
delta.round(4).to_csv(OUT / "delta_auc.csv", index=False)
ors.to_csv(OUT / "odds_ratios.csv", index=False)
print(summary.round(3).to_string(index=False))
print("\n" + delta.round(4).to_string(index=False))
print("\n" + ors.to_string(index=False))

# =====================================================================
# 2. Plot
# =====================================================================
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 8.5, "lines.linewidth": 2, "font.size": 9.5,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def spread(values, min_gap):
    """Nudge label positions apart so they do not overlap (keeps the order)."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    shift = (pos - np.array(values)[order]).mean()
    out = np.empty_like(pos)
    out[order] = pos - shift
    return out


# --- Figure 1: association (Jannis) --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
labels = {"Model 1": "Model 1: age, sex", "Model 2": "Model 2: + risk factors",
          "Model 3": "Model 3: + lipid medication"}
for ax, marker in zip(axes, ["NLR", "hs-CRP"]):
    sub = ors[ors.marker == marker].reset_index(drop=True)
    y = np.arange(len(sub))
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.errorbar(sub["or"], y, xerr=[sub["or"] - sub.lo, sub.hi - sub["or"]], fmt="o", ms=7,
                color="#2a78d6", ecolor="#2a78d6", elinewidth=2, capsize=0,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    for yi, r in sub.iterrows():
        p_text = f"p < {r.p[1:]}" if r.p.startswith("<") else f"p = {r.p}"
        ax.text(1.53, yi, f"{r['or']:.2f} ({r.lo:.2f}–{r.hi:.2f})  {p_text}", va="center",
                fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1.52)
    ax.set_xticks([0.8, 1, 1.2, 1.4], ["0.8", "1", "1.2", "1.4"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_title(f"log {marker}, per 1 SD", loc="left")
    ax.set_xlabel("Odds ratio for CHD/MI (95% CI)")
axes[0].set_yticks(range(3), [labels[m] for m in ["Model 1", "Model 2", "Model 3"]])
axes[0].invert_yaxis()
fig.subplots_adjust(wspace=0.75)
save(fig, "1_association_odds_ratios")

# --- Figure 2: added value of the markers (H1c) ------------------------
fig, ax = plt.subplots(figsize=(7.5, 2.9))
ax.axvline(0, color=MUTED, lw=1, ls="--")
ax.text(0.0003, 0.5, "no gain", color=MUTED, fontsize=8, va="center")
for i, r in delta.iterrows():
    m = MODELS[r.model]
    ax.errorbar(r.delta_auc, i, xerr=[[r.delta_auc - r.lo], [r.hi - r.delta_auc]], fmt=m["marker"], ms=7,
                color=m["color"], ecolor=m["color"], elinewidth=2, capsize=0,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(0.0205, i, f"{r.delta_auc:+.3f} ({r.lo:+.3f} to {r.hi:+.3f})".replace("-", "−"),
            va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(delta)), [f"{m} ({MODELS[m]['owner']})" for m in delta.model])
ax.invert_yaxis()
ax.set_xlim(-0.02, 0.02)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Change in test ROC-AUC when adding log NLR + log hs-CRP to M2 (95% bootstrap CI)")
ax.set_title("Do the markers improve prediction? M3 vs. M2", loc="left")
save(fig, "2_marker_gain_delta_auc")

# --- Figure 3: performance across feature sets ------------------------
panels = [("roc_auc", "Test ROC-AUC", "higher is better", 0.006, "{:.3f}"),
          ("pr_auc", "Test PR-AUC", "higher is better · random = 0.074", 0.016, "{:.3f}"),
          ("brier", "Test Brier score", "lower is better · always 7.4% = 0.068", 0.0012, "{:.4f}")]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
x = np.arange(len(FEATURE_SETS))
for ax, (col, title, note, gap, fmt) in zip(axes, panels):
    ends = []
    for name, m in MODELS.items():
        s = summary[summary.model == name].set_index("feature_set").loc[FEATURE_SETS]
        if col == "brier":  # class-weighted Brier scores are not comparable: leave them out
            s = s.assign(brier=s.brier.where(~s.class_weighted))
        ax.plot(x, s[col], color=m["color"], lw=2, zorder=2)
        for xi, (v, w) in enumerate(zip(s[col], s.class_weighted)):
            if np.isnan(v):
                continue
            ax.plot(xi, v, m["marker"], ms=7, zorder=3, markeredgewidth=1.5,
                    color=SURFACE if w else m["color"], markeredgecolor=m["color"])
        ends.append((name, s[col].iloc[-1]))
    ys = spread([v for _, v in ends], gap)
    for (name, v), yl in zip(ends, ys):
        ax.plot([3.06, 3.22], [v, yl], color=AXIS, lw=0.8)
        ax.text(3.27, yl, f"{name} " + fmt.format(v), va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, FEATURE_SETS)
    ax.set_xlim(-0.25, 3.25)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", pad=18)
    ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    ax.set_xlabel("Feature set")
axes[0].legend(handles=[plt.Line2D([], [], color=m["color"], marker=m["marker"], lw=2, label=n)
                        for n, m in MODELS.items()],
               loc="lower right", fontsize=8)
fig.text(0.01, -0.06, "M0 = age, sex · M1 = + BMI, race/ethnicity, income · M2 = + smoking, diabetes, "
         "hypertension, cholesterol, HDL, eGFR · M3 = M2 + log NLR + log hs-CRP. "
         "Hollow markers = random forest trained with class weighting\n(its Brier scores for M0–M2 are not comparable and are left out). Test-set CIs are about ±0.04 for ROC-AUC (see the individual results); Figure 2 shows the marker gain with CIs.",
         fontsize=8, color=INK2)
fig.subplots_adjust(wspace=0.75)
save(fig, "3_performance_by_feature_set")
print("\nFigures written to", FIG.relative_to(ROOT))
