"""
02 - Descriptive analysis, missing-data diagnostics, selection bias, sample size.

  Table 1                  analysis sample by CHD/MI status (median [IQR] / n (%), SMD)
  Selection bias           analysis sample vs. excluded adults (SMD instead of p-values:
                           with thousands of people every small difference is "significant")
  Missingness mechanism    logistic model for "income missing" -> is MCAR plausible? MAR?
  Sample size              events per parameter (rule of thumb >= 10; course W3 checklist)

Figures: flowchart, marker distributions, the "statin paradox", correlations,
missingness, selection bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import config as cfg
from common import (AQUA, AXIS, BLUE, BLUE_RAMP, DEEMPH, DIVERGING_MID, GRID, INK, INK2,
                    MUTED, ORANGE, RED, SURFACE, Logger, bonferroni, fmt_p, load_analysis_sample,
                    save_fig, set_style, smd)

log = Logger("02_descriptives")
set_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

RACE = {"nh_white": "Non-Hispanic White", "nh_black": "Non-Hispanic Black",
        "mexican_american": "Mexican American", "other_hispanic": "Other Hispanic",
        "nh_asian": "Non-Hispanic Asian", "other_multi": "Other / multiracial"}
SMOKE = {"never": "Never", "former": "Former", "current": "Current"}


def med_iqr(x, digits=1):
    x = x.dropna()
    return f"{x.median():.{digits}f} [{x.quantile(.25):.{digits}f}, {x.quantile(.75):.{digits}f}]"


def n_pct(x):
    x = x.dropna()
    return f"{int(x.sum()):,} ({100 * x.mean():.1f})"


def table1(df: pd.DataFrame) -> pd.DataFrame:
    cases, ctrls = df[df.chd_mi == 1], df[df.chd_mi == 0]
    groups = {f"Overall (n={len(df):,})": df, f"No CHD/MI (n={len(ctrls):,})": ctrls,
              f"CHD/MI (n={len(cases):,})": cases}
    rows, pvals = [], []

    def chi2_p(var):
        tab = pd.crosstab(df[var], df.chd_mi)
        return stats.chi2_contingency(tab, correction=False)[1]

    def add(label, fn, var=None, binary=False, smd_val=None, test=None):
        row = {"Characteristic": label}
        for g, d in groups.items():
            row[g] = fn(d)
        if smd_val is None and var is not None:
            smd_val = smd(cases[var], ctrls[var], binary=binary)
        row["SMD"] = "" if smd_val is None else f"{smd_val:.2f}"
        row["Test"] = ""
        row["_p"] = np.nan
        if test == "mw":       # continuous, skewed -> Mann-Whitney U (course W3)
            row["Test"] = "Mann-Whitney U"
            row["_p"] = stats.mannwhitneyu(cases[var].dropna(), ctrls[var].dropna()).pvalue
        elif test == "chi2":   # categorical -> chi-square test of independence
            row["Test"] = "Chi-square"
            row["_p"] = chi2_p(var)
        row["Missing %"] = "" if var is None else f"{100 * df[var].isna().mean():.1f}"
        rows.append(row)

    cont = [("Age, years", "age", 0), ("Income-to-poverty ratio", "income_pir", 2),
            ("BMI, kg/m2", "bmi", 1), ("Systolic BP, mmHg", "sbp", 0),
            ("Total cholesterol, mg/dL", "total_chol", 0), ("HDL cholesterol, mg/dL", "hdl_chol", 0),
            ("eGFR, mL/min/1.73m2", "egfr", 0), ("NLR", "nlr", 2), ("hs-CRP, mg/L", "crp", 2)]
    for label, var, dg in cont:
        add(f"{label}, median [IQR]", lambda d, v=var, k=dg: med_iqr(d[v], k), var, test="mw")
    for label, var in [("Male sex", "male"), ("Diabetes", "diabetes"),
                       ("Hypertension", "hypertension"), ("hs-CRP > 10 mg/L", "crp_above_10"),
                       ("Lipid-lowering medication", "lipid_med"), ("BP medication", "bp_med")]:
        add(f"{label}, n (%)", lambda d, v=var: n_pct(d[v]), var, binary=True, test="chi2")
    for header, var, levels in [("Race/ethnicity, n (%)", "race_eth", RACE),
                                ("Smoking, n (%)", "smoking", SMOKE)]:
        rows.append({"Characteristic": header, **{g: "" for g in groups}, "SMD": "",
                     "Test": "Chi-square", "_p": chi2_p(var),
                     "Missing %": f"{100 * df[var].isna().mean():.1f}"})
        for code, lab in levels.items():
            add(f"    {lab}", lambda d, v=var, c=code: n_pct((d[v] == c).astype(float).where(d[v].notna())),
                None, smd_val=smd((cases[var] == code).astype(float), (ctrls[var] == code).astype(float), True))
    t = pd.DataFrame(rows)
    has_p = t["_p"].notna()
    t["p"] = ""
    t["p (Bonferroni)"] = ""
    t.loc[has_p, "p"] = [fmt_p(p) for p in t.loc[has_p, "_p"]]
    t.loc[has_p, "p (Bonferroni)"] = [fmt_p(p) for p in bonferroni(t.loc[has_p, "_p"].to_numpy())]
    return t.drop(columns="_p")


def shape_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Skewness/kurtosis on raw vs log scale and outlier counts (course W2/W3)."""
    rows = []
    for raw, lg, lab in [("nlr", "log_nlr", "NLR"), ("crp", "log_crp", "hs-CRP")]:
        for scale, v in [("raw", df[raw]), ("log", df[lg])]:
            v = v.dropna()
            q1, q3 = v.quantile([.25, .75])
            iqr = q3 - q1
            z = (v - v.mean()) / v.std()
            rows.append({
                "marker": lab, "scale": scale, "skewness": stats.skew(v),
                "excess_kurtosis": stats.kurtosis(v),
                "shapiro_p_n5000": stats.shapiro(v.sample(5000, random_state=cfg.SEED)).pvalue,
                "n_outliers_iqr": int(((v < q1 - cfg.IQR_FACTOR * iqr) | (v > q3 + cfg.IQR_FACTOR * iqr)).sum()),
                "n_outliers_z3": int((z.abs() > 3).sum()),
                "min": v.min(), "max": v.max(),
            })
    return pd.DataFrame(rows)


def missing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Co-missingness (course W2 checklist: does missingness co-occur?)."""
    cols = [c for c in cfg.M3 if df[c].isna().any()]
    pat = df[cols].isna().apply(lambda r: " + ".join(c for c in cols if r[c]) or "complete", axis=1)
    out = pat.value_counts().rename_axis("missing_variables").reset_index(name="n")
    out["pct"] = (100 * out["n"] / len(df)).round(2)
    out["chd_mi_prevalence_pct"] = [100 * df.loc[pat == p, "chd_mi"].mean() for p in out.missing_variables]
    return out


def selection_bias(adults: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """SMD of each excluded group vs the analysis sample, on interview variables."""
    vars_ = [("Age", "age", False), ("Male sex", "male", True), ("Income-to-poverty ratio", "income_pir", False),
             ("Education (1-5)", "education", False), ("Diabetes", "diabetes", True),
             ("Told hypertension", "told_htn", True), ("Lipid-lowering medication", "lipid_med", True),
             ("CHD/MI (outcome)", "chd_mi", True)]
    for code, lab in RACE.items():
        adults[f"race_{code}"] = (adults.race_eth == code).astype(float)
        sample[f"race_{code}"] = (sample.race_eth == code).astype(float)
        vars_.append((lab, f"race_{code}", True))
    for code, lab in SMOKE.items():
        adults[f"smk_{code}"] = (adults.smoking == code).astype(float).where(adults.smoking.notna())
        sample[f"smk_{code}"] = (sample.smoking == code).astype(float).where(sample.smoking.notna())
        vars_.append((f"Smoking: {lab}", f"smk_{code}", True))
    vars_ += [("NLR (log)", "log_nlr", False), ("hs-CRP (log)", "log_crp", False)]
    ref = adults[adults.group == "Analysis sample"]
    comps = {"No examination": adults[adults.group == "Excluded: no examination"],
             "No blood markers": adults[adults.group == "Excluded: no blood markers"],
             "Incomplete covariates (kept!)": sample[~sample.complete_case]}
    ref_cc = sample[sample.complete_case]
    rows = []
    for label, var, binary in vars_:
        row = {"variable": label}
        for name, g in comps.items():
            base = ref_cc if name.startswith("Incomplete") else ref
            row[name] = smd(g[var], base[var], binary=binary) if g[var].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def income_missingness_model(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["income_missing"] = d.income_pir.isna().astype(int)
    d["age10"] = d.age / 10
    f = ("income_missing ~ age10 + male + C(race_eth, Treatment('nh_white')) + education"
         " + C(smoking, Treatment('never')) + diabetes + hypertension + chd_mi + log_nlr + log_crp")
    m = smf.logit(f, data=d).fit(disp=0)
    ci = np.exp(m.conf_int())
    out = pd.DataFrame({"OR": np.exp(m.params), "lo": ci[0], "hi": ci[1], "p": m.pvalues}).drop("Intercept")
    nice = {"age10": "Age (per 10 y)", "male": "Male sex", "education": "Education (per level)",
            "diabetes": "Diabetes", "hypertension": "Hypertension", "chd_mi": "CHD/MI (outcome)",
            "log_nlr": "NLR (log)", "log_crp": "hs-CRP (log)"}
    lab = []
    for k in out.index:
        if "race_eth" in k:
            lab.append(RACE[k.split("T.")[1].rstrip("]")] + " (vs White)")
        elif "smoking" in k:
            lab.append("Smoking: " + SMOKE[k.split("T.")[1].rstrip("]")] + " (vs never)")
        else:
            lab.append(nice.get(k, k))
    out.index = lab
    out.attrs["n"] = int(m.nobs)
    return out


# =============================================================================
# Figures
# =============================================================================
def fig_flowchart(flow: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    main = flow.iloc[:5]
    ys = np.linspace(9.0, 1.6, len(main))
    for i, (y, r) in enumerate(zip(ys, main.itertuples())):
        last = i == len(main) - 1
        box = FancyBboxPatch((0.4, y - 0.42), 5.0, 0.84, boxstyle="round,pad=0.02,rounding_size=0.12",
                             fc=BLUE_RAMP[0] if last else SURFACE, ec=BLUE if last else AXIS, lw=1.2)
        ax.add_patch(box)
        ax.text(2.9, y + 0.12, r.step.replace(" = analysis sample", ""), ha="center", va="center",
                fontsize=8.8, color=INK, weight="bold" if last else "normal")
        ax.text(2.9, y - 0.2, f"n = {r.n:,}   ·   CHD/MI cases = {r.cases:,} ({r.prevalence_pct:.1f}%)",
                ha="center", va="center", fontsize=8.2, color=INK2)
        if i > 0:
            ax.annotate("", xy=(2.9, y + 0.44), xytext=(2.9, ys[i - 1] - 0.44),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
            ex = FancyBboxPatch((6.0, (y + ys[i - 1]) / 2 - 0.27), 3.7, 0.54,
                                boxstyle="round,pad=0.02,rounding_size=0.1", fc=SURFACE, ec=GRID, lw=1)
            ax.add_patch(ex)
            ax.text(7.85, (y + ys[i - 1]) / 2, f"excluded: {r.excluded:,}", ha="center",
                    va="center", fontsize=8.2, color=INK2)
            ax.plot([2.9, 6.0], [(y + ys[i - 1]) / 2] * 2, color=GRID, lw=1, zorder=0)
    cc = flow.iloc[5]
    ax.text(2.9, 0.65, f"Old draft additionally dropped incomplete covariates -> n = {cc.n:,}, "
                       f"cases = {cc.cases} (-{main.iloc[-1].cases - cc.cases} cases).\n"
                       "New draft keeps them and handles gaps by imputation inside the model pipeline.",
            ha="center", va="center", fontsize=7.8, color=INK2)
    ax.set_title("Who is in the analysis? (STROBE flow)", loc="left")
    save_fig(fig, "fig01_flowchart")


def fig_markers(df):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    groups = [(df[df.chd_mi == 0], "No CHD/MI", BLUE), (df[df.chd_mi == 1], "CHD/MI", ORANGE)]
    specs = [("nlr", "NLR (raw scale)", np.linspace(0, 10, 50)),
             ("log_nlr", "log NLR", np.linspace(-1.2, 2.6, 50)),
             ("crp", "hs-CRP, mg/L (raw scale)", np.linspace(0, 40, 50)),
             ("log_crp", "log hs-CRP", np.linspace(-2.4, 5.0, 50))]
    for ax, (var, title, bins) in zip(axes.ravel(), specs):
        for g, lab, col in groups:
            ax.hist(g[var].clip(bins[0], bins[-1]), bins=bins, density=True, histtype="stepfilled",
                    color=col, alpha=0.10)
            ax.hist(g[var].clip(bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                    color=col, lw=1.6, label=f"{lab} (median {np.exp(g[var].median()) if var.startswith('log') else g[var].median():.2f})")
        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
    fig.suptitle("Inflammatory markers are right-skewed; the log scale makes them near-symmetric",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, "fig02_marker_distributions")


def fig_statin(df):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    x = np.array([0, 1])
    for ax, var, title, ylab in [(axes[0], "total_chol", "Total cholesterol", "mg/dL"),
                                 (axes[1], "crp", "hs-CRP", "mg/L (log scale)")]:
        for k, (med, lab, col) in enumerate([(0, "No lipid-lowering medication", BLUE),
                                             (1, "On lipid-lowering medication", ORANGE)]):
            vals = [df[(df.chd_mi == o) & (df.lipid_med == med)][var] for o in (0, 1)]
            mid = [v.median() for v in vals]
            lo = [v.quantile(.25) for v in vals]
            hi = [v.quantile(.75) for v in vals]
            xx = x + (k - 0.5) * 0.18
            ax.vlines(xx, lo, hi, color=col, lw=2)
            ax.plot(xx, mid, "o", ms=8, color=col, mec=SURFACE, mew=2, label=lab)
        ax.set_xticks(x, ["No CHD/MI", "CHD/MI"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title)
        ax.set_ylabel(f"Median and IQR, {ylab}")
        ax.grid(axis="x", visible=False)
        if var == "crp":
            ax.set_yscale("log")
            ax.set_yticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
            ax.minorticks_off()
    axes[0].legend(loc="lower left", fontsize=7.8)
    share = df.groupby("chd_mi").lipid_med.mean() * 100
    ax = axes[2]
    ax.bar([0, 1], share.values, width=0.25, color=BLUE)
    for i, v in enumerate(share.values):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks([0, 1], ["No CHD/MI", "CHD/MI"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% on lipid-lowering medication")
    ax.set_title("Treatment after diagnosis")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Reverse causation: diagnosed patients are treated, which lowers cholesterol and CRP",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, "fig03_statin_paradox")


def fig_correlation(df):
    cols = ["age", "male", "income_pir", "bmi", "diabetes", "hypertension", "total_chol",
            "hdl_chol", "egfr", "log_nlr", "log_crp", "lipid_med"]
    rho = df[cols].corr(method="spearman")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, DIVERGING_MID, RED])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(rho.values, cmap=cmap, vmin=-1, vmax=1)
    labels = [cfg.LABELS.get(c, c) for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(rho.values[i, j]) >= 0.3:
                ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(rho.values[i, j]) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman correlation", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Correlations between predictors (|rho| >= 0.3 labelled)")
    save_fig(fig, "fig04_correlations")
    return rho


def fig_missingness(miss, mod):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [0.8, 1.2]})
    ax = axes[0]
    m = miss.loc[[c for c in cfg.M3 if miss.loc[c, "pct_missing"] > 0]].sort_values("pct_missing")
    ax.barh(range(len(m)), m.pct_missing, height=0.55, color=BLUE)
    for i, v in enumerate(m.pct_missing):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(m)), [cfg.LABELS.get(c, c) for c in m.index])
    ax.set_xlabel("% missing in the analysis sample")
    ax.set_title("Where are the gaps?")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    yy = np.arange(len(mod))[::-1]
    sig = (mod.lo > 1) | (mod.hi < 1)
    col = np.where(sig, BLUE, DEEMPH)
    ax.hlines(yy, mod.lo, mod.hi, color=col, lw=2)
    ax.scatter(mod.OR, yy, s=42, c=col, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.axvline(1, color=AXIS, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.7, 1, 1.4, 2], ["0.5", "0.7", "1", "1.4", "2"])
    ax.minorticks_off()
    ax.set_yticks(yy, mod.index)
    ax.set_xlabel("Odds ratio for 'income missing' (95% CI), log scale")
    ax.set_title("Who does not report income? (blue = CI excludes 1)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, "fig05_missingness")


def fig_selection(sb):
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    yy = np.arange(len(sb))[::-1]
    ax.axvspan(-0.1, 0.1, color=GRID, alpha=0.45, lw=0)
    ax.axvline(0, color=AXIS, lw=1)
    for (col, name), off in zip([(BLUE, "No examination"), (ORANGE, "No blood markers"),
                                 (AQUA, "Incomplete covariates (kept!)")], (-0.22, 0, 0.22)):
        ax.scatter(sb[name], yy + off, s=36, color=col, edgecolors=SURFACE, linewidths=1.2,
                   label=name, zorder=3)
    ax.set_yticks(yy, sb.variable)
    ax.set_xlabel("Standardised mean difference vs. analysis sample (shaded: |SMD| < 0.1)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Group compared",
              title_fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_title("Who is missing from the analysis, and does it matter?")
    save_fig(fig, "fig06_selection_bias")


def fig_qq(df, shape):
    """Q-Q plots raw vs log (course W3: judge normality visually, not by p-values at n > 5,000)."""
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0))
    for row, (raw, lg, lab) in enumerate([("nlr", "log_nlr", "NLR"), ("crp", "log_crp", "hs-CRP")]):
        for col, (var, scale) in enumerate([(raw, "raw"), (lg, "log")]):
            ax = axes[row, col]
            v = df[var].dropna().to_numpy()
            (osm, osr), (slope, icpt, _) = stats.probplot(v, dist="norm")
            ax.plot(osm, slope * osm + icpt, color=MUTED, lw=1)
            ax.plot(osm[::7], osr[::7], "o", ms=3.5, color=BLUE, alpha=0.6, mew=0)
            sk = shape[(shape.marker == lab) & (shape.scale == scale)].skewness.iloc[0]
            ax.set_title(f"{lab}, {scale} scale (skewness {sk:.2f})")
            ax.set_xlabel("Theoretical normal quantiles")
            ax.set_ylabel("Observed quantiles")
    fig.suptitle("Log transformation removes most of the right skew (Q-Q plots)",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, "fig07_qq_plots")


def main():
    df = load_analysis_sample()
    adults = pd.read_parquet(cfg.DATA / "adults_with_outcome.parquet")
    flow = pd.read_csv(cfg.TABLES / "flowchart.csv")
    miss = pd.read_csv(cfg.TABLES / "missingness.csv", index_col=0)

    t1 = table1(df)
    t1.to_csv(cfg.TABLES / "table1.csv", index=False)
    log("Table 1:\n" + t1.to_string(index=False))

    sb = selection_bias(adults, df.copy())
    sb.to_csv(cfg.TABLES / "selection_bias_smd.csv", index=False)
    log("Selection bias (SMD vs analysis sample):\n" + sb.round(3).to_string(index=False))

    mod = income_missingness_model(df)
    mod.to_csv(cfg.TABLES / "income_missingness_model.csv")
    log(f"Income-missingness model (n = {mod.attrs['n']:,}):\n" + mod.round(3).to_string())
    inc = df.assign(m=df.income_pir.isna()).groupby("m").chd_mi.agg(["size", "mean"])
    log("CHD/MI prevalence by income missingness:\n" + inc.to_string())

    prev = df.chd_mi.mean()
    # Sample size (course W3 checklist: "was the sample size justified?"): events per
    # parameter (EPV). Rule of thumb: at least 10 cases per estimated coefficient.
    # M3 coefficients: age, male, 5 race dummies, income, income flag, BMI, 2 smoking
    # dummies, diabetes, hypertension, total and HDL cholesterol, eGFR, NLR, CRP = 19
    n_par_m3 = 1 + 1 + 5 + 1 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1
    cases = int(df.chd_mi.sum())
    rows = []
    for label, n_cases in [("Full analysis sample", cases),
                           ("Outer training fold (4/5)", int(cases * 0.8)),
                           ("Inner training fold (4/5 of 4/5)", int(cases * 0.64)),
                           ("Old draft: 80% of the complete cases",
                            int(df[df.complete_case].chd_mi.sum() * 0.8))]:
        rows.append({"setting": label, "cases": n_cases, "parameters_M3": n_par_m3,
                     "events_per_parameter": n_cases / n_par_m3})
    epv = pd.DataFrame(rows)
    epv.to_csv(cfg.TABLES / "sample_size_epv.csv", index=False)
    log("Sample size (events per parameter; rule of thumb >= 10):\n"
        + epv.round(1).to_string(index=False))

    fig_flowchart(flow)
    fig_markers(df)
    fig_statin(df)
    rho = fig_correlation(df)
    rho.round(3).to_csv(cfg.TABLES / "spearman_correlations.csv")
    fig_missingness(miss, mod)
    fig_selection(sb)

    shape = shape_and_outliers(df)
    shape.to_csv(cfg.TABLES / "marker_shape_outliers.csv", index=False)
    log("Shape and outliers (course W2/W3):\n" + shape.round(3).to_string(index=False))
    fig_qq(df, shape)
    pats = missing_patterns(df)
    pats.to_csv(cfg.TABLES / "missing_patterns.csv", index=False)
    log("Missingness patterns:\n" + pats.head(10).round(2).to_string(index=False))

    # key numbers for the text
    key = {
        "n": len(df), "cases": int(df.chd_mi.sum()), "prevalence": prev,
        "n_complete_case": int(df.complete_case.sum()),
        "cases_complete_case": int(df[df.complete_case].chd_mi.sum()),
        "lipid_med_cases": df[df.chd_mi == 1].lipid_med.mean(),
        "lipid_med_controls": df[df.chd_mi == 0].lipid_med.mean(),
        "rho_nlr_crp": float(df[["log_nlr", "log_crp"]].corr(method="spearman").iloc[0, 1]),
    }
    pd.Series(key).to_csv(cfg.TABLES / "key_numbers_descriptive.csv", header=["value"])
    log("Key numbers:", key)


if __name__ == "__main__":
    main()
