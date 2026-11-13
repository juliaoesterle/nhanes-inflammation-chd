"""
Shared helpers for the NewMarco pipeline:

  * logging to console + results/logs/<script>.log
  * preprocessing pipelines (imputation inside the pipeline -> no leakage)
  * the model zoo with its hyperparameter grids (D10, D11)
  * metrics (discrimination, calibration, overall accuracy)
  * statistical tests: paired t-test over CV folds with the correction for
    overlapping training sets (Nadeau & Bengio 2003), Bonferroni correction,
    survey-weighted logistic regression (sensitivity analysis)
  * SMOTE (course W9), written out step by step
  * the figure style (validated categorical palette, recessive chrome)
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config as C


# =============================================================================
# Logging
# =============================================================================
class Logger:
    """print() that also appends to results/logs/<name>.log with a time stamp."""

    def __init__(self, name: str):
        self.path = C.LOGS / f"{name}.log"
        self.path.write_text("", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, *msg) -> None:
        line = " ".join(str(m) for m in msg)
        out = f"[{time.time() - self.t0:7.1f}s] {line}"
        print(out, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(out + "\n")


def write_json(obj, path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_analysis_sample() -> pd.DataFrame:
    return pd.read_parquet(C.DATA / "analysis_sample.parquet")


# =============================================================================
# Preprocessing (fitted inside every training fold only)
# =============================================================================
def split_types(features):
    cat = [f for f in features if f in C.CATEGORICAL]
    binary = [f for f in features if f in C.BINARY]
    cont = [f for f in features if f not in cat and f not in binary]
    ind = [f for f in features if f in C.INDICATOR_VARS]
    return cont, binary, cat, ind


class MissingFlag(BaseEstimator, TransformerMixin):
    """0/1 float indicator 'x_missing' for each input column (pandas in, pandas out)."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(
            {f"{c}_missing": X[c].isna().astype(float).to_numpy() for c in self.columns_},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_missing" for c in self.columns_], dtype=object)


def make_preprocessor(features, imputation="median", drop_first=False, indicators=True):
    """Two pandas-out steps: (1) impute + missing flags, (2) scale + one-hot.

    imputation 'median' (course W2): median for continuous variables, most frequent
    value for binary and categorical ones; for the variables in C.INDICATOR_VARS a
    0/1 column "<name>_missing" is added ("missing data can hold information").
    Both steps are fitted on the training data only (inside the Pipeline).
    """
    cont, binary, cat, ind = split_types(features)
    if not indicators:
        ind = []
    steps = []
    if imputation == "median":
        steps += [("cont", SimpleImputer(strategy="median"), cont),
                  ("bin", SimpleImputer(strategy="most_frequent"), binary)]
    else:
        raise ValueError(imputation)
    if cat:
        steps.append(("cat", SimpleImputer(strategy="most_frequent"), cat))
    if ind:
        steps.append(("ind", MissingFlag(), ind))
    steps = [s for s in steps if len(s[2])]
    impute = ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)
    impute.set_output(transform="pandas")

    enc = []
    if cont:
        enc.append(("scale", StandardScaler(), cont))
    if cat:
        enc.append(("onehot", OneHotEncoder(drop="first" if drop_first else None,
                                            handle_unknown="ignore", sparse_output=False), cat))
    encode = ColumnTransformer(enc, remainder="passthrough", verbose_feature_names_out=False)
    encode.set_output(transform="pandas")
    return [("impute", impute), ("encode", encode)]


# =============================================================================
# Model zoo and hyperparameter grids (D11)
# =============================================================================
def make_model(algo: str, class_weight=None):
    """Return (estimator, grid). Grids are deliberately small and centred on
    values that are sensible for n ~ 4,400 training rows and ~7 % prevalence."""
    if algo == "LR":      # unpenalised = the classical clinical risk-model form
        return LogisticRegression(C=np.inf, max_iter=10_000, class_weight=class_weight), {}
    if algo == "L1":      # lasso, used for the feature-selection experiment (E2)
        return (LogisticRegression(l1_ratio=1.0, solver="liblinear", C=0.1, max_iter=5_000,
                                   class_weight=class_weight, random_state=C.SEED),
                {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]})
    if algo == "EN":
        return (LogisticRegression(solver="saga", l1_ratio=0.5, C=1.0, max_iter=20_000,
                                   class_weight=class_weight, random_state=C.SEED),
                {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                 "clf__l1_ratio": [0.0, 0.5, 1.0]})
    if algo == "DT":      # single CART tree (course W7-W8); depth/leaf size limit overfitting
        return (DecisionTreeClassifier(criterion="gini", random_state=C.SEED,
                                       class_weight=class_weight),
                {"clf__max_depth": [3, 4, 5, 6, 8],
                 "clf__min_samples_leaf": [20, 50, 100]})
    if algo == "kNN":
        return (KNeighborsClassifier(),
                {"clf__n_neighbors": [50, 100, 200, 400, 800],
                 "clf__weights": ["uniform", "distance"],
                 "clf__p": [1, 2]})
    if algo == "SVM":     # probabilities via Platt scaling on out-of-fold decision values
        cw = C.SVM_CLASS_WEIGHT if class_weight is None else class_weight   # see config
        return (CalibratedClassifierCV(SVC(kernel="rbf", class_weight=cw),
                                       method="sigmoid", cv=3, ensemble=False),
                {"clf__estimator__C": [0.03, 0.1, 0.3, 1.0],
                 "clf__estimator__gamma": [0.002, 0.005, 0.02]})
    if algo == "RF":
        return (RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=C.SEED,
                                       class_weight=class_weight),
                {"clf__max_features": ["sqrt", 0.33],
                 "clf__min_samples_leaf": [10, 30]})
    raise ValueError(algo)


def make_pipeline(features, algo, imputation="median", class_weight=None, indicators=True):
    clf, grid = make_model(algo, class_weight)
    pre = make_preprocessor(features, imputation=imputation,
                            drop_first=(algo == "LR"), indicators=indicators)
    return Pipeline(pre + [("clf", clf)]), grid


def fit_tuned(pipe, grid, X, y, seed, scoring=None):
    """Inner loop of nested CV: grid search on the training part only."""
    if not grid:
        return pipe.fit(X, y), {}
    inner = StratifiedKFold(C.INNER_SPLITS, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, scoring=scoring or C.TUNING_SCORING, cv=inner,
                      n_jobs=1, refit=True, error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def outer_folds(strat, n_repeats):
    """Repeated stratified K-fold; returns (repeat, fold, train_idx, test_idx)."""
    rskf = RepeatedStratifiedKFold(n_splits=C.OUTER_SPLITS, n_repeats=n_repeats,
                                   random_state=C.SEED)
    out = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(strat)), strat)):
        out.append((k // C.OUTER_SPLITS, k % C.OUTER_SPLITS, tr, te))
    return out


# =============================================================================
# Metrics
# =============================================================================
def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p):
    """Calibration-in-the-large (intercept with logit(p) as offset) and slope."""
    import statsmodels.api as sm
    y = np.asarray(y, float)
    lp = logit(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(),
                           offset=lp).fit().params[0]
    return float(intercept), float(slope)


def all_metrics(y, p, base_rate):
    """base_rate = training prevalence (reference for the scaled Brier score)."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    brier = brier_score_loss(y, p)
    brier_null = brier_score_loss(y, np.full(len(y), base_rate))
    cal_int, cal_slope = calibration_intercept_slope(y, p)
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier,
        "scaled_brier": 1 - brier / brier_null,
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "cal_intercept": cal_int,
        "cal_slope": cal_slope,
        "mean_pred": float(p.mean()),
        "observed": float(y.mean()),
    }


def youden_threshold(y, p):
    """Threshold maximising sensitivity + specificity - 1 (Youden's J)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def threshold_metrics(y, p, threshold):
    """Confusion-matrix metrics at a fixed threshold (course W9)."""
    y = np.asarray(y, int)
    pred = np.asarray(p, float) >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))

    def ratio(a, b):
        return a / b if b else np.nan
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "precision": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "accuracy": ratio(tp + tn, len(y))}


def resample_training(X, y, how, seed):
    """Random over- or undersampling of the TRAINING fold only (course W9)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if how == "oversample":
        idx = np.concatenate([neg, pos, rng.choice(pos, len(neg) - len(pos), replace=True)])
    elif how == "undersample":
        idx = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
    else:
        raise ValueError(how)
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


# =============================================================================
# Statistical tests
# =============================================================================
def corrected_ttest(values, ratio=C.TEST_TRAIN_RATIO, alpha=0.05):
    """Corrected resampled t-test / CI (Nadeau & Bengio 2003; Bouckaert & Frank 2004).

    values: one number per outer fold (e.g. AUC difference). The naive variance
    of the fold mean, s^2/J, is too small because training sets overlap; the
    correction inflates it to (1/J + n_test/n_train) * s^2.
    """
    d = np.asarray(values, float)
    J = d.size
    mean = d.mean()
    sd = d.std(ddof=1)
    se = np.sqrt(1.0 / J + ratio) * sd
    df = J - 1
    t = mean / se if se > 0 else np.inf * np.sign(mean)
    p = 2 * stats.t.sf(abs(t), df)
    q = stats.t.ppf(1 - alpha / 2, df)
    return {"mean": mean, "se": se, "lo": mean - q * se, "hi": mean + q * se,
            "t": t, "df": df, "p": p, "n_folds": J}



def bonferroni(pvals):
    """Bonferroni correction (course, tutorial 4): multiply every p-value by the
    number of tests m (capped at 1). Equivalent to testing each at alpha / m."""
    p = np.asarray(pvals, float)
    return np.minimum(1.0, p * p.size)


def smote(X, y, k=5, seed=0):
    """SMOTE (Chawla et al. 2002; course W9), written out so that every step is visible.

    For each synthetic case: pick a random minority case A, pick one of its k nearest
    minority neighbours B, and place a new case at a random point on the line A -> B:
        new = A + u * (B - A),  u ~ Uniform(0, 1)
    New cases are added until both classes are equally large. Use it ONLY on the
    (already imputed and scaled) training part, never on test data.
    Simplification: 0/1 columns are interpolated like numbers (SMOTE-NC would treat
    them separately); the classifier simply sees values between 0 and 1.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    minority = X[y == 1]
    n_new = int((y == 0).sum() - (y == 1).sum())
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority)
    neighbours = nn.kneighbors(minority, return_distance=False)[:, 1:]   # column 0 = the case itself
    a = rng.integers(0, len(minority), n_new)
    b = neighbours[a, rng.integers(0, k, n_new)]
    u = rng.random((n_new, 1))
    synthetic = minority[a] + u * (minority[b] - minority[a])
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def svy_logit(y, X, w, strata, psu):
    """Survey-weighted logistic regression with design-based variance.

    Point estimate: weighted (pseudo-) maximum likelihood.
    Variance: Taylor linearisation (Binder 1983) with PSUs nested in strata,
    with-replacement approximation - the estimator used by R's survey::svyglm.
    Returns (beta, cov, df) with df = #PSU - #strata.
    """
    import statsmodels.api as sm
    y = np.asarray(y, float)
    Xm = np.asarray(X, float)
    w = np.asarray(w, float)
    w = w / w.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(y, Xm, family=sm.families.Binomial(), freq_weights=w).fit()
    beta = np.asarray(res.params)
    p = 1 / (1 + np.exp(-Xm @ beta))
    H = (Xm * (w * p * (1 - p))[:, None]).T @ Xm
    Hinv = np.linalg.inv(H)
    U = Xm * (w * (y - p))[:, None]
    scores = pd.DataFrame(U)
    scores["h"] = np.asarray(strata)
    scores["i"] = np.asarray(psu)
    meat = np.zeros((Xm.shape[1], Xm.shape[1]))
    n_psu, n_strata = 0, 0
    for _, g in scores.groupby("h"):
        z = g.groupby("i").sum().drop(columns="h", errors="ignore").to_numpy()
        nh = z.shape[0]
        n_psu += nh
        n_strata += 1
        if nh < 2:
            continue
        zc = z - z.mean(axis=0)
        meat += nh / (nh - 1) * zc.T @ zc
    cov = Hinv @ meat @ Hinv
    return beta, cov, n_psu - n_strata


def smd(a, b, binary=False):
    """Standardised mean difference (Austin 2009). |SMD| > 0.1 = meaningful imbalance."""
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if binary:
        pa, pb = a.mean(), b.mean()
        den = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        return (pa - pb) / den if den > 0 else 0.0
    den = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / den if den > 0 else 0.0


# =============================================================================
# Figure style: validated categorical palette (slots in fixed order),
# hairline solid grid, text in ink colours only.
# =============================================================================
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DEEMPH = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_MID = "#f0efec"


def set_style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "font.size": 9.5, "text.color": INK, "figure.dpi": 100,
    })


def legend_below(ax, ncol=3, y=-0.2, fig=None):
    """Legend below the x-axis so it can never cover data points."""
    target = fig if fig is not None else ax
    handles, labels = ax.get_legend_handles_labels()
    target.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, fontsize=8.5,
                  bbox_to_anchor=(0.5, y) if fig is None else (0.5, 0.0))


def save_fig(fig, name):
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fmt_ci(est, lo, hi, digits=3):
    return f"{est:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"
