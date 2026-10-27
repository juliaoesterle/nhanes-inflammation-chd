"""
Shared data preparation for the NHANES 2021-2023 group project
(Foundations of Data Science, ETH Zurich).

    Research question: are NLR and hs-CRP associated with a self-reported
    history of coronary heart disease / myocardial infarction (CHD/MI),
    independent of established cardiovascular risk factors?

EVERYONE loads the output of this script -- nobody builds their own cohort.
If you want a different choice (e.g. include angina in the outcome), change
it in CONFIG, re-run, and log the decision in the decision log (B6).

Pipeline
--------
  1. clean_raw()                  generic cleaning of all 763 columns
  2. build_analysis_variables()   outcome, exposures, covariates (row-wise only)
  3. select_cohort()              exclusion steps -> flowchart (Figure 1)
  4. add_split()                  fixed stratified train/test split

Outputs (data/processed/ and reports/)
  nhanes_clean.parquet       all participants, all columns, cleaned
  eligible_sample.parquet    adults with outcome + markers, incl. missing
                             covariates (-> imputation, selection-bias check)
  analytic_sample.parquet    complete-case subset of the eligible sample
  flowchart.csv              n after every exclusion step
  cleaning_log.csv           per column: how many cells were set to NaN and why
  missingness.csv            % missing per analysis variable in eligible sample

Usage
  python prepare_data.py                 # full pipeline
  from prepare_data import load_data     # in notebooks / scripts
  train, test = load_data(split=True)

What this script deliberately does NOT do
  No scaling, imputation, encoding or outlier removal fitted on the data.
  Those go inside a scikit-learn Pipeline fitted on the training set only
  (otherwise: test-set leakage).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================================
# CONFIG -- every analysis decision lives here (and in the decision log)
# =====================================================================
ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw" / "nhanes_raw.csv"
METADATA = ROOT / "data" / "raw" / "nhanes_metadata.csv"
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

CONFIG = {
    "min_age": 20,                # MCQ160 heart questions are only asked >= 20 y
    "include_angina": False,      # add MCQ160D to the composite outcome?
    "diabetes_borderline_as_yes": False,   # DIQ010 == 3
    "hypertension_sbp": 140,      # measured SBP threshold (130 = ACC/AHA 2017)
    "hypertension_dbp": 90,       # measured DBP threshold (80 = ACC/AHA 2017)
    "exclude_crp_above_10": False,  # acute inflammation; keep for main analysis,
                                    # flag column `crp_above_10` for sensitivity
    "test_size": 0.20,
    "random_state": 42,
}

# Covariates that must be non-missing for the complete-case analytic sample.
# (= feature set M2 from the brainstorming section B4)
REQUIRED_COVARIATES = [
    "age", "male", "race_eth", "bmi", "income_pir",
    "smoking", "diabetes", "hypertension",
    "total_chol", "hdl_chol", "egfr",
    "lipid_med",
]

# Value labels in the metadata that mean "no valid answer" -> NaN
NON_RESPONSE = re.compile(
    r"^(refused|don.?t know|don\W*t know|missing|sp refusal|participant refusal)$",
    re.IGNORECASE,
)

# SAS transport (.xpt) files store exact 0 as 5.397605346934028e-79
SAS_ZERO_EPS = 1e-70


# =====================================================================
# 1. Generic cleaning of the raw file
# =====================================================================
def parse_value_codes(spec: str | float) -> tuple[list[tuple[float, float]], dict[float, str]]:
    """Parse a metadata `value_codes` string.

    '0 to 79=Range of Values | 80=80 years and over | .=Missing'
      -> ranges [(0, 79)], codes {80.0: '80 years and over'}
    """
    ranges, codes = [], {}
    if not isinstance(spec, str):
        return ranges, codes
    for part in spec.split(" | "):
        key, _, label = part.partition("=")
        key, label = key.strip(), label.strip()
        m = re.fullmatch(r"(-?[\d.]+)\s+to\s+(-?[\d.]+)", key)
        if m:
            ranges.append((float(m.group(1)), float(m.group(2))))
            continue
        try:
            codes[float(key)] = label
        except ValueError:
            pass  # '.' = missing
    return ranges, codes


def clean_raw(raw: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every column using the documented value codes in the metadata.

    Per cell, in this order:
      a) non-numeric junk (random strings, timestamps)        -> NaN  'non_numeric'
      b) SAS zero artefact 5.4e-79                             -> 0
      c) documented non-response code (Refused/Don't know)     -> NaN  'non_response'
      d) value neither a documented code nor inside a
         documented range (e.g. 123456789, -1000000)           -> NaN  'out_of_range'

    Returns the cleaned frame and a per-column log.
    """
    meta = meta.set_index("column_name")
    clean = {}
    log = []

    for col in raw.columns:
        s_raw = raw[col]
        s = pd.to_numeric(s_raw, errors="coerce")
        n_non_numeric = int((s_raw.notna() & s.isna()).sum())
        s = s.mask(s.abs() < SAS_ZERO_EPS, 0.0)

        n_non_response = n_out_of_range = 0
        if col in meta.index and col != "SEQN":
            ranges, codes = parse_value_codes(meta.at[col, "value_codes"])
            nr_codes = [k for k, lab in codes.items() if NON_RESPONSE.match(lab)]
            is_nr = s.isin(nr_codes)
            n_non_response = int(is_nr.sum())
            s = s.mask(is_nr)

            if ranges or codes:
                valid = s.isin([k for k in codes if k not in nr_codes])
                for lo, hi in ranges:
                    # small tolerance: ranges in the codebook are rounded
                    tol = 1e-6 * max(1.0, abs(hi))
                    valid |= s.between(lo - tol, hi + tol)
                bad = s.notna() & ~valid
                n_out_of_range = int(bad.sum())
                s = s.mask(bad)

        clean[col] = s
        log.append({
            "column": col,
            "source_file": meta.at[col, "source_file"] if col in meta.index else None,
            "non_numeric": n_non_numeric,
            "non_response": n_non_response,
            "out_of_range": n_out_of_range,
            "n_valid": int(s.notna().sum()),
        })

    clean = pd.DataFrame(clean)
    clean["SEQN"] = clean["SEQN"].astype("int64")
    assert clean["SEQN"].is_unique, "SEQN must identify exactly one participant"
    return clean, pd.DataFrame(log)


# =====================================================================
# 2. Analysis variables (row-wise transformations only -> no leakage)
# =====================================================================
def _yes_no(s: pd.Series) -> pd.Series:
    """NHANES 1=Yes / 2=No -> 1.0 / 0.0, everything else NaN."""
    return s.map({1.0: 1.0, 2.0: 0.0})


def egfr_ckd_epi_2021(scr_mg_dl: pd.Series, age: pd.Series, male: pd.Series) -> pd.Series:
    """Race-free CKD-EPI 2021 creatinine equation (mL/min/1.73 m^2)."""
    kappa = np.where(male == 1, 0.9, 0.7)
    alpha = np.where(male == 1, -0.302, -0.241)
    ratio = scr_mg_dl / kappa
    egfr = (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200
            * 0.9938 ** age * np.where(male == 1, 1.0, 1.012))
    return pd.Series(egfr, index=scr_mg_dl.index)


def build_analysis_variables(df: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """Derive outcome, exposures and covariates with readable names."""
    out = pd.DataFrame({"SEQN": df["SEQN"]})

    # ---- outcome: CHD or MI (optionally angina) --------------------------
    parts = ["MCQ160C", "MCQ160E"] + (["MCQ160D"] if cfg["include_angina"] else [])
    yn = pd.concat([_yes_no(df[c]) for c in parts], axis=1)
    out["chd_mi"] = np.select(
        [(yn == 1).any(axis=1), (yn == 0).all(axis=1)], [1.0, 0.0], default=np.nan
    )
    out["chd"] = _yes_no(df["MCQ160C"])
    out["mi"] = _yes_no(df["MCQ160E"])
    out["angina"] = _yes_no(df["MCQ160D"])

    # ---- exposures --------------------------------------------------------
    out["crp"] = df["LBXHSCRP"]                       # mg/L, <LOD filled by NHANES
    out["crp_below_lod"] = df["LBDHRPLC"]
    out["crp_above_10"] = (df["LBXHSCRP"] > 10).astype(float).where(df["LBXHSCRP"].notna())
    out["log_crp"] = np.log(df["LBXHSCRP"])
    # NLR from percentages (less rounding than LBDNENO / LBDLYMNO)
    lymph = df["LBXLYPCT"].where(df["LBXLYPCT"] > 0)
    out["nlr"] = df["LBXNEPCT"] / lymph
    out["log_nlr"] = np.log(out["nlr"])
    out["wbc"] = df["LBXWBCSI"]

    # ---- demographics -----------------------------------------------------
    out["age"] = df["RIDAGEYR"]                       # top-coded at 80
    out["male"] = df["RIAGENDR"].map({1.0: 1.0, 2.0: 0.0})
    out["race_eth"] = df["RIDRETH3"].map({
        1.0: "mexican_american", 2.0: "other_hispanic", 3.0: "nh_white",
        4.0: "nh_black", 6.0: "nh_asian", 7.0: "other_multi",
    })
    out["income_pir"] = df["INDFMPIR"]                # top-coded at 5
    out["education"] = df["DMDEDUC2"]                 # 1-5, adults only

    # ---- risk factors -----------------------------------------------------
    out["bmi"] = df["BMXBMI"]
    smq020, smq040 = df["SMQ020"], df["SMQ040"]
    out["smoking"] = np.select(
        [smq020 == 2, (smq020 == 1) & (smq040 == 3), (smq020 == 1) & smq040.isin([1, 2])],
        ["never", "former", "current"], default=None,
    )
    diab_yes = [1.0, 3.0] if cfg["diabetes_borderline_as_yes"] else [1.0]
    out["diabetes"] = np.where(df["DIQ010"].isin(diab_yes), 1.0,
                               np.where(df["DIQ010"].isin([1.0, 2.0, 3.0]), 0.0, np.nan))

    out["sbp"] = df[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    out["dbp"] = df[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    told_htn = _yes_no(df["BPQ020"])
    bp_high = ((out["sbp"] >= cfg["hypertension_sbp"]) | (out["dbp"] >= cfg["hypertension_dbp"]))
    bp_high = bp_high.astype(float).where(out["sbp"].notna() | out["dbp"].notna())
    out["hypertension"] = np.select(
        [(told_htn == 1) | (bp_high == 1), (told_htn == 0) & (bp_high == 0)],
        [1.0, 0.0], default=np.nan,
    )

    out["total_chol"] = df["LBXTC"]                   # mg/dL
    out["hdl_chol"] = df["LBDHDD"]                    # mg/dL
    out["creatinine"] = df["LBXSCR"]                  # mg/dL
    out["egfr"] = egfr_ckd_epi_2021(df["LBXSCR"], out["age"], out["male"])

    # ---- treatment (consequence of diagnosis: adjust, never use as ML feature)
    out["lipid_med"] = _yes_no(df["BPQ101D"])
    # BPQ150 is only asked if told high BP -> "not told" means not on BP meds
    out["bp_med"] = np.where(told_htn == 0, 0.0, _yes_no(df["BPQ150"]))

    # ---- survey design ----------------------------------------------------
    # NOTE: in this merged file WTPH2YR comes from AGP_L (females 12-49 only),
    # so it is ~80% missing for our cohort. Use the MEC exam weight instead.
    out["weight_mec"] = df["WTMEC2YR"]
    out["strata"] = df["SDMVSTRA"]
    out["psu"] = df["SDMVPSU"]
    out["exam_status"] = df["RIDSTATR"]               # 2 = interviewed + examined
    return out


# =====================================================================
# 3. Cohort selection with flowchart
# =====================================================================
def select_cohort(av: pd.DataFrame, cfg: dict = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply exclusions step by step; return (eligible, analytic, flowchart)."""
    steps = [
        ("All NHANES 2021-2023 participants", lambda d: d),
        (f"Age >= {cfg['min_age']} years", lambda d: d[d["age"] >= cfg["min_age"]]),
        ("Outcome (CHD/MI) available", lambda d: d[d["chd_mi"].notna()]),
        ("NLR and hs-CRP available", lambda d: d[d["nlr"].notna() & d["crp"].notna()]),
    ]
    if cfg["exclude_crp_above_10"]:
        steps.append(("hs-CRP <= 10 mg/L", lambda d: d[d["crp"] <= 10]))

    flow, d = [], av
    for label, f in steps:
        d = f(d)
        flow.append({"step": label, "n": len(d), "cases": int(d["chd_mi"].eq(1).sum())})
    eligible = d.copy()

    analytic = eligible.dropna(subset=REQUIRED_COVARIATES)
    flow.append({"step": "Complete covariates (analytic sample)", "n": len(analytic),
                 "cases": int(analytic["chd_mi"].eq(1).sum())})
    eligible["complete_case"] = eligible.index.isin(analytic.index)

    flow = pd.DataFrame(flow)
    flow["excluded"] = (-flow["n"].diff()).fillna(0).astype(int)
    flow["prevalence_%"] = (100 * flow["cases"] / flow["n"]).round(2)
    return eligible, analytic, flow


# =====================================================================
# 4. Fixed train/test split
# =====================================================================
def add_split(eligible: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """Stratified (outcome x complete_case) split on the ELIGIBLE sample, so that
    complete-case and imputation analyses share the same test participants."""
    strat = eligible["chd_mi"].astype(int).astype(str) + "_" + eligible["complete_case"].astype(str)
    train_idx, _ = train_test_split(
        eligible.index, test_size=cfg["test_size"],
        stratify=strat, random_state=cfg["random_state"],
    )
    eligible = eligible.copy()
    eligible["split"] = np.where(eligible.index.isin(train_idx), "train", "test")
    return eligible


# =====================================================================
# Loader for everyone's analyses
# =====================================================================
def load_data(sample: str = "analytic", split: bool = False):
    """Load the prepared data.

    sample: 'analytic' (complete case), 'eligible' (with missing covariates)
            or 'clean' (all participants, all 763 cleaned raw columns)
    split:  if True return (train, test); the test set is used ONCE, at the end.
    """
    df = pd.read_parquet(OUT_DIR / f"{ {'clean': 'nhanes_clean'}.get(sample, sample + '_sample') }.parquet")
    if split:
        if "split" not in df:
            raise ValueError("split=True only works for 'analytic' or 'eligible'")
        return df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()
    return df


# =====================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_DATA.name} ...")
    raw = pd.read_csv(RAW_DATA, dtype=str, low_memory=False)
    meta = pd.read_csv(METADATA)

    clean, log = clean_raw(raw, meta)
    log.to_csv(REPORT_DIR / "cleaning_log.csv", index=False)
    clean.to_parquet(OUT_DIR / "nhanes_clean.parquet", index=False)
    print(f"  cleaned {clean.shape[0]} x {clean.shape[1]}: "
          f"{log.non_numeric.sum()} non-numeric, {log.non_response.sum()} non-response, "
          f"{log.out_of_range.sum()} out-of-range cells -> NaN")

    av = build_analysis_variables(clean)
    eligible, _, flow = select_cohort(av)
    eligible = add_split(eligible)
    analytic = eligible[eligible["complete_case"]]

    eligible.to_parquet(OUT_DIR / "eligible_sample.parquet", index=False)
    analytic.to_parquet(OUT_DIR / "analytic_sample.parquet", index=False)
    flow.to_csv(REPORT_DIR / "flowchart.csv", index=False)

    analysis_cols = [c for c in eligible.columns if c not in ("SEQN", "split", "complete_case")]
    miss = eligible[analysis_cols].isna().agg(["sum", "mean"]).T
    miss.columns = ["n_missing", "pct_missing"]
    miss["pct_missing"] = (100 * miss["pct_missing"]).round(1)
    miss.sort_values("n_missing", ascending=False).to_csv(REPORT_DIR / "missingness.csv")

    print("\nFlowchart:\n" + flow.to_string(index=False))
    print("\nSplit (analytic sample):")
    print(analytic.groupby("split")["chd_mi"].agg(n="size", cases="sum").to_string())
    print(f"\nWritten to {OUT_DIR.relative_to(ROOT)}/ and {REPORT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
