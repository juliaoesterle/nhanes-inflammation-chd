"""
01 - Data preparation: generic cleaning -> analysis variables -> cohort.

Stage 1 re-uses the team's shared, codebook-driven cleaning function
`prepare_data.clean_raw` unchanged (reviewed: it maps junk strings, SAS zero
artefacts, documented non-response codes and undocumented sentinels to NaN or 0
using nhanes_metadata.csv, without per-variable hard-coding).

One gap was found during the review and is fixed here: text-coded columns such
as the physical-activity UNIT columns PAD790U / PAD810U (D/W/M/Y) are
non-numeric and are therefore wiped by the generic cleaner. Without the unit,
"3" could mean 3 times a day, week, month or year, so we read them separately
from the raw file.

Differences to the old draft (decision records in docs/TECHNICAL_DOCUMENTATION.md):
  D3  NO complete-case restriction. People with missing covariates stay in the
      analysis sample; missing values are handled inside the model pipelines
      (median imputation + missing indicator, course W2), both for prediction
      and for the association analysis.
  D5  additional candidate variables for the hybrid feature-selection pool (S3)
  D2  alternative outcome definitions for sensitivity analyses

Outputs
  data/analysis_sample.parquet            n = 5,538 adults with outcome + markers
  data/adults_with_outcome.parquet        all adults with an outcome (selection-bias check)
  data/clean_raw_analysis_sample.parquet  all 763 cleaned raw columns (naive pool S2)
  results/tables/flowchart.csv, missingness.csv, cleaning_summary.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config as C
from common import Logger

sys.path.insert(0, str(C.ROOT))
from prepare_data import clean_raw, egfr_ckd_epi_2021  # noqa: E402  (shared team code)

log = Logger("01_prepare_data")


def yes_no(s: pd.Series) -> pd.Series:
    """NHANES 1 = yes / 2 = no -> 1.0 / 0.0; anything else NaN."""
    return s.map({1.0: 1.0, 2.0: 0.0})


def weekly_minutes(freq, unit, minutes):
    """Leisure-time activity minutes per week from frequency + unit + minutes.
    Frequency 0 means no activity (the follow-up questions are then skipped)."""
    per_week = unit.map({"D": 7.0, "W": 1.0, "M": 7.0 / 30.44, "Y": 7.0 / 365.25})
    out = freq * per_week * minutes
    return out.where(freq != 0, 0.0)


# times per week for the ALQ121 answer categories
ALQ121_PER_WEEK = {0: 0.0, 1: 7.0, 2: 5.5, 3: 3.5, 4: 2.0, 5: 1.0, 6: 2.5 * 12 / 52,
                   7: 12 / 52, 8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52}


def build_variables(d: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"SEQN": d["SEQN"].astype("int64")})

    # ---- outcomes (D2) -------------------------------------------------------
    chd, mi, ang = yes_no(d["MCQ160C"]), yes_no(d["MCQ160E"]), yes_no(d["MCQ160D"])
    out["chd"], out["mi"], out["angina"] = chd, mi, ang
    out["chd_mi"] = np.select([(chd == 1) | (mi == 1), (chd == 0) & (mi == 0)], [1.0, 0.0], np.nan)
    out["chd_mi_angina"] = np.select(
        [(chd == 1) | (mi == 1) | (ang == 1), (chd == 0) & (mi == 0) & (ang == 0)], [1.0, 0.0], np.nan)
    out["mi_strict"] = np.select([mi == 1, (mi == 0) & (chd == 0)], [1.0, 0.0], np.nan)
    out["chd_strict"] = np.select([chd == 1, (chd == 0) & (mi == 0)], [1.0, 0.0], np.nan)

    # ---- inflammatory markers ------------------------------------------------
    out["crp"] = d["LBXHSCRP"]                              # mg/L; < LOD filled by NHANES
    out["log_crp"] = np.log(d["LBXHSCRP"])
    out["crp_above_10"] = (d["LBXHSCRP"] > 10).astype(float).where(d["LBXHSCRP"].notna())
    lymph_pct = d["LBXLYPCT"].where(d["LBXLYPCT"] > 0)
    out["nlr"] = d["LBXNEPCT"] / lymph_pct                  # ratio of percentages = ratio of counts
    out["log_nlr"] = np.log(out["nlr"])
    lymph_n = d["LBDLYMNO"].where(d["LBDLYMNO"] > 0)
    out["log_nlr_counts"] = np.log(d["LBDNENO"] / lymph_n)  # sensitivity: absolute counts

    # ---- demographics / socio-economic --------------------------------------
    out["age"] = d["RIDAGEYR"]                              # top-coded at 80
    out["male"] = d["RIAGENDR"].map({1.0: 1.0, 2.0: 0.0})
    out["race_eth"] = d["RIDRETH3"].map({1.0: "mexican_american", 2.0: "other_hispanic",
                                         3.0: "nh_white", 4.0: "nh_black",
                                         6.0: "nh_asian", 7.0: "other_multi"})
    out["income_pir"] = d["INDFMPIR"]                       # top-coded at 5
    out["education"] = d["DMDEDUC2"]                        # 1 (<9th grade) ... 5 (college+)

    # ---- established risk factors (M2) --------------------------------------
    out["bmi"] = d["BMXBMI"]
    smq020, smq040 = d["SMQ020"], d["SMQ040"]
    out["smoking"] = np.select(
        [smq020 == 2, (smq020 == 1) & (smq040 == 3), (smq020 == 1) & smq040.isin([1, 2])],
        ["never", "former", "current"], default=None)
    out["smoking"] = out["smoking"].where(out["smoking"].notna(), None)
    out["diabetes"] = np.where(d["DIQ010"] == 1, 1.0,
                               np.where(d["DIQ010"].isin([2.0, 3.0]), 0.0, np.nan))
    out["sbp"] = d[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    out["dbp"] = d[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    told_htn = yes_no(d["BPQ020"])
    out["told_htn"] = told_htn               # interview-only part (selection-bias check)
    measured = ((out["sbp"] >= 140) | (out["dbp"] >= 90)).astype(float)
    measured = measured.where(out["sbp"].notna() | out["dbp"].notna())
    out["hypertension"] = np.select([(told_htn == 1) | (measured == 1),
                                     (told_htn == 0) & (measured == 0)], [1.0, 0.0], np.nan)
    out["total_chol"] = d["LBXTC"]
    out["hdl_chol"] = d["LBDHDD"]
    out["egfr"] = egfr_ckd_epi_2021(d["LBXSCR"], out["age"], out["male"])

    # ---- treatment (D4: adjustment / sensitivity only, never an ML predictor)
    out["lipid_med"] = yes_no(d["BPQ101D"])
    out["bp_med"] = np.where(told_htn == 0, 0.0, yes_no(d["BPQ150"]))

    # ---- extended candidate pool for the hybrid strategy S3 (D5) -------------
    out["waist"] = d["BMXWAIST"]
    out["pulse"] = d[["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"]].mean(axis=1)
    out["told_high_chol"] = yes_no(d["BPQ080"])
    out["log_tg"] = np.log(d["LBXSTR"])                     # non-fasting serum triglycerides
    out["glucose"] = d["LBXSGL"]                            # non-fasting serum glucose
    out["uric_acid"] = d["LBXSUA"]
    out["albumin"] = d["LBXSAL"]
    out["log_ggt"] = np.log(d["LBXSGTSI"])
    out["log_alt"] = np.log(d["LBXSATSI"])
    out["log_uacr"] = np.log1p(d["URDACT"])
    out["log_cotinine"] = np.log(d["LBXCOT"])
    out["wbc"] = d["LBXWBCSI"].where(d["LBXWBCSI"] < 100)   # code 400 = '400 and over'
    out["hemoglobin"] = d["LBXHGB"]
    out["platelets"] = d["LBXPLTSI"]
    out["rdw"] = d["LBXRDW"]
    out["monocyte_pct"] = d["LBXMOPCT"]
    out["eosinophil_pct"] = d["LBXEOPCT"]
    out["vitamin_d"] = d["LBXVIDMS"]
    out["sleep_hours"] = d["SLD012"]
    out["sedentary_min"] = d["PAD680"]
    moderate = weekly_minutes(d["PAD790Q"], units["PAD790U"], d["PAD800"])
    vigorous = weekly_minutes(d["PAD810Q"], units["PAD810U"], d["PAD820"])
    out["activity_min_wk"] = moderate + 2 * vigorous        # WHO moderate-equivalent minutes
    out["log_activity"] = np.log1p(out["activity_min_wk"])
    freq = d["ALQ121"].map(ALQ121_PER_WEEK)
    drinks = freq * d["ALQ130"]
    drinks = drinks.where(~(freq == 0), 0.0).where(~(d["ALQ111"] == 2), 0.0)
    out["drinks_wk"] = drinks
    out["log_drinks_wk"] = np.log1p(drinks)
    dpq = d[[f"DPQ0{i}0" for i in range(1, 10)]]
    out["phq9"] = dpq.sum(axis=1).where(dpq.notna().all(axis=1))

    # ---- survey design --------------------------------------------------------
    out["weight_mec"] = d["WTMEC2YR"]
    out["strata"] = d["SDMVSTRA"]
    out["psu"] = d["SDMVPSU"]
    out["exam_status"] = d["RIDSTATR"]                      # 2 = interviewed + examined
    return out


def main() -> None:
    log(f"Reading {C.RAW_DATA.name} and {C.METADATA.name}")
    raw = pd.read_csv(C.RAW_DATA, dtype=str, low_memory=False)
    meta = pd.read_csv(C.METADATA)

    clean, clean_log = clean_raw(raw, meta)
    totals = clean_log[["non_numeric", "non_response", "out_of_range"]].sum()
    log(f"Generic cleaning of {clean.shape[0]:,} x {clean.shape[1]} cells: "
        f"{totals.non_numeric:,} non-numeric, {totals.non_response:,} non-response, "
        f"{totals.out_of_range:,} out-of-range values set to NaN")
    pd.DataFrame({"reason": totals.index, "cells_set_to_nan": totals.values}).to_csv(
        C.TABLES / "cleaning_summary.csv", index=False)

    # text-coded unit columns (gap in the generic cleaner, see module docstring)
    units = raw[["SEQN", "PAD790U", "PAD810U"]].copy()
    units["SEQN"] = pd.to_numeric(units["SEQN"]).astype("int64")
    for c in ("PAD790U", "PAD810U"):
        units[c] = units[c].where(units[c].isin(["D", "W", "M", "Y"]))
    units = units.set_index("SEQN").reindex(clean["SEQN"]).reset_index(drop=True)
    log(f"Recovered activity units: PAD790U {units.PAD790U.notna().sum():,}, "
        f"PAD810U {units.PAD810U.notna().sum():,} (were all NaN after generic cleaning)")

    av = build_variables(clean, units)

    # ---- cohort flow (STROBE) -------------------------------------------------
    steps = [
        ("All NHANES 2021-2023 participants", pd.Series(True, index=av.index)),
        (f"Aged >= {C.MIN_AGE} years", av["age"] >= C.MIN_AGE),
        ("Answered the CHD/MI questions", av["chd_mi"].notna()),
        ("Attended the examination centre (MEC)", av["exam_status"] == 2),
        ("NLR and hs-CRP measured = analysis sample", av["nlr"].notna() & av["crp"].notna()),
    ]
    keep = pd.Series(True, index=av.index)
    flow = []
    for label, cond in steps:
        keep &= cond
        flow.append({"step": label, "n": int(keep.sum()),
                     "cases": int(av.loc[keep, "chd_mi"].eq(1).sum())})
    flow = pd.DataFrame(flow)
    flow["excluded"] = (-flow["n"].diff()).fillna(0).astype(int)
    flow["prevalence_pct"] = (100 * flow["cases"] / flow["n"]).round(2)

    sample = av[keep].copy()
    sample["complete_case"] = sample[C.M2].notna().all(axis=1)
    cc = sample["complete_case"]
    flow.loc[len(flow)] = {"step": "  of which complete on all M2 covariates (old draft's sample)",
                           "n": int(cc.sum()), "cases": int(sample.loc[cc, "chd_mi"].sum()),
                           "excluded": 0,
                           "prevalence_pct": round(100 * sample.loc[cc, "chd_mi"].mean(), 2)}
    flow.to_csv(C.TABLES / "flowchart.csv", index=False)
    log("Flowchart:\n" + flow.to_string(index=False))

    # adults with an outcome -> who was excluded and why (selection-bias check)
    adults = av[(av["age"] >= C.MIN_AGE) & av["chd_mi"].notna()].copy()
    adults["group"] = np.select(
        [adults["SEQN"].isin(sample["SEQN"]), adults["exam_status"] != 2],
        ["Analysis sample", "Excluded: no examination"], default="Excluded: no blood markers")
    log("Adults with outcome by group:\n" + adults["group"].value_counts().to_string())

    # missingness in the analysis sample
    cols = [c for c in C.S3_POOL + C.TREATMENT_VARS if c in sample]
    miss = sample[cols].isna().agg(["sum", "mean"]).T
    miss.columns = ["n_missing", "pct_missing"]
    miss["pct_missing"] = (100 * miss["pct_missing"]).round(1)
    miss.sort_values("n_missing", ascending=False).to_csv(C.TABLES / "missingness.csv")
    log("Missingness of M3 variables (%):\n"
        + miss.loc[C.M3, "pct_missing"].sort_values(ascending=False).to_string())

    raw_clean = clean[clean["SEQN"].isin(sample["SEQN"])].copy()
    sample.to_parquet(C.DATA / "analysis_sample.parquet", index=False)
    adults.to_parquet(C.DATA / "adults_with_outcome.parquet", index=False)
    raw_clean.to_parquet(C.DATA / "clean_raw_analysis_sample.parquet", index=False)
    log(f"Saved analysis sample: n = {len(sample):,}, cases = {int(sample.chd_mi.sum())}, "
        f"complete cases = {int(cc.sum()):,}")


if __name__ == "__main__":
    main()
