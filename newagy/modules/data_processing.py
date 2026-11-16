"""
Data cleaning with the codebook, analysis variables and the study sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NON_RESPONSE_LABELS = ["Refused", "Don't know", "SP refusal", "Participant refusal"]
SAS_ZERO = 1e-70      # SAS stores an exact 0 as 5.4e-79


def parse_value_codes(text):
    """'0 to 79=Range of Values | 80=80 years and over | 7=Refused | .=Missing'
    -> ranges [(0, 79)], allowed codes [80], non-response codes [7]"""
    ranges, codes, nonresponse = [], [], []
    if not isinstance(text, str):
        return ranges, codes, nonresponse
    for part in text.split(" | "):
        code, _, label = part.partition("=")
        code, label = code.strip(), label.strip()
        if " to " in code:
            low, high = code.split(" to ")
            ranges.append((float(low), float(high)))
            continue
        try:
            value = float(code)
        except ValueError:          # "." or text codes such as D=Day
            continue
        if label in NON_RESPONSE_LABELS:
            nonresponse.append(value)
        else:
            codes.append(value)
    return ranges, codes, nonresponse


def clean_raw_data(raw, meta):
    """Keeps only the values that the codebook allows.

    text in numeric columns -> NaN, SAS zero -> 0, refused / don't know -> NaN,
    values outside the documented codes and ranges -> NaN.
    Returns the cleaned data and the number of cells changed per reason.
    """
    meta = meta.set_index("column_name")
    clean = {}
    counts = {"non_numeric": 0, "non_response": 0, "out_of_range": 0}

    for col in raw.columns:
        x = pd.to_numeric(raw[col], errors="coerce")
        counts["non_numeric"] += int((raw[col].notna() & x.isna()).sum())
        x[x.abs() < SAS_ZERO] = 0

        if col in meta.index and col != "SEQN":
            ranges, codes, nonresponse = parse_value_codes(meta.loc[col, "value_codes"])
            is_nr = x.isin(nonresponse)
            counts["non_response"] += int(is_nr.sum())
            x[is_nr] = np.nan
            if ranges or codes:
                ok = x.isna() | x.isin(codes)
                for low, high in ranges:
                    tol = 1e-6 * max(1, abs(high))      # ranges in the codebook are rounded
                    ok = ok | ((x >= low - tol) & (x <= high + tol))
                counts["out_of_range"] += int((~ok).sum())
                x[~ok] = np.nan
        clean[col] = x

    clean = pd.DataFrame(clean)
    clean["SEQN"] = clean["SEQN"].astype("int64")
    return clean, counts


def yes_no(s):
    """NHANES 1 = yes, 2 = no -> 1.0 / 0.0, everything else NaN"""
    return s.map({1.0: 1.0, 2.0: 0.0})


def egfr_ckd_epi_2021(scr, age, male):
    """Race-free CKD-EPI 2021 creatinine equation (Inker et al. 2021), mL/min/1.73 m2."""
    kappa = np.where(male == 1, 0.9, 0.7)
    alpha = np.where(male == 1, -0.302, -0.241)
    ratio = scr / kappa
    egfr = (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200
            * 0.9938 ** age * np.where(male == 1, 1.0, 1.012))
    return pd.Series(egfr, index=scr.index)


def weekly_minutes(freq, unit, minutes):
    per_week = unit.map({"D": 7.0, "W": 1.0, "M": 7.0 / 30.44, "Y": 7.0 / 365.25})
    out = freq * per_week * minutes
    return out.where(freq != 0, 0.0)


ALQ121_PER_WEEK = {0: 0.0, 1: 7.0, 2: 5.5, 3: 3.5, 4: 2.0, 5: 1.0, 6: 2.5 * 12 / 52,
                   7: 12 / 52, 8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52}

RACE_NAMES = {1.0: "mexican_american", 2.0: "other_hispanic", 3.0: "nh_white",
              4.0: "nh_black", 6.0: "nh_asian", 7.0: "other_multi"}


def build_analysis_variables(d, raw):
    """Outcome, markers, risk factors and the extra variables of the hybrid pool."""
    out = pd.DataFrame({"SEQN": d["SEQN"].astype("int64")})

    # outcome: CHD or heart attack; angina only for a sensitivity analysis
    chd, mi, ang = yes_no(d["MCQ160C"]), yes_no(d["MCQ160E"]), yes_no(d["MCQ160D"])
    out["chd"], out["mi"], out["angina"] = chd, mi, ang
    out["chd_mi"] = np.select([(chd == 1) | (mi == 1), (chd == 0) & (mi == 0)], [1.0, 0.0], np.nan)
    out["chd_mi_angina"] = np.select(
        [(chd == 1) | (mi == 1) | (ang == 1), (chd == 0) & (mi == 0) & (ang == 0)], [1.0, 0.0], np.nan)

    # markers
    out["crp"] = d["LBXHSCRP"]
    out["log_crp"] = np.log(d["LBXHSCRP"])
    out["nlr"] = d["LBXNEPCT"] / d["LBXLYPCT"].where(d["LBXLYPCT"] > 0)
    out["log_nlr"] = np.log(out["nlr"])

    # demographics
    out["age"] = d["RIDAGEYR"]                    # top-coded at 80
    out["male"] = d["RIAGENDR"].map({1.0: 1.0, 2.0: 0.0})
    out["race_eth"] = d["RIDRETH3"].map(RACE_NAMES)
    out["income_pir"] = d["INDFMPIR"]             # top-coded at 5
    out["education"] = d["DMDEDUC2"]

    # established risk factors (M2)
    out["bmi"] = d["BMXBMI"]
    smq020, smq040 = d["SMQ020"], d["SMQ040"]
    out["smoking"] = np.select(
        [smq020 == 2, (smq020 == 1) & (smq040 == 3), (smq020 == 1) & smq040.isin([1, 2])],
        ["never", "former", "current"], default=None)
    out["smoking"] = out["smoking"].where(out["smoking"].notna(), None)
    out["diabetes"] = np.where(d["DIQ010"] == 1, 1.0, np.where(d["DIQ010"].isin([2.0, 3.0]), 0.0, np.nan))
    out["sbp"] = d[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    out["dbp"] = d[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    told_htn = yes_no(d["BPQ020"])
    out["told_htn"] = told_htn
    measured = ((out["sbp"] >= 140) | (out["dbp"] >= 90)).astype(float)
    measured = measured.where(out["sbp"].notna() | out["dbp"].notna())
    out["hypertension"] = np.select([(told_htn == 1) | (measured == 1),
                                     (told_htn == 0) & (measured == 0)], [1.0, 0.0], np.nan)
    out["total_chol"] = d["LBXTC"]
    out["hdl_chol"] = d["LBDHDD"]
    out["egfr"] = egfr_ckd_epi_2021(d["LBXSCR"], out["age"], out["male"])

    # treatment: only for adjustment / sensitivity, never an ML predictor
    out["lipid_med"] = yes_no(d["BPQ101D"])

    # extra variables of the hybrid pool (S3)
    out["waist"] = d["BMXWAIST"]
    out["pulse"] = d[["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"]].mean(axis=1)
    out["told_high_chol"] = yes_no(d["BPQ080"])
    out["log_tg"] = np.log(d["LBXSTR"])
    out["glucose"] = d["LBXSGL"]
    out["uric_acid"] = d["LBXSUA"]
    out["albumin"] = d["LBXSAL"]
    out["log_ggt"] = np.log(d["LBXSGTSI"])
    out["log_alt"] = np.log(d["LBXSATSI"])
    out["log_uacr"] = np.log1p(d["URDACT"])
    out["log_cotinine"] = np.log(d["LBXCOT"])
    out["wbc"] = d["LBXWBCSI"].where(d["LBXWBCSI"] < 100)     # 400 = "400 and over"
    out["hemoglobin"] = d["LBXHGB"]
    out["platelets"] = d["LBXPLTSI"]
    out["rdw"] = d["LBXRDW"]
    out["monocyte_pct"] = d["LBXMOPCT"]
    out["eosinophil_pct"] = d["LBXEOPCT"]
    out["vitamin_d"] = d["LBXVIDMS"]
    out["sleep_hours"] = d["SLD012"]
    out["sedentary_min"] = d["PAD680"]
    # the unit columns (D/W/M/Y) are text and therefore NaN after cleaning -> take them from raw
    units = raw[["PAD790U", "PAD810U"]].where(raw[["PAD790U", "PAD810U"]].isin(["D", "W", "M", "Y"]))
    moderate = weekly_minutes(d["PAD790Q"], units["PAD790U"], d["PAD800"])
    vigorous = weekly_minutes(d["PAD810Q"], units["PAD810U"], d["PAD820"])
    out["log_activity"] = np.log1p(moderate + 2 * vigorous)
    freq = d["ALQ121"].map(ALQ121_PER_WEEK)
    drinks = (freq * d["ALQ130"]).where(~(freq == 0), 0.0).where(~(d["ALQ111"] == 2), 0.0)
    out["log_drinks_wk"] = np.log1p(drinks)
    dpq = d[[f"DPQ0{i}0" for i in range(1, 10)]]
    out["phq9"] = dpq.sum(axis=1).where(dpq.notna().all(axis=1))

    # survey design
    out["exam_status"] = d["RIDSTATR"]        # 2 = interviewed and examined
    out["weight_mec"] = d["WTMEC2YR"]
    out["strata"] = d["SDMVSTRA"]
    out["psu"] = d["SDMVPSU"]
    return out


def select_cohort(df, m2_features):
    """Inclusion criteria one after the other; returns sample, adults with outcome, flowchart."""
    steps = [
        ("All participants", pd.Series(True, index=df.index)),
        ("Age >= 20 years", df["age"] >= 20),
        ("Answered the CHD/MI questions", df["chd_mi"].notna()),
        ("Examined in the MEC", df["exam_status"] == 2),
        ("NLR and hs-CRP measured (analysis sample)", df["nlr"].notna() & df["crp"].notna()),
    ]
    keep = pd.Series(True, index=df.index)
    flow = []
    for label, condition in steps:
        keep &= condition
        flow.append({"step": label, "n": int(keep.sum()), "cases": int((df.loc[keep, "chd_mi"] == 1).sum())})
    flow = pd.DataFrame(flow)
    flow["excluded"] = (-flow["n"].diff()).fillna(0).astype(int)
    flow["prevalence_pct"] = (100 * flow["cases"] / flow["n"]).round(2)

    sample = df[keep].copy()
    sample["complete_case"] = sample[m2_features].notna().all(axis=1)

    adults = df[(df["age"] >= 20) & df["chd_mi"].notna()].copy()
    adults["group"] = np.select(
        [adults["SEQN"].isin(sample["SEQN"]), adults["exam_status"] != 2],
        ["Analysis sample", "Excluded: no examination"], default="Excluded: no blood markers")
    return sample, adults, flow
