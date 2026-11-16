"""
Step 05: evaluation of the 15 outer folds of step 04.
- performance per algorithm and feature set (AUC, PR-AUC, Brier, calibration slope, overfitting gap)
- marker gain dAUC = M3 - M2 per algorithm (Nadeau-Bengio corrected t-test, Bonferroni over 6)
- each algorithm vs logistic regression on M3 (Bonferroni over 5)
- metrics at the Youden threshold (sensitivity, specificity, PPV, NPV)
- final model: LR unless another algorithm is significantly better
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import config as cfg
from evaluation import bonferroni, corrected_ttest


def paired(fm, algo, fs_a, fs_b, algo_b=None):
    a = fm[(fm["algo"] == algo) & (fm["fs"] == fs_a)].set_index(["repeat", "fold"])
    b = fm[(fm["algo"] == (algo_b or algo)) & (fm["fs"] == fs_b)].set_index(["repeat", "fold"])
    return a, b


def main():
    fm = pd.read_csv(cfg.CV_DIR / "fold_metrics.csv")

    perf = (fm.groupby(["algo", "fs"])[["auc", "pr_auc", "brier", "scaled_brier", "cal_slope", "auc_train"]]
            .mean().reset_index())
    perf["auc_sd"] = fm.groupby(["algo", "fs"])["auc"].std().to_numpy()
    perf["overfit_gap"] = perf["auc_train"] - perf["auc"]
    perf.to_csv(cfg.TABLES_DIR / "main_performance.csv", index=False)
    print(perf[perf["fs"].isin(["M2", "M3"])].round(4).to_string(index=False))

    rows = []
    for algo in cfg.ALGORITHMS:
        m3, m2 = paired(fm, algo, "M3", "M2")
        t = corrected_ttest(m3["auc"] - m2["auc"])
        rows.append({"algo": algo, "auc_M2": m2["auc"].mean(), "auc_M3": m3["auc"].mean(),
                     "delta_auc": t["mean"], "lo": t["lo"], "hi": t["hi"],
                     "delta_brier": (m3["brier"] - m2["brier"]).mean(), "p": t["p"]})
    gain = pd.DataFrame(rows)
    gain["p_bonferroni"] = bonferroni(gain["p"])
    gain.to_csv(cfg.TABLES_DIR / "main_marker_gain.csv", index=False)
    print("\nmarker gain M3 - M2:\n" + gain.round(4).to_string(index=False))

    rows = []
    for algo in cfg.ALGORITHMS[1:]:
        a, lr = paired(fm, algo, "M3", "M3", algo_b="LR")
        t = corrected_ttest(a["auc"] - lr["auc"])
        rows.append({"comparison": f"{algo} vs LR (M3)", "delta_auc": t["mean"], "lo": t["lo"],
                     "hi": t["hi"], "p": t["p"]})
    vs_lr = pd.DataFrame(rows)
    vs_lr["p_bonferroni"] = bonferroni(vs_lr["p"])
    vs_lr.to_csv(cfg.TABLES_DIR / "main_algorithm_vs_lr.csv", index=False)
    print("\nalgorithm vs LR:\n" + vs_lr.round(4).to_string(index=False))

    lr = fm[fm["algo"] == "LR"].groupby("fs")["auc"].agg(["mean", "std"])
    print("\nLR by feature set:\n" + lr.reindex(list(cfg.FEATURE_SETS)).round(3).to_string())

    thr_cols = [c for c in fm.columns if c.startswith("thr_")]
    thr = fm[fm["fs"].isin(["M2", "M3"])].groupby(["algo", "fs"])[thr_cols].mean().reset_index()
    thr.columns = [c.replace("thr_", "") for c in thr.columns]
    thr.to_csv(cfg.TABLES_DIR / "main_threshold_metrics.csv", index=False)
    print("\nYouden threshold (from the training part):\n"
          + thr[["algo", "fs", "threshold", "sensitivity", "specificity", "precision", "npv"]].round(3)
          .to_string(index=False))

    better = vs_lr[(vs_lr["delta_auc"] > 0) & (vs_lr["p_bonferroni"] < cfg.ALPHA)]
    final = better.sort_values("delta_auc").iloc[-1]["comparison"].split()[0] if len(better) else "LR"
    with open(cfg.DECISIONS_DIR / "final_model.json", "w") as f:
        json.dump({"rule": "LR unless another algorithm is significantly better (Bonferroni p < 0.05)",
                   "decision": final}, f, indent=2)
    print("\nfinal model:", final)


if __name__ == "__main__":
    main()
