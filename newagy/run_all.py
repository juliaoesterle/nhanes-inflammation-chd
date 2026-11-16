"""
Master Execution Pipeline for the newagy project.

Allows running the entire data science pipeline sequentially or executing
specific stages on demand.

Usage:
    python run_all.py           # Runs all steps in sequence
    python run_all.py 01 02     # Runs only steps 01 and 02
    python run_all.py 05 06     # Runs evaluation and association analysis
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE / "scripts"

PIPELINE_STEPS = [
    ("01", SCRIPTS_DIR / "01_prepare_cohort.py", "Cohort Preparation & Data Cleaning"),
    ("02", SCRIPTS_DIR / "02_exploratory_analysis.py", "Exploratory Analysis & Table 1"),
    ("03", SCRIPTS_DIR / "03_design_experiments.py", "Four Pivotal Design Experiments (E1-E4)"),
    ("04", SCRIPTS_DIR / "04_run_nested_cv.py", "Repeated Nested Cross-Validation"),
    ("05", SCRIPTS_DIR / "05_model_evaluations.py", "Model Discrimination & Statistical Tests"),
    ("06", SCRIPTS_DIR / "06_association_analysis.py", "Epidemiological Association & Odds Ratios"),
    ("07", SCRIPTS_DIR / "07_interpretability.py", "Model Interpretability & Feature Importance"),
]


def main():
    target_steps = sys.argv[1:]
    steps_to_run = [
        (num, script, desc)
        for num, script, desc in PIPELINE_STEPS
        if not target_steps or num in target_steps or script.stem in target_steps
    ]

    if not steps_to_run:
        print(f"Error: No matching steps found for arguments {target_steps}")
        print("Available step numbers: 01, 02, 03, 04, 05, 06, 07")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("NEWAGY: ETH FOUNDATIONS OF DATA SCIENCE PIPELINE")
    print("Coronary Heart Disease & Inflammatory Biomarkers (NHANES 2021-2023)")
    print("=" * 75)
    print(f"Executing {len(steps_to_run)} pipeline stage(s)...\n")

    overall_start = time.time()
    for num, script, desc in steps_to_run:
        step_start = time.time()
        print(f">>> Running Step {num}: {desc} ({script.name})...")
        cmd = [sys.executable, str(script)]
        ret = subprocess.run(cmd, cwd=str(HERE))
        if ret.returncode != 0:
            print(f"\n[ERROR] Step {num} ({script.name}) failed with exit code {ret.returncode}.")
            sys.exit(ret.returncode)
        print(f">>> Step {num} completed in {time.time() - step_start:.1f}s.\n")

    print("=" * 75)
    print(f"ALL SELECTED PIPELINE STAGES COMPLETED IN {time.time() - overall_start:.1f}s.")
    print("Results and artifacts are available in 'results/' and 'data/'.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
