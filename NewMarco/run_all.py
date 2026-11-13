"""Run the whole NewMarco pipeline in order (about 1.2 hours on 20 cores, mostly step 04).

    pip install -r requirements.txt  # exact versions (see requirements.txt)
    cd NewMarco
    python run_all.py                # everything
    python run_all.py 05 06          # only selected steps (their inputs must exist)
"""
import subprocess
import sys
import time
from pathlib import Path

STEPS = ["01_prepare_data.py", "02_descriptives.py", "03_design_experiments.py",
         "04_nested_cv.py", "05_evaluate_models.py", "06_association.py", "07_interpretation.py"]

here = Path(__file__).resolve().parent
wanted = sys.argv[1:]
for step in STEPS:
    if wanted and not any(step.startswith(w) for w in wanted):
        continue
    t0 = time.time()
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(here / step)], cwd=here, check=True)
    print(f"=== {step} finished in {(time.time() - t0) / 60:.1f} min ===", flush=True)
