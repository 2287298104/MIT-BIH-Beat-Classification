"""End-to-end runner: dataset -> training/evaluation -> figures.

Usage
    python run_all.py            # full pipeline
    python run_all.py --skip-data   # reuse outputs/features_mitdb.csv
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    ("build_dataset.py", "Building beat-level feature table from 48 records"),
    ("train_eval.py", "Training classifiers and evaluating on DS2"),
    ("make_figures.py", "Rendering report figures"),
]


def main() -> None:
    skip = "--skip-data" in sys.argv
    for script, msg in STEPS:
        if skip and script == "build_dataset.py":
            print(f"[skip] {msg}")
            continue
        print(f"\n{'=' * 70}\n>>> {msg}\n{'=' * 70}", flush=True)
        res = subprocess.run([sys.executable, str(SRC / script)], cwd=str(SRC))
        if res.returncode != 0:
            sys.exit(f"step {script} failed with code {res.returncode}")
    print("\nall steps finished. Results in outputs/ and outputs/figures/")


if __name__ == "__main__":
    main()
