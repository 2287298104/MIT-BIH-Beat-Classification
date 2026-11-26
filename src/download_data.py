"""Download the analysed MIT-BIH records (42 of the 48, ~90 MB) from PhysioNet.

Usage
    python src/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, RECORDS

import wfdb


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in DATA_DIR.glob("*.hea")}
    todo = [r for r in RECORDS if str(r) not in have]
    print(f"{len(have)} records already present, {len(todo)} to download")

    for i, rec in enumerate(todo, 1):
        wfdb.dl_files("mitdb", str(DATA_DIR), [f"{rec}.hea", f"{rec}.dat", f"{rec}.atr"])
        print(f"[{i:2d}/{len(todo)}] record {rec}", flush=True)
    print("done ->", DATA_DIR)


if __name__ == "__main__":
    main()
