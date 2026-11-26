"""Build the beat-level feature table from the 48 MIT-BIH records.

For every record:
    read signal -> choose lead -> Pan-Tompkins R-peak detection
    -> match detections to cardiologist annotations (AAMI super-classes)
    -> 0.5-40 Hz filtering -> fixed 250/450 ms beat windows
    -> 40 hand-crafted features -> append to the global table

Outputs
    outputs/features_mitdb.csv     one row per labelled beat
    outputs/detection_stats.csv    per-record detector sensitivity / PPV
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (AAMI_MAP, CLASSES, DATA_DIR, DS1, FS, MATCH_TOL_S, OUT_DIR,
                    RECORDS)
from features import all_feature_names, extract_all
from preprocessing import (detect_r_peaks, detect_r_peaks_dual_threshold,
                            match_peaks, preprocess_ecg)
from segmentation import extract_beats, label_beats, rr_features

BEAT_SYMBOLS = set(AAMI_MAP.keys()) | {"?"}


def pick_lead(rec: wfdb.Record) -> int:
    names = [n.strip() for n in rec.sig_name]
    for pref in ("MLII", "II", "V5", "V1"):
        if pref in names:
            return names.index(pref)
    return 0


def process_record(rec_id: int) -> tuple[pd.DataFrame, dict]:
    rec = wfdb.rdrecord(str(DATA_DIR / str(rec_id)))
    ann = wfdb.rdann(str(DATA_DIR / str(rec_id)), "atr")

    lead_idx = pick_lead(rec)
    raw = np.asarray(rec.p_signal[:, lead_idx], dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0)

    # ---- beat annotations only (exclude rhythm / artefact annotations) ----
    beat_mask = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    ann_samples = ann.sample[beat_mask]
    ann_symbols = [s for s, m in zip(ann.symbol, beat_mask) if m]

    # ---- R-peak detection ------------------------------------------------
    t0 = time.time()
    peaks = detect_r_peaks(raw, FS)
    det_time = time.time() - t0

    tol = int(round(MATCH_TOL_S * FS))
    tp, fp, fn = match_peaks(peaks, ann_samples, tol)

    # ablation: literal Pan-Tompkins dual-threshold rule (reported in the report)
    peaks_dual = detect_r_peaks_dual_threshold(raw, FS)
    tp_d, _, _ = match_peaks(peaks_dual, ann_samples, tol)

    n_ref = ann_samples.size
    sens = tp.size / n_ref if n_ref else np.nan
    ppv = tp.size / peaks.size if peaks.size else np.nan

    # ---- segmentation on the 0.5-40 Hz filtered signal -------------------
    clean = preprocess_ecg(raw, FS)
    beats, valid = extract_beats(clean, peaks)

    labels, matched = label_beats(peaks, ann_samples, ann_symbols, tol)

    # keep beats that are valid, matched and belong to one of the 5 AAMI classes
    keep = valid & matched & (labels >= 0)
    rr = rr_features(peaks.astype(float), FS)

    if keep.sum() < 10:
        df = pd.DataFrame()
    else:
        feats = extract_all(beats[keep], rr[keep])
        df = pd.DataFrame(feats, columns=all_feature_names())
        df.insert(0, "record", rec_id)
        df.insert(1, "peak_sample", peaks[keep])
        df.insert(2, "label_idx", labels[keep])
        df.insert(3, "label", [CLASSES[i] for i in labels[keep]])
        df.insert(4, "split", "DS1" if rec_id in DS1 else "DS2")

    stats = {
        "record": rec_id,
        "lead": rec.sig_name[lead_idx].strip(),
        "n_samples": raw.size,
        "duration_min": round(raw.size / FS / 60.0, 1),
        "n_ann_beats": int(n_ref),
        "n_detected": int(peaks.size),
        "TP": int(tp.size),
        "FP": int(fp.size),
        "FN": int(fn.size),
        "Se": round(float(sens), 4),
        "PPV": round(float(ppv), 4),
        "n_detected_dual": int(peaks_dual.size),
        "TP_dual": int(tp_d.size),
        "Se_dual": round(float(tp_d.size / n_ref) if n_ref else np.nan, 4),
        "PPV_dual": round(float(tp_d.size / peaks_dual.size) if peaks_dual.size else np.nan, 4),
        "n_labelled_beats": int(keep.sum()),
        "detect_time_s": round(det_time, 2),
    }
    return df, stats


def main() -> None:
    frames, stats = [], []
    for i, rec_id in enumerate(RECORDS, 1):
        t0 = time.time()
        df, st = process_record(rec_id)
        if not df.empty:
            frames.append(df)
        stats.append(st)
        print(f"[{i:2d}/{len(RECORDS)}] record {rec_id}: "
              f"Se={st['Se']:.4f} PPV={st['PPV']:.4f} "
              f"beats={st['n_labelled_beats']:6d}  ({time.time() - t0:.1f}s)",
              flush=True)

    feats = pd.concat(frames, ignore_index=True)
    feats_path = OUT_DIR / "features_mitdb.csv"
    feats.to_csv(feats_path, index=False)

    st_df = pd.DataFrame(stats)
    st_df.to_csv(OUT_DIR / "detection_stats.csv", index=False)

    total_ref = st_df["n_ann_beats"].sum()
    total_tp = st_df["TP"].sum()
    print("\n=== summary ===")
    print(f"records           : {len(st_df)}")
    print(f"annotated beats   : {total_ref}")
    print(f"labelled beats    : {len(feats)}")
    print(f"detector Se (pool): {total_tp / total_ref:.4f}")
    print(f"detector PPV(pool): {total_tp / st_df['n_detected'].sum():.4f}")
    print(f"class counts      : {feats['label'].value_counts().to_dict()}")
    print(f"saved -> {feats_path}")


if __name__ == "__main__":
    main()
