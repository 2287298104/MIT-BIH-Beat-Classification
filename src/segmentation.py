"""Beat segmentation and AAMI EC57 label mapping."""

from __future__ import annotations

import numpy as np

from config import AAMI_MAP, CLASSES, POST, PRE, WIN


def aami_label(symbol: str) -> str | None:
    """Map a WFDB annotation symbol to an AAMI super-class (None = not modelled)."""
    cls = AAMI_MAP.get(symbol)
    if cls is None or cls not in CLASSES:
        return None
    return cls


def extract_beats(
    signal: np.ndarray, peaks: np.ndarray, pre: int = PRE, post: int = POST
) -> tuple[np.ndarray, np.ndarray]:
    """Cut fixed-length windows around the given R-peak indices.

    Returns (beats[n, pre+post], valid_mask) where valid_mask marks beats whose
    window lies fully inside the recording.
    """
    n = peaks.size
    beats = np.zeros((n, pre + post), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i, r in enumerate(peaks):
        lo, hi = r - pre, r + post
        if lo < 0 or hi > signal.size:
            continue
        beats[i] = signal[lo:hi]
        valid[i] = True
    return beats, valid


def label_beats(detected: np.ndarray, ann_samples: np.ndarray, ann_symbols: list[str],
                tol: int) -> tuple[np.ndarray, np.ndarray]:
    """Attach an AAMI label to every detected peak via one-to-one matching.

    A detected peak is labelled only when an annotated beat lies within `tol`
    samples. Detected peaks without a matching annotation are discarded because
    they carry no reliable ground truth.

    Returns (label_index[n_det], matched_mask[n_det]) with label_index == -1 for
    unmatched / non-beat detections.
    """
    labels = np.full(detected.size, -1, dtype=int)
    matched = np.zeros(detected.size, dtype=bool)
    used = np.zeros(ann_samples.size, dtype=bool)

    order = np.argsort(ann_samples)
    ann_sorted = ann_samples[order]
    sym_sorted = [ann_symbols[i] for i in order]

    for i, d in enumerate(detected):
        lo = np.searchsorted(ann_sorted, d - tol, side="left")
        hi = np.searchsorted(ann_sorted, d + tol, side="right")
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        idx = idx[~used[idx]]
        if idx.size == 0:
            continue
        j = idx[np.argmin(np.abs(ann_sorted[idx] - d))]
        used[j] = True
        matched[i] = True
        cls = aami_label(sym_sorted[j])
        labels[i] = CLASSES.index(cls) if cls is not None else -1

    return labels, matched


def rr_features(r_locs: np.ndarray, fs: float) -> np.ndarray:
    """Beat-to-beat timing features computed on the detected R-peak sequence.

    Columns
        0 pre-RR      interval that ends at the beat (s)
        1 post-RR     interval that starts at the beat (s)
        2 local RR    median of the +/-3 surrounding intervals (s)
        3 pre ratio   pre-RR / local RR
        4 post ratio  post-RR / local RR
        5 RR change   |post-RR - pre-RR| (s)

    The pair (short pre-RR, long post-RR) is the classic signature of a premature
    beat followed by a compensatory pause, so both directions are kept.
    """
    n = r_locs.size
    out = np.zeros((n, 6), dtype=np.float64)
    if n < 3:
        return out

    iv = np.diff(np.asarray(r_locs, dtype=float)) / fs      # iv[k] = t[k+1]-t[k]
    med = float(np.median(iv))
    med = med if med > 1e-3 else 0.8

    pre = np.concatenate(([med], iv))          # interval ending at beat i
    post = np.concatenate((iv, [med]))         # interval starting at beat i

    for i in range(n):
        lo = max(0, i - 3)
        hi = min(iv.size, i + 2)
        loc = float(np.median(iv[lo:hi])) if hi > lo else med
        loc = loc if loc > 1e-3 else med
        out[i] = [pre[i], post[i], loc, pre[i] / loc, post[i] / loc,
                  abs(post[i] - pre[i])]
    return out


def window_bounds(n_samples: int, pre: int = PRE, post: int = POST) -> int:
    return pre + post if n_samples >= WIN else n_samples
