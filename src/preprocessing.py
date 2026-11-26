"""Signal preprocessing and QRS (R-peak) detection.

The detector follows the Pan & Tompkins (1985) processing chain:
    1) 5-15 Hz band-pass      2) 5-point derivative      3) squaring
    4) 150 ms moving-window integration
    5) thresholding of the integration envelope
    6) fiducial refinement on the band-passed signal

Two thresholding rules are provided:

* `detect_r_peaks_dual_threshold` - the literal adaptive dual-threshold rule of
  the paper (running SPKI/NPKI levels). It is kept for the ablation reported in
  the course report; it systematically fails on records whose beat amplitude
  alternates strongly (e.g. bigeminy in record 228), because a single running
  signal level cannot follow the alternation.
* `detect_r_peaks` - the variant used by the pipeline: the same envelope, but the
  decision threshold is a fixed fraction of the 90th percentile of the envelope
  amplitude in a +/-3 s neighbourhood, followed by suppression of T-wave humps.
  This is a *local* (rather than a running) amplitude reference, which keeps the
  detector stable when the beat amplitude is non-stationary.

Author: <YOUR NAME>
Course: Biomedical Signal Processing - Course Project, Fall 2026
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

# tuneable parameters of the local-percentile rule
LOCAL_WIN_S = 3.0        # half width of the neighbourhood used for the amplitude reference
LOCAL_RATIO = 0.12       # threshold = LOCAL_RATIO x local 90th percentile
T_WAVE_WIN_S = 0.36      # a smaller peak this close after a bigger one is a T wave


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (no phase shift, no group delay)."""
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64))


def preprocess_ecg(x: np.ndarray, fs: float) -> np.ndarray:
    """Clinical preprocessing: 0.5-40 Hz band-pass removes baseline wander and
    high-frequency muscle noise while keeping the QRS morphology intact."""
    return bandpass(x, fs, 0.5, 40.0, order=3)


# --------------------------------------------------------------------------- #
# Pan-Tompkins envelope
# --------------------------------------------------------------------------- #
def _pan_tompkins_envelope(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (band-passed signal 5-15 Hz, integrated squared-derivative envelope)."""
    bp = bandpass(x, fs, 5.0, 15.0, order=3)

    # 5-point derivative (Pan & Tompkins, Eq. 1) implemented as convolution.
    h = np.array([-1, -2, 0, 2, 1], dtype=np.float64) * (fs / 8.0)
    deriv = np.convolve(bp, h, mode="same")

    squared = deriv ** 2

    win = max(1, int(round(0.150 * fs)))
    mwi = np.convolve(squared, np.ones(win) / win, mode="same")
    return bp, mwi


def _refine(bp: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Move every envelope peak to the largest |signal| deflection within +/-100 ms."""
    half = int(round(0.100 * fs))
    out = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(bp), p + half + 1)
        if hi <= lo:
            continue
        out.append(lo + int(np.argmax(np.abs(bp[lo:hi]))))
    return np.array(sorted(set(out)), dtype=int)


# --------------------------------------------------------------------------- #
# Adopted detector: local percentile threshold
# --------------------------------------------------------------------------- #
def detect_r_peaks(
    x: np.ndarray,
    fs: float,
    refractory_s: float = 0.200,
    win_s: float = LOCAL_WIN_S,
    ratio: float = LOCAL_RATIO,
) -> np.ndarray:
    """Detect R peaks on the Pan-Tompkins integration envelope.

    The envelope is computed exactly as in Pan & Tompkins; the amplitude reference
    is then the 90th percentile of the envelope peak amplitudes inside a +/-3 s
    window, which adapts to slow amplitude drift but is insensitive to single
    artefacts and to alternating beat amplitudes. Candidates below
    `ratio` x reference are discarded, and a candidate that follows a larger one
    within 360 ms is treated as a T-wave hump and removed.

    Returns ascending R-peak sample indices.
    """
    bp, mwi = _pan_tompkins_envelope(x, fs)
    refr = max(1, int(round(refractory_s * fs)))

    cand, _ = find_peaks(mwi, distance=refr)
    if cand.size == 0:
        return np.array([], dtype=int)

    v = mwi[cand]
    half = int(win_s * fs)
    lo = np.searchsorted(cand, cand - half, side="left")
    hi = np.searchsorted(cand, cand + half, side="right")
    local = np.array([np.percentile(v[a:b], 90) for a, b in zip(lo, hi)])

    keep = v > ratio * local
    cand, v = cand[keep], v[keep]
    if cand.size == 0:
        return np.array([], dtype=int)

    # T-wave suppression: within a QRS-T pair keep the larger deflection.
    sel: list[tuple[int, float]] = []
    twin = int(T_WAVE_WIN_S * fs)
    for c, val in zip(cand, v):
        if sel:
            pc, pv = sel[-1]
            if c - pc < twin:
                if val > pv:
                    sel[-1] = (c, val)
                continue
        sel.append((c, val))

    peaks = np.array([c for c, _ in sel], dtype=int)
    return _refine(bp, peaks, fs)


# --------------------------------------------------------------------------- #
# Baseline detector: literal Pan-Tompkins dual threshold (ablation only)
# --------------------------------------------------------------------------- #
def detect_r_peaks_dual_threshold(
    x: np.ndarray,
    fs: float,
    refractory_s: float = 0.200,
    searchback_ratio: float = 1.66,
) -> np.ndarray:
    """Literal implementation of the adaptive SPKI/NPKI dual-threshold rule."""
    bp, mwi = _pan_tompkins_envelope(x, fs)
    refractory = max(1, int(round(refractory_s * fs)))

    cand, _ = find_peaks(mwi, distance=refractory)
    if cand.size == 0:
        return np.array([], dtype=int)

    init = cand[cand < int(2 * fs)]
    if init.size < 2:
        init = cand[: min(10, cand.size)]
    spki = float(np.max(mwi[init])) / 3.0
    npki = float(np.mean(mwi[init])) / 2.0

    signal_peaks: list[int] = []
    rr: list[float] = []

    for c in cand:
        th1 = npki + 0.25 * (spki - npki)
        th2 = 0.5 * th1
        v = mwi[c]

        if signal_peaks and (c - signal_peaks[-1]) < refractory:
            if v > mwi[signal_peaks[-1]]:
                signal_peaks[-1] = c
            continue

        if v > th1:
            signal_peaks.append(c)
            spki = 0.125 * v + 0.875 * spki
            if len(signal_peaks) > 1:
                rr.append(float(signal_peaks[-1] - signal_peaks[-2]))
        else:
            npki = 0.125 * v + 0.875 * npki
            if len(rr) >= 3 and signal_peaks:
                rr_mean = float(np.mean(rr[-8:]))
                if c - signal_peaks[-1] > searchback_ratio * rr_mean:
                    seg = cand[(cand > signal_peaks[-1]) & (cand <= c)]
                    if seg.size:
                        best = int(seg[np.argmax(mwi[seg])])
                        if mwi[best] > th2:
                            signal_peaks.append(best)
                            rr.append(float(best - signal_peaks[-2]))

    peaks = np.array(sorted(signal_peaks), dtype=int)
    if peaks.size == 0:
        return peaks
    return _refine(bp, peaks, fs)


# --------------------------------------------------------------------------- #
# Detection scoring
# --------------------------------------------------------------------------- #
def match_peaks(det: np.ndarray, ref: np.ndarray, tol: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy one-to-one matching of detected peaks to reference annotations.

    Both arrays must be sorted in time (true for WFDB annotations).
    Returns (tp_det_idx, fp_det_idx, fn_ref_idx).
    """
    det = np.asarray(det, dtype=int)
    ref = np.asarray(ref, dtype=int)
    used = np.zeros(ref.size, dtype=bool)
    tp, fp = [], []

    for i, d in enumerate(det):
        lo = np.searchsorted(ref, d - tol, side="left")
        hi = np.searchsorted(ref, d + tol, side="right")
        if hi <= lo:
            fp.append(i)
            continue
        cand = np.arange(lo, hi)
        cand = cand[~used[cand]]
        if cand.size == 0:
            fp.append(i)
            continue
        j = cand[np.argmin(np.abs(ref[cand] - d))]
        used[j] = True
        tp.append(i)

    fn = np.where(~used)[0]
    return np.array(tp, dtype=int), np.array(fp, dtype=int), fn
