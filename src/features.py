"""Hand-crafted beat descriptors: morphology, timing, spectral and wavelet.

All functions operate on a whole beat matrix (n_beats x 252 samples) so that the
feature extraction is vectorised and stays fast for ~100k beats.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy.stats import kurtosis, skew

from config import FS, PRE

AMP_OFFSETS_MS = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
SPECTRAL_BANDS = [(1.0, 5.0), (5.0, 15.0), (15.0, 40.0), (40.0, 100.0)]

MORPH_NAMES = [
    "abs_max", "p2p", "mean_abs", "area", "pos_area", "neg_area",
    "std_shape", "skew", "kurtosis", "qrs_width_ms", "n_zero_cross",
    "max_up_slope", "max_down_slope",
]
AMP_NAMES = [f"amp_{o:+d}ms" for o in AMP_OFFSETS_MS]
SPEC_NAMES = [f"bp_{int(lo)}_{int(hi)}Hz_rel" for lo, hi in SPECTRAL_BANDS] + [
    "spec_entropy", "centroid_hz", "hf_lf_ratio",
]
WAVELET_NAMES = [f"wt_e_{n}" for n in ["cD1", "cD2", "cD3", "cD4", "cA4"]]


def morphology_features(beats: np.ndarray, fs: float = FS, pre: int = PRE) -> np.ndarray:
    x = beats - np.median(beats, axis=1, keepdims=True)
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    scale[scale < 1e-9] = 1e-9
    z = x / scale                                   # shape-normalised beat

    abs_max = scale[:, 0]
    p2p = np.ptp(x, axis=1)
    mean_abs = np.mean(np.abs(x), axis=1)
    area = np.sum(x, axis=1) * (1000.0 / fs)        # mV*ms
    pos_area = np.sum(np.clip(x, 0, None), axis=1) * (1000.0 / fs)
    neg_area = np.sum(np.clip(x, None, 0), axis=1) * (1000.0 / fs)
    std_shape = np.std(z, axis=1)
    sk = skew(z, axis=1)
    ku = kurtosis(z, axis=1)

    above = np.abs(z) > 0.5
    qrs_width = np.sum(above, axis=1) * (1000.0 / fs)

    sign = np.sign(z)
    nzc = np.sum(np.abs(np.diff(sign, axis=1)) > 0, axis=1)

    dz = np.diff(z, axis=1) * fs
    max_up = np.max(dz, axis=1)
    max_down = np.min(dz, axis=1)

    feats = np.column_stack([abs_max, p2p, mean_abs, area, pos_area, neg_area,
                             std_shape, sk, ku, qrs_width, nzc, max_up, max_down])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def amplitude_features(beats: np.ndarray, fs: float = FS, pre: int = PRE) -> np.ndarray:
    x = beats - np.median(beats, axis=1, keepdims=True)
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    scale[scale < 1e-9] = 1e-9
    z = x / scale

    idx = [int(round(pre + o * fs / 1000.0)) for o in AMP_OFFSETS_MS]
    idx = [min(max(i, 0), z.shape[1] - 1) for i in idx]
    return z[:, idx]


def spectral_features(beats: np.ndarray, fs: float = FS) -> np.ndarray:
    x = beats - np.mean(beats, axis=1, keepdims=True)
    win = np.hanning(x.shape[1])
    spec = np.abs(np.fft.rfft(x * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / fs)
    total = np.sum(spec, axis=1, keepdims=True)
    total[total < 1e-12] = 1e-12

    bands = []
    for lo, hi in SPECTRAL_BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bands.append(np.sum(spec[:, m], axis=1, keepdims=True) / total)
    bands = np.hstack(bands)

    p = spec / total
    ent = -np.sum(p * np.log2(p + 1e-12), axis=1, keepdims=True) / np.log2(p.shape[1])
    centroid = np.sum(p * freqs[None, :], axis=1, keepdims=True)

    hf = bands[:, [2]]
    lf = bands[:, [1]]
    ratio = hf / np.maximum(lf, 1e-6)

    out = np.hstack([bands, ent, centroid, ratio])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def wavelet_features(beats: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    x = beats - np.mean(beats, axis=1, keepdims=True)
    coeffs = pywt.wavedec(x, wavelet, level=level, axis=1)  # [cA4, cD4, cD3, cD2, cD1]
    energies = np.column_stack([np.sum(c ** 2, axis=1) for c in coeffs])
    total = np.sum(energies, axis=1, keepdims=True)
    total[total < 1e-12] = 1e-12
    rel = energies / total
    # store in cD1..cD4, cA4 order
    rel = rel[:, ::-1]
    return np.nan_to_num(rel, nan=0.0)


def feature_names() -> list[str]:
    return MORPH_NAMES + AMP_NAMES + SPEC_NAMES + WAVELET_NAMES


def extract_all(beats: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """Concatenate every feature family into a single (n_beats x n_feat) matrix."""
    f = np.hstack([
        morphology_features(beats),
        amplitude_features(beats),
        spectral_features(beats),
        wavelet_features(beats),
        rr,
    ])
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def all_feature_names() -> list[str]:
    return feature_names() + ["rr_pre", "rr_post", "rr_local_mean",
                              "rr_pre_ratio", "rr_post_ratio", "rr_abs_diff"]
