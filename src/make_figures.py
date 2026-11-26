"""Generate every figure used in the report (all labels in English)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (AAMI_MAP, CLASSES, DATA_DIR, FIG_DIR, FS, MATCH_TOL_S, OUT_DIR,
                    PRE, WIN)
from preprocessing import detect_r_peaks, preprocess_ecg
from segmentation import extract_beats, label_beats

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

C_N, C_S, C_V, C_F, C_Q = "#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B3"
CLASS_COLORS = {"N": C_N, "S": C_S, "V": C_V, "F": C_F, "Q": C_Q}


def save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path.name)


# --------------------------------------------------------------------------- #
def fig_pipeline() -> None:
    stages = [
        "MIT-BIH\n42 records",
        "Filter\n0.5-40 Hz",
        "Pan-Tompkins\nQRS detect",
        "Beats\n-250/+450",
        "Features\n40 dims",
        "Classifiers\nRF/SVM/LR",
        "Evaluation\nSe, PPV, F1",
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(6.9, 1.35))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bw, bh = 0.90, 0.62
    for i, title in enumerate(stages):
        x = i + 0.06
        ax.add_patch(FancyBboxPatch((x, 0.19), bw, bh,
                                    boxstyle="round,pad=0.012,rounding_size=0.06",
                                    linewidth=0.9, edgecolor="#28527A", facecolor="#EAF1F8"))
        ax.text(x + bw / 2, 0.19 + bh / 2, title, ha="center", va="center",
                fontsize=5.9, weight="bold", linespacing=1.45)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.005, 0.5), (x + 0.995, 0.5),
                                         arrowstyle="-|>", mutation_scale=7,
                                         linewidth=0.8, color="#28527A"))
    save(fig, "fig01_pipeline.png")


# --------------------------------------------------------------------------- #
def fig_detection(rec_id: int = 119, t0: float = 30.0, dur: float = 9.0) -> None:
    rec = wfdb.rdrecord(str(DATA_DIR / str(rec_id)))
    ann = wfdb.rdann(str(DATA_DIR / str(rec_id)), "atr")
    names = [n.strip() for n in rec.sig_name]
    li = names.index("MLII") if "MLII" in names else 0
    raw = np.nan_to_num(np.asarray(rec.p_signal[:, li], dtype=float))
    clean = preprocess_ecg(raw, FS)

    peaks = detect_r_peaks(raw, FS)
    tol = int(round(MATCH_TOL_S * FS))
    labels, _ = label_beats(peaks, ann.sample, ann.symbol, tol)

    # centre the excerpt on a ventricular beat so that the figure shows both an
    # abnormal and normal morphology
    v_idx = np.where(labels == CLASSES.index("V"))[0]
    if v_idx.size:
        t0 = max(0.0, peaks[v_idx[len(v_idx) // 2]] / FS - 2.0)
    s0, s1 = int(t0 * FS), int((t0 + dur) * FS)
    t = np.arange(s0, s1) / FS - t0

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.6),
                             gridspec_kw={"height_ratios": [1, 1.25]})
    axes[0].plot(t, clean[s0:s1], lw=0.8, color="#333333")

    pk = peaks[(peaks >= s0) & (peaks < s1)]
    axes[0].plot(pk / FS - t0, clean[pk], "o", ms=4.0, mfc="none",
                 mec=C_V, mew=1.0, label="detected R peaks")
    ann_in = (ann.sample >= s0) & (ann.sample < s1)
    axes[0].plot(ann.sample[ann_in] / FS - t0, clean[ann.sample[ann_in]], "|", ms=9,
                 color="#2E7D32", mew=1.1, label="cardiologist annotation")
    axes[0].set_xlim(t[0], t[-1])
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("ECG (mV)")
    axes[0].set_title(f"Record {rec_id}, lead {names[li]}: Pan-Tompkins detection "
                      f"({pk.size} R peaks in this excerpt)")
    axes[0].legend(loc="upper right", fontsize=7.5, frameon=False, ncol=2)

    # lower panel: the segmented beats of this excerpt, coloured by AAMI class
    seen = set()
    for p in pk:
        i = int(np.where(peaks == p)[0][0])
        if labels[i] < 0:
            continue
        lab = CLASSES[labels[i]]
        w = clean[p - PRE: p + WIN - PRE]
        axes[1].plot(np.arange(-PRE, WIN - PRE) / FS * 1000, w,
                     color=CLASS_COLORS[lab], lw=1.0, alpha=0.85,
                     label=None if lab in seen else f"class {lab}")
        seen.add(lab)
    axes[1].axvline(0, color="k", lw=0.6, ls=":")
    axes[1].set_xlim(-250, 450)
    axes[1].set_xlabel("time relative to R peak (ms)")
    axes[1].set_ylabel("ECG (mV)")
    axes[1].set_title("Segmented beats (-250 / +450 ms), coloured by AAMI class")
    axes[1].legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig02_detection_example.png")


# --------------------------------------------------------------------------- #
def fig_class_distribution(summary: dict) -> None:
    tr = [summary["train_dist"][c] for c in CLASSES]
    te = [summary["test_dist"][c] for c in CLASSES]
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    b1 = ax.bar(x - 0.19, tr, 0.38, label="DS1 (train)", color="#4C72B0")
    b2 = ax.bar(x + 0.19, te, 0.38, label="DS2 (test)", color="#C44E52")
    ax.set_yscale("log")
    short = {"N": "N\nnormal", "S": "S\nsupravent.", "V": "V\nventricular",
             "F": "F\nfusion"}
    ax.set_xticks(x, [short.get(c, c) for c in CLASSES])
    ax.set_ylabel("number of beats (log scale)")
    ax.set_title("AAMI class distribution (inter-patient split)")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h * 1.12, f"{int(h)}",
                        ha="center", va="bottom", fontsize=6.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(1, max(max(tr), max(te)) * 4)
    save(fig, "fig03_class_distribution.png")


# --------------------------------------------------------------------------- #
def fig_templates(records=(119, 208, 232, 209)) -> None:
    """Median beat per AAMI class, pooled over records that contain the class."""
    pool: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    for rid in records:
        rec = wfdb.rdrecord(str(DATA_DIR / str(rid)))
        ann = wfdb.rdann(str(DATA_DIR / str(rid)), "atr")
        names = [n.strip() for n in rec.sig_name]
        li = names.index("MLII") if "MLII" in names else 0
        raw = np.nan_to_num(np.asarray(rec.p_signal[:, li], dtype=float))
        clean = preprocess_ecg(raw, FS)
        peaks = detect_r_peaks(raw, FS)
        beats, valid = extract_beats(clean, peaks)
        labels, matched = label_beats(peaks, ann.sample, ann.symbol, int(0.15 * FS))
        for c in CLASSES:
            idx = np.where(valid & matched & (labels == CLASSES.index(c)))[0]
            if idx.size:
                pool[c].append(beats[idx][:300])

    t = np.arange(-PRE, WIN - PRE) / FS * 1000
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for c in CLASSES:
        if not pool[c]:
            continue
        B = np.vstack(pool[c])
        med = np.median(B, axis=0)
        q1, q3 = np.percentile(B, 25, axis=0), np.percentile(B, 75, axis=0)
        ax.plot(t, med, color=CLASS_COLORS[c], lw=1.5, label=f"{c} (n={len(B)})")
        ax.fill_between(t, q1, q3, color=CLASS_COLORS[c], alpha=0.12, lw=0)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(-250, 400)
    ax.set_xlabel("time relative to R peak (ms)")
    ax.set_ylabel("ECG (mV)")
    ax.set_title("Median beat waveform per AAMI class (shaded: inter-quartile range)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "fig04_beat_templates.png")


# --------------------------------------------------------------------------- #
def fig_confusion(summary: dict) -> None:
    models = list(summary["confusion"].keys())
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.3))
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        cm = np.array(summary["confusion"][m], dtype=float)
        row = cm.sum(axis=1, keepdims=True)
        row[row == 0] = 1
        cmn = cm / row
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        n_cls = len(CLASSES)
        ax.set_xticks(range(n_cls), CLASSES)
        ax.set_yticks(range(n_cls), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(m.replace("_", " "), fontsize=11)
        ax.tick_params(labelsize=9.5)
        ax.grid(False)
        for i in range(n_cls):
            for j in range(n_cls):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                            fontsize=9.0, color="white" if cmn[i, j] > 0.55 else "#222222")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="row-normalised rate")
    fig.suptitle("Confusion matrices on DS2 (counts)", y=1.02, fontsize=10)
    save(fig, "fig05_confusion_matrices.png")


# --------------------------------------------------------------------------- #
def fig_importance() -> None:
    imp = pd.read_csv(OUT_DIR / "feature_importance.csv").head(15)[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.barh(imp["feature"], imp["importance"], color="#4C72B0", height=0.7)
    ax.set_xlabel("mean decrease in impurity")
    ax.set_title("Random forest - 15 most important features")
    ax.grid(axis="y", visible=False)
    save(fig, "fig06_feature_importance.png")


# --------------------------------------------------------------------------- #
def fig_pr(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for m, d in summary["pr_curves"].items():
        ax.plot(d["recall"], d["precision"], lw=1.3, label=m.replace("_", " "))
    ax.set_xlabel("recall (V class)")
    ax.set_ylabel("precision (V class)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curve, ventricular ectopic beats (DS2)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "fig07_pr_curves.png")


# --------------------------------------------------------------------------- #
def fig_detector_stats() -> None:
    st = pd.read_csv(OUT_DIR / "detection_stats.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    axes[0].hist(st["Se"] * 100, bins=np.arange(90, 100.6, 1.0), color="#4C72B0",
                 edgecolor="white")
    axes[0].set_xlabel("sensitivity (%)")
    axes[0].set_ylabel("number of records")
    axes[0].set_title("Per-record detection sensitivity")

    axes[1].hist(st["PPV"] * 100, bins=np.arange(90, 100.6, 1.0), color="#C44E52",
                 edgecolor="white")
    axes[1].set_xlabel("positive predictive value (%)")
    axes[1].set_title("Per-record detection PPV")
    fig.tight_layout()
    save(fig, "fig08_detector_per_record.png")


def main() -> None:
    with open(OUT_DIR / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    fig_pipeline()
    fig_detection()
    fig_class_distribution(summary)
    fig_templates()
    fig_confusion(summary)
    fig_importance()
    fig_pr(summary)
    fig_detector_stats()


if __name__ == "__main__":
    main()
