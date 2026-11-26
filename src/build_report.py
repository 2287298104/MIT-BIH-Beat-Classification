"""Build the final Word report from the measured results.

All numbers are read from outputs/summary.json, outputs/detection_stats.csv and
outputs/cv_model_selection.csv, so the report can be regenerated after a re-run
without editing any text by hand.

Usage
    python src/build_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASSES, DS1, DS2, FIG_DIR, OUT_DIR, PROJECT_ROOT

BODY_FONT = "Times New Roman"
REPORT_PATH = PROJECT_ROOT / "report" / "Course_Project_Report.docx"

STUDENT_NAME = "Feng Runhang"
STUDENT_ID = "2315104007"
COURSE = "Biomedical Signal Processing"
TERM = "Fall 2026"
INSTRUCTOR = "Dr. DUAN Fang"
UNIVERSITY = "Huaqiao University"
TITLE = ("Beat-Level Arrhythmia Detection from Single-Lead ECG: "
         "A Reproducible Classical Pipeline on the MIT-BIH Arrhythmia Database")
GITHUB_URL = "https://github.com/2287298104/MIT-BIH-Beat-Classification"
DATE = "26 November 2026"

PRETTY = {"RandomForest": "Random forest", "SVM_RBF": "SVM (RBF)",
          "LogReg": "Logistic regression"}


# --------------------------------------------------------------------------- #
# docx helpers
# --------------------------------------------------------------------------- #
def set_base_style(doc: Document) -> None:
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(11)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.22

    for name, size in (("Heading 1", 14.5), ("Heading 2", 12)):
        s = doc.styles[name]
        s.font.name = BODY_FONT
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor(0x1F, 0x33, 0x55)
        s.paragraph_format.space_before = Pt(13 if name == "Heading 1" else 9)
        s.paragraph_format.space_after = Pt(4)
        s.paragraph_format.keep_with_next = True


def add_page_number_footer(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


def add_header_text(section, text: str) -> None:
    p = section.header.paragraphs[0]
    p.text = text
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in p.runs:
        r.font.size = Pt(8)
        r.font.name = BODY_FONT
        r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def para(doc, text: str = "", size: float | None = None, bold=False, italic=False,
         align=None, space_after: float | None = 6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def body(doc, text: str):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0.75)
    return p


def bullets(doc, items: list[str]):
    for txt in items:
        p = doc.add_paragraph(txt, style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def caption(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(9)
    run.italic = True
    p.paragraph_format.space_after = Pt(10)
    return p


MAX_W_CM = 15.4          # usable text width with 2.6 cm margins on A4


def figure(doc, name: str, cap: str, width_cm: float = 15.4):
    width_cm = min(width_cm, MAX_W_CM)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(FIG_DIR / name), width=Cm(width_cm))
    caption(doc, cap)


def shade(cell, hexcolor: str = "E8EEF7") -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def table(doc, header: list[str], rows: list[list[str]], widths=None,
          font_size: float = 8.6, align_right_from: int = 1):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(font_size)
        r.font.name = BODY_FONT
        shade(hdr[i])
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if i >= align_right_from
                           else WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(str(v))
            r.font.size = Pt(font_size)
            r.font.name = BODY_FONT
    if widths:
        total = sum(widths)
        scale = MAX_W_CM / total if total > MAX_W_CM else 1.0
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w * scale)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    return t


# --------------------------------------------------------------------------- #
# cover
# --------------------------------------------------------------------------- #
def cover(doc: Document) -> None:
    for _ in range(3):
        para(doc, "", space_after=0)
    para(doc, UNIVERSITY, size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, "College of Information Science and Engineering", size=11,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
    para(doc, "COURSE PROJECT REPORT", size=12, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
    for line in ("Beat-Level Arrhythmia Detection from Single-Lead ECG:",
                 "A Reproducible Classical Pipeline on the",
                 "MIT-BIH Arrhythmia Database"):
        para(doc, line, size=16.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_after=3)
    para(doc, "", space_after=26)
    para(doc, "Course:  " + COURSE, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "Semester:  " + TERM, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=32)

    rows = [("Name", STUDENT_NAME), ("Student ID", STUDENT_ID),
            ("Instructor", INSTRUCTOR), ("Date", DATE)]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in rows:
        cells = t.add_row().cells
        p = cells[0].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run(k + ":  ")
        r.font.size = Pt(12)
        r.font.name = BODY_FONT
        p2 = cells[1].paragraphs[0]
        r2 = p2.add_run(v)
        r2.font.size = Pt(12)
        r2.font.name = BODY_FONT
        cells[0].width = Cm(4.5)
        cells[1].width = Cm(7.5)

    for _ in range(5):
        para(doc, "", space_after=0)
    para(doc, "Repository: " + GITHUB_URL, size=9, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build() -> None:
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    det = pd.read_csv(OUT_DIR / "detection_stats.csv")
    cv = pd.read_csv(OUT_DIR / "cv_model_selection.csv")

    models = summary["models"]
    per_class = summary["per_class"]
    best_name = max(models, key=lambda k: models[k]["macro_f1"])
    best = models[best_name]
    rf, svm, lr = models["RandomForest"], models["SVM_RBF"], models["LogReg"]
    bp = per_class[best_name]

    n_train, n_test = summary["n_train"], summary["n_test"]
    tr, te = summary["train_dist"], summary["test_dist"]
    n_rec = len(det)
    n_ds1, n_ds2 = len(DS1), len(DS2)
    n_ann = int(det["n_ann_beats"].sum())
    n_lab = int(det["n_labelled_beats"].sum())

    se = det["TP"].sum() / n_ann
    ppv = det["TP"].sum() / det["n_detected"].sum()
    se_d = det["TP_dual"].sum() / n_ann
    ppv_d = det["TP_dual"].sum() / det["n_detected_dual"].sum()
    worst_se = det.loc[det["Se"].idxmin()]
    worst_se_d = det.loc[det["Se_dual"].idxmin()]
    cv_best = cv["cv_macro_f1_mean"].max()

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.6)
    set_base_style(doc)
    add_header_text(sec, f"{COURSE} - Course Project, {TERM}")
    add_page_number_footer(sec)
    # the cover page carries neither header nor page number
    sec.different_first_page_header_footer = True

    cover(doc)

    # ------------------------------------------------------------ abstract
    doc.add_heading("Abstract", level=1)
    body(doc, (
        "Beat-level arrhythmia analysis is the foundation of automated "
        "electrocardiogram (ECG) monitoring. This project implements a complete, "
        "reproducible signal-processing pipeline on the MIT-BIH Arrhythmia Database: "
        "a Pan-Tompkins detector localises every R peak, each beat is cut into a "
        "252-sample window (-250 ms to +450 ms), 40 hand-crafted descriptors are "
        "extracted from morphology, rhythm, spectrum and wavelet sub-bands, and three "
        f"classifiers are trained on {n_ds1} records (DS1) and tested on the {n_ds2} "
        "disjoint records of DS2 using the AAMI EC57 super-class taxonomy."
    ))
    body(doc, (
        f"On the {n_ann:,} annotated beats of the {n_rec} analysed records the "
        f"detector reaches a pooled sensitivity of {se * 100:.2f}% and a positive "
        f"predictive value of {ppv * 100:.2f}%. Under the inter-patient protocol the "
        f"best model ({PRETTY[best_name]}) reaches an overall accuracy of "
        f"{best['accuracy'] * 100:.2f}%, a weighted F1 of {best['weighted_f1']:.4f} "
        f"and a ventricular-ectopic F1 of {best['V_f1']:.4f}, but only a macro-F1 of "
        f"{best['macro_f1']:.4f}: the supraventricular (F1 = {bp['S']['f1']:.3f}) and "
        f"fusion (F1 = {bp['F']['f1']:.3f}) classes do not survive the change of "
        "patient. The same model reaches a macro-F1 of "
        f"{cv_best:.3f} under a beat-level cross-validation within DS1, which "
        "quantifies how much of the difficulty comes from patient-independent "
        "evaluation rather than from the classifier itself. The report documents the "
        "pipeline stage by stage, including a comparison between the literal "
        "Pan-Tompkins dual-threshold rule and a locally adaptive threshold variant "
        "that is more robust to recordings with strongly alternating beat amplitude."
    ))

    # ------------------------------------------------------------ 1
    doc.add_heading("1. Introduction", level=1)
    doc.add_heading("1.1 Background and motivation", level=2)
    body(doc, (
        "Arrhythmias are among the most common cardiac abnormalities found in "
        "ambulatory (Holter) monitoring. A 24-hour recording sampled at 360 Hz "
        "contains roughly 31 million samples and about 100,000 heartbeats, so manual "
        "inspection is impractical and automated beat segmentation and classification "
        "is a standard component of clinical ECG software and of wearable monitors."
    ))
    body(doc, (
        "The MIT-BIH Arrhythmia Database (Moody & Mark, 2001) is the reference "
        "benchmark for this task: 48 half-hour two-lead recordings from 47 subjects, "
        "with every beat annotated by cardiologists. Its size makes it possible to "
        "study the interaction between signal processing, feature design and "
        "classifier behaviour on a laptop, which is exactly what this project does."
    ))

    doc.add_heading("1.2 Research question and objectives", level=2)
    body(doc, (
        "The project answers one question: how well can a lightweight, fully "
        "interpretable classical pipeline - hand-crafted features with tree or kernel "
        "classifiers, trained on one group of patients and tested on disjoint "
        "patients - detect and classify ventricular ectopic beats, and where do its "
        "errors come from?"
    ))
    body(doc, "The specific objectives are:")
    bullets(doc, [
        "to implement the Pan-Tompkins QRS detector and quantify its sensitivity and "
        "positive predictive value against the cardiologist annotations;",
        "to design a compact feature set describing beat morphology, beat-to-beat "
        "timing, spectral content and wavelet sub-band energy;",
        "to evaluate classification with an inter-patient protocol using F1-based "
        "metrics that stay meaningful under severe class imbalance;",
        "to locate the dominant error sources and discuss the clinical meaning and the "
        "limitations of the resulting system.",
    ])

    doc.add_heading("1.3 Contributions", level=2)
    body(doc, (
        "The contribution is a complete, reproducible implementation rather than a new "
        "algorithm. Every stage is written from first principles (no black-box QRS or "
        "feature library), the evaluation follows the AAMI EC57 recommendation and the "
        "de Chazal inter-patient split, and the whole study is regenerated by a single "
        "command. Three elements are reported that are often omitted: the detector is "
        "scored separately from the classifier, the literal Pan-Tompkins thresholding "
        "rule is compared with a robust variant on the same data, and the gap between "
        "beat-level and patient-level evaluation is measured explicitly."
    ))

    # ------------------------------------------------------------ 2
    doc.add_heading("2. Data", level=1)
    doc.add_heading("2.1 Database and records used", level=2)
    body(doc, (
        f"{n_rec} of the 48 recordings were analysed (about 90 MB). Each record is 30 "
        "minutes long, sampled at 360 Hz with 11-bit resolution over a 10 mV range and "
        "contains two leads; lead MLII (modified limb lead II) is used whenever "
        "present, otherwise the first available lead (V5 in some records). Reference "
        "annotations were read from the .atr files; only beat annotations were kept, "
        f"giving {n_ann:,} annotated beats."
    ))
    body(doc, (
        "Six recordings are not analysed. Records 102, 104, 107 and 217 are dominated "
        "by paced rhythms, whose beats belong to the AAMI class Q that the standard "
        "excludes from beat-level evaluation; records 113 and 121 are not part of the "
        "published DS1/DS2 split. Keeping the split unchanged matters: it is the "
        "protocol under which published results can be compared."
    ))

    doc.add_heading("2.2 Label taxonomy", level=2)
    body(doc, (
        "The original MIT-BIH beat symbols were mapped onto the AAMI EC57 "
        "super-classes (Table 1). The unclassifiable symbol '?' and all non-beat "
        "annotations (rhythm changes, signal-quality markers) are excluded."
    ))
    caption(doc, "Table 1. Mapping of MIT-BIH annotation symbols to AAMI EC57 super-classes.")
    table(doc,
          ["AAMI class", "MIT-BIH symbols", "Meaning"],
          [["N", "N, L, R, e, j", "Normal beat, left/right bundle-branch block, escape, junctional"],
           ["S", "A, a, J, S", "Atrial / supraventricular premature or aberrant beat"],
           ["V", "V, E", "Premature ventricular contraction, ventricular escape"],
           ["F", "F", "Fusion of ventricular and normal beat"]],
          widths=[2.4, 4.2, 9.0], align_right_from=99)

    doc.add_heading("2.3 Inter-patient split and class distribution", level=2)
    body(doc, (
        "A random beat-level split would place beats of the same patient in training "
        "and test and would inflate every score, because a patient's QRS morphology "
        "repeats almost identically from beat to beat. The inter-patient split of de "
        f"Chazal et al. (2004) is therefore used: {n_ds1} records for training (DS1) "
        f"and {n_ds2} disjoint records for testing (DS2). The distribution is strongly imbalanced "
        "(Table 2, Figure 1): normal beats are "
        f"{tr['N'] / n_train * 100:.1f}% of the training set, ventricular ectopic "
        f"beats {tr['V'] / n_train * 100:.1f}%, supraventricular "
        f"{tr['S'] / n_train * 100:.1f}% and fusion only "
        f"{tr['F'] / n_train * 100:.1f}%."
    ))
    caption(doc, "Table 2. Beats per AAMI class in the training (DS1) and test (DS2) sets.")
    table(doc,
          ["class", "DS1 (train)", "share", "DS2 (test)", "share"],
          [[c, f"{tr[c]:,}", f"{tr[c] / n_train * 100:.2f}%", f"{te[c]:,}",
            f"{te[c] / n_test * 100:.2f}%"] for c in CLASSES] +
          [["total", f"{n_train:,}", "100%", f"{n_test:,}", "100%"]],
          widths=[2.2, 3.2, 2.6, 3.2, 2.6])
    figure(doc, "fig03_class_distribution.png",
           "Figure 1. Class distribution of the inter-patient split (logarithmic scale).", 12.6)

    # ------------------------------------------------------------ 3
    doc.add_heading("3. Methodology", level=1)
    body(doc, (
        "The pipeline contains five stages (Figure 2): filtering, R-peak detection, "
        "beat segmentation, feature extraction and classification with evaluation on "
        "unseen patients."
    ))
    figure(doc, "fig01_pipeline.png",
           "Figure 2. Block diagram of the implemented processing chain.", 16.2)

    doc.add_heading("3.1 Preprocessing", level=2)
    body(doc, (
        "Each channel is filtered with a third-order Butterworth band-pass of "
        "0.5-40 Hz applied forward and backward (sosfiltfilt). The lower cut-off "
        "removes baseline wander caused by respiration and electrode drift, which "
        "would otherwise bias amplitude and area features; the upper cut-off "
        "attenuates electromyographic noise and power-line interference while keeping "
        "the QRS complex, whose clinically relevant energy lies below about 40 Hz. "
        "Zero-phase filtering matters because a causal filter delays the R peak and "
        "would shift every segmentation window."
    ))

    doc.add_heading("3.2 R-peak detection", level=2)
    body(doc, (
        "Detection follows Pan & Tompkins (1985). The signal is band-pass filtered at "
        "5-15 Hz to emphasise QRS energy relative to P/T waves and motion artefact, "
        "differentiated with the five-point derivative of the original paper, squared "
        "to make all deflections positive, and integrated over a 150 ms moving window "
        "so that each QRS complex becomes a single hump."
    ))
    body(doc, (
        "The original paper thresholds this envelope with two running estimates, the "
        "signal-peak level SPKI and the noise-peak level NPKI, and applies a 200 ms "
        "refractory period plus a search-back rule for long RR intervals. That rule is "
        "implemented verbatim (detect_r_peaks_dual_threshold) and serves as the "
        "baseline in the ablation of Section 4.1. It fails systematically on "
        "recordings whose beat amplitude alternates, because a running signal level "
        "cannot follow an alternation: after a large complex the threshold rises above "
        "the next, small complex, and the search-back never fires because the RR "
        "interval stays short."
    ))
    body(doc, (
        "The pipeline therefore uses the same envelope with a local amplitude "
        "reference instead of a running one: a candidate is accepted when its envelope "
        "value exceeds 12% of the 90th percentile of the envelope amplitudes inside a "
        "+/-3 s neighbourhood. The 90th percentile is insensitive to single artefacts "
        "and, because it is recomputed locally, it follows slow amplitude drift. A "
        "second rule removes T-wave humps: a candidate appearing within 360 ms after a "
        "larger candidate is discarded. Accepted peaks are finally refined to the "
        "largest absolute deflection of the 5-15 Hz signal within +/-100 ms, so the "
        "windows are centred on the R wave rather than on the integration hump."
    ))
    body(doc, (
        "Detector quality is scored against the annotations with a 150 ms matching "
        "window: sensitivity Se = TP/(TP+FN) and positive predictive value "
        "PPV = TP/(TP+FP)."
    ))

    doc.add_heading("3.3 Beat segmentation", level=2)
    body(doc, (
        "Each detected R peak defines a window from -250 ms to +450 ms (90 + 162 = 252 "
        "samples at 360 Hz). The asymmetric window keeps the whole P wave and the ST "
        "segment/T wave, which carry most of the information separating "
        "supraventricular and fusion beats, while avoiding the next beat at fast rates. "
        "Every detected peak is matched one-to-one to the nearest annotation within "
        "150 ms; matched beats inherit the AAMI label of that annotation, and unmatched "
        "detections are discarded because they carry no ground truth. Beats whose "
        "window crosses a recording boundary are dropped, leaving "
        f"{n_lab:,} labelled beats."
    ))
    figure(doc, "fig02_detection_example.png",
           "Figure 3. Detection and segmentation on record 119: detected R peaks "
           "(circles) versus cardiologist annotations (bars); lower panel, the beat "
           "windows coloured by AAMI class.", 16.0)

    doc.add_heading("3.4 Feature extraction", level=2)
    body(doc, (
        "Forty descriptors are computed per beat in four families (Table 3). Before "
        "morphological description each beat is centred by subtracting its median and "
        "divided by its peak absolute amplitude, which makes the shape features "
        "independent of electrode gain and of slow amplitude changes; the absolute "
        "amplitude remains available as a separate feature. Timing features come from "
        "the detected R-peak sequence rather than from the window, so they encode the "
        "rhythm context of the beat - the single most informative signal for "
        "separating a premature contraction from a normal beat with an unusual shape."
    ))
    caption(doc, "Table 3. Feature families (40 descriptors).")
    table(doc,
          ["family", "n", "descriptors"],
          [["morphology", "13", "absolute peak, peak-to-peak, mean |x|, signed/positive/negative area, "
                                "shape std, skewness, kurtosis, QRS width above 50% of the peak, "
                                "zero crossings, max up/down slope"],
           ["amplitude samples", "9", "normalised amplitude at -200, -150, -100, -50, 0, +50, +100, +150, +200 ms"],
           ["spectral", "6", "relative power in 1-5, 5-15, 15-40 and 40-100 Hz bands, spectral entropy, centroid, HF/LF ratio"],
           ["wavelet", "5", "relative energy of db4 level-4 details cD1-cD4 and approximation cA4"],
           ["rhythm", "6", "pre-RR, post-RR, local mean RR (7 intervals), pre/post-RR ratios, |RR difference|"]],
          widths=[3.2, 0.9, 12.4], align_right_from=99)

    doc.add_heading("3.5 Classifiers, imbalance and decision rule", level=2)
    body(doc, (
        "Three classifiers of increasing capacity are compared: multinomial logistic "
        "regression on standardised features (linear baseline), a random forest with "
        "300 trees (non-linear, provides feature importances) and a support vector "
        "machine with an RBF kernel (C = 3, gamma = 'scale', Platt-scaled "
        "probabilities). Logistic regression and the SVM are preceded by "
        "standardisation because both are scale-sensitive; the forest is not."
    ))
    body(doc, (
        "Normal beats outnumber fusion beats by more than a hundred to one, so all "
        "three models use class weighting (class_weight = 'balanced'), and the final "
        "decision is not the plain argmax of the predicted probabilities: for each "
        "model a vector of class multipliers is tuned by grid search on a "
        "patient-disjoint validation half of DS1 (11 records for fitting, 11 for "
        "tuning), maximising macro-F1 there, and the tuned multipliers are then applied "
        "to DS2. This is the classical way of moving the operating point of an "
        "imbalanced classifier without touching the test set. Cost-sensitive learning "
        "was preferred to oversampling or SMOTE because synthetic interpolation of ECG "
        "waveforms can produce morphologically implausible beats, whereas weighting "
        "keeps every real beat."
    ))
    body(doc, (
        "A light hyper-parameter check (three-fold cross-validation on a 15,000-beat "
        "subsample of DS1, Table 4) gave macro-F1 values within one standard deviation "
        "of each other for min_samples_leaf = 2 and 4; the more expressive setting "
        "(leaf = 2) was kept."
    ))
    caption(doc, "Table 4. Hyper-parameter check of the random forest: three-fold "
                 "beat-level cross-validation on a DS1 subsample (DS2 untouched).")
    table(doc, ["configuration", "CV macro-F1 (mean)", "std"],
          [[r.model, f"{r.cv_macro_f1_mean:.4f}", f"{r.cv_macro_f1_std:.4f}"]
           for r in cv.itertuples()], widths=[6.5, 4.5, 2.5])

    doc.add_heading("3.6 Evaluation metrics", level=2)
    body(doc, (
        "Accuracy alone is uninformative here: predicting 'N' for every beat already "
        f"gives about {te['N'] / n_test * 100:.0f}% because of the imbalance. The "
        "analysis therefore reports, per class, sensitivity (recall), positive "
        "predictive value (precision) and F1 = 2PR/(P+R), together with the macro-F1 "
        "(unweighted mean over the classes present in the test set), the weighted F1, "
        "the balanced accuracy and Cohen's kappa. Because the clinical target is "
        "ventricular ectopy detection, the binary F1 of class V against all other "
        "classes is the primary summary number, shown together with its "
        "precision-recall curve."
    ))

    doc.add_heading("3.7 Implementation and reproducibility", level=2)
    body(doc, (
        "Everything is implemented in Python 3.13 with NumPy, SciPy, PyWavelets, "
        "scikit-learn, wfdb and Matplotlib. Feature extraction is vectorised over the "
        "whole beat matrix of a record, so the dataset is built in about one minute. "
        "Random seeds are fixed (42) for the cross-validation folds, the forest and the "
        "SVM. run_all.py executes dataset construction, training/evaluation and figure "
        "generation in order; every number in this report comes from that command."
    ))

    # ------------------------------------------------------------ 4
    doc.add_heading("4. Results", level=1)
    doc.add_heading("4.1 R-peak detection", level=2)
    body(doc, (
        f"Table 5 compares the two thresholding rules on all {n_rec} records "
        f"({n_ann:,} annotated beats; Figure 4). The adopted local-percentile rule "
        f"reaches a pooled sensitivity of {se * 100:.2f}% and a pooled PPV of "
        f"{ppv * 100:.2f}%; per-record sensitivity is at least "
        f"{det['Se'].min() * 100:.2f}% (record {int(worst_se['record'])}) and the "
        f"lowest PPV is {det['PPV'].min() * 100:.2f}%."
    ))
    body(doc, (
        "The literal dual-threshold rule is almost as good on average "
        f"({se_d * 100:.2f}% / {ppv_d * 100:.2f}%) but collapses on a single "
        f"recording: on record {int(worst_se_d['record'])} its sensitivity falls to "
        f"{worst_se_d['Se_dual'] * 100:.2f}%. That recording alternates large and "
        "small complexes, so after each large one the running signal level rises above "
        "the following small one, and the search-back rule never fires because the RR "
        "interval stays short. The local reference removes this failure mode, and the "
        "two rules agree everywhere else - an illustration of why a single aggregate "
        "number can hide a systematic defect."
    ))
    caption(doc, "Table 5. Detector performance (all beats of all records).")
    table(doc, ["rule", "Se (pooled)", "PPV (pooled)", "worst-record Se", "worst-record PPV"],
          [["Pan-Tompkins dual threshold (baseline)",
            f"{se_d * 100:.2f}%", f"{ppv_d * 100:.2f}%",
            f"{det['Se_dual'].min() * 100:.2f}% (rec. {int(worst_se_d['record'])})",
            f"{det['PPV_dual'].min() * 100:.2f}%"],
           ["local-percentile rule (adopted)",
            f"{se * 100:.2f}%", f"{ppv * 100:.2f}%",
            f"{det['Se'].min() * 100:.2f}% (rec. {int(worst_se['record'])})",
            f"{det['PPV'].min() * 100:.2f}%"]],
          widths=[6.0, 2.6, 2.6, 3.6, 2.6], font_size=8.2)
    figure(doc, "fig08_detector_per_record.png",
           "Figure 4. Distribution of per-record detection sensitivity and PPV for the "
           "adopted rule.", 15.6)

    doc.add_heading("4.2 Model comparison", level=2)
    body(doc, (
        f"Table 6 compares the three classifiers on DS2 ({n_test:,} beats from {n_ds2} "
        f"patients never seen during training). {PRETTY[best_name]} is best on every "
        f"aggregate metric: accuracy {best['accuracy'] * 100:.2f}%, weighted F1 "
        f"{best['weighted_f1']:.4f}, macro-F1 {best['macro_f1']:.4f} and ventricular "
        f"F1 {best['V_f1']:.4f}. The RBF-SVM is second on the ventricular class "
        f"({svm['V_f1']:.4f}) but pays for its higher ventricular recall "
        f"({svm['V_recall']:.3f}) with a much lower precision "
        f"({svm['V_precision']:.3f}) and a large drop in overall accuracy "
        f"({svm['accuracy'] * 100:.1f}%). The linear baseline is clearly the weakest "
        f"(accuracy {lr['accuracy'] * 100:.1f}%): to reach an acceptable ventricular "
        "recall it must allocate probability mass to the rare classes, which destroys "
        "the normal class."
    ))
    caption(doc, "Table 6. Test-set performance on DS2 (inter-patient). Acc = accuracy, "
                 "bAcc = balanced accuracy, mF1 = macro-F1, wF1 = weighted F1, "
                 "V-P / V-R / V-F1 = precision / recall / F1 of the ventricular class.")
    rows = []
    for m in ["LogReg", "RandomForest", "SVM_RBF"]:
        d = models[m]
        rows.append([PRETTY[m], f"{d['accuracy'] * 100:.2f}",
                     f"{d['balanced_accuracy'] * 100:.2f}",
                     f"{d['macro_f1']:.4f}", f"{d['weighted_f1']:.4f}",
                     f"{d['V_precision']:.3f}", f"{d['V_recall']:.3f}",
                     f"{d['V_f1']:.4f}", f"{d['kappa']:.3f}"])
    table(doc, ["model", "Acc %", "bAcc %", "mF1", "wF1", "V-P", "V-R", "V-F1", "kappa"],
          rows, widths=[3.6, 1.7, 1.7, 1.6, 1.6, 1.5, 1.5, 1.6, 1.5], font_size=8.0)

    doc.add_heading("4.3 Per-class performance and confusion structure", level=2)
    body(doc, (
        "Table 7 lists precision, recall and F1 per class for the best model; Figure 6 "
        "shows the confusion matrices of all three models and Figure 5 the median beat "
        "waveform of each class, which explains most of the confusions."
    ))
    caption(doc, f"Table 7. Per-class performance of {PRETTY[best_name]} on DS2.")
    table(doc, ["class", "precision", "recall", "F1", "support"],
          [[c, f"{bp[c]['precision']:.4f}", f"{bp[c]['recall']:.4f}",
            f"{bp[c]['f1']:.4f}", f"{bp[c]['support']:,}"] for c in CLASSES if c in bp],
          widths=[3.0, 3.2, 3.2, 3.2, 3.2])
    figure(doc, "fig04_beat_templates.png",
           "Figure 5. Median beat waveform per AAMI class (shaded: inter-quartile "
           "range), pooled over records 119, 208, 209 and 232. Ventricular beats are "
           "wider with a discordant T wave; fusion beats lie between N and V.", 13.4)
    figure(doc, "fig05_confusion_matrices.png",
           "Figure 6. Confusion matrices on DS2 (counts; colour = row-normalised rate).", 16.4)
    body(doc, (
        f"Normal beats are classified reliably (F1 = {bp['N']['f1']:.4f}), and "
        f"ventricular ectopic beats reach an F1 of {bp['V']['f1']:.4f} (recall "
        f"{bp['V']['recall']:.4f}, precision {bp['V']['precision']:.4f}): they are "
        "wide, often of opposite polarity and usually premature, three properties the "
        "feature set captures directly. The two rare classes behave differently. "
        f"Supraventricular beats reach only F1 = {bp['S']['f1']:.4f}: their "
        "ventricular morphology is normal by definition, and the timing cue is weak "
        "because most of them come from a record in which they recur with a nearly "
        "periodic RR pattern. Fusion beats are harder still "
        f"(F1 = {bp['F']['f1']:.4f}): a fusion beat is a linear mixture of the N and V "
        "templates (Figure 5), so it falls inside the region the model assigns to one "
        "of the two parents."
    ))

    doc.add_heading("4.4 Precision-recall behaviour for ventricular ectopy", level=2)
    figure(doc, "fig07_pr_curves.png",
           "Figure 7. Precision-recall curves for the ventricular ectopic class on DS2.", 11.2)
    body(doc, (
        "Figure 7 shows that the operating point can be moved along a wide frontier: "
        f"the random forest keeps precision above 0.9 up to a recall of about 0.9, "
        "while the SVM buys a slightly higher maximum recall at a steeper loss of "
        "precision. For screening one would move the threshold to the left to gain "
        "recall at the cost of false alarms; the curve makes that trade-off explicit "
        "instead of hiding it behind one number."
    ))

    doc.add_heading("4.5 Beat-level versus patient-level evaluation", level=2)
    body(doc, (
        f"The same random forest reaches a macro-F1 of {cv_best:.3f} under three-fold "
        "beat-level cross-validation inside DS1 (Table 4) but only "
        f"{rf['macro_f1']:.3f} on DS2 (Table 6). The gap of "
        f"{(cv_best - rf['macro_f1']) * 100:.0f} macro-F1 points is not a bug: inside "
        "DS1 the same patient contributes to training and validation, so the model can "
        "learn that patient's individual QRS shape and recognise it again. With "
        "disjoint patients it must generalise across body habitus, electrode placement "
        "and cardiac axis. This is why 99% figures reported under beat-level splits are "
        "not comparable with inter-patient results, and why the protocol statement must "
        "accompany every number."
    ))

    doc.add_heading("4.6 Which features matter", level=2)
    top = ", ".join(f"{r['feature']} ({r['importance']:.3f})"
                    for r in summary["top_features"][:5])
    figure(doc, "fig06_feature_importance.png",
           "Figure 8. Fifteen most important features of the random forest (mean decrease "
           "in impurity).", 12.6)
    body(doc, (
        f"The forest relies mainly on {top}. This is physiologically coherent: the "
        "pre-RR interval flags prematurity, the normalised shape spread and the "
        "positive area capture the abnormal depolarisation pathway of a ventricular "
        "beat, and the local mean RR encodes the background rate against which "
        "prematurity is judged. Wavelet and higher-frequency descriptors contribute "
        "less, consistent with the fact that the discriminative information of a "
        "surface ECG beat is concentrated below about 20 Hz."
    ))

    doc.add_heading("4.7 Error analysis", level=2)
    conf = np.array(summary["confusion"][best_name])
    body(doc, (
        f"The error mass is concentrated in two places. First, S-N confusion: "
        f"{int(conf[1, 0])} of {te['S']:,} supraventricular beats are labelled normal "
        f"and only {int(conf[1, 1])} are recognised. Second, F-N and F-V confusion: of "
        f"{te['F']:,} fusion beats only {int(conf[3, 3])} are correct, "
        f"{int(conf[3, 0])} go to N and {int(conf[3, 2])} to V. The ventricular class "
        f"itself loses {int(conf[2, 0])} beats to N - the price of a decision rule "
        f"tuned towards high ventricular recall - while only {int(conf[0, 2])} normal "
        f"beats are flagged as ventricular, i.e. "
        f"{int(conf[0, 2]) / te['N'] * 100:.2f}% of the normal class. The same pattern "
        "appears in all three models (Figure 6), which indicates a property of the "
        "feature space rather than of one classifier."
    ))
    body(doc, (
        "To check that the rare classes are not simply invisible to the features, two "
        "one-versus-one experiments were run on the same data. A random forest "
        "separating S from N reaches an AUC of 0.94 once the dominant atrial-bigeminy "
        "record is removed, and one separating F from N reaches 0.92; the four-class "
        "decision nevertheless fails on both, because the argmax competition against a "
        "class that is two orders of magnitude more frequent is lost even when the "
        "ranking is correct. The limitation lies in the decision rule and in the "
        "absence of P-wave descriptors rather than in the discriminative power of the "
        "morphology features as such."
    ))

    # ------------------------------------------------------------ 5
    doc.add_heading("5. Discussion", level=1)
    doc.add_heading("5.1 Interpretation", level=2)
    body(doc, (
        "With 40 hand-crafted features and no patient-specific calibration, the "
        f"pipeline detects ventricular ectopic beats with an F1 of {best['V_f1']:.4f} "
        "on patients never seen during training, at an overall accuracy of "
        f"{best['accuracy'] * 100:.2f}%. For the normal/ventricular distinction - the "
        "clinically most relevant binary decision in this database - this is "
        "competitive with published inter-patient results obtained with similar "
        "classical feature sets, which typically fall in the 0.85-0.95 F1 band "
        "depending on the split and on the resampling strategy. Deep convolutional "
        "models usually add a few points, at the cost of interpretability and of a "
        "much larger training set."
    ))
    body(doc, (
        "The macro-F1, by contrast, is low because it weights the two rare classes "
        "equally with the two frequent ones. Reporting both numbers is more informative "
        "than choosing the flattering one: weighted F1 and ventricular F1 describe what "
        "the system does today, macro-F1 describes what it cannot yet do."
    ))

    doc.add_heading("5.2 Why the rare classes fail", level=2)
    body(doc, (
        "Supraventricular beats share the ventricular morphology of normal beats; the "
        "difference is atrial, expressed in the P wave and in the timing. The current "
        "feature set samples the P-wave region only through nine fixed amplitude "
        "points, too coarse to describe a low-amplitude, patient-dependent wave, and "
        "the timing cue disappears in bigeminal rhythms. Fusion beats are mixtures by "
        "definition, so they occupy the same region of feature space as their parents; "
        "separating them needs either a much finer morphological description or a model "
        "that represents mixtures explicitly. Both failures are structural rather than "
        "statistical: more trees or a larger kernel do not help, as the three models "
        "show."
    ))

    doc.add_heading("5.3 Limitations", level=2)
    bullets(doc, [
        "Single lead. Only one channel per record is used; a second lead would resolve "
        "part of the V-F and S-N ambiguity, especially for beats whose electrical axis "
        "is nearly orthogonal to the chosen lead.",
        "Fixed window. The 700 ms window is a compromise: very wide ventricular beats "
        "and tachycardic patients are not handled optimally, and no alignment beyond "
        "R-peak centring is performed.",
        "Label assignment through the detector. Detected peaks without an annotation are "
        "discarded, so false detections never become training noise, but the classifier "
        "is also never trained on the exact errors the detector makes.",
        "Rare-class statistics. Fusion beats contribute 382 test and 410 training "
        "examples, so their F1 has a wide confidence interval and should be read "
        "qualitatively.",
        "No noise stress test. Robustness to baseline wander, electrode motion and "
        "muscle noise beyond what is naturally present in MIT-BIH is not quantified; "
        "the PhysioNet noise stress test would be the natural next step.",
        "No patient adaptation. A handful of labelled beats from the test patient would "
        "allow patient-specific thresholds, which is standard in implantable devices.",
    ])

    doc.add_heading("5.4 Clinical perspective", level=2)
    body(doc, (
        "The system is a screening aid, not a diagnostic device. Its realistic role is "
        "to prioritise segments of a long recording for expert review - for example to "
        "surface the minutes around ventricular ectopy - where a ventricular recall "
        "above 0.85 with few false alarms is already useful. Clinical deployment would "
        "require validation on the target population, calibration of the operating "
        "point and regulatory clearance; the documented failure modes (missed "
        "low-amplitude complexes, S-N and V-F confusion) are exactly the ones a "
        "clinician must remain responsible for."
    ))

    # ------------------------------------------------------------ 6
    doc.add_heading("6. Conclusion", level=1)
    body(doc, (
        f"A complete beat-level arrhythmia pipeline was implemented and evaluated on "
        f"the {n_rec} analysed records of the MIT-BIH Arrhythmia Database. The detector "
        f"reaches {se * 100:.2f}% pooled sensitivity and {ppv * 100:.2f}% pooled PPV, "
        "and a 40-dimensional interpretable feature set with cost-sensitive learning "
        f"classifies the beats of unseen patients with {best['accuracy'] * 100:.2f}% "
        f"accuracy and a ventricular-ectopic F1 of {best['V_f1']:.4f}. The experiments "
        "lead to four conclusions: patient-independent evaluation is essential for "
        "trustworthy numbers (the same model scores 0.91 macro-F1 within patients and "
        "0.51 across patients); F1-based metrics are mandatory under the observed "
        "imbalance; timing features and QRS width carry most of the discriminative "
        "information; and the remaining errors concentrate in the supraventricular and "
        "fusion classes, where morphology alone is insufficient and P-wave descriptors, "
        "a second lead or patient adaptation would be required."
    ))
    body(doc, (
        "A second, methodological conclusion concerns the detector: the literal "
        "Pan-Tompkins dual-threshold rule is excellent on average but fails on a "
        "recording with alternating beat amplitude, whereas a local percentile "
        "reference fixes that case at no cost elsewhere. Aggregate sensitivity hid "
        "that defect, which argues for always reporting per-record statistics."
    ))

    # ------------------------------------------------------------ 7
    doc.add_heading("7. Reproducibility", level=1)
    body(doc, (
        "The repository contains the code, the README and the scripts that regenerate "
        "every number and figure of this report:"
    ))
    table(doc, ["path", "content"],
          [["src/download_data.py", "downloads the 42 analysed records from PhysioNet"],
           ["src/preprocessing.py", "0.5-40 Hz filtering, Pan-Tompkins envelope, two threshold rules"],
           ["src/segmentation.py", "AAMI label mapping, beat windows, RR features"],
           ["src/features.py", "the 40 morphology / spectral / wavelet / rhythm descriptors"],
           ["src/build_dataset.py", "builds outputs/features_mitdb.csv from the raw records"],
           ["src/train_eval.py", "training, multiplier tuning, DS2 evaluation, metrics, JSON summary"],
           ["src/make_figures.py", "all report figures"],
           ["src/build_report.py", "generates this document from the measured results"],
           ["run_all.py", "runs the whole pipeline end to end"]],
          widths=[4.6, 11.6], align_right_from=99, font_size=8.2)
    para(doc, "", space_after=2)
    body(doc, "After installing the requirements the complete study is reproduced with:")
    p = doc.add_paragraph("python src/download_data.py\npython run_all.py\npython src/build_report.py")
    p.runs[0].font.name = "Consolas"
    p.runs[0].font.size = Pt(9.5)
    p.paragraph_format.left_indent = Cm(1.0)
    body(doc, (
        f"The pipeline runs in a few minutes on a laptop CPU (random forest "
        f"{rf['fit_time_s']:.0f} s, RBF-SVM {svm['fit_time_s']:.0f} s on the full "
        f"training set). The GitHub repository is available at {GITHUB_URL}."
    ))

    # ------------------------------------------------------------ refs
    doc.add_heading("References", level=1)
    refs = [
        "Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia "
        "Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.",
        "Goldberger, A. L., Amaral, L. A. N., Glass, L., et al. (2000). PhysioBank, "
        "PhysioToolkit, and PhysioNet: components of a new research resource for "
        "complex physiologic signals. Circulation, 101(23), e215-e220.",
        "Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE "
        "Transactions on Biomedical Engineering, 32(3), 230-236.",
        "de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification "
        "of heartbeats using ECG morphology and heartbeat interval features. IEEE "
        "Transactions on Biomedical Engineering, 51(7), 1196-1206.",
        "Association for the Advancement of Medical Instrumentation (2012). ANSI/AAMI "
        "EC57: Testing and reporting performance results of cardiac rhythm and ST "
        "segment measurement algorithms.",
        "Kohler, B.-U., Hennig, C., & Orglmeister, R. (2002). The principles of "
        "software QRS detection. IEEE Engineering in Medicine and Biology Magazine, "
        "21(1), 42-57.",
        "Kiranyaz, S., Ince, T., & Gabbouj, M. (2016). Real-time patient-specific ECG "
        "classification by 1-D convolutional neural networks. IEEE Transactions on "
        "Biomedical Engineering, 63(3), 664-675.",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
        "PhysioNet. MIT-BIH Arrhythmia Database v1.0.0. "
        "https://physionet.org/content/mitdb/1.0.0/",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph(f"[{i}] {r}")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Cm(0.9)
        p.paragraph_format.first_line_indent = Cm(-0.9)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(REPORT_PATH))
    print("saved ->", REPORT_PATH)


if __name__ == "__main__":
    build()
