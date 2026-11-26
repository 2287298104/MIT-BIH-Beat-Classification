# Beat-Level Arrhythmia Detection from Single-Lead ECG

Course project for **Biomedical Signal Processing, Fall 2026** (Huaqiao University).

**Author:** Feng Runhang (2315104007) · **Instructor:** Dr. DUAN Fang<br>
**Repository:** https://github.com/2287298104/MIT-BIH-Beat-Classification

A complete, reproducible ECG signal-processing pipeline on the **MIT-BIH Arrhythmia
Database**: Pan-Tompkins R-peak detection → beat segmentation → 40 hand-crafted
features → classical classifiers, evaluated with the **inter-patient DS1/DS2
protocol** of de Chazal et al. (2004) and the **AAMI EC57** super-class taxonomy.

**Research question.** How well can a lightweight, fully interpretable classical
pipeline — hand-crafted features plus tree/kernel classifiers, trained on one group
of patients and tested on disjoint patients — detect and classify ventricular
ectopic beats, and where do its errors come from?

## Headline results

| stage | result |
|---|---|
| R-peak detection (42 records, 97,075 annotated beats) | Se **99.26 %**, PPV **99.73 %** (pooled) |
| Beat classification (DS2, 45,791 beats from 22 unseen patients) | accuracy **93.04 %**, weighted F1 **0.919**, **V-F1 0.906**, N-F1 0.963 |
| rare classes | S-F1 0.143, F-F1 0.014 |
| same model, beat-level CV inside DS1 | macro-F1 **0.907** vs **0.507** inter-patient |

The gap between 0.907 and 0.507 is the main finding: most of the reported difficulty
of MIT-BIH is caused by patient-independent evaluation, not by the classifier.

## Quick start

```bash
pip install -r requirements.txt

python src/download_data.py     # ~90 MB from PhysioNet (42 of the 48 records are used)
python run_all.py               # features -> train/evaluate -> figures
python src/build_report.py      # rebuild report/Course_Project_Report.docx
```

`run_all.py --skip-data` reuses an existing `outputs/features_mitdb.csv`.
Runtime on a laptop CPU: about 1 minute for the dataset, 7 s for the random forest,
~2.5 minutes for the Platt-scaled RBF-SVM.

## Repository layout

| path | content |
|---|---|
| `src/config.py` | paths, sampling rate, record lists (DS1/DS2), AAMI mapping, window sizes |
| `src/download_data.py` | downloads the analysed records from PhysioNet |
| `src/preprocessing.py` | 0.5-40 Hz zero-phase band-pass, Pan-Tompkins envelope, two threshold rules, peak matching |
| `src/segmentation.py` | AAMI label mapping, beat windows, RR-interval features |
| `src/features.py` | the 40 morphology / spectral / wavelet / rhythm descriptors |
| `src/build_dataset.py` | per-record pipeline → `outputs/features_mitdb.csv` |
| `src/train_eval.py` | trains LR / RF / SVM, tunes the decision rule on a DS1 validation half, evaluates on DS2 |
| `src/make_figures.py` | renders every figure into `outputs/figures/` |
| `src/build_report.py` | builds the Word report from the measured results |
| `run_all.py` | end-to-end runner |
| `outputs/` | feature table, metrics, per-class reports, `summary.json`, `figures/` |
| `report/` | the course report (`Course_Project_Report.docx`) |

## Method

1. **Preprocessing** — Butterworth band-pass 0.5-40 Hz applied forward and backward
   (`sosfiltfilt`), so no phase shift displaces the R peak.
2. **R-peak detection** — Pan & Tompkins (1985): 5-15 Hz band-pass, five-point
   derivative, squaring, 150 ms moving-window integration, thresholding, 200 ms
   refractory period, fiducial refinement to the largest deflection within ±100 ms.
   Two threshold rules are implemented:
   * `detect_r_peaks_dual_threshold` — the literal SPKI/NPKI adaptive rule of the
     paper (used as the baseline in the report);
   * `detect_r_peaks` — the adopted rule: threshold = 12 % of the **local** 90th
     percentile of the envelope in a ±3 s window, plus T-wave suppression. It is
     robust to recordings whose beat amplitude alternates (record 228, where the
     literal rule drops to 29 % sensitivity while the local rule keeps 96 %).
3. **Segmentation** — one window per detected R peak, −250 ms to +450 ms
   (252 samples at 360 Hz), labelled by one-to-one matching to the nearest
   cardiologist annotation (150 ms tolerance).
4. **Features (40)** — 13 morphological descriptors, 9 normalised amplitude samples
   (−200 … +200 ms), 6 spectral descriptors (band powers, entropy, centroid, HF/LF),
   5 db4 level-4 wavelet sub-band energies, 6 rhythm descriptors (pre/post RR, local
   mean RR, RR ratios, |ΔRR|).
5. **Classification** — multinomial logistic regression (linear baseline), random
   forest (300 trees), RBF-SVM. All use `class_weight='balanced'` (no synthetic
   beats), and each model's class multipliers are tuned on a **patient-disjoint**
   validation half of DS1 before being applied to DS2.
6. **Evaluation** — per-class precision / recall / F1, macro-F1, weighted F1,
   balanced accuracy, Cohen's κ, and the binary F1 of the ventricular class with its
   precision-recall curve.

### Data protocol

* 42 of the 48 MIT-BIH records are analysed. Records 102, 104, 107 and 217 are
  excluded (paced rhythms, outside the AAMI class set); 113 and 121 are not part of
  the published DS1/DS2 split that the project follows.
* **DS1** (22 records) for training, **DS2** (22 disjoint records) for testing. No
  patient appears in both sets.
* Lead MLII when present, otherwise the first available lead.

## Outputs

| figure | content |
|---|---|
| `fig01_pipeline.png` | block diagram of the processing chain |
| `fig02_detection_example.png` | detected R peaks vs annotations; segmented beats by class |
| `fig03_class_distribution.png` | class balance of DS1 / DS2 (log scale) |
| `fig04_beat_templates.png` | median beat waveform per AAMI class |
| `fig05_confusion_matrices.png` | confusion matrices of the three models |
| `fig06_feature_importance.png` | 15 most important random-forest features |
| `fig07_pr_curves.png` | precision-recall curves for the ventricular class |
| `fig08_detector_per_record.png` | per-record detector Se and PPV |

Numeric results: `outputs/model_metrics.csv`, `outputs/per_class_<model>.csv`,
`outputs/detection_stats.csv`, `outputs/summary.json`.

## Reproducibility notes

* Python 3.13, NumPy, SciPy, PyWavelets, scikit-learn, wfdb, Matplotlib, python-docx
  (see `requirements.txt`).
* All random operations are seeded (`RANDOM_STATE = 42`): CV folds, forest, SVM.
* Hyper-parameters and decision multipliers were selected without touching DS2.
* Feature extraction is vectorised over the whole beat matrix of a record, which
  keeps the dataset build at about one minute for ~96k beats.

## Data and licence

The MIT-BIH Arrhythmia Database is distributed by PhysioNet under the ODC Attribution
licence (https://physionet.org/content/mitdb/1.0.0/). The data are not
re-distributed here; `src/download_data.py` fetches them.

## References

1. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database.
   *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.
2. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
   *IEEE TBME*, 32(3), 230-236.
3. de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification of
   heartbeats using ECG morphology and heartbeat interval features. *IEEE TBME*,
   51(7), 1196-1206.
4. ANSI/AAMI EC57 (2012). Testing and reporting performance results of cardiac rhythm
   and ST segment measurement algorithms.
5. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
   *Circulation*, 101(23), e215-e220.
