"""Train and evaluate beat classifiers with an inter-patient (DS1 -> DS2) protocol.

Models
    * Logistic regression : linear baseline
    * Random forest       : non-linear ensemble, feature importances
    * SVM (RBF)           : strong classical classifier on standardised features

Because the classes are extremely unbalanced, the decision rule is not the plain
argmax of the predicted probabilities. For each model a vector of class
multipliers is tuned on a *patient-disjoint* validation half of DS1 (the records
of DS1 are divided in two halves by patient), maximising macro-F1 there; the
tuned multipliers are then applied to DS2. DS2 is never used for any tuning
decision, and the DS1/DS2 split of de Chazal et al. (2004) guarantees that no
patient appears in both training and test sets.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASSES, DS1, OUT_DIR, RANDOM_STATE
from features import all_feature_names

GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]


def load_data():
    df = pd.read_csv(OUT_DIR / "features_mitdb.csv")
    feats = all_feature_names()
    train = df[df["split"] == "DS1"]
    test = df[df["split"] == "DS2"]
    return (train[feats].to_numpy(), train["label_idx"].to_numpy(),
            test[feats].to_numpy(), test["label_idx"].to_numpy(), feats, df)


def build_models() -> dict:
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                       n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced",
                        probability=True, cache_size=1000, random_state=RANDOM_STATE)),
        ]),
    }


def apply_multipliers(proba: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.argmax(proba * w[None, :], axis=1)


def tune_multipliers(proba: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Grid-search class multipliers maximising macro-F1 (N multiplier fixed to 1)."""
    best_w = np.ones(len(CLASSES))
    best_f1 = -1.0
    for combo in itertools.product(GRID, repeat=len(CLASSES) - 1):
        w = np.array([1.0, *combo])
        f1 = f1_score(y, apply_multipliers(proba, w), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, float(best_f1)


def main() -> None:
    Xtr, ytr, Xte, yte, feats, df = load_data()
    present = [c for c in range(len(CLASSES)) if (yte == c).sum() > 0]
    target_names = [CLASSES[c] for c in present]

    print(f"train beats {Xtr.shape[0]}  test beats {Xte.shape[0]}  features {Xtr.shape[1]}")
    print("train dist:", {CLASSES[c]: int((ytr == c).sum()) for c in range(len(CLASSES))})
    print("test  dist:", {CLASSES[c]: int((yte == c).sum()) for c in range(len(CLASSES))})

    # ---- patient-disjoint validation halves of DS1 ------------------------
    tune_records = set(DS1[0::2])
    fit_records = set(DS1) - tune_records
    rec_tr = df.loc[df["split"] == "DS1", "record"].to_numpy()
    m_tune = np.isin(rec_tr, list(tune_records))
    Xa, ya = Xtr[~m_tune], ytr[~m_tune]     # fitting half
    Xb, yb = Xtr[m_tune], ytr[m_tune]       # tuning half (patients never seen by Xa)

    summary: dict = {
        "n_features": int(Xtr.shape[1]),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "train_dist": {CLASSES[c]: int((ytr == c).sum()) for c in range(len(CLASSES))},
        "test_dist": {CLASSES[c]: int((yte == c).sum()) for c in range(len(CLASSES))},
        "val_fit_records": sorted(fit_records),
        "val_tune_records": sorted(tune_records),
        "models": {},
    }
    print(f"validation: fit on {len(fit_records)} DS1 records, "
          f"tune on {len(tune_records)} DS1 records")

    # ---- light hyper-parameter sanity check (3-fold CV on DS1 only) -------
    cv_rows = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    sub = np.random.default_rng(RANDOM_STATE).choice(
        Xtr.shape[0], size=min(15000, Xtr.shape[0]), replace=False)
    for leaf in (1, 2, 4):
        rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=leaf,
                                    class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=RANDOM_STATE)
        s = cross_val_score(rf, Xtr[sub], ytr[sub], cv=skf, scoring="f1_macro", n_jobs=1)
        cv_rows.append({"model": f"RF(min_samples_leaf={leaf})",
                        "cv_macro_f1_mean": round(float(s.mean()), 4),
                        "cv_macro_f1_std": round(float(s.std()), 4)})
        print(f"CV RF leaf={leaf}: macro-F1 {s.mean():.4f} +/- {s.std():.4f}")
    pd.DataFrame(cv_rows).to_csv(OUT_DIR / "cv_model_selection.csv", index=False)

    rows, per_class_all, cms, curves = [], {}, {}, {}
    for name, model in build_models().items():
        # 1) tune the class multipliers on the patient-disjoint DS1 half
        model.fit(Xa, ya)
        w, val_f1 = tune_multipliers(model.predict_proba(Xb), yb)
        base = f1_score(yb, model.predict(Xb), average="macro", zero_division=0)
        print(f"\n[{name}] multipliers {dict(zip(CLASSES, np.round(w, 2)))} "
              f"-> validation macro-F1 {base:.4f} (argmax) vs {val_f1:.4f} (tuned)")

        # 2) refit on all of DS1 and evaluate once on DS2
        t0 = time.time()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        yp = apply_multipliers(proba, w)
        fit_s = time.time() - t0

        acc = accuracy_score(yte, yp)
        bacc = balanced_accuracy_score(yte, yp)
        macro_f1 = f1_score(yte, yp, labels=present, average="macro", zero_division=0)
        weighted_f1 = f1_score(yte, yp, labels=present, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(yte, yp)

        per = pd.DataFrame(classification_report(
            yte, yp, labels=present, target_names=target_names, output_dict=True)).T
        per.to_csv(OUT_DIR / f"per_class_{name}.csv")
        cms[name] = confusion_matrix(yte, yp, labels=list(range(len(CLASSES)))).tolist()

        print(f"\n=== {name} ({fit_s:.1f}s) ===")
        print(classification_report(yte, yp, labels=present, target_names=target_names,
                                    zero_division=0, digits=4))

        v = CLASSES.index("V")
        yt_v, yp_v = (yte == v).astype(int), (yp == v).astype(int)
        row = {
            "model": name,
            "fit_time_s": round(fit_s, 1),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "kappa": kappa,
            "V_precision": precision_score(yt_v, yp_v, zero_division=0),
            "V_recall": recall_score(yt_v, yp_v, zero_division=0),
            "V_f1": f1_score(yt_v, yp_v, zero_division=0),
            "validation_macro_f1": val_f1,
            "multipliers": {CLASSES[c]: float(w[c]) for c in range(len(CLASSES))},
        }
        per_class_all[name] = {
            CLASSES[c]: {
                "precision": float(per.loc[CLASSES[c], "precision"]),
                "recall": float(per.loc[CLASSES[c], "recall"]),
                "f1": float(per.loc[CLASSES[c], "f1-score"]),
                "support": int(per.loc[CLASSES[c], "support"]),
            } for c in present
        }
        rows.append(row)

        # precision-recall data for the ventricular-ectopic class
        score = proba[:, list(model.classes_).index(v)]
        prec, rec, _ = precision_recall_curve((yte == v).astype(int), score)
        step = max(1, len(rec) // 500)
        curves[name] = {"recall": rec[::step].tolist(), "precision": prec[::step].tolist()}

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    summary["models"] = {
        r["model"]: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in r.items()} for r in rows}
    summary["per_class"] = per_class_all
    summary["confusion"] = cms
    summary["pr_curves"] = curves

    rf = build_models()["RandomForest"]
    rf.fit(Xtr, ytr)
    imp = pd.DataFrame({"feature": feats,
                        "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    summary["top_features"] = imp.head(15).to_dict("records")

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== model comparison ===")
    print(metrics[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                   "V_precision", "V_recall", "V_f1"]].to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
