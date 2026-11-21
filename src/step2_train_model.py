"""Smart Stadium Canopy — Weather-only rain AI, walk-forward training.

Run from the project root:
    python src/step2_train_rf.py

This script REPLACES the former step2_train_rf.py. No new training scripts
or duplicate model files are created. It preserves the existing .pkl name
for compatibility; consult JSON metadata for the actual algorithm.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/kl_stadium_training_clean.csv")
FOLDS_PATH = Path("data/walk_forward_folds.csv")
MODEL_PATH = Path("models/rf_rain_predictor.pkl")  # existing filename
METADATA_PATH = Path("models/rain_predictor_metadata.json")
DETAILS_PATH = Path("models/model_validation_results.csv")
COMPARISON_PATH = Path("data/model_comparison_results.csv")

TEST_START = pd.Timestamp("2026-05-22 22:00:00")
HORIZON = pd.Timedelta(hours=1)
TARGET = "target_rain_next_1h"
FEATURES = [
    "temperature_2m", "relative_humidity_2m", "rain_mm",
    "cloud_cover", "wind_speed_10m", "is_day",
    "shortwave_radiation", "direct_radiation",
    "temperature_diff_1h", "humidity_diff_1h",
    "wind_diff_1h", "cloud_diff_1h",
    "shortwave_radiation_diff_1h", "rain_lag_1h",
    "rain_roll_3h", "humidity_roll_3h", "hour", "month",
]


def build_models():
    # Only scikit-learn: no extra installations needed yet.
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            ),
        ),
        "Gaussian NB": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=160, max_depth=14, min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=160, max_depth=18, min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
        "HistGBM": HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=42,
        ),
        "HistGBM Regularized": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=2.0,
            random_state=42,
        ),
    }


def rain_scores(model, X):
    rain_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, rain_index]


def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y, pred),
        "rain_precision": precision_score(y, pred, zero_division=0),
        "rain_recall": recall_score(y, pred, zero_division=0),
        "rain_f1": f1_score(y, pred, zero_division=0),
        "rain_f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "false_negatives": int(cm[1, 0]),
        "false_positives": int(cm[0, 1]),
    }


def main():
    if not DATA_PATH.exists() or not FOLDS_PATH.exists():
        raise FileNotFoundError(
            "Run step1_clean_data.py and step5_prepare_time_folds.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    missing = set(["time", TARGET] + FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing feature/target values found")
    if df["time"].duplicated().any():
        raise ValueError("Duplicate timestamps found")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary 0/1")

    folds_file = pd.read_csv(FOLDS_PATH)
    required = {"fold", "train_end", "train_rows", "val_start", "val_end", "val_rows"}
    if not required.issubset(folds_file.columns):
        raise ValueError("walk_forward_folds.csv has unexpected columns")

    folds = []
    for row in folds_file.itertuples(index=False):
        train_end = pd.Timestamp(row.train_end)
        val_start = pd.Timestamp(row.val_start)
        val_end = pd.Timestamp(row.val_end)
        if val_end >= TEST_START:
            raise ValueError("A validation fold overlaps the reserved test set")
        train = df[(df["time"] <= train_end) &
                   (df["time"] + HORIZON < val_start)]
        valid = df[df["time"].between(val_start, val_end)]
        if len(train) != row.train_rows or len(valid) != row.val_rows:
            raise ValueError(f"Fold {row.fold} no longer matches dataset; rerun step5")
        if train.empty or valid.empty or (
            train["time"].max() + HORIZON >= valid["time"].min()
        ):
            raise ValueError(f"Time leakage / empty data in fold {row.fold}")
        if train[TARGET].nunique() != 2 or valid[TARGET].nunique() != 2:
            raise ValueError(f"Fold {row.fold} needs both target classes")
        folds.append((row.fold, train, valid))

    print(f"\nDataset: {len(df):,} rows | Validation folds: {len(folds)}")
    print("Reserved Test starts:", TEST_START, "(not used for model selection)")

    summaries, detailed = [], []
    for name, model in build_models().items():
        print(f"\nTraining {name} across {len(folds)} folds ...", flush=True)
        fold_predictions = []
        for fold_name, train, valid in folds:
            model.fit(train[FEATURES], train[TARGET])
            scores = rain_scores(model, valid[FEATURES])
            fold_predictions.append((fold_name, valid[TARGET].to_numpy(), scores))

        # One shared threshold for each model, selected using validation ONLY.
        # Maximize mean Rain F2 across seasons (recall is weighted higher).
        best_threshold, best_f2 = 0.50, -1.0
        for threshold in np.round(np.arange(0.05, 0.901, 0.025), 3):
            fold_f2 = [
                fbeta_score(y, scores >= threshold, beta=2, zero_division=0)
                for _, y, scores in fold_predictions
            ]
            mean_f2 = float(np.mean(fold_f2))
            if mean_f2 > best_f2:
                best_threshold, best_f2 = float(threshold), mean_f2

        fold_metrics = []
        for fold_name, y, scores in fold_predictions:
            m = metrics(y, scores, best_threshold)
            detailed.append({"model": name, "fold": fold_name,
                             "threshold": best_threshold, **m})
            fold_metrics.append(m)

        summary = {
            "model": name,
            "threshold": best_threshold,
            "mean_rain_f2": float(np.mean([m["rain_f2"] for m in fold_metrics])),
            "mean_rain_f1": float(np.mean([m["rain_f1"] for m in fold_metrics])),
            "mean_rain_recall": float(np.mean([m["rain_recall"] for m in fold_metrics])),
            "worst_fold_recall": float(min(m["rain_recall"] for m in fold_metrics)),
            "mean_rain_precision": float(np.mean([m["rain_precision"] for m in fold_metrics])),
            "mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics])),
            "mean_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
            "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        }
        summaries.append(summary)
        print(f"  Mean F2={summary['mean_rain_f2']:.3f}  "
              f"Mean Recall={summary['mean_rain_recall']:.3f}  "
              f"Worst Recall={summary['worst_fold_recall']:.3f}  "
              f"Threshold={best_threshold:.3f}", flush=True)

    results = pd.DataFrame(summaries).sort_values(
        ["mean_rain_f2", "worst_fold_recall", "mean_pr_auc"],
        ascending=False,
    ).reset_index(drop=True)
    print("\n========== WALK-FORWARD MODEL COMPARISON ==========")
    print(results.round(3).to_string(index=False))

    winner = results.iloc[0]
    chosen_name = winner["model"]
    threshold = float(winner["threshold"])

    # Refit chosen algorithm on every eligible pre-test row. The last
    # training label must occur BEFORE the test period begins.
    train_final = df[df["time"] + HORIZON < TEST_START]
    test = df[df["time"] >= TEST_START]
    if train_final.empty or test.empty:
        raise ValueError("Final training or test set is empty")
    best_model = build_models()[chosen_name]
    best_model.fit(train_final[FEATURES], train_final[TARGET])
    test_scores = rain_scores(best_model, test[FEATURES])
    test_metrics = metrics(test[TARGET], test_scores, threshold)

    print(f"\n========== HOLDOUT REPORT (ONE-TIME CHECK) ==========")
    print(f"Model: {chosen_name} | threshold={threshold:.3f}")
    for key, value in test_metrics.items():
        print(f"{key:20s}: {value:.4f}" if isinstance(value, float)
              else f"{key:20s}: {value}")
    print("NOTE: This test period was evaluated in previous versions. "
          "Use newly collected future data for a truly untouched final test.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    metadata = {
        "model_name": chosen_name,
        "pipeline_version": "weather_only_walk_forward_v2",
        "threshold": threshold,
        "features": FEATURES,
        "target": TARGET,
        "rain_threshold_mm": 0.1,
        "selection_metric": "mean rain F2 across 5 walk-forward folds",
        "train_end_time": str(train_final["time"].max()),
        "test_start_time": str(test["time"].min()),
        "test_end_time": str(test["time"].max()),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    results.to_csv(COMPARISON_PATH, index=False)
    pd.DataFrame(detailed).to_csv(DETAILS_PATH, index=False)

    print("\nUpdated existing files (no duplicate models):")
    for p in [MODEL_PATH, METADATA_PATH, COMPARISON_PATH, DETAILS_PATH]:
        print(" -", p)
    print("Step 6 complete!")


if __name__ == "__main__":
    main()
