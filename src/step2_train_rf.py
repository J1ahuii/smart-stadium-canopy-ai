
# ============================================================
# AI Canopy System - Rain Prediction Model Training
# Input : data/kl_stadium_training_clean.csv
# Output: models/rf_rain_predictor.pkl
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------
DATA_PATH = "data/kl_stadium_training_clean.csv"
MODEL_PATH = "models/rf_rain_predictor.pkl"
METADATA_PATH = "models/rain_predictor_metadata.json"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

os.makedirs("models", exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}. "
        "Run the clean data script first."
    )

df = pd.read_csv(DATA_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

# ------------------------------------------------------------
# 2. Features matching the cleaning script
# ------------------------------------------------------------
features = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain_mm",
    "cloud_cover",
    "wind_speed_10m",
    "is_day",
    "shortwave_radiation",
    "direct_radiation",
    "temperature_diff_1h",
    "humidity_diff_1h",
    "wind_diff_1h",
    "cloud_diff_1h",
    "shortwave_radiation_diff_1h",
    "rain_lag_1h",
    "rain_roll_3h",
    "humidity_roll_3h",
    "hour",
    "month",
]

target = "target_rain_next_1h"

required_columns = ["time"] + features + [target]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}. "
        "Run the matching clean data script first."
    )

df = df.dropna(subset=features + [target]).copy()
df[target] = df[target].astype(int)

if df["time"].duplicated().any():
    raise ValueError("Duplicate timestamps found.")

if not set(df[target].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

# ------------------------------------------------------------
# 3. Chronological train / validation / test split
# ------------------------------------------------------------
n = len(df)

train_end = int(n * TRAIN_RATIO)
validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()

X_train = train_df[features]
y_train = train_df[target]

X_val = validation_df[features]
y_val = validation_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("\n========== DATA SPLIT ==========")

for name, subset in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:
    print(
        f"{name}: {len(subset):,} rows | "
        f"{subset['time'].min()} to {subset['time'].max()} | "
        f"Rain rate = {subset[target].mean():.3f}"
    )

if any(
    labels.nunique() < 2
    for labels in [y_train, y_val, y_test]
):
    raise ValueError(
        "At least one split contains only one target class. "
        "Review the dataset and split boundaries."
    )

# ------------------------------------------------------------
# 4. Define the seven models
# ------------------------------------------------------------
models = {
    "1. Linear Regression (LPM benchmark)": LinearRegression(),

    "2. Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
    ),

    "3. Gaussian Naive Bayes": GaussianNB(),

    "4. Single Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=RANDOM_SEED,
    ),

    "5. Random Forest (250 Trees)": RandomForestClassifier(
        n_estimators=250,
        max_depth=16,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),

    "6. Gradient Boosting (HistGBM)": HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=10,
        random_state=RANDOM_SEED,
    ),

    "7. Ensemble (RF + HistGBM)": VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=16,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
            (
                "gbm",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    max_depth=10,
                    random_state=RANDOM_SEED,
                ),
            ),
        ],
        voting="soft",
    ),
}

# ------------------------------------------------------------
# 5. Prediction score helper
# ------------------------------------------------------------
def get_rain_probability(model, X):
    """
    Return a score for the Rain class.

    Linear Regression is included only as a benchmark.
    Its clipped output is not a calibrated probability.
    """
    if isinstance(model, LinearRegression):
        return np.clip(model.predict(X), 0.0, 1.0)

    probabilities = model.predict_proba(X)
    class_labels = list(model.classes_)
    rain_column = class_labels.index(1)

    return probabilities[:, rain_column]


def tune_threshold(y_true, scores):
    """
    Select the threshold that maximises Rain-class F1
    on the validation set only.
    """
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (scores >= threshold).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold


def evaluate_scores(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "rain_precision": precision_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "rain_recall": recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "rain_f1": f1_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, scores),
        "predictions": predictions,
    }


# ------------------------------------------------------------
# 6. Train and evaluate on validation set
# ------------------------------------------------------------
trained_models = {}
validation_results = []

for name, model in models.items():
    print(f"\nTraining {name} ...")

    model.fit(X_train, y_train)
    val_scores = get_rain_probability(model, X_val)

    threshold = tune_threshold(y_val, val_scores)
    metrics = evaluate_scores(y_val, val_scores, threshold)

    trained_models[name] = model

    validation_results.append({
        "model": name,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "rain_precision": metrics["rain_precision"],
        "rain_recall": metrics["rain_recall"],
        "rain_f1": metrics["rain_f1"],
        "roc_auc": metrics["roc_auc"],
    })

validation_results_df = pd.DataFrame(validation_results)

validation_results_df = validation_results_df.sort_values(
    by="rain_f1",
    ascending=False,
).reset_index(drop=True)

print("\n========== VALIDATION RESULTS ==========")
print(validation_results_df.to_string(index=False))

# ------------------------------------------------------------
# 7. Select model using validation Rain F1
# ------------------------------------------------------------
best_model_name = validation_results_df.iloc[0]["model"]
best_threshold = float(
    validation_results_df.iloc[0]["threshold"]
)
best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)
print("Selected threshold:", round(best_threshold, 2))

# ------------------------------------------------------------
# 8. Final test evaluation
# ------------------------------------------------------------
# The test set has not been used for model selection or
# threshold tuning.
test_scores = get_rain_probability(best_model, X_test)

test_metrics = evaluate_scores(
    y_test,
    test_scores,
    best_threshold,
)

test_predictions = test_metrics["predictions"]

print("\n========== FINAL TEST RESULTS ==========")
print("Model:", best_model_name)
print("Threshold:", round(best_threshold, 2))
print(f"Accuracy:       {test_metrics['accuracy']:.4f}")
print(f"Rain Precision: {test_metrics['rain_precision']:.4f}")
print(f"Rain Recall:    {test_metrics['rain_recall']:.4f}")
print(f"Rain F1-score:  {test_metrics['rain_f1']:.4f}")
print(f"ROC-AUC:        {test_metrics['roc_auc']:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["No Rain", "Rain"],
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)

print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        cm,
        index=["Actual No Rain", "Actual Rain"],
        columns=["Predicted No Rain", "Predicted Rain"],
    )
)


# ------------------------------------------------------------
# 9. Save model, metadata, and model comparison results
# ------------------------------------------------------------

import json

# Keep the original model file path and format.
joblib.dump(best_model, MODEL_PATH)

metadata = {
    "model_name": best_model_name,
    "threshold": best_threshold,
    "features": features,
    "target": target,
    "rain_threshold_mm": 0.1,

    "train_start_time": str(train_df["time"].min()),
    "train_end_time": str(train_df["time"].max()),

    "validation_start_time": str(validation_df["time"].min()),
    "validation_end_time": str(validation_df["time"].max()),

    "test_start_time": str(test_df["time"].min()),
    "test_end_time": str(test_df["time"].max()),

    "test_accuracy": float(test_metrics["accuracy"]),
    "test_rain_precision": float(test_metrics["rain_precision"]),
    "test_rain_recall": float(test_metrics["rain_recall"]),
    "test_rain_f1": float(test_metrics["rain_f1"]),
    "test_roc_auc": float(test_metrics["roc_auc"]),
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

# Keep the validation results in the models folder.
validation_results_df.to_csv(
    "models/model_validation_results.csv",
    index=False,
)

# ------------------------------------------------------------
# Update the existing comparison CSV in its original location.
# Keep its original filename and column format.
# Metrics in this file are validation metrics, not test metrics.
# ------------------------------------------------------------

comparison_results = validation_results_df.rename(columns={
    "model": "Model Name",
    "accuracy": "Accuracy (%)",
    "roc_auc": "ROC-AUC",
    "rain_precision": "Rain Precision",
    "rain_recall": "Rain Recall",
    "rain_f1": "Rain F1",
}).copy()

# Convert accuracy from 0-1 to percentage, matching the old CSV.
comparison_results["Accuracy (%)"] = (
    comparison_results["Accuracy (%)"] * 100
).round(2)

# Keep the original column names and order.
comparison_results = comparison_results[
    [
        "Model Name",
        "Accuracy (%)",
        "ROC-AUC",
        "Rain Precision",
        "Rain Recall",
        "Rain F1",
    ]
]

# Round metric values for easier reading.
metric_columns = [
    "ROC-AUC",
    "Rain Precision",
    "Rain Recall",
    "Rain F1",
]

comparison_results[metric_columns] = (
    comparison_results[metric_columns].round(4)
)

comparison_path = "data/model_comparison_results.csv"

comparison_results.to_csv(
    comparison_path,
    index=False,
)

print("\n========== FILES SAVED ==========")
print("Model:", MODEL_PATH)
print("Metadata:", METADATA_PATH)
print("Detailed validation results: models/model_validation_results.csv")
print("Updated comparison results:", comparison_path)

print("\nUpdated model comparison table:")
print(comparison_results.to_string(index=False))

print("\nFinal test metrics:")
print(f"Accuracy:       {test_metrics['accuracy']:.4f}")
print(f"Rain Precision: {test_metrics['rain_precision']:.4f}")
print(f"Rain Recall:    {test_metrics['rain_recall']:.4f}")
print(f"Rain F1-score:  {test_metrics['rain_f1']:.4f}")
print(f"ROC-AUC:        {test_metrics['roc_auc']:.4f}")
