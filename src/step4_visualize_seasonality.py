"""Weather AI figures for Smart Stadium Canopy.

Run from project root:
    python src/step4_visualize_seasonality.py

Updates existing plots, adds correlation and explainability. Does NOT
modify/retrain the saved model artifact or use the previously exposed test set.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

DATA_PATH = Path("data/kl_stadium_training_clean.csv")
SEASONAL_PATH = Path("data/seasonal_rain_summary.csv")
FOLDS_PATH = Path("data/walk_forward_folds.csv")
MODEL_PATH = Path("models/rf_rain_predictor.pkl")
METADATA_PATH = Path("models/rain_predictor_metadata.json")
COMPARISON_PATH = Path("data/model_comparison_results.csv")
OUTPUT_DIR = Path("reports/figures")

TARGET = "target_rain_next_1h"
HORIZON = pd.Timedelta(hours=1)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", OUTPUT_DIR / filename)


def main():
    with METADATA_PATH.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)

    features = metadata["features"]
    threshold = float(metadata["threshold"])
    test_start = pd.Timestamp(metadata["test_start_time"])

    df = pd.read_csv(DATA_PATH, parse_dates=["time"]).sort_values("time")
    df = df.reset_index(drop=True)
    if not set(features + [TARGET, "time"]).issubset(df.columns):
        raise ValueError("Missing one or more expected features in cleaned dataset")
    if df[features + [TARGET]].isna().any().any():
        raise ValueError("Missing training values")

    # The dataset before the reserved test period is used for exploratory charts.
    # To respect the next-hour label boundary, exclude the last hour before test.
    pretest = df[df["time"] + HORIZON < test_start]

    # -------- 1: Original seasonality chart --------
    seasonal = pd.read_csv(SEASONAL_PATH)
    plt.figure(figsize=(11, 6))
    for year in sorted(seasonal["year"].unique()):
        rows = seasonal[seasonal["year"] == year].sort_values("month_number")
        plt.plot(rows["month_number"], rows["rain_percentage"],
                 marker="o", linewidth=2, label=str(year))
    plt.title("Kuala Lumpur - Monthly Next-Hour Rain Rate")
    plt.xlabel("Month")
    plt.ylabel("Rain (1) Percentage (%)")
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.grid(alpha=0.25)
    plt.legend(title="Year")
    save_figure("monthly_rain_by_year.png")

    # -------- 2: Original chronological split distribution --------
    n = len(df)
    sections = [df.iloc[:int(n * .70)],
                df.iloc[int(n * .70):int(n * .85)],
                df.iloc[int(n * .85):]]
    rates = [s[TARGET].mean() * 100 for s in sections]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Training", "Validation", "Testing"], rates)
    for bar, pct in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.2f}%", ha="center")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.set_ylabel("Rain (1) Percentage (%)")
    ax.set_title("Original 70/15/15 Dataset Split")
    ax.grid(axis="y", alpha=0.25)
    save_figure("rain_distribution_splits.png")

    # -------- 3: Correlation heatmap, pre-test records only --------
    corr = pretest[features + [TARGET]].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(13, 11))
    matrix = np.ma.array(corr.to_numpy(), mask=np.triu(
        np.ones(corr.shape, dtype=bool), k=1))
    img = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=85, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
    ax.set_title("Pearson Correlation - Pre-test Weather Data")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    save_figure("correlation_heatmap.png")

    # -------- 4: Re-fit a CLONE for the latest walk-forward fold --------
    # The saved model remains unchanged. This fold's validation dates
    # are later than its training dates (no label crosses the boundary).
    folds = pd.read_csv(FOLDS_PATH, parse_dates=["train_end", "val_start", "val_end"])
    fold = folds.sort_values("val_end").iloc[-1]
    if fold["val_end"] >= test_start:
        raise ValueError("Validation fold overlaps reserved Test")

    train_fold = df[(df["time"] <= fold["train_end"]) &
                    (df["time"] + HORIZON < fold["val_start"])]
    val_fold = df[df["time"].between(fold["val_start"], fold["val_end"])]
    if len(train_fold) != int(fold["train_rows"]) or len(val_fold) != int(fold["val_rows"]):
        raise ValueError("Fold rows changed: rerun Step 5 and Step 6")
    if train_fold["time"].max() + HORIZON >= val_fold["time"].min():
        raise ValueError("Leakage at validation boundary")

    trained_model = joblib.load(MODEL_PATH)
    model = clone(trained_model)
    model.fit(train_fold[features], train_fold[TARGET])
    rain_class_idx = list(model.classes_).index(1)
    scores = model.predict_proba(val_fold[features])[:, rain_class_idx]
    y = val_fold[TARGET].to_numpy()
    predictions = (scores >= threshold).astype(int)

    print(f"\nExplainability uses {fold['fold']} walk-forward validation, NOT the Test set.")
    print("Selected model:", metadata["model_name"], "| threshold:", threshold)
    print("Rain F2 on this fold:", round(fbeta_score(y, predictions, beta=2), 3))
    print("Recall:", round(recall_score(y, predictions), 3),
          "| Precision:", round(precision_score(y, predictions, zero_division=0), 3))

    # -------- 5: Permutation Importance on fold validation --------
    # Custom scorer uses the selected Rain threshold (not the default 0.50).
    def rain_f2_scorer(estimator, X, y_true):
        idx = list(estimator.classes_).index(1)
        p = estimator.predict_proba(X)[:, idx]
        return fbeta_score(y_true, p >= threshold, beta=2, zero_division=0)

    importance = permutation_importance(
        model, val_fold[features], y,
        scoring=rain_f2_scorer,
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importance.importances_mean,
    }).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Decrease in Rain F2 after feature shuffle")
    ax.set_title(f"Permutation Feature Importance - {fold['fold']} Validation")
    ax.grid(axis="x", alpha=0.25)
    save_figure("permutation_feature_importance.png")

    # -------- 6: Confusion Matrix on same validation fold --------
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for (r, c), value in np.ndenumerate(cm):
        ax.text(c, r, f"{value:,}", ha="center", va="center", fontsize=15)
    ax.set_xticks([0, 1], ["Predicted No Rain", "Predicted Rain"])
    ax.set_yticks([0, 1], ["Actual No Rain", "Actual Rain"])
    ax.set_title(f"Confusion Matrix - {fold['fold']} Validation")
    save_figure("confusion_matrix_validation.png")

    # -------- 7: Precision-Recall Curve on same fold --------
    precision, recall, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label="Model PR curve")
    ax.scatter([recall_score(y, predictions)],
               [precision_score(y, predictions, zero_division=0)],
               label=f"Selected threshold {threshold:.3f}", zorder=3)
    ax.axhline(y.mean(), linestyle="--", label="Rain prevalence baseline")
    ax.set_xlabel("Rain Recall")
    ax.set_ylabel("Rain Precision")
    ax.set_ylim(0, 1.01)
    ax.set_xlim(0, 1.01)
    ax.set_title(f"Precision-Recall Curve - {fold['fold']} Validation")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure("precision_recall_curve_validation.png")

    # -------- 8: Existing model comparison results --------
    compare = pd.read_csv(COMPARISON_PATH).sort_values("mean_rain_f2")
    indices = np.arange(len(compare))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(indices - .18, compare["mean_rain_f2"], height=.34, label="Mean Rain F2")
    ax.barh(indices + .18, compare["mean_rain_recall"], height=.34, label="Mean Rain Recall")
    ax.set_yticks(indices, compare["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score across walk-forward validation folds")
    ax.set_title("Rain Prediction Model Comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    save_figure("model_comparison.png")

    print("\nStep 7 completed! Saved model has NOT been changed.")


if __name__ == "__main__":
    main()
