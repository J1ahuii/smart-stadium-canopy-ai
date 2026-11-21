
from pathlib import Path
import pandas as pd

DATA_FILE = Path("data/kl_stadium_training_clean.csv")
OUTPUT_FILE = Path("data/walk_forward_folds.csv")

TARGET = "target_rain_next_1h"
HORIZON = pd.Timedelta(hours=1)

# Preserve existing test period
TEST_START = pd.Timestamp("2026-05-22 22:00:00")

# Quarterly validation periods
FOLD_PERIODS = [
    ("2025_Q1", "2025-01-01", "2025-04-01"),
    ("2025_Q2", "2025-04-01", "2025-07-01"),
    ("2025_Q3", "2025-07-01", "2025-10-01"),
    ("2025_Q4", "2025-10-01", "2026-01-01"),
    ("2026_Q1", "2026-01-01", "2026-04-01"),
]

# Load data
df = pd.read_csv(
    DATA_FILE, parse_dates=["time"]
).sort_values("time").reset_index(drop=True)

if df["time"].duplicated().any():
    raise ValueError("Duplicate timestamps found")

if (
    df["time"].diff().iloc[1:] != HORIZON
).any():
    raise ValueError("Non-hourly timestamps found")

if df[TARGET].isna().any():
    raise ValueError("Missing target values")

# Reserve final test data
test_df = df[df["time"] >= TEST_START]

if test_df.empty:
    raise ValueError("No test records found")

print("\n========== RESERVED TEST ==========")
print("Rows:", len(test_df))
print("Start:", test_df["time"].min())
print("End:", test_df["time"].max())

# Build expanding-window folds
records = []

print("\n========== WALK-FORWARD FOLDS ==========")

for name, start, end in FOLD_PERIODS:

    val_start = pd.Timestamp(start)
    val_end = pd.Timestamp(end)

    if val_end > TEST_START:
        raise ValueError("Validation overlaps test period")

    # Target at t describes rainfall at t+1 hour.
    # Exclude boundary-crossing labels.
    train = df[
        df["time"] + HORIZON < val_start
    ]

    valid = df[
        (df["time"] >= val_start)
        & (df["time"] + HORIZON < val_end)
    ]

    if train.empty or valid.empty:
        raise ValueError(f"Empty fold: {name}")

    if (
        train["time"].max() + HORIZON
        >= valid["time"].min()
    ):
        raise ValueError("Target leakage detected")

    for subset in [train, valid]:
        if subset[TARGET].nunique() != 2:
            raise ValueError(
                f"Missing target class in {name}"
            )

    record = {
        "fold": name,
        "train_start": train["time"].min(),
        "train_end": train["time"].max(),
        "train_rows": len(train),
        "train_rain_pct": round(
            train[TARGET].mean() * 100, 2
        ),
        "val_start": valid["time"].min(),
        "val_end": valid["time"].max(),
        "val_rows": len(valid),
        "val_rain_pct": round(
            valid[TARGET].mean() * 100, 2
        ),
    }

    records.append(record)

    print(
        f"{name}: "
        f"Train={len(train):,}, "
        f"Validation={len(valid):,}, "
        f"Val Rain={record['val_rain_pct']}%"
    )

# Eligible rows for final training after model selection
final_train = df[
    df["time"] + HORIZON < TEST_START
]

print("\nFinal pre-test training rows:", len(final_train))

pd.DataFrame(records).to_csv(
    OUTPUT_FILE, index=False
)

print("\nSaved:", OUTPUT_FILE)
print("Step 5 completed successfully!")
