
import os
import pandas as pd

# ==========================================
# STEP 3: Dataset Audit & Seasonal Analysis
# ==========================================

DATA_PATH = "data/kl_stadium_training_clean.csv"
TARGET = "target_rain_next_1h"

os.makedirs("data", exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

print("\n========== DATASET OVERVIEW ==========")
print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Start date:", df["time"].min())
print("End date:", df["time"].max())

# 2. Check data quality
print("\n========== DATA QUALITY ==========")

duplicates = df["time"].duplicated().sum()
missing = df.isnull().sum().sum()

time_diff = df["time"].diff().iloc[1:]
gaps = (time_diff != pd.Timedelta(hours=1)).sum()

print("Duplicate timestamps:", duplicates)
print("Missing values:", missing)
print("Non-hourly intervals:", gaps)

# 3. Rain distribution
print("\n========== RAIN DISTRIBUTION ==========")

rain_count = (df[TARGET] == 1).sum()
no_rain_count = (df[TARGET] == 0).sum()

print("Rain (1):", rain_count)
print("No Rain (0):", no_rain_count)
print("Rain percentage:", round(rain_count / len(df) * 100, 2))

# 4. Malaysia monthly seasonal analysis
df["year"] = df["time"].dt.year
df["month_number"] = df["time"].dt.month

monthly = df.groupby(
    ["year", "month_number"]
).agg(
    total_hours=(TARGET, "size"),
    rainy_hours=(TARGET, "sum")
).reset_index()

monthly["rain_percentage"] = (
    monthly["rainy_hours"] / monthly["total_hours"] * 100
).round(2)

print("\n========== MONTHLY RAIN ANALYSIS ==========")
print(monthly.to_string(index=False))

monthly.to_csv(
    "data/seasonal_rain_summary.csv",
    index=False
)

# 5. Examine existing chronological split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "Training": df.iloc[:train_end],
    "Validation": df.iloc[train_end:val_end],
    "Testing": df.iloc[val_end:]
}

print("\n========== CURRENT DATA SPLIT ==========")

for name, subset in splits.items():
    rain_pct = subset[TARGET].mean() * 100

    print(f"\n{name}")
    print("Records:", len(subset))
    print("Start:", subset["time"].min())
    print("End:", subset["time"].max())
    print("Rain percentage:", round(rain_pct, 2))

print("\nDataset audit completed!")
print("Saved: data/seasonal_rain_summary.csv")
