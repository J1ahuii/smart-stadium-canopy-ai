
# ============================================================
# AI Canopy System - Real Weather Data Cleaning
# Input : data/open-meteo-KL-240101-261023.csv
# Output: data/kl_stadium_training_clean.csv
# ============================================================

import os
import numpy as np
import pandas as pd

RAW_PATH = "data/open-meteo-KL-240101-261023.csv"
OUTPUT_PATH = "data/kl_stadium_training_clean.csv"

RAIN_THRESHOLD_MM = 0.1

os.makedirs("data", exist_ok=True)

# 1. Load raw CSV
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(
        f"Raw CSV not found: {RAW_PATH}"
    )

df = pd.read_csv(RAW_PATH, skiprows=3)
df.columns = df.columns.str.strip()

# 2. Verify columns against the actual raw CSV
required_columns = [
    "time",
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "rain (mm)",
    "cloud_cover (%)",
    "wind_speed_10m (km/h)",
    "is_day ()",
    "shortwave_radiation (W/m²)",
    "direct_radiation (W/m²)",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Unexpected raw CSV format. Missing columns: {missing_columns}"
    )

# 3. Parse time and numeric values
df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

numeric_columns = [
    col for col in required_columns
    if col != "time"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time"])
df = df.sort_values("time")
df = df.drop_duplicates(subset=["time"], keep="first")
df = df.reset_index(drop=True)

# 4. Rename source columns to clear, consistent names
df_clean = df.rename(columns={
    "temperature_2m (°C)": "temperature_2m",
    "relative_humidity_2m (%)": "relative_humidity_2m",
    "rain (mm)": "rain_mm",
    "cloud_cover (%)": "cloud_cover",
    "wind_speed_10m (km/h)": "wind_speed_10m",
    "is_day ()": "is_day",
    "shortwave_radiation (W/m²)": "shortwave_radiation",
    "direct_radiation (W/m²)": "direct_radiation",
}).copy()

# 5. Check hourly continuity
# Target labels are only created when the next record is exactly
# one hour later. This avoids incorrectly labelling a time gap
# as the next-hour rainfall.
next_time = df_clean["time"].shift(-1)
next_rain = df_clean["rain_mm"].shift(-1)

has_next_hour = (
    next_time == df_clean["time"] + pd.Timedelta(hours=1)
)

# 6. Create time features
df_clean["hour"] = df_clean["time"].dt.hour
df_clean["day"] = df_clean["time"].dt.day
df_clean["month"] = df_clean["time"].dt.month

# 7. Create historical features
# These features use current or past records only.
df_clean["temperature_diff_1h"] = (
    df_clean["temperature_2m"].diff()
)

df_clean["humidity_diff_1h"] = (
    df_clean["relative_humidity_2m"].diff()
)

df_clean["wind_diff_1h"] = (
    df_clean["wind_speed_10m"].diff()
)

df_clean["cloud_diff_1h"] = (
    df_clean["cloud_cover"].diff()
)

df_clean["shortwave_radiation_diff_1h"] = (
    df_clean["shortwave_radiation"].diff()
)

df_clean["rain_lag_1h"] = df_clean["rain_mm"].shift(1)

df_clean["rain_roll_3h"] = (
    df_clean["rain_mm"]
    .rolling(window=3, min_periods=3)
    .mean()
)

df_clean["humidity_roll_3h"] = (
    df_clean["relative_humidity_2m"]
    .rolling(window=3, min_periods=3)
    .mean()
)

# 8. Create next-hour rainfall target
# 1 = rainfall in the next hourly record >= 0.1 mm
# 0 = rainfall in the next hourly record < 0.1 mm
# Rows without a valid next-hour record are marked missing.
df_clean["target_rain_next_1h"] = np.nan

valid_target_rows = has_next_hour

df_clean.loc[
    valid_target_rows,
    "target_rain_next_1h"
] = (
    next_rain.loc[valid_target_rows] >= RAIN_THRESHOLD_MM
).astype(int)

# 9. Remove rows that cannot be used by the model
model_features = [
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

required_training_columns = (
    model_features + ["target_rain_next_1h"]
)

df_clean = df_clean.dropna(
    subset=required_training_columns
).copy()

df_clean["target_rain_next_1h"] = (
    df_clean["target_rain_next_1h"].astype(int)
)

df_clean = df_clean.sort_values("time").reset_index(drop=True)

# 10. Validate and save
if df_clean.empty:
    raise ValueError("No usable rows remain after preprocessing.")

if df_clean["time"].duplicated().any():
    raise ValueError("Duplicate timestamps remain.")

if not set(df_clean["target_rain_next_1h"].unique()).issubset({0, 1}):
    raise ValueError("Target contains values other than 0 and 1.")

df_clean.to_csv(OUTPUT_PATH, index=False)

print("\nData cleaning completed successfully.")
print("Output:", OUTPUT_PATH)
print("Records:", len(df_clean))
print("Date range:", df_clean["time"].min(), "to", df_clean["time"].max())

print("\nFeatures used by the model:")
print(model_features)

print("\nTarget distribution:")
print(
    df_clean["target_rain_next_1h"]
    .value_counts()
    .sort_index()
    .rename(index={0: "No Rain", 1: "Rain"})
)

print("\nMissing values in training columns:")
print(df_clean[required_training_columns].isna().sum())

print("\nFirst five rows:")
print(df_clean.head())
