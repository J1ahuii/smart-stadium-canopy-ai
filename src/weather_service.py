# 实时 Open-Meteo API + 历史 CSV 回放(防止开机失忆误报)

"""Weather data ingestion and 18-feature engineering service.

Implements:
- Live Open-Meteo Forecast API client with cold-start 4-hour ring buffer (Point 3)
  using `past_days=1` and strictly excluding future hours (t+1, ...) to prevent leakage.
- Historical CSV replay loader for offline software demonstrations (Mode 3).
- Strict schema, unit, and NaN validation matching `rain_predictor_metadata.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.config import (
    CLEAN_DATASET_PATH,
    MODEL_METADATA_PATH,
    OPEN_METEO_FORECAST_URL,
    STADIUM_LATITUDE,
    STADIUM_LONGITUDE,
    STADIUM_TIMEZONE,
)

HOURLY_API_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "is_day",
    "shortwave_radiation",
    "direct_radiation",
]


@dataclass
class WeatherFeaturePacket:
    """Validated weather snapshot and exact 18-feature vector for ML inference."""

    source: str  # "OPEN_METEO_LIVE_API" | "HISTORICAL_CSV_REPLAY" | "UNAVAILABLE"
    timestamp: str
    valid: bool
    error_message: str = ""
    raw_current: Dict[str, float] = field(default_factory=dict)
    feature_dict: Dict[str, float] = field(default_factory=dict)
    feature_vector: Optional[np.ndarray] = None
    history_window: Optional[pd.DataFrame] = None


def load_expected_features() -> List[str]:
    """Read exact 18 feature names in training order from model metadata."""
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    return list(metadata["features"])


def validate_feature_dict(
    feature_dict: Dict[str, float],
    expected_features: List[str],
) -> Tuple[bool, str, Optional[np.ndarray]]:
    """Verify exact feature presence, finite numeric values, and physical bounds."""
    missing = [f for f in expected_features if f not in feature_dict]
    if missing:
        return False, f"Missing required weather features: {missing}", None

    values: List[float] = []
    for feat in expected_features:
        val = feature_dict[feat]
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return False, f"Invalid/NaN value in feature '{feat}'", None
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return False, f"Non-numeric value in feature '{feat}': {val}", None
        if not np.isfinite(fval):
            return False, f"Non-finite value in feature '{feat}'", None
        values.append(fval)

    if not (-10.0 <= feature_dict["temperature_2m"] <= 55.0):
        return False, "temperature_2m out of physical bounds", None
    if not (0.0 <= feature_dict["relative_humidity_2m"] <= 100.0):
        return False, "relative_humidity_2m out of [0, 100] bounds", None
    if feature_dict["rain_mm"] < 0.0 or feature_dict["wind_speed_10m"] < 0.0:
        return False, "Negative rainfall or wind speed detected", None

    arr = np.asarray(values, dtype=np.float64).reshape(1, -1)
    return True, "OK", arr


class OpenMeteoWeatherService:
    """Fetches live Open-Meteo Forecast API data and builds the 18-feature vector."""

    def __init__(
        self,
        latitude: float = STADIUM_LATITUDE,
        longitude: float = STADIUM_LONGITUDE,
        timezone: str = STADIUM_TIMEZONE,
        timeout_sec: float = 8.0,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.timeout_sec = timeout_sec
        self.expected_features = load_expected_features()

    def fetch_live_features(self) -> WeatherFeaturePacket:
        """Fetch current + past 3h observations from Open-Meteo and compute 18 features."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(HOURLY_API_VARIABLES),
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "past_days": 1,
            "forecast_days": 1,
            "timezone": self.timezone,
        }
        try:
            resp = requests.get(
                OPEN_METEO_FORECAST_URL, params=params, timeout=self.timeout_sec
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            return WeatherFeaturePacket(
                source="OPEN_METEO_LIVE_API",
                timestamp=datetime.now(ZoneInfo(self.timezone)).isoformat(timespec="seconds"),
                valid=False,
                error_message=f"Open-Meteo API request failed: {exc}",
            )

        hourly = payload.get("hourly")
        if not hourly or "time" not in hourly:
            return WeatherFeaturePacket(
                source="OPEN_METEO_LIVE_API",
                timestamp=datetime.now(ZoneInfo(self.timezone)).isoformat(timespec="seconds"),
                valid=False,
                error_message="Malformed Open-Meteo response: missing 'hourly' block.",
            )

        try:
            df = pd.DataFrame(hourly)
            df["time"] = pd.to_datetime(df["time"])
            df = df.rename(columns={"rain": "rain_mm"}).sort_values("time").reset_index(drop=True)

            # Strictly exclude future forecast timestamps (> current local hour)
            now_local = datetime.now(ZoneInfo(self.timezone)).replace(
                minute=0, second=0, microsecond=0, tzinfo=None
            )
            observed = df[df["time"] <= now_local].copy()
            if len(observed) < 4:
                return WeatherFeaturePacket(
                    source="OPEN_METEO_LIVE_API",
                    timestamp=now_local.isoformat(),
                    valid=False,
                    error_message="Insufficient past hourly records (<4h) in API response.",
                )

            # Keep last 4 consecutive hourly records: t-3, t-2, t-1, t0
            ring_buffer = observed.iloc[-4:].reset_index(drop=True)
            diffs = ring_buffer["time"].diff().iloc[1:]
            if (diffs != pd.Timedelta(hours=1)).any():
                return WeatherFeaturePacket(
                    source="OPEN_METEO_LIVE_API",
                    timestamp=str(ring_buffer["time"].iloc[-1]),
                    valid=False,
                    error_message="Non-consecutive hourly timestamps in API ring buffer.",
                )

            t_prev = ring_buffer.iloc[-2]  # t - 1h
            t_curr = ring_buffer.iloc[-1]  # t0
            last_3h = ring_buffer.iloc[-3:]  # t-2h, t-1h, t0

            feature_dict: Dict[str, float] = {
                "temperature_2m": float(t_curr["temperature_2m"]),
                "relative_humidity_2m": float(t_curr["relative_humidity_2m"]),
                "rain_mm": float(t_curr["rain_mm"]),
                "cloud_cover": float(t_curr["cloud_cover"]),
                "wind_speed_10m": float(t_curr["wind_speed_10m"]),
                "is_day": float(t_curr["is_day"]),
                "shortwave_radiation": float(t_curr["shortwave_radiation"]),
                "direct_radiation": float(t_curr["direct_radiation"]),
                "temperature_diff_1h": float(t_curr["temperature_2m"] - t_prev["temperature_2m"]),
                "humidity_diff_1h": float(
                    t_curr["relative_humidity_2m"] - t_prev["relative_humidity_2m"]
                ),
                "wind_diff_1h": float(t_curr["wind_speed_10m"] - t_prev["wind_speed_10m"]),
                "cloud_diff_1h": float(t_curr["cloud_cover"] - t_prev["cloud_cover"]),
                "shortwave_radiation_diff_1h": float(
                    t_curr["shortwave_radiation"] - t_prev["shortwave_radiation"]
                ),
                "rain_lag_1h": float(t_prev["rain_mm"]),
                "rain_roll_3h": float(last_3h["rain_mm"].mean()),
                "humidity_roll_3h": float(last_3h["relative_humidity_2m"].mean()),
                "hour": float(t_curr["time"].hour),
                "month": float(t_curr["time"].month),
            }

            is_valid, msg, vec = validate_feature_dict(feature_dict, self.expected_features)
            raw_current = {
                "temperature_2m": feature_dict["temperature_2m"],
                "relative_humidity_2m": feature_dict["relative_humidity_2m"],
                "rain_mm": feature_dict["rain_mm"],
                "cloud_cover": feature_dict["cloud_cover"],
                "wind_speed_10m": feature_dict["wind_speed_10m"],
                "is_day": feature_dict["is_day"],
                "shortwave_radiation": feature_dict["shortwave_radiation"],
                "direct_radiation": feature_dict["direct_radiation"],
            }
            return WeatherFeaturePacket(
                source="OPEN_METEO_LIVE_API",
                timestamp=str(t_curr["time"]),
                valid=is_valid,
                error_message="" if is_valid else msg,
                raw_current=raw_current,
                feature_dict=feature_dict,
                feature_vector=vec,
                history_window=ring_buffer,
            )
        except Exception as exc:
            return WeatherFeaturePacket(
                source="OPEN_METEO_LIVE_API",
                timestamp=datetime.now(ZoneInfo(self.timezone)).isoformat(timespec="seconds"),
                valid=False,
                error_message=f"Feature construction error: {exc}",
            )


class HistoricalCSVReplayService:
    """Streams pre-cleaned historical rows from `kl_stadium_training_clean.csv` for Mode 3."""

    def __init__(self, csv_path=CLEAN_DATASET_PATH) -> None:
        self.csv_path = csv_path
        self.expected_features = load_expected_features()
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)
        return self._df

    def total_rows(self) -> int:
        return len(self._load())

    def find_sample_rain_event_index(self) -> int:
        """Locate an index right before an actual rainfall onset for demo purposes."""
        df = self._load()
        sub = df.iloc[20000:]
        hits = sub[(sub["rain_mm"] == 0.0) & (sub["target_rain_next_1h"] == 1)]
        if not hits.empty:
            return int(hits.index[0])
        return max(0, len(df) - 100)

    def get_row_packet(self, row_index: int) -> Tuple[WeatherFeaturePacket, Optional[int]]:
        """Return a validated WeatherFeaturePacket and ground-truth label for a CSV row."""
        df = self._load()
        idx = int(np.clip(row_index, 0, len(df) - 1))
        row = df.iloc[idx]

        feature_dict = {feat: float(row[feat]) for feat in self.expected_features}
        is_valid, msg, vec = validate_feature_dict(feature_dict, self.expected_features)

        start_win = max(0, idx - 3)
        history_win = df.iloc[start_win : idx + 1][
            ["time", "temperature_2m", "relative_humidity_2m", "rain_mm", "wind_speed_10m"]
        ].copy()

        raw_current = {
            "temperature_2m": feature_dict["temperature_2m"],
            "relative_humidity_2m": feature_dict["relative_humidity_2m"],
            "rain_mm": feature_dict["rain_mm"],
            "cloud_cover": feature_dict["cloud_cover"],
            "wind_speed_10m": feature_dict["wind_speed_10m"],
            "is_day": feature_dict["is_day"],
            "shortwave_radiation": feature_dict["shortwave_radiation"],
            "direct_radiation": feature_dict["direct_radiation"],
        }

        actual_target = int(row["target_rain_next_1h"]) if "target_rain_next_1h" in row else None
        packet = WeatherFeaturePacket(
            source="HISTORICAL_CSV_REPLAY",
            timestamp=str(row["time"]),
            valid=is_valid,
            error_message="" if is_valid else msg,
            raw_current=raw_current,
            feature_dict=feature_dict,
            feature_vector=vec,
            history_window=history_win,
        )
        return packet, actual_target