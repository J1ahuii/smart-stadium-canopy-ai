# 全局参数与安全阈值配置(强风，冷却时间，手控密码2026)

"""Centralized configuration for Intelligent Stadium Microclimate & Canopy Control System.

Distinguishes completed offline ML artifacts from proposed runtime/hardware parameters.
"""

from pathlib import Path

# Project paths (relative to repository root)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Completed ML artifacts (DO NOT overwrite during runtime)
CLEAN_DATASET_PATH = DATA_DIR / "kl_stadium_training_clean.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "rf_rain_predictor.pkl"  # Stores HistGBM Regularized
MODEL_METADATA_PATH = MODELS_DIR / "rain_predictor_metadata.json"
MODEL_COMPARISON_PATH = DATA_DIR / "model_comparison_results.csv"
SEASONAL_SUMMARY_PATH = DATA_DIR / "seasonal_rain_summary.csv"

# Runtime persistence (Point 6: SQLite audit database)
TELEMETRY_DB_PATH = DATA_DIR / "stadium_telemetry.db"

# Stadium location: Kuala Lumpur, Malaysia
STADIUM_LATITUDE = 3.1390
STADIUM_LONGITUDE = 101.6869
STADIUM_TIMEZONE = "Asia/Kuala_Lumpur"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Four Operating Modes
MODE_1_AUTO_HYBRID = "Mode 1: Automatic Hybrid Mode"
MODE_2_WEATHER_ONLY = "Mode 2: Weather-Only Mode"
MODE_3_MANUAL_SIM = "Mode 3: Manual Simulation Mode"
MODE_4_FAULT_FALLBACK = "Mode 4: Hardware Fault / Safe Fallback"

ALL_MODES = [
    MODE_1_AUTO_HYBRID,
    MODE_2_WEATHER_ONLY,
    MODE_3_MANUAL_SIM,
    MODE_4_FAULT_FALLBACK,
]

# Configurable Safety & Control Parameters (Points 1, 2, 4, 5)
# Prototype defaults — must be verified by structural engineering assessment.
DEFAULT_CRITICAL_WIND_KMH = 35.0         # Wind speed triggering structural safety override
DEFAULT_WIND_SAFE_POSTURE_PCT = 0.0      # Structural wind-safe posture (0% = fully retracted)
DEFAULT_RAIN_CLOSURE_PCT = 90.0          # Protective canopy closure when rain predicted/detected
DEFAULT_MIN_DEADBAND_PCT = 10.0          # Point 5: Minimum position delta (%) to trigger movement
DEFAULT_COOLDOWN_SECONDS = 30.0          # Point 5: Cooldown between non-urgent moves (30s for demo; 300s in prod)
DEFAULT_SENSOR_STALE_SECONDS = 15.0      # Point 4: Max age before ESP32 telemetry is stale
DEFAULT_ACTUATOR_TIMEOUT_SECONDS = 10.0  # Point 2: Max wait for ESP32 command ACK
DEFAULT_MANUAL_PIN = "2026"              # Point 5: PIN required to unlock manual motor controls