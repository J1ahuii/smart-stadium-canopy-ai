# 🏟️ Intelligent Stadium Microclimate & Canopy Control System

A hybrid **Machine Learning + Mamdani Fuzzy Logic + Safety Supervisor + IoT-Ready** system for autonomous retractable stadium canopy control in Kuala Lumpur, Malaysia.

## Project Implementation Status
- **Completed (`src/step1` – `src/step5`, `models/`, `reports/`):**
  - Cleaned 24,645 hourly Kuala Lumpur historical weather records (`data/kl_stadium_training_clean.csv`).
  - Evaluated 6 classifiers using 5-fold chronological walk-forward validation.
  - Selected **HistGradientBoosting Regularized** (`threshold = 0.175`, Recall = `79.94%`, F2 = `74.39%`).
- **Software Operational System (`src/weather_service.py`, `src/ai_predictor.py`, `src/fuzzy_controller.py`, `src/safety_supervisor.py`, `src/esp32_bridge.py`, `src/telemetry_logger.py`, `app.py`):**
  - Live Open-Meteo Forecast API integration with cold-start 4-hour ring buffer (`past_days=1`) and zero future leakage.
  - Mamdani Fuzzy Inference System for thermal/UV/light comfort optimization (`0–100%` closure).
  - Independent Safety Supervisor enforcing structural wind overrides, anti-chattering hysteresis (deadband + cooldown), PIN-locked manual overrides, and hardware fault fallback.
  - Interactive 5-tab Streamlit dashboard with SQLite audit persistence and offline CSV replay.

## How to Run (Windows / VS Code / `stadium_env`)
```bash
conda activate stadium_env
streamlit run app.py