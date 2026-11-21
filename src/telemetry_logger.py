# SQLite 历史记录持久化——刷新不丢失
# 每次运行自动把气象、AI 概率、模糊逻辑输出、安全决策写入 data/stadium_telemetry.db

"""Point 6 — Persistent SQLite & CSV Telemetry and Decision Audit Logger.

Ensures operational history, AI predictions, Fuzzy recommendations, safety overrides,
and canopy positions persist across Streamlit page refreshes.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import TELEMETRY_DB_PATH


class TelemetryLogger:
    """SQLite persistence manager for runtime telemetry and control decisions."""

    def __init__(self, db_path: Path = TELEMETRY_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at TEXT NOT NULL,
                    operating_mode TEXT NOT NULL,
                    weather_source TEXT,
                    temperature_c REAL,
                    humidity_pct REAL,
                    wind_speed_kmh REAL,
                    uv_index REAL,
                    light_pct REAL,
                    local_rain INTEGER,
                    ai_valid INTEGER,
                    ai_rain_prob REAL,
                    ai_rain_pred INTEGER,
                    fuzzy_closure_pct REAL,
                    approved_action TEXT,
                    requested_pos_pct REAL,
                    confirmed_pos_pct REAL,
                    simulated_pos_pct REAL,
                    physical_dispatched INTEGER,
                    decision_source TEXT,
                    safety_warnings TEXT
                )
                """
            )
            conn.commit()

    def log_cycle(self, record: Dict[str, Any]) -> None:
        """Insert a single control cycle record into SQLite."""
        now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        warnings_json = json.dumps(record.get("safety_warnings", []), ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO telemetry_logs (
                    logged_at, operating_mode, weather_source,
                    temperature_c, humidity_pct, wind_speed_kmh, uv_index, light_pct,
                    local_rain, ai_valid, ai_rain_prob, ai_rain_pred,
                    fuzzy_closure_pct, approved_action, requested_pos_pct,
                    confirmed_pos_pct, simulated_pos_pct, physical_dispatched,
                    decision_source, safety_warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("logged_at", now_iso),
                    record.get("operating_mode", "UNKNOWN"),
                    record.get("weather_source", "NONE"),
                    record.get("temperature_c"),
                    record.get("humidity_pct"),
                    record.get("wind_speed_kmh"),
                    record.get("uv_index"),
                    record.get("light_pct"),
                    int(bool(record.get("local_rain", False))),
                    int(bool(record.get("ai_valid", False))),
                    record.get("ai_rain_prob"),
                    None if record.get("ai_rain_pred") is None else int(bool(record["ai_rain_pred"])),
                    record.get("fuzzy_closure_pct"),
                    record.get("approved_action", "HOLD"),
                    record.get("requested_pos_pct"),
                    record.get("confirmed_pos_pct"),
                    record.get("simulated_pos_pct"),
                    int(bool(record.get("physical_dispatched", False))),
                    record.get("decision_source", ""),
                    warnings_json,
                ),
            )
            conn.commit()

    def fetch_recent(self, limit: int = 200) -> pd.DataFrame:
        """Return recent logged records ordered newest-first."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT {int(limit)}",
                conn,
            )
        return df

    def clear_logs(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM telemetry_logs")
            conn.commit()