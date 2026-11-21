# 加载你已训练好的 HistGBM 模型，18个特征，缺失特征时安全拒绝推理

"""Layer A — Completed AI Rainfall Prediction Wrapper.

Loads the pre-trained HistGradientBoosting Regularized classifier
(`models/rf_rain_predictor.pkl`) and metadata (`models/rain_predictor_metadata.json`).
Enforces strict 18-feature compatibility and refuses inference on invalid/incomplete data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib

from src.config import MODEL_ARTIFACT_PATH, MODEL_METADATA_PATH
from src.weather_service import WeatherFeaturePacket


@dataclass
class AIPredictionResult:
    """Structured output of Layer A AI Rainfall Predictor."""

    available: bool
    rain_probability: Optional[float]
    rain_predicted: Optional[bool]
    threshold: float
    model_name: str
    data_source: str
    status_message: str


class RainfallAIPredictor:
    """Inference wrapper preserving exact training feature order and validation threshold."""

    def __init__(
        self,
        model_path=MODEL_ARTIFACT_PATH,
        metadata_path=MODEL_METADATA_PATH,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.metadata: Dict[str, Any] = self._load_metadata()
        self.model_name: str = str(self.metadata.get("model_name", "HistGBM Regularized"))
        self.threshold: float = float(self.metadata.get("threshold", 0.175))
        self.expected_features: List[str] = list(self.metadata.get("features", []))
        self.model = None
        self.load_error: Optional[str] = None
        self._load_model()

    def _load_metadata(self) -> Dict[str, Any]:
        with self.metadata_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _load_model(self) -> None:
        try:
            self.model = joblib.load(self.model_path)
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = f"Model artifact load failed: {exc}"

    def predict(self, packet: Optional[WeatherFeaturePacket]) -> AIPredictionResult:
        """Predict next-1h rainfall probability (>= 0.1 mm) if valid 18-feature input exists."""
        if self.model is None:
            return AIPredictionResult(
                available=False,
                rain_probability=None,
                rain_predicted=None,
                threshold=self.threshold,
                model_name=self.model_name,
                data_source=packet.source if packet else "NONE",
                status_message=self.load_error or "Model artifact unavailable.",
            )

        if packet is None or not packet.valid or packet.feature_vector is None:
            reason = packet.error_message if packet else "No weather packet supplied."
            return AIPredictionResult(
                available=False,
                rain_probability=None,
                rain_predicted=None,
                threshold=self.threshold,
                model_name=self.model_name,
                data_source=packet.source if packet else "NONE",
                status_message=f"AI inference blocked (invalid/incomplete features): {reason}",
            )

        if packet.feature_vector.shape != (1, len(self.expected_features)):
            return AIPredictionResult(
                available=False,
                rain_probability=None,
                rain_predicted=None,
                threshold=self.threshold,
                model_name=self.model_name,
                data_source=packet.source,
                status_message=(
                    f"Shape mismatch: expected (1, {len(self.expected_features)}), "
                    f"got {packet.feature_vector.shape}"
                ),
            )

        try:
            classes = list(self.model.classes_)
            rain_idx = classes.index(1)
            prob = float(self.model.predict_proba(packet.feature_vector)[0, rain_idx])
            pred = bool(prob >= self.threshold)
            return AIPredictionResult(
                available=True,
                rain_probability=prob,
                rain_predicted=pred,
                threshold=self.threshold,
                model_name=self.model_name,
                data_source=packet.source,
                status_message=(
                    f"Rain Alert (>= {self.threshold:.3f})"
                    if pred
                    else f"Below Threshold (< {self.threshold:.3f})"
                ),
            )
        except Exception as exc:
            return AIPredictionResult(
                available=False,
                rain_probability=None,
                rain_predicted=None,
                threshold=self.threshold,
                model_name=self.model_name,
                data_source=packet.source,
                status_message=f"Inference runtime error: {exc}",
            )