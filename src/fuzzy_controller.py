# Mamdani 模糊逻辑舒适度控制器(根据温度、湿度、UV、光照计算 0%–100% 顶棚闭合建议)

"""Layer B — Proposed Mamdani Fuzzy Logic Controller for Microclimate Comfort.

Evaluates local environmental comfort using:
- Temperature (°C): Low, Medium, High
- Relative Humidity (%): Low, Medium, High
- UV Intensity (UV Index 0-12): Low, Moderate, High
- Light Intensity (0-100%): Dark, Normal, Bright

Output:
- Recommended Canopy Closure Percentage (0% = Fully Open, 100% = Fully Closed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False


@dataclass
class FuzzyComfortResult:
    """Explainable output from the Mamdani Fuzzy Inference System."""

    valid: bool
    recommended_closure_pct: float
    comfort_label: str
    inputs_used: Dict[str, float] = field(default_factory=dict)
    activated_rules: List[Tuple[str, float]] = field(default_factory=list)
    engine_backend: str = "Mamdani FIS"
    status_message: str = "OK"


def _trimf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


class MamdaniCanopyFuzzyController:
    """Mamdani FIS recommending stadium canopy closure (0–100%) for thermal/visual comfort."""

    def __init__(self) -> None:
        self._sim = None
        if HAS_SKFUZZY:
            self._build_skfuzzy_system()

    def _build_skfuzzy_system(self) -> None:
        temp = ctrl.Antecedent(np.arange(15, 42.1, 0.5), "temperature")
        hum = ctrl.Antecedent(np.arange(20, 100.1, 1.0), "humidity")
        uv = ctrl.Antecedent(np.arange(0, 12.1, 0.2), "uv")
        light = ctrl.Antecedent(np.arange(0, 100.1, 1.0), "light")
        closure = ctrl.Consequent(np.arange(0, 100.1, 1.0), "closure", defuzzify_method="centroid")

        # Input Membership Functions
        temp["low"] = fuzz.trapmf(temp.universe, [15, 15, 22, 25])
        temp["medium"] = fuzz.trimf(temp.universe, [23, 27, 31])
        temp["high"] = fuzz.trapmf(temp.universe, [29, 32, 42, 42])

        hum["low"] = fuzz.trapmf(hum.universe, [20, 20, 45, 60])
        hum["medium"] = fuzz.trimf(hum.universe, [50, 67, 82])
        hum["high"] = fuzz.trapmf(hum.universe, [75, 85, 100, 100])

        uv["low"] = fuzz.trapmf(uv.universe, [0, 0, 2.0, 3.5])
        uv["moderate"] = fuzz.trimf(uv.universe, [2.5, 5.0, 7.5])
        uv["high"] = fuzz.trapmf(uv.universe, [6.5, 8.5, 12.0, 12.0])

        light["dark"] = fuzz.trapmf(light.universe, [0, 0, 15, 30])
        light["normal"] = fuzz.trimf(light.universe, [20, 50, 75])
        light["bright"] = fuzz.trapmf(light.universe, [65, 80, 100, 100])

        # Output Membership Functions (Canopy Closure %)
        closure["open"] = fuzz.trapmf(closure.universe, [0, 0, 10, 25])
        closure["partial_low"] = fuzz.trimf(closure.universe, [15, 35, 55])
        closure["partial_high"] = fuzz.trimf(closure.universe, [45, 68, 85])
        closure["closed"] = fuzz.trapmf(closure.universe, [75, 90, 100, 100])

        # Mamdani IF-THEN Rule Base
        rules = [
            ctrl.Rule(temp["high"] & uv["high"], closure["closed"]),
            ctrl.Rule(temp["high"] & light["bright"], closure["closed"]),
            ctrl.Rule(uv["high"] & light["bright"], closure["closed"]),
            ctrl.Rule(temp["high"] & hum["high"] & uv["moderate"], closure["partial_high"]),
            ctrl.Rule(temp["medium"] & uv["high"], closure["partial_high"]),
            ctrl.Rule(temp["medium"] & light["bright"] & uv["moderate"], closure["partial_high"]),
            ctrl.Rule(temp["medium"] & hum["medium"] & uv["moderate"], closure["partial_low"]),
            ctrl.Rule(temp["high"] & hum["high"] & uv["low"], closure["partial_low"]),
            ctrl.Rule(temp["low"] & uv["low"] & light["normal"], closure["open"]),
            ctrl.Rule(temp["low"] & hum["low"] & uv["low"], closure["open"]),
            ctrl.Rule(light["dark"] & uv["low"], closure["open"]),
        ]

        system = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(system)

    def _compute_rule_activations(
        self, t: float, h: float, u: float, l: float
    ) -> List[Tuple[str, float]]:
        """Evaluate transparent firing strengths for UI rule explainability."""
        mu_t_low = _trapmf(t, 15, 15, 22, 25)
        mu_t_med = _trimf(t, 23, 27, 31)
        mu_t_high = _trapmf(t, 29, 32, 42, 42)

        mu_h_low = _trapmf(h, 20, 20, 45, 60)
        mu_h_med = _trimf(h, 50, 67, 82)
        mu_h_high = _trapmf(h, 75, 85, 100, 100)

        mu_u_low = _trapmf(u, 0, 0, 2.0, 3.5)
        mu_u_mod = _trimf(u, 2.5, 5.0, 7.5)
        mu_u_high = _trapmf(u, 6.5, 8.5, 12.0, 12.0)

        mu_l_dark = _trapmf(l, 0, 0, 15, 30)
        mu_l_norm = _trimf(l, 20, 50, 75)
        mu_l_bright = _trapmf(l, 65, 80, 100, 100)

        candidates = [
            ("R1: IF Temp=High AND UV=High THEN Closure=Closed (90%)", min(mu_t_high, mu_u_high)),
            ("R2: IF Temp=High AND Light=Bright THEN Closure=Closed (90%)", min(mu_t_high, mu_l_bright)),
            ("R3: IF UV=High AND Light=Bright THEN Closure=Closed (90%)", min(mu_u_high, mu_l_bright)),
            ("R4: IF Temp=High AND Humidity=High AND UV=Moderate THEN Closure=Partial_High (68%)", min(mu_t_high, mu_h_high, mu_u_mod)),
            ("R5: IF Temp=Medium AND UV=High THEN Closure=Partial_High (68%)", min(mu_t_med, mu_u_high)),
            ("R6: IF Temp=Medium AND Light=Bright AND UV=Moderate THEN Closure=Partial_High (68%)", min(mu_t_med, mu_l_bright, mu_u_mod)),
            ("R7: IF Temp=Medium AND Humidity=Medium AND UV=Moderate THEN Closure=Partial_Low (35%)", min(mu_t_med, mu_h_med, mu_u_mod)),
            ("R8: IF Temp=High AND Humidity=High AND UV=Low THEN Closure=Partial_Low (35% - Vent Heat)", min(mu_t_high, mu_h_high, mu_u_low)),
            ("R9: IF Temp=Low AND UV=Low AND Light=Normal THEN Closure=Open (10%)", min(mu_t_low, mu_u_low, mu_l_norm)),
            ("R10: IF Temp=Low AND Humidity=Low AND UV=Low THEN Closure=Open (10%)", min(mu_t_low, mu_h_low, mu_u_low)),
            ("R11: IF Light=Dark AND UV=Low THEN Closure=Open (10% - Evening Ventilation)", min(mu_l_dark, mu_u_low)),
        ]
        active = [(desc, round(strength, 3)) for desc, strength in candidates if strength > 0.01]
        active.sort(key=lambda item: item[1], reverse=True)
        return active

    def evaluate(
        self,
        temperature_c: Optional[float],
        humidity_pct: Optional[float],
        uv_index: Optional[float],
        light_pct: Optional[float],
    ) -> FuzzyComfortResult:
        """Compute defuzzified canopy closure percentage (0–100%)."""
        if any(v is None or not np.isfinite(float(v)) for v in [temperature_c, humidity_pct, uv_index, light_pct]):
            return FuzzyComfortResult(
                valid=False,
                recommended_closure_pct=50.0,
                comfort_label="Fallback Neutral (Missing Comfort Inputs)",
                status_message="One or more fuzzy inputs missing; using conservative 50% default.",
            )

        t = float(np.clip(temperature_c, 15.0, 42.0))
        h = float(np.clip(humidity_pct, 20.0, 100.0))
        u = float(np.clip(uv_index, 0.0, 12.0))
        l = float(np.clip(light_pct, 0.0, 100.0))

        activated = self._compute_rule_activations(t, h, u, l)

        closure_pct: Optional[float] = None
        backend = "Mamdani Centroid Fallback"

        if self._sim is not None:
            try:
                self._sim.input["temperature"] = t
                self._sim.input["humidity"] = h
                self._sim.input["uv"] = u
                self._sim.input["light"] = l
                self._sim.compute()
                closure_pct = float(self._sim.output["closure"])
                backend = "scikit-fuzzy Mamdani FIS (Centroid)"
            except Exception:
                closure_pct = None

        if closure_pct is None:
            rule_centers = {
                "R1": 90.0, "R2": 90.0, "R3": 90.0,
                "R4": 68.0, "R5": 68.0, "R6": 68.0,
                "R7": 35.0, "R8": 35.0,
                "R9": 10.0, "R10": 10.0, "R11": 10.0,
            }
            if activated:
                num = sum(rule_centers[r.split(":")[0]] * w for r, w in activated)
                den = sum(w for _, w in activated)
                closure_pct = num / den
            else:
                closure_pct = 35.0

        closure_pct = float(np.clip(closure_pct, 0.0, 100.0))

        if closure_pct < 20.0:
            label = "Fully / Mostly Open (Cool & Mild)"
        elif closure_pct < 50.0:
            label = "Partial Low Closure (Balanced Daylight)"
        elif closure_pct < 80.0:
            label = "Partial High Closure (Heat/UV Shading)"
        else:
            label = "Maximum Shading Closure (High Thermal/UV Load)"

        return FuzzyComfortResult(
            valid=True,
            recommended_closure_pct=round(closure_pct, 1),
            comfort_label=label,
            inputs_used={
                "temperature_c": round(t, 2),
                "humidity_pct": round(h, 2),
                "uv_index": round(u, 2),
                "light_pct": round(l, 2),
            },
            activated_rules=activated,
            engine_backend=backend,
            status_message="OK",
        )