# 独立安全监督器
# 大风一票否决防雨关棚
# 防电机抽风死区 + 冷却过滤 + PIN 码解锁手动控制


"""Layer C — Independent Safety Supervisor & Hybrid Control Arbiter.

Implements Points 1, 2, 4, 5:
- Mechanical & structural safety strictly overrides AI, Fuzzy Logic, and Manual commands.
- Wind-rain conflict rule (Point 1): critical wind speed overrides rain closure and commands
  the structural wind-safe posture (configurable).
- Anti-chattering protection (Point 5): minimum movement deadband + cooldown timer
  (bypassed for Emergency Stop, Critical Wind, and Urgent Rain protection).
- PIN-authenticated manual motor override (Point 5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.ai_predictor import AIPredictionResult
from src.config import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_CRITICAL_WIND_KMH,
    DEFAULT_MANUAL_PIN,
    DEFAULT_MIN_DEADBAND_PCT,
    DEFAULT_RAIN_CLOSURE_PCT,
    DEFAULT_SENSOR_STALE_SECONDS,
    DEFAULT_WIND_SAFE_POSTURE_PCT,
    MODE_1_AUTO_HYBRID,
    MODE_2_WEATHER_ONLY,
    MODE_3_MANUAL_SIM,
    MODE_4_FAULT_FALLBACK,
)
from src.esp32_bridge import ActuatorCommand, ESP32HardwareBridge
from src.fuzzy_controller import FuzzyComfortResult


@dataclass
class SafetyConfig:
    """Configurable engineering thresholds for Safety Supervisor."""

    critical_wind_kmh: float = DEFAULT_CRITICAL_WIND_KMH
    wind_safe_posture_pct: float = DEFAULT_WIND_SAFE_POSTURE_PCT
    rain_protective_closure_pct: float = DEFAULT_RAIN_CLOSURE_PCT
    min_deadband_pct: float = DEFAULT_MIN_DEADBAND_PCT
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    sensor_stale_seconds: float = DEFAULT_SENSOR_STALE_SECONDS
    manual_pin: str = DEFAULT_MANUAL_PIN


@dataclass
class SupervisorDecision:
    """Complete auditable decision record produced on every control cycle."""

    effective_mode: str
    approved_action: str  # "MOVE" | "HOLD" | "STOP" | "SAFETY_OVERRIDE"
    candidate_closure_pct: float
    approved_target_pct: float
    physical_actuation_permitted: bool
    decision_source: str
    safety_warnings: List[str] = field(default_factory=list)
    command: Optional[ActuatorCommand] = None


class SafetySupervisor:
    """Final safety authority governing stadium canopy actuation."""

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()
        self.last_movement_epoch: float = 0.0

    def verify_pin(self, entered_pin: str) -> bool:
        return str(entered_pin).strip() == str(self.config.manual_pin).strip()

    def reset_cooldown(self) -> None:
        self.last_movement_epoch = 0.0

    def evaluate_and_actuate(
        self,
        requested_mode: str,
        bridge: ESP32HardwareBridge,
        ai_result: AIPredictionResult,
        fuzzy_result: FuzzyComfortResult,
        wind_speed_kmh: Optional[float],
        local_rain_detected: bool = False,
        manual_motor_request: Optional[Dict] = None,
        virtual_hardware_loopback: bool = False,
    ) -> SupervisorDecision:
        """Fuse Layer A (AI) + Layer B (Fuzzy) and enforce Layer C (Safety Supervisor)."""
        warnings: List[str] = []
        bridge.check_ack_timeout()

        if bridge.actuator.connected and bridge.actuator.confirmed_position_pct is not None:
            current_pos = bridge.actuator.confirmed_position_pct
        else:
            current_pos = bridge.simulated_canopy_pct

        # --- PRIORITY 1: Hard Hardware Fault / E-Stop Lockout ---
        if bridge.actuator.estop_active or bridge.actuator.actuator_fault:
            msg = bridge.actuator.fault_detail or "Emergency Stop or Actuator Fault active"
            warnings.append(f"CRITICAL LOCKOUT: {msg}")
            cmd = bridge.dispatch_command(
                action="STOP",
                target_position_pct=current_pos,
                reason=f"Safety Lockout: {msg}",
                allow_physical_actuation=False,
            )
            return SupervisorDecision(
                effective_mode=MODE_4_FAULT_FALLBACK,
                approved_action="STOP",
                candidate_closure_pct=current_pos,
                approved_target_pct=current_pos,
                physical_actuation_permitted=False,
                decision_source="Safety Supervisor (Hard Fault / E-Stop)",
                safety_warnings=warnings,
                command=cmd,
            )

        # --- PRIORITY 2: Structural Wind Safety Override (Point 1: Wind vs Rain Conflict) ---
        if wind_speed_kmh is not None and wind_speed_kmh > self.config.critical_wind_kmh:
            warnings.append(
                f"WIND OVERRIDE: {wind_speed_kmh:.1f} km/h > {self.config.critical_wind_kmh:.1f} km/h limit. "
                f"Overriding AI/Fuzzy/Manual to wind-safe posture ({self.config.wind_safe_posture_pct:.0f}%)."
            )
            can_move_physically = (
                requested_mode in (MODE_1_AUTO_HYBRID, MODE_2_WEATHER_ONLY)
                and bridge.actuator.connected
            )
            if requested_mode in (MODE_1_AUTO_HYBRID, MODE_2_WEATHER_ONLY) and not bridge.actuator.connected:
                warnings.append("Actuator disconnected during wind alert — simulated safe posture only.")

            cmd = bridge.dispatch_command(
                action="MOVE",
                target_position_pct=self.config.wind_safe_posture_pct,
                reason="Structural Wind Safety Override",
                allow_physical_actuation=can_move_physically,
                virtual_hardware_loopback=virtual_hardware_loopback,
            )
            self.last_movement_epoch = time.time()
            return SupervisorDecision(
                effective_mode=(
                    MODE_4_FAULT_FALLBACK
                    if (not can_move_physically and requested_mode != MODE_3_MANUAL_SIM)
                    else requested_mode
                ),
                approved_action="SAFETY_OVERRIDE",
                candidate_closure_pct=self.config.wind_safe_posture_pct,
                approved_target_pct=self.config.wind_safe_posture_pct,
                physical_actuation_permitted=can_move_physically,
                decision_source="Safety Supervisor (Critical Wind Override)",
                safety_warnings=warnings,
                command=cmd,
            )

        # --- PRIORITY 3: Operating Mode & Telemetry Health Gate ---
        effective_mode = requested_mode
        physical_permitted = False

        if requested_mode == MODE_1_AUTO_HYBRID:
            sensor_bad = bridge.sensors.is_stale(self.config.sensor_stale_seconds) or bridge.sensors.sensor_fault
            actuator_bad = not bridge.actuator.connected
            if sensor_bad and actuator_bad:
                warnings.append("ESP32 Sensor & Actuator offline -> Entering Mode 4 Safe Fallback (Virtual Sim Active).")
                effective_mode = MODE_4_FAULT_FALLBACK
            elif sensor_bad:
                detail = bridge.sensors.fault_detail or "ESP32 Sensor Node offline/stale"
                warnings.append(f"{detail} -> Degrading to Mode 2 Weather-Only.")
                effective_mode = MODE_2_WEATHER_ONLY if ai_result.available else MODE_4_FAULT_FALLBACK
                physical_permitted = bridge.actuator.connected and ai_result.available
            elif actuator_bad:
                warnings.append("ESP32 Actuator disconnected -> Physical motion blocked (Simulation only).")
                effective_mode = MODE_4_FAULT_FALLBACK
            else:
                physical_permitted = True

        elif requested_mode == MODE_2_WEATHER_ONLY:
            if not ai_result.available:
                warnings.append("Weather API unavailable in Weather-Only Mode -> Entering Mode 4 Safe Fallback.")
                effective_mode = MODE_4_FAULT_FALLBACK
            elif not bridge.actuator.connected:
                warnings.append("ESP32 Actuator offline -> Running Weather-Only in virtual simulation.")
                physical_permitted = False
            else:
                physical_permitted = True

        elif requested_mode == MODE_3_MANUAL_SIM:
            physical_permitted = False

        elif requested_mode == MODE_4_FAULT_FALLBACK:
            physical_permitted = False
            warnings.append("System in Mode 4 Safe Fallback — physical motor actuation disabled.")

        # --- PRIORITY 4: Candidate Target Generation (Manual vs AI + Fuzzy Fusion) ---
        urgent_rain = bool(local_rain_detected or (ai_result.available and ai_result.rain_predicted))
        fuzzy_target = fuzzy_result.recommended_closure_pct

        if manual_motor_request is not None:
            if not self.verify_pin(manual_motor_request.get("pin", "")):
                warnings.append("Unauthorized manual motor command rejected (Invalid PIN).")
                manual_motor_request = None

        if manual_motor_request is not None:
            m_action = manual_motor_request.get("action", "MOVE")
            if m_action == "STOP":
                cmd = bridge.dispatch_command(
                    action="STOP",
                    target_position_pct=current_pos,
                    reason="Authorized Manual STOP",
                    allow_physical_actuation=physical_permitted,
                    virtual_hardware_loopback=virtual_hardware_loopback,
                )
                return SupervisorDecision(
                    effective_mode=effective_mode,
                    approved_action="STOP",
                    candidate_closure_pct=current_pos,
                    approved_target_pct=current_pos,
                    physical_actuation_permitted=physical_permitted,
                    decision_source="Authorized Manual Command (STOP)",
                    safety_warnings=warnings,
                    command=cmd,
                )
            candidate_pct = float(manual_motor_request.get("target_pct", current_pos))
            source_label = "Authorized Manual Motor Override"
            bypass_anti_chatter = True
        elif urgent_rain:
            candidate_pct = max(fuzzy_target, self.config.rain_protective_closure_pct)
            source_label = (
                "Hybrid AI + Local Rain Protection"
                if (local_rain_detected and ai_result.rain_predicted)
                else ("Local Rain Sensor Protection" if local_rain_detected else "AI Next-1h Rain Alert Protection")
            )
            bypass_anti_chatter = True
        else:
            candidate_pct = fuzzy_target
            source_label = "Mamdani Fuzzy Microclimate Comfort"
            bypass_anti_chatter = False

        # --- PRIORITY 5: Anti-Chattering Hysteresis (Deadband + Cooldown, Point 5) ---
        delta = abs(candidate_pct - current_pos)
        elapsed_since_move = time.time() - self.last_movement_epoch

        if not bypass_anti_chatter:
            if delta < self.config.min_deadband_pct:
                warnings.append(
                    f"Hysteresis Deadband Hold: delta {delta:.1f}% < {self.config.min_deadband_pct:.1f}% threshold."
                )
                cmd = bridge.dispatch_command(
                    action="HOLD",
                    target_position_pct=current_pos,
                    reason="Deadband Hold (Anti-Chattering)",
                    allow_physical_actuation=False,
                )
                return SupervisorDecision(
                    effective_mode=effective_mode,
                    approved_action="HOLD",
                    candidate_closure_pct=round(candidate_pct, 1),
                    approved_target_pct=round(current_pos, 1),
                    physical_actuation_permitted=physical_permitted,
                    decision_source=f"{source_label} -> Deadband Hold",
                    safety_warnings=warnings,
                    command=cmd,
                )

            if self.last_movement_epoch > 0 and elapsed_since_move < self.config.cooldown_seconds:
                remaining = int(self.config.cooldown_seconds - elapsed_since_move)
                warnings.append(f"Cooldown Active: {remaining}s remaining before next non-urgent move.")
                cmd = bridge.dispatch_command(
                    action="HOLD",
                    target_position_pct=current_pos,
                    reason="Cooldown Hold (Anti-Chattering)",
                    allow_physical_actuation=False,
                )
                return SupervisorDecision(
                    effective_mode=effective_mode,
                    approved_action="HOLD",
                    candidate_closure_pct=round(candidate_pct, 1),
                    approved_target_pct=round(current_pos, 1),
                    physical_actuation_permitted=physical_permitted,
                    decision_source=f"{source_label} -> Cooldown Hold",
                    safety_warnings=warnings,
                    command=cmd,
                )

        # --- APPROVED MOVEMENT ---
        cmd = bridge.dispatch_command(
            action="MOVE",
            target_position_pct=candidate_pct,
            reason=source_label,
            allow_physical_actuation=physical_permitted,
            virtual_hardware_loopback=virtual_hardware_loopback,
        )
        if delta >= 0.5:
            self.last_movement_epoch = time.time()

        return SupervisorDecision(
            effective_mode=effective_mode,
            approved_action="MOVE",
            candidate_closure_pct=round(candidate_pct, 1),
            approved_target_pct=round(candidate_pct, 1),
            physical_actuation_permitted=physical_permitted,
            decision_source=source_label,
            safety_warnings=warnings,
            command=cmd,
        )