# 硬件抽象接口 & 纯软件模拟器桥梁
# 严格区分“指令要求位置”与“限位开关/编码器实际确认位置”，未接 ESP32 时绝不虚报物理动作
# DHT11 连续 3 次读取失败才判定故障

"""Proposed ESP32 IoT Telemetry & Actuator Communication Bridge (Points 2 & 4).

In Software-Only Mode (no physical ESP32 attached), this module cleanly separates
virtual simulation state from confirmed physical hardware feedback and never falsely
claims physical motor movement when the actuator controller is disconnected.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from src.config import (
    DEFAULT_ACTUATOR_TIMEOUT_SECONDS,
    DEFAULT_SENSOR_STALE_SECONDS,
)


@dataclass
class ESP32SensorState:
    """Latest hardware readings from ESP32 Sensor Node (distinct from Weather API)."""

    connected: bool = False
    last_seen_epoch: float = 0.0
    timestamp_iso: str = "Never"
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    rain_detected: bool = False
    rain_raw_analog: Optional[int] = None
    uv_index: Optional[float] = None
    light_pct: Optional[float] = None
    # IMPORTANT: PIR detects local proximity only, NEVER whole-stadium occupancy.
    pir_zone_activity: bool = False
    dht11_consecutive_failures: int = 0
    sensor_fault: bool = False
    fault_detail: str = ""

    def is_stale(self, max_age_sec: float = DEFAULT_SENSOR_STALE_SECONDS) -> bool:
        if not self.connected or self.last_seen_epoch <= 0:
            return True
        return (time.time() - self.last_seen_epoch) > max_age_sec


@dataclass
class ActuatorCommand:
    """Tracked canopy actuation command with unique UUID."""

    command_id: str
    action: str  # "MOVE" | "STOP" | "HOLD"
    requested_position_pct: float
    reason: str
    issued_at_iso: str
    physical_dispatch: bool


@dataclass
class ESP32ActuatorState:
    """Physical actuator feedback state (distinguishes requested vs confirmed position)."""

    connected: bool = False
    last_heartbeat_epoch: float = 0.0
    requested_position_pct: float = 0.0
    confirmed_position_pct: Optional[float] = None
    position_feedback_mode: str = "Limit Switches + Encoder/Travel-Time"
    limit_open_triggered: bool = True
    limit_close_triggered: bool = False
    motor_moving: bool = False
    estop_active: bool = False
    actuator_fault: bool = False
    fault_detail: str = ""
    last_command_id: str = "NONE"
    last_ack_command_id: str = "NONE"
    ack_status: str = "IDLE"  # "IDLE" | "PENDING_ACK" | "CONFIRMED" | "TIMEOUT_FAULT" | "SIMULATED_ONLY"
    pending_since_epoch: float = 0.0


class ESP32HardwareBridge:
    """Manages sensor ingestion, actuator command dispatch, and ACK tracking."""

    def __init__(self) -> None:
        self.sensors = ESP32SensorState()
        self.actuator = ESP32ActuatorState()
        self.simulated_canopy_pct: float = 0.0

    def ingest_sensor_payload(self, payload: Dict) -> None:
        """Process JSON telemetry packet from ESP32 sensor node (or bench simulator)."""
        now = time.time()
        self.sensors.connected = True
        self.sensors.last_seen_epoch = now
        self.sensors.timestamp_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

        failures = int(payload.get("dht11_failures", 0))
        self.sensors.dht11_consecutive_failures = failures
        if failures >= 3:
            self.sensors.sensor_fault = True
            self.sensors.fault_detail = f"DHT11 read fault ({failures} consecutive failures)"
        else:
            self.sensors.sensor_fault = bool(payload.get("sensor_fault", False))
            self.sensors.fault_detail = str(payload.get("fault_detail", ""))

        self.sensors.temperature_c = payload.get("temperature_c")
        self.sensors.humidity_pct = payload.get("humidity_pct")
        self.sensors.rain_detected = bool(payload.get("rain_detected", False))
        self.sensors.rain_raw_analog = payload.get("rain_raw_analog")
        self.sensors.uv_index = payload.get("uv_index")
        self.sensors.light_pct = payload.get("light_pct")
        self.sensors.pir_zone_activity = bool(payload.get("pir_zone_activity", False))

    def ingest_actuator_ack(self, payload: Dict) -> None:
        """Process physical confirmation packet from ESP32 actuator controller."""
        now = time.time()
        self.actuator.connected = True
        self.actuator.last_heartbeat_epoch = now
        self.actuator.estop_active = bool(payload.get("estop_active", False))
        self.actuator.actuator_fault = bool(payload.get("actuator_fault", False))
        self.actuator.fault_detail = str(payload.get("fault_detail", ""))
        self.actuator.limit_open_triggered = bool(payload.get("limit_open", False))
        self.actuator.limit_close_triggered = bool(payload.get("limit_close", False))
        self.actuator.motor_moving = bool(payload.get("motor_moving", False))

        if "confirmed_position_pct" in payload and payload["confirmed_position_pct"] is not None:
            self.actuator.confirmed_position_pct = float(payload["confirmed_position_pct"])

        ack_id = str(payload.get("command_id", ""))
        if ack_id:
            self.actuator.last_ack_command_id = ack_id
            if ack_id == self.actuator.last_command_id:
                self.actuator.ack_status = "CONFIRMED"

    def check_ack_timeout(self, timeout_sec: float = DEFAULT_ACTUATOR_TIMEOUT_SECONDS) -> bool:
        """Detect unacknowledged motor commands to prevent false physical state claims."""
        if (
            self.actuator.ack_status == "PENDING_ACK"
            and self.actuator.pending_since_epoch > 0
            and (time.time() - self.actuator.pending_since_epoch) > timeout_sec
        ):
            self.actuator.ack_status = "TIMEOUT_FAULT"
            self.actuator.actuator_fault = True
            self.actuator.fault_detail = (
                f"Actuator ACK timeout (>{timeout_sec:.0f}s) for command {self.actuator.last_command_id}"
            )
            return True
        return False

    def dispatch_command(
        self,
        action: str,
        target_position_pct: float,
        reason: str,
        allow_physical_actuation: bool,
        virtual_hardware_loopback: bool = False,
    ) -> ActuatorCommand:
        """Issue tracked movement command or update simulation state safely."""
        cmd_id = str(uuid.uuid4())[:8].upper()
        now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        target = round(float(max(0.0, min(100.0, target_position_pct))), 1)

        if allow_physical_actuation and self.actuator.connected:
            self.actuator.requested_position_pct = target
            self.actuator.last_command_id = cmd_id
            self.actuator.ack_status = "PENDING_ACK"
            self.actuator.pending_since_epoch = time.time()

            if virtual_hardware_loopback:
                self.simulated_canopy_pct = target
                self.ingest_actuator_ack(
                    {
                        "command_id": cmd_id,
                        "confirmed_position_pct": target,
                        "limit_open": target <= 0.5,
                        "limit_close": target >= 99.5,
                        "motor_moving": False,
                        "estop_active": False,
                        "actuator_fault": False,
                    }
                )
            return ActuatorCommand(
                command_id=cmd_id,
                action=action,
                requested_position_pct=target,
                reason=reason,
                issued_at_iso=now_iso,
                physical_dispatch=True,
            )

        # Software Simulation / Disconnected Actuator: update virtual state ONLY
        if action != "STOP":
            self.simulated_canopy_pct = target
        self.actuator.requested_position_pct = target
        self.actuator.ack_status = "SIMULATED_ONLY"
        return ActuatorCommand(
            command_id=cmd_id,
            action=action,
            requested_position_pct=target,
            reason=reason,
            issued_at_iso=now_iso,
            physical_dispatch=False,
        )

    def clear_faults(self) -> None:
        """Operator reset after inspection."""
        self.actuator.actuator_fault = False
        self.actuator.estop_active = False
        self.actuator.fault_detail = ""
        self.actuator.ack_status = "IDLE"
        self.sensors.sensor_fault = False
        self.sensors.dht11_consecutive_failures = 0
        self.sensors.fault_detail = ""