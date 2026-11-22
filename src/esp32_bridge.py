# 硬件抽象接口 & USB串口 / 纯软件模拟器桥梁
# 严格区分“指令要求位置”与“限位开关/编码器实际确认位置”，未接 ESP32 时绝不虚报物理动作
# DHT11 连续 3 次读取失败才判定故障

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    serial = None  # type: ignore
    PYSERIAL_AVAILABLE = False

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
    """Manages USB Serial link, sensor ingestion, actuator command dispatch, and ACK tracking."""

    def __init__(self, baudrate: int = 115200) -> None:
        self.sensors = ESP32SensorState()
        self.actuator = ESP32ActuatorState()
        self.simulated_canopy_pct: float = 0.0

        # USB Serial transport state
        self.baudrate: int = baudrate
        self.serial_port_name: Optional[str] = None
        self._ser = None
        self._rx_buffer: str = ""
        self._last_scan_epoch: float = 0.0

    @staticmethod
    def list_available_ports() -> List[str]:
        """Return list of available serial COM ports on this PC."""
        if not PYSERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    def _find_esp32_port(self) -> Optional[str]:
        """Auto-detect common ESP32 USB-UART bridges (CP210x, CH340, FTDI, Espressif)."""
        if not PYSERIAL_AVAILABLE:
            return None
        ports = list(serial.tools.list_ports.comports())
        keywords = ("cp210", "ch340", "ch910", "ftdi", "usb serial", "uart", "espressif", "jtag")
        for p in ports:
            desc = f"{p.description} {p.manufacturer or ''}".lower()
            if any(k in desc for k in keywords):
                return p.device
        return ports[0].device if len(ports) == 1 else None

    def connect_serial(self, port: Optional[str] = None) -> bool:
        """Open non-blocking USB Serial connection to ESP32."""
        if not PYSERIAL_AVAILABLE:
            return False
        if self._ser is not None and getattr(self._ser, "is_open", False):
            return True

        target_port = port or self._find_esp32_port()
        if not target_port:
            return False

        try:
            self._ser = serial.Serial(target_port, self.baudrate, timeout=0, write_timeout=0.5)
            self.serial_port_name = target_port
            self._rx_buffer = ""
            return True
        except Exception:
            self._ser = None
            self.serial_port_name = None
            return False

    def disconnect_serial(self) -> None:
        """Close USB Serial port cleanly."""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.serial_port_name = None
        self.sensors.connected = False
        self.actuator.connected = False

    def poll_serial(self, preferred_port: Optional[str] = None) -> int:
        """Read all pending JSON telemetry lines from ESP32 over USB Serial."""
        if not PYSERIAL_AVAILABLE:
            return 0

        # Auto-connect every 2 seconds if unplugged
        if self._ser is None or not getattr(self._ser, "is_open", False):
            now = time.time()
            if now - self._last_scan_epoch >= 2.0:
                self._last_scan_epoch = now
                self.connect_serial(preferred_port)
            if self._ser is None:
                return 0

        packets_read = 0
        try:
            waiting = self._ser.in_waiting
            if waiting > 0:
                raw = self._ser.read(waiting).decode("utf-8", errors="ignore")
                self._rx_buffer += raw
                while "\n" in self._rx_buffer:
                    line, self._rx_buffer = self._rx_buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        data = json.loads(line)
                        self.ingest_sensor_payload(data)
                        self.ingest_actuator_ack(data)
                        packets_read += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            self.disconnect_serial()
            return packets_read

        if self.sensors.connected and self.sensors.is_stale():
            self.sensors.connected = False
            self.actuator.connected = False

        self.check_ack_timeout()
        return packets_read

    def _send_serial_json(self, payload: Dict) -> bool:
        """Transmit JSON command line to ESP32 over USB Serial."""
        if self._ser is None or not getattr(self._ser, "is_open", False):
            return False
        try:
            msg = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            self._ser.write(msg)
            self._ser.flush()
            return True
        except Exception:
            self.disconnect_serial()
            return False

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
        """Issue tracked movement command over USB Serial or update simulation state safely."""
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
            else:
                self._send_serial_json(
                    {
                        "cmd": action,
                        "command_id": cmd_id,
                        "target_pos": target,
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