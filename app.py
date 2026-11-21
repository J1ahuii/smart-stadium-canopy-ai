"""Modern Weather-Integrated SaaS Dashboard for Intelligent Stadium Canopy Control.

Run from project root:
    streamlit run app.py
"""

from datetime import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.ai_predictor import RainfallAIPredictor
from src.config import (
    MODE_1_AUTO_HYBRID,
    MODE_2_WEATHER_ONLY,
    MODE_3_MANUAL_SIM,
    MODE_4_FAULT_FALLBACK,
    MODEL_COMPARISON_PATH,
    REPORTS_FIGURES_DIR,
)
from src.esp32_bridge import ESP32HardwareBridge
from src.fuzzy_controller import MamdaniCanopyFuzzyController
from src.safety_supervisor import SafetySupervisor
from src.telemetry_logger import TelemetryLogger
from src.weather_service import HistoricalCSVReplayService, OpenMeteoWeatherService

st.set_page_config(
    page_title="Stadium Canopy & Microclimate",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Hybrid Fintech Matte-Black Sidebar + Clean Weather Bento CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #F6F8FA;
        color: #111827;
    }
    .block-container {
        padding-top: 3.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1380px;
    }
    [data-testid="stAppDeployButton"], #MainMenu, footer {
        display: none !important;
    }

    /* Matte Black Left Navigation Rail */
    [data-testid="stSidebar"] {
        background-color: #141518 !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }
    [data-testid="stSidebar"] .stRadio label {
        background-color: transparent;
        padding: 10px 14px;
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.92rem;
        transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #22242A;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #22242A !important;
        color: #F9FAFB !important;
        border: 1px solid #32353E !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2E313A !important;
        border-color: #4B5563 !important;
    }

    /* Brand Mark in Sidebar */
    .brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.25rem 0 1.25rem 0;
        border-bottom: 1px solid #26282E;
        margin-bottom: 1.1rem;
    }
    .brand-logo {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background-color: #10B981;
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .brand-name {
        font-weight: 700;
        font-size: 0.98rem;
        color: #FFFFFF !important;
        line-height: 1.2;
    }
    .brand-sub {
        font-size: 0.75rem;
        color: #9CA3AF !important;
    }

    /* Top Page Header */
    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.25rem;
    }
    .page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.025em;
        margin: 0;
    }
    .page-subtitle {
        font-size: 0.86rem;
        color: #6B7280;
        margin-top: 2px;
    }
    .status-badge {
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-standby {
        background-color: #FFFFFF;
        color: #374151;
        border: 1px solid #E5E7EB;
    }
    .badge-active {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-alert {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }

    /* White Elevated Surface Panels */
    .white-panel {
        background-color: #FFFFFF;
        border: 1px solid #EAECEF;
        border-radius: 20px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 2px 10px rgba(17, 24, 39, 0.03);
        margin-bottom: 0.9rem;
    }
    .panel-heading {
        font-size: 1.02rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.9rem;
    }

    /* Weather Hero Card */
    .loc-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #10B981;
        color: #FFFFFF;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
    }
    .unit-pill {
        display: inline-block;
        background-color: #F3F4F6;
        color: #374151;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 999px;
    }
    .weather-hero-grid {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1.1rem;
        gap: 1rem;
    }
    .weather-day {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    .weather-date {
        font-size: 0.82rem;
        color: #6B7280;
        margin-top: 3px;
    }
    .weather-temp-big {
        font-size: 2.35rem;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.03em;
        line-height: 1.05;
        text-align: right;
    }
    .weather-cond {
        font-size: 0.92rem;
        font-weight: 600;
        color: #111827;
        text-align: right;
        margin-top: 6px;
    }
    .weather-sub-note {
        font-size: 0.78rem;
        color: #6B7280;
        text-align: right;
        margin-top: 2px;
    }

    /* 6-Tile "Today's Highlight" Sub-Cards */
    .hl-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }
    .hl-tile {
        background-color: #F8FAFC;
        border: 1px solid #EEF2F6;
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    .hl-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 0.35rem;
    }
    .hl-val {
        font-size: 1.32rem;
        font-weight: 700;
        color: #111827;
        font-variant-numeric: tabular-nums;
    }
    .hl-unit {
        font-size: 0.80rem;
        font-weight: 500;
        color: #6B7280;
    }
    .hl-foot {
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 0.35rem;
    }

    /* Stacked Source Rows */
    .source-row {
        background-color: #F8FAFC;
        border: 1px solid #EEF2F6;
        border-radius: 14px;
        padding: 0.85rem 1.05rem;
        margin-bottom: 0.65rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .src-kicker {
        font-size: 0.72rem;
        color: #6B7280;
        font-weight: 500;
    }
    .src-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #111827;
    }
    .src-status {
        font-size: 0.78rem;
        color: #4B5563;
        margin-top: 1px;
    }
    .src-right {
        text-align: right;
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
    }

    /* Pastel Bento Cards & Dark Contrast Card */
    .bento-card {
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        min-height: 152px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 0.65rem;
    }
    .card-blue { background-color: #EEF5FF; }
    .card-purple { background-color: #EFEBFD; }
    .card-green { background-color: #E7F6EC; }
    .card-cream { background-color: #FDF6E7; }
    .card-white { background-color: #FFFFFF; border: 1px solid #EAECEF; }

    .card-label {
        font-size: 0.80rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 0.4rem;
    }
    .card-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .card-sub {
        font-size: 0.80rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }

    .dark-promo-card {
        background-color: #141518;
        color: #FFFFFF;
        border-radius: 20px;
        padding: 1.45rem 1.55rem;
        min-height: 255px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .dark-card-tag {
        display: inline-block;
        background-color: #26282E;
        color: #E5E7EB;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .dark-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.3;
        margin-bottom: 0.45rem;
    }
    .dark-card-body {
        font-size: 0.83rem;
        color: #9CA3AF;
        line-height: 1.48;
    }

    /* Mode Selection Cards */
    .mode-box {
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
        min-height: 126px;
        margin-bottom: 0.45rem;
        border: 1.5px solid #EAECEF;
        background-color: #FFFFFF;
    }
    .mode-box-active {
        border-color: #141518;
        background-color: #EEF5FF;
        box-shadow: 0 6px 16px rgba(17, 24, 39, 0.06);
    }
    .mode-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #10B981;
        margin-bottom: 4px;
    }
    .mode-name {
        font-size: 1.0rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 5px;
    }
    .mode-text {
        font-size: 0.80rem;
        color: #6B7280;
        line-height: 1.4;
    }

    /* Clean Key-Value List Table */
    .kv-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #FFFFFF;
        border-radius: 14px;
        overflow: hidden;
    }
    .kv-table th {
        text-align: left;
        font-size: 0.76rem;
        font-weight: 600;
        color: #9CA3AF;
        padding: 10px 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    .kv-table td {
        padding: 11px 12px;
        font-size: 0.88rem;
        color: #111827;
        border-bottom: 1px solid #F1F5F9;
    }
    .kv-key { font-weight: 600; color: #111827; }
    .kv-sub { font-size: 0.75rem; color: #9CA3AF; display: block; }
    .kv-val { font-weight: 600; font-variant-numeric: tabular-nums; }
    .pill-ok { color: #059669; font-weight: 600; font-size: 0.80rem; }
    .pill-warn { color: #D97706; font-weight: 600; font-size: 0.80rem; }
    .pill-alert { color: #DC2626; font-weight: 600; font-size: 0.80rem; }

    /* Forecast Strip Cards */
    .fc-card {
        background-color: #F8FAFC;
        border: 1px solid #EEF2F6;
        border-radius: 16px;
        padding: 0.95rem 0.65rem;
        text-align: center;
    }
    .fc-card-now {
        background-color: #EEF5FF;
        border-color: #BFDBFE;
    }
    .fc-time {
        font-size: 0.78rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 0.45rem;
    }
    .fc-temp {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.45rem;
    }
    .fc-rain {
        font-size: 0.74rem;
        color: #6B7280;
        margin-top: 2px;
    }

    .stMain .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_weather_svg(is_rain: bool, cloud_pct: float, size: int = 74) -> str:
    """Crisp flat-vector weather illustration matching modern weather SaaS cards."""
    if is_rain:
        return f"""
        <svg width="{size}" height="{size}" viewBox="0 0 80 80" fill="none">
            <circle cx="52" cy="26" r="13" fill="#FBBF24"/>
            <path d="M24 52H56C62.6 52 68 46.6 68 40C68 33.8 63.2 28.7 57.1 28.1C55.1 20.6 48.2 15 40 15C30.6 15 23 22.6 23 32C23 32.4 23 32.8 23.1 33.2C18.5 34.4 15 38.6 15 43.5C15 48.2 18.8 52 24 52Z" fill="#E2E8F0"/>
            <line x1="28" y1="57" x2="24" y2="66" stroke="#3B82F6" stroke-width="3" stroke-linecap="round"/>
            <line x1="40" y1="57" x2="36" y2="66" stroke="#3B82F6" stroke-width="3" stroke-linecap="round"/>
            <line x1="52" y1="57" x2="48" y2="66" stroke="#3B82F6" stroke-width="3" stroke-linecap="round"/>
        </svg>
        """
    if cloud_pct >= 45.0:
        return f"""
        <svg width="{size}" height="{size}" viewBox="0 0 80 80" fill="none">
            <circle cx="52" cy="28" r="14" fill="#FBBF24"/>
            <path d="M24 56H56C62.6 56 68 50.6 68 44C68 37.8 63.2 32.7 57.1 32.1C55.1 24.6 48.2 19 40 19C30.6 19 23 26.6 23 36C23 36.4 23 36.8 23.1 37.2C18.5 38.4 15 42.6 15 47.5C15 52.2 18.8 56 24 56Z" fill="#CBD5E1"/>
        </svg>
        """
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 80 80" fill="none">
        <circle cx="40" cy="40" r="18" fill="#FBBF24"/>
        <circle cx="40" cy="40" r="25" fill="#FBBF24" fill-opacity="0.22"/>
    </svg>
    """


def render_page_header(title: str, subtitle: str, active_mode: str | None, has_warning: bool = False) -> None:
    if active_mode is None:
        badge = '<span class="status-badge badge-standby">Status: Standby (No Mode Selected)</span>'
    elif has_warning:
        badge = f'<span class="status-badge badge-alert">Active: {active_mode.split(":")[0]} (Safety Alert)</span>'
    else:
        badge = f'<span class="status-badge badge-active">Active: {active_mode.split(":")[0]}</span>'

    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
            </div>
            <div>{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kv_table(rows: list[tuple[str, str, str, str, str]]) -> None:
    body_html = []
    for name, sub, val, ref, cls in rows:
        sub_html = f'<span class="kv-sub">{sub}</span>' if sub else ""
        body_html.append(
            f"<tr>"
            f"<td><span class='kv-key'>{name}</span>{sub_html}</td>"
            f"<td class='kv-val'>{val}</td>"
            f"<td style='color:#6B7280; font-size:0.83rem;'>{ref}</td>"
            f"<td><span class='{cls}'>{cls.replace('pill-', '').upper()}</span></td>"
            f"</tr>"
        )
    st.markdown(
        f"""
        <table class="kv-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Current Value</th>
                    <th>Reference / Source</th>
                    <th>State</th>
                </tr>
            </thead>
            <tbody>{''.join(body_html)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_canopy_schematic(closure_pct: float, is_raining: bool, wind_alert: bool, authority: str) -> None:
    pct = max(0.0, min(100.0, float(closure_pct)))
    wing = round(185.0 * (pct / 100.0), 1)
    lx2 = 110.0 + wing
    rx1 = 490.0 - wing

    panel_color = "#DC2626" if wind_alert else ("#2563EB" if is_raining else "#10B981")
    weather_note = (
        "Critical Wind Override Active"
        if wind_alert
        else ("Rain Protection Active" if is_raining else "Normal Microclimate")
    )

    rain_lines = ""
    if is_raining:
        rain_lines = "".join(
            f'<line x1="{x}" y1="{y}" x2="{x-5}" y2="{y+12}" stroke="#3B82F6" stroke-width="1.8" stroke-linecap="round"/>'
            for x in range(150, 460, 32)
            for y in (45, 78)
        )

    html = f"""
    <div style="background:#FFFFFF; border:1px solid #EAECEF; border-radius:18px; padding:14px 18px;
                font-family: Inter, -apple-system, sans-serif;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:13px; font-weight:700; color:#111827;">Retractable Canopy Cross-Section</span>
            <span style="font-size:13px; font-weight:700; color:{panel_color};">Closure: {pct:.1f}% • {weather_note}</span>
        </div>
        <svg viewBox="0 0 600 175" width="100%" height="165">
            {rain_lines}
            <polygon points="45,105 110,105 145,152 455,152 490,105 555,105 525,166 75,166"
                     fill="#F1F5F9" stroke="#CBD5E1" stroke-width="1.5"/>
            <rect x="145" y="146" width="310" height="9" rx="3" fill="#10B981"/>
            <polygon points="35,105 110,88 110,105" fill="#475569"/>
            <polygon points="565,105 490,88 490,105" fill="#475569"/>
            <rect x="110" y="90" width="{wing}" height="9" rx="4" fill="{panel_color}"/>
            <rect x="{rx1}" y="90" width="{wing}" height="9" rx="4" fill="{panel_color}"/>
            <circle cx="{lx2}" cy="94.5" r="4.5" fill="#111827"/>
            <circle cx="{rx1}" cy="94.5" r="4.5" fill="#111827"/>
            <text x="300" y="130" text-anchor="middle" fill="#374151" font-size="11.5" font-weight="600">
                Decision Authority: {authority}
            </text>
        </svg>
    </div>
    """
    components.html(html, height=205)


# --- Session State Initialization ---
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = None  # Standby on initial load
if "bridge" not in st.session_state:
    st.session_state.bridge = ESP32HardwareBridge()
if "supervisor" not in st.session_state:
    st.session_state.supervisor = SafetySupervisor()
if "logger" not in st.session_state:
    st.session_state.logger = TelemetryLogger()
if "ai_predictor" not in st.session_state:
    st.session_state.ai_predictor = RainfallAIPredictor()
if "fuzzy_ctrl" not in st.session_state:
    st.session_state.fuzzy_ctrl = MamdaniCanopyFuzzyController()
if "live_weather_svc" not in st.session_state:
    st.session_state.live_weather_svc = OpenMeteoWeatherService()
if "csv_replay_svc" not in st.session_state:
    st.session_state.csv_replay_svc = HistoricalCSVReplayService()
if "csv_idx" not in st.session_state:
    st.session_state.csv_idx = st.session_state.csv_replay_svc.find_sample_rain_event_index()
if "pin_unlocked" not in st.session_state:
    st.session_state.pin_unlocked = False
if "sim_hw_online" not in st.session_state:
    st.session_state.sim_hw_online = False
if "sim_dht_fault" not in st.session_state:
    st.session_state.sim_dht_fault = False

bridge: ESP32HardwareBridge = st.session_state.bridge
supervisor: SafetySupervisor = st.session_state.supervisor
logger: TelemetryLogger = st.session_state.logger

# --- Left Matte-Black Navigation Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-logo">SC</div>
            <div>
                <div class="brand-name">Stadium Canopy</div>
                <div class="brand-sub">Kuala Lumpur Microclimate</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_page = st.radio(
        "Navigation",
        [
            "Overview",
            "Sensors & Weather",
            "AI & Fuzzy Logic",
            "Canopy Control",
            "Analytics & Logs",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("SYSTEM WORKBENCH")
    if st.session_state.selected_mode:
        st.write(f"**Active:** {st.session_state.selected_mode.split(':')[0]}")
        if st.button("Exit Mode (Standby)", use_container_width=True):
            st.session_state.selected_mode = None
            st.rerun()
    else:
        st.write("**State:** Standby")

    estop_txt = "Release E-Stop" if bridge.actuator.estop_active else "Emergency Stop"
    if st.button(estop_txt, use_container_width=True):
        bridge.actuator.estop_active = not bridge.actuator.estop_active
        if bridge.actuator.estop_active:
            bridge.actuator.fault_detail = "Emergency Stop Triggered by Operator"
        else:
            bridge.clear_faults()
        st.rerun()

# Sync hardware bench state
if st.session_state.sim_hw_online:
    bridge.actuator.connected = True
    if bridge.actuator.confirmed_position_pct is None:
        bridge.actuator.confirmed_position_pct = bridge.simulated_canopy_pct
else:
    bridge.actuator.connected = False
    bridge.sensors.connected = False

active_mode = st.session_state.selected_mode

# ==============================================================================
# Evaluate Control Pipeline When an Operating Mode Is Active
# ==============================================================================
weather_packet = None
actual_csv_label = None
sim_temp = sim_hum = sim_wind = sim_uv = sim_light = None
sim_cloud = 40.0
sim_rad = 450.0
sim_rain_mm = 0.0
sim_local_rain = False
ai_res = None
fuzzy_res = None
decision = None
active_pos = bridge.simulated_canopy_pct
pos_source_txt = "Virtual Simulation"

if active_mode is not None:
    if active_mode == MODE_3_MANUAL_SIM:
        sim_source = st.session_state.get("m3_source", "Historical CSV Replay")
        if sim_source == "Historical CSV Replay":
            weather_packet, actual_csv_label = st.session_state.csv_replay_svc.get_row_packet(
                st.session_state.csv_idx
            )
            sim_temp = weather_packet.raw_current["temperature_2m"]
            sim_hum = weather_packet.raw_current["relative_humidity_2m"]
            sim_wind = weather_packet.raw_current["wind_speed_10m"]
            sim_cloud = weather_packet.raw_current["cloud_cover"]
            sim_rad = weather_packet.raw_current["shortwave_radiation"]
            sim_rain_mm = weather_packet.raw_current["rain_mm"]
            sim_uv = min(11.0, round(sim_rad / 90.0, 1))
            sim_light = min(100.0, round(sim_rad / 9.0, 1))
            sim_local_rain = sim_rain_mm >= 0.1
        else:
            sim_temp = st.session_state.get("sl_temp", 31.5)
            sim_hum = st.session_state.get("sl_hum", 78.0)
            sim_uv = st.session_state.get("sl_uv", 8.0)
            sim_light = st.session_state.get("sl_light", 82.0)
            sim_wind = st.session_state.get("sl_wind", 14.0)
            sim_local_rain = st.session_state.get("sl_rain", False)
            sim_cloud = 85.0 if sim_local_rain else 35.0
            sim_rad = round(sim_light * 9.0, 1)
            sim_rain_mm = 1.2 if sim_local_rain else 0.0
    else:
        if "cached_live_packet" not in st.session_state:
            st.session_state.cached_live_packet = st.session_state.live_weather_svc.fetch_live_features()
        weather_packet = st.session_state.cached_live_packet

        if st.session_state.sim_hw_online:
            base_t = (
                weather_packet.raw_current.get("temperature_2m", 30.0)
                if weather_packet and weather_packet.valid
                else 30.0
            )
            base_h = (
                weather_packet.raw_current.get("relative_humidity_2m", 75.0)
                if weather_packet and weather_packet.valid
                else 75.0
            )
            bridge.ingest_sensor_payload(
                {
                    "temperature_c": None if st.session_state.sim_dht_fault else round(base_t + 0.4, 1),
                    "humidity_pct": None if st.session_state.sim_dht_fault else round(min(100.0, base_h + 2.0), 1),
                    "rain_detected": False,
                    "uv_index": 6.5,
                    "light_pct": 72.0,
                    "pir_zone_activity": True,
                    "dht11_failures": 3 if st.session_state.sim_dht_fault else 0,
                }
            )

        if weather_packet and weather_packet.valid:
            sim_cloud = weather_packet.raw_current.get("cloud_cover", 40.0)
            sim_rad = weather_packet.raw_current.get("shortwave_radiation", 450.0)
            sim_rain_mm = weather_packet.raw_current.get("rain_mm", 0.0)

        sim_temp = (
            bridge.sensors.temperature_c
            if (bridge.sensors.connected and bridge.sensors.temperature_c is not None)
            else (weather_packet.raw_current.get("temperature_2m") if weather_packet and weather_packet.valid else None)
        )
        sim_hum = (
            bridge.sensors.humidity_pct
            if (bridge.sensors.connected and bridge.sensors.humidity_pct is not None)
            else (weather_packet.raw_current.get("relative_humidity_2m") if weather_packet and weather_packet.valid else None)
        )
        sim_wind = (
            weather_packet.raw_current.get("wind_speed_10m")
            if weather_packet and weather_packet.valid
            else None
        )
        sim_uv = (
            bridge.sensors.uv_index
            if bridge.sensors.connected
            else (min(11.0, round(sim_rad / 90.0, 1)) if weather_packet and weather_packet.valid else None)
        )
        sim_light = (
            bridge.sensors.light_pct
            if bridge.sensors.connected
            else (min(100.0, round(sim_rad / 9.0, 1)) if weather_packet and weather_packet.valid else None)
        )
        sim_local_rain = bridge.sensors.rain_detected if bridge.sensors.connected else (sim_rain_mm >= 0.1)

    ai_res = st.session_state.ai_predictor.predict(weather_packet)
    fuzzy_res = st.session_state.fuzzy_ctrl.evaluate(sim_temp, sim_hum, sim_uv, sim_light)

    queued_manual = st.session_state.pop("queued_manual_cmd", None)
    decision = supervisor.evaluate_and_actuate(
        requested_mode=active_mode,
        bridge=bridge,
        ai_result=ai_res,
        fuzzy_result=fuzzy_res,
        wind_speed_kmh=sim_wind,
        local_rain_detected=sim_local_rain,
        manual_motor_request=queued_manual,
        virtual_hardware_loopback=st.session_state.sim_hw_online,
    )

    active_pos = (
        bridge.actuator.confirmed_position_pct
        if (bridge.actuator.connected and bridge.actuator.confirmed_position_pct is not None)
        else bridge.simulated_canopy_pct
    )
    pos_source_txt = (
        "Confirmed Hardware"
        if (bridge.actuator.connected and bridge.actuator.confirmed_position_pct is not None)
        else "Virtual Simulation"
    )

    logger.log_cycle(
        {
            "operating_mode": decision.effective_mode,
            "weather_source": weather_packet.source if weather_packet else "MANUAL_SLIDERS",
            "temperature_c": sim_temp,
            "humidity_pct": sim_hum,
            "wind_speed_kmh": sim_wind,
            "uv_index": sim_uv,
            "light_pct": sim_light,
            "local_rain": sim_local_rain,
            "ai_valid": ai_res.available,
            "ai_rain_prob": ai_res.rain_probability,
            "ai_rain_pred": ai_res.rain_predicted,
            "fuzzy_closure_pct": fuzzy_res.recommended_closure_pct,
            "approved_action": decision.approved_action,
            "requested_pos_pct": bridge.actuator.requested_position_pct,
            "confirmed_pos_pct": bridge.actuator.confirmed_position_pct,
            "simulated_pos_pct": bridge.simulated_canopy_pct,
            "physical_dispatched": decision.physical_actuation_permitted,
            "decision_source": decision.decision_source,
            "safety_warnings": decision.safety_warnings,
        }
    )

with st.sidebar:
    st.caption("Weather API: Open-Meteo (open-meteo.com)  \nLocation: Kuala Lumpur (3.14°N, 101.69°E)")

# ==============================================================================
# PAGE 1: OVERVIEW (Mode Selector + Weather Hero + Today's Highlights + Zones)
# ==============================================================================
if nav_page == "Overview":
    render_page_header(
        "Overview",
        "Intelligent Stadium Microclimate & Retractable Canopy Command",
        active_mode,
        has_warning=bool(decision and decision.safety_warnings),
    )

    st.markdown("#### Operating Modes")
    modes_spec = [
        (MODE_1_AUTO_HYBRID, "Mode 01", "Automatic Hybrid", "Live Open-Meteo AI rain forecast + ESP32 local sensors + Fuzzy comfort."),
        (MODE_2_WEATHER_ONLY, "Mode 02", "Weather-Only", "Live Open-Meteo API + AI prediction when ESP32 sensors are disconnected."),
        (MODE_3_MANUAL_SIM, "Mode 03", "Manual Simulation", "Historical CSV replay or manual environmental sliders for software testing."),
        (MODE_4_FAULT_FALLBACK, "Mode 04", "Fault / Safe Fallback", "Disables physical motor dispatch and holds canopy in protective fallback."),
    ]
    m_cols = st.columns(4)
    for i, (m_const, kicker, m_name, m_desc) in enumerate(modes_spec):
        is_sel = active_mode == m_const
        active_cls = "mode-box-active" if is_sel else ""
        with m_cols[i]:
            st.markdown(
                f"""
                <div class="mode-box {active_cls}">
                    <div class="mode-kicker">{kicker}{' • ACTIVE' if is_sel else ''}</div>
                    <div class="mode-name">{m_name}</div>
                    <div class="mode-text">{m_desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Active" if is_sel else f"Select {kicker}",
                key=f"ov_mode_{i}",
                type="primary" if is_sel else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_mode = m_const
                st.rerun()

    if active_mode is None:
        st.markdown("")
        st.markdown(
            """
            <div class="white-panel">
                <div class="panel-heading">System Standby</div>
                <div style="color:#6B7280; font-size:0.90rem;">
                    No operating mode is active on startup. Choose any of the four modes above to begin live weather forecasting, CSV replay, or safety fallback testing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # Compact Mode 3 Simulation Bar
    if active_mode == MODE_3_MANUAL_SIM:
        st.markdown("")
        c_src, c_ctrl1, c_ctrl2 = st.columns([2, 3, 2])
        with c_src:
            m3_choice = st.radio(
                "Simulation Source",
                ["Historical CSV Replay", "Manual Environmental Sliders"],
                key="m3_source",
                horizontal=True,
            )
        if m3_choice == "Historical CSV Replay":
            max_r = st.session_state.csv_replay_svc.total_rows() - 1
            with c_ctrl1:
                st.session_state.csv_idx = st.slider("CSV Timeline Index", 0, max_r, st.session_state.csv_idx)
            with c_ctrl2:
                st.write("")
                b_next, b_rain = st.columns(2)
                if b_next.button("Next +1h", use_container_width=True):
                    st.session_state.csv_idx = min(max_r, st.session_state.csv_idx + 1)
                    st.rerun()
                if b_rain.button("Rain Event", use_container_width=True):
                    st.session_state.csv_idx = st.session_state.csv_replay_svc.find_sample_rain_event_index()
                    st.rerun()
        else:
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            sc1.slider("Temp (°C)", 18.0, 40.0, 31.5, 0.5, key="sl_temp")
            sc2.slider("Humidity (%)", 30.0, 100.0, 78.0, 1.0, key="sl_hum")
            sc3.slider("UV Index", 0.0, 12.0, 8.0, 0.5, key="sl_uv")
            sc4.slider("Light (%)", 0.0, 100.0, 82.0, 1.0, key="sl_light")
            sc5.slider("Wind (km/h)", 0.0, 60.0, 14.0, 1.0, key="sl_wind")
            sc6.checkbox("Local Rain", value=False, key="sl_rain")

    for w in decision.safety_warnings:
        st.warning(w)

    st.markdown("")

    # --- ROW 2: Weather Hero Card (Left) + Today's Microclimate Highlights 6-Tile Panel (Right) ---
    col_weather, col_highlights = st.columns([1.15, 1.45])

    is_rain_now = bool(sim_local_rain or (ai_res and ai_res.rain_predicted))
    weather_svg = get_weather_svg(is_rain=is_rain_now, cloud_pct=float(sim_cloud or 0.0), size=78)
    ts_display = (
        weather_packet.timestamp
        if (weather_packet and weather_packet.valid)
        else datetime.now().strftime("%d %b, %Y • %H:%M")
    )
    cond_label = (
        "Active Rain / Alert"
        if sim_local_rain
        else ("Rain Expected (Next 1h)" if (ai_res and ai_res.rain_predicted) else ("Partly Cloudy" if sim_cloud >= 45 else "Clear / Sunny"))
    )
    prob_note = (
        f"Next-1h AI Rain Risk: {ai_res.rain_probability * 100:.1f}%"
        if (ai_res and ai_res.available)
        else "AI Forecast Bypassed (Slider Input)"
    )

    with col_weather:
        st.markdown(
            f"""
            <div class="white-panel" style="min-height:275px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="loc-pill">Kuala Lumpur • National Stadium</span>
                    <span class="unit-pill">°C / SI</span>
                </div>
                <div class="weather-hero-grid">
                    <div>
                        <div class="weather-day">Microclimate</div>
                        <div class="weather-date">{ts_display}</div>
                        <div style="margin-top:12px;">
                            <span style="font-size:0.78rem; color:#6B7280;">Canopy Closure</span><br/>
                            <span style="font-size:1.35rem; font-weight:800; color:#10B981;">{active_pos:.1f}%</span>
                            <span style="font-size:0.78rem; color:#6B7280;"> ({decision.approved_action})</span>
                        </div>
                    </div>
                    <div>{weather_svg}</div>
                    <div>
                        <div class="weather-temp-big">{f'{sim_temp:.1f}°C' if sim_temp is not None else 'N/A'}</div>
                        <div class="weather-cond">{cond_label}</div>
                        <div class="weather-sub-note">{prob_note}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    wind_str = f"{sim_wind:.1f}" if sim_wind is not None else "N/A"
    wind_note = (
        "Exceeds critical limit"
        if (sim_wind is not None and sim_wind > supervisor.config.critical_wind_kmh)
        else f"Limit {supervisor.config.critical_wind_kmh:.0f} km/h"
    )
    hum_str = f"{sim_hum:.0f}" if sim_hum is not None else "N/A"
    hum_note = "High moisture load" if (sim_hum is not None and sim_hum > 80) else "Humidity is normal"
    uv_str = f"{sim_uv:.1f}" if sim_uv is not None else "N/A"
    uv_note = "High UV exposure" if (sim_uv is not None and sim_uv >= 7.0) else "Moderate UV"

    with col_highlights:
        st.markdown(
            f"""
            <div class="white-panel" style="min-height:275px;">
                <div class="panel-heading">Today's Microclimate Highlights</div>
                <div class="hl-grid">
                    <div class="hl-tile">
                        <div class="hl-label">Wind Status</div>
                        <div class="hl-val">{wind_str} <span class="hl-unit">km/h</span></div>
                        <div class="hl-foot">{wind_note}</div>
                    </div>
                    <div class="hl-tile">
                        <div class="hl-label">Humidity</div>
                        <div class="hl-val">{hum_str} <span class="hl-unit">%</span></div>
                        <div class="hl-foot">{hum_note}</div>
                    </div>
                    <div class="hl-tile">
                        <div class="hl-label">UV Index</div>
                        <div class="hl-val">{uv_str} <span class="hl-unit">UV</span></div>
                        <div class="hl-foot">{uv_note}</div>
                    </div>
                    <div class="hl-tile">
                        <div class="hl-label">Cloud Cover</div>
                        <div class="hl-val">{sim_cloud:.0f} <span class="hl-unit">%</span></div>
                        <div class="hl-foot">Sky opacity index</div>
                    </div>
                    <div class="hl-tile">
                        <div class="hl-label">Solar Radiation</div>
                        <div class="hl-val">{sim_rad:.0f} <span class="hl-unit">W/m²</span></div>
                        <div class="hl-foot">Shortwave irradiance</div>
                    </div>
                    <div class="hl-tile">
                        <div class="hl-label">Fuzzy Comfort</div>
                        <div class="hl-val">{fuzzy_res.recommended_closure_pct:.1f} <span class="hl-unit">%</span></div>
                        <div class="hl-foot">{fuzzy_res.comfort_label.split('(')[0].strip()}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- ROW 3: Stacked Subsystem Nodes (Left) + Matte-Black Supervisor Card (Right) ---
    bot_left, bot_right = st.columns([1.45, 1.15])

    with bot_left:
        hw_status_txt = (
            f"Online (Simulated) • DHT11 {bridge.sensors.temperature_c}°C"
            if bridge.sensors.connected
            else "Disconnected • Software Fallback Active"
        )
        hw_badge = "Online" if bridge.sensors.connected else "Offline"
        api_source_txt = weather_packet.source if weather_packet else "Manual Sliders"

        st.markdown(
            f"""
            <div class="white-panel">
                <div class="panel-heading">Subsystem Telemetry & Actuation Nodes</div>
                <div class="source-row">
                    <div>
                        <div class="src-kicker">Cloud Weather Feed</div>
                        <div class="src-title">Open-Meteo / Replay Vector</div>
                        <div class="src-status">{api_source_txt} • Rain 1h: {sim_rain_mm:.2f} mm</div>
                    </div>
                    <div class="src-right">{f'{sim_temp:.1f}°C' if sim_temp is not None else 'N/A'}</div>
                </div>
                <div class="source-row">
                    <div>
                        <div class="src-kicker">Edge Microclimate Node</div>
                        <div class="src-title">ESP32 Arena Sensor Rig</div>
                        <div class="src-status">{hw_status_txt}</div>
                    </div>
                    <div class="src-right">{hw_badge}</div>
                </div>
                <div class="source-row" style="margin-bottom:0;">
                    <div>
                        <div class="src-kicker">Roof Drive Mechanism</div>
                        <div class="src-title">Retractable Canopy Actuator</div>
                        <div class="src-status">Authority: {decision.decision_source}</div>
                    </div>
                    <div class="src-right">{active_pos:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with bot_right:
        st.markdown(
            f"""
            <div class="dark-promo-card">
                <div>
                    <span class="dark-card-tag">SAFETY SUPERVISOR</span>
                    <div class="dark-card-title">Three-Layer Hybrid Arbitration Active</div>
                    <div class="dark-card-body">
                        Wind safety limits ({supervisor.config.critical_wind_kmh:.0f} km/h) strictly override rain closure.
                        Anti-chattering deadband ({supervisor.config.min_deadband_pct:.0f}%) and cooldown ({supervisor.config.cooldown_seconds:.0f}s) protect motor actuators.
                    </div>
                </div>
                <div style="margin-top:1.1rem; padding-top:0.85rem; border-top:1px solid #26282E; font-size:0.82rem; color:#10B981; font-weight:600;">
                    Current Dispatch: {'Physical Bench Loopback' if decision.physical_actuation_permitted else 'Virtual Digital Twin Only'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: SENSORS & WEATHER (5-Hour Trajectory Strip + Cloud vs ESP32 Tables)
# ==============================================================================
elif nav_page == "Sensors & Weather":
    render_page_header(
        "Sensors & Weather",
        "Hourly Microclimate Trajectory, Open-Meteo Cloud Stream, and ESP32 Hardware Bench",
        active_mode,
    )

    if active_mode is None:
        st.info("System is in Standby. Go to Overview and select an operating mode to stream telemetry.")
        st.stop()

    # Horizontal 5-Step Hourly Strip (Inspired by Forecast Strip in Reference UI)
    st.markdown("#### Hourly Microclimate Trajectory")
    fc_cols = st.columns(5)
    if active_mode == MODE_3_MANUAL_SIM and st.session_state.get("m3_source", "Historical CSV Replay") == "Historical CSV Replay":
        max_r = st.session_state.csv_replay_svc.total_rows() - 1
        base_i = st.session_state.csv_idx
        for step in range(5):
            r_idx = min(max_r, base_i + step)
            pkt, _ = st.session_state.csv_replay_svc.get_row_packet(r_idx)
            t_val = pkt.raw_current["temperature_2m"]
            r_val = pkt.raw_current["rain_mm"]
            c_val = pkt.raw_current["cloud_cover"]
            ts_short = str(pkt.timestamp)[-5:] if len(str(pkt.timestamp)) >= 5 else f"+{step}h"
            lbl = f"Now ({ts_short})" if step == 0 else f"+{step}h ({ts_short})"
            svg_mini = get_weather_svg(is_rain=(r_val >= 0.1), cloud_pct=c_val, size=44)
            card_cls = "fc-card fc-card-now" if step == 0 else "fc-card"
            with fc_cols[step]:
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <div class="fc-time">{lbl}</div>
                        <div>{svg_mini}</div>
                        <div class="fc-temp">{t_val:.1f}°C</div>
                        <div class="fc-rain">Rain: {r_val:.2f} mm</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        # Show T-2h, T-1h (from lag features if present), Current, and +1h AI Forecast
        fdict = weather_packet.feature_dict if (weather_packet and weather_packet.valid) else {}
        t_now = float(sim_temp) if sim_temp is not None else 30.0
        t_lag1 = float(fdict.get("temperature_2m_lag1", t_now - 0.3))
        t_lag2 = float(fdict.get("temperature_2m_lag2", t_now - 0.6))
        r_lag1 = float(fdict.get("rain_lag1", 0.0))
        r_lag2 = float(fdict.get("rain_lag2", 0.0))
        ai_p = (ai_res.rain_probability * 100.0) if (ai_res and ai_res.available) else 0.0

        slots = [
            ("T - 2h", t_lag2, f"Rain: {r_lag2:.1f} mm", r_lag2 >= 0.1, False),
            ("T - 1h", t_lag1, f"Rain: {r_lag1:.1f} mm", r_lag1 >= 0.1, False),
            ("Current", t_now, f"Rain: {sim_rain_mm:.1f} mm", sim_rain_mm >= 0.1, True),
            ("Next +1h (AI)", t_now - 0.5 if ai_p >= 17.5 else t_now, f"Prob: {ai_p:.1f}%", ai_p >= 17.5, False),
            ("Target Shade", t_now, f"Canopy: {active_pos:.0f}%", bool(sim_local_rain or (ai_res and ai_res.rain_predicted)), False),
        ]
        for step, (title_s, temp_s, sub_s, rain_flag, is_now) in enumerate(slots):
            svg_mini = get_weather_svg(is_rain=rain_flag, cloud_pct=sim_cloud, size=44)
            card_cls = "fc-card fc-card-now" if is_now else "fc-card"
            with fc_cols[step]:
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <div class="fc-time">{title_s}</div>
                        <div>{svg_mini}</div>
                        <div class="fc-temp">{temp_s:.1f}°C</div>
                        <div class="fc-rain">{sub_s}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    c_top1, c_top2 = st.columns([3, 1])
    with c_top1:
        st.markdown("#### Software-Only Hardware Bench")
    with c_top2:
        if active_mode != MODE_3_MANUAL_SIM:
            if st.button("Refresh Live Open-Meteo", use_container_width=True):
                st.session_state.cached_live_packet = st.session_state.live_weather_svc.fetch_live_features()
                st.rerun()

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.session_state.sim_hw_online = st.checkbox(
            "Simulate ESP32 Sensor & Actuator Nodes Online", value=st.session_state.sim_hw_online
        )
    with b_col2:
        st.session_state.sim_dht_fault = st.checkbox(
            "Simulate DHT11 Sensor Fault (3x Consecutive Failures)",
            value=st.session_state.sim_dht_fault,
            disabled=not st.session_state.sim_hw_online,
        )

    st.markdown("")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Cloud Weather / Replay Stream")
        if weather_packet and weather_packet.valid:
            st.caption(f"Source: {weather_packet.source} | Timestamp: {weather_packet.timestamp}")
            w_rows = [
                ("Temperature (2m)", "Open-Meteo hourly", f"{weather_packet.raw_current['temperature_2m']:.1f} °C", "Valid", "pill-ok"),
                ("Relative Humidity", "Open-Meteo hourly", f"{weather_packet.raw_current['relative_humidity_2m']:.0f} %", "Valid", "pill-ok"),
                ("Precipitation (1h)", "Rain volume", f"{weather_packet.raw_current['rain_mm']:.2f} mm", "Rain >= 0.1 mm", "pill-alert" if weather_packet.raw_current['rain_mm'] >= 0.1 else "pill-ok"),
                ("Wind Speed (10m)", "Structural check", f"{weather_packet.raw_current['wind_speed_10m']:.1f} km/h", f"Limit {supervisor.config.critical_wind_kmh:.0f} km/h", "pill-alert" if weather_packet.raw_current['wind_speed_10m'] > supervisor.config.critical_wind_kmh else "pill-ok"),
                ("Cloud Cover", "Sky opacity", f"{weather_packet.raw_current['cloud_cover']:.0f} %", "Valid", "pill-ok"),
                ("Shortwave Radiation", "Global solar irradiance", f"{weather_packet.raw_current['shortwave_radiation']:.0f} W/m²", "Valid", "pill-ok"),
            ]
            render_kv_table(w_rows)
        else:
            st.info(
                weather_packet.error_message
                if weather_packet
                else "Manual environmental sliders are active (no cloud weather packet)."
            )

    with right_col:
        st.markdown("#### ESP32 Local Microclimate Node")
        if bridge.sensors.connected:
            st.caption(f"Status: Online (Simulated Bench) | Updated: {bridge.sensors.timestamp_iso}")
            hw_rows = [
                (
                    "DHT11 Temperature",
                    "Local arena air temp",
                    f"{bridge.sensors.temperature_c} °C" if bridge.sensors.temperature_c is not None else "READ FAULT",
                    "3-strike fault filter",
                    "pill-alert" if bridge.sensors.sensor_fault else "pill-ok",
                ),
                (
                    "DHT11 Humidity",
                    "Local arena RH",
                    f"{bridge.sensors.humidity_pct} %" if bridge.sensors.humidity_pct is not None else "READ FAULT",
                    "3-strike fault filter",
                    "pill-alert" if bridge.sensors.sensor_fault else "pill-ok",
                ),
                (
                    "Local Rain Sensor",
                    "Conductive rain plate",
                    "Wet (Rain Detected)" if bridge.sensors.rain_detected else "Dry",
                    "Immediate override",
                    "pill-alert" if bridge.sensors.rain_detected else "pill-ok",
                ),
                (
                    "UV & Light Sensors",
                    "UV index / LDR %",
                    f"UV {bridge.sensors.uv_index} • Light {bridge.sensors.light_pct}%",
                    "Fuzzy comfort inputs",
                    "pill-ok",
                ),
                (
                    "PIR Maintenance Zone",
                    "Local rig proximity only",
                    "Activity" if bridge.sensors.pir_zone_activity else "Idle",
                    "Not stadium crowd count",
                    "pill-ok",
                ),
            ]
            render_kv_table(hw_rows)
        else:
            st.markdown(
                """
                <div class="white-panel">
                    <div class="panel-heading">No Physical ESP32 Connected</div>
                    <div style="color:#6B7280; font-size:0.86rem;">
                        Enable "Simulate ESP32 Sensor & Actuator Nodes Online" above to test local hardware telemetry and DHT11 fault fallback.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# PAGE 3: AI & FUZZY LOGIC
# ==============================================================================
elif nav_page == "AI & Fuzzy Logic":
    render_page_header(
        "AI & Fuzzy Logic",
        "Layer A Rainfall Classifier (HistGBM Regularized) & Layer B Mamdani Comfort Engine",
        active_mode,
    )

    if active_mode is None:
        st.info("System is in Standby. Go to Overview and select an operating mode to view live inference.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        p_txt = f"{ai_res.rain_probability * 100:.2f}%" if ai_res.available else "N/A"
        st.markdown(
            f"""
            <div class="bento-card card-purple">
                <div class="card-label">Layer A • Predicted Rain Probability</div>
                <div class="card-value">{p_txt}</div>
                <div class="card-sub">Model: {ai_res.model_name} • Cutoff: {ai_res.threshold:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="bento-card card-green">
                <div class="card-label">Layer B • Fuzzy Comfort Target</div>
                <div class="card-value">{fuzzy_res.recommended_closure_pct:.1f}%</div>
                <div class="card-sub">{fuzzy_res.comfort_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        gt_txt = (
            ("Actual Rain (1)" if actual_csv_label == 1 else "Actual No Rain (0)")
            if actual_csv_label is not None
            else "Live / Slider Input"
        )
        st.markdown(
            f"""
            <div class="bento-card card-blue">
                <div class="card-label">Validation Reference</div>
                <div class="card-value" style="font-size:1.45rem;">{gt_txt}</div>
                <div class="card-sub">Target definition: >= 0.1 mm rain in next 1h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    col_feat, col_rules = st.columns([1.15, 1.0])

    with col_feat:
        st.markdown("#### 18-Feature Inference Vector (Exact Training Order)")
        if weather_packet and weather_packet.valid:
            feat_df = pd.DataFrame(
                [{"Feature Name": k, "Input Value": round(v, 4)} for k, v in weather_packet.feature_dict.items()]
            )
            st.dataframe(feat_df, use_container_width=True, height=390, hide_index=True)
        else:
            st.info("18-feature time-series vector is unavailable in manual slider mode; AI inference is safely bypassed.")

    with col_rules:
        st.markdown("#### Activated Mamdani Fuzzy Rules")
        st.caption(f"Inference Backend: {fuzzy_res.engine_backend}")
        if fuzzy_res.activated_rules:
            rule_rows = [
                (
                    rule_txt.split("THEN")[0].replace("R", "Rule "),
                    "THEN" + rule_txt.split("THEN")[1],
                    f"{strength:.2f}",
                    "Firing strength",
                    "pill-ok",
                )
                for rule_txt, strength in fuzzy_res.activated_rules
            ]
            render_kv_table(rule_rows)
        else:
            st.info("No thermal/UV comfort rules strongly triggered; neutral baseline applied.")


# ==============================================================================
# PAGE 4: CANOPY CONTROL
# ==============================================================================
elif nav_page == "Canopy Control":
    render_page_header(
        "Canopy Control",
        "Retractable Canopy Schematic, Safety Supervisor Interlocks, and PIN-Locked Override",
        active_mode,
        has_warning=bool(decision and decision.safety_warnings),
    )

    if active_mode is None:
        st.info("System is in Standby. Go to Overview and select an operating mode to operate canopy controls.")
        st.stop()

    render_canopy_schematic(
        closure_pct=active_pos,
        is_raining=bool(sim_local_rain or (ai_res.available and ai_res.rain_predicted)),
        wind_alert=bool(sim_wind is not None and sim_wind > supervisor.config.critical_wind_kmh),
        authority=decision.decision_source,
    )

    st.markdown("")
    left_ctrl, right_ctrl = st.columns([1.15, 1.0])

    with left_ctrl:
        st.markdown("#### Actuator Feedback & Position Integrity")
        act_rows = [
            ("Requested Position", "Commanded target", f"{bridge.actuator.requested_position_pct:.1f}%", f"Cmd {bridge.actuator.last_command_id}", "pill-ok"),
            (
                "Confirmed Physical Position",
                "Limit switch / encoder feedback",
                f"{bridge.actuator.confirmed_position_pct:.1f}%" if bridge.actuator.confirmed_position_pct is not None else "N/A (Disconnected)",
                bridge.actuator.ack_status,
                "pill-ok" if bridge.actuator.confirmed_position_pct is not None else "pill-warn",
            ),
            ("Virtual Simulated Position", "Software digital twin state", f"{bridge.simulated_canopy_pct:.1f}%", "In-memory model", "pill-ok"),
            ("Anti-Chattering Filter", "Minimum movement deadband", f"{supervisor.config.min_deadband_pct:.0f}% delta", f"Cooldown {supervisor.config.cooldown_seconds:.0f}s", "pill-ok"),
        ]
        render_kv_table(act_rows)

        with st.expander("Adjust Safety Supervisor Thresholds", expanded=False):
            w_lim = st.number_input("Critical Wind Limit (km/h)", 15.0, 90.0, supervisor.config.critical_wind_kmh, 1.0)
            w_pos = st.slider("Wind-Safe Posture (%)", 0.0, 100.0, supervisor.config.wind_safe_posture_pct, 5.0)
            db_pct = st.slider("Minimum Deadband (%)", 0.0, 25.0, supervisor.config.min_deadband_pct, 1.0)
            cd_sec = st.slider("Cooldown Timer (sec)", 0.0, 300.0, supervisor.config.cooldown_seconds, 5.0)
            supervisor.config.critical_wind_kmh = w_lim
            supervisor.config.wind_safe_posture_pct = w_pos
            supervisor.config.min_deadband_pct = db_pct
            supervisor.config.cooldown_seconds = cd_sec
            if st.button("Reset Cooldown Timer Now"):
                supervisor.reset_cooldown()
                st.success("Cooldown timer reset.")

    with right_ctrl:
        st.markdown("#### Authorized Manual Motor Override")
        st.caption("Requires operator PIN (default demo PIN: 2026). Still subject to E-Stop and Critical Wind overrides.")

        p_in, p_btn = st.columns([2, 1])
        pin_val = p_in.text_input("Operator PIN", type="password", label_visibility="collapsed", placeholder="Enter PIN (2026)")
        with p_btn:
            if st.button("Verify PIN", use_container_width=True):
                st.session_state.pin_unlocked = supervisor.verify_pin(pin_val)
                if not st.session_state.pin_unlocked:
                    st.error("Invalid PIN")

        if st.session_state.pin_unlocked:
            st.success("Manual Controls Unlocked")
            c_b1, c_b2 = st.columns(2)
            if c_b1.button("Open Canopy (0%)", use_container_width=True):
                st.session_state.queued_manual_cmd = {"pin": pin_val, "action": "MOVE", "target_pct": 0.0}
                st.rerun()
            if c_b2.button("Half Shade (50%)", use_container_width=True):
                st.session_state.queued_manual_cmd = {"pin": pin_val, "action": "MOVE", "target_pct": 50.0}
                st.rerun()
            c_b3, c_b4 = st.columns(2)
            if c_b3.button("Close Canopy (100%)", use_container_width=True):
                st.session_state.queued_manual_cmd = {"pin": pin_val, "action": "MOVE", "target_pct": 100.0}
                st.rerun()
            if c_b4.button("Stop Motor", use_container_width=True):
                st.session_state.queued_manual_cmd = {"pin": pin_val, "action": "STOP", "target_pct": 0.0}
                st.rerun()
        else:
            st.markdown(
                """
                <div class="white-panel">
                    <div class="panel-heading">Controls Locked</div>
                    <div style="color:#6B7280; font-size:0.86rem;">
                        Enter PIN 2026 above to unlock manual Open, Half, Close, and Stop motor commands.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# PAGE 5: ANALYTICS & LOGS
# ==============================================================================
elif nav_page == "Analytics & Logs":
    render_page_header(
        "Analytics & Logs",
        "Persistent SQLite Decision History & Completed Offline Walk-Forward ML Benchmarks",
        active_mode,
    )

    st.markdown("#### Persistent Telemetry & Decision Audit Log (SQLite)")
    logs_df = logger.fetch_recent(limit=200)
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True, height=260, hide_index=True)
        dl1, dl2, _ = st.columns([1, 1, 3])
        dl1.download_button(
            "Download CSV",
            logs_df.to_csv(index=False).encode("utf-8"),
            file_name="stadium_canopy_audit_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if dl2.button("Clear Logs", use_container_width=True):
            logger.clear_logs()
            st.rerun()
    else:
        st.info("No runtime records logged yet. Select an operating mode in Overview to begin recording.")

    st.markdown("---")
    st.markdown("#### Completed Offline Model Comparison (5-Fold Chronological Walk-Forward)")
    if MODEL_COMPARISON_PATH.exists():
        st.dataframe(pd.read_csv(MODEL_COMPARISON_PATH), use_container_width=True, hide_index=True)

    st.markdown("#### Validation & Seasonality Figures")
    fig_files = [
        ("Model Comparison (Walk-Forward F2 & Recall)", "model_comparison.png"),
        ("Confusion Matrix (Validation Fold)", "confusion_matrix_validation.png"),
        ("Precision-Recall Curve (Threshold 0.175)", "precision_recall_curve_validation.png"),
        ("Permutation Feature Importance", "permutation_feature_importance.png"),
        ("Feature Correlation Heatmap", "correlation_heatmap.png"),
        ("Kuala Lumpur Monthly Rainfall Seasonality", "monthly_rain_by_year.png"),
    ]
    cols = st.columns(2)
    for idx, (title, fname) in enumerate(fig_files):
        fpath = REPORTS_FIGURES_DIR / fname
        if fpath.exists():
            cols[idx % 2].image(str(fpath), caption=title, use_container_width=True)