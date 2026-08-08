"""
app.py — BridgePulse Phase 1 Dashboard

Pixel-perfect match to the reference design.
Run with:   streamlit run app.py
"""

import sys
import os
import time
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

# ---------------------------------------------------------------------------
# Import our modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sensor_simulator import (
    get_sensor_readings,
    inject_anomaly,
    SENSOR_LABELS,
    SENSOR_UNITS,
    BASELINES,
)
from mathematical_engine import (
    run_pipeline,
    WARNING_THRESHOLD,
    CRITICAL_THRESHOLD,
)

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                           ║
# ╚═════════════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="BridgePulse — Bridge Health Monitoring Dashboard",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  LOAD BRIDGE IMAGE                                                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝
BRIDGE_IMG_PATH = os.path.join(os.path.dirname(__file__), "assets", "bridge.jpg")
bridge_b64 = ""
if os.path.exists(BRIDGE_IMG_PATH):
    with open(BRIDGE_IMG_PATH, "rb") as f:
        bridge_b64 = base64.b64encode(f.read()).decode()

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  CSS — Exact match to reference design                                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* === GLOBAL === */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: #f5f6fa;
}
/* Remove default padding */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}

/* === HEADER BAR === */
.header-bar {
    background: #ffffff;
    border-bottom: 1px solid #e8eaed;
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 16px -1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-logo {
    display: flex;
    align-items: center;
    gap: 8px;
}
.header-logo-icon {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.header-logo-icon svg {
    width: 28px;
    height: 28px;
}
.header-logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.3px;
}
.header-sep {
    width: 1px;
    height: 24px;
    background: #d1d5db;
    margin: 0 4px;
}
.header-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #374151;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 20px;
    font-size: 0.85rem;
    color: #6b7280;
}
.header-date {
    font-weight: 500;
    color: #374151;
}
.live-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 500;
    color: #374151;
}
.live-dot {
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse-live 1.5s ease-in-out infinite;
}
@keyframes pulse-live {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}
.header-icons {
    display: flex;
    align-items: center;
    gap: 14px;
}
.header-icon-btn {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    position: relative;
    font-size: 1rem;
    border: 1px solid #e5e7eb;
}
.notif-dot {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 16px;
    height: 16px;
    background: #ef4444;
    border-radius: 50%;
    border: 2px solid #ffffff;
    font-size: 0.55rem;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* === CARDS === */
.card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    border: 1px solid #f0f1f3;
    overflow: hidden;
}

/* === BRIDGE HERO === */
.bridge-card {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    height: 340px;
}
.bridge-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    background-size: cover;
    background-position: center;
}
.bridge-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 20px 24px;
    background: linear-gradient(transparent 0%, rgba(0,0,0,0.55) 100%);
}
.bridge-name {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
/* Sensor tags on bridge */
.stag {
    position: absolute;
    background: rgba(255,255,255,0.93);
    backdrop-filter: blur(6px);
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #374151;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}
.stag::before {
    content: '';
    width: 6px;
    height: 6px;
    background: #22c55e;
    border-radius: 50%;
    flex-shrink: 0;
}
.stag.alert::before {
    background: #ef4444;
}
/* Connector lines from tags */
.stag-line {
    position: absolute;
    border-left: 1.5px dashed rgba(55,65,81,0.25);
}

/* === METRIC CARDS (right of bridge) === */
.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #f0f1f3;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 10px;
}
.metric-icon {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.metric-icon svg {
    width: 24px;
    height: 24px;
}
.metric-body {
    flex: 1;
    min-width: 0;
}
.metric-label {
    font-size: 0.73rem;
    font-weight: 500;
    color: #9ca3af;
    line-height: 1.3;
}
.metric-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #1f2937;
    letter-spacing: -0.5px;
}
.metric-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-green { background: #22c55e; }
.dot-blue  { background: #3b82f6; }
.dot-orange { background: #f59e0b; }
.dot-red   { background: #ef4444; }

/* === RECENT ALERTS === */
.alerts-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #f0f1f3;
    height: 100%;
}
.alerts-heading {
    font-size: 1rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 14px;
}
.alert-row {
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-green {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
}
.alert-orange {
    background: #fffbeb;
    border: 1px solid #fde68a;
}
.alert-red {
    background: #fef2f2;
    border: 1px solid #fecaca;
}
.alert-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.alert-triangle {
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 10px solid #f59e0b;
    flex-shrink: 0;
}
.alert-content {
    flex: 1;
}
.alert-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #374151;
}
.alert-time {
    font-size: 0.7rem;
    color: #9ca3af;
    font-weight: 400;
    font-style: italic;
}

/* === BOTTOM ROW === */
/* Health Score */
.health-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #f0f1f3;
    height: 100%;
}
.health-row {
    display: flex;
    align-items: center;
    justify-content: space-around;
}
.health-left {
    text-align: center;
}
.health-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 12px;
}
.donut-wrap {
    position: relative;
    width: 140px;
    height: 140px;
    margin: 0 auto;
}
.donut-wrap svg {
    transform: rotate(-90deg);
}
.donut-center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
}
.donut-pct {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1f2937;
    line-height: 1;
}
.donut-sub {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-top: 4px;
}
.health-right {
    text-align: center;
    padding-left: 8px;
}
.health-right-label {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 14px;
}
.status-dot-big {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: inline-block;
    margin-bottom: 6px;
}
.health-status-text {
    font-size: 0.92rem;
    font-weight: 600;
    color: #374151;
    line-height: 1.4;
}

/* Chart cards */
.chart-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px 20px 8px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #f0f1f3;
    height: 100%;
}
.chart-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 4px;
}

/* === CRITICAL / WARNING BANNERS === */
.crit-banner {
    background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%);
    border: 1px solid #fecaca;
    border-left: 5px solid #ef4444;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 12px;
}
.warn-banner {
    background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
    border: 1px solid #fde68a;
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 12px;
}
.banner-t {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.banner-t-red { color: #dc2626; }
.banner-t-orange { color: #d97706; }
.banner-body {
    font-size: 0.82rem;
    color: #4b5563;
    line-height: 1.55;
}

/* === MATH PIPELINE === */
.math-section {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #f0f1f3;
    margin-top: 12px;
}
.math-heading {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 12px;
}
.math-grid {
    display: flex;
    gap: 12px;
}
.math-box {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 14px;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.65;
}
.math-box-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.mv { color: #1e293b; font-weight: 600; }
.mg { color: #16a34a; font-weight: 700; }
.mo { color: #d97706; font-weight: 700; }
.mr { color: #dc2626; font-weight: 700; }
.mb { color: #2563eb; font-weight: 700; }

/* === BUTTON OVERRIDES === */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 8px 16px;
    width: 100%;
    border: 1px solid #e5e7eb;
    transition: all 0.2s ease;
}

/* === HIDE STREAMLIT CHROME === */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝
if "time_step" not in st.session_state:
    st.session_state.time_step = 0
if "anomaly_active" not in st.session_state:
    st.session_state.anomaly_active = False
if "history" not in st.session_state:
    st.session_state.history = {
        "time": [], "time_labels": [],
        "strain_obs": [], "strain_exp": [],
        "vibration_obs": [], "vibration_exp": [],
        "tilt_obs": [], "tilt_exp": [],
        "temperature_obs": [], "temperature_exp": [],
        "wind_obs": [], "traffic_obs": [],
        "residual_magnitude": [],
        "status": [],
    }
if "prev_reading" not in st.session_state:
    st.session_state.prev_reading = np.array([
        BASELINES["strain"], BASELINES["vibration"],
        BASELINES["tilt"], BASELINES["temperature"],
    ])
if "alerts_log" not in st.session_state:
    now_t = datetime.now()
    st.session_state.alerts_log = [
        {"type": "green", "icon": "dot", "text": "Normal Monitoring", "time": "1m ago"},
        {"type": "orange", "icon": "tri", "text": "High Wind", "time": "15m ago"},
        {"type": "green", "icon": "dot", "text": "Model Updated", "time": "1h ago"},
        {"type": "orange", "icon": "dot", "text": "All Sensors Online", "time": "2h ago"},
    ]


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  GENERATE DATA                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝
t = st.session_state.time_step
raw_reading = get_sensor_readings(t)

if st.session_state.anomaly_active:
    current_reading = inject_anomaly(raw_reading)
else:
    current_reading = raw_reading

result = run_pipeline(current_reading, st.session_state.prev_reading)
st.session_state.prev_reading = current_reading.copy()

# Simulated wind and traffic for the chart (visual only)
wind_val = 12.0 + 8.0 * np.sin(t / 30.0) + np.random.normal(0, 2.0)
traffic_val = 45.0 + 20.0 * np.sin(t / 20.0) + np.random.normal(0, 5.0)
if st.session_state.anomaly_active:
    wind_val += 25.0
    traffic_val += 30.0

# Create time labels like "10:00", "11:00" etc.
base_time = datetime.now().replace(hour=10, minute=0, second=0) + timedelta(minutes=t * 5)
time_label = base_time.strftime("%H:%M")

h = st.session_state.history
h["time"].append(t)
h["time_labels"].append(time_label)
h["strain_obs"].append(result["x_observed"][0])
h["strain_exp"].append(result["x_expected"][0])
h["vibration_obs"].append(result["x_observed"][1])
h["vibration_exp"].append(result["x_expected"][1])
h["tilt_obs"].append(result["x_observed"][2])
h["tilt_exp"].append(result["x_expected"][2])
h["temperature_obs"].append(result["x_observed"][3])
h["temperature_exp"].append(result["x_expected"][3])
h["wind_obs"].append(wind_val)
h["traffic_obs"].append(traffic_val)
h["residual_magnitude"].append(result["residual_magnitude"])
h["status"].append(result["status"])

st.session_state.time_step += 1

status = result["status"]
rm = result["residual_magnitude"]
now_str = datetime.now().strftime("%H:%M:%S")

# Update alerts when status changes
if status == "CRITICAL":
    st.session_state.alerts_log.insert(0, {"type": "red", "icon": "dot", "text": "Structural Deviation Detected", "time": "just now"})
    st.session_state.alerts_log = st.session_state.alerts_log[:4]
elif status == "WARNING":
    st.session_state.alerts_log.insert(0, {"type": "orange", "icon": "tri", "text": "Elevated Deviation", "time": "just now"})
    st.session_state.alerts_log = st.session_state.alerts_log[:4]


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  HEADER BAR                                                            ║
# ╚═════════════════════════════════════════════════════════════════════════╝
now = datetime.now()
date_display = now.strftime("%A, %B %d, %Y")
notif_count = sum(1 for a in st.session_state.alerts_log if a["type"] in ("red", "orange"))

st.markdown(f"""
<div class="header-bar">
    <div class="header-left">
        <div class="header-logo">
            <div class="header-logo-icon">
                <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 24 L16 8 L28 24" stroke="#1a1a2e" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                    <line x1="4" y1="24" x2="28" y2="24" stroke="#1a1a2e" stroke-width="2.5" stroke-linecap="round"/>
                    <line x1="10" y1="24" x2="13" y2="14" stroke="#1a1a2e" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="22" y1="24" x2="19" y2="14" stroke="#1a1a2e" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="16" y1="8" x2="16" y2="24" stroke="#1a1a2e" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
            </div>
            <span class="header-logo-text">BridgePulse</span>
        </div>
        <div class="header-sep"></div>
        <span class="header-title">Bridge Health Monitoring Dashboard</span>
    </div>
    <div class="header-right">
        <span class="header-date">{date_display}</span>
        <div class="live-indicator">
            <div class="live-dot"></div>
            <span>Live clock</span>
        </div>
        <div class="header-icons">
            <div class="header-icon-btn">
                🔔
                {f'<div class="notif-dot">{notif_count}</div>' if notif_count > 0 else ''}
            </div>
            <div class="header-icon-btn">⚙️</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  CONTROL BUTTONS (small, above content)                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝
bc1, bc2, bc_spacer = st.columns([1, 1, 5])
with bc1:
    if st.button("💥 Inject Anomaly", type="primary", width="stretch"):
        st.session_state.anomaly_active = True
        st.rerun()
with bc2:
    if st.button("🔄 Reset to Normal", width="stretch"):
        st.session_state.anomaly_active = False
        st.session_state.prev_reading = np.array([
            BASELINES["strain"], BASELINES["vibration"],
            BASELINES["tilt"], BASELINES["temperature"],
        ])
        st.session_state.alerts_log = [
            {"type": "green", "icon": "dot", "text": "System Reset — Normal", "time": "just now"},
            {"type": "green", "icon": "dot", "text": "Normal Monitoring", "time": "1m ago"},
            {"type": "green", "icon": "dot", "text": "All Sensors Online", "time": "2m ago"},
            {"type": "green", "icon": "dot", "text": "Model Updated", "time": "5m ago"},
        ]
        st.rerun()


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ALERT BANNER                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════╝
if status == "CRITICAL":
    st.markdown(f"""
    <div class="crit-banner">
        <div class="banner-t banner-t-red">🚨 STRUCTURAL DEVIATION DETECTED</div>
        <div class="banner-body">
            Observed behaviour exceeds the configured critical threshold (D<sub>k</sub> = {rm:.3f} &gt; τ₂ = {CRITICAL_THRESHOLD}).
            <b>Recommended action:</b> Initiate immediate structural inspection.
            <em>This is an early warning indicator — not a guaranteed collapse prediction.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif status == "WARNING":
    st.markdown(f"""
    <div class="warn-banner">
        <div class="banner-t banner-t-orange">⚠️ ELEVATED STRUCTURAL DEVIATION</div>
        <div class="banner-body">
            Residual magnitude above warning level (D<sub>k</sub> = {rm:.3f} &gt; τ₁ = {WARNING_THRESHOLD}).
            <b>Recommended action:</b> Increase monitoring frequency and schedule inspection.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ROW 1 — BRIDGE IMAGE | METRICS | ALERTS                              ║
# ╚═════════════════════════════════════════════════════════════════════════╝
r1_bridge, r1_metrics, r1_alerts = st.columns([3.5, 1.5, 1.5])

# --- Bridge hero ---
with r1_bridge:
    stag_class = "stag alert" if st.session_state.anomaly_active else "stag"
    st.markdown(f"""
    <div class="card bridge-card" style="background-image: url('data:image/jpeg;base64,{bridge_b64}'); background-size:cover; background-position:center;">
        <div class="{stag_class}" style="top:15%; left:12%;">Sensors data</div>
        <div class="{stag_class}" style="top:10%; right:28%;">Sensor data</div>
        <div class="{stag_class}" style="top:42%; right:12%;">Sensor data</div>
        <div class="{stag_class}" style="bottom:28%; left:25%;">Sensors data</div>
        <div class="bridge-overlay">
            <div class="bridge-name">Rajiv Gandhi Sea Link</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Metric cards ---
with r1_metrics:
    # Residual
    res_dot_class = "dot-green" if status == "NORMAL" else ("dot-orange" if status == "WARNING" else "dot-red")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
        </div>
        <div class="metric-body">
            <div class="metric-label">Residual</div>
            <div class="metric-value">{rm:.3f}</div>
        </div>
        <div class="metric-dot {res_dot_class}"></div>
    </div>
    """, unsafe_allow_html=True)

    # Dominant Eigenvalue
    eigenval = 0.93 if status == "NORMAL" else (0.78 if status == "WARNING" else 0.52)
    eig_dot = "dot-blue" if status == "NORMAL" else ("dot-orange" if status == "WARNING" else "dot-red")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a10 10 0 0 1 10 10"/>
                <path d="M12 2a10 10 0 0 0-10 10"/>
                <circle cx="12" cy="12" r="6" fill="none"/>
                <circle cx="12" cy="12" r="2" fill="#6b7280"/>
            </svg>
        </div>
        <div class="metric-body">
            <div class="metric-label">Dominant Eigenvalue</div>
            <div class="metric-value">{eigenval:.2f}</div>
        </div>
        <div class="metric-dot {eig_dot}"></div>
    </div>
    """, unsafe_allow_html=True)

    # Sensors Active
    sensors_active = "64 / 64" if status != "CRITICAL" else "61 / 64"
    sa_dot = "dot-green" if status == "NORMAL" else ("dot-orange" if status == "WARNING" else "dot-red")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M5 12.55a11 11 0 0 1 14.08 0"/>
                <path d="M1.42 9a16 16 0 0 1 21.16 0"/>
                <path d="M8.53 16.11a6 6 0 0 1 6.95 0"/>
                <circle cx="12" cy="20" r="1" fill="#6b7280"/>
            </svg>
        </div>
        <div class="metric-body">
            <div class="metric-label">Sensors Active</div>
            <div class="metric-value">{sensors_active}</div>
        </div>
        <div class="metric-dot {sa_dot}"></div>
    </div>
    """, unsafe_allow_html=True)

    # Predicted Useful Life
    useful_life = 18.2 if status == "NORMAL" else (12.5 if status == "WARNING" else 5.1)
    ul_dot = "dot-green" if status == "NORMAL" else ("dot-orange" if status == "WARNING" else "dot-red")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                <line x1="16" y1="2" x2="16" y2="6"/>
                <line x1="8" y1="2" x2="8" y2="6"/>
                <line x1="3" y1="10" x2="21" y2="10"/>
            </svg>
        </div>
        <div class="metric-body">
            <div class="metric-label">Predicted Remaining<br>Useful Life</div>
            <div class="metric-value">{useful_life} years</div>
        </div>
        <div class="metric-dot {ul_dot}"></div>
    </div>
    """, unsafe_allow_html=True)

# --- Recent Alerts ---
with r1_alerts:
    alert_html = ""
    for alert in st.session_state.alerts_log[:4]:
        if alert["type"] == "green":
            row_cls = "alert-green"
            dot_color = "#22c55e"
        elif alert["type"] == "orange":
            row_cls = "alert-orange"
            dot_color = "#f59e0b"
        else:
            row_cls = "alert-red"
            dot_color = "#ef4444"

        if alert.get("icon") == "tri" and alert["type"] == "orange":
            icon_html = '<div class="alert-triangle"></div>'
        else:
            icon_html = f'<div class="alert-dot" style="background:{dot_color}"></div>'

        alert_html += f"""
        <div class="alert-row {row_cls}">
            {icon_html}
            <div class="alert-content">
                <div class="alert-name">{alert['text']}</div>
                <div class="alert-time">{alert['time']}</div>
            </div>
        </div>
        """

    st.markdown(f"""
    <div class="alerts-card">
        <div class="alerts-heading">Recent Alerts</div>
        {alert_html}
    </div>
    """, unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ROW 2 — HEALTH SCORE | RESIDUAL CHART | SENSOR CHART                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

r2_health, r2_residual, r2_sensors = st.columns([1.2, 2, 2.5])

times = h["time_labels"]

# Plotly light theme
PL = dict(
    template="plotly_white",
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Inter", color="#6b7280", size=11),
    margin=dict(l=45, r=15, t=10, b=35),
    height=250,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, font=dict(size=10),
    ),
    xaxis=dict(gridcolor="#f3f4f6", showgrid=True),
    yaxis=dict(gridcolor="#f3f4f6", showgrid=True),
)

# --- Health Score ---
with r2_health:
    health_score = max(0, min(100, 100 - (rm / CRITICAL_THRESHOLD) * 100))

    if health_score > 70:
        hs_color = "#22c55e"
        hs_label = "EXCELLENT"
        status_text = "Healthy"
        status_cn = "green"
    elif health_score > 40:
        hs_color = "#f59e0b"
        hs_label = "WARNING"
        status_text = "At Risk"
        status_cn = "orange"
    else:
        hs_color = "#ef4444"
        hs_label = "CRITICAL"
        status_text = "Critical"
        status_cn = "red"

    r = 56
    circ = 2 * 3.14159265 * r
    filled = circ * (health_score / 100)
    gap = circ - filled

    st.markdown(f"""
    <div class="health-card">
        <div class="health-row">
            <div class="health-left">
                <div class="health-title">Bridge Health Score</div>
                <div class="donut-wrap">
                    <svg width="140" height="140" viewBox="0 0 140 140">
                        <circle cx="70" cy="70" r="{r}" fill="none" stroke="#f3f4f6" stroke-width="12"/>
                        <circle cx="70" cy="70" r="{r}" fill="none" stroke="{hs_color}" stroke-width="12"
                                stroke-dasharray="{filled} {gap}" stroke-linecap="round"/>
                    </svg>
                    <div class="donut-center">
                        <div class="donut-pct">{health_score:.0f}%</div>
                    </div>
                </div>
                <div class="donut-sub" style="color:{hs_color}; text-align:center; margin-top:6px;">{hs_label}</div>
            </div>
            <div class="health-right">
                <div class="health-right-label">Bridge Status</div>
                <div class="status-dot-big" style="background:{hs_color};"></div>
                <div class="health-status-text">{status_text}<br>({status_cn})</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Residual vs Time ---
with r2_residual:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Residual vs Time</div>', unsafe_allow_html=True)

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=times, y=h["residual_magnitude"],
        mode="lines", name="Residual",
        line=dict(color="#3b82f6", width=2),
    ))
    fig_r.add_hline(
        y=WARNING_THRESHOLD, line_dash="dash", line_color="#ef4444", line_width=1.5,
    )
    fig_r.update_layout(**PL)
    st.plotly_chart(fig_r, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Live Sensor Readings ---
with r2_sensors:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Live Sensor Readings</div>', unsafe_allow_html=True)

    fig_s = go.Figure()
    # 6 sensor lines to match the reference
    traces = [
        ("Strain", h["strain_obs"], "#3b82f6"),
        ("Vibration", h["vibration_obs"], "#a855f7"),
        ("Tilt", h["tilt_obs"], "#22c55e"),
        ("Temperature", h["temperature_obs"], "#f59e0b"),
        ("Wind", h["wind_obs"], "#06b6d4"),
        ("Traffic Load", h["traffic_obs"], "#ef4444"),
    ]
    for name, data, color in traces:
        fig_s.add_trace(go.Scatter(
            x=times, y=data,
            mode="lines", name=name,
            line=dict(color=color, width=1.5),
        ))
    fig_s.update_layout(**PL)
    st.plotly_chart(fig_s, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ROW 3 — MATHEMATICAL ENGINE PIPELINE                                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

obs = result["x_observed"]
exp = result["x_expected"]
res_vec = result["residual"]

if status == "NORMAL":
    dc = "mg"
    dt = "✅ All systems nominal."
    ds = "&lt;"
elif status == "WARNING":
    dc = "mo"
    dt = "⚠️ Schedule inspection."
    ds = "&gt;"
else:
    dc = "mr"
    dt = "🚨 Immediate inspection required."
    ds = "&gt;"

st.markdown(f"""
<div class="math-section">
    <div class="math-heading">🧮 Mathematical Engine — Live Pipeline</div>
    <div class="math-grid">
        <div class="math-box">
            <div class="math-box-title">Observed x<sub>k</sub></div>
            <div class="mv">[ {obs[0]:>8.3f} ] strain</div>
            <div class="mv">[ {obs[1]:>8.3f} ] vibration</div>
            <div class="mv">[ {obs[2]:>8.4f} ] tilt</div>
            <div class="mv">[ {obs[3]:>8.3f} ] temperature</div>
        </div>
        <div class="math-box">
            <div class="math-box-title">Expected x̂<sub>k</sub></div>
            <div class="mv">[ {exp[0]:>8.3f} ]</div>
            <div class="mv">[ {exp[1]:>8.3f} ]</div>
            <div class="mv">[ {exp[2]:>8.4f} ]</div>
            <div class="mv">[ {exp[3]:>8.3f} ]</div>
        </div>
        <div class="math-box">
            <div class="math-box-title">Residual r<sub>k</sub> = x<sub>k</sub> − x̂<sub>k</sub></div>
            <div class="mv">[ {res_vec[0]:>+8.3f} ]</div>
            <div class="mv">[ {res_vec[1]:>+8.3f} ]</div>
            <div class="mv">[ {res_vec[2]:>+8.4f} ]</div>
            <div class="mv">[ {res_vec[3]:>+8.3f} ]</div>
        </div>
        <div class="math-box" style="flex:1.3;">
            <div class="math-box-title">Threshold Decision</div>
            <div class="mv">D<sub>k</sub> = ‖r<sub>k</sub>‖₂ = <span class="mb">{rm:.4f}</span></div>
            <div style="margin:4px 0;"><span class="{dc}">D<sub>k</sub> {ds} τ₁ = {WARNING_THRESHOLD}</span></div>
            <div class="{dc}" style="font-size:1.1rem; margin-top:4px;">→ {status}</div>
            <div style="font-size:0.75rem; color:#6b7280; margin-top:4px;">{dt}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  FOOTER                                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<div style="text-align:center; padding:12px 0 4px 0; color:#9ca3af; font-size:0.7rem;">
    ⚡ SIMULATED SENSOR DATA — Mathematical Structural Health Monitoring Prototype &nbsp;|&nbsp; BridgePulse Phase 1
</div>
""", unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  AUTO-REFRESH                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════╝
time.sleep(1.5)
st.rerun()
