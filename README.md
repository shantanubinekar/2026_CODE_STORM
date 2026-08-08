<<<<<<< HEAD
# 2026_CODE_STORM
=======
# 🌉 BridgePulse

**Mathematical Structural Health Monitoring Prototype**

> ⚠️ All sensor data is **SIMULATED** for demonstration purposes.  
> This prototype demonstrates a mathematical anomaly detection mechanism — not a clinically validated collapse prediction system.

## What It Does

BridgePulse monitors a simulated bridge using four sensor channels (strain, vibration, tilt, temperature) and applies a state-space mathematical model to detect deviations from expected structural behaviour:

```
Sensors → State Vector → Expected State → Residual → ‖Residual‖₂ → Threshold → NORMAL / WARNING / CRITICAL
```

## Quick Start

### Option 1: Standalone Web Dashboard (Instant)
Simply open `index.html` in any web browser:
```bash
open index.html
```

### Option 2: Streamlit Dashboard (Python)
```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

The Streamlit dashboard will open automatically at `http://localhost:8501`.

## Live Demo (2 minutes)

1. Open BridgePulse and observe the **four sensor channels** streaming data.
2. Point out the **state vector**, **expected state**, and **residual** in the Mathematical Engine panel.
3. Note the **residual magnitude** stays well below the threshold → status is **NORMAL**.
4. Click **💥 Inject Anomaly**.
5. Watch: sensor values spike → residual magnitude crosses the threshold → status changes to **CRITICAL** → inspection alert appears.
6. Click **🔄 Reset Normal** → the system recovers to **NORMAL**.

## Mathematical Model

**State vector:**

$$x_k = [\text{strain}, \text{vibration}, \text{tilt}, \text{temperature}]^T$$

**State transition:**

$$\hat{x}_{k+1} = A \cdot x_k$$

**Residual:**

$$r_k = x_k - \hat{x}_k$$

**Deviation magnitude:**

$$D_k = \|r_k\|_2$$

**Classification:**

| Condition | Rule |
|-----------|------|
| NORMAL | $D_k < \tau_1$ |
| WARNING | $\tau_1 \leq D_k < \tau_2$ |
| CRITICAL | $D_k \geq \tau_2$ |

## Project Structure

```
BridgePulse/
├── index.html              # Standalone web dashboard (HTML + CSS + JS)
├── app.py                  # Streamlit dashboard (Python entry point)
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
└── src/
    ├── sensor_simulator.py # Simulated bridge sensor data
    └── mathematical_engine.py  # State-space anomaly detection
```

## Tech Stack

- **Python 3** — core language
- **NumPy** — linear algebra and matrix operations
- **Pandas** — data handling
- **Plotly** — interactive graphs
- **Streamlit** — dashboard framework

No databases, no cloud, no external APIs. Everything runs locally.

---

*Built for hackathon Phase 1 — demonstrating mathematical anomaly detection for structural health monitoring.*
>>>>>>> e8f5b8c (First Commit : Made dashboard)
