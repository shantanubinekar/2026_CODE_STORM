"""
sensor_simulator.py — BridgePulse Phase 1

Simulates realistic bridge sensor data for:
  - Strain (µε)
  - Vibration / acceleration (m/s²)
  - Tilt (degrees)
  - Temperature (°C)

All data is SIMULATED. No physical sensors are used.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Baseline values for a healthy bridge
# ---------------------------------------------------------------------------
BASELINES = {
    "strain": 100.0,       # microstrain (µε)
    "vibration": 0.0,      # m/s²
    "tilt": 0.0,           # degrees
    "temperature": 20.0,   # °C
}

# Noise levels for a healthy bridge (standard deviations)
NOISE = {
    "strain": 0.5,
    "vibration": 0.2,
    "tilt": 0.01,
    "temperature": 0.1,
}

# Labels and units for display
SENSOR_LABELS = ["Strain", "Vibration", "Tilt", "Temperature"]
SENSOR_UNITS  = ["µε", "m/s²", "°", "°C"]


# ---------------------------------------------------------------------------
# Healthy sensor readings
# ---------------------------------------------------------------------------
def get_sensor_readings(time_step):
    """
    Generate simulated sensor readings for a HEALTHY bridge at a given time_step.

    Returns a 1-D numpy array: [strain, vibration, tilt, temperature]

    The data looks realistic because:
      - Strain fluctuates slightly around a constant baseline.
      - Vibration oscillates rapidly around zero.
      - Tilt has tiny slow drift.
      - Temperature follows a smooth sine wave (daily cycle) plus noise.
    """
    strain = (
        BASELINES["strain"]
        + np.random.normal(0.0, NOISE["strain"])
    )

    vibration = (
        BASELINES["vibration"]
        + 0.15 * np.sin(time_step / 5.0)          # subtle oscillation
        + np.random.normal(0.0, NOISE["vibration"])
    )

    tilt = (
        BASELINES["tilt"]
        + 0.005 * np.sin(time_step / 80.0)        # very slow drift
        + np.random.normal(0.0, NOISE["tilt"])
    )

    temperature = (
        BASELINES["temperature"]
        + 5.0 * np.sin(time_step / 50.0)          # daily-like cycle
        + np.random.normal(0.0, NOISE["temperature"])
    )

    return np.array([strain, vibration, tilt, temperature])


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------
def inject_anomaly(healthy_reading):
    """
    Take a healthy sensor reading and deliberately corrupt it to simulate
    structural damage.

    Changes applied:
      - Strain:      large spike  (+15 µε)
      - Vibration:   large spike  (+3 m/s²)
      - Tilt:        noticeable shift (+0.5°)
      - Temperature: slight rise  (+2 °C)   (friction / stress heating)

    These values are chosen to CLEARLY exceed our detection threshold
    every time, so the live demo is 100 % reliable.
    """
    anomaly_offsets = np.array([15.0, 3.0, 0.5, 2.0])
    return healthy_reading + anomaly_offsets


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Healthy readings ===")
    for t in range(5):
        r = get_sensor_readings(t)
        print(f"  t={t}: strain={r[0]:.2f} µε, vib={r[1]:.3f} m/s², "
              f"tilt={r[2]:.4f}°, temp={r[3]:.2f} °C")

    print("\n=== Anomalous reading (injected) ===")
    healthy = get_sensor_readings(0)
    bad     = inject_anomaly(healthy)
    print(f"  Healthy : {healthy}")
    print(f"  Anomaly : {bad}")
