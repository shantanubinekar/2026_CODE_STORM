"""
mathematical_engine.py — BridgePulse Phase 1

Implements the full anomaly-detection pipeline:

    Observed state   x_k
            ↓
    Expected state   x̂_k = A · x_k
            ↓
    Residual         r_k = x_k − x̂_k
            ↓
    Residual magnitude  D_k = ‖r_k‖₂
            ↓
    Threshold comparison → NORMAL / WARNING / CRITICAL

All mathematics use simple linear algebra (NumPy).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds — tuned so that normal noise stays below WARNING
# and an injected anomaly clearly reaches CRITICAL.
# ---------------------------------------------------------------------------
WARNING_THRESHOLD  = 3.0    # τ₁
CRITICAL_THRESHOLD = 10.0   # τ₂


# ---------------------------------------------------------------------------
# Step 2 — State vector
# ---------------------------------------------------------------------------
def build_state_vector(sensor_data):
    """
    Convert a 1-D array of 4 sensor readings into a 4×1 column vector.
    This is our mathematical state vector x_k.
    """
    return sensor_data.reshape(4, 1)


# ---------------------------------------------------------------------------
# Step 3 — Expected state
# ---------------------------------------------------------------------------
# State-transition matrix A.
# For Phase 1 we use a near-identity matrix.
# Each sensor is expected to stay close to its previous value.
# The slight diagonal values < 1.0 model the expectation that
# vibration and tilt naturally damp toward zero.
A = np.array([
    [1.0,  0.0,  0.0,  0.0],   # strain stays stable
    [0.0,  0.95, 0.0,  0.0],   # vibration damps slightly
    [0.0,  0.0,  0.98, 0.0],   # tilt damps slightly
    [0.0,  0.0,  0.0,  1.0],   # temperature stays stable
])


def calculate_expected_state(x_k):
    """
    Predict the next state using the state-transition matrix.

    Formula:  x̂_{k+1} = A · x_k

    Returns a 4×1 column vector.
    """
    return A @ x_k


# ---------------------------------------------------------------------------
# Step 4 — Residual
# ---------------------------------------------------------------------------
def calculate_residual(x_observed, x_expected):
    """
    Compute the residual (deviation) vector.

    Formula:  r_k = x_observed − x_expected

    A large residual means the bridge is NOT behaving as expected.
    """
    return x_observed - x_expected


# ---------------------------------------------------------------------------
# Step 5 — Residual magnitude (L2 norm)
# ---------------------------------------------------------------------------
def calculate_residual_magnitude(residual):
    """
    Compute the scalar magnitude of the residual vector.

    Formula:  D_k = ‖r_k‖₂  =  sqrt( r₁² + r₂² + r₃² + r₄² )

    This collapses the 4-element residual into ONE number
    that tells us "how far off is the bridge, overall?"
    """
    return float(np.linalg.norm(residual))


# ---------------------------------------------------------------------------
# Step 6 — Threshold-based classification
# ---------------------------------------------------------------------------
def classify_health(residual_magnitude):
    """
    Compare residual magnitude D_k against thresholds.

    Returns one of: "NORMAL", "WARNING", "CRITICAL"
    """
    if residual_magnitude < WARNING_THRESHOLD:
        return "NORMAL"
    elif residual_magnitude < CRITICAL_THRESHOLD:
        return "WARNING"
    else:
        return "CRITICAL"


# ---------------------------------------------------------------------------
# Convenience: run the full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(x_observed_1d, x_prev_1d):
    """
    Execute the complete anomaly-detection pipeline.

    Parameters
    ----------
    x_observed_1d : 1-D array (4,)   — current sensor readings
    x_prev_1d     : 1-D array (4,)   — previous sensor readings

    Returns
    -------
    dict with keys:
        x_observed, x_expected, residual,
        residual_magnitude, status,
        warning_threshold, critical_threshold
    """
    x_prev     = build_state_vector(x_prev_1d)
    x_observed = build_state_vector(x_observed_1d)
    x_expected = calculate_expected_state(x_prev)

    residual           = calculate_residual(x_observed, x_expected)
    residual_magnitude = calculate_residual_magnitude(residual)
    status             = classify_health(residual_magnitude)

    return {
        "x_observed":          x_observed.flatten(),
        "x_expected":          x_expected.flatten(),
        "residual":            residual.flatten(),
        "residual_magnitude":  residual_magnitude,
        "status":              status,
        "warning_threshold":   WARNING_THRESHOLD,
        "critical_threshold":  CRITICAL_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate two consecutive healthy readings
    prev    = np.array([100.0, 0.0, 0.0, 20.0])
    current = np.array([100.3, 0.1, 0.005, 20.1])

    result = run_pipeline(current, prev)

    print("=== Pipeline test (healthy) ===")
    for k, v in result.items():
        print(f"  {k:>22s} : {v}")

    # Simulate an anomalous reading
    anomaly = np.array([115.0, 3.0, 0.5, 22.0])
    result2 = run_pipeline(anomaly, prev)

    print("\n=== Pipeline test (anomaly) ===")
    for k, v in result2.items():
        print(f"  {k:>22s} : {v}")
