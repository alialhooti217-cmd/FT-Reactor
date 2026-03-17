"""Shared surrogate model loader and prediction interface.

Used by both the Streamlit dashboard (app.py) and the terminal UI
(run_interactive.py) to get instant ML-based predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # repo root (one level up from ml/)
MODEL_PATH = PROJECT_ROOT / "models" / "surrogate_model.joblib"
META_PATH  = PROJECT_ROOT / "models" / "surrogate_metadata.json"

# Column name mapping: short key → feature column name in model
FEATURE_MAP = {
    "temperature_C":         "input_temperature_C",
    "pressure_bar":          "input_pressure_bar",
    "ghsv_h":                "input_ghsv_h",
    "target_x_co":           "input_target_x_co",
    "total_flow_kmol_h":     "input_total_flow_kmol_h",
    "tube_inner_diameter_m": "input_tube_inner_diameter_m",
    "particle_diameter_m":   "input_particle_diameter_m",
    "void_fraction":         "input_void_fraction",
    "purge_fraction":        "input_purge_fraction",
}

TARGET_COLUMNS = [
    "target_rate_kgph",
    "target_fraction",
    "specific_energy_kwh_per_kg_target",
    "compressor_power_mw",
    "cooling_duty_mw",
    "delta_p_bar",
]


def surrogate_available() -> bool:
    """Return True if both the model file and metadata exist."""
    return MODEL_PATH.exists() and META_PATH.exists()


def load_surrogate() -> tuple:
    """Load and return (model, metadata). Raises FileNotFoundError if missing."""
    if not surrogate_available():
        raise FileNotFoundError(
            "Surrogate model not found. Run `python batch/run_batch.py` first."
        )
    model = joblib.load(MODEL_PATH)
    meta  = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


def predict(params: dict, model=None) -> dict:
    """Predict KPIs from *params* using the surrogate model.

    Args:
        params: Dict with keys matching FEATURE_MAP (e.g. 'temperature_C').
        model:  Pre-loaded model (optional). If None, loads from disk.

    Returns:
        Dict mapping each TARGET_COLUMN to its predicted float value,
        plus 'feasible' (bool) and 'violation_reason' (str).
    """
    if model is None:
        model, _ = load_surrogate()

    # Build the feature row in the exact order the model expects
    row = {col: params[key] for key, col in FEATURE_MAP.items()}
    X = pd.DataFrame([row])

    y_pred = model.predict(X)[0]
    result = dict(zip(TARGET_COLUMNS, (float(v) for v in y_pred)))

    # Evaluate feasibility constraints
    max_delta_p = 4.0
    min_target_fraction = 0.08
    violations = []

    if result["delta_p_bar"] > max_delta_p:
        violations.append(f"ΔP {result['delta_p_bar']:.2f} > {max_delta_p} bar")
    if result["target_fraction"] < min_target_fraction:
        violations.append(
            f"C8–C16 fraction {result['target_fraction']:.4f} < {min_target_fraction}"
        )

    result["feasible"]         = len(violations) == 0
    result["violation_reason"] = "; ".join(violations) if violations else "All constraints satisfied"
    result["source"]           = "ML Surrogate"
    return result


def r2_summary(meta: dict) -> dict[str, float]:
    """Return a dict of {target_column: r2_score} from surrogate metadata."""
    return {
        col: meta.get("metrics", {}).get(col, {}).get("r2", float("nan"))
        for col in TARGET_COLUMNS
    }
