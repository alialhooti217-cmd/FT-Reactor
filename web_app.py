"""Flask web application for the FT Reactor ML Surrogate.

Serves a standalone HTML interface that lets users interactively
adjust reactor parameters and see instant ML predictions.

Usage:
    python web_app.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress sklearn version mismatch warnings on model load
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from ml.predictor import load_surrogate, predict as ml_predict  # noqa: E402

app = Flask(__name__, template_folder="templates")

# Load model once at startup
MODEL, META = load_surrogate()


@app.route("/")
def index():
    """Serve the main HTML interface."""
    return render_template("index.html")


@app.route("/api/metadata")
def metadata():
    """Return model metadata: parameter ranges, metrics, constraints."""
    return jsonify({
        "ranges": META.get("ranges", {}),
        "metrics": META.get("metrics", {}),
        "constraints": META.get("constraints", {}),
        "model_type": META.get("model_type", ""),
        "n_training_rows": META.get("n_training_rows", 0),
        "n_cases_completed": META.get("n_cases_completed", 0),
        "n_feasible": META.get("n_feasible", 0),
        "target_columns": META.get("target_columns", []),
        "feature_columns": META.get("feature_columns", []),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept 9 parameters as JSON, return 6 KPIs + feasibility."""
    data = request.get_json(force=True)
    required_keys = [
        "temperature_C", "pressure_bar", "ghsv_h", "target_x_co",
        "total_flow_kmol_h", "tube_inner_diameter_m", "particle_diameter_m",
        "void_fraction", "purge_fraction",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return jsonify({"error": f"Missing parameters: {missing}"}), 400

    try:
        params = {k: float(data[k]) for k in required_keys}
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid parameter value: {exc}"}), 400

    result = ml_predict(params, model=MODEL)
    return jsonify(result)


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serve plot images from the plots directory."""
    return send_from_directory(PROJECT_ROOT / "plots", filename)


if __name__ == "__main__":
    print("FT Reactor ML Interface running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
