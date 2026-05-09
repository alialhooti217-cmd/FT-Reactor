"""Flask web application for the FT Reactor ML Surrogate.

Serves a standalone HTML interface that lets users interactively
adjust reactor parameters and see instant ML predictions.

Usage:
    python web_app.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import yaml
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


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """Scan 4,000 Latin-Hypercube candidates in one batch predict call, then
    return the best feasible point for the chosen objective."""
    data = request.get_json(force=True)
    goal = data.get("goal", "min_energy")
    if goal not in ("min_energy", "max_rate", "min_dp"):
        return jsonify({"error": f"Invalid goal: {goal}"}), 400

    try:
        import numpy as np
        import pandas as pd

        FEATURE_COLS = [
            "input_temperature_C", "input_pressure_bar", "input_ghsv_h",
            "input_target_x_co", "input_total_flow_kmol_h",
            "input_tube_inner_diameter_m", "input_particle_diameter_m",
            "input_void_fraction", "input_purge_fraction",
        ]
        PARAM_KEYS = [
            "temperature_C", "pressure_bar", "ghsv_h", "target_x_co",
            "total_flow_kmol_h", "tube_inner_diameter_m", "particle_diameter_m",
            "void_fraction", "purge_fraction",
        ]

        lo = np.array([210.0, 20.0, 1500.0, 0.60,  900.0, 0.040,  0.0011, 0.41, 0.020])
        hi = np.array([235.0, 28.0, 2200.0, 0.78, 1800.0, 0.046,  0.0015, 0.46, 0.060])

        N   = 4000
        rng = np.random.default_rng(42)

        # Latin-Hypercube sampling: each dimension gets evenly-spaced strata
        samples = np.zeros((N, 9))
        for j in range(9):
            perm = rng.permutation(N)
            samples[:, j] = (perm + rng.uniform(0.0, 1.0, N)) / N
        X = lo + samples * (hi - lo)

        # Single batch prediction — much faster than N individual calls
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        Y  = MODEL.predict(df)   # shape (N, 6)
        # column order: 0:rate  1:fraction  2:energy  3:comp  4:cool  5:dp

        feasible = (Y[:, 5] <= 4.0) & (Y[:, 1] >= 0.08)
        if feasible.sum() == 0:           # relax ΔP limit slightly
            feasible = Y[:, 5] <= 4.5
        if feasible.sum() == 0:           # fall back: use all candidates
            feasible = np.ones(N, dtype=bool)

        idx = np.where(feasible)[0]
        if goal == "min_energy":
            best = idx[np.argmin(Y[idx, 2])]
        elif goal == "max_rate":
            best = idx[np.argmax(Y[idx, 0])]
        else:  # min_dp
            best = idx[np.argmin(Y[idx, 5])]

        optimal_params = dict(zip(PARAM_KEYS, X[best].tolist()))
        optimal_kpis   = ml_predict(optimal_params, model=MODEL)

        return jsonify({
            "optimal_params":  optimal_params,
            "optimal_kpis":    optimal_kpis,
            "goal":            goal,
            "n_evaluations":   N,
            "feasible_found":  int(feasible.sum()),
            "converged":       True,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/geometry", methods=["POST"])
def geometry():
    """Run full physics simulation to return reactor geometry details."""
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

    try:
        from src.feed import build_total_feed
        from src.reactor import FTReactor

        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["operating_conditions"]["temperature_C"]      = params["temperature_C"]
        config["operating_conditions"]["pressure_bar"]       = params["pressure_bar"]
        config["design_basis"]["ghsv_h"]                    = params["ghsv_h"]
        config["target_x_co"]                               = params["target_x_co"]
        config["feed"]["total_flow_kmol_h"]                 = params["total_flow_kmol_h"]
        config["reactor_geometry"]["tube_inner_diameter_m"] = params["tube_inner_diameter_m"]
        config["reactor_geometry"]["tube_outer_diameter_m"] = params["tube_inner_diameter_m"] + 0.0032
        config["bed_properties"]["particle_diameter_m"]     = params["particle_diameter_m"]
        config["bed_properties"]["void_fraction"]           = params["void_fraction"]
        config["loop_configuration"]["purge_fraction"]      = params["purge_fraction"]

        feed = build_total_feed(config)
        r = FTReactor(config=config, feed_composition=feed).run()

        return jsonify({
            "tube_inner_diameter_m":   params["tube_inner_diameter_m"],
            "tube_outer_diameter_m":   params["tube_inner_diameter_m"] + 0.0032,
            "tube_length_m":           r.tube_length_m,
            "shell_diameter_m":        r.shell_diameter_m,
            "l_over_d":                r.l_over_d,
            "n_parallel":              r.n_parallel,
            "nt_per_reactor":          r.nt_per_reactor,
            "total_catalyst_volume_m3": r.total_catalyst_volume_m3,
            "reactor_volume_m3":       r.reactor_volume_m3,
            "superficial_velocity_m_s": r.superficial_velocity_m_s,
            "x_co_single_pass":        r.x_co_single_pass,
            "alpha":                   r.alpha,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serve plot images from the plots directory."""
    return send_from_directory(PROJECT_ROOT / "plots", filename)


if __name__ == "__main__":
    print("FT Reactor ML Interface running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
