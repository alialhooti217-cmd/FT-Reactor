from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from .services.simulation_service import SimulationValidationError, run_simulation

bp = Blueprint("web", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@bp.post("/simulate")
def simulate():
    payload = request.get_json(silent=True) or {}
    try:
        result = run_simulation(payload)
        return jsonify(result)
    except SimulationValidationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"status": "error", "message": f"Simulation failed: {exc}"}), 500
