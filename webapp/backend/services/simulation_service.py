from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.feed import build_total_feed
from src.reactor import FTReactor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class SimulationValidationError(ValueError):
    """Raised when incoming simulation inputs are invalid."""


def _load_base_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_input(payload: dict) -> dict:
    required = {
        "temperature_C": (210.0, 235.0),
        "pressure_bar": (20.0, 28.0),
        "ghsv_h": (1500.0, 2200.0),
        "target_x_co": (0.60, 0.78),
        "total_flow_kmol_h": (900.0, 1800.0),
        "tube_inner_diameter_m": (0.040, 0.046),
        "particle_diameter_m": (0.0011, 0.0015),
        "void_fraction": (0.41, 0.46),
        "purge_fraction": (0.020, 0.060),
    }

    clean: dict[str, float] = {}
    for key, (min_v, max_v) in required.items():
        if key not in payload:
            raise SimulationValidationError(f"Missing required field: {key}")
        try:
            value = float(payload[key])
        except (TypeError, ValueError) as exc:
            raise SimulationValidationError(f"Field '{key}' must be numeric") from exc

        if value < min_v or value > max_v:
            raise SimulationValidationError(
                f"Field '{key}'={value} outside allowed range [{min_v}, {max_v}]"
            )
        clean[key] = value

    return clean


def run_simulation(payload: dict) -> dict:
    params = _validate_input(payload)

    config = copy.deepcopy(_load_base_config())
    config["operating_conditions"]["temperature_C"] = params["temperature_C"]
    config["operating_conditions"]["pressure_bar"] = params["pressure_bar"]
    config["design_basis"]["ghsv_h"] = params["ghsv_h"]
    config["target_x_co"] = params["target_x_co"]
    config["feed"]["total_flow_kmol_h"] = params["total_flow_kmol_h"]
    config["reactor_geometry"]["tube_inner_diameter_m"] = params["tube_inner_diameter_m"]
    config["reactor_geometry"]["tube_outer_diameter_m"] = params["tube_inner_diameter_m"] + 0.0032
    config["bed_properties"]["particle_diameter_m"] = params["particle_diameter_m"]
    config["bed_properties"]["void_fraction"] = params["void_fraction"]
    config["loop_configuration"]["purge_fraction"] = params["purge_fraction"]

    feed_total = build_total_feed(config)
    results = FTReactor(config=config, feed_composition=feed_total).run()

    return {
        "status": "ok",
        "inputs": params,
        "outputs": {
            "target_rate_kgph": results.target_rate_kgph,
            "target_fraction": results.target_fraction,
            "specific_energy_kwh_per_kg_target": results.specific_energy_kwh_per_kg_target,
            "compressor_power_mw": results.compressor_power_mw,
            "cooling_duty_mw": results.cooling_duty_mw,
            "delta_p_bar": results.delta_p_bar,
            "feasible": results.feasible,
            "violation_reason": results.violation_reason,
            "shell_diameter_m": results.shell_diameter_m,
            "tube_length_m": results.tube_length_m,
            "l_over_d": results.l_over_d,
            "n_parallel": results.n_parallel,
            "nt_per_reactor": results.nt_per_reactor,
            "loop_iterations": results.loop_iterations,
            "recycle_kmol_h": results.recycle_kmol_h,
            "purge_kmol_h": results.purge_kmol_h,
        },
    }
