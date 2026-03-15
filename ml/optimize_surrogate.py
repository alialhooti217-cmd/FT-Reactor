from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/surrogate_model.joblib")
META_PATH = Path("models/surrogate_metadata.json")
CONFIG_PATH = Path("config.yaml")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_input(ranges: dict, rng: random.Random) -> dict:
    return {
        "input_temperature_C": rng.uniform(
            ranges["temperature_C"]["min"],
            ranges["temperature_C"]["max"],
        ),
        "input_pressure_bar": rng.uniform(
            ranges["pressure_bar"]["min"],
            ranges["pressure_bar"]["max"],
        ),
        "input_ghsv_h": rng.uniform(
            ranges["ghsv_h"]["min"],
            ranges["ghsv_h"]["max"],
        ),
        "input_target_x_co": rng.uniform(
            ranges["target_x_co"]["min"],
            ranges["target_x_co"]["max"],
        ),
        "input_total_flow_kmol_h": rng.uniform(
            ranges["total_flow_kmol_h"]["min"],
            ranges["total_flow_kmol_h"]["max"],
        ),
        "input_tube_inner_diameter_m": rng.uniform(
            ranges["tube_inner_diameter_m"]["min"],
            ranges["tube_inner_diameter_m"]["max"],
        ),
        "input_particle_diameter_m": rng.uniform(
            ranges["particle_diameter_m"]["min"],
            ranges["particle_diameter_m"]["max"],
        ),
        "input_void_fraction": rng.uniform(
            ranges["void_fraction"]["min"],
            ranges["void_fraction"]["max"],
        ),
        "input_purge_fraction": rng.uniform(
            ranges["purge_fraction"]["min"],
            ranges["purge_fraction"]["max"],
        ),
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    print("Loading surrogate model and metadata...")

    model = joblib.load(MODEL_PATH)
    metadata = load_json(META_PATH)
    config = load_yaml_config(CONFIG_PATH)

    feature_columns = metadata.get("feature_columns", [])
    target_columns = metadata.get("target_columns", [])

    if not feature_columns:
        raise ValueError("feature_columns missing in surrogate_metadata.json")

    if not target_columns:
        raise ValueError("target_columns missing in surrogate_metadata.json")

    ranges = config["dataset_generation"]["ranges"]
    max_delta_p = config["design_basis"]["max_delta_p_bar"]
    min_target_fraction = config["design_basis"]["min_target_fraction"]

    rng = random.Random(42)

    # Change this if needed
    n_trials = 10000

    print(f"Starting surrogate optimization with {n_trials} trials...")
    print(f"Constraints: delta_p_bar <= {max_delta_p}, target_fraction >= {min_target_fraction}")

    start_time = time.time()

    best_row = None
    best_objective = float("inf")

    all_results = []
    feasible_count = 0

    for i in range(n_trials):
        if i % 1000 == 0:
            print(f"Trial {i}/{n_trials}")

        x = sample_input(ranges, rng)

        # Build dataframe in the exact trained feature order
        X = pd.DataFrame([x])
        X = X.reindex(columns=feature_columns)

        missing_cols = [col for col in feature_columns if col not in X.columns]
        if missing_cols:
            raise KeyError(f"Missing required feature columns: {missing_cols}")

        y_pred = model.predict(X)[0]

        result = dict(x)
        for name, value in zip(target_columns, y_pred):
            result[name] = float(value)

        feasible = (
            result["delta_p_bar"] <= max_delta_p
            and result["target_fraction"] >= min_target_fraction
        )
        result["surrogate_feasible"] = feasible

        all_results.append(result)

        if feasible:
            feasible_count += 1

        if feasible and result["specific_energy_kwh_per_kg_target"] < best_objective:
            best_objective = result["specific_energy_kwh_per_kg_target"]
            best_row = result

    elapsed = time.time() - start_time

    print(f"Optimization finished in {elapsed:.2f} s")
    print(f"Feasible surrogate candidates found: {feasible_count}/{n_trials}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("models/surrogate_optimization_trials.csv", index=False)
    print("Saved trial results to: models/surrogate_optimization_trials.csv")

    if best_row is None:
        print("No feasible optimum found in surrogate search.")
        return

    print("\nBest surrogate optimum found:\n")
    for k, v in best_row.items():
        if isinstance(v, float):
            print(f"{k:35s} : {v:.6f}")
        else:
            print(f"{k:35s} : {v}")

    Path("models/surrogate_optimum.json").write_text(
        json.dumps(best_row, indent=2),
        encoding="utf-8",
    )
    print("\nSaved optimum to: models/surrogate_optimum.json")


if __name__ == "__main__":
    main()
