from __future__ import annotations

import copy
import json
import random
import sys
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reactor import run_case  # noqa: E402
from ml.surrogate import train_and_save_surrogate  # noqa: E402
from ml.plots import (  # noqa: E402
    plot_delta_p_histogram,
    plot_feature_importance,
    plot_feasibility_scatter,
    plot_predicted_vs_actual,
)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_dataset(rows: list[dict], out_path: Path) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to: {out_path}")
    print(f"Number of rows: {len(df)}")
    return df


def save_dataframe(df: pd.DataFrame, out_path: Path, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"{label} saved to: {out_path}")
    print(f"{label} rows: {len(df)}")


def _stratified_samples(
    min_val: float,
    max_val: float,
    n_samples: int,
    rng: random.Random,
) -> list[float]:
    if n_samples <= 1:
        return [(min_val + max_val) / 2.0]

    width = (max_val - min_val) / n_samples
    values = []

    for i in range(n_samples):
        lo = min_val + i * width
        hi = min(max_val, lo + width)
        values.append(rng.uniform(lo, hi))

    rng.shuffle(values)
    return values


def sample_cases(base_config: dict) -> list[dict]:
    dg = base_config.get("dataset_generation", {})
    n_cases = int(dg.get("n_cases", 250))
    random_seed = int(dg.get("random_seed", 42))
    ranges = dg.get("ranges", {})
    method = dg.get("sampling_method", "stratified_random")
    rng = random.Random(random_seed)

    if not ranges:
        raise ValueError(
            "dataset_generation.ranges is empty or missing in config.yaml. "
            "Make sure the active config file contains the full "
            "'dataset_generation -> ranges' block."
        )

    value_map: dict[str, list[float]] = {}

    for key, bounds in ranges.items():
        if "min" not in bounds or "max" not in bounds:
            raise KeyError(f"Range for '{key}' must contain 'min' and 'max'")

        min_val = bounds["min"]
        max_val = bounds["max"]

        if max_val < min_val:
            raise ValueError(f"Invalid range for '{key}': max < min")

        if method == "stratified_random":
            value_map[key] = _stratified_samples(min_val, max_val, n_cases, rng)
        else:
            raise ValueError(f"Unsupported sampling_method: {method}")

    cases: list[dict] = []

    for i in range(n_cases):
        cfg = copy.deepcopy(base_config)

        for key, values in value_map.items():
            val = values[i]

            if key == "temperature_C":
                cfg["operating_conditions"]["temperature_C"] = float(val)

            elif key == "pressure_bar":
                cfg["operating_conditions"]["pressure_bar"] = float(val)

            elif key == "ghsv_h":
                cfg["design_basis"]["ghsv_h"] = float(val)

            elif key == "target_x_co":
                cfg["target_x_co"] = float(val)

            elif key == "total_flow_kmol_h":
                cfg["feed"]["total_flow_kmol_h"] = float(val)

            elif key == "tube_inner_diameter_m":
                cfg["reactor_geometry"]["tube_inner_diameter_m"] = float(val)
                wall = float(cfg["dataset_generation"].get("tube_wall_thickness_m", 0.0032))
                cfg["reactor_geometry"]["tube_outer_diameter_m"] = float(val) + wall

            elif key == "particle_diameter_m":
                cfg["bed_properties"]["particle_diameter_m"] = float(val)

            elif key == "void_fraction":
                cfg["bed_properties"]["void_fraction"] = float(val)

            elif key == "purge_fraction":
                cfg["loop_configuration"]["purge_fraction"] = float(val)

            elif key == "reactors_max_search":
                cfg["design_basis"]["reactors_max_search"] = int(round(val))

            else:
                raise KeyError(f"Unsupported range key: {key}")

        cases.append(cfg)

    return cases


def build_run_metadata(config: dict, df_all: pd.DataFrame, df_feasible: pd.DataFrame) -> dict:
    dg = config.get("dataset_generation", {})
    constraints = {
        "max_delta_p_bar": config.get("design_basis", {}).get("max_delta_p_bar", 4.0),
        "min_target_fraction": config.get("design_basis", {}).get("min_target_fraction", 0.0),
    }

    n_success = 0
    n_failed = 0
    n_feasible = 0

    if "run_status" in df_all.columns:
        n_success = int((df_all["run_status"] == "success").sum())
        n_failed = int((df_all["run_status"] == "failed").sum())

    if "feasible" in df_all.columns:
        n_feasible = int((df_all["feasible"] == True).sum())  # noqa: E712

    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "random_seed": int(dg.get("random_seed", 42)),
        "n_cases_requested": int(dg.get("n_cases", 250)),
        "n_cases_completed": int(len(df_all)),
        "n_success": n_success,
        "n_failed": n_failed,
        "n_feasible": n_feasible,
        "sampling_method": dg.get("sampling_method", "stratified_random"),
        "ranges": dg.get("ranges", {}),
        "kpis": config.get("kpi_definitions", {}),
        "constraints": constraints,
        "output_files": {
            "dataset_all_csv": dg.get("output_dataset_csv", "data/processed/dataset.csv"),
            "dataset_feasible_csv": dg.get(
                "output_feasible_dataset_csv",
                "data/processed/dataset_feasible.csv",
            ),
            "run_metadata_json": dg.get(
                "output_run_metadata_json",
                "data/processed/run_metadata.json",
            ),
            "model_dir": dg.get("output_model_dir", "models"),
            "plots_dir": dg.get("output_plots_dir", "plots"),
        },
        "training_rows": int(len(df_feasible)),
    }

    return metadata


def run_batch(config_path: str = "config.yaml") -> None:
    full_config_path = PROJECT_ROOT / config_path
    print(f"Loading config from: {full_config_path}")
    config = load_yaml(full_config_path)

    dg = config.get("dataset_generation", {})

    out_csv = PROJECT_ROOT / dg.get("output_dataset_csv", "data/processed/dataset.csv")
    feasible_csv = PROJECT_ROOT / dg.get(
        "output_feasible_dataset_csv",
        "data/processed/dataset_feasible.csv",
    )
    metadata_out = PROJECT_ROOT / dg.get(
        "output_run_metadata_json",
        "data/processed/run_metadata.json",
    )
    model_dir = PROJECT_ROOT / dg.get("output_model_dir", "models")
    plots_dir = PROJECT_ROOT / dg.get("output_plots_dir", "plots")

    cases = sample_cases(config)

    rows: list[dict] = []

    for i, case_config in enumerate(cases, start=1):
        try:
            row = run_case(config=case_config)
            row["case_id"] = i
            row["run_status"] = "success"
        except Exception as exc:
            row = {
                "case_id": i,
                "run_status": "failed",
                "error_message": str(exc),
                "feasible": False,
            }

        rows.append(row)
        print(f"Finished case {i}/{len(cases)} - {row['run_status']}")

    df_all = save_dataset(rows, out_csv)

    if "run_status" in df_all.columns:
        df_success = df_all[df_all["run_status"] == "success"].copy()
    else:
        df_success = df_all.copy()

    if "feasible" in df_success.columns:
        df_feasible = df_success[df_success["feasible"] == True].copy()  # noqa: E712
    else:
        df_feasible = df_success.copy()

    save_dataframe(df_feasible, feasible_csv, "Feasible dataset")

    metadata = build_run_metadata(config, df_all, df_feasible)

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Run metadata saved to: {metadata_out}")

    if df_feasible.empty:
        print("No feasible successful rows available for surrogate training.")
        return

    training_info = train_and_save_surrogate(df_feasible, model_dir, metadata)

    print("Surrogate training complete")
    print(json.dumps(training_info, indent=2))

    # ---------------------------
    # Generate plots
    # ---------------------------
    plot_feasibility_scatter(
        df=df_all,
        out_dir=plots_dir,
        x_col="input_temperature_C",
        y_col="input_pressure_bar",
    )

    plot_delta_p_histogram(
        df=df_all,
        out_dir=plots_dir,
        delta_p_limit=config["design_basis"].get("max_delta_p_bar", 4.0),
    )

    # Feature importance for:
    # 0 -> target_rate_kgph
    # 2 -> specific_energy_kwh_per_kg_target
    # 5 -> delta_p_bar
    plot_feature_importance(
        model_path=training_info["model_path"],
        out_dir=plots_dir,
        target_index=0,
    )

    plot_feature_importance(
        model_path=training_info["model_path"],
        out_dir=plots_dir,
        target_index=2,
    )

    plot_feature_importance(
        model_path=training_info["model_path"],
        out_dir=plots_dir,
        target_index=5,
    )

    plot_predicted_vs_actual(
        model_path=training_info["model_path"],
        training_data_path=training_info["training_data_path"],
        out_dir=plots_dir,
        target_cols=[
            "target_rate_kgph",
            "specific_energy_kwh_per_kg_target",
            "delta_p_bar",
        ],
    )

    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    run_batch()