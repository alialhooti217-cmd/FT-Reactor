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


def _stratified_samples(min_val: float, max_val: float, n_samples: int, rng: random.Random) -> list[float]:
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

    value_map = {}
    for key, bounds in ranges.items():
        min_val = bounds["min"]
        max_val = bounds["max"]
        if method == "stratified_random":
            value_map[key] = _stratified_samples(min_val, max_val, n_cases, rng)
        else:
            raise ValueError(f"Unsupported sampling_method: {method}")

    cases = []
    for i in range(n_cases):
        cfg = copy.deepcopy(base_config)
        for key, values in value_map.items():
            val = values[i]
            if key == "temperature_C":
                cfg["operating_conditions"]["temperature_C"] = val
            elif key == "pressure_bar":
                cfg["operating_conditions"]["pressure_bar"] = val
            elif key == "ghsv_h":
                cfg["design_basis"]["ghsv_h"] = val
            elif key == "target_x_co":
                cfg["target_x_co"] = val
            elif key == "total_flow_kmol_h":
                cfg["feed"]["total_flow_kmol_h"] = val
            elif key == "tube_inner_diameter_m":
                cfg["reactor_geometry"]["tube_inner_diameter_m"] = val
                wall = cfg["dataset_generation"].get("tube_wall_thickness_m", 0.0032)
                cfg["reactor_geometry"]["tube_outer_diameter_m"] = val + wall
            elif key == "particle_diameter_m":
                cfg["bed_properties"]["particle_diameter_m"] = val
            elif key == "void_fraction":
                cfg["bed_properties"]["void_fraction"] = val
            elif key == "purge_fraction":
                cfg["loop_configuration"]["purge_fraction"] = val
            elif key == "reactors_max_search":
                cfg["design_basis"]["reactors_max_search"] = int(round(val))
            else:
                raise KeyError(f"Unsupported range key: {key}")
        cases.append(cfg)
    return cases


def run_batch(config_path: str = "config.yaml") -> None:
    config = load_yaml(PROJECT_ROOT / config_path)
    dg = config.get("dataset_generation", {})
    out_csv = PROJECT_ROOT / dg.get("output_dataset_csv", "data/processed/dataset.csv")
    metadata_out = PROJECT_ROOT / dg.get("output_run_metadata_json", "data/processed/run_metadata.json")
    model_dir = PROJECT_ROOT / dg.get("output_model_dir", "models")

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
            }
        rows.append(row)
        print(f"Finished case {i}/{len(cases)} - {row['run_status']}")

    df = save_dataset(rows, out_csv)

    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "random_seed": int(dg.get("random_seed", 42)),
        "n_cases_requested": int(dg.get("n_cases", 250)),
        "n_cases_completed": int(len(df)),
        "sampling_method": dg.get("sampling_method", "stratified_random"),
        "ranges": dg.get("ranges", {}),
        "kpis": config.get("kpi_definitions", {}),
        "constraints": {
            "max_delta_p_bar": config["design_basis"].get("max_delta_p_bar", 4.0),
            "min_target_fraction": config["design_basis"].get("min_target_fraction", 0.0),
        },
    }

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    training_info = train_and_save_surrogate(df, model_dir, metadata)
    print("Surrogate training complete")
    print(json.dumps(training_info, indent=2))


if __name__ == "__main__":
    run_batch()
