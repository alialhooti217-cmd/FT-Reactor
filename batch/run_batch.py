from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reactor import run_case  # noqa: E402


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_dataset(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to: {out_path}")
    print(f"Number of rows: {len(df)}")


def get_base_feed(config: dict) -> dict:
    feed_block = config.get("feed", {})
    composition = feed_block.get("composition")
    total_flow = feed_block.get("total_flow_kmol_h")

    if composition is None or total_flow is None:
        raise ValueError("config.yaml must contain feed.total_flow_kmol_h and feed.composition")

    component_flows = {
        comp: frac * total_flow
        for comp, frac in composition.items()
    }
    return component_flows


def sample_case(base_config: dict) -> dict:
    cfg = copy.deepcopy(base_config)

    # Operating conditions
    cfg["operating_conditions"]["temperature_C"] = random.uniform(210.0, 260.0)
    cfg["operating_conditions"]["pressure_bar"] = random.uniform(20.0, 30.0)

    # Design basis
    cfg["design_basis"]["ghsv_h"] = random.uniform(1500.0, 2500.0)
    cfg["target_x_co"] = random.uniform(0.50, 0.90)

    # Bed properties
    cfg["bed_properties"]["particle_diameter_m"] = random.uniform(0.0005, 0.0015)
    cfg["bed_properties"]["void_fraction"] = random.uniform(0.40, 0.68)

    # Geometry-driving variables
    cfg["reactor_geometry"]["tube_inner_diameter_m"] = random.uniform(0.020, 0.032)

    # Scale-driving variable
    cfg["feed"]["total_flow_kmol_h"] = random.uniform(5000.0, 50000.0)

    return cfg


def build_feed_from_case_config(case_config: dict) -> dict:
    feed_block = case_config.get("feed", {})
    composition = feed_block.get("composition")
    total_flow = feed_block.get("total_flow_kmol_h")

    if composition is None or total_flow is None:
        raise ValueError("Case config must contain feed.total_flow_kmol_h and feed.composition")

    return {
        comp: frac * total_flow
        for comp, frac in composition.items()
    }


def run_batch(
    config_path: str = "config.yaml",
    out_csv: str = "data/processed/dataset.csv",
    n_cases: int = 20,
    random_seed: int | None = None,
) -> None:
    if random_seed is not None:
        random.seed(random_seed)

    config = load_yaml(PROJECT_ROOT / config_path)

    rows: list[dict] = []

    for i in range(n_cases):
        case_config = sample_case(config)
        case_feed = build_feed_from_case_config(case_config)

        try:
            row = run_case(config=case_config, feed_composition=case_feed)
            row["case_id"] = i + 1
            row["run_status"] = "success"
        except Exception as exc:
            row = {
                "case_id": i + 1,
                "run_status": "failed",
                "error_message": str(exc),
            }

        rows.append(row)
        print(f"Finished case {i + 1}/{n_cases} - {row['run_status']}")

    save_dataset(rows, PROJECT_ROOT / out_csv)


if __name__ == "__main__":
    run_batch()
