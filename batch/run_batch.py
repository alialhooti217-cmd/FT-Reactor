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
    """
    Reads feed composition from your YAML structure:
    feed:
      total_flow_kmol_h: ...
      composition:
        H2: ...
        CO: ...
        ...
    """
    feed_block = config.get("feed", {})
    composition = feed_block.get("composition")

    if composition is None:
        raise ValueError(
            "config.yaml must contain feed -> composition block."
        )

    return copy.deepcopy(composition)


def sample_case(base_config: dict) -> dict:
    """
    Create one random case from the base config.
    Adjust ranges later as needed.
    """
    cfg = copy.deepcopy(base_config)

    cfg["operating_conditions"]["temperature_C"] = random.uniform(210.0, 260.0)
    cfg["operating_conditions"]["pressure_bar"] = random.uniform(20.0, 30.0)

    cfg["design_basis"]["ghsv_h"] = random.uniform(1500.0, 2500.0)
    cfg["target_x_co"] = random.uniform(0.50, 0.90)

    cfg["bed_properties"]["particle_diameter_m"] = random.uniform(0.0005, 0.0015)
    cfg["bed_properties"]["void_fraction"] = random.uniform(0.40, 0.68)

    return cfg


def run_batch(
    config_path: str = "config.yaml",
    out_csv: str = "data/processed/dataset.csv",
    n_cases: int = 20,
    random_seed: int = 42,
) -> None:
    random.seed(random_seed)

    config = load_yaml(PROJECT_ROOT / config_path)
    base_feed = get_base_feed(config)

    rows: list[dict] = []

    for i in range(n_cases):
        case_config = sample_case(config)

        try:
            row = run_case(config=case_config, feed_composition=base_feed)
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