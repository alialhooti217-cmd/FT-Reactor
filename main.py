"""Run the FT loop model using a YAML configuration file."""

from __future__ import annotations

import yaml

from src.feed import build_total_feed
from src.reactor import FTReactor, run_case


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config("config.yaml")
    feed_total = build_total_feed(config)
    reactor = FTReactor(config=config, feed_composition=feed_total)
    results = reactor.run()
    print(results.summary())
    row = run_case(config=config, feed_composition=feed_total)
    print("\nDataset-style row preview:")
    for key in [
        "target_rate_kgph",
        "target_fraction",
        "specific_energy_kwh_per_kg_target",
        "compressor_power_mw",
        "cooling_duty_mw",
        "delta_p_bar",
        "feasible",
        "violation_reason",
    ]:
        print(f"{key}: {row.get(key)}")


if __name__ == "__main__":
    main()
