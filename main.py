"""
main.py
-------
Run the FT reactor model using a YAML configuration file.
"""

import yaml

from src.feed import build_total_feed
from src.reactor import FTReactor, run_case


def load_config(yaml_path: str) -> dict:
    """Load YAML configuration."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    # Load YAML config
    config = load_config("config.yaml")

    # Build feed stream from YAML
    feed_total = build_total_feed(config)

    # Map YAML conversion field if using reaction_model block
    if "reaction_model" in config:
        config["target_x_co"] = config["reaction_model"].get("target_co_conversion", None)

    # Run reactor model
    reactor = FTReactor(config=config, feed_composition=feed_total)

    results = reactor.run()

    print(results.summary())

    # Generate dataset row (for ML / optimization)
    row = run_case(config=config, feed_composition=feed_total)

    print("\nDataset row:")
    for k, v in row.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
