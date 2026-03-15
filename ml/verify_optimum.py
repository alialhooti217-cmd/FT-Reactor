from pathlib import Path
import json
import yaml
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reactor import run_case  # noqa


OPTIMUM_PATH = Path("models/surrogate_optimum.json")
CONFIG_PATH = Path("config.yaml")


def load_json(path):
    return json.loads(path.read_text())


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    if not OPTIMUM_PATH.exists():
        raise FileNotFoundError("Run optimization first!")

    optimum = load_json(OPTIMUM_PATH)
    config = load_yaml(CONFIG_PATH)

    # Update config with optimum inputs
    config["operating_conditions"]["temperature_C"] = optimum["input_temperature_C"]
    config["operating_conditions"]["pressure_bar"] = optimum["input_pressure_bar"]
    config["design_basis"]["ghsv_h"] = optimum["input_ghsv_h"]
    config["target_x_co"] = optimum["input_target_x_co"]
    config["feed"]["total_flow_kmol_h"] = optimum["input_total_flow_kmol_h"]
    config["reactor_geometry"]["tube_inner_diameter_m"] = optimum["input_tube_inner_diameter_m"]
    config["bed_properties"]["particle_diameter_m"] = optimum["input_particle_diameter_m"]
    config["bed_properties"]["void_fraction"] = optimum["input_void_fraction"]
    config["loop_configuration"]["purge_fraction"] = optimum["input_purge_fraction"]

    print("\nRunning full reactor simulation for optimum...\n")

    result = run_case(config)

    print("\nReal reactor result:\n")

    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:35s} : {v:.6f}")
        else:
            print(f"{k:35s} : {v}")


if __name__ == "__main__":
    main()
