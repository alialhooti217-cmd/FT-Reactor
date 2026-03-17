"""Interactive FT reactor simulation with user-defined inputs.

Allows the user to enter reactor operating parameters at the command line
(pressing Enter uses the default value), runs a full FTReactor simulation,
and displays formatted results including feasibility status and KPIs.

Usage:
    python run_interactive.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feed import build_total_feed
from src.reactor import FTReactor


# ── Parameter definitions ──────────────────────────────────────────────
# Each tuple: (label, config path, default, min, max, unit)
PARAMETERS = [
    ("Temperature",            "operating_conditions.temperature_C",      220.0, 210.0, 235.0, "C"),
    ("Pressure",               "operating_conditions.pressure_bar",        25.0,  20.0,  28.0, "bar"),
    ("GHSV",                   "design_basis.ghsv_h",                   1800.0, 1500.0, 2200.0, "1/h"),
    ("Target CO conversion",   "target_x_co",                             0.72,  0.60,  0.78, "-"),
    ("Total feed flow",        "feed.total_flow_kmol_h",                1200.0, 900.0, 1800.0, "kmol/h"),
    ("Tube inner diameter",    "reactor_geometry.tube_inner_diameter_m",  0.042, 0.040, 0.046, "m"),
    ("Particle diameter",      "bed_properties.particle_diameter_m",     0.0012, 0.0011, 0.0015, "m"),
    ("Void fraction",          "bed_properties.void_fraction",            0.42,  0.41,  0.46, "-"),
    ("Purge fraction",         "loop_configuration.purge_fraction",       0.03,  0.02,  0.06, "-"),
]


def load_config() -> dict:
    """Load the base configuration from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_nested(cfg: dict, dotpath: str, value: float) -> None:
    """Set a value in a nested dict using a dot-separated path."""
    keys = dotpath.split(".")
    d = cfg
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def get_nested(cfg: dict, dotpath: str):
    """Get a value from a nested dict using a dot-separated path."""
    keys = dotpath.split(".")
    d = cfg
    for key in keys:
        d = d[key]
    return d


def prompt_parameter(label: str, default: float, lo: float, hi: float, unit: str) -> float:
    """Prompt the user for a parameter value, with validation."""
    while True:
        prompt_str = f"  {label} [{lo}–{hi} {unit}] (default {default}): "
        raw = input(prompt_str).strip()
        if raw == "":
            return default
        try:
            val = float(raw)
        except ValueError:
            print(f"    Invalid number. Please enter a value between {lo} and {hi}.")
            continue
        if val < lo or val > hi:
            print(f"    Out of range. Please enter a value between {lo} and {hi}.")
            continue
        return val


def print_divider(char: str = "=", width: int = 60) -> None:
    """Print a horizontal divider line."""
    print(char * width)


def print_results(results) -> None:
    """Print formatted simulation results."""
    print()
    print_divider()
    print("  FT REACTOR SIMULATION RESULTS")
    print_divider()

    sections = [
        ("Feed & Conversion", [
            ("Fresh feed flow",         f"{results.fresh_feed_kmol_h:,.2f} kmol/h"),
            ("Reactor inlet flow",      f"{results.reactor_inlet_kmol_h:,.2f} kmol/h"),
            ("Recycle flow",            f"{results.recycle_kmol_h:,.2f} kmol/h"),
            ("Purge flow",              f"{results.purge_kmol_h:,.2f} kmol/h"),
            ("CO single-pass conversion", f"{results.x_co_single_pass:.4f}"),
            ("ASF alpha",               f"{results.alpha:.4f}"),
            ("Loop iterations",         f"{results.loop_iterations}"),
        ]),
        ("Product KPIs", [
            ("C8-C16 product rate",     f"{results.target_rate_kgph:,.2f} kg/h"),
            ("C8-C16 mass fraction",    f"{results.target_fraction:.4f}"),
            ("Total hydrocarbon rate",  f"{results.total_hydrocarbon_rate_kgph:,.2f} kg/h"),
            ("Specific energy",         f"{results.specific_energy_kwh_per_kg_target:.4f} kWh/kg"),
        ]),
        ("Energy", [
            ("Compressor power",        f"{results.compressor_power_mw:.4f} MW"),
            ("Cooling duty",            f"{results.cooling_duty_mw:.4f} MW"),
            ("Heat of reaction",        f"{results.heat_of_reaction_kj_per_kmol_co:,.0f} kJ/kmol_CO"),
        ]),
        ("Reactor Geometry", [
            ("Parallel reactors (N)",   f"{results.n_parallel}"),
            ("Tubes per reactor",       f"{results.nt_per_reactor:,d}"),
            ("Tube length",             f"{results.tube_length_m:.2f} m"),
            ("Shell diameter",          f"{results.shell_diameter_m:.2f} m"),
            ("L/D ratio",              f"{results.l_over_d:.2f}"),
            ("Catalyst volume (total)", f"{results.total_catalyst_volume_m3:.2f} m3"),
            ("Reactor volume (total)",  f"{results.reactor_volume_m3:.2f} m3"),
        ]),
        ("Hydraulics & Gas Properties", [
            ("Superficial velocity",    f"{results.superficial_velocity_m_s:.3f} m/s"),
            ("Pressure drop",           f"{results.delta_p_bar:.3f} bar (limit {results.max_delta_p_bar:.1f})"),
            ("Gas density",             f"{results.gas_density_kg_m3:.3f} kg/m3"),
            ("Cp mixture",             f"{results.cp_mix_kj_kmolk:.2f} kJ/kmol.K"),
            ("r_CO volumetric",        f"{results.rco_kmol_m3_s:.6e} kmol/m3/s"),
        ]),
    ]

    for section_title, rows in sections:
        print()
        print(f"  --- {section_title} ---")
        max_label = max(len(r[0]) for r in rows)
        for label, value in rows:
            print(f"  {label:<{max_label}}  :  {value}")

    print()
    print_divider("-")
    status = "FEASIBLE" if results.feasible else "INFEASIBLE"
    print(f"  Status  :  {status}")
    print(f"  Reason  :  {results.violation_reason}")
    print_divider()
    print()


def main() -> None:
    """Run the interactive simulation loop."""
    print()
    print_divider()
    print("  FT REACTOR - INTERACTIVE SIMULATION")
    print_divider()
    print()
    print("  Enter values for each parameter below.")
    print("  Press Enter to use the default value.")
    print("  Type 'q' at any prompt to quit.")
    print()

    base_config = load_config()

    while True:
        config = copy.deepcopy(base_config)

        print_divider("-")
        print("  INPUT PARAMETERS")
        print_divider("-")

        quit_flag = False
        for label, dotpath, default, lo, hi, unit in PARAMETERS:
            prompt_str = f"  {label} [{lo}–{hi} {unit}] (default {default}): "
            raw = input(prompt_str).strip()
            if raw.lower() == "q":
                quit_flag = True
                break
            if raw == "":
                val = default
            else:
                try:
                    val = float(raw)
                except ValueError:
                    print(f"    Invalid input, using default {default}")
                    val = default
                if val < lo or val > hi:
                    print(f"    Out of range [{lo}–{hi}], clamping.")
                    val = max(lo, min(hi, val))

            set_nested(config, dotpath, val)

            # Keep tube OD consistent if tube ID changes
            if dotpath == "reactor_geometry.tube_inner_diameter_m":
                wall = 0.0032
                set_nested(config, "reactor_geometry.tube_outer_diameter_m", val + wall)

        if quit_flag:
            print("\n  Goodbye!\n")
            break

        print()
        print("  Running simulation...")

        try:
            feed = build_total_feed(config)
            reactor = FTReactor(config=config, feed_composition=feed)
            results = reactor.run()
            print_results(results)
        except Exception as exc:
            print(f"\n  Simulation failed: {exc}\n")

        again = input("  Run another simulation? [Y/n]: ").strip().lower()
        if again in ("n", "no", "q"):
            print("\n  Goodbye!\n")
            break
        print()


if __name__ == "__main__":
    main()
