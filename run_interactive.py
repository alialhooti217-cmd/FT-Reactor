"""Interactive FT reactor simulation with rich terminal visualisation.

Simulation modes:
  - ML Surrogate (default): instant predictions from the ExtraTreesRegressor
    model trained on 21,000 reactor simulations.
  - Full Physics: runs the complete equation-based FTReactor model.

Usage:
    python run_interactive.py
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import yaml
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from ml.predictor import predict as ml_predict, load_surrogate, r2_summary, surrogate_available

# ── Console ─────────────────────────────────────────────────────────────
theme = Theme({
    "primary":   "bold cyan",
    "accent":    "bold yellow",
    "good":      "bold green",
    "bad":       "bold red",
    "muted":     "dim white",
    "section":   "bold blue",
    "value":     "white",
    "unit":      "dim cyan",
    "ml":        "bold #74b9ff",
    "physics":   "bold #55efc4",
})
console = Console(theme=theme, highlight=False)

# ── Parameters ───────────────────────────────────────────────────────────
PARAMETERS = [
    ("Temperature",          "operating_conditions.temperature_C",      220.0, 210.0, 235.0, "°C"),
    ("Pressure",             "operating_conditions.pressure_bar",        25.0,  20.0,  28.0, "bar"),
    ("GHSV",                 "design_basis.ghsv_h",                   1800.0, 1500.0, 2200.0, "h⁻¹"),
    ("Target CO Conversion", "target_x_co",                             0.72,  0.60,  0.78, ""),
    ("Total Feed Flow",      "feed.total_flow_kmol_h",                1200.0,  500.0, 10000.0, "kmol/h"),
    ("H₂/CO Ratio",         "_composition.h2_co_ratio",                2.2,   1.5,   3.0, ""),
    ("CO₂ Fraction",        "_composition.co2_fraction",               0.02,  0.0,   0.10, ""),
    ("N₂ / Inerts Fraction","_composition.n2_fraction",                0.02,  0.0,   0.15, ""),
    ("Tube Inner Diameter",  "reactor_geometry.tube_inner_diameter_m",  0.042, 0.040, 0.046, "m"),
    ("Particle Diameter",    "bed_properties.particle_diameter_m",     0.0012, 0.0011, 0.0015, "m"),
    ("Void Fraction",        "bed_properties.void_fraction",            0.42,  0.41,  0.46, ""),
    ("Purge Fraction",       "loop_configuration.purge_fraction",       0.03,  0.02,  0.06, ""),
]


def load_config() -> dict:
    """Load base configuration from config.yaml."""
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_nested(cfg: dict, dotpath: str, value: float) -> None:
    """Set a nested dict value using a dot-separated path."""
    keys = dotpath.split(".")
    d = cfg
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def bar_chart(value: float, max_val: float, width: int = 22, over_limit: bool = False) -> Text:
    """Return a Unicode block bar representing value/max_val."""
    filled = max(0, min(width, int(round(value / max_val * width))))
    bar = Text()
    style = "bold red" if over_limit else "bar.full"
    bar.append("█" * filled, style=style)
    bar.append("░" * (width - filled), style="grey23")
    return bar


def print_header() -> None:
    """Print the application header banner."""
    console.print()
    console.rule("[primary]⚗  Fischer–Tropsch Reactor Simulator[/primary]")
    console.print("[muted]  Powered by a surrogate ML model trained on 21,000 simulations[/muted]")
    console.print()


def choose_mode() -> str:
    """Ask user which simulation mode to use. Returns 'ml' or 'physics'."""
    if not surrogate_available():
        console.print("[bad]  Surrogate model not found.[/bad] Using Full Physics.")
        return "physics"

    console.print(Panel(
        "[ml]🤖 ML Surrogate[/ml]  — instant predictions from a trained model\n"
        "[physics]⚗️  Full Physics[/physics]   — runs the full reactor equations (~1 s per run)\n\n"
        "[muted]Type [/muted][accent]1[/accent][muted] for ML, [/muted][accent]2[/accent][muted] for Physics (default: 1)[/muted]",
        title="[primary]Choose Simulation Mode[/primary]",
        border_style="blue",
    ))

    choice = Prompt.ask("  Mode", default="1", choices=["1", "2"], console=console)
    return "ml" if choice == "1" else "physics"


def print_ml_model_info(meta: dict) -> None:
    """Print a summary of the ML model's accuracy."""
    t = Table(box=box.ROUNDED, border_style="blue",
              title="[ml]🧠 ML Model — Validation R² Scores[/ml]",
              show_header=True, header_style="bold blue")
    t.add_column("Target KPI",     style="accent",  min_width=36)
    t.add_column("R² Score",       style="value",   min_width=8,  justify="right")
    t.add_column("Accuracy",       min_width=22)

    for col, r2 in r2_summary(meta).items():
        bar = Text()
        filled = max(0, min(22, int(round(r2 * 22))))
        bar.append("█" * filled, style="#74b9ff")
        bar.append("░" * (22 - filled), style="grey23")
        label = "Excellent" if r2 >= 0.99 else "Good" if r2 >= 0.97 else "Fair"
        t.add_row(col.replace("_", " "), f"{r2:.4f}  {label}", bar)

    console.print(t)
    n = meta.get("n_training_rows", "—")
    console.print(f"  [muted]Trained on {n:,} feasible cases  ·  model: {meta.get('model_type', '')}[/muted]\n")


def print_kpi_bars_ml(result: dict) -> None:
    """Print KPI bar chart for ML prediction results."""
    kpis = [
        ("C8–C16 Rate",      result["target_rate_kgph"],                  1500.0, "kg/h"),
        ("C8–C16 Fraction",  result["target_fraction"] * 100,              25.0,  "%"),
        ("Specific Energy",  result["specific_energy_kwh_per_kg_target"],  60.0,  "kWh/kg"),
        ("Compressor Power", result["compressor_power_mw"],                 3.0,  "MW"),
        ("Cooling Duty",     result["cooling_duty_mw"],                    40.0,  "MW"),
        ("Pressure Drop",    result["delta_p_bar"],                         6.0,  "bar (limit 4.0)"),
    ]

    t = Table(box=box.ROUNDED, border_style="blue",
              title="[ml]🤖 ML Surrogate Predictions[/ml]",
              show_header=True, header_style="bold blue")
    t.add_column("KPI",    style="accent",  min_width=20)
    t.add_column("Value",  style="value",   min_width=12, justify="right")
    t.add_column("Unit",   style="unit",    min_width=14)
    t.add_column("Visual", min_width=24)

    for label, val, max_ref, unit in kpis:
        over = (label == "Pressure Drop" and val > 4.0)
        val_text = Text(f"{val:.3g}")
        if over:
            val_text.stylize("bold red")
        t.add_row(label, val_text, unit, bar_chart(val, max_ref, over_limit=over))

    console.print(t)


def print_kpi_bars_physics(r) -> None:
    """Print KPI bar chart for full physics simulation results."""
    kpis = [
        ("C8–C16 Rate",      r.target_rate_kgph,                  1500.0, "kg/h"),
        ("C8–C16 Fraction",  r.target_fraction * 100,              25.0,  "%"),
        ("Specific Energy",  r.specific_energy_kwh_per_kg_target,  60.0,  "kWh/kg"),
        ("Compressor Power", r.compressor_power_mw,                 3.0,  "MW"),
        ("Cooling Duty",     r.cooling_duty_mw,                    40.0,  "MW"),
        ("Pressure Drop",    r.delta_p_bar,                         6.0,  "bar (limit 4.0)"),
    ]

    t = Table(box=box.ROUNDED, border_style="blue",
              title="[physics]⚗️ Full Physics Simulation Results[/physics]",
              show_header=True, header_style="bold blue")
    t.add_column("KPI",    style="accent",  min_width=20)
    t.add_column("Value",  style="value",   min_width=12, justify="right")
    t.add_column("Unit",   style="unit",    min_width=14)
    t.add_column("Visual", min_width=24)

    for label, val, max_ref, unit in kpis:
        over = (label == "Pressure Drop" and val > 4.0)
        val_text = Text(f"{val:.3g}")
        if over:
            val_text.stylize("bold red")
        t.add_row(label, val_text, unit, bar_chart(val, max_ref, over_limit=over))

    console.print(t)


def print_physics_details(r) -> None:
    """Print detailed result tables (only available in full physics mode)."""
    t_feed = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t_feed.add_column("", style="muted",  min_width=28)
    t_feed.add_column("", style="value",  min_width=16, justify="right")
    t_feed.add_row("Fresh feed flow",           f"{r.fresh_feed_kmol_h:,.2f} kmol/h")
    t_feed.add_row("Reactor inlet flow",        f"{r.reactor_inlet_kmol_h:,.2f} kmol/h")
    t_feed.add_row("Recycle flow",              f"{r.recycle_kmol_h:,.2f} kmol/h")
    t_feed.add_row("Purge flow",                f"{r.purge_kmol_h:,.2f} kmol/h")
    t_feed.add_row("Single-pass CO conversion", f"{r.x_co_single_pass:.4f}")
    t_feed.add_row("ASF alpha",                 f"{r.alpha:.4f}")
    t_feed.add_row("Loop iterations",           f"{r.loop_iterations}")

    t_geo = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t_geo.add_column("", style="muted",  min_width=28)
    t_geo.add_column("", style="value",  min_width=16, justify="right")
    t_geo.add_row("Parallel reactors",     str(r.n_parallel))
    t_geo.add_row("Tubes per reactor",     f"{r.nt_per_reactor:,d}")
    t_geo.add_row("Tube length",           f"{r.tube_length_m:.2f} m")
    t_geo.add_row("Shell diameter",        f"{r.shell_diameter_m:.2f} m")
    t_geo.add_row("L/D ratio",             f"{r.l_over_d:.2f}")
    t_geo.add_row("Total catalyst volume", f"{r.total_catalyst_volume_m3:.2f} m³")
    t_geo.add_row("Total reactor volume",  f"{r.reactor_volume_m3:.2f} m³")

    t_energy = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t_energy.add_column("", style="muted",  min_width=28)
    t_energy.add_column("", style="value",  min_width=16, justify="right")
    t_energy.add_row("Compressor power",    f"{r.compressor_power_mw:.4f} MW")
    t_energy.add_row("Cooling duty",        f"{r.cooling_duty_mw:.4f} MW")
    t_energy.add_row("Heat of reaction",    f"{r.heat_of_reaction_kj_per_kmol_co:,.0f} kJ/kmol_CO")
    t_energy.add_row("Gas density",         f"{r.gas_density_kg_m3:.3f} kg/m³")
    t_energy.add_row("Superficial velocity",f"{r.superficial_velocity_m_s:.3f} m/s")
    t_energy.add_row("Pressure drop",       f"{r.delta_p_bar:.3f} bar")

    console.print(Columns([
        Panel(t_feed,   title="[section]Feed & Conversion[/section]",  border_style="blue"),
        Panel(t_geo,    title="[section]Reactor Geometry[/section]",   border_style="blue"),
    ], equal=True, expand=True))
    console.print(Panel(t_energy, title="[section]Energy & Hydraulics[/section]", border_style="blue"))


def print_compare(ml_result: dict, r_phys) -> None:
    """Print ML vs physics comparison table."""
    pairs = [
        ("Target Rate (kg/h)",   "target_rate_kgph",                  r_phys.target_rate_kgph),
        ("C8–C16 Fraction",      "target_fraction",                   r_phys.target_fraction),
        ("Specific Energy",      "specific_energy_kwh_per_kg_target", r_phys.specific_energy_kwh_per_kg_target),
        ("Compressor Power (MW)","compressor_power_mw",               r_phys.compressor_power_mw),
        ("Cooling Duty (MW)",    "cooling_duty_mw",                   r_phys.cooling_duty_mw),
        ("Pressure Drop (bar)",  "delta_p_bar",                       r_phys.delta_p_bar),
    ]

    t = Table(box=box.ROUNDED, border_style="blue",
              title="[primary]ML Surrogate vs Full Physics Comparison[/primary]",
              show_header=True, header_style="bold blue")
    t.add_column("KPI",          style="accent", min_width=24)
    t.add_column("ML Surrogate", style="ml",     min_width=14, justify="right")
    t.add_column("Full Physics", style="physics",min_width=14, justify="right")
    t.add_column("Error %",      min_width=10,   justify="right")

    for label, key, phys_val in pairs:
        ml_val = ml_result[key]
        err_pct = abs(ml_val - phys_val) / abs(phys_val) * 100 if phys_val != 0 else 0
        err_text = Text(f"{err_pct:.2f}%")
        if err_pct > 5:
            err_text.stylize("yellow")
        elif err_pct > 10:
            err_text.stylize("bold red")
        t.add_row(label, f"{ml_val:.4g}", f"{phys_val:.4g}", err_text)

    console.print(t)


def prompt_parameters(base_config: dict, use_ml: bool) -> tuple[dict, dict]:
    """Interactively prompt for all reactor parameters."""
    flat_params: dict[str, float] = {}

    p_table = Table(box=box.ROUNDED, border_style="dim blue",
                    title="[primary]Parameters[/primary]",
                    show_header=True, header_style="bold blue")
    p_table.add_column("Parameter", style="accent", min_width=24)
    p_table.add_column("Range",     style="muted",  min_width=18)
    p_table.add_column("Default",   style="value",  min_width=10)
    for label, _, default, lo, hi, unit in PARAMETERS:
        p_table.add_row(label, f"{lo}–{hi} {unit}", str(default))
    console.print(p_table)
    console.print("[muted]  Press Enter to accept the default. Type 'q' to quit.[/muted]\n")

    config = copy.deepcopy(base_config)
    composition_params = {}

    for label, dotpath, default, lo, hi, unit in PARAMETERS:
        unit_str = f" {unit}" if unit else ""
        console.print(f"  [accent]{label}[/accent] [muted]([{lo}–{hi}{unit_str}, default {default})[/muted]")

        while True:
            raw = Prompt.ask("    >", default=str(default), console=console)
            if raw.strip().lower() == "q":
                return None, None
            try:
                val = float(raw)
            except ValueError:
                console.print(f"    [bad]Invalid. Enter a number.[/bad]")
                continue
            if val < lo or val > hi:
                console.print(f"    [bad]Out of range [{lo}–{hi}]. Clamping.[/bad]")
                val = max(lo, min(hi, val))
            break

        if dotpath.startswith("_composition."):
            comp_key = dotpath.split(".", 1)[1]
            composition_params[comp_key] = val
        else:
            set_nested(config, dotpath, val)
        flat_params[dotpath] = val
        if dotpath == "reactor_geometry.tube_inner_diameter_m":
            set_nested(config, "reactor_geometry.tube_outer_diameter_m", val + 0.0032)

    # Apply composition to feed
    ratio = composition_params.get("h2_co_ratio", 2.2)
    co2_f = composition_params.get("co2_fraction", 0.02)
    n2_f = composition_params.get("n2_fraction", 0.02)
    remaining = max(1.0 - co2_f - n2_f, 0.01)
    h2_f = remaining * ratio / (ratio + 1.0)
    co_f = remaining / (ratio + 1.0)
    comp = config["feed"]["composition"]
    comp["H2"] = h2_f
    comp["CO"] = co_f
    comp["CO2"] = co2_f
    comp["N2"] = n2_f
    for sp in comp:
        if sp not in ("H2", "CO", "CO2", "N2"):
            comp[sp] = 0.0
    total_comp = sum(comp.values())
    if total_comp > 0:
        for sp in comp:
            comp[sp] /= total_comp

    return config, flat_params


def main() -> None:
    """Run the interactive simulation loop."""
    print_header()
    base_config = load_config()
    mode = choose_mode()
    console.print()

    # Preload ML model
    ml_model = ml_meta = None
    if mode == "ml":
        ml_model, ml_meta = load_surrogate()
        print_ml_model_info(ml_meta)

    while True:
        console.rule(
            f"[ml]🤖 New ML Prediction[/ml]" if mode == "ml"
            else "[physics]⚗️ New Physics Simulation[/physics]"
        )
        console.print()

        config, flat_params = prompt_parameters(base_config, use_ml=(mode == "ml"))
        if config is None:
            console.print("\n[muted]  Goodbye![/muted]\n")
            break

        console.print()

        try:
            if mode == "ml":
                # Build the short-key dict from the config
                comp = config["feed"]["composition"]
                ml_params = {
                    "temperature_C":         config["operating_conditions"]["temperature_C"],
                    "pressure_bar":          config["operating_conditions"]["pressure_bar"],
                    "ghsv_h":                config["design_basis"]["ghsv_h"],
                    "target_x_co":           config["target_x_co"],
                    "total_flow_kmol_h":     config["feed"]["total_flow_kmol_h"],
                    "tube_inner_diameter_m": config["reactor_geometry"]["tube_inner_diameter_m"],
                    "particle_diameter_m":   config["bed_properties"]["particle_diameter_m"],
                    "void_fraction":         config["bed_properties"]["void_fraction"],
                    "purge_fraction":        config["loop_configuration"]["purge_fraction"],
                    "h2_co_ratio":           comp.get("H2", 0.66) / max(comp.get("CO", 0.30), 1e-12),
                    "co2_fraction":          comp.get("CO2", 0.02),
                    "n2_fraction":           comp.get("N2", 0.02),
                }
                console.print("[muted]  Predicting with ML model...[/muted]")
                result = ml_predict(ml_params, model=ml_model)

                # Feasibility banner
                if result["feasible"]:
                    console.print(Panel("[good]✅  FEASIBLE — All design constraints satisfied[/good]",
                                        border_style="green", padding=(0, 2)))
                else:
                    console.print(Panel(f"[bad]❌  INFEASIBLE — {result['violation_reason']}[/bad]",
                                        border_style="red", padding=(0, 2)))
                console.print()
                print_kpi_bars_ml(result)
                console.print()

                # Optional: compare with physics
                compare = Prompt.ask(
                    "\n  [muted]Compare against full physics simulation?[/muted] [Y/n]",
                    default="n", console=console,
                )
                if compare.strip().lower() in ("y", "yes"):
                    from src.feed import build_total_feed
                    from src.reactor import FTReactor
                    console.print("[muted]  Running physics simulation...[/muted]")
                    feed = build_total_feed(config)
                    r_phys = FTReactor(config=config, feed_composition=feed).run()
                    console.print()
                    print_compare(result, r_phys)

            else:
                from src.feed import build_total_feed
                from src.reactor import FTReactor
                console.print("[muted]  Running reactor simulation...[/muted]")
                feed = build_total_feed(config)
                r_phys = FTReactor(config=config, feed_composition=feed).run()

                if r_phys.feasible:
                    console.print(Panel("[good]✅  FEASIBLE — All design constraints satisfied[/good]",
                                        border_style="green", padding=(0, 2)))
                else:
                    console.print(Panel(f"[bad]❌  INFEASIBLE — {r_phys.violation_reason}[/bad]",
                                        border_style="red", padding=(0, 2)))
                console.print()
                print_kpi_bars_physics(r_phys)
                console.print()
                print_physics_details(r_phys)

        except Exception as exc:
            console.print(f"[bad]  Simulation failed: {exc}[/bad]\n")

        again = Prompt.ask("\n  Run another simulation? [Y/n]", default="Y", console=console)
        if again.strip().lower() in ("n", "no", "q"):
            console.print("\n[muted]  Goodbye![/muted]\n")
            break
        console.print()


if __name__ == "__main__":
    main()
