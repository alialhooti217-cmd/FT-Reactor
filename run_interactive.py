"""Interactive FT reactor simulation with rich terminal visualisation.

Displays a colorful terminal UI with formatted tables, KPI bar charts,
an ASCII reactor diagram, and grouped result sections.

Usage:
    python run_interactive.py
"""

from __future__ import annotations

import copy
import sys
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

from src.feed import build_total_feed
from src.reactor import FTReactor

# ── Console with custom theme ───────────────────────────────────────────
theme = Theme({
    "primary":   "bold cyan",
    "accent":    "bold yellow",
    "good":      "bold green",
    "bad":       "bold red",
    "muted":     "dim white",
    "section":   "bold blue",
    "value":     "white",
    "unit":      "dim cyan",
    "bar.full":  "cyan",
    "bar.empty": "grey23",
})
console = Console(theme=theme, highlight=False)

# ── Parameter definitions ───────────────────────────────────────────────
PARAMETERS = [
    ("Temperature",           "operating_conditions.temperature_C",      220.0, 210.0, 235.0, "°C"),
    ("Pressure",              "operating_conditions.pressure_bar",        25.0,  20.0,  28.0, "bar"),
    ("GHSV",                  "design_basis.ghsv_h",                   1800.0, 1500.0, 2200.0, "h⁻¹"),
    ("Target CO Conversion",  "target_x_co",                             0.72,  0.60,  0.78, ""),
    ("Total Feed Flow",       "feed.total_flow_kmol_h",                1200.0,  900.0, 1800.0, "kmol/h"),
    ("Tube Inner Diameter",   "reactor_geometry.tube_inner_diameter_m",  0.042, 0.040, 0.046, "m"),
    ("Particle Diameter",     "bed_properties.particle_diameter_m",     0.0012, 0.0011, 0.0015, "m"),
    ("Void Fraction",         "bed_properties.void_fraction",            0.42,  0.41,  0.46, ""),
    ("Purge Fraction",        "loop_configuration.purge_fraction",       0.03,  0.02,  0.06, ""),
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


def bar_chart(value: float, max_val: float, width: int = 20) -> Text:
    """Return a Rich Text bar representing value/max_val."""
    filled = max(0, min(width, int(round(value / max_val * width))))
    bar = Text()
    bar.append("█" * filled, style="bar.full")
    bar.append("░" * (width - filled), style="bar.empty")
    return bar


def print_header() -> None:
    """Print the application header banner."""
    console.print()
    console.rule("[primary]⚗  Fischer–Tropsch Reactor Simulator[/primary]")
    console.print("[muted]  Interactive single-case simulation with real-time results[/muted]")
    console.print()


def print_reactor_diagram(n_parallel: int, nt_per_reactor: int,
                           tube_length_m: float, shell_d_m: float,
                           tube_id_mm: float) -> None:
    """Print an ASCII reactor diagram panel."""
    diagram = (
        f"  [dim]╔══════════════════════════════════╗[/dim]\n"
        f"  [dim]║[/dim] [primary]MULTITUBULAR FT REACTOR[/primary]         [dim]║[/dim]\n"
        f"  [dim]║                                  ║[/dim]\n"
        f"  [dim]║[/dim]  [accent]Feed (syngas) →→→→→→→→→→→→[/accent]  [dim]║[/dim]\n"
        f"  [dim]║[/dim]  [muted]╔══════════════════════════╗[/muted]  [dim]║[/dim]\n"
        f"  [dim]║[/dim]  [muted]║  [/muted][value]{nt_per_reactor:>6,d} tubes per shell[/value][muted]  ║[/muted]  [dim]║[/dim]\n"
        f"  [dim]║[/dim]  [muted]║[/muted]  [muted]Ø {tube_id_mm:.0f} mm ID  ×  {tube_length_m:.1f} m long [/muted] [muted]║[/muted]  [dim]║[/dim]\n"
        f"  [dim]║[/dim]  [muted]║[/muted]  [muted]Shell Ø {shell_d_m:.2f} m           [/muted] [muted]║[/muted]  [dim]║[/dim]\n"
        f"  [dim]║[/dim]  [muted]╚══════════════════════════╝[/muted]  [dim]║[/dim]\n"
        f"  [dim]║[/dim]  [accent]↓  Products (hydrocarbons)    [/accent]  [dim]║[/dim]\n"
        f"  [dim]║  × {n_parallel} reactors in parallel          ║[/dim]\n"
        f"  [dim]╚══════════════════════════════════╝[/dim]\n"
    )
    console.print(Panel(diagram, title="[primary]Reactor Configuration[/primary]",
                        border_style="blue", padding=(0, 1)))


def print_kpi_bars(r) -> None:
    """Print a visual KPI dashboard with bar charts."""
    kpis = [
        ("C8–C16 Rate",       r.target_rate_kgph,                  1500.0, "kg/h"),
        ("C8–C16 Fraction",   r.target_fraction * 100,              25.0,  "% (mass)"),
        ("Specific Energy",   r.specific_energy_kwh_per_kg_target,  60.0,  "kWh/kg"),
        ("Compressor Power",  r.compressor_power_mw,                 3.0,  "MW"),
        ("Cooling Duty",      r.cooling_duty_mw,                    40.0,  "MW"),
        ("Pressure Drop",     r.delta_p_bar,                         6.0,  "bar"),
    ]

    table = Table(
        box=box.ROUNDED,
        border_style="blue",
        show_header=True,
        header_style="bold blue",
        padding=(0, 1),
        title="[primary]Key Performance Indicators[/primary]",
    )
    table.add_column("KPI",          style="accent",  min_width=20)
    table.add_column("Value",        style="value",   min_width=12, justify="right")
    table.add_column("Unit",         style="unit",    min_width=10)
    table.add_column("Visual",       min_width=22, no_wrap=True)

    limits = {5: 4.0}  # index 5 = pressure drop has a hard limit

    for idx, (label, val, max_ref, unit) in enumerate(kpis):
        # Colour the bar red if over limit
        bar = bar_chart(val, max_ref, width=20)
        at_limit = idx in limits and val > limits[idx]
        value_text = Text(f"{val:.3g}")
        if at_limit:
            value_text.stylize("bold red")
            bar = Text()
            bar.append("█" * 20, style="bold red")
        table.add_row(label, value_text, unit, bar)

    console.print(table)


def print_details(r, params: dict) -> None:
    """Print detailed result tables for feed, geometry, energy, and hydraulics."""

    # Feed & conversion
    t_feed = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t_feed.add_column("", style="muted",   min_width=28)
    t_feed.add_column("", style="value",   min_width=16, justify="right")

    t_feed.add_row("Fresh feed flow",          f"{r.fresh_feed_kmol_h:,.2f} kmol/h")
    t_feed.add_row("Reactor inlet flow",       f"{r.reactor_inlet_kmol_h:,.2f} kmol/h")
    t_feed.add_row("Recycle flow",             f"{r.recycle_kmol_h:,.2f} kmol/h")
    t_feed.add_row("Purge flow",               f"{r.purge_kmol_h:,.2f} kmol/h")
    t_feed.add_row("Single-pass CO conversion",f"{r.x_co_single_pass:.4f}")
    t_feed.add_row("ASF alpha",                f"{r.alpha:.4f}")
    t_feed.add_row("Loop iterations",          f"{r.loop_iterations}")

    # Geometry
    t_geo = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t_geo.add_column("", style="muted",   min_width=28)
    t_geo.add_column("", style="value",   min_width=16, justify="right")

    t_geo.add_row("Parallel reactors (N)",     str(r.n_parallel))
    t_geo.add_row("Tubes per reactor",         f"{r.nt_per_reactor:,d}")
    t_geo.add_row("Tube length",               f"{r.tube_length_m:.2f} m")
    t_geo.add_row("Shell diameter",            f"{r.shell_diameter_m:.2f} m")
    t_geo.add_row("L/D ratio",                 f"{r.l_over_d:.2f}")
    t_geo.add_row("Total catalyst volume",     f"{r.total_catalyst_volume_m3:.2f} m³")
    t_geo.add_row("Total reactor volume",      f"{r.reactor_volume_m3:.2f} m³")

    # Energy & hydraulics
    t_energy = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t_energy.add_column("", style="muted",   min_width=28)
    t_energy.add_column("", style="value",   min_width=16, justify="right")

    t_energy.add_row("Compressor power",       f"{r.compressor_power_mw:.4f} MW")
    t_energy.add_row("Cooling duty",           f"{r.cooling_duty_mw:.4f} MW")
    t_energy.add_row("Heat of reaction",       f"{r.heat_of_reaction_kj_per_kmol_co:,.0f} kJ/kmol_CO")
    t_energy.add_row("Gas density",            f"{r.gas_density_kg_m3:.3f} kg/m³")
    t_energy.add_row("Superficial velocity",   f"{r.superficial_velocity_m_s:.3f} m/s")
    t_energy.add_row("Pressure drop",          f"{r.delta_p_bar:.3f} bar")
    t_energy.add_row("Cp mixture",             f"{r.cp_mix_kj_kmolk:.2f} kJ/kmol·K")

    feed_panel   = Panel(t_feed,   title="[section]Feed & Conversion[/section]",   border_style="blue")
    geo_panel    = Panel(t_geo,    title="[section]Reactor Geometry[/section]",    border_style="blue")
    energy_panel = Panel(t_energy, title="[section]Energy & Hydraulics[/section]", border_style="blue")

    console.print(Columns([feed_panel, geo_panel], equal=True, expand=True))
    console.print(energy_panel)


def print_results(r, params: dict) -> None:
    """Print the full result display for a completed simulation."""
    # Feasibility banner
    if r.feasible:
        console.print(Panel(
            "[good]✅  FEASIBLE — All design constraints satisfied[/good]",
            border_style="green", padding=(0, 2),
        ))
    else:
        console.print(Panel(
            f"[bad]❌  INFEASIBLE — {r.violation_reason}[/bad]",
            border_style="red", padding=(0, 2),
        ))

    console.print()

    # Reactor diagram + KPI bars side by side
    print_reactor_diagram(
        n_parallel=r.n_parallel,
        nt_per_reactor=r.nt_per_reactor,
        tube_length_m=r.tube_length_m,
        shell_d_m=r.shell_diameter_m,
        tube_id_mm=params["reactor_geometry.tube_inner_diameter_m"] * 1000,
    )
    console.print()
    print_kpi_bars(r)
    console.print()
    print_details(r, params)
    console.print()


def prompt_parameters(base_config: dict) -> tuple[dict, dict]:
    """Interactively prompt for all reactor parameters.

    Returns:
        (config, flat_params) where config is updated for FTReactor
        and flat_params is a dict keyed by dotpath for the diagram.
    """
    flat_params: dict[str, float] = {}

    # Show parameter table before prompting
    p_table = Table(box=box.ROUNDED, border_style="dim blue",
                    title="[primary]Parameters to configure[/primary]",
                    show_header=True, header_style="bold blue")
    p_table.add_column("Parameter",  style="accent",  min_width=24)
    p_table.add_column("Range",      style="muted",   min_width=16)
    p_table.add_column("Default",    style="value",   min_width=10)

    for label, _, default, lo, hi, unit in PARAMETERS:
        p_table.add_row(label, f"{lo}–{hi} {unit}", str(default))
    console.print(p_table)
    console.print("[muted]  Press Enter to use the default value. Type 'q' to quit.[/muted]\n")

    config = copy.deepcopy(base_config)

    for label, dotpath, default, lo, hi, unit in PARAMETERS:
        unit_str = f" {unit}" if unit else ""
        prompt_str = f"  [accent]{label}[/accent] [muted]([{lo}–{hi}{unit_str}, default {default})[/muted]"
        console.print(prompt_str)

        while True:
            raw = Prompt.ask("    >", default=str(default), console=console)
            if raw.strip().lower() == "q":
                return None, None
            try:
                val = float(raw)
            except ValueError:
                console.print(f"    [bad]Invalid. Enter a number between {lo} and {hi}.[/bad]")
                continue
            if val < lo or val > hi:
                console.print(f"    [bad]Out of range [{lo}–{hi}]. Clamping.[/bad]")
                val = max(lo, min(hi, val))
            break

        set_nested(config, dotpath, val)
        flat_params[dotpath] = val

        if dotpath == "reactor_geometry.tube_inner_diameter_m":
            set_nested(config, "reactor_geometry.tube_outer_diameter_m", val + 0.0032)

    return config, flat_params


def main() -> None:
    """Run the interactive simulation loop."""
    print_header()
    base_config = load_config()

    while True:
        console.rule("[primary]New Simulation[/primary]")
        console.print()

        config, flat_params = prompt_parameters(base_config)

        if config is None:
            console.print("\n[muted]  Goodbye![/muted]\n")
            break

        console.print()
        console.print("[muted]  Running reactor simulation...[/muted]")

        try:
            feed = build_total_feed(config)
            reactor = FTReactor(config=config, feed_composition=feed)
            results = reactor.run()
            print_results(results, flat_params)
        except Exception as exc:
            console.print(f"[bad]  Simulation failed: {exc}[/bad]\n")

        again = Prompt.ask("\n  Run another simulation? [Y/n]",
                           default="Y", console=console)
        if again.strip().lower() in ("n", "no", "q"):
            console.print("\n[muted]  Goodbye![/muted]\n")
            break
        console.print()


if __name__ == "__main__":
    main()
