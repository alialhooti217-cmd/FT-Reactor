"""Generate 2D and 3D visualisation plots from the FT reactor dataset.

Reads the simulation dataset (all cases and feasible-only) and produces:

2D plots:
  1. Feasibility scatter (Temperature vs Pressure)
  2. Pressure-drop histogram with limit line
  3. CO conversion vs C8-C16 selectivity
  4. Specific energy vs target product rate
  5. Pairwise correlation heatmap of KPIs

3D plots:
  6. T x P x GHSV coloured by specific energy (feasible only)
  7. T x P x target_x_co coloured by target fraction (feasible only)
  8. T x P coloured by pressure drop (3D surface-like scatter)

Usage:
    python visualize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for 3D projection
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "plots"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the full and feasible-only datasets."""
    all_path = DATA_DIR / "dataset.csv"
    feasible_path = DATA_DIR / "dataset_feasible.csv"

    if not all_path.exists():
        print(f"Error: {all_path} not found. Run batch/run_batch.py first.")
        sys.exit(1)

    df_all = pd.read_csv(all_path)
    df_feasible = pd.read_csv(feasible_path) if feasible_path.exists() else df_all[df_all.get("feasible", True) == True].copy()

    print(f"Loaded {len(df_all)} total cases, {len(df_feasible)} feasible cases")
    return df_all, df_feasible


# ─────────────────────────────────────────────────────────────────────
# 2D PLOTS
# ─────────────────────────────────────────────────────────────────────

def plot_feasibility_map(df: pd.DataFrame) -> str:
    """1. Temperature vs Pressure feasibility scatter."""
    fig, ax = plt.subplots(figsize=(9, 6))

    feasible = df[df["feasible"] == True]
    infeasible = df[df["feasible"] == False]

    ax.scatter(infeasible["input_temperature_C"], infeasible["input_pressure_bar"],
               c="#d62728", alpha=0.3, s=12, label=f"Infeasible ({len(infeasible)})")
    ax.scatter(feasible["input_temperature_C"], feasible["input_pressure_bar"],
               c="#2ca02c", alpha=0.4, s=12, label=f"Feasible ({len(feasible)})")

    ax.set_xlabel("Temperature (C)", fontsize=12)
    ax.set_ylabel("Pressure (bar)", fontsize=12)
    ax.set_title("FT Reactor Design Space: Feasibility Map", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = PLOTS_DIR / "01_feasibility_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_pressure_drop_hist(df: pd.DataFrame, limit: float = 4.0) -> str:
    """2. Pressure-drop distribution with constraint line."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(df["delta_p_bar"].dropna(), bins=50, color="#1f77b4", edgecolor="white", alpha=0.8)
    ax.axvline(limit, color="#d62728", linestyle="--", linewidth=2,
               label=f"Max limit = {limit} bar")

    ax.set_xlabel("Pressure Drop (bar)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Pressure Drop Distribution Across All Cases", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = PLOTS_DIR / "02_pressure_drop_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_conversion_vs_selectivity(df: pd.DataFrame) -> str:
    """3. CO conversion vs C8-C16 selectivity (feasible only)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(df["x_co_single_pass"], df["target_fraction"],
                    c=df["input_temperature_C"], cmap="plasma", s=14, alpha=0.6)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Temperature (C)", fontsize=11)

    ax.set_xlabel("Single-Pass CO Conversion", fontsize=12)
    ax.set_ylabel("C8-C16 Mass Fraction", fontsize=12)
    ax.set_title("CO Conversion vs Product Selectivity", fontsize=14)
    ax.grid(True, alpha=0.3)

    path = PLOTS_DIR / "03_conversion_vs_selectivity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_energy_vs_production(df: pd.DataFrame) -> str:
    """4. Specific energy vs target product rate."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(df["target_rate_kgph"], df["specific_energy_kwh_per_kg_target"],
                    c=df["input_pressure_bar"], cmap="viridis", s=14, alpha=0.6)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Pressure (bar)", fontsize=11)

    ax.set_xlabel("Target Product Rate (kg/h)", fontsize=12)
    ax.set_ylabel("Specific Energy (kWh/kg)", fontsize=12)
    ax.set_title("Energy Efficiency vs Production Rate", fontsize=14)
    ax.grid(True, alpha=0.3)

    path = PLOTS_DIR / "04_energy_vs_production.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_kpi_correlation(df: pd.DataFrame) -> str:
    """5. Pairwise correlation heatmap of key KPIs and inputs."""
    cols = [
        "input_temperature_C", "input_pressure_bar", "input_ghsv_h",
        "input_target_x_co", "input_total_flow_kmol_h",
        "target_rate_kgph", "target_fraction",
        "specific_energy_kwh_per_kg_target",
        "compressor_power_mw", "cooling_duty_mw", "delta_p_bar",
    ]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)

    labels = [c.replace("input_", "").replace("_", "\n") for c in available]
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(len(available)):
        for j in range(len(available)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("Correlation Matrix: Inputs and KPIs", fontsize=14)

    path = PLOTS_DIR / "05_kpi_correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


# ─────────────────────────────────────────────────────────────────────
# 3D PLOTS
# ─────────────────────────────────────────────────────────────────────

def plot_3d_energy_landscape(df: pd.DataFrame) -> str:
    """6. T x P x GHSV coloured by specific energy (3D scatter)."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample for readability if needed
    sample = df.sample(n=min(3000, len(df)), random_state=42)

    sc = ax.scatter(
        sample["input_temperature_C"],
        sample["input_pressure_bar"],
        sample["input_ghsv_h"],
        c=sample["specific_energy_kwh_per_kg_target"],
        cmap="coolwarm", s=10, alpha=0.6,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Specific Energy (kWh/kg)", fontsize=10)

    ax.set_xlabel("Temperature (C)", fontsize=10)
    ax.set_ylabel("Pressure (bar)", fontsize=10)
    ax.set_zlabel("GHSV (1/h)", fontsize=10)
    ax.set_title("3D Energy Landscape (T x P x GHSV)", fontsize=13)

    path = PLOTS_DIR / "06_3d_energy_landscape.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_3d_selectivity(df: pd.DataFrame) -> str:
    """7. T x P x target_x_co coloured by target fraction (3D scatter)."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    sample = df.sample(n=min(3000, len(df)), random_state=42)

    sc = ax.scatter(
        sample["input_temperature_C"],
        sample["input_pressure_bar"],
        sample["input_target_x_co"],
        c=sample["target_fraction"],
        cmap="YlGn", s=10, alpha=0.6,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("C8-C16 Fraction", fontsize=10)

    ax.set_xlabel("Temperature (C)", fontsize=10)
    ax.set_ylabel("Pressure (bar)", fontsize=10)
    ax.set_zlabel("Target CO Conversion", fontsize=10)
    ax.set_title("3D Selectivity Map (T x P x X_CO)", fontsize=13)

    path = PLOTS_DIR / "07_3d_selectivity_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_3d_pressure_drop(df: pd.DataFrame) -> str:
    """8. T x P coloured/sized by pressure drop (3D view with delta_p as Z)."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    sample = df.sample(n=min(3000, len(df)), random_state=42)

    sc = ax.scatter(
        sample["input_temperature_C"],
        sample["input_pressure_bar"],
        sample["delta_p_bar"],
        c=sample["delta_p_bar"],
        cmap="hot_r", s=10, alpha=0.6,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Pressure Drop (bar)", fontsize=10)

    ax.set_xlabel("Temperature (C)", fontsize=10)
    ax.set_ylabel("Pressure (bar)", fontsize=10)
    ax.set_zlabel("Pressure Drop (bar)", fontsize=10)
    ax.set_title("3D Pressure Drop Surface (T x P x dP)", fontsize=13)

    path = PLOTS_DIR / "08_3d_pressure_drop.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all visualisations and save to plots/ directory."""
    ensure_dir(PLOTS_DIR)

    df_all, df_feasible = load_datasets()

    print("\nGenerating 2D plots...")
    plots = []

    plots.append(plot_feasibility_map(df_all))
    print(f"  [1/8] Feasibility map")

    plots.append(plot_pressure_drop_hist(df_all))
    print(f"  [2/8] Pressure drop histogram")

    plots.append(plot_conversion_vs_selectivity(df_feasible))
    print(f"  [3/8] Conversion vs selectivity")

    plots.append(plot_energy_vs_production(df_feasible))
    print(f"  [4/8] Energy vs production")

    plots.append(plot_kpi_correlation(df_feasible))
    print(f"  [5/8] KPI correlation heatmap")

    print("\nGenerating 3D plots...")

    plots.append(plot_3d_energy_landscape(df_feasible))
    print(f"  [6/8] 3D energy landscape")

    plots.append(plot_3d_selectivity(df_feasible))
    print(f"  [7/8] 3D selectivity map")

    plots.append(plot_3d_pressure_drop(df_all))
    print(f"  [8/8] 3D pressure drop surface")

    print(f"\nAll {len(plots)} plots saved to: {PLOTS_DIR}/")
    print("Files:")
    for p in plots:
        print(f"  {Path(p).name}")
    print()


if __name__ == "__main__":
    main()
