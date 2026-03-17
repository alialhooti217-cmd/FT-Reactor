"""Streamlit web dashboard for the FT Reactor Simulation.

Provides an interactive browser-based UI with sliders for all reactor
operating parameters, live KPI results, bar charts, and reactor geometry
details.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feed import build_total_feed
from src.reactor import FTReactor

# ── Page setup ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FT Reactor Simulator",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load base config once ───────────────────────────────────────────────
@st.cache_resource
def load_base_config() -> dict:
    import yaml
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_simulation(params: dict) -> object | None:
    """Run a reactor simulation with the given parameter overrides."""
    config = copy.deepcopy(load_base_config())

    config["operating_conditions"]["temperature_C"]      = params["temperature_C"]
    config["operating_conditions"]["pressure_bar"]       = params["pressure_bar"]
    config["design_basis"]["ghsv_h"]                    = params["ghsv_h"]
    config["target_x_co"]                               = params["target_x_co"]
    config["feed"]["total_flow_kmol_h"]                 = params["total_flow_kmol_h"]
    config["reactor_geometry"]["tube_inner_diameter_m"] = params["tube_inner_diameter_m"]
    config["reactor_geometry"]["tube_outer_diameter_m"] = params["tube_inner_diameter_m"] + 0.0032
    config["bed_properties"]["particle_diameter_m"]     = params["particle_diameter_m"]
    config["bed_properties"]["void_fraction"]            = params["void_fraction"]
    config["loop_configuration"]["purge_fraction"]      = params["purge_fraction"]

    feed = build_total_feed(config)
    reactor = FTReactor(config=config, feed_composition=feed)
    return reactor.run()


# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #1e2433;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #2e3757;
    }
    .metric-label { color: #8b95b0; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #e8eaf6; font-size: 26px; font-weight: 700; }
    .metric-unit  { color: #8b95b0; font-size: 12px; }
    .feasible-badge {
        background: #1b4332;
        color: #52b788;
        border: 1px solid #52b788;
        border-radius: 20px;
        padding: 6px 20px;
        font-size: 16px;
        font-weight: 700;
    }
    .infeasible-badge {
        background: #3b0a0a;
        color: #e63946;
        border: 1px solid #e63946;
        border-radius: 20px;
        padding: 6px 20px;
        font-size: 16px;
        font-weight: 700;
    }
    [data-testid="stSidebar"] { background: #0f1117; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: parameter inputs ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ FT Reactor")
    st.markdown("### Operating Parameters")
    st.markdown("---")

    temperature_C = st.slider("🌡️ Temperature (°C)",
        min_value=210.0, max_value=235.0, value=220.0, step=0.5)

    pressure_bar = st.slider("🔵 Pressure (bar)",
        min_value=20.0, max_value=28.0, value=25.0, step=0.5)

    ghsv_h = st.slider("💨 GHSV (h⁻¹)",
        min_value=1500.0, max_value=2200.0, value=1800.0, step=50.0)

    target_x_co = st.slider("🎯 Target CO Conversion",
        min_value=0.60, max_value=0.78, value=0.72, step=0.01,
        format="%.2f")

    total_flow_kmol_h = st.slider("🌊 Total Feed Flow (kmol/h)",
        min_value=900.0, max_value=1800.0, value=1200.0, step=50.0)

    st.markdown("---")
    st.markdown("### Reactor Geometry")

    tube_inner_diameter_m = st.slider("🔩 Tube Inner Diameter (m)",
        min_value=0.040, max_value=0.046, value=0.042, step=0.0005,
        format="%.4f")

    particle_diameter_m = st.slider("⚫ Particle Diameter (m)",
        min_value=0.0011, max_value=0.0015, value=0.0012, step=0.00005,
        format="%.5f")

    void_fraction = st.slider("🕳️ Void Fraction",
        min_value=0.41, max_value=0.46, value=0.42, step=0.005,
        format="%.3f")

    purge_fraction = st.slider("💧 Purge Fraction",
        min_value=0.020, max_value=0.060, value=0.030, step=0.002,
        format="%.3f")

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ── Main panel ──────────────────────────────────────────────────────────
st.markdown("# ⚗️ Fischer–Tropsch Reactor Dashboard")
st.markdown("Adjust parameters in the sidebar and click **Run Simulation**.")

params = {
    "temperature_C": temperature_C,
    "pressure_bar": pressure_bar,
    "ghsv_h": ghsv_h,
    "target_x_co": target_x_co,
    "total_flow_kmol_h": total_flow_kmol_h,
    "tube_inner_diameter_m": tube_inner_diameter_m,
    "particle_diameter_m": particle_diameter_m,
    "void_fraction": void_fraction,
    "purge_fraction": purge_fraction,
}

# Auto-run or button-triggered
if run_btn or "results" not in st.session_state:
    with st.spinner("Running reactor simulation..."):
        try:
            r = run_simulation(params)
            st.session_state["results"] = r
            st.session_state["params"] = params.copy()
            st.session_state["error"] = None
        except Exception as exc:
            st.session_state["error"] = str(exc)
            st.session_state["results"] = None

# ── Display results ─────────────────────────────────────────────────────
if st.session_state.get("error"):
    st.error(f"Simulation failed: {st.session_state['error']}")

elif st.session_state.get("results"):
    r = st.session_state["results"]

    # Feasibility badge
    if r.feasible:
        st.markdown('<span class="feasible-badge">✅ FEASIBLE</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<span class="infeasible-badge">❌ INFEASIBLE — {r.violation_reason}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("")

    # ── KPI metric cards ──────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    def metric_card(col, label: str, value: str, unit: str) -> None:
        col.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-unit">{unit}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    metric_card(c1, "C8–C16 Rate",      f"{r.target_rate_kgph:,.0f}",             "kg/h")
    metric_card(c2, "C8–C16 Fraction",  f"{r.target_fraction * 100:.2f}%",        "mass fraction")
    metric_card(c3, "Specific Energy",  f"{r.specific_energy_kwh_per_kg_target:.2f}", "kWh/kg")
    metric_card(c4, "Pressure Drop",    f"{r.delta_p_bar:.3f}",                   "bar  (limit 4.0)")
    metric_card(c5, "Cooling Duty",     f"{r.cooling_duty_mw:.3f}",               "MW")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 KPI Charts", "🏗️ Reactor Geometry", "⚡ Energy & Loop", "📈 Dataset Plots"])

    # --- Tab 1: KPI bar chart ---
    with tab1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        kpi_labels = [
            "Target Rate\n(kg/h)",
            "Selectivity\n(%)",
            "Spec. Energy\n(kWh/kg)",
            "Compressor\nPower (MW)",
            "Cooling Duty\n(MW)",
            "Pressure Drop\n(bar)",
        ]
        kpi_values = [
            r.target_rate_kgph,
            r.target_fraction * 100,
            r.specific_energy_kwh_per_kg_target,
            r.compressor_power_mw,
            r.cooling_duty_mw,
            r.delta_p_bar,
        ]

        # Limits / reference lines
        limits = [None, 8.0, None, None, None, 4.0]
        colors = ["#4dabf7", "#69db7c", "#ffa94d", "#cc5de8", "#f783ac", "#ff6b6b"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), facecolor="#0f1117")
        axes = axes.flatten()

        for i, (ax, label, val, lim, color) in enumerate(
            zip(axes, kpi_labels, kpi_values, limits, colors)
        ):
            ax.set_facecolor("#1e2433")
            bar = ax.bar([0], [val], color=color, width=0.5, alpha=0.85)
            if lim:
                ax.axhline(lim, color="white", linestyle="--", linewidth=1.5, alpha=0.7,
                           label=f"Limit: {lim}")
                ax.legend(fontsize=8, facecolor="#1e2433", labelcolor="white", framealpha=0.5)
            ax.set_xticks([])
            ax.set_title(label, color="white", fontsize=10, pad=8)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2e3757")
            ax.yaxis.label.set_color("white")
            # Value annotation
            ax.text(0, val * 0.5, f"{val:.3g}", ha="center", va="center",
                    color="white", fontsize=13, fontweight="bold")

        fig.suptitle("Key Performance Indicators", color="white", fontsize=14, y=1.01)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Tab 2: Geometry ---
    with tab2:
        col_geo, col_diag = st.columns([1, 1])

        with col_geo:
            st.markdown("#### Tube Bundle Configuration")
            geo_data = {
                "Parameter": [
                    "Parallel Reactors (N)",
                    "Tubes per Reactor",
                    "Tube Length",
                    "Shell Diameter",
                    "L/D Ratio",
                    "Total Catalyst Volume",
                    "Total Reactor Volume",
                    "Tube Inner Diameter",
                    "Particle Diameter",
                    "Void Fraction",
                ],
                "Value": [
                    f"{r.n_parallel}",
                    f"{r.nt_per_reactor:,d}",
                    f"{r.tube_length_m:.2f} m",
                    f"{r.shell_diameter_m:.2f} m",
                    f"{r.l_over_d:.2f}",
                    f"{r.total_catalyst_volume_m3:.2f} m³",
                    f"{r.reactor_volume_m3:.2f} m³",
                    f"{tube_inner_diameter_m * 1000:.1f} mm",
                    f"{particle_diameter_m * 1000:.2f} mm",
                    f"{void_fraction:.3f}",
                ],
            }
            st.dataframe(
                pd.DataFrame(geo_data),
                use_container_width=True,
                hide_index=True,
            )

        with col_diag:
            st.markdown("#### Reactor Diagram")
            n = r.n_parallel
            nt = r.nt_per_reactor
            L = r.tube_length_m
            Ds = r.shell_diameter_m
            # ASCII reactor diagram
            diagram = f"""
```
  ╔═══════════════════════╗
  ║   MULTITUBULAR FT     ║
  ║     REACTOR           ║
  ║  ┌─────────────────┐  ║
  ║  │  Feed → → → →   │  ║  ← Shell Ø {Ds:.2f} m
  ║  │ ┊┊┊┊┊┊┊┊┊┊┊┊┊┊ │  ║
  ║  │ ┊  {nt:>5,d} tubes  ┊ │  ║  ← Tube Ø {tube_inner_diameter_m*1000:.0f} mm ID
  ║  │ ┊  each {L:.1f} m  ┊ │  ║
  ║  │ ┊┊┊┊┊┊┊┊┊┊┊┊┊┊ │  ║
  ║  │  ↓ Products     │  ║
  ║  └─────────────────┘  ║
  ║  × {n} in parallel    ║
  ╚═══════════════════════╝
```
"""
            st.markdown(diagram)

    # --- Tab 3: Energy & Loop ---
    with tab3:
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown("#### Recycle Loop")
            loop_data = {
                "Stream": ["Fresh Feed", "Reactor Inlet", "Recycle", "Purge"],
                "Flow (kmol/h)": [
                    f"{r.fresh_feed_kmol_h:,.2f}",
                    f"{r.reactor_inlet_kmol_h:,.2f}",
                    f"{r.recycle_kmol_h:,.2f}",
                    f"{r.purge_kmol_h:,.2f}",
                ],
            }
            st.dataframe(pd.DataFrame(loop_data), use_container_width=True, hide_index=True)

            st.markdown("#### Kinetics")
            kin_data = {
                "Parameter": ["Single-pass CO conversion", "ASF alpha", "Loop iterations"],
                "Value": [
                    f"{r.x_co_single_pass:.4f}",
                    f"{r.alpha:.4f}",
                    f"{r.loop_iterations}",
                ],
            }
            st.dataframe(pd.DataFrame(kin_data), use_container_width=True, hide_index=True)

        with col_e2:
            st.markdown("#### Energy Balance")
            energy_data = {
                "Component": [
                    "Compressor Power",
                    "Cooling Duty",
                    "Specific Energy",
                    "Heat of Reaction",
                    "Gas Density",
                    "Superficial Velocity",
                ],
                "Value": [
                    f"{r.compressor_power_mw:.4f} MW",
                    f"{r.cooling_duty_mw:.4f} MW",
                    f"{r.specific_energy_kwh_per_kg_target:.4f} kWh/kg",
                    f"{r.heat_of_reaction_kj_per_kmol_co:,.0f} kJ/kmol_CO",
                    f"{r.gas_density_kg_m3:.3f} kg/m³",
                    f"{r.superficial_velocity_m_s:.3f} m/s",
                ],
            }
            st.dataframe(pd.DataFrame(energy_data), use_container_width=True, hide_index=True)

    # --- Tab 4: Dataset plots ---
    with tab4:
        plots_dir = PROJECT_ROOT / "plots"
        plot_files = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

        if not plot_files:
            st.info("No plots found. Run `python visualize.py` first to generate them.")
        else:
            st.markdown(f"Showing {len(plot_files)} pre-generated plots from `plots/`.")
            # Show in pairs
            for i in range(0, len(plot_files), 2):
                row_cols = st.columns(2)
                for j, col in enumerate(row_cols):
                    if i + j < len(plot_files):
                        pf = plot_files[i + j]
                        col.image(str(pf), caption=pf.stem.replace("_", " ").title(),
                                  use_container_width=True)
