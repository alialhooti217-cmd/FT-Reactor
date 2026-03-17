"""Streamlit web dashboard for the FT Reactor Simulation.

Supports two simulation modes:
  - ML Surrogate (instant): uses the trained ExtraTreesRegressor model
    that learned from 21,000 reactor simulations.
  - Full Physics: runs the complete FTReactor equation-based model.

The comparison tab shows ML vs physics side-by-side with error %.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress sklearn version mismatch warnings on model load
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from ml.predictor import predict as ml_predict, load_surrogate, r2_summary, surrogate_available

# ── Page setup ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FT Reactor Simulator",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached loaders ──────────────────────────────────────────────────────
@st.cache_resource
def get_surrogate():
    """Load surrogate model once and keep in memory."""
    return load_surrogate()


@st.cache_resource
def load_base_config() -> dict:
    import yaml
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_physics(params: dict) -> object:
    """Run the full equation-based FTReactor simulation."""
    from src.feed import build_total_feed
    from src.reactor import FTReactor

    config = copy.deepcopy(load_base_config())
    config["operating_conditions"]["temperature_C"]      = params["temperature_C"]
    config["operating_conditions"]["pressure_bar"]       = params["pressure_bar"]
    config["design_basis"]["ghsv_h"]                    = params["ghsv_h"]
    config["target_x_co"]                               = params["target_x_co"]
    config["feed"]["total_flow_kmol_h"]                 = params["total_flow_kmol_h"]
    config["reactor_geometry"]["tube_inner_diameter_m"] = params["tube_inner_diameter_m"]
    config["reactor_geometry"]["tube_outer_diameter_m"] = params["tube_inner_diameter_m"] + 0.0032
    config["bed_properties"]["particle_diameter_m"]     = params["particle_diameter_m"]
    config["bed_properties"]["void_fraction"]           = params["void_fraction"]
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
        background: #1b4332; color: #52b788;
        border: 1px solid #52b788; border-radius: 20px;
        padding: 6px 20px; font-size: 16px; font-weight: 700;
    }
    .infeasible-badge {
        background: #3b0a0a; color: #e63946;
        border: 1px solid #e63946; border-radius: 20px;
        padding: 6px 20px; font-size: 16px; font-weight: 700;
    }
    .ml-badge {
        background: #1a2a4a; color: #74b9ff;
        border: 1px solid #74b9ff; border-radius: 20px;
        padding: 4px 14px; font-size: 13px; font-weight: 600;
    }
    .physics-badge {
        background: #1a3a2a; color: #55efc4;
        border: 1px solid #55efc4; border-radius: 20px;
        padding: 4px 14px; font-size: 13px; font-weight: 600;
    }
    [data-testid="stSidebar"] { background: #0f1117; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ FT Reactor")

    # Mode selector
    ml_ok = surrogate_available()
    mode_options = ["🤖 ML Surrogate (instant)", "⚗️ Full Physics"]
    if not ml_ok:
        mode_options = ["⚗️ Full Physics"]
        st.warning("Surrogate model not found. Using full physics only.")

    sim_mode = st.radio(
        "**Simulation Mode**",
        mode_options,
        index=0,
        help=(
            "ML Surrogate: instant predictions from a model trained on 21,000 cases.\n"
            "Full Physics: runs all equations (~1 s)."
        ),
    )
    use_ml = sim_mode.startswith("🤖")

    st.markdown("---")
    st.markdown("### Operating Parameters")

    temperature_C = st.slider("🌡️ Temperature (°C)", 210.0, 235.0, 220.0, 0.5)
    pressure_bar  = st.slider("🔵 Pressure (bar)",    20.0,  28.0,  25.0,  0.5)
    ghsv_h        = st.slider("💨 GHSV (h⁻¹)",      1500.0, 2200.0, 1800.0, 50.0)
    target_x_co   = st.slider("🎯 Target CO Conversion", 0.60, 0.78, 0.72, 0.01, format="%.2f")
    total_flow    = st.slider("🌊 Feed Flow (kmol/h)", 900.0, 1800.0, 1200.0, 50.0)

    st.markdown("---")
    st.markdown("### Geometry")
    tube_id   = st.slider("🔩 Tube Inner Diameter (m)", 0.040, 0.046, 0.042, 0.0005, format="%.4f")
    dp_m      = st.slider("⚫ Particle Diameter (m)",  0.0011, 0.0015, 0.0012, 0.00005, format="%.5f")
    void_f    = st.slider("🕳️ Void Fraction",          0.41, 0.46, 0.42, 0.005, format="%.3f")
    purge_f   = st.slider("💧 Purge Fraction",         0.020, 0.060, 0.030, 0.002, format="%.3f")

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

params = {
    "temperature_C": temperature_C,
    "pressure_bar": pressure_bar,
    "ghsv_h": ghsv_h,
    "target_x_co": target_x_co,
    "total_flow_kmol_h": total_flow,
    "tube_inner_diameter_m": tube_id,
    "particle_diameter_m": dp_m,
    "void_fraction": void_f,
    "purge_fraction": purge_f,
}


# ── Main ─────────────────────────────────────────────────────────────────
st.markdown("# ⚗️ Fischer–Tropsch Reactor Dashboard")

mode_label = (
    '<span class="ml-badge">🤖 ML Surrogate — trained on 21,000 simulations</span>'
    if use_ml else
    '<span class="physics-badge">⚗️ Full Physics Equations</span>'
)
st.markdown(mode_label, unsafe_allow_html=True)
st.markdown("Adjust parameters in the sidebar and click **Run Simulation**.")

# ── Run on button or first load ─────────────────────────────────────────
if run_btn or "results" not in st.session_state:
    with st.spinner("Predicting with ML model..." if use_ml else "Running reactor simulation..."):
        try:
            if use_ml:
                model, meta = get_surrogate()
                result_dict = ml_predict(params, model=model)
                st.session_state["results"]      = result_dict
                st.session_state["results_type"] = "ml"
                st.session_state["meta"]         = meta
            else:
                r = run_physics(params)
                st.session_state["results"]      = r
                st.session_state["results_type"] = "physics"
            st.session_state["params"] = params.copy()
            st.session_state["error"]  = None
        except Exception as exc:
            st.session_state["error"]  = str(exc)
            st.session_state["results"] = None

# ── Error display ───────────────────────────────────────────────────────
if st.session_state.get("error"):
    st.error(f"Simulation failed: {st.session_state['error']}")

# ── Results display ─────────────────────────────────────────────────────
elif st.session_state.get("results") is not None:
    res_type = st.session_state["results_type"]

    # Normalise into a flat dict for display
    if res_type == "ml":
        r = st.session_state["results"]
        kpis = {k: r[k] for k in [
            "target_rate_kgph", "target_fraction",
            "specific_energy_kwh_per_kg_target",
            "compressor_power_mw", "cooling_duty_mw", "delta_p_bar",
        ]}
        feasible         = r["feasible"]
        violation_reason = r["violation_reason"]
    else:
        r = st.session_state["results"]
        kpis = {
            "target_rate_kgph":                    r.target_rate_kgph,
            "target_fraction":                     r.target_fraction,
            "specific_energy_kwh_per_kg_target":   r.specific_energy_kwh_per_kg_target,
            "compressor_power_mw":                 r.compressor_power_mw,
            "cooling_duty_mw":                     r.cooling_duty_mw,
            "delta_p_bar":                         r.delta_p_bar,
        }
        feasible         = r.feasible
        violation_reason = r.violation_reason

    # Feasibility banner
    if feasible:
        st.markdown('<span class="feasible-badge">✅ FEASIBLE</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<span class="infeasible-badge">❌ INFEASIBLE — {violation_reason}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("")

    # KPI cards
    def metric_card(col, label, value, unit):
        col.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-unit">{unit}</div>'
            f'</div>', unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "C8–C16 Rate",     f"{kpis['target_rate_kgph']:,.0f}",                       "kg/h")
    metric_card(c2, "C8–C16 Fraction", f"{kpis['target_fraction'] * 100:.2f}%",                  "mass fraction")
    metric_card(c3, "Specific Energy", f"{kpis['specific_energy_kwh_per_kg_target']:.2f}",        "kWh/kg")
    metric_card(c4, "Pressure Drop",   f"{kpis['delta_p_bar']:.3f}",                             "bar  (limit 4.0)")
    metric_card(c5, "Cooling Duty",    f"{kpis['cooling_duty_mw']:.3f}",                         "MW")

    st.markdown("---")

    # ── Tabs ────────────────────────────────────────────────────────────
    if res_type == "ml":
        tab_labels = ["📊 KPI Charts", "🧠 ML Model Info", "🔬 Compare with Physics", "📈 Dataset Plots"]
    else:
        tab_labels = ["📊 KPI Charts", "🏗️ Reactor Geometry", "⚡ Energy & Loop", "📈 Dataset Plots"]

    tabs = st.tabs(tab_labels)

    # ── Tab 1: KPI bar charts ────────────────────────────────────────────
    with tabs[0]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kpi_labels = [
            "Target Rate\n(kg/h)", "Selectivity\n(%)",
            "Spec. Energy\n(kWh/kg)", "Compressor\nPower (MW)",
            "Cooling Duty\n(MW)", "Pressure Drop\n(bar)",
        ]
        kpi_values = [
            kpis["target_rate_kgph"],
            kpis["target_fraction"] * 100,
            kpis["specific_energy_kwh_per_kg_target"],
            kpis["compressor_power_mw"],
            kpis["cooling_duty_mw"],
            kpis["delta_p_bar"],
        ]
        ref_maxes = [1500.0, 25.0, 60.0, 3.0, 40.0, 6.0]
        limits    = [None,   8.0,  None, None, None, 4.0]
        colors    = ["#4dabf7", "#69db7c", "#ffa94d", "#cc5de8", "#f783ac", "#ff6b6b"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), facecolor="#0f1117")
        for ax, label, val, ref, lim, color in zip(
            axes.flatten(), kpi_labels, kpi_values, ref_maxes, limits, colors
        ):
            ax.set_facecolor("#1e2433")
            ax.bar([0], [val], color=color, width=0.5, alpha=0.85)
            if lim:
                ax.axhline(lim, color="white", linestyle="--", linewidth=1.5, alpha=0.7,
                           label=f"Limit: {lim}")
                ax.legend(fontsize=8, facecolor="#1e2433", labelcolor="white", framealpha=0.5)
            ax.set_xticks([])
            ax.set_title(label, color="white", fontsize=10, pad=8)
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#2e3757")
            ax.text(0, val * 0.5, f"{val:.3g}", ha="center", va="center",
                    color="white", fontsize=13, fontweight="bold")

        source_label = "ML Surrogate Prediction" if res_type == "ml" else "Physics Simulation"
        fig.suptitle(f"KPI Dashboard — {source_label}", color="white", fontsize=14, y=1.01)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 2: ML info OR Geometry ───────────────────────────────────────
    with tabs[1]:
        if res_type == "ml":
            meta = st.session_state.get("meta", {})
            st.markdown("#### 🧠 Surrogate Model Details")
            st.markdown(
                f"The ML model was trained on **{meta.get('n_training_rows', '—'):,} feasible cases** "
                f"out of **{meta.get('n_cases_completed', '—'):,} total simulations**. "
                f"It uses a `{meta.get('model_type', 'ExtraTreesRegressor')}` with 9 input features "
                f"and predicts 6 KPIs simultaneously."
            )
            st.markdown("---")
            st.markdown("#### R² Validation Scores (higher = better)")

            r2_data = []
            for col, r2_val in r2_summary(meta).items():
                bar_filled = int(round(r2_val * 20)) if not pd.isna(r2_val) else 0
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                quality = "Excellent" if r2_val >= 0.99 else "Good" if r2_val >= 0.97 else "Fair"
                r2_data.append({
                    "Target KPI":  col.replace("_", " "),
                    "R² Score":    f"{r2_val:.4f}",
                    "Visual":      bar,
                    "Quality":     quality,
                })
            st.dataframe(pd.DataFrame(r2_data), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### Input Feature Ranges (training data)")
            ranges = meta.get("ranges", {})
            if ranges:
                range_rows = [
                    {"Feature": k.replace("_", " "), "Min": v["min"], "Max": v["max"]}
                    for k, v in ranges.items()
                ]
                st.dataframe(pd.DataFrame(range_rows), use_container_width=True, hide_index=True)

        else:
            r_phys = st.session_state["results"]
            col_geo, col_diag = st.columns([1, 1])

            with col_geo:
                st.markdown("#### Tube Bundle Configuration")
                geo_data = {
                    "Parameter": [
                        "Parallel Reactors", "Tubes per Reactor", "Tube Length",
                        "Shell Diameter", "L/D Ratio", "Total Catalyst Volume",
                        "Total Reactor Volume",
                    ],
                    "Value": [
                        str(r_phys.n_parallel),
                        f"{r_phys.nt_per_reactor:,d}",
                        f"{r_phys.tube_length_m:.2f} m",
                        f"{r_phys.shell_diameter_m:.2f} m",
                        f"{r_phys.l_over_d:.2f}",
                        f"{r_phys.total_catalyst_volume_m3:.2f} m³",
                        f"{r_phys.reactor_volume_m3:.2f} m³",
                    ],
                }
                st.dataframe(pd.DataFrame(geo_data), use_container_width=True, hide_index=True)

            with col_diag:
                st.markdown("#### Reactor Diagram")
                diagram = f"""
```
  ╔═══════════════════════════╗
  ║   MULTITUBULAR FT REACTOR ║
  ║  ┌───────────────────┐    ║
  ║  │  Feed → → → → →   │    ║  Shell Ø {r_phys.shell_diameter_m:.2f} m
  ║  │ ┊┊┊ {r_phys.nt_per_reactor:>6,d} tubes ┊┊┊ │    ║  Tube Ø {tube_id*1000:.0f} mm ID
  ║  │ ┊┊┊  {r_phys.tube_length_m:.1f} m long  ┊┊┊ │    ║
  ║  │ ┊┊┊┊┊┊┊┊┊┊┊┊┊┊┊┊┊ │    ║
  ║  │  ↓ Products (C8–C16) │    ║
  ║  └───────────────────┘    ║
  ║  × {r_phys.n_parallel} reactors in parallel ║
  ╚═══════════════════════════╝
```
"""
                st.markdown(diagram)

    # ── Tab 3: Compare ML vs Physics OR Energy & Loop ────────────────────
    with tabs[2]:
        if res_type == "ml":
            st.markdown("#### 🔬 Compare ML Prediction vs Full Physics Simulation")
            st.markdown(
                "Click below to run the full equation-based simulation with the same inputs "
                "and see how close the ML model is."
            )

            if st.button("▶  Run Full Physics Simulation", type="secondary"):
                with st.spinner("Running physics simulation..."):
                    try:
                        r_phys = run_physics(st.session_state["params"])
                        st.session_state["compare_physics"] = r_phys
                    except Exception as exc:
                        st.error(f"Physics simulation failed: {exc}")

            if "compare_physics" in st.session_state:
                r_phys = st.session_state["compare_physics"]
                ml_res = st.session_state["results"]

                comparison = []
                pairs = [
                    ("Target Rate",        "target_rate_kgph",                  r_phys.target_rate_kgph,                   "kg/h"),
                    ("C8–C16 Fraction",    "target_fraction",                   r_phys.target_fraction,                    ""),
                    ("Specific Energy",    "specific_energy_kwh_per_kg_target", r_phys.specific_energy_kwh_per_kg_target,  "kWh/kg"),
                    ("Compressor Power",   "compressor_power_mw",               r_phys.compressor_power_mw,               "MW"),
                    ("Cooling Duty",       "cooling_duty_mw",                   r_phys.cooling_duty_mw,                   "MW"),
                    ("Pressure Drop",      "delta_p_bar",                       r_phys.delta_p_bar,                       "bar"),
                ]

                for label, key, phys_val, unit in pairs:
                    ml_val = ml_res[key]
                    diff_pct = abs(ml_val - phys_val) / abs(phys_val) * 100 if phys_val != 0 else 0
                    comparison.append({
                        "KPI":           f"{label} ({unit})" if unit else label,
                        "ML Surrogate":  f"{ml_val:.4g}",
                        "Full Physics":  f"{phys_val:.4g}",
                        "Error %":       f"{diff_pct:.2f}%",
                    })

                st.dataframe(
                    pd.DataFrame(comparison),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Error % = |ML − Physics| / |Physics| × 100. "
                    "The surrogate was trained with 80/20 split validation."
                )

        else:
            r_phys = st.session_state["results"]
            col_e1, col_e2 = st.columns(2)

            with col_e1:
                st.markdown("#### Recycle Loop")
                loop_data = {
                    "Stream":       ["Fresh Feed", "Reactor Inlet", "Recycle", "Purge"],
                    "Flow (kmol/h)": [
                        f"{r_phys.fresh_feed_kmol_h:,.2f}",
                        f"{r_phys.reactor_inlet_kmol_h:,.2f}",
                        f"{r_phys.recycle_kmol_h:,.2f}",
                        f"{r_phys.purge_kmol_h:,.2f}",
                    ],
                }
                st.dataframe(pd.DataFrame(loop_data), use_container_width=True, hide_index=True)

                kin_data = {
                    "Parameter": ["Single-pass CO conversion", "ASF alpha", "Loop iterations"],
                    "Value":     [
                        f"{r_phys.x_co_single_pass:.4f}",
                        f"{r_phys.alpha:.4f}",
                        str(r_phys.loop_iterations),
                    ],
                }
                st.dataframe(pd.DataFrame(kin_data), use_container_width=True, hide_index=True)

            with col_e2:
                st.markdown("#### Energy Balance")
                energy_data = {
                    "Component": [
                        "Compressor Power", "Cooling Duty", "Specific Energy",
                        "Heat of Reaction", "Gas Density", "Superficial Velocity",
                    ],
                    "Value": [
                        f"{r_phys.compressor_power_mw:.4f} MW",
                        f"{r_phys.cooling_duty_mw:.4f} MW",
                        f"{r_phys.specific_energy_kwh_per_kg_target:.4f} kWh/kg",
                        f"{r_phys.heat_of_reaction_kj_per_kmol_co:,.0f} kJ/kmol_CO",
                        f"{r_phys.gas_density_kg_m3:.3f} kg/m³",
                        f"{r_phys.superficial_velocity_m_s:.3f} m/s",
                    ],
                }
                st.dataframe(pd.DataFrame(energy_data), use_container_width=True, hide_index=True)

    # ── Tab 4: Dataset plots ─────────────────────────────────────────────
    with tabs[3]:
        plots_dir = PROJECT_ROOT / "plots"
        plot_files = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

        if not plot_files:
            st.info("No plots found. Run `python visualize.py` to generate them.")
        else:
            st.markdown(f"{len(plot_files)} pre-generated plots from the 21,000-case dataset.")
            for i in range(0, len(plot_files), 2):
                row_cols = st.columns(2)
                for j, col in enumerate(row_cols):
                    if i + j < len(plot_files):
                        pf = plot_files[i + j]
                        col.image(str(pf),
                                  caption=pf.stem.replace("_", " ").title(),
                                  use_container_width=True)
