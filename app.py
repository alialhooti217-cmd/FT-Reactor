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
import json
import sys
import warnings
from datetime import datetime
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

    # Apply variable feed composition
    ratio = params.get("h2_co_ratio", 2.2)
    co2_f = params.get("co2_fraction", 0.02)
    n2_f = params.get("n2_fraction", 0.02)
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
    st.markdown("### Presets")
    PRESETS = {
        "Custom": {},
        "Default (Natural Gas)": {
            "temperature_C": 220.0, "pressure_bar": 25.0, "ghsv_h": 1800.0,
            "target_x_co": 0.72, "total_flow_kmol_h": 1200.0,
            "tube_inner_diameter_m": 0.042, "particle_diameter_m": 0.0012,
            "void_fraction": 0.42, "purge_fraction": 0.03,
            "h2_co_ratio": 2.2, "co2_fraction": 0.02, "n2_fraction": 0.02,
        },
        "Coal-Derived Syngas": {
            "temperature_C": 215.0, "pressure_bar": 26.0, "ghsv_h": 1600.0,
            "target_x_co": 0.68, "total_flow_kmol_h": 1500.0,
            "tube_inner_diameter_m": 0.044, "particle_diameter_m": 0.0013,
            "void_fraction": 0.43, "purge_fraction": 0.04,
            "h2_co_ratio": 1.7, "co2_fraction": 0.08, "n2_fraction": 0.05,
        },
        "Biomass-Derived Syngas": {
            "temperature_C": 218.0, "pressure_bar": 24.0, "ghsv_h": 1700.0,
            "target_x_co": 0.65, "total_flow_kmol_h": 800.0,
            "tube_inner_diameter_m": 0.042, "particle_diameter_m": 0.0012,
            "void_fraction": 0.42, "purge_fraction": 0.035,
            "h2_co_ratio": 1.5, "co2_fraction": 0.10, "n2_fraction": 0.08,
        },
        "High Throughput": {
            "temperature_C": 225.0, "pressure_bar": 27.0, "ghsv_h": 2000.0,
            "target_x_co": 0.75, "total_flow_kmol_h": 8000.0,
            "tube_inner_diameter_m": 0.045, "particle_diameter_m": 0.0014,
            "void_fraction": 0.44, "purge_fraction": 0.025,
            "h2_co_ratio": 2.1, "co2_fraction": 0.02, "n2_fraction": 0.02,
        },
    }

    preset_choice = st.selectbox("Load Preset", list(PRESETS.keys()), index=0)
    if preset_choice != "Custom" and PRESETS[preset_choice]:
        for k, v in PRESETS[preset_choice].items():
            st.session_state[f"preset_{k}"] = v

    def _get_preset(key, default):
        return st.session_state.pop(f"preset_{key}", default)

    st.markdown("---")
    st.markdown("### Operating Parameters")

    temperature_C = st.slider("🌡️ Temperature (°C)", 210.0, 235.0, _get_preset("temperature_C", 220.0), 0.5,
        help="Reactor operating temperature. Higher T → higher conversion but lower selectivity to heavy products")
    pressure_bar  = st.slider("🔵 Pressure (bar)",    20.0,  28.0,  _get_preset("pressure_bar", 25.0),  0.5,
        help="Reactor operating pressure. Higher P → higher conversion and heavier products")
    ghsv_h        = st.slider("💨 GHSV (h⁻¹)",      1500.0, 2200.0, _get_preset("ghsv_h", 1800.0), 50.0,
        help="Gas Hourly Space Velocity. Higher GHSV → less residence time → lower conversion but higher throughput")
    target_x_co   = st.slider("🎯 Target CO Conversion", 0.60, 0.78, _get_preset("target_x_co", 0.72), 0.01, format="%.2f",
        help="Desired single-pass CO conversion fraction")
    total_flow    = st.slider("🌊 Feed Flow (kmol/h)", 500.0, 10000.0, _get_preset("total_flow_kmol_h", 1200.0), 100.0,
        help="Total fresh syngas molar flow rate")

    st.markdown("---")
    st.markdown("### Feed Composition")
    h2_co_ratio = st.slider("⚛️ H₂/CO Ratio", 1.5, 3.0, _get_preset("h2_co_ratio", 2.2), 0.1,
        help="Hydrogen-to-carbon-monoxide molar ratio. Natural gas ~2.0–2.5, Coal ~0.7–1.0 (after WGS), Biomass ~1.0–2.0")
    co2_frac = st.slider("🫧 CO₂ Fraction", 0.0, 0.10, _get_preset("co2_fraction", 0.02), 0.005, format="%.3f",
        help="Carbon dioxide mole fraction in syngas feed")
    n2_frac = st.slider("🌬️ N₂ / Inerts Fraction", 0.0, 0.15, _get_preset("n2_fraction", 0.02), 0.005, format="%.3f",
        help="Nitrogen and inert gas mole fraction")

    st.markdown("---")
    st.markdown("### Geometry")
    tube_id   = st.slider("🔩 Tube Inner Diameter (m)", 0.040, 0.046, _get_preset("tube_inner_diameter_m", 0.042), 0.0005, format="%.4f",
        help="Inner diameter of reactor tubes. Larger tubes → more catalyst per tube but harder heat removal")
    dp_m      = st.slider("⚫ Particle Diameter (m)",  0.0011, 0.0015, _get_preset("particle_diameter_m", 0.0012), 0.00005, format="%.5f",
        help="Catalyst pellet diameter. Smaller particles → better kinetics but higher pressure drop")
    void_f    = st.slider("🕳️ Void Fraction",          0.41, 0.46, _get_preset("void_fraction", 0.42), 0.005, format="%.3f",
        help="Packed bed void fraction (porosity)")
    purge_f   = st.slider("💧 Purge Fraction",         0.020, 0.060, _get_preset("purge_fraction", 0.03), 0.002, format="%.3f",
        help="Fraction of recycle stream purged to prevent inert buildup")

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
    "h2_co_ratio": h2_co_ratio,
    "co2_fraction": co2_frac,
    "n2_fraction": n2_frac,
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
    error_msg = st.session_state["error"]
    if "convergence" in error_msg.lower() or "iteration" in error_msg.lower():
        st.error("Recycle loop did not converge. Try adjusting purge fraction or reducing CO conversion target.")
    elif "negative" in error_msg.lower() or "zero" in error_msg.lower():
        st.error("Invalid parameter combination produced negative flows. Try more moderate parameter values.")
    elif "feature" in error_msg.lower() or "column" in error_msg.lower():
        st.error("ML model feature mismatch. You may need to retrain: `python batch/run_batch.py`")
    else:
        st.error(f"Simulation error: {error_msg}")

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
        # Actionable fix suggestions
        fix_suggestions = []
        vr_lower = violation_reason.lower()
        if "pressure drop" in vr_lower or "δp" in vr_lower or "dp" in vr_lower:
            fix_suggestions.append("**Pressure drop too high**: Try increasing tube diameter, increasing particle size, reducing flow rate, or reducing GHSV.")
        if "target fraction" in vr_lower or "c8" in vr_lower or "selectivity" in vr_lower:
            fix_suggestions.append("**Target fraction too low**: Try lowering temperature (increases alpha → heavier products) or adjusting H₂/CO ratio closer to 2.0.")
        if "shell diameter" in vr_lower:
            fix_suggestions.append("**Shell too large**: Try reducing total flow or allowing more parallel reactors.")
        if "velocity" in vr_lower:
            fix_suggestions.append("**Velocity out of range**: Try adjusting flow rate, tube diameter, or number of reactors.")
        if "h2" in vr_lower or "insufficient" in vr_lower:
            fix_suggestions.append("**Insufficient H₂**: Try increasing H₂/CO ratio or reducing target CO conversion.")
        if fix_suggestions:
            with st.expander("💡 How to fix this"):
                for s in fix_suggestions:
                    st.markdown(f"- {s}")
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

    # Export buttons
    export_data = {**params, **kpis, "feasible": feasible, "violation_reason": violation_reason,
                   "source": res_type, "timestamp": datetime.now().isoformat()}
    col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 3])
    with col_exp1:
        csv_row = pd.DataFrame([export_data]).to_csv(index=False)
        st.download_button("📥 CSV", csv_row, "ft_results.csv", "text/csv", use_container_width=True)
    with col_exp2:
        json_str = json.dumps(export_data, indent=2, default=str)
        st.download_button("📥 JSON", json_str, "ft_results.json", "application/json", use_container_width=True)

    # Run history
    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []
    history = st.session_state["run_history"]
    # Add current run if it's new
    if run_btn or len(history) == 0:
        entry = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Mode": res_type.upper(),
            "T (°C)": params["temperature_C"],
            "P (bar)": params["pressure_bar"],
            "H₂/CO": params.get("h2_co_ratio", 2.2),
            "Rate (kg/h)": f"{kpis['target_rate_kgph']:.0f}",
            "Energy (kWh/kg)": f"{kpis['specific_energy_kwh_per_kg_target']:.2f}",
            "ΔP (bar)": f"{kpis['delta_p_bar']:.3f}",
            "Feasible": "Yes" if feasible else "No",
        }
        history.append(entry)
        if len(history) > 50:
            history.pop(0)

    st.markdown("---")

    # ── Tabs ────────────────────────────────────────────────────────────
    if res_type == "ml":
        tab_labels = ["📊 KPI Charts", "🔍 Sensitivity", "🧠 ML Model Info", "🔬 Compare with Physics", "📈 Dataset Plots"]
    else:
        tab_labels = ["📊 KPI Charts", "🔍 Sensitivity", "🏗️ Reactor Geometry", "⚡ Energy & Loop", "📈 Dataset Plots"]

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

    # ── Tab 2: Sensitivity Analysis ──────────────────────────────────────
    with tabs[1]:
        st.markdown("#### 🔍 Parameter Sensitivity Analysis")
        st.markdown("See how each parameter affects KPIs (uses ML surrogate for instant sweeps).")

        if not ml_ok:
            st.warning("Sensitivity analysis requires the ML surrogate model. Run `python batch/run_batch.py` first.")
        else:
            import numpy as np
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            if "sensitivity_model" not in st.session_state:
                st.session_state["sensitivity_model"], _ = get_surrogate()

            sens_model = st.session_state["sensitivity_model"]

            # Parameter sweep definitions
            sweep_params = [
                ("temperature_C",         "Temperature (°C)",       210.0, 235.0),
                ("pressure_bar",          "Pressure (bar)",          20.0,  28.0),
                ("ghsv_h",                "GHSV (h⁻¹)",           1500.0, 2200.0),
                ("target_x_co",           "CO Conversion",           0.60,  0.78),
                ("total_flow_kmol_h",     "Feed Flow (kmol/h)",    500.0, 10000.0),
                ("tube_inner_diameter_m", "Tube ID (m)",             0.040, 0.046),
                ("particle_diameter_m",   "Particle Dia (m)",        0.0011, 0.0015),
                ("void_fraction",         "Void Fraction",           0.41,  0.46),
                ("purge_fraction",        "Purge Fraction",          0.020, 0.060),
                ("h2_co_ratio",           "H₂/CO Ratio",            1.5,   3.0),
                ("co2_fraction",          "CO₂ Fraction",            0.0,   0.10),
                ("n2_fraction",           "N₂ Fraction",             0.0,   0.15),
            ]

            kpi_names = [
                ("target_rate_kgph",                  "C8-C16 Rate (kg/h)"),
                ("target_fraction",                   "C8-C16 Fraction"),
                ("specific_energy_kwh_per_kg_target", "Spec. Energy (kWh/kg)"),
                ("compressor_power_mw",               "Compressor (MW)"),
                ("cooling_duty_mw",                   "Cooling Duty (MW)"),
                ("delta_p_bar",                       "Pressure Drop (bar)"),
            ]

            selected_kpi_key, selected_kpi_label = kpi_names[0]  # default
            kpi_choice = st.selectbox(
                "Select KPI to analyze",
                options=[label for _, label in kpi_names],
                index=0,
            )
            for key, label in kpi_names:
                if label == kpi_choice:
                    selected_kpi_key = key
                    selected_kpi_label = label
                    break

            n_points = 50
            fig = make_subplots(rows=3, cols=4, subplot_titles=[label for _, label, _, _ in sweep_params])

            for idx, (pkey, plabel, pmin, pmax) in enumerate(sweep_params):
                row_idx = idx // 4 + 1
                col_idx = idx % 4 + 1
                x_vals = np.linspace(pmin, pmax, n_points)
                y_vals = []

                for xv in x_vals:
                    test_params = dict(params)
                    test_params[pkey] = float(xv)
                    pred = ml_predict(test_params, model=sens_model)
                    y_vals.append(pred[selected_kpi_key])

                fig.add_trace(
                    go.Scatter(x=x_vals, y=y_vals, mode="lines", name=plabel,
                               line=dict(width=2), showlegend=False),
                    row=row_idx, col=col_idx,
                )
                # Mark current operating point
                fig.add_trace(
                    go.Scatter(x=[params[pkey]], y=[kpis[selected_kpi_key]],
                               mode="markers", marker=dict(size=10, color="red", symbol="diamond"),
                               name="Current", showlegend=(idx == 0)),
                    row=row_idx, col=col_idx,
                )

            fig.update_layout(
                height=700, title_text=f"Sensitivity: {selected_kpi_label}",
                template="plotly_dark",
                margin=dict(t=80, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: ML info OR Geometry ───────────────────────────────────────
    with tabs[2]:
        if res_type == "ml":
            meta = st.session_state.get("meta", {})
            st.markdown("#### 🧠 Surrogate Model Details")
            st.markdown(
                f"The ML model was trained on **{meta.get('n_training_rows', '—'):,} feasible cases** "
                f"out of **{meta.get('n_cases_completed', '—'):,} total simulations**. "
                f"It uses a `{meta.get('model_type', 'ExtraTreesRegressor')}` with 12 input features "
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

    # ── Tab 4: Compare ML vs Physics OR Energy & Loop ────────────────────
    with tabs[3]:
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

    # ── Tab 5: Dataset plots ─────────────────────────────────────────────
    with tabs[4]:
        plots_dir = PROJECT_ROOT / "plots"
        plot_files = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

        if not plot_files:
            st.info("No plots found. Run `python visualize.py` to generate them.")
        else:
            st.markdown(f"{len(plot_files)} pre-generated plots from the dataset.")
            for i in range(0, len(plot_files), 2):
                row_cols = st.columns(2)
                for j, col in enumerate(row_cols):
                    if i + j < len(plot_files):
                        pf = plot_files[i + j]
                        col.image(str(pf),
                                  caption=pf.stem.replace("_", " ").title(),
                                  use_container_width=True)

    # ── Run History ──────────────────────────────────────────────────────
    if st.session_state.get("run_history"):
        with st.expander(f"📋 Run History ({len(st.session_state['run_history'])} runs)", expanded=False):
            st.dataframe(
                pd.DataFrame(reversed(st.session_state["run_history"])),
                use_container_width=True,
                hide_index=True,
            )
            if st.button("🗑️ Clear History"):
                st.session_state["run_history"] = []
                st.rerun()
