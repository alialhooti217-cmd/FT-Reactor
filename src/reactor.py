"""FT loop model wrapper for reactor, recycle, separator, hydraulics and KPIs."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

from src.asf import dynamic_alpha, product_molar_flows_from_conversion
from src.constants import DEFAULT_SEPARATOR_GAS_SPLIT, DEFAULT_TARGET_RANGE, MW, TUBE_BUNDLE
from src.feed import build_total_feed
from src.geometry import search_geometry
from src.hydraulics import ergun_pressure_drop_bar
from src.kinetics import catalyst_volume_from_kinetics, kinetic_rate_kmol_m3_s
from src.mass import (
    apply_ft_stoichiometry,
    assert_nonnegative_stream,
    combine_streams,
    recycle_and_purge,
    separator_split,
    target_range_metrics,
)
from src.thermo import (
    cp_mixture_kj_kmolk,
    effective_ft_heat_of_reaction_kj_per_kmol_co,
    gas_density,
    volumetric_flow_m3_h,
)


@dataclasses.dataclass
class ReactorResults:
    fresh_feed_kmol_h: float
    reactor_inlet_kmol_h: float
    recycle_kmol_h: float
    purge_kmol_h: float
    purge_fraction: float
    x_co_single_pass: float
    alpha: float
    target_rate_kgph: float
    target_fraction: float
    total_hydrocarbon_rate_kgph: float
    specific_energy_kwh_per_kg_target: float
    compressor_power_mw: float
    cooling_duty_mw: float
    heat_of_reaction_kj_per_kmol_co: float
    total_inlet_flow_kmol_h: float
    total_catalyst_volume_m3: float
    reactor_volume_m3: float
    n_parallel: int
    nt_per_reactor: int
    tube_length_m: float
    shell_diameter_m: float
    l_over_d: float
    superficial_velocity_m_s: float
    delta_p_bar: float
    max_delta_p_bar: float
    gas_density_kg_m3: float
    cp_mix_kj_kmolk: float
    rco_kmol_m3_s: float
    feasible: bool
    violation_reason: str
    loop_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        return (
            "\n========================================\n"
            " FT LOOP SIMULATION\n"
            "========================================\n"
            f"Fresh feed                : {self.fresh_feed_kmol_h:,.2f} kmol/h\n"
            f"Reactor inlet             : {self.reactor_inlet_kmol_h:,.2f} kmol/h\n"
            f"Recycle / Purge           : {self.recycle_kmol_h:,.2f} / {self.purge_kmol_h:,.2f} kmol/h\n"
            f"Single-pass CO conversion : {self.x_co_single_pass:.4f}\n"
            f"ASF alpha                 : {self.alpha:.4f}\n"
            f"Target product rate       : {self.target_rate_kgph:,.2f} kg/h\n"
            f"Target fraction           : {self.target_fraction:.4f}\n"
            f"Specific energy           : {self.specific_energy_kwh_per_kg_target:.4f} kWh/kg\n"
            f"Compression power         : {self.compressor_power_mw:.4f} MW\n"
            f"Cooling duty              : {self.cooling_duty_mw:.4f} MW\n"
            f"Pressure drop             : {self.delta_p_bar:.4f} bar (limit {self.max_delta_p_bar:.4f})\n"
            f"Reactors / tubes          : {self.n_parallel} / {self.nt_per_reactor}\n"
            f"Tube length / shell D     : {self.tube_length_m:.2f} m / {self.shell_diameter_m:.2f} m\n"
            f"Feasible?                 : {'YES' if self.feasible else 'NO'}\n"
            f"Reason                    : {self.violation_reason}\n"
            "========================================\n"
        )


class FTReactor:
    def __init__(self, config: dict, feed_composition: dict | None = None):
        self.config = config
        self.fresh_feed = feed_composition or build_total_feed(config)

        op = config["operating_conditions"]
        design = config["design_basis"]
        geom_const = config["geometry_constraints"]
        reactor_geom = config["reactor_geometry"]
        bed = config["bed_properties"]
        kin = config["kinetics"]
        loop = config.get("loop_configuration", {})
        target = config.get("target_product", DEFAULT_TARGET_RANGE)

        self.T_C = op["temperature_C"]
        self.P_bar = op["pressure_bar"]
        self.Z = op.get("compressibility_factor", 1.0)

        self.ghsv = design["ghsv_h"]
        self.max_Ds = design["max_shell_diameter_m"]
        self.N_max_search = design["reactors_max_search"]
        self.max_delta_p_bar = design.get("max_delta_p_bar", 4.0)
        self.min_superficial_velocity = design.get("min_superficial_velocity_m_s", 0.0)
        self.max_superficial_velocity = design.get("max_superficial_velocity_m_s", 5.0)
        self.min_target_fraction = design.get("min_target_fraction", 0.0)

        self.LD_min = geom_const["LD_min"]
        self.LD_max = geom_const["LD_max"]
        self.L_tube_min = geom_const["tube_length_min_m"]
        self.L_tube_max = geom_const["tube_length_max_m"]
        self.L_tube_step = geom_const["tube_length_step_m"]

        self.Di = reactor_geom["tube_inner_diameter_m"]
        self.Do = reactor_geom["tube_outer_diameter_m"]
        self.clearance_m = reactor_geom["bundle_clearance_m"]

        self.eps = bed["void_fraction"]
        self.dp = bed["particle_diameter_m"]
        self.mu_Pa_s = bed["gas_viscosity_Pa_s"]

        self.k_rate = kin["rate_constant"]
        self.a_ads = kin["adsorption_constant"]
        self.rho_cat_kg_m3 = kin["catalyst_density_kg_m3"]
        self.Vcat_mode = kin.get("catalyst_volume_mode", "max").lower()

        self.Kt = TUBE_BUNDLE["Kt"]
        self.n_exp = TUBE_BUNDLE["n_exp"]

        self.target_x_co = config.get("target_x_co", op.get("single_pass_co_conversion", 0.7))
        self.purge_fraction = loop.get("purge_fraction", 0.05)
        self.recycle_pressure_ratio = loop.get("compressor_pressure_ratio", 1.05)
        self.compressor_efficiency = loop.get("compressor_efficiency", 0.75)
        self.max_loop_iterations = loop.get("max_loop_iterations", 50)
        self.loop_tolerance = loop.get("loop_tolerance", 1e-6)
        self.separator_gas_split = {**DEFAULT_SEPARATOR_GAS_SPLIT, **loop.get("separator_gas_split", {})}

        self.target_c_min = target.get("c_min", DEFAULT_TARGET_RANGE["c_min"])
        self.target_c_max = target.get("c_max", DEFAULT_TARGET_RANGE["c_max"])
        self.asf_params = config.get("asf_model", {})
        self.heat_cfg = config.get("heat_of_reaction", {})

    def _estimate_feed_h2_co_ratio(self, stream: dict) -> float:
        fco = stream.get("CO", 0.0)
        fh2 = stream.get("H2", 0.0)
        if fco <= 0.0:
            return 0.0
        return fh2 / fco

    def _calculate_gas_density(self, feed: dict) -> float:
        return gas_density(self.P_bar, self.T_C, feed, mw_dict=MW, z_factor=self.Z)

    def _calculate_total_volumetric_flow(self, feed: dict) -> float:
        return volumetric_flow_m3_h(sum(feed.values()), self.T_C, self.P_bar, z_factor=self.Z)

    def _calculate_cp_mix(self, feed: dict) -> float:
        return cp_mixture_kj_kmolk(feed, self.T_C)

    def _calculate_kinetic_rate(self, feed: dict) -> float:
        return kinetic_rate_kmol_m3_s(self.P_bar, feed, self.k_rate, self.a_ads, self.rho_cat_kg_m3)

    def _calculate_bed_volume(self, feed: dict, rco_kmol_m3_s: float):
        total_vol_flow_m3_h = self._calculate_total_volumetric_flow(feed)
        vcat_ghsv = total_vol_flow_m3_h / max(self.ghsv, 1e-9)
        vcat_kin = catalyst_volume_from_kinetics(feed, self.target_x_co, rco_kmol_m3_s)
        if self.Vcat_mode == "kinetics":
            cat_volume_total = vcat_kin
        elif self.Vcat_mode == "ghsv":
            cat_volume_total = vcat_ghsv
        else:
            cat_volume_total = max(vcat_ghsv, vcat_kin)
        return cat_volume_total, vcat_ghsv, vcat_kin

    def _compressor_power_mw(self, recycle_stream: dict) -> float:
        flow_kmol_h = sum(recycle_stream.values())
        if flow_kmol_h <= 0.0:
            return 0.0
        vdot_m3_h = volumetric_flow_m3_h(flow_kmol_h, self.T_C, self.P_bar, z_factor=self.Z)
        delta_p_bar = max((self.recycle_pressure_ratio - 1.0) * self.P_bar, 0.0)
        power_kw = (vdot_m3_h / 3600.0) * delta_p_bar * 1e5 / max(self.compressor_efficiency, 1e-9) / 1000.0
        return power_kw / 1000.0

    def _check_feasibility(self, geometry, delta_p_bar: float, superficial_velocity_m_s: float, target_fraction: float, inlet_stream: dict):
        violations = []
        if geometry.Ds > self.max_Ds:
            violations.append(f"Shell diameter exceeds limit ({geometry.Ds:.2f} > {self.max_Ds:.2f} m)")
        if not (self.LD_min <= geometry.L_over_D <= self.LD_max):
            violations.append(f"L/D out of bounds ({geometry.L_over_D:.2f} not in [{self.LD_min:.2f}, {self.LD_max:.2f}])")
        if delta_p_bar > self.max_delta_p_bar:
            violations.append(f"Pressure drop exceeds limit ({delta_p_bar:.3f} > {self.max_delta_p_bar:.3f} bar)")
        if superficial_velocity_m_s < self.min_superficial_velocity:
            violations.append(f"Superficial velocity too low ({superficial_velocity_m_s:.3f} < {self.min_superficial_velocity:.3f} m/s)")
        if superficial_velocity_m_s > self.max_superficial_velocity:
            violations.append(f"Superficial velocity too high ({superficial_velocity_m_s:.3f} > {self.max_superficial_velocity:.3f} m/s)")
        if target_fraction < self.min_target_fraction:
            violations.append(f"Target fraction below minimum ({target_fraction:.3f} < {self.min_target_fraction:.3f})")
        if inlet_stream.get("CO", 0.0) <= 0.0:
            violations.append("No CO available at reactor inlet")
        h2_needed = inlet_stream.get("CO", 0.0) * self.target_x_co * 2.1
        if inlet_stream.get("H2", 0.0) < h2_needed:
            violations.append("Insufficient H2 for requested single-pass CO conversion")
        return (len(violations) == 0), ("All constraints satisfied" if not violations else "; ".join(violations))

    def _run_loop(self):
        recycle_stream = {comp: 0.0 for comp in self.fresh_feed}
        last_total = None
        loop_data = {}
        for iteration in range(1, self.max_loop_iterations + 1):
            inlet_stream = combine_streams(self.fresh_feed, recycle_stream)
            assert_nonnegative_stream(inlet_stream, "reactor inlet")
            fco_in = inlet_stream.get("CO", 0.0)
            x_co = max(0.0, min(self.target_x_co, 0.98)) if fco_in > 0 else 0.0
            h2_co_ratio = self._estimate_feed_h2_co_ratio(inlet_stream)
            alpha = dynamic_alpha(self.T_C, h2_co_ratio, self.asf_params)
            co_reacted = fco_in * x_co
            asf_result = product_molar_flows_from_conversion(
                co_reacted_kmol_h=co_reacted,
                alpha=alpha,
                nmax=self.asf_params.get("nmax", 20),
                params=self.asf_params,
            )
            outlet_stream, stoich = apply_ft_stoichiometry(inlet_stream, asf_result["products_kmol_h"])
            gas_stream, liquid_stream = separator_split(outlet_stream, self.separator_gas_split)
            recycle_stream, purge_stream = recycle_and_purge(gas_stream, self.purge_fraction)
            total_recycle = sum(recycle_stream.values())
            loop_data = {
                "iteration": iteration,
                "inlet_stream": inlet_stream,
                "outlet_stream": outlet_stream,
                "gas_stream": gas_stream,
                "liquid_stream": liquid_stream,
                "recycle_stream": recycle_stream,
                "purge_stream": purge_stream,
                "stoich": stoich,
                "alpha": alpha,
                "x_co": x_co,
                "asf_result": asf_result,
            }
            if last_total is not None and abs(total_recycle - last_total) < self.loop_tolerance * max(1.0, total_recycle):
                break
            last_total = total_recycle
        return loop_data

    def run(self) -> ReactorResults:
        loop = self._run_loop()
        inlet_stream = loop["inlet_stream"]
        recycle_stream = loop["recycle_stream"]
        purge_stream = loop["purge_stream"]
        liquid_stream = loop["liquid_stream"]
        products = loop["asf_result"]["products_kmol_h"]

        rho = self._calculate_gas_density(inlet_stream)
        cp_mix = self._calculate_cp_mix(inlet_stream)
        rco_kmol_m3_s = self._calculate_kinetic_rate(inlet_stream)
        cat_volume_total, _, _ = self._calculate_bed_volume(inlet_stream, rco_kmol_m3_s)

        geometry = search_geometry(
            cat_volume_total=cat_volume_total,
            eps=self.eps,
            di_m=self.Di,
            do_m=self.Do,
            kt=self.Kt,
            n_exp=self.n_exp,
            clearance_m=self.clearance_m,
            max_Ds=self.max_Ds,
            LD_min=self.LD_min,
            LD_max=self.LD_max,
            L_tube_min=self.L_tube_min,
            L_tube_max=self.L_tube_max,
            L_tube_step=self.L_tube_step,
            N_max_search=self.N_max_search,
        )

        total_vol_flow_m3_h = self._calculate_total_volumetric_flow(inlet_stream)
        delta_p_bar = ergun_pressure_drop_bar(
            flow_m3_h_total=total_vol_flow_m3_h,
            N_parallel=geometry.N,
            nt_per_reactor=geometry.Nt,
            di_m=self.Di,
            eps=self.eps,
            dp_m=self.dp,
            mu_pa_s=self.mu_Pa_s,
            rho_kg_m3=rho,
            L_tube_m=geometry.L_tube,
        )
        superficial_velocity_m_s = total_vol_flow_m3_h / 3600.0 / geometry.At_m2
        reactor_volume_total = cat_volume_total / (1.0 - self.eps)

        target_metrics = target_range_metrics(products, self.target_c_min, self.target_c_max)
        dh_rxn = effective_ft_heat_of_reaction_kj_per_kmol_co(self.T_C, self.heat_cfg)
        q_rxn_kj_h = loop["stoich"]["CO_consumed_kmol_h"] * dh_rxn
        cooling_duty_mw = abs(q_rxn_kj_h) / 3.6e6
        compressor_power_mw = self._compressor_power_mw(recycle_stream)
        specific_energy = ((compressor_power_mw + cooling_duty_mw) * 1000.0 / target_metrics["target_rate_kgph"])
        if target_metrics["target_rate_kgph"] <= 0:
            specific_energy = float("inf")

        feasible, reason = self._check_feasibility(
            geometry=geometry,
            delta_p_bar=delta_p_bar,
            superficial_velocity_m_s=superficial_velocity_m_s,
            target_fraction=target_metrics["target_fraction"],
            inlet_stream=inlet_stream,
        )

        return ReactorResults(
            fresh_feed_kmol_h=sum(self.fresh_feed.values()),
            reactor_inlet_kmol_h=sum(inlet_stream.values()),
            recycle_kmol_h=sum(recycle_stream.values()),
            purge_kmol_h=sum(purge_stream.values()),
            purge_fraction=self.purge_fraction,
            x_co_single_pass=loop["x_co"],
            alpha=loop["alpha"],
            target_rate_kgph=target_metrics["target_rate_kgph"],
            target_fraction=target_metrics["target_fraction"],
            total_hydrocarbon_rate_kgph=target_metrics["total_hydrocarbon_rate_kgph"],
            specific_energy_kwh_per_kg_target=specific_energy,
            compressor_power_mw=compressor_power_mw,
            cooling_duty_mw=cooling_duty_mw,
            heat_of_reaction_kj_per_kmol_co=dh_rxn,
            total_inlet_flow_kmol_h=sum(inlet_stream.values()),
            total_catalyst_volume_m3=cat_volume_total,
            reactor_volume_m3=reactor_volume_total,
            n_parallel=geometry.N,
            nt_per_reactor=geometry.Nt,
            tube_length_m=geometry.L_tube,
            shell_diameter_m=geometry.Ds,
            l_over_d=geometry.L_over_D,
            superficial_velocity_m_s=superficial_velocity_m_s,
            delta_p_bar=delta_p_bar,
            max_delta_p_bar=self.max_delta_p_bar,
            gas_density_kg_m3=rho,
            cp_mix_kj_kmolk=cp_mix,
            rco_kmol_m3_s=rco_kmol_m3_s,
            feasible=feasible,
            violation_reason=reason,
            loop_iterations=loop["iteration"],
        )


def run_case(config: dict, feed_composition: dict | None = None) -> dict:
    reactor = FTReactor(config=config, feed_composition=feed_composition)
    results = reactor.run()
    row = results.to_dict()

    # Input features for surrogate model
    row["input_temperature_C"] = config["operating_conditions"]["temperature_C"]
    row["input_pressure_bar"] = config["operating_conditions"]["pressure_bar"]
    row["input_ghsv_h"] = config["design_basis"]["ghsv_h"]
    row["input_target_x_co"] = config.get("target_x_co", None)
    row["input_total_flow_kmol_h"] = config["feed"]["total_flow_kmol_h"]
    row["input_tube_inner_diameter_m"] = config["reactor_geometry"]["tube_inner_diameter_m"]
    row["input_particle_diameter_m"] = config["bed_properties"]["particle_diameter_m"]
    row["input_void_fraction"] = config["bed_properties"]["void_fraction"]
    row["input_reactors_max_search"] = config["design_basis"]["reactors_max_search"]
    row["input_purge_fraction"] = config.get("loop_configuration", {}).get("purge_fraction", 0.05)

    # Optional consistency check for ML target columns
    required_targets = [
        "target_rate_kgph",
        "target_fraction",
        "specific_energy_kwh_per_kg_target",
        "compressor_power_mw",
        "cooling_duty_mw",
        "delta_p_bar",
    ]

    missing_targets = [col for col in required_targets if col not in row]
    if missing_targets:
        raise KeyError(
            f"run_case() is missing required target columns for surrogate training: {missing_targets}"
        )

    return row