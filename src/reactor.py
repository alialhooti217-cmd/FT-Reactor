"""
reactor.py
----------
Main FT reactor model wrapper for sizing, hydraulics, thermal estimates,
and optimization-ready outputs.
"""

import dataclasses
from typing import Dict, Any

from src.constants import MW, CARBON_NUMBER, TUBE_BUNDLE
from src.thermo import gas_density, volumetric_flow_m3_h, cp_mixture_kj_kmolk
from src.kinetics import kinetic_rate_kmol_m3_s, catalyst_volume_from_kinetics
from src.geometry import search_geometry
from src.hydraulics import ergun_pressure_drop_bar


@dataclasses.dataclass
class ReactorResults:
    total_inlet_flow_kmol_h: float
    catalyst_sizing_mode: str
    total_catalyst_volume_m3: float
    reactor_volume_m3: float
    n_parallel: int
    nt_per_reactor: int
    tube_length_m: float
    shell_length_m: float
    shell_diameter_m: float
    l_over_d: float
    superficial_velocity_m_s: float
    delta_p_bar: float
    gas_density_kg_m3: float
    cp_mix_kj_kmolk: float
    rco_kmol_m3_s: float
    x_co: float
    co_consumed_kmol_h: float
    q_rxn_kj_h: float
    q_rxn_mw: float
    delta_t_ad_C: float
    vcat_ghsv_m3: float
    vcat_kin_m3: float
    feasible: bool
    violation_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        return (
            "\n========================================\n"
            " FT REACTOR ISOLATED SIMULATION\n"
            "========================================\n"
            f"Total Inlet Flow          : {self.total_inlet_flow_kmol_h:,.2f} kmol/h\n"
            f"Catalyst Sizing Mode      : {self.catalyst_sizing_mode}\n"
            f"Total Catalyst Volume     : {self.total_catalyst_volume_m3:.2f} m3\n"
            f"Reactor Volume            : {self.reactor_volume_m3:.2f} m3\n"
            f"Vcat by GHSV              : {self.vcat_ghsv_m3:.2f} m3\n"
            f"Vcat by Kinetics          : {self.vcat_kin_m3:.2f} m3\n"
            f"CO Conversion             : {self.x_co:.4f}\n"
            f"CO Consumed               : {self.co_consumed_kmol_h:.2f} kmol/h\n"
            f"Number of Reactors (N)    : {self.n_parallel}\n"
            f"Tubes per Reactor (Nt)    : {self.nt_per_reactor:,}\n"
            f"Tube Length Selected      : {self.tube_length_m:.2f} m\n"
            f"Shell Length              : {self.shell_length_m:.2f} m\n"
            f"Shell Diameter (Ds)       : {self.shell_diameter_m:.2f} m\n"
            f"L/D Ratio                 : {self.l_over_d:.2f}\n"
            f"Superficial Velocity      : {self.superficial_velocity_m_s:.3f} m/s\n"
            f"Pressure Drop (dP)        : {self.delta_p_bar:.3f} bar\n"
            f"Gas Density               : {self.gas_density_kg_m3:.3f} kg/m3\n"
            f"Cp, mix                   : {self.cp_mix_kj_kmolk:.2f} kJ/(kmol.K)\n"
            f"r_CO                      : {self.rco_kmol_m3_s:.6f} kmol/m3_cat/s\n"
            f"Q_rxn                     : {self.q_rxn_mw:.3f} MW\n"
            f"Adiabatic dT              : {self.delta_t_ad_C:.2f} C\n"
            f"Feasible?                 : {'YES' if self.feasible else 'NO'}\n"
            f"Reason                    : {self.violation_reason}\n"
            "========================================\n"
        )


class FTReactor:
    def __init__(self, config: dict, feed_composition: dict):
        self.config = config
        self.feed_composition = feed_composition or {}

        op = config["operating_conditions"]
        design = config["design_basis"]
        geom_const = config["geometry_constraints"]
        reactor_geom = config["reactor_geometry"]
        bed = config["bed_properties"]
        kin = config["kinetics"]

        self.T_C = op["temperature_C"]
        self.P_bar = op["pressure_bar"]
        self.Z = op.get("compressibility_factor", 1.0)

        self.ghsv = design["ghsv_h"]
        self.max_Ds = design["max_shell_diameter_m"]
        self.N_max_search = design["reactors_max_search"]

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

        self.target_x_co = config.get("target_x_co", None)
        self.min_superficial_velocity = config.get("min_superficial_velocity_m_s", 0.0)
        self.max_superficial_velocity = config.get("max_superficial_velocity_m_s", 5.0)
        self.max_delta_p_bar = config.get("max_delta_p_bar", 5.0)

        self.reaction_extents = config.get("reaction_extents_kmol_h", {})
        self.dh_kj_per_kmol = config.get("dh_kj_per_kmol", {})

    def _total_inlet_flow(self) -> float:
        return sum(self.feed_composition.values())

    def _calculate_gas_density(self) -> float:
        return gas_density(
            P_bar=self.P_bar,
            T_C=self.T_C,
            feed=self.feed_composition,
            mw_dict=MW,
            z_factor=self.Z,
        )

    def _calculate_total_volumetric_flow(self) -> float:
        return volumetric_flow_m3_h(
            flow_kmol_h=self._total_inlet_flow(),
            T_C=self.T_C,
            P_bar=self.P_bar,
            z_factor=self.Z,
        )

    def _calculate_cp_mix(self) -> float:
        return cp_mixture_kj_kmolk(self.feed_composition, self.T_C)

    def _calculate_kinetic_rate(self) -> float:
        return kinetic_rate_kmol_m3_s(
            P_bar=self.P_bar,
            feed=self.feed_composition,
            k_rate=self.k_rate,
            a_ads=self.a_ads,
            rho_cat_kg_m3=self.rho_cat_kg_m3,
        )

    def _estimate_feed_h2_co_ratio(self) -> float:
        fco = self.feed_composition.get("CO", 0.0)
        fh2 = self.feed_composition.get("H2", 0.0)
        if fco <= 0.0:
            return 0.0
        return fh2 / fco

    def _co_consumed_from_extents(self) -> float:
        co_consumed = 0.0
        for comp, extent in self.reaction_extents.items():
            if comp in CARBON_NUMBER:
                co_consumed += CARBON_NUMBER[comp] * extent
        return co_consumed

    def _estimate_co_conversion_from_extents(self) -> float:
        fco_in = self.feed_composition.get("CO", 0.0)
        if fco_in <= 0.0:
            return 0.0
        x_co = self._co_consumed_from_extents() / fco_in
        return max(0.0, min(x_co, 0.999999))

    def _get_target_co_conversion(self) -> float:
        if self.target_x_co is not None:
            return max(0.0, min(self.target_x_co, 0.999999))
        return self._estimate_co_conversion_from_extents()

    def _calculate_bed_volume(self, rco_kmol_m3_s: float):
        total_vol_flow_m3_h = self._calculate_total_volumetric_flow()
        vcat_ghsv = total_vol_flow_m3_h / self.ghsv

        x_co = self._get_target_co_conversion()
        h2_co_ratio = self._estimate_feed_h2_co_ratio()

        vcat_kin = catalyst_volume_from_kinetics(
            feed=self.feed_composition,
            flow_in_kmol_h=self._total_inlet_flow(),
            h2_co_ratio=h2_co_ratio,
            target_x_co=x_co,
            rco_kmol_m3_s=rco_kmol_m3_s,
        )

        if self.Vcat_mode == "kinetics":
            cat_volume_total = vcat_kin
            sizing_mode = "KINETICS"
        elif self.Vcat_mode == "ghsv":
            cat_volume_total = vcat_ghsv
            sizing_mode = "GHSV"
        else:
            cat_volume_total = max(vcat_ghsv, vcat_kin)
            sizing_mode = "MAX"

        return cat_volume_total, vcat_ghsv, vcat_kin, sizing_mode, x_co

    def _calculate_heat_release(self):
        q_rxn_kj_h = 0.0
        for comp, extent in self.reaction_extents.items():
            dh = self.dh_kj_per_kmol.get(comp, 0.0)
            q_rxn_kj_h += extent * dh
        q_rxn_mw = q_rxn_kj_h / 3.6e6
        return q_rxn_kj_h, q_rxn_mw

    def _calculate_adiabatic_delta_t(self, q_rxn_kj_h: float, cp_mix_kj_kmolk: float) -> float:
        f_total = self._total_inlet_flow()
        if f_total <= 0.0 or cp_mix_kj_kmolk <= 0.0:
            return 0.0
        # Exothermic dh is negative, temperature rise reported as positive
        return (-q_rxn_kj_h) / (f_total * cp_mix_kj_kmolk)

    def _check_feasibility(self, geometry, delta_p_bar: float, superficial_velocity_m_s: float):
        violations = []

        if geometry.Ds > self.max_Ds:
            violations.append(f"Shell diameter exceeds limit ({geometry.Ds:.2f} > {self.max_Ds:.2f} m)")

        if not (self.LD_min <= geometry.L_over_D <= self.LD_max):
            violations.append(
                f"L/D out of bounds ({geometry.L_over_D:.2f} not in [{self.LD_min:.2f}, {self.LD_max:.2f}])"
            )

        if delta_p_bar > self.max_delta_p_bar:
            violations.append(f"Pressure drop exceeds limit ({delta_p_bar:.3f} > {self.max_delta_p_bar:.3f} bar)")

        if superficial_velocity_m_s < self.min_superficial_velocity:
            violations.append(
                f"Superficial velocity too low ({superficial_velocity_m_s:.3f} < {self.min_superficial_velocity:.3f} m/s)"
            )

        if superficial_velocity_m_s > self.max_superficial_velocity:
            violations.append(
                f"Superficial velocity too high ({superficial_velocity_m_s:.3f} > {self.max_superficial_velocity:.3f} m/s)"
            )

        if violations:
            return False, "; ".join(violations)

        return True, "All constraints satisfied"

    def run(self) -> ReactorResults:
        rho = self._calculate_gas_density()
        cp_mix = self._calculate_cp_mix()
        rco_kmol_m3_s = self._calculate_kinetic_rate()

        cat_volume_total, vcat_ghsv, vcat_kin, sizing_mode, x_co = self._calculate_bed_volume(
            rco_kmol_m3_s
        )

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

        total_vol_flow_m3_h = self._calculate_total_volumetric_flow()

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

        q_rxn_kj_h, q_rxn_mw = self._calculate_heat_release()
        delta_t_ad_C = self._calculate_adiabatic_delta_t(q_rxn_kj_h, cp_mix)

        feasible, reason = self._check_feasibility(
            geometry=geometry,
            delta_p_bar=delta_p_bar,
            superficial_velocity_m_s=superficial_velocity_m_s,
        )

        return ReactorResults(
            total_inlet_flow_kmol_h=self._total_inlet_flow(),
            catalyst_sizing_mode=sizing_mode,
            total_catalyst_volume_m3=cat_volume_total,
            reactor_volume_m3=reactor_volume_total,
            n_parallel=geometry.N,
            nt_per_reactor=geometry.Nt,
            tube_length_m=geometry.L_tube,
            shell_length_m=geometry.Lshell_m,
            shell_diameter_m=geometry.Ds,
            l_over_d=geometry.L_over_D,
            superficial_velocity_m_s=superficial_velocity_m_s,
            delta_p_bar=delta_p_bar,
            gas_density_kg_m3=rho,
            cp_mix_kj_kmolk=cp_mix,
            rco_kmol_m3_s=rco_kmol_m3_s,
            x_co=x_co,
            co_consumed_kmol_h=self._co_consumed_from_extents(),
            q_rxn_kj_h=q_rxn_kj_h,
            q_rxn_mw=q_rxn_mw,
            delta_t_ad_C=delta_t_ad_C,
            vcat_ghsv_m3=vcat_ghsv,
            vcat_kin_m3=vcat_kin,
            feasible=feasible,
            violation_reason=reason,
        )


def run_case(config: dict, feed_composition: dict) -> dict:
    """
    Optimization/ML-friendly single-case runner.
    Returns one flat dictionary row.
    """
    reactor = FTReactor(config=config, feed_composition=feed_composition)
    results = reactor.run()

    row = results.to_dict()

    # Add key inputs to the same row for dataset generation
    row["input_temperature_C"] = config["operating_conditions"]["temperature_C"]
    row["input_pressure_bar"] = config["operating_conditions"]["pressure_bar"]
    row["input_ghsv_h"] = config["design_basis"]["ghsv_h"]
    row["input_target_x_co"] = config.get("target_x_co", None)
    row["input_tube_inner_diameter_m"] = config["reactor_geometry"]["tube_inner_diameter_m"]
    row["input_particle_diameter_m"] = config["bed_properties"]["particle_diameter_m"]
    row["input_void_fraction"] = config["bed_properties"]["void_fraction"]

    return row
