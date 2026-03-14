"""Thermodynamic helper functions."""

from __future__ import annotations

from src.constants import (
    BAR_TO_MPA,
    BAR_TO_PA,
    CP_GAS_COEFF,
    FT_HEAT_OF_REACTION,
    MW,
    R_PA_M3_PER_KMOLK,
    paraffin_cp_constant,
)


def bar_to_pa(p_bar: float) -> float:
    return p_bar * BAR_TO_PA


def bar_to_mpa(p_bar: float) -> float:
    return p_bar * BAR_TO_MPA


def kmol_h_to_kmol_s(flow_kmol_h: float) -> float:
    return flow_kmol_h / 3600.0


def mixture_mw(feed: dict, mw_dict: dict = MW) -> float:
    total = sum(feed.values())
    if total <= 0:
        return 0.0
    return sum((flow / total) * mw_dict.get(comp, 0.0) for comp, flow in feed.items())


def volumetric_flow_m3_h(flow_kmol_h: float, T_C: float, P_bar: float, z_factor: float = 1.0) -> float:
    T_K = T_C + 273.15
    P_Pa = bar_to_pa(P_bar)
    flow_kmol_s = kmol_h_to_kmol_s(flow_kmol_h)
    vdot_m3_s = z_factor * flow_kmol_s * R_PA_M3_PER_KMOLK * T_K / max(P_Pa, 1e-9)
    return vdot_m3_s * 3600.0


def gas_density(P_bar: float, T_C: float, feed: dict, mw_dict: dict = MW, z_factor: float = 1.0) -> float:
    T_K = T_C + 273.15
    P_Pa = bar_to_pa(P_bar)
    mw_mix = mixture_mw(feed, mw_dict)
    return P_Pa * mw_mix / (max(z_factor, 1e-9) * R_PA_M3_PER_KMOLK * T_K)


def cp_species_kj_kmolk(species: str, T_K: float) -> float:
    if species in CP_GAS_COEFF:
        a, b, c, d = CP_GAS_COEFF[species]
        return a + b * T_K + c * (T_K ** 2) + d * (T_K ** 3)
    if species.startswith("C") and species[1:].isdigit():
        return paraffin_cp_constant(int(species[1:]))
    raise KeyError(f"No Cp data found for species '{species}'.")


def cp_mixture_kj_kmolk(feed: dict, T_C: float) -> float:
    total = sum(feed.values())
    if total <= 0:
        return 0.0
    T_K = T_C + 273.15
    cp_mix = 0.0
    for comp, flow in feed.items():
        cp_mix += (flow / total) * cp_species_kj_kmolk(comp, T_K)
    return cp_mix


def effective_ft_heat_of_reaction_kj_per_kmol_co(T_C: float, heat_cfg: dict | None = None) -> float:
    cfg = {**FT_HEAT_OF_REACTION, **(heat_cfg or {})}
    dh_ref = cfg["dh_ref_kj_per_kmol_co"]
    t_ref = cfg["t_ref_C"]
    delta_cp = cfg["delta_cp_kj_per_kmolco_K"]
    return dh_ref + delta_cp * (T_C - t_ref)
