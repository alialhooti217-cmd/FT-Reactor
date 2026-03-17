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
    """Convert pressure from bar to Pascal."""
    return p_bar * BAR_TO_PA


def bar_to_mpa(p_bar: float) -> float:
    """Convert pressure from bar to MPa."""
    return p_bar * BAR_TO_MPA


def kmol_h_to_kmol_s(flow_kmol_h: float) -> float:
    """Convert molar flow from kmol/h to kmol/s."""
    return flow_kmol_h / 3600.0


def mixture_mw(feed: dict, mw_dict: dict = MW) -> float:
    """Calculate the mole-fraction-averaged mixture molecular weight (kg/kmol).

    Args:
        feed: Dict of species → molar flow (kmol/h).
        mw_dict: Molecular weight lookup table. Defaults to the project-wide
            :data:`~src.constants.MW` dict.

    Returns:
        Mixture MW in kg/kmol, or 0 if the total flow is zero.
    """
    total = sum(feed.values())
    if total <= 0:
        return 0.0
    return sum((flow / total) * mw_dict.get(comp, 0.0) for comp, flow in feed.items())


def volumetric_flow_m3_h(flow_kmol_h: float, T_C: float, P_bar: float, z_factor: float = 1.0) -> float:
    """Calculate volumetric flow rate from molar flow using the ideal gas law.

    V̇ = z × ṅ × R × T / P

    Args:
        flow_kmol_h: Total molar flow rate (kmol/h).
        T_C: Temperature (°C).
        P_bar: Pressure (bar).
        z_factor: Compressibility factor. Defaults to 1.0 (ideal gas).

    Returns:
        Volumetric flow rate in m³/h.
    """
    T_K = T_C + 273.15
    P_Pa = bar_to_pa(P_bar)
    flow_kmol_s = kmol_h_to_kmol_s(flow_kmol_h)
    vdot_m3_s = z_factor * flow_kmol_s * R_PA_M3_PER_KMOLK * T_K / max(P_Pa, 1e-9)
    return vdot_m3_s * 3600.0


def gas_density(P_bar: float, T_C: float, feed: dict, mw_dict: dict = MW, z_factor: float = 1.0) -> float:
    """Calculate gas mixture density using the ideal gas law (kg/m³).

    ρ = P × MW_mix / (z × R × T)

    Args:
        P_bar: Pressure (bar).
        T_C: Temperature (°C).
        feed: Dict of species → molar flow (kmol/h) used to compute MW_mix.
        mw_dict: Molecular weight lookup table. Defaults to project-wide MW.
        z_factor: Compressibility factor. Defaults to 1.0.

    Returns:
        Gas density in kg/m³.
    """
    T_K = T_C + 273.15
    P_Pa = bar_to_pa(P_bar)
    mw_mix = mixture_mw(feed, mw_dict)
    return P_Pa * mw_mix / (max(z_factor, 1e-9) * R_PA_M3_PER_KMOLK * T_K)


def cp_species_kj_kmolk(species: str, T_K: float) -> float:
    """Return the ideal-gas heat capacity of a pure species (kJ/kmol·K).

    Uses a cubic polynomial (a + bT + cT² + dT³) from :data:`~src.constants.CP_GAS_COEFF`
    for light gases and the engineering approximation from
    :func:`~src.constants.paraffin_cp_constant` for heavier paraffins (C3+).

    Args:
        species: Species identifier (e.g. ``"H2"``, ``"CO"``, ``"C5"``).
        T_K: Temperature in Kelvin.

    Returns:
        Cp in kJ/(kmol·K).

    Raises:
        KeyError: If the species is not in the Cp tables.
    """
    if species in CP_GAS_COEFF:
        a, b, c, d = CP_GAS_COEFF[species]
        return a + b * T_K + c * (T_K ** 2) + d * (T_K ** 3)
    if species.startswith("C") and species[1:].isdigit():
        return paraffin_cp_constant(int(species[1:]))
    raise KeyError(f"No Cp data found for species '{species}'.")


def cp_mixture_kj_kmolk(feed: dict, T_C: float) -> float:
    """Calculate the mole-fraction-averaged mixture heat capacity (kJ/kmol·K).

    Args:
        feed: Dict of species → molar flow (kmol/h).
        T_C: Temperature (°C).

    Returns:
        Mixture Cp in kJ/(kmol·K), or 0 if the total flow is zero.
    """
    total = sum(feed.values())
    if total <= 0:
        return 0.0
    T_K = T_C + 273.15
    cp_mix = 0.0
    for comp, flow in feed.items():
        cp_mix += (flow / total) * cp_species_kj_kmolk(comp, T_K)
    return cp_mix


def effective_ft_heat_of_reaction_kj_per_kmol_co(T_C: float, heat_cfg: dict | None = None) -> float:
    """Return the effective FT heat of reaction on a per-kmol-CO basis (kJ/kmol_CO).

    Uses a linear temperature correction around a reference point:
        ΔH(T) = ΔH_ref + ΔCp × (T − T_ref)

    The value is negative (exothermic). Default reference: −165,000 kJ/kmol_CO
    at 220 °C.

    Args:
        T_C: Reactor temperature (°C).
        heat_cfg: Optional dict to override keys in
            :data:`~src.constants.FT_HEAT_OF_REACTION`
            (``dh_ref_kj_per_kmol_co``, ``t_ref_C``,
            ``delta_cp_kj_per_kmolco_K``).

    Returns:
        Effective heat of reaction in kJ/kmol_CO (negative = exothermic).
    """
    cfg = {**FT_HEAT_OF_REACTION, **(heat_cfg or {})}
    dh_ref = cfg["dh_ref_kj_per_kmol_co"]
    t_ref = cfg["t_ref_C"]
    delta_cp = cfg["delta_cp_kj_per_kmolco_K"]
    return dh_ref + delta_cp * (T_C - t_ref)
