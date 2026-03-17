"""Pressure-drop calculation for a packed-bed reactor using the Ergun equation."""

import math


def ergun_pressure_drop_bar(
    flow_m3_h_total,
    N_parallel,
    nt_per_reactor,
    di_m,
    eps,
    dp_m,
    mu_pa_s,
    rho_kg_m3,
    L_tube_m,
):
    """Calculate packed-bed pressure drop using the Ergun equation.

    The total volumetric flow is split equally among N_parallel reactors, then
    distributed across all tubes within a single reactor to obtain the
    superficial velocity. Both the viscous (Blake-Kozeny) and inertial
    (Burke-Plummer) terms are included.

    Args:
        flow_m3_h_total: Total volumetric flow rate at reactor conditions (m³/h).
        N_parallel: Number of parallel reactor trains.
        nt_per_reactor: Number of tubes per reactor.
        di_m: Tube inner diameter (m).
        eps: Bed void fraction.
        dp_m: Catalyst particle diameter (m).
        mu_pa_s: Gas dynamic viscosity (Pa·s).
        rho_kg_m3: Gas density (kg/m³).
        L_tube_m: Tube (packed-bed) length (m).

    Returns:
        Pressure drop across the bed in bar.

    Raises:
        ValueError: If the computed tube flow area is zero or negative.
    """
    flow_m3_s_per_reactor = (flow_m3_h_total / 3600.0) / N_parallel
    area_per_tube = math.pi * (di_m ** 2) / 4.0
    total_flow_area = nt_per_reactor * area_per_tube

    if total_flow_area <= 0:
        raise ValueError("Total tube flow area must be positive.")

    superficial_velocity = flow_m3_s_per_reactor / total_flow_area
    term1 = 150.0 * ((1.0 - eps) ** 2 / eps ** 3) * (mu_pa_s * superficial_velocity) / (dp_m ** 2)
    term2 = 1.75 * ((1.0 - eps) / eps ** 3) * (rho_kg_m3 * superficial_velocity ** 2) / dp_m
    delta_p_pa = (term1 + term2) * L_tube_m
    delta_p_bar = delta_p_pa / 1e5
    return delta_p_bar
