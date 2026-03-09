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
    """
    Calculate pressure drop across packed bed using Ergun equation.

    Parameters
    ----------
    flow_m3_h_total : float
        Total gas volumetric flow rate for all reactors (m3/h)
    N_parallel : int
        Number of parallel reactors
    nt_per_reactor : int
        Number of tubes per reactor
    di_m : float
        Tube inner diameter (m)
    eps : float
        Bed void fraction
    dp_m : float
        Catalyst particle diameter (m)
    mu_pa_s : float
        Gas viscosity (Pa.s)
    rho_kg_m3 : float
        Gas density (kg/m3)
    L_tube_m : float
        Tube length (m)

    Returns
    -------
    float
        Pressure drop in bar
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