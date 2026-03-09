def kinetic_rate_kmol_m3_s(P_bar, feed, k_rate, a_ads, rho_cat_kg_m3):
    f_tot = sum(feed.values())

    if f_tot <= 0:
        return 1e-9

    xco = feed.get("CO", 0.0) / f_tot
    xh2 = feed.get("H2", 0.0) / f_tot

    p_tot_mpa = P_bar * 0.1
    pco = xco * p_tot_mpa
    ph2 = xh2 * p_tot_mpa

    denom = 1.0 + a_ads * pco * (ph2 ** 2)
    rco_mol_kg_s = k_rate * (ph2 ** 2) * pco / denom

    return max((rco_mol_kg_s * rho_cat_kg_m3) / 1000.0, 1e-9)


def catalyst_volume_from_kinetics(feed, flow_in_kmol_h, h2_co_ratio, target_x_co, rco_kmol_m3_s):
    """
    Calculate catalyst volume from kinetic rate on CO basis.

    Parameters
    ----------
    feed : dict
        Feed composition in kmol/h
    flow_in_kmol_h : float
        Total inlet flow in kmol/h (used only as fallback)
    h2_co_ratio : float
        H2/CO ratio (used only as fallback)
    target_x_co : float
        CO conversion target
    rco_kmol_m3_s : float
        CO reaction rate in kmol/m3_cat/s
    """

    if feed and sum(feed.values()) > 0:
        fco_kmol_h = feed.get("CO", 0.0)
    else:
        fco_kmol_h = flow_in_kmol_h * (1.0 / (h2_co_ratio + 1.0))

    fco_kmol_s = fco_kmol_h / 3600.0

    if rco_kmol_m3_s <= 0:
        return float("inf")

    return (fco_kmol_s * target_x_co) / rco_kmol_m3_s