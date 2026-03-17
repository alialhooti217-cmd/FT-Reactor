"""Simple FT kinetic/severity helpers."""

from __future__ import annotations


def kinetic_rate_kmol_m3_s(P_bar, feed, k_rate, a_ads, rho_cat_kg_m3):
    """Calculate the volumetric CO reaction rate using the FT rate law.

    Rate form (consistent with the paper):
        r_CO = k * P_H2² * P_CO / (1 + a * P_CO * P_H2²)   [mol/kg_cat/s]

    The result is converted to a volumetric rate by multiplying by the
    catalyst bulk density.

    Args:
        P_bar: Total pressure (bar).
        feed: Dict of species → molar flow (kmol/h); used for mole fractions.
        k_rate: Rate constant in mol/(kg_cat·s·MPa).
        a_ads: Adsorption constant in MPa⁻¹.
        rho_cat_kg_m3: Catalyst bulk/packing density (kg_cat/m³_bed).

    Returns:
        Volumetric CO consumption rate in kmol/(m³_bed·s). Returns 1e-9 if
        the feed is empty or the computed rate is non-positive.
    """
    f_tot = sum(feed.values())
    if f_tot <= 0:
        return 1e-9
    xco = feed.get("CO", 0.0) / f_tot
    xh2 = feed.get("H2", 0.0) / f_tot
    p_tot_mpa = P_bar * 0.1
    pco = xco * p_tot_mpa
    ph2 = xh2 * p_tot_mpa
    denom = 1.0 + a_ads * pco * (ph2 ** 2)
    rco_mol_kg_s = k_rate * (ph2 ** 2) * pco / max(denom, 1e-12)
    return max((rco_mol_kg_s * rho_cat_kg_m3) / 1000.0, 1e-9)


def catalyst_volume_from_kinetics(feed, target_x_co, rco_kmol_m3_s):
    """Estimate the required catalyst volume from a plug-flow design equation.

    Uses the simplified integral form for a first-order-like approximation:
        V_cat = F_CO0 * X_CO / r_CO

    Args:
        feed: Dict of species → molar flow (kmol/h).
        target_x_co: Target single-pass CO conversion (0–1).
        rco_kmol_m3_s: Volumetric CO reaction rate (kmol/m³/s).

    Returns:
        Required catalyst volume in m³. Returns ``float("inf")`` if the rate
        is zero or negative.
    """
    fco_kmol_h = feed.get("CO", 0.0)
    fco_kmol_s = fco_kmol_h / 3600.0
    if rco_kmol_m3_s <= 0:
        return float("inf")
    return (fco_kmol_s * target_x_co) / rco_kmol_m3_s
