"""Simple FT kinetic/severity helpers."""

from __future__ import annotations


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
    rco_mol_kg_s = k_rate * (ph2 ** 2) * pco / max(denom, 1e-12)
    return max((rco_mol_kg_s * rho_cat_kg_m3) / 1000.0, 1e-9)


def catalyst_volume_from_kinetics(feed, target_x_co, rco_kmol_m3_s):
    fco_kmol_h = feed.get("CO", 0.0)
    fco_kmol_s = fco_kmol_h / 3600.0
    if rco_kmol_m3_s <= 0:
        return float("inf")
    return (fco_kmol_s * target_x_co) / rco_kmol_m3_s
