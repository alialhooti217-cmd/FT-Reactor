"""ASF (Anderson-Schulz-Flory) utilities.

This module isolates ASF logic from the mass-balance workflow so that the
project can keep `mass.py` focused on stoichiometry and loop balances.
"""

from __future__ import annotations

import math
from typing import Dict

from src.constants import ASF_DEFAULTS


def dynamic_alpha(T_C: float, h2_co_ratio: float, params: dict | None = None) -> float:
    params = {**ASF_DEFAULTS, **(params or {})}
    ka = params["ka"]
    beta0 = params["beta0"]
    ea = params["Ea_J_mol"]
    rg = 8.314
    t_ref = params["T_ref_K"]
    T_K = T_C + 273.15
    ratio = max(h2_co_ratio, 1e-9)
    exponent = (ea / rg) * (1.0 / t_ref - 1.0 / T_K)
    alpha = 1.0 / (1.0 + ka * (ratio ** beta0) * math.exp(exponent))
    return max(1e-6, min(alpha, 0.999999))


def modified_asf_distribution(alpha: float, nmax: int = 20, params: dict | None = None) -> Dict[int, float]:
    params = {**ASF_DEFAULTS, **(params or {})}
    y1 = params["y1"]
    y2 = params["y2"]
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")
    if nmax < 2:
        raise ValueError("nmax must be at least 2")

    gamma = 1.0 - (1.0 - y1) / alpha
    gamma = max(-10.0, min(gamma, 10.0))
    denom_rpar = (1.0 - alpha) * alpha * (1.0 - gamma)
    if abs(denom_rpar) < 1e-12:
        raise ValueError("Modified ASF parameters created a singular Rpar denominator")
    rpar = y2 / denom_rpar
    denom_beta = 1.0 - rpar * (1.0 - alpha)
    if abs(denom_beta) < 1e-12:
        raise ValueError("Modified ASF parameters created a singular beta denominator")
    beta = (1.0 - rpar) / denom_beta
    denom_series = 1.0 - beta * (1.0 - alpha)
    if abs(denom_series) < 1e-12:
        raise ValueError("Modified ASF parameters created a singular series denominator")

    raw: Dict[int, float] = {}
    raw[1] = 1.0 - alpha * (1.0 - gamma)
    raw[2] = (1.0 - alpha) * alpha * ((1.0 - beta) / denom_series) * (1.0 - gamma)
    for n in range(3, nmax + 1):
        raw[n] = (1.0 - alpha) * (alpha ** (n - 1)) * (1.0 - gamma) / denom_series

    total = sum(max(v, 0.0) for v in raw.values())
    if total <= 0.0:
        raise ValueError("ASF distribution sum is zero or negative")
    return {n: max(v, 0.0) / total for n, v in raw.items()}


def product_molar_flows_from_conversion(
    co_reacted_kmol_h: float,
    alpha: float,
    nmax: int = 20,
    params: dict | None = None,
) -> dict:
    """Map reacted CO into paraffin product molar flows using ASF carbon fractions."""
    dist = modified_asf_distribution(alpha=alpha, nmax=nmax, params=params)
    products = {}
    co_reacted_by_product = {}
    for n, frac in dist.items():
        carbon_to_n = co_reacted_kmol_h * frac
        co_reacted_by_product[f"C{n}"] = carbon_to_n
        products[f"C{n}"] = carbon_to_n / n
    return {
        "distribution": dist,
        "products_kmol_h": products,
        "co_reacted_kmol_h": co_reacted_kmol_h,
        "co_reacted_by_product_kmol_h": co_reacted_by_product,
    }
