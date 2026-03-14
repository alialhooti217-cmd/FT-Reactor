"""Global constants and helper data for the FT loop model."""

from __future__ import annotations

from typing import Dict

# Universal constants
R_PA_M3_PER_KMOLK = 8.314e3  # Pa.m^3/(kmol.K)
BAR_TO_PA = 1e5
BAR_TO_MPA = 0.1
SECONDS_PER_HOUR = 3600.0

# Tube-bundle correlation constants
TUBE_BUNDLE = {
    "Kt": 0.215,
    "n_exp": 2.207,
}

# Reactor/loop defaults
DEFAULT_TARGET_RANGE = {"c_min": 8, "c_max": 16}
DEFAULT_SEPARATOR_GAS_SPLIT = {
    "H2": 1.0,
    "CO": 1.0,
    "CO2": 1.0,
    "N2": 1.0,
    "Ar": 1.0,
    "H2O": 0.05,
}

# Modified ASF parameters from Marwan's latest implementation
ASF_DEFAULTS = {
    "ka": 0.157,
    "beta0": 0.28,
    "Ea_J_mol": 30100.0,
    "T_ref_K": 493.15,
    "y1": 0.632898507303832,
    "y2": 0.0447425425659602,
    "nmax": 20,
}

# Effective FT heat model on a CO-converted basis
FT_HEAT_OF_REACTION = {
    "dh_ref_kj_per_kmol_co": -165000.0,
    "t_ref_C": 220.0,
    "delta_cp_kj_per_kmolco_K": 35.0,
}


def paraffin_mw(n_carbon: int) -> float:
    """Approximate molecular weight of a linear paraffin CnH(2n+2) in kg/kmol."""
    if n_carbon < 1:
        raise ValueError("Carbon number must be >= 1")
    return n_carbon * 12.011 + (2 * n_carbon + 2) * 1.008


def build_mw_dict(nmax: int = 20) -> Dict[str, float]:
    mw = {
        "H2": 2.016,
        "CO": 28.01,
        "CO2": 44.01,
        "H2O": 18.015,
        "N2": 28.014,
        "Ar": 39.948,
    }
    for n in range(1, nmax + 1):
        mw[f"C{n}"] = paraffin_mw(n)
    return mw


MW = build_mw_dict(20)

# Cp correlations / approximations in kJ/(kmol.K)
CP_GAS_COEFF = {
    "H2": (29.11, -1.92e-3, 4.00e-6, -8.70e-10),
    "CO": (28.16, 1.67e-3, 5.37e-6, -2.22e-9),
    "CO2": (22.26, 5.981e-2, -3.501e-5, 7.47e-9),
    "H2O": (32.24, 1.92e-3, 1.055e-5, -3.60e-9),
    "N2": (28.90, -0.1571e-2, 0.8081e-5, -2.873e-9),
    "Ar": (20.786, 0.0, 0.0, 0.0),
    "C1": (19.89, 5.024e-2, 1.269e-5, -1.10e-8),
    "C2": (5.409, 1.781e-1, -6.938e-5, 1.095e-8),
}


def paraffin_cp_constant(n_carbon: int) -> float:
    """Simple engineering approximation for paraffin Cp above C2 in kJ/(kmol.K)."""
    return 10.0 + 41.0 * n_carbon
