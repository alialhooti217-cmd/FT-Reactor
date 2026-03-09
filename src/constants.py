"""
constants.py
------------
Global constants and fixed physical-property data.
"""

# Lumped representative FT species
LUMPED_FT_SPECIES = ("C1", "C3", "C6", "C11", "C18")

# Molecular weights (kg/kmol)
MW = {
    "H2": 2.016,
    "CO": 28.01,
    "CO2": 44.01,
    "H2O": 18.015,
    "C1": 16.043,
    "C3": 44.11,
    "C6": 86.1779,
    "C11": 156.31,
    "C18": 254.49,
}

# Carbon numbers for lumped FT paraffins
CARBON_NUMBER = {
    "C1": 1,
    "C3": 3,
    "C6": 6,
    "C11": 11,
    "C18": 18,
}

# Universal constants
R_PA_M3_PER_KMOLK = 8.314e3  # Pa.m^3/(kmol.K)

# Unit conversions
SECONDS_PER_HOUR = 3600.0
BAR_TO_PA = 1e5
BAR_TO_MPA = 0.1

# Tube-bundle correlation constants
TUBE_BUNDLE = {
    "Kt": 0.215,
    "n_exp": 2.207,
}

# ------------------------------------------------------------------
# Cp correlations
# Form:
#   Cp = a + b*T + c*T^2 + d*T^3
# T in K
# Cp in kJ/(kmol.K)
#
# For permanent gases, polynomial coefficients are used.
# For lumped heavy FT species, constant Cp is used as a practical
# engineering approximation unless better correlations are available.
# ------------------------------------------------------------------

CP_GAS_COEFF = {
    "H2":  (29.11, -1.92e-3, 4.00e-6, -8.70e-10),
    "CO":  (28.16,  1.67e-3, 5.37e-6, -2.22e-9),
    "CO2": (22.26,  5.981e-2, -3.501e-5, 7.47e-9),
    "H2O": (32.24,  1.92e-3, 1.055e-5, -3.60e-9),
    "C1":  (19.89,  5.024e-2, 1.269e-5, -1.10e-8),
}

# Constant Cp for lumped hydrocarbons at representative FT conditions
CP_LUMPED_CONST = {
    "C3": 134.5821805,
    "C6": 259.1919756,
    "C11": 477.5820572,
    "C18": 752.7197428,
}