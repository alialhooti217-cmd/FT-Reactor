# src/utilities.py
import math

# =========================
# THERMODYNAMIC CONSTANTS
# =========================
MW_DICT = {
    "H2": 2.016, "CO": 28.01, "CO2": 44.01, "H2O": 18.015,
    "C1": 16.043, "C3": 44.11, "C6": 86.1779, "C11": 156.31, "C18": 254.49,
}

CP_KJ_KMOL_K = {
    "C1": 102.244515, "C3": 134.5821805, "C6": 259.1919756,
    "C11": 477.5820572, "C18": 752.7197428, "CO": 30.7507513,
    "CO2": 48.40242581, "H2": 29.37548769, "H2O": 36.81863916,
}

# =========================
# HELPER FUNCTIONS
# =========================

def calculate_mixture_mw(flow_kmol_h: dict) -> float:
    """Calculates the average molecular weight (kg/kmol) of a gas mixture."""
    Ftot = sum(flow_kmol_h.values())
    if Ftot <= 0:
        return 0.0
    
    mw_mix = sum((F / Ftot) * MW_DICT.get(sp, 0.0) for sp, F in flow_kmol_h.items())
    return mw_mix

def calculate_mixture_cp(flow_kmol_h: dict) -> float:
    """Calculates the average heat capacity (kJ/kmol·K) of a gas mixture."""
    Ftot = sum(flow_kmol_h.values())
    if Ftot <= 0:
        return 0.0
    
    cp_mix = 0.0
    missing = []
    
    for sp, F in flow_kmol_h.items():
        if sp not in CP_KJ_KMOL_K:
            missing.append(sp)
            continue
        cp_mix += (F / Ftot) * CP_KJ_KMOL_K[sp]
        
    if missing:
        raise ValueError(f"Missing Cp for species: {missing}. Add them to utilities.py.")
        
    return cp_mix

def ideal_gas_density(P_bar: float, T_K: float, mw_mix: float, Z: float = 1.0) -> float:
    """Calculates gas density in kg/m³ using the Ideal Gas Law."""
    P_Pa = P_bar * 1e5
    R = 8.314e3  # Pa·m3/(kmol·K)
    return (P_Pa * mw_mix) / (Z * R * T_K)

# =========================
# ASF DISTRIBUTION MODELS
# =========================

def calculate_dynamic_alpha(T_C: float, h2_co_ratio: float, asf_params: dict) -> float:
    """
    Dynamically calculates chain growth probability (alpha) based on reactor 
    temperature and H2/CO ratio using Arrhenius kinetics.
    """
    ka = asf_params.get('ka', 0.157)
    beta = asf_params.get('beta', 0.28)
    Ea = asf_params.get('Ea_J_mol', 30100.0)
    
    T_K = T_C + 273.15
    T_ref_K = 220.0 + 273.15  # Reference temperature from Excel
    R = 8.314
    
    # Calculate temperature-dependent k_a
    k_T = ka * math.exp((Ea / R) * (1.0 / T_ref_K - 1.0 / T_K))
    
    # Calculate alpha
    alpha = 1.0 / (1.0 + k_T * (h2_co_ratio ** beta))
    
    return alpha

def get_pure_alkane_mw(n: int) -> float:
    """Calculates the standard molecular weight of a linear alkane (C_n H_2n+2)."""
    return (n * 12.011) + ((2 * n + 2) * 1.008)

def generate_asf_distribution(alpha: float, max_carbon: int = 30, 
                              is_modified: bool = True, 
                              y1_raw: float = 0.6329, 
                              y2_raw: float = 0.0447, 
                              y3_raw: float = 0.0523) -> dict:
    """
    Generates the ASF distribution (mole fractions) from C1 to C_max.
    """
    dist = {}
    
    if is_modified:
        # Apply empirical offsets for light gases
        dist[1] = y1_raw
        dist[2] = y2_raw
        dist[3] = y3_raw
        # Chain growth dictates the rest of the distribution
        for n in range(4, max_carbon + 1):
            dist[n] = dist[n - 1] * alpha
    else:
        # Normal ASF Equation
        for n in range(1, max_carbon + 1):
            dist[n] = (1.0 - alpha) * (alpha ** (n - 1))
            
    # Normalize the fractions so they sum exactly to 1.0
    total_y = sum(dist.values())
    normalized_dist = {n: val / total_y for n, val in dist.items()}
    
    return normalized_dist

def calculate_lumped_groups(normalized_dist: dict, group_ranges: dict) -> dict:
    """
    Groups the discrete carbon distribution into defined lumped fractions.
    """
    groups_result = {}
    
    for group_name, limits in group_ranges.items():
        # Handle YAML parsing if limits are passed as lists instead of tuples
        c_start, c_end = limits[0], limits[1]
        
        y_group_total = 0.0
        weighted_n = 0.0
        
        for n in range(c_start, c_end + 1):
            y_i = normalized_dist.get(n, 0.0)
            y_group_total += y_i
            weighted_n += (n * y_i)
            
        avg_n = weighted_n / y_group_total if y_group_total > 0 else 0.0
        avg_H = (2 * avg_n) + 2  # Assuming predominantly alkanes
        avg_MW = (avg_n * 12.011) + (avg_H * 1.008)
        
        groups_result[group_name] = {
            "mole_fraction": y_group_total,
            "avg_carbon": avg_n,
            "avg_mw": avg_MW,
            "formula": f"C{avg_n:.2f}H{avg_H:.2f}"
        }
        
    return groups_result
