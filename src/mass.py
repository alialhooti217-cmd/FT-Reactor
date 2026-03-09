##code for mass balance &ASF
import math


def asf_weight_fraction(alpha, n):
    """
    Anderson-Schulz-Flory mass fraction for carbon number n.

    Wn = n * (1 - alpha)^2 * alpha^(n - 1)

    Parameters
    ----------
    alpha : float
        Chain growth probability (0 < alpha < 1)
    n : int
        Carbon number

    Returns
    -------
    float
        ASF weight fraction for carbon number n
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    if n < 1:
        raise ValueError("Carbon number n must be >= 1.")

    return n * ((1.0 - alpha) ** 2) * (alpha ** (n - 1))


def asf_distribution(alpha, n_max):
    """
    Generate ASF distribution from C1 to Cn_max.

    Returns
    -------
    dict
        {carbon_number: fraction}
    """
    dist = {}
    for n in range(1, n_max + 1):
        dist[n] = asf_weight_fraction(alpha, n)

    total = sum(dist.values())
    if total <= 0:
        raise ValueError("ASF distribution sum is zero.")

    # Normalize so fractions sum to 1 over truncated range
    for n in dist:
        dist[n] /= total

    return dist


def range_fraction(distribution, c_min, c_max):
    """
    Sum ASF fraction over target carbon range.
    """
    if c_min < 1 or c_max < c_min:
        raise ValueError("Invalid carbon range.")

    return sum(frac for n, frac in distribution.items() if c_min <= n <= c_max)


def co_reacted_from_conversion(feed, x_co):
    """
    Calculate reacted CO flow from target CO conversion.

    Parameters
    ----------
    feed : dict
        Feed in kmol/h
    x_co : float
        CO conversion

    Returns
    -------
    float
        Reacted CO in kmol/h
    """
    fco_in = feed.get("CO", 0.0)
    return fco_in * x_co


def product_molar_flows_from_co_conversion(feed, x_co, alpha, n_max):
    """
    Convert reacted CO into hydrocarbon product molar flows using ASF.

    Assumption:
    - All hydrocarbons are paraffins CnH2n+2
    - ASF fractions are used on a carbon basis

    Logic:
    - Total reacted CO = total reacted carbon atoms
    - Carbon allocated to each n using ASF fraction
    - Product molar flow of Cn = carbon allocated to n / n

    Returns
    -------
    dict
        {
            "distribution": {n: fraction},
            "products_kmol_h": {"C1": ..., "C2": ..., ...},
            "co_reacted_kmol_h": ...,
        }
    """
    distribution = asf_distribution(alpha, n_max)
    co_reacted = co_reacted_from_conversion(feed, x_co)

    products = {}
    for n, frac in distribution.items():
        carbon_to_n = co_reacted * frac
        product_flow = carbon_to_n / n
        products[f"C{n}"] = product_flow

    return {
        "distribution": distribution,
        "products_kmol_h": products,
        "co_reacted_kmol_h": co_reacted,
    }


def component_balances(products_kmol_h):
    """
    Compute stoichiometric balances for paraffin FT products.

    Stoichiometry:
        n CO + (2n+1) H2 -> CnH2n+2 + n H2O

    For each 1 kmol of Cn produced:
    - CO consumed = n
    - H2 consumed = 2n + 1
    - H2O formed = n

    Returns
    -------
    dict
        {
            "CO_consumed_kmol_h": ...,
            "H2_consumed_kmol_h": ...,
            "H2O_formed_kmol_h": ...
        }
    """
    co_consumed = 0.0
    h2_consumed = 0.0
    h2o_formed = 0.0

    for comp, flow in products_kmol_h.items():
        if not comp.startswith("C"):
            continue

        n = int(comp[1:])
        co_consumed += n * flow
        h2_consumed += (2 * n + 1) * flow
        h2o_formed += n * flow

    return {
        "CO_consumed_kmol_h": co_consumed,
        "H2_consumed_kmol_h": h2_consumed,
        "H2O_formed_kmol_h": h2o_formed,
    }


def outlet_stream(feed, products_kmol_h):
    """
    Calculate reactor outlet stream based on feed and FT stoichiometry.

    Returns
    -------
    dict
        Outlet stream in kmol/h
    """
    balances = component_balances(products_kmol_h)

    out = dict(feed)

    out["CO"] = feed.get("CO", 0.0) - balances["CO_consumed_kmol_h"]
    out["H2"] = feed.get("H2", 0.0) - balances["H2_consumed_kmol_h"]
    out["H2O"] = feed.get("H2O", 0.0) + balances["H2O_formed_kmol_h"]

    for comp, flow in products_kmol_h.items():
        out[comp] = out.get(comp, 0.0) + flow

    return out


def target_range_production(products_kmol_h, c_min, c_max):
    """
    Calculate total production in a selected carbon range.

    Returns
    -------
    float
        Total target-range production in kmol/h
    """
    total = 0.0
    for comp, flow in products_kmol_h.items():
        if comp.startswith("C"):
            n = int(comp[1:])
            if c_min <= n <= c_max:
                total += flow
    return total


def ft_mass_balance(feed, x_co, alpha, n_max, c_min, c_max):
    """
    Full FT mass-balance workflow.

    Returns
    -------
    dict
        {
            "distribution": ...,
            "products_kmol_h": ...,
            "co_reacted_kmol_h": ...,
            "balances": ...,
            "outlet_stream_kmol_h": ...,
            "target_range_fraction": ...,
            "target_range_production_kmol_h": ...
        }
    """
    result = product_molar_flows_from_co_conversion(
        feed=feed,
        x_co=x_co,
        alpha=alpha,
        n_max=n_max,
    )

    distribution = result["distribution"]
    products = result["products_kmol_h"]

    balances = component_balances(products)
    out = outlet_stream(feed, products)
    target_frac = range_fraction(distribution, c_min, c_max)
    target_prod = target_range_production(products, c_min, c_max)

    return {
        "distribution": distribution,
        "products_kmol_h": products,
        "co_reacted_kmol_h": result["co_reacted_kmol_h"],
        "balances": balances,
        "outlet_stream_kmol_h": out,
        "target_range_fraction": target_frac,
        "target_range_production_kmol_h": target_prod,
    }
