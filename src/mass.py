"""Mass-balance helpers for the FT loop model."""

from __future__ import annotations

from typing import Dict, Tuple

from src.constants import MW


def component_balances(products_kmol_h: Dict[str, float]) -> dict:
    """Compute stoichiometric CO, H2, and H2O flows from paraffin products.

    For each paraffin Cn (linear alkane CnH(2n+2)):
        CO consumed  = n  × F_Cn
        H2 consumed  = (2n+1) × F_Cn
        H2O formed   = n  × F_Cn

    Args:
        products_kmol_h: Dict of hydrocarbon species (``"C1"``, ``"C2"``, …)
            → molar flow in kmol/h. Non-hydrocarbon keys are ignored.

    Returns:
        Dict with keys ``"CO_consumed_kmol_h"``, ``"H2_consumed_kmol_h"``,
        and ``"H2O_formed_kmol_h"``.
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


def apply_ft_stoichiometry(inlet_stream: Dict[str, float], products_kmol_h: Dict[str, float]) -> Tuple[dict, dict]:
    """Apply FT reaction stoichiometry to produce the reactor outlet stream.

    Subtracts consumed CO and H2 from the inlet, adds produced H2O and
    hydrocarbons, then returns both the updated outlet stream and the
    component balance summary.

    Args:
        inlet_stream: Dict of species → molar flow (kmol/h) entering the reactor.
        products_kmol_h: Dict of hydrocarbon products from the ASF model (kmol/h).

    Returns:
        Tuple of (outlet_stream, balances) where balances is the dict returned
        by :func:`component_balances`.
    """
    balances = component_balances(products_kmol_h)
    outlet = dict(inlet_stream)
    outlet["CO"] = max(0.0, inlet_stream.get("CO", 0.0) - balances["CO_consumed_kmol_h"])
    outlet["H2"] = max(0.0, inlet_stream.get("H2", 0.0) - balances["H2_consumed_kmol_h"])
    outlet["H2O"] = inlet_stream.get("H2O", 0.0) + balances["H2O_formed_kmol_h"]
    for comp, flow in products_kmol_h.items():
        outlet[comp] = outlet.get(comp, 0.0) + flow
    return outlet, balances


def separator_split(outlet_stream: Dict[str, float], gas_split: Dict[str, float]) -> Tuple[dict, dict]:
    """Split the reactor outlet between a gas phase and a liquid product phase.

    For each species, the fraction sent to gas is ``gas_split.get(comp)``.
    Light permanent gases (H2, CO, CO2, N2, Ar) default to 1.0 (all gas);
    all other species default to 0.0 (all liquid) if not in ``gas_split``.

    Args:
        outlet_stream: Dict of species → molar flow (kmol/h) leaving the reactor.
        gas_split: Dict of species → fraction routed to the gas phase (0–1).

    Returns:
        Tuple of (gas_stream, liquid_stream) dicts (kmol/h).
    """
    gas, liquid = {}, {}
    for comp, flow in outlet_stream.items():
        frac_to_gas = gas_split.get(comp)
        if frac_to_gas is None:
            frac_to_gas = 1.0 if comp in {"H2", "CO", "CO2", "N2", "Ar"} else 0.0
        frac_to_gas = max(0.0, min(frac_to_gas, 1.0))
        gas[comp] = flow * frac_to_gas
        liquid[comp] = flow * (1.0 - frac_to_gas)
    return gas, liquid


def recycle_and_purge(gas_stream: Dict[str, float], purge_fraction: float) -> Tuple[dict, dict]:
    """Split the separator gas into a recycle stream and a purge stream.

    A small purge fraction is required to prevent inerts (N2, Ar) and light
    hydrocarbons from accumulating in the recycle loop.

    Args:
        gas_stream: Dict of species → molar flow (kmol/h) from the separator.
        purge_fraction: Fraction of gas sent to purge (0–0.95). Values outside
            this range are clamped.

    Returns:
        Tuple of (recycle_stream, purge_stream) dicts (kmol/h).
    """
    purge_fraction = max(0.0, min(purge_fraction, 0.95))
    recycle = {comp: flow * (1.0 - purge_fraction) for comp, flow in gas_stream.items()}
    purge = {comp: flow * purge_fraction for comp, flow in gas_stream.items()}
    return recycle, purge


def combine_streams(*streams: Dict[str, float]) -> dict:
    """Combine multiple stream dicts by summing flows for each species.

    Args:
        *streams: Any number of dicts mapping species name to kmol/h.

    Returns:
        Combined stream dict (kmol/h).
    """
    total: Dict[str, float] = {}
    for stream in streams:
        for comp, flow in stream.items():
            total[comp] = total.get(comp, 0.0) + flow
    return total


def target_range_metrics(products_kmol_h: Dict[str, float], c_min: int, c_max: int) -> dict:
    """Calculate mass-flow and selectivity metrics for the target carbon-number cut.

    The target cut (default C8–C16) represents the diesel/gasoline-range
    hydrocarbons that the reactor is optimised to produce.

    Args:
        products_kmol_h: Dict of hydrocarbon species → molar flow (kmol/h).
        c_min: Lowest carbon number in the target range (inclusive).
        c_max: Highest carbon number in the target range (inclusive).

    Returns:
        Dict with keys:
            - ``"target_rate_kgph"``: Mass flow of target cut (kg/h).
            - ``"target_rate_kmolph"``: Molar flow of target cut (kmol/h).
            - ``"target_fraction"``: Mass-based selectivity to target cut.
            - ``"total_hydrocarbon_rate_kgph"``: Total hydrocarbon mass flow (kg/h).
    """
    target_kmol_h = 0.0
    target_kg_h = 0.0
    total_kg_h = 0.0
    for comp, flow in products_kmol_h.items():
        if not comp.startswith("C"):
            continue
        mw = MW.get(comp, 0.0)
        mass_rate = flow * mw
        total_kg_h += mass_rate
        n = int(comp[1:])
        if c_min <= n <= c_max:
            target_kmol_h += flow
            target_kg_h += mass_rate
    target_fraction = target_kg_h / total_kg_h if total_kg_h > 0 else 0.0
    return {
        "target_rate_kgph": target_kg_h,
        "target_rate_kmolph": target_kmol_h,
        "target_fraction": target_fraction,
        "total_hydrocarbon_rate_kgph": total_kg_h,
    }


def assert_nonnegative_stream(stream: Dict[str, float], name: str) -> None:
    """Raise ValueError if any species flow in the stream is significantly negative.

    A small tolerance of 1e-9 kmol/h is allowed for floating-point rounding.

    Args:
        stream: Dict of species → molar flow (kmol/h).
        name: Human-readable label used in the error message.

    Raises:
        ValueError: If any flow is below −1e-9 kmol/h.
    """
    bad = {k: v for k, v in stream.items() if v < -1e-9}
    if bad:
        raise ValueError(f"Negative flows detected in {name}: {bad}")
