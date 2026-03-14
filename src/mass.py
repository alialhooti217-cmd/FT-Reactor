"""Mass-balance helpers for the FT loop model."""

from __future__ import annotations

from typing import Dict, Tuple

from src.constants import MW


def component_balances(products_kmol_h: Dict[str, float]) -> dict:
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
    balances = component_balances(products_kmol_h)
    outlet = dict(inlet_stream)
    outlet["CO"] = max(0.0, inlet_stream.get("CO", 0.0) - balances["CO_consumed_kmol_h"])
    outlet["H2"] = max(0.0, inlet_stream.get("H2", 0.0) - balances["H2_consumed_kmol_h"])
    outlet["H2O"] = inlet_stream.get("H2O", 0.0) + balances["H2O_formed_kmol_h"]
    for comp, flow in products_kmol_h.items():
        outlet[comp] = outlet.get(comp, 0.0) + flow
    return outlet, balances


def separator_split(outlet_stream: Dict[str, float], gas_split: Dict[str, float]) -> Tuple[dict, dict]:
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
    purge_fraction = max(0.0, min(purge_fraction, 0.95))
    recycle = {comp: flow * (1.0 - purge_fraction) for comp, flow in gas_stream.items()}
    purge = {comp: flow * purge_fraction for comp, flow in gas_stream.items()}
    return recycle, purge


def combine_streams(*streams: Dict[str, float]) -> dict:
    total: Dict[str, float] = {}
    for stream in streams:
        for comp, flow in stream.items():
            total[comp] = total.get(comp, 0.0) + flow
    return total


def target_range_metrics(products_kmol_h: Dict[str, float], c_min: int, c_max: int) -> dict:
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
    bad = {k: v for k, v in stream.items() if v < -1e-9}
    if bad:
        raise ValueError(f"Negative flows detected in {name}: {bad}")
