"""Feed helpers."""

from __future__ import annotations


def normalize_composition(composition: dict) -> dict:
    frac_sum = sum(composition.values())
    if frac_sum <= 0:
        raise ValueError("Feed composition sum must be positive")
    if abs(frac_sum - 1.0) > 1e-9:
        return {k: v / frac_sum for k, v in composition.items()}
    return dict(composition)


def build_total_feed(config: dict) -> dict:
    feed_cfg = config["feed"]
    total_flow = feed_cfg["total_flow_kmol_h"]
    composition = normalize_composition(feed_cfg["composition"])
    return {comp: total_flow * frac for comp, frac in composition.items()}
