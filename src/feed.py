"""Feed composition helpers for the FT loop model.

Provides utilities to normalise mole-fraction compositions and build
the fresh-feed stream dict from a YAML config block.
"""

from __future__ import annotations


def normalize_composition(composition: dict) -> dict:
    """Normalise a composition dict so that all fractions sum to 1.

    If the values already sum to 1 (within 1e-9) the dict is returned
    unchanged. Otherwise each value is divided by the total.

    Args:
        composition: Dict mapping species name to mole fraction or flow.

    Returns:
        Normalised composition dict.

    Raises:
        ValueError: If the sum of all values is zero or negative.
    """
    frac_sum = sum(composition.values())
    if frac_sum <= 0:
        raise ValueError("Feed composition sum must be positive")
    if abs(frac_sum - 1.0) > 1e-9:
        return {k: v / frac_sum for k, v in composition.items()}
    return dict(composition)


def build_total_feed(config: dict) -> dict:
    """Build the fresh-feed stream (kmol/h) from the YAML configuration.

    Reads ``config["feed"]["total_flow_kmol_h"]`` and
    ``config["feed"]["composition"]``, normalises the composition, then
    scales by total flow.

    Args:
        config: Full configuration dict loaded from ``config.yaml``.

    Returns:
        Dict mapping species name to molar flow rate in kmol/h.
    """
    feed_cfg = config["feed"]
    total_flow = feed_cfg["total_flow_kmol_h"]
    composition = normalize_composition(feed_cfg["composition"])
    return {comp: total_flow * frac for comp, frac in composition.items()}
