"""Reactor geometry sizing utilities for multitubular fixed-bed reactors.

Provides tube-count, bundle-diameter, shell-ID, and pressure-drop geometry
calculations, together with ``search_geometry`` which performs a 2-D scan
over N (parallel reactors) and tube length to find the smallest feasible
configuration satisfying shell-diameter and L/D constraints.
"""

import math
import dataclasses


@dataclasses.dataclass
class GeometryResult:
    """Results from a single reactor geometry candidate.

    Attributes:
        N: Number of parallel reactor trains.
        Nt: Number of tubes per reactor.
        L_tube: Tube length in metres.
        Db: Bundle diameter in metres.
        Ds: Shell inner diameter (= Db + clearance) in metres.
        Lshell_m: Shell length (= 1.2 × L_tube) in metres.
        L_over_D: Shell length-to-diameter ratio.
        At_m2: Total tube-side flow area across all tubes (m²).
        status: ``"OK"`` if all constraints are met, otherwise ``"COMPROMISE"``.
        penalty: Sum-of-squared constraint violations (0 for feasible designs).
    """

    N: int
    Nt: int
    L_tube: float
    Db: float
    Ds: float
    Lshell_m: float
    L_over_D: float
    At_m2: float
    status: str
    penalty: float


def tube_internal_volume(di_m: float, L_m: float) -> float:
    """Return the internal volume of a single tube (m³)."""
    return (math.pi * di_m**2 / 4.0) * L_m


def number_of_tubes(v_reactor_m3: float, di_m: float, L_m: float) -> int:
    """Return the minimum integer tube count to hold a given reactor volume.

    Ceiling division ensures the design is never undersized.

    Args:
        v_reactor_m3: Required reactor (bed) volume in m³.
        di_m: Tube inner diameter in metres.
        L_m: Tube length in metres.

    Returns:
        Number of tubes (≥ 1).
    """
    v_tube = tube_internal_volume(di_m, L_m)
    if v_tube <= 0:
        raise ValueError("Tube volume must be positive.")
    return max(1, int(math.ceil(v_reactor_m3 / v_tube)))


def bundle_diameter(nt: int, do_m: float, kt: float, n_exp: float) -> float:
    """Calculate the tube-bundle outer diameter using the standard correlation.

    Formula: Db = Do × (Nt / Kt)^(1/n_exp)

    Args:
        nt: Number of tubes.
        do_m: Tube outer diameter in metres.
        kt: Bundle correlation constant (default 0.215 for triangular pitch).
        n_exp: Bundle correlation exponent (default 2.207).

    Returns:
        Bundle diameter Db in metres.
    """
    if nt <= 0:
        raise ValueError("Number of tubes must be positive.")
    if do_m <= 0 or kt <= 0 or n_exp <= 0:
        raise ValueError("Bundle diameter inputs must be positive.")
    return do_m * ((nt / kt) ** (1.0 / n_exp))


def shell_id(bundle_diameter_m: float, clearance_m: float) -> float:
    """Return the shell inner diameter (Ds = Db + clearance) in metres."""
    if bundle_diameter_m <= 0:
        raise ValueError("Bundle diameter must be positive.")
    if clearance_m < 0:
        raise ValueError("Clearance must be non-negative.")
    return bundle_diameter_m + clearance_m


def shell_length(L_tube_m: float) -> float:
    """Return the shell length (= 1.2 × tube length) to allow for end-caps."""
    if L_tube_m <= 0:
        raise ValueError("Tube length must be positive.")
    return 1.2 * L_tube_m


def total_tube_flow_area(nt: int, di_m: float) -> float:
    """Return the combined tube-side cross-sectional flow area (m²)."""
    if nt <= 0 or di_m <= 0:
        raise ValueError("Tube count and tube diameter must be positive.")
    return nt * (math.pi * di_m**2 / 4.0)


def frange(start: float, stop: float, step: float):
    """Generate a list of evenly spaced floats from start to stop (inclusive)."""
    if step <= 0:
        raise ValueError("Step must be positive.")
    values = []
    x = start
    while x <= stop + 1e-12:
        values.append(round(x, 10))
        x += step
    return values


def penalty(value: float, low: float, high: float) -> float:
    """Return a squared-violation penalty for a value outside [low, high].

    Returns 0 if ``low <= value <= high``, otherwise ``(distance to bound)²``.
    """
    if value < low:
        return (low - value) ** 2
    if value > high:
        return (value - high) ** 2
    return 0.0


def _selection_score(result: GeometryResult) -> tuple:
    return (
        result.penalty,
        result.N,
        result.Nt,
        result.L_tube,
        result.Ds,
    )


def search_geometry(
    cat_volume_total: float,
    eps: float,  # noqa: E741
    di_m: float,
    do_m: float,
    kt: float,
    n_exp: float,
    clearance_m: float,
    max_Ds: float,
    LD_min: float,
    LD_max: float,
    L_tube_min: float,
    L_tube_max: float,
    L_tube_step: float,
    N_max_search: int,
):
    """Search for the optimal reactor geometry over N and tube length.

    Performs a 2-D scan: for each number of parallel reactors N (1 … N_max_search)
    and each tube length in [L_tube_min, L_tube_max] (step L_tube_step), the
    function computes tube count, bundle diameter, shell diameter, and L/D.

    Preference order:
    1. Feasible designs (Ds ≤ max_Ds AND L/D in [LD_min, LD_max]).
    2. If no feasible design exists, the compromise with the smallest penalty.

    Among feasible designs the one with the lowest (N, Nt, L_tube, Ds) is
    returned, favouring fewer and larger reactors over many small ones.

    Args:
        cat_volume_total: Total catalyst volume required across all reactors (m³).
        eps: Bed void fraction (0 < eps < 1).
        di_m: Tube inner diameter (m).
        do_m: Tube outer diameter (m).
        kt: Bundle-diameter correlation constant.
        n_exp: Bundle-diameter correlation exponent.
        clearance_m: Diametral clearance between bundle and shell (m).
        max_Ds: Maximum allowable shell inner diameter (m).
        LD_min: Minimum acceptable shell L/D ratio.
        LD_max: Maximum acceptable shell L/D ratio.
        L_tube_min: Minimum tube length to search (m).
        L_tube_max: Maximum tube length to search (m).
        L_tube_step: Step size for tube length search (m).
        N_max_search: Maximum number of parallel reactors to try.

    Returns:
        :class:`GeometryResult` for the best found configuration, or ``None``
        if the search space is empty (should not occur for valid inputs).
    """
    if not (0.0 < eps < 1.0):
        raise ValueError("Void fraction eps must be between 0 and 1.")
    if cat_volume_total <= 0:
        raise ValueError("Total catalyst volume must be positive.")
    if di_m <= 0 or do_m <= 0:
        raise ValueError("Tube diameters must be positive.")
    if do_m <= di_m:
        raise ValueError("Outer tube diameter must be larger than inner diameter.")
    if max_Ds <= 0:
        raise ValueError("Maximum shell diameter must be positive.")
    if LD_min <= 0 or LD_max <= 0 or LD_max < LD_min:
        raise ValueError("Invalid L/D bounds.")
    if L_tube_min <= 0 or L_tube_max <= 0 or L_tube_max < L_tube_min:
        raise ValueError("Invalid tube length bounds.")
    if N_max_search < 1:
        raise ValueError("N_max_search must be at least 1.")

    v_reactor_total = cat_volume_total / (1.0 - eps)
    valid_results = []
    best_compromise = None

    for N in range(1, N_max_search + 1):
        v_reactor_each = v_reactor_total / N

        for L_tube in frange(L_tube_min, L_tube_max, L_tube_step):
            Nt = number_of_tubes(v_reactor_each, di_m, L_tube)
            Db = bundle_diameter(Nt, do_m, kt, n_exp)
            Ds = shell_id(Db, clearance_m)
            Lshell = shell_length(L_tube)
            LD = Lshell / Ds if Ds > 0 else float("inf")
            At = total_tube_flow_area(Nt, di_m)

            ok_D = Ds <= max_Ds
            ok_LD = LD_min <= LD <= LD_max

            result = GeometryResult(
                N=N,
                Nt=Nt,
                L_tube=L_tube,
                Db=Db,
                Ds=Ds,
                Lshell_m=Lshell,
                L_over_D=LD,
                At_m2=At,
                status="OK" if (ok_D and ok_LD) else "COMPROMISE",
                penalty=penalty(Ds, 0.0, max_Ds) + penalty(LD, LD_min, LD_max),
            )

            if ok_D and ok_LD:
                valid_results.append(result)
            else:
                if best_compromise is None or _selection_score(result) < _selection_score(best_compromise):
                    best_compromise = result

    if valid_results:
        return min(valid_results, key=_selection_score)

    return best_compromise
