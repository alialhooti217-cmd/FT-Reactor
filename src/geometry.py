"""
geometry.py
-----------
Geometry and reactor sizing utilities.
"""

import math
import dataclasses


@dataclasses.dataclass
class GeometryResult:
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
    return (math.pi * di_m**2 / 4.0) * L_m


def number_of_tubes(v_reactor_m3: float, di_m: float, L_m: float) -> int:
    v_tube = tube_internal_volume(di_m, L_m)
    if v_tube <= 0:
        raise ValueError("Tube volume must be positive.")
    return max(1, int(math.ceil(v_reactor_m3 / v_tube)))


def bundle_diameter(nt: int, do_m: float, kt: float, n_exp: float) -> float:
    return do_m * ((nt / kt) ** (1.0 / n_exp))


def shell_id(bundle_diameter_m: float, clearance_m: float) -> float:
    return bundle_diameter_m + clearance_m


def shell_length(L_tube_m: float) -> float:
    return 1.2 * L_tube_m


def total_tube_flow_area(nt: int, di_m: float) -> float:
    return nt * (math.pi * di_m**2 / 4.0)


def frange(start, stop, step):
    values = []
    x = start
    while x <= stop + 1e-12:
        values.append(round(x, 10))
        x += step
    return values


def penalty(value, low, high):
    if value < low:
        return (low - value) ** 2
    if value > high:
        return (value - high) ** 2
    return 0.0


def search_geometry(
    cat_volume_total: float,
    eps: float,
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
    v_reactor_total = cat_volume_total / (1.0 - eps)
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
                return result

            if best_compromise is None or result.penalty < best_compromise.penalty:
                best_compromise = result

    return best_compromise