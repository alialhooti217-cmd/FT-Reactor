from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feed import build_total_feed
from src.hydraulics import ergun_pressure_drop_bar
from src.reactor import FTReactor


def load_config() -> dict:
    with open(PROJECT_ROOT / 'config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_checks() -> None:
    cfg = load_config()
    base = FTReactor(cfg, build_total_feed(cfg)).run()
    assert base.delta_p_bar <= cfg['design_basis']['max_delta_p_bar'] + 1e-9, 'Base case should satisfy max delta P'

    low_dp = ergun_pressure_drop_bar(
        flow_m3_h_total=1000.0, N_parallel=2, nt_per_reactor=2000, di_m=0.042, eps=0.42,
        dp_m=0.0014, mu_pa_s=2.0e-5, rho_kg_m3=8.0, L_tube_m=8.0
    )
    high_dp = ergun_pressure_drop_bar(
        flow_m3_h_total=1000.0, N_parallel=2, nt_per_reactor=2000, di_m=0.042, eps=0.42,
        dp_m=0.0008, mu_pa_s=2.0e-5, rho_kg_m3=8.0, L_tube_m=8.0
    )
    assert high_dp > low_dp, 'Smaller particle diameter should increase Ergun pressure drop'

    cfg_more_purge = copy.deepcopy(cfg)
    cfg_more_purge['loop_configuration']['purge_fraction'] = 0.08
    res_more_purge = FTReactor(cfg_more_purge, build_total_feed(cfg_more_purge)).run()
    assert res_more_purge.recycle_kmol_h < base.recycle_kmol_h, 'Higher purge should reduce recycle flow'
    print('All sanity checks passed.')


if __name__ == '__main__':
    run_checks()
