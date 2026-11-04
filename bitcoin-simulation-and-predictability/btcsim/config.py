"""The whole experimental protocol in one place."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    data_path: Path = ROOT / "data" / "btc_daily_coinmetrics.csv"
    results_dir: Path = ROOT / "results"
    figures_dir: Path = ROOT / "figures"
    seed: int = 42

    start: pd.Timestamp = pd.Timestamp("2014-01-01")  # before 2014 the market was tiny and illiquid
    train_end: pd.Timestamp = pd.Timestamp("2020-12-31")  # models are fitted on 2014-2020 ...
    test_start: pd.Timestamp = pd.Timestamp("2021-01-01")  # ... and judged on 2021 - May 2026

    # Monte Carlo
    n_paths: int = 20_000
    fan_origin: pd.Timestamp = pd.Timestamp("2025-05-23")  # 1-year-ahead simulation, compared with what happened
    fan_horizon_days: int = 365
    block_length: int = 20  # mean block length of the stationary bootstrap

    # Risk
    var_levels: tuple[float, ...] = (0.95, 0.99)
    var_window: int = 365  # crypto trades every day
    refit_every: int = 30  # days between re-estimations in walk-forward loops

    # Predictability
    min_train_days: int = 730


CFG = Config()
