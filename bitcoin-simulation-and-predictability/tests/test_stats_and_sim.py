import numpy as np
import pandas as pd
import pytest
from scipy import stats

from btcsim import stats as S
from btcsim.simulate import GBM, GARCHt, StationaryBootstrap, simulate_garch


def test_ljung_box_has_correct_size_on_white_noise():
    rng = np.random.default_rng(0)
    p = [S.ljung_box(rng.normal(size=1000), 10)["p_value"] for _ in range(300)]
    assert 0.02 < np.mean(np.array(p) < 0.05) < 0.09  # rejects ~5 % of the time under H0


def test_ljung_box_detects_ar1():
    rng = np.random.default_rng(1)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.3 * x[t - 1] + rng.normal()
    assert S.ljung_box(x, 10)["p_value"] < 1e-6
    assert S.acf(x, 1)[0] == pytest.approx(0.3, abs=0.05)


def test_jarque_bera_normal_vs_t():
    rng = np.random.default_rng(2)
    assert S.jarque_bera(rng.normal(size=5000))["p_value"] > 0.01
    jb = S.jarque_bera(rng.standard_t(4, size=5000))
    assert jb["p_value"] < 1e-6 and jb["excess_kurtosis"] > 1


def test_max_drawdown():
    path = np.log(np.array([[1.0, 2.0, 1.0, 1.5, 0.5]]))
    assert S.max_drawdown(path)[0] == pytest.approx(0.75)  # from 2.0 down to 0.5


def test_gbm_moments_match_theory():
    g = GBM()
    g.mu_, g.sigma_ = 0.001, 0.03
    T = 100
    logret = g.simulate(200_000, T, np.random.default_rng(3)).sum(axis=1)
    assert np.mean(np.exp(logret)) == pytest.approx(np.exp(g.mu_ * T), rel=0.01)  # E[S_T/S_0] = e^{μT}
    assert np.std(logret) == pytest.approx(g.sigma_ * np.sqrt(T), rel=0.01)


def test_garch_mle_recovers_parameters():
    x = simulate_garch(omega=1e-5, alpha=0.10, beta=0.85, nu=5.0, mu=5e-4, n=20_000, rng=np.random.default_rng(4))
    g = GARCHt().fit(x)
    assert g.converged_
    assert g.alpha_ == pytest.approx(0.10, abs=0.02)
    assert g.beta_ == pytest.approx(0.85, abs=0.03)
    assert g.nu_ == pytest.approx(5.0, abs=0.8)
    assert g.half_life_days == pytest.approx(np.log(0.5) / np.log(g.alpha_ + g.beta_))


def test_garch_variance_forecast_uses_only_the_past():
    x = simulate_garch(1e-5, 0.1, 0.85, 5.0, 0.0, 3000, np.random.default_rng(5))
    g = GARCHt().fit(x)
    v = g.next_variance(x[:2000])
    y = x.copy()
    y[2000:] *= 10  # change the future
    assert g.next_variance(y[:2000]) == v


def test_block_bootstrap_keeps_blocks():
    r = np.arange(1000, dtype=float)
    sims = StationaryBootstrap(mean_block=50).fit(r).simulate(200, 500, np.random.default_rng(6))
    steps = np.diff(sims, axis=1)
    continued = np.mean((steps == 1) | (steps == -999))  # next day of the same block (or wrap-around)
    assert continued == pytest.approx(1 - 1 / 50, abs=0.01)
    assert set(np.unique(sims)) <= set(r)
