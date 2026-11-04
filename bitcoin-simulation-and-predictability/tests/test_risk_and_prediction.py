import numpy as np
import pandas as pd
import pytest
from scipy import stats

from btcsim.data import random_walk_control, uniform_price_control
from btcsim.predict import diebold_mariano, price_features, r2_oos, return_features, walk_forward
from btcsim.risk import christoffersen, kupiec_pof, quantile_loss


def test_kupiec_matches_hand_computation():
    hits = np.zeros(1000, dtype=int)
    hits[:20] = 1  # 20 violations where 10 are expected at p = 1 %
    r = kupiec_pof(hits, 0.01)
    lr = -2 * (980 * np.log(0.99) + 20 * np.log(0.01) - 980 * np.log(0.98) - 20 * np.log(0.02))
    assert r["LR_pof"] == pytest.approx(lr)
    assert r["p_pof"] == pytest.approx(stats.chi2.sf(lr, 1))


def test_christoffersen_flags_clustered_violations():
    rng = np.random.default_rng(0)
    spread = (rng.random(2000) < 0.05).astype(int)
    clustered = np.zeros(2000, dtype=int)
    clustered[500:600] = 1  # same number of hits, all in one run
    assert christoffersen(spread, 0.05)["p_ind"] > 0.01
    assert christoffersen(clustered, 0.05)["p_ind"] < 1e-6


def test_quantile_loss_prefers_the_true_quantile():
    r = np.random.default_rng(1).standard_t(4, 100_000)
    true_q = np.quantile(r, 0.01)
    assert quantile_loss(r, true_q, 0.01) < quantile_loss(r, true_q * 0.7, 0.01)
    assert quantile_loss(r, true_q, 0.01) < quantile_loss(r, true_q * 1.3, 0.01)


def test_iid_prices_have_mean_reverting_changes():
    """For i.i.d. X: corr(ΔX_{t+1}, ΔX_t) = −1/2  (the positive control relies on this)."""
    p = uniform_price_control(50_000, 0)["price_usd"].diff().dropna().to_numpy()
    assert np.corrcoef(p[1:], p[:-1])[0, 1] == pytest.approx(-0.5, abs=0.02)


def test_random_walk_returns_are_uncorrelated():
    r = random_walk_control(50_000, 0.03, 0)["log_return"].dropna().to_numpy()
    assert abs(np.corrcoef(r[1:], r[:-1])[0, 1]) < 0.02


def test_features_never_use_the_future():
    df = random_walk_control(600, 0.03, 1)
    t0 = df.index[400]
    altered = df.copy()
    altered.loc[altered.index > t0, ["price_usd", "volume_usd", "active_addresses", "transactions", "hash_rate"]] *= 3
    altered["log_return"] = np.log(altered["price_usd"]).diff()
    for f in (return_features, price_features):
        a, b = f(df), f(altered)
        rows = a.index[a.index <= t0]
        pd.testing.assert_frame_equal(a.loc[rows], b.loc[rows])


def test_walk_forward_trains_only_on_the_past():
    """A model that memorises its training targets can only 'know' days before the block it predicts."""
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    X = pd.DataFrame({"t": np.arange(300.0)}, index=idx)
    y = pd.Series(np.arange(300.0), index=idx)

    class MaxSeen:
        def fit(self, X, y):
            self.m = y.max()
            return self

        def predict(self, X):
            return np.full(len(X), self.m)

    res = walk_forward(X, y, {"max": MaxSeen}, idx[200], refit_every=30, min_train=10)
    assert (res["max"] < res["actual"]).all()


def test_dm_and_r2_oos():
    rng = np.random.default_rng(2)
    y = rng.normal(size=3000)
    good, bad = y + rng.normal(0, 0.5, 3000), y + rng.normal(0, 1.0, 3000)
    d = diebold_mariano(y - good, y - bad)
    assert d["DM"] < 0 and d["p_value"] < 1e-6
    assert r2_oos(y, y, np.zeros_like(y)) == 1.0
    assert r2_oos(y, np.zeros_like(y), np.zeros_like(y)) == 0.0
