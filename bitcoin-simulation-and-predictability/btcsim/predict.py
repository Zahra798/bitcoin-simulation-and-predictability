r"""Can tomorrow's Bitcoin move be predicted from public daily data?

Everything is evaluated walk-forward: on each test day t, models may only use
data up to t, and they are re-estimated every `refit_every` days on all data
seen so far (an expanding window). This mimics real use and makes leakage
impossible by construction.

Scores:
* out-of-sample R² against a benchmark b (Campbell & Thompson, 2008):
      R²_oos = 1 − Σ (r_t − r̂_t)² / Σ (r_t − b_t)²,
  R²_oos > 0 means the model beats the benchmark.
* Diebold–Mariano test of equal squared error against the random walk.
* Directional accuracy with an exact binomial test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------------------
# Features (all known at the end of day t)
# ---------------------------------------------------------------------------
def return_features(df: pd.DataFrame) -> pd.DataFrame:
    r = df["log_return"]
    lp = np.log(df["price_usd"])
    X = pd.DataFrame(index=df.index)
    for k in range(10):
        X[f"r_lag{k}"] = r.shift(k)
    X["abs_r"] = r.abs()
    for w in (7, 30):
        X[f"mean_r_{w}d"] = r.rolling(w).mean()
        X[f"vol_{w}d"] = r.rolling(w).std()
    X["vol_ratio"] = X["vol_7d"] / X["vol_30d"]
    X["dist_ma50"] = lp - lp.rolling(50).mean()
    X["dist_ma200"] = lp - lp.rolling(200).mean()
    lv = np.log(df["volume_usd"].clip(lower=1))
    X["volume_change_1d"] = lv.diff()
    X["volume_vs_30d"] = lv - lv.rolling(30).mean()
    for c in ("active_addresses", "transactions", "hash_rate"):
        X[f"{c}_change_7d"] = np.log(df[c].clip(lower=1)).diff(7)
    dow = df.index.dayofweek.to_numpy()
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    return X


def price_features(df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """The 'textbook' setup: yesterday's prices as inputs for tomorrow's price."""
    p = df["price_usd"]
    return pd.DataFrame({f"p_lag{k}": p.shift(k) for k in range(lags)}, index=df.index).dropna()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def regressors(seed: int) -> dict:
    return {
        "Ridge regression": lambda: make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))),
        "Gradient boosting": lambda: HistGradientBoostingRegressor(learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
                                                                  min_samples_leaf=50, random_state=seed),
        "Neural network (MLP)": lambda: make_pipeline(SimpleImputer(), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, max_iter=300, random_state=seed)),
    }


def classifiers(seed: int) -> dict:
    return {
        "Logistic regression": lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
                                                                   min_samples_leaf=50, random_state=seed),
    }


def price_models(seed: int) -> dict:
    return {
        "Ridge on past prices": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))),
        "Gradient boosting on past prices": lambda: HistGradientBoostingRegressor(max_iter=300, random_state=seed),
        "Neural network on min-max scaled prices": lambda: make_pipeline(MinMaxScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=seed)),
    }


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def walk_forward(X: pd.DataFrame, y: pd.Series, factories: dict, test_start, refit_every: int,
                 min_train: int, proba: bool = False) -> pd.DataFrame:
    """Predict y_t for every t ≥ test_start; each model is refitted every `refit_every`
    steps on all rows whose TARGET is already known at the time of fitting."""
    ok = y.notna()
    X, y = X[ok], y[ok]
    test_idx = X.index[X.index >= test_start]
    preds = {name: pd.Series(np.nan, index=test_idx) for name in factories}
    for i0 in range(0, len(test_idx), refit_every):
        block = test_idx[i0:i0 + refit_every]
        # rows strictly before the first day of the block; their targets (t+1) are ≤ block start
        train = X.index[X.index < block[0]]
        if len(train) < min_train:
            continue
        for name, make in factories.items():
            m = make().fit(X.loc[train], y.loc[train])
            p = m.predict_proba(X.loc[block])[:, 1] if proba else m.predict(X.loc[block])
            preds[name].loc[block] = p
    out = pd.DataFrame(preds)
    out.insert(0, "actual", y.loc[test_idx])
    return out


# ---------------------------------------------------------------------------
# Tests and scores
# ---------------------------------------------------------------------------
def diebold_mariano(e1, e2, lags: int | None = None) -> dict:
    """DM test of equal mean squared error (Newey–West variance, HLN correction, h = 1)."""
    d = np.asarray(e1, dtype=float) ** 2 - np.asarray(e2, dtype=float) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    L = lags if lags is not None else int(4 * (n / 100) ** (2 / 9))
    dc = d - d.mean()
    v = np.dot(dc, dc) / n + sum(2 * (1 - j / (L + 1)) * np.dot(dc[j:], dc[:-j]) / n for j in range(1, L + 1))
    stat = d.mean() / np.sqrt(v / n) * np.sqrt((n - 1) / n)
    return {"DM": float(stat), "p_value": float(2 * stats.t.sf(abs(stat), n - 1))}


def r2_oos(y, pred, bench) -> float:
    y, pred, bench = (np.asarray(v, dtype=float) for v in (y, pred, bench))
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - bench) ** 2))


def return_scores(res: pd.DataFrame, hist_mean: pd.Series) -> pd.DataFrame:
    y = res["actual"]
    rows = []
    cands = {"Random walk (predict 0)": pd.Series(0.0, index=res.index), "Historical mean": hist_mean.loc[res.index]}
    cands.update({c: res[c] for c in res.columns if c != "actual"})
    for name, p in cands.items():
        dm = diebold_mariano(y - p, y) if name != "Random walk (predict 0)" else {"DM": np.nan, "p_value": np.nan}
        hit = np.mean(np.sign(p) == np.sign(y)) if name != "Random walk (predict 0)" else np.nan
        rows.append({"model": name, "RMSE (%)": 100 * np.sqrt(np.mean((y - p) ** 2)),
                     "R2_oos vs random walk (%)": 100 * r2_oos(y, p, 0 * y),
                     "R2_oos vs historical mean (%)": 100 * r2_oos(y, p, hist_mean.loc[res.index]),
                     "DM stat vs random walk": dm["DM"], "DM p-value": dm["p_value"], "sign accuracy": hit})
    return pd.DataFrame(rows)


def direction_scores(res: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of up/down calls, with an exact binomial test against the best trivial rule."""
    y = res["actual"].astype(int)
    n = len(y)
    base_rate = max(y.mean(), 1 - y.mean())  # 'always predict the majority class'
    rows = [{"model": "Always 'up'", "accuracy": y.mean(), "n": n, "p vs 50 %": np.nan, "p vs majority rule": np.nan}]
    for c in res.columns:
        if c == "actual":
            continue
        k = int(np.sum((res[c] > 0.5).astype(int) == y))
        rows.append({"model": c, "accuracy": k / n, "n": n,
                     "p vs 50 %": stats.binomtest(k, n, 0.5, alternative="greater").pvalue,
                     "p vs majority rule": stats.binomtest(k, n, base_rate, alternative="greater").pvalue})
    return pd.DataFrame(rows)


def price_scores(res: pd.DataFrame, last_price: pd.Series) -> pd.DataFrame:
    """Price-level forecasts: R² looks spectacular for everyone; Theil's U = RMSE / RMSE(naive) tells the truth."""
    y = res["actual"]
    naive = last_price.loc[res.index]
    rmse_naive = np.sqrt(np.mean((y - naive) ** 2))
    rows = []
    for name, p in {"Naive (tomorrow = today)": naive, **{c: res[c] for c in res.columns if c != "actual"}}.items():
        rmse = np.sqrt(np.mean((y - p) ** 2))
        rows.append({"model": name, "R2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
                     "RMSE (USD)": rmse, "Theil U (vs naive)": rmse / rmse_naive,
                     "corr with today's price": np.corrcoef(p, naive)[0, 1],
                     "corr of predicted vs actual change": np.corrcoef(p - naive, y - naive)[0, 1]})
    return pd.DataFrame(rows)
