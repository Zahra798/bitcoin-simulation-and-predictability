r"""Loading the Coin Metrics daily data and computing log returns.

Daily log return:  r_t = ln(P_t / P_{t-1}).  Log returns add up over time
(the k-day return is the sum of k daily returns), which is what makes them
the natural object for both simulation and statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load(path, start=None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    if len(full) != len(df):
        raise ValueError("calendar gaps in the data")
    df["log_return"] = np.log(df["price_usd"]).diff()
    if start is not None:
        df = df.loc[start:]
    return df


def _frame(price, rng, start="2015-03-14"):
    n = len(price)
    idx = pd.date_range(start, periods=n, freq="D")
    df = pd.DataFrame({"price_usd": price, "volume_usd": rng.uniform(1, 100, n),
                       "active_addresses": rng.uniform(1, 100, n), "transactions": rng.uniform(1, 100, n),
                       "hash_rate": rng.uniform(1, 100, n)}, index=idx)
    df["log_return"] = np.log(df["price_usd"]).diff()
    return df


def uniform_price_control(n: int, seed: int) -> pd.DataFrame:
    """The data of the original course project: every day's price drawn independently
    from U(1, 100). The *level* is pure noise, but the *changes* are not: for i.i.d.
    X_t, corr(X_{t+1} − X_t, X_t − X_{t−1}) = −Var(X)/(2 Var(X)) = −1/2,
    so a high price today predicts a fall tomorrow. A sound pipeline must detect this."""
    rng = np.random.default_rng(seed)
    return _frame(rng.uniform(1, 100, n), rng)


def random_walk_control(n: int, sigma: float, seed: int) -> pd.DataFrame:
    """A price whose daily log returns are i.i.d. N(0, σ²): tomorrow's move is
    unpredictable by construction, so a sound pipeline must score at chance level."""
    rng = np.random.default_rng(seed)
    return _frame(100 * np.exp(np.cumsum(rng.normal(0, sigma, n))), rng)
