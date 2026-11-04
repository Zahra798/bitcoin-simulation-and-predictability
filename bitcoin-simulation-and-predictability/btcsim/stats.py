r"""Descriptive statistics and classical tests, implemented from their formulas.

Stylised facts of financial returns (Cont, 2001) checked here:
  1. heavy tails            -> excess kurtosis, Jarque-Bera test
  2. almost no linear memory -> Ljung-Box on r_t
  3. volatility clustering   -> Ljung-Box on |r_t| (large moves follow large moves)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def acf(x, nlags: int) -> np.ndarray:
    """Sample autocorrelation ρ̂_k = Σ (x_t − x̄)(x_{t+k} − x̄) / Σ (x_t − x̄)²,  k = 1..nlags."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)] - np.nanmean(x)
    d = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:-k]) / d for k in range(1, nlags + 1)])


def ljung_box(x, lags: int) -> dict:
    r"""Q = n(n+2) Σ_{k=1}^{m} ρ̂_k² / (n−k)  ~  χ²(m)  under 'no autocorrelation'."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    rho = acf(x, lags)
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, lags + 1)))
    return {"Q": float(q), "p_value": float(stats.chi2.sf(q, lags)), "lags": lags}


def jarque_bera(x) -> dict:
    r"""JB = n/6 · [S² + (K−3)²/4]  ~  χ²(2)  under normality (S = skewness, K = kurtosis)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    z = (x - x.mean()) / x.std()
    s, k = np.mean(z**3), np.mean(z**4)
    jb = n / 6 * (s**2 + (k - 3) ** 2 / 4)
    return {"JB": float(jb), "p_value": float(stats.chi2.sf(jb, 2)), "skewness": float(s), "excess_kurtosis": float(k - 3)}


def summary(r) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    jb = jarque_bera(r)
    return {
        "days": len(r),
        "mean daily log return": r.mean(),
        "daily volatility": r.std(ddof=1),
        "annualised volatility": r.std(ddof=1) * np.sqrt(365),
        "skewness": jb["skewness"],
        "excess kurtosis": jb["excess_kurtosis"],
        "worst day": r.min(),
        "best day": r.max(),
        "days beyond 4 sigma (observed)": int(np.sum(np.abs(r - r.mean()) > 4 * r.std())),
        "days beyond 4 sigma (if normal)": len(r) * 2 * stats.norm.sf(4),
    }


def max_drawdown(log_path: np.ndarray) -> np.ndarray:
    """Largest peak-to-trough fall of each path; paths are cumulative log returns (rows)."""
    log_path = np.atleast_2d(log_path)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((len(log_path), 1)), log_path], axis=1), axis=1)[:, 1:]
    return 1 - np.exp(np.min(log_path - peak, axis=1))
