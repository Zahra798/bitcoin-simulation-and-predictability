r"""Value-at-Risk forecasting and its statistical backtests.

The one-day VaR at level c (e.g. 99 %) is the loss that tomorrow's return
exceeds with probability only p = 1 − c:   P(r_{t+1} < −VaR_t) = p.

A VaR model is judged by its "hits" I_t = 1{r_t < −VaR_t}:

* Kupiec (1995) proportion-of-failures test, H0: P(I_t = 1) = p,
      LR_pof = −2 ln[(1−p)^{n−x} p^x] + 2 ln[(1−π̂)^{n−x} π̂^x] ~ χ²(1),  π̂ = x/n.
* Christoffersen (1998) independence test, H0: hits do not cluster,
      using the transition counts n_ij of the Markov chain I_{t−1} → I_t,
      LR_ind ~ χ²(1), and the conditional-coverage test LR_cc = LR_pof + LR_ind ~ χ²(2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .simulate import GARCHt, std_t_ppf


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits, p: float) -> dict:
    hits = np.asarray(hits, dtype=int)
    n, x = len(hits), int(hits.sum())
    pi = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = -2 * (ll0 - ll1)
    return {"violations": x, "expected": n * p, "rate": pi, "LR_pof": lr, "p_pof": float(stats.chi2.sf(lr, 1))}


def christoffersen(hits, p: float) -> dict:
    h = np.asarray(hits, dtype=int)
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    ll_0 = _xlogy(n00 + n10, 1 - p1) + _xlogy(n01 + n11, p1)
    lr_ind = -2 * (ll_0 - ll_ind)
    pof = kupiec_pof(h, p)
    lr_cc = pof["LR_pof"] + lr_ind
    return {**pof, "LR_ind": lr_ind, "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "LR_cc": lr_cc, "p_cc": float(stats.chi2.sf(lr_cc, 2)), "consecutive_hits": int(n11)}


def quantile_loss(r, q, p: float) -> float:
    """Tick loss ρ_p(r − q) of the quantile forecast q = −VaR: a strictly consistent
    score for quantiles, so lower is better and models can be ranked, not only passed or failed."""
    u = np.asarray(r, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(u * (p - (u < 0))))


def rolling_var(r: pd.Series, test_start, levels, window: int, refit_every: int) -> pd.DataFrame:
    """One-day-ahead VaR for every test day, each computed from data strictly before that day.

    * Normal (variance-covariance): μ̂ and σ̂ from the last `window` days, VaR = −(μ̂ + σ̂ Φ⁻¹(p)).
    * Historical simulation: the empirical p-quantile of the last `window` days.
    * GARCH(1,1)-t: re-estimated every `refit_every` days on all data so far,
      VaR = −(μ + σ_{t+1|t} · q_ν(p)) with the standardised-t quantile q_ν.
    """
    r = r.dropna()
    test_days = r.loc[test_start:].index
    out = pd.DataFrame(index=test_days)
    out["return"] = r.loc[test_days]
    model, since_fit = None, refit_every
    rows = []
    for day in test_days:
        past = r.loc[:day - pd.Timedelta(days=1)]
        win = past.iloc[-window:]
        if since_fit >= refit_every:
            model, since_fit = GARCHt().fit(past), 0
        since_fit += 1
        sig = np.sqrt(model.next_variance(past.to_numpy()))
        row = {}
        for c in levels:
            p = 1 - c
            row[f"Normal {c:.0%}"] = -(win.mean() + win.std(ddof=1) * stats.norm.ppf(p))
            row[f"Historical {c:.0%}"] = -np.quantile(win, p)
            row[f"GARCH-t {c:.0%}"] = -(model.mu_ + sig * std_t_ppf(p, model.nu_))
        row["garch_sigma"] = sig
        rows.append(row)
    return out.join(pd.DataFrame(rows, index=test_days))


def backtest_table(var: pd.DataFrame, levels) -> pd.DataFrame:
    rows = []
    for c in levels:
        for method in ("Normal", "Historical", "GARCH-t"):
            col = f"{method} {c:.0%}"
            hits = (var["return"] < -var[col]).astype(int)
            rows.append({"level": c, "method": method, "mean VaR": var[col].mean(),
                         "quantile loss x1000": 1000 * quantile_loss(var["return"], -var[col], 1 - c),
                         **christoffersen(hits, 1 - c)})
    return pd.DataFrame(rows)
