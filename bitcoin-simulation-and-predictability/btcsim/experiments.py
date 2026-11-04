"""The four studies. Each returns pandas objects; __main__ saves and plots them."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from . import predict as P
from . import stats as S
from .config import Config
from .data import random_walk_control, uniform_price_control
from .risk import backtest_table, rolling_var
from .simulate import GBM, GARCHt, IIDBootstrap, StationaryBootstrap


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# 1. Stylised facts
# ===========================================================================
def stylized_facts(df: pd.DataFrame, cfg: Config):
    r = df["log_return"].dropna()
    periods = {"2014–2020 (fit)": r.loc[:cfg.train_end], "2021–2026 (test)": r.loc[cfg.test_start:], "2014–2026": r}
    summary = pd.DataFrame({k: S.summary(v) for k, v in periods.items()})
    tests = []
    for name, x in [("returns r_t", r), ("absolute returns |r_t|", r.abs()), ("squared returns r_t²", r**2)]:
        for lags in (10, 30):
            lb = S.ljung_box(x, lags)
            tests.append({"series": name, "lags": lags, "Q": lb["Q"], "p_value": lb["p_value"]})
    jb = S.jarque_bera(r)
    tests.append({"series": "normality (Jarque–Bera)", "lags": np.nan, "Q": jb["JB"], "p_value": jb["p_value"]})
    return summary, pd.DataFrame(tests)


# ===========================================================================
# 2. Monte Carlo simulation
# ===========================================================================
def make_models(cfg):
    return [GBM(), IIDBootstrap(), StationaryBootstrap(cfg.block_length), GARCHt()]


def simulation_study(df: pd.DataFrame, cfg: Config, n_series: int = 200):
    """Do simulated return series look like the real ones? Each model is fitted on
    2014–2020, then n_series synthetic histories of the same length are generated."""
    rng = np.random.default_rng(cfg.seed)
    r = df["log_return"].loc[:cfg.train_end].dropna().to_numpy()
    n = len(r)

    def facts(x):
        x = np.atleast_2d(x)
        a_abs = np.array([S.acf(np.abs(row), 30) for row in x])
        return {
            "excess kurtosis": np.array([S.jarque_bera(row)["excess_kurtosis"] for row in x]),
            "ACF of r, lag 1": np.array([S.acf(row, 1)[0] for row in x]),
            "ACF of |r|, lag 1": a_abs[:, 0],
            "ACF of |r|, mean lags 1–30": a_abs.mean(axis=1),
            "1 % quantile of r": np.quantile(x, 0.01, axis=1),
        }, a_abs

    real, real_acf = facts(r)
    rows = {"Real data (2014–2020)": {k: float(v[0]) for k, v in real.items()}}
    acf_curves = {"Real data (2014–2020)": real_acf[0]}
    fitted = {}
    for m in make_models(cfg):
        m.fit(r)
        fitted[m.name] = m
        if isinstance(m, GARCHt):
            sims = m.simulate(n_series, n, rng, start_from_last=False)
        else:
            sims = m.simulate(n_series, n, rng)
        f, a = facts(sims)
        rows[m.name] = {k: f"{np.median(v):.3f} [{np.quantile(v, 0.05):.3f}, {np.quantile(v, 0.95):.3f}]"
                        for k, v in f.items()}
        acf_curves[m.name] = np.median(a, axis=0)
        log(f"simulated {m.name}")
    table = pd.DataFrame(rows).T
    for c in table.columns:
        table.loc["Real data (2014–2020)", c] = f"{float(table.loc['Real data (2014–2020)', c]):.3f}"
    g = fitted["GARCH(1,1)-t"]
    garch = {"mu (daily)": g.mu_, "omega": g.omega_, "alpha": g.alpha_, "beta": g.beta_, "nu (t d.o.f.)": g.nu_,
             "persistence alpha+beta": g.persistence, "volatility half-life (days)": g.half_life_days,
             "log-likelihood": g.loglik_, "converged": g.converged_}
    gbm = fitted["GBM (normal, i.i.d.)"]
    garch_ll_gain = g.loglik_ - np.sum(stats.norm.logpdf(r, r.mean(), r.std(ddof=1)))
    return table, pd.DataFrame(acf_curves), garch, {"GBM mu": gbm.mu_, "GBM sigma": gbm.sigma_,
                                                    "loglik gain GARCH-t vs normal": float(garch_ll_gain)}


def fan_study(df: pd.DataFrame, cfg: Config):
    """Fit on everything up to `fan_origin`, simulate one year of prices, compare with what happened."""
    rng = np.random.default_rng(cfg.seed + 1)
    r = df["log_return"].loc[:cfg.fan_origin].dropna().to_numpy()
    p0 = float(df["price_usd"].loc[cfg.fan_origin])
    actual = df["price_usd"].loc[cfg.fan_origin:].iloc[1:cfg.fan_horizon_days + 1]
    h = len(actual)
    bands, rows = {}, []
    actual_dd = S.max_drawdown(np.log(actual.to_numpy() / p0)[None, :])[0]
    for m in make_models(cfg):
        m.fit(r)
        sims = m.simulate(cfg.n_paths, h, rng)
        logp = np.cumsum(sims, axis=1)
        q = np.quantile(logp, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        bands[m.name] = pd.DataFrame(p0 * np.exp(q.T), index=actual.index, columns=["q05", "q25", "q50", "q75", "q95"])
        final = p0 * np.exp(logp[:, -1])
        dd = S.max_drawdown(logp)
        inside = np.mean((actual.to_numpy() >= bands[m.name]["q05"]) & (actual.to_numpy() <= bands[m.name]["q95"]))
        rows.append({"model": m.name, "median final price": np.median(final),
                     "90 % range of final price": f"{np.quantile(final, 0.05):,.0f} – {np.quantile(final, 0.95):,.0f}",
                     "P(final < start)": np.mean(final < p0),
                     "percentile of actual final price": stats.percentileofscore(final, actual.iloc[-1]) / 100,
                     "days inside 90 % band": inside,
                     "P(drawdown ≥ actual)": np.mean(dd >= actual_dd)})
    info = {"start date": str(cfg.fan_origin.date()), "start price": p0, "end date": str(actual.index[-1].date()),
            "actual final price": float(actual.iloc[-1]), "actual max drawdown": float(actual_dd)}
    return pd.DataFrame(rows), bands, actual, info


def mc_convergence(df: pd.DataFrame, cfg: Config):
    """Monte Carlo error shrinks like 1/√N. Under GBM the target probability is known exactly:
        P(S_T < S_0/2) = Φ( (ln ½ − (μ − σ²/2)T) / (σ√T) ),
    so the true error of every estimate can be measured."""
    r = df["log_return"].loc[:cfg.train_end].dropna().to_numpy()
    g = GBM().fit(r)
    T = 365
    exact = stats.norm.cdf((np.log(0.5) - (g.mu_ - g.sigma_**2 / 2) * T) / (g.sigma_ * np.sqrt(T)))
    rng = np.random.default_rng(cfg.seed + 2)
    rows = []
    for n in [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000]:
        for rep in range(50):
            # the sum of T i.i.d. normal daily returns is itself normal: draw it directly
            logret = rng.normal((g.mu_ - g.sigma_**2 / 2) * T, g.sigma_ * np.sqrt(T), n)
            est = np.mean(logret < np.log(0.5))
            rows.append({"N": n, "rep": rep, "estimate": est})
    res = pd.DataFrame(rows)
    summ = res.groupby("N")["estimate"].agg(["mean", "std"]).reset_index()
    summ["theoretical std"] = np.sqrt(exact * (1 - exact) / summ["N"])
    summ["RMSE vs exact"] = res.groupby("N")["estimate"].apply(lambda e: np.sqrt(np.mean((e - exact) ** 2))).to_numpy()
    return summ, float(exact)


# ===========================================================================
# 3. Value at Risk
# ===========================================================================
def var_study(df: pd.DataFrame, cfg: Config):
    var = rolling_var(df["log_return"], cfg.test_start, cfg.var_levels, cfg.var_window, cfg.refit_every)
    return var, backtest_table(var, cfg.var_levels)


# ===========================================================================
# 4. Predictability
# ===========================================================================
def predictability(df: pd.DataFrame, cfg: Config):
    out = {}
    X = P.return_features(df)
    y = df["log_return"].shift(-1)
    reg = P.walk_forward(X, y, P.regressors(cfg.seed), cfg.test_start, cfg.refit_every, cfg.min_train_days)
    hist_mean = df["log_return"].expanding().mean()
    out["returns"] = P.return_scores(reg, hist_mean)
    out["returns_pred"] = reg
    log("return forecasts done")

    up = (y > 0).astype(float).where(y.notna())
    cls = P.walk_forward(X, up, P.classifiers(cfg.seed), cfg.test_start, cfg.refit_every, cfg.min_train_days, proba=True)
    out["direction"] = P.direction_scores(cls)
    log("direction forecasts done")

    Xp = P.price_features(df)
    yp = df["price_usd"].shift(-1).loc[Xp.index]
    pr = P.walk_forward(Xp, yp, P.price_models(cfg.seed), cfg.test_start, cfg.refit_every, cfg.min_train_days)
    out["price"] = P.price_scores(pr, df["price_usd"])
    out["price_pred"] = pr
    log("price forecasts done")

    # controls: same pipeline, data with known answers
    n_test = len(cls)
    controls = []
    sigma = float(df["log_return"].std())
    for name, data, truth in [
        ("Negative control: random walk (unpredictable by construction)", random_walk_control(len(df), sigma, cfg.seed), "≈ 50 %"),
        ("Positive control: i.i.d. uniform prices (original course data)", uniform_price_control(len(df), cfg.seed), "> 50 %"),
    ]:
        Xc, yc = P.return_features(data), data["log_return"].shift(-1)
        upc = (yc > 0).astype(float).where(yc.notna())
        start = data.index[len(data) - n_test]
        res = P.walk_forward(Xc, upc, P.classifiers(cfg.seed), start, cfg.refit_every, cfg.min_train_days, proba=True)
        sc = P.direction_scores(res)
        sc.insert(0, "dataset", name)
        sc.insert(1, "expected", truth)
        controls.append(sc)
    real = out["direction"].copy()
    real.insert(0, "dataset", "Real Bitcoin, 2021–2026")
    real.insert(1, "expected", "?")
    out["controls"] = pd.concat(controls + [real], ignore_index=True)
    log("controls done")
    return out
