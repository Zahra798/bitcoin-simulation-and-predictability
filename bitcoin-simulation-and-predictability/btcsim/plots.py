"""Figures for the README."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
SIM_COLORS = {"GBM (normal, i.i.d.)": ORANGE, "i.i.d. bootstrap": YELLOW, "block bootstrap": AQUA, "GARCH(1,1)-t": BLUE}


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })


def _save(fig, path):
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def price_and_returns(df, test_start, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    a1.plot(df.index, df["price_usd"], color=BLUE, linewidth=1.2)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    a1.set_ylabel("Price (log scale)")
    a1.set_title("Bitcoin, 2014 – 2026: price and daily log returns")
    r = df["log_return"]
    a2.plot(r.index, 100 * r, color=INK2, linewidth=0.5)
    a2.set_ylabel("Daily log return (%)")
    for a in (a1, a2):
        a.axvspan(test_start, df.index.max(), color=GRID, alpha=0.6, linewidth=0)
    a1.text(test_start + pd.Timedelta(days=40), df["price_usd"].min() * 1.5, "test period", color=INK2)
    worst = r.idxmin()
    a2.annotate(f"{worst:%d %b %Y}: {100 * (np.exp(r.min()) - 1):.0f} %", (worst, 100 * r.min()), xytext=(25, 5),
                textcoords="offset points", fontsize=8, color=INK2)
    _save(fig, path)


def return_distribution(r, path):
    r = r.dropna().to_numpy()
    z = (r - r.mean()) / r.std()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(-14, 8, 111)
    dens, edges = np.histogram(z, bins=bins, density=True)
    mid = (edges[1:] + edges[:-1]) / 2
    a1.scatter(mid[dens > 0], dens[dens > 0], s=10, color=INK, label="Bitcoin (standardised)", zorder=5)
    xs = np.linspace(-14, 8, 500)
    a1.plot(xs, stats.norm.pdf(xs), color=ORANGE, label="Normal distribution")
    nu, loc, sc = stats.t.fit(z)
    a1.plot(xs, stats.t.pdf(xs, nu, loc, sc), color=BLUE, label=f"Student-t, ν = {nu:.1f}")
    a1.set_yscale("log")
    a1.set_ylim(1e-5, 1)
    a1.set_xlabel("Daily return in standard deviations")
    a1.set_ylabel("Density (log scale)")
    a1.legend(fontsize=8, loc="lower center")
    a1.set_title("Heavy tails")
    q = np.sort(z)
    th = stats.norm.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q))
    a2.plot(th, q, ".", color=INK, markersize=2)
    a2.plot([-4, 4], [-4, 4], color=ORANGE, linewidth=1.2)
    a2.set_xlabel("Normal quantiles")
    a2.set_ylabel("Bitcoin quantiles")
    a2.set_title("Normal Q–Q plot: the tails bend away")
    _save(fig, path)


def acf_plot(r, path, nlags=60):
    from .stats import acf
    r = r.dropna().to_numpy()
    band = 1.96 / np.sqrt(len(r))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for a, x, title in [(a1, r, "Returns r_t: almost no memory"), (a2, np.abs(r), "Absolute returns |r_t|: long memory")]:
        lags = np.arange(1, nlags + 1)
        a.vlines(lags, 0, acf(x, nlags), color=BLUE, linewidth=1.6)
        a.axhspan(-band, band, color=GRID, alpha=0.9, linewidth=0)
        a.axhline(0, color=INK2, linewidth=0.8)
        a.set_xlabel("Lag (days)")
        a.set_title(title)
    a1.set_ylabel("Autocorrelation")
    a2.text(nlags, band + 0.01, "95 % band under 'no correlation'", ha="right", fontsize=8, color=INK2)
    _save(fig, path)


def simulated_acf(acf_curves, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    lags = np.arange(1, len(acf_curves) + 1)
    ax.plot(lags, acf_curves["Real data (2014–2020)"], color=INK, linewidth=2.6, label="Real data (2014–2020)")
    for name, c in SIM_COLORS.items():
        ax.plot(lags, acf_curves[name], color=c, label=f"{name} (median of 200 simulations)")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation of |r_t|")
    ax.legend(fontsize=8)
    ax.set_title("Which simulation reproduces volatility clustering?")
    _save(fig, path)


def fan_chart(bands, actual, info, history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["GBM (normal, i.i.d.)", "GARCH(1,1)-t"]):
        b = bands[name]
        c = SIM_COLORS[name]
        ax.fill_between(b.index, b["q05"], b["q95"], color=c, alpha=0.15, linewidth=0, label="90 % of simulated paths")
        ax.fill_between(b.index, b["q25"], b["q75"], color=c, alpha=0.3, linewidth=0, label="50 % of simulated paths")
        ax.plot(b.index, b["q50"], color=c, linewidth=1.4, label="Median path")
        ax.plot(history.index, history, color=INK2, linewidth=1)
        ax.plot(actual.index, actual, color=INK, linewidth=1.6, label="What actually happened")
        ax.axvline(pd.Timestamp(info["start date"]), color=MUTED, linewidth=1, linestyle=":")
        ax.set_yscale("log")
        ax.set_yticks([40_000, 60_000, 100_000, 150_000, 250_000, 400_000])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.set_title(name)
    axes[0].set_ylabel("Price (log scale)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"20,000 simulated years from {info['start date']}, compared with reality", x=0.06, ha="left",
                 fontweight="bold", y=1.02)
    _save(fig, path)


def convergence(summ, exact, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.errorbar(summ["N"], summ["mean"], yerr=1.96 * summ["std"], fmt="o", color=BLUE, capsize=3, markersize=4)
    a1.axhline(exact, color=INK, linestyle="--", linewidth=1)
    a1.text(summ["N"].iloc[-1], exact * 1.06, f"exact = {exact:.4f}", ha="right", fontsize=8)
    a1.set_xscale("log")
    a1.set_xlabel("Number of simulated paths N")
    a1.set_ylabel("Estimate of P(price halves in a year)")
    a1.set_title("Monte Carlo estimate ± 1.96 sd")
    a2.loglog(summ["N"], summ["RMSE vs exact"], "o", color=BLUE, label="Observed error (50 repetitions)")
    a2.loglog(summ["N"], summ["theoretical std"], color=ORANGE, label="Theory: √(p(1−p)/N)")
    a2.set_xlabel("Number of simulated paths N")
    a2.set_ylabel("Root mean squared error")
    a2.legend(fontsize=8)
    a2.set_title("Error falls like 1/√N")
    _save(fig, path)


def var_plot(var, level, path):
    v = var.copy()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(v.index, 100 * v["return"], color=MUTED, linewidth=0.6, label="Daily return")
    for method, c in [("Normal", ORANGE), ("GARCH-t", BLUE)]:
        col = f"{method} {level:.0%}"
        ax.plot(v.index, -100 * v[col], color=c, linewidth=1.2, label=f"−VaR {level:.0%}, {method}")
        hit = v["return"] < -v[col]
        ax.scatter(v.index[hit], 100 * v["return"][hit], color=c, s=16, zorder=5)
    ax.set_ylabel("Daily log return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.0), framealpha=0.9, frameon=True,
              edgecolor="white")
    ax.set_ylim(None, 24)
    ax.set_title(f"{level:.0%} Value-at-Risk, 2021 – 2026: dots are days when the loss exceeded VaR")
    _save(fig, path)


def price_illusion(pr, price, path):
    p = pr.copy()
    p.index = p.index + pd.Timedelta(days=1)  # plot each forecast at the day it is about
    name = "Neural network on min-max scaled prices"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(p.index, p["actual"], color=INK, linewidth=1, label="Actual price")
    a1.plot(p.index, p["Ridge on past prices"], color=BLUE, linewidth=1, label="Ridge forecast")
    a1.plot(p.index, p[name], color=ORANGE, linewidth=1, label="Neural-network forecast")
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a1.legend(fontsize=8)
    a1.set_title("Looks impressive: R² > 0.98")
    w = p.loc["2024-02-01":"2024-03-15"]
    yesterday = price.reindex(w.index - pd.Timedelta(days=1)).to_numpy()
    a2.plot(w.index, w["actual"], color=INK, marker="o", markersize=3, linewidth=1.2, label="Actual price")
    a2.plot(w.index, w["Ridge on past prices"], color=BLUE, marker="o", markersize=3, linewidth=1.2,
            label="Ridge forecast")
    a2.plot(w.index, yesterday, color=MUTED, linestyle=":", linewidth=1.4, label="Yesterday's price")
    a2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    a2.legend(fontsize=8)
    a2.set_title("Zoom in: the forecast is yesterday's price")
    _save(fig, path)


def predictability_summary(ret, controls, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    r = ret[ret["model"] != "Random walk (predict 0)"]
    vals = r["R2_oos vs random walk (%)"].to_numpy()
    colors = [BLUE if v >= 0 else ORANGE for v in vals]
    a1.barh(r["model"][::-1], vals[::-1], color=colors[::-1], height=0.55)
    a1.axvline(0, color=INK, linewidth=1)
    for y, v in enumerate(vals[::-1]):
        a1.text(v + (1.0 if v >= 0 else -1.0), y, f"{v:+.2f} %", va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=INK2)
    a1.set_xlabel("Out-of-sample R² vs. 'predict zero' (%)")
    a1.grid(axis="y", visible=False)
    lo = min(vals.min() * 1.35, -5)
    a1.set_xlim(lo, 6)
    a1.set_title("Next-day return forecasts")
    c = controls[controls["model"] == "Logistic regression"].reset_index(drop=True)
    labels = ["Random walk\n(negative control)", "i.i.d. uniform prices\n(positive control)", "Real Bitcoin\n2021–2026"]
    acc = c["accuracy"].to_numpy()
    n = c["n"].to_numpy()
    ci = 1.96 * np.sqrt(0.25 / n)
    a2.axhspan(100 * (0.5 - ci[0]), 100 * (0.5 + ci[0]), color=GRID, alpha=0.9, linewidth=0, zorder=1)
    a2.bar(labels, 100 * acc, color=[MUTED, AQUA, BLUE], width=0.55, zorder=2)
    a2.axhline(50, color=INK, linewidth=1, linestyle="--", zorder=3)
    for x, v in enumerate(acc):
        top = max(100 * v, 100 * (0.5 + ci[0]))
        a2.text(x, top + 0.8, f"{100 * v:.1f} %", ha="center", fontsize=9, zorder=4)
    a2.set_ylim(40, 80)
    a2.set_ylabel("Accuracy of up/down calls (%)")
    a2.grid(axis="x", visible=False)
    a2.set_title("Same pipeline, three datasets (grey band: chance ± 95 %)")
    _save(fig, path)
