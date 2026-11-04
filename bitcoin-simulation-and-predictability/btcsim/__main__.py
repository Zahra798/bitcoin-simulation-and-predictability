"""Run every study:  python -m btcsim  [--skip-prediction]"""

from __future__ import annotations

import argparse
import json
import warnings

import pandas as pd

from . import plots
from .config import CFG
from .data import load
from .experiments import fan_study, log, mc_convergence, predictability, simulation_study, stylized_facts, var_study


def save(df: pd.DataFrame, name: str, index=False):
    df.to_csv(CFG.results_dir / f"{name}.csv", index=index)
    (CFG.results_dir / f"{name}.md").write_text(df.to_markdown(index=index, floatfmt=".4g"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prediction", action="store_true", help="skip the walk-forward models (saves ~8 min)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    CFG.results_dir.mkdir(exist_ok=True)
    CFG.figures_dir.mkdir(exist_ok=True)
    plots.setup()
    F = lambda n: CFG.figures_dir / n  # noqa: E731

    df = load(CFG.data_path, CFG.start)
    r = df["log_return"]
    summary, tests = stylized_facts(df, CFG)
    save(summary.reset_index(names="statistic"), "stylized_facts")
    save(tests, "dependence_tests")
    plots.price_and_returns(df, CFG.test_start, F("01_price_and_returns.png"))
    plots.return_distribution(r, F("02_return_distribution.png"))
    plots.acf_plot(r, F("03_autocorrelation.png"))
    log("stylised facts done")

    table, acf_curves, garch, gbm = simulation_study(df, CFG)
    save(table.reset_index(names="model"), "simulation_vs_reality")
    plots.simulated_acf(acf_curves, F("04_simulated_volatility_clustering.png"))
    fan, bands, actual, info = fan_study(df, CFG)
    save(fan, "one_year_simulation")
    history = df["price_usd"].loc[CFG.fan_origin - pd.Timedelta(days=200):CFG.fan_origin]
    plots.fan_chart(bands, actual, info, history, F("05_one_year_fan_chart.png"))
    conv, exact = mc_convergence(df, CFG)
    save(conv, "monte_carlo_convergence")
    plots.convergence(conv, exact, F("06_monte_carlo_convergence.png"))
    log("simulation done")

    var, bt = var_study(df, CFG)
    save(bt, "var_backtest")
    plots.var_plot(var, 0.99, F("07_value_at_risk.png"))
    log("value at risk done")

    summary_json = {"garch_2014_2020": garch, "gbm_2014_2020": gbm, "fan_chart": info, "mc_exact_probability": exact}
    if not args.skip_prediction:
        pred = predictability(df, CFG)
        save(pred["returns"], "return_forecasts")
        save(pred["direction"], "direction_forecasts")
        save(pred["price"], "price_forecasts")
        save(pred["controls"], "controls")
        plots.price_illusion(pred["price_pred"], df["price_usd"], F("08_price_forecast_illusion.png"))
        plots.predictability_summary(pred["returns"], pred["controls"], F("09_predictability.png"))
    (CFG.results_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, default=float))
    log("all done")


if __name__ == "__main__":
    main()
