"""Run portfolio optimization backtests."""

import argparse
import logging
from pathlib import Path
import random
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from ei.backtest import run_all
from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares
from ei.metrics import metrics_table
from ei.plots import (
    plot_active_weights_last,
    plot_cumulative_returns,
    plot_drawdown,
    plot_rolling_te,
    plot_turnover,
)
from ei.synthetic import make_synthetic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_cost_sensitivity(
    prices: pd.DataFrame,
    shares: pd.DataFrame,
    sectors: pd.DataFrame,
    cfg: dict,
    tc_levels: list[float] | None = None,
    strategy: str = "enhanced_lw",
) -> pd.DataFrame:
    """Evaluate enhanced strategy across varying transaction cost levels tc_bps."""
    import copy
    from ei.backtest import run_backtest
    from ei.metrics import compute_metrics

    if tc_levels is None:
        tc_levels = [0.0, 5.0, 10.0, 20.0, 50.0]

    logger.info("Computing cost sensitivity across tc_bps = %s...", tc_levels)
    records = []
    rf = float(cfg.get("risk_free_annual", 0.065))

    for tc in tc_levels:
        cfg_tc = copy.deepcopy(cfg)
        cfg_tc["costs"]["tc_bps"] = float(tc)
        res_e = run_backtest(prices, shares, sectors, cfg_tc, strategy=strategy)
        res_b = run_backtest(prices, shares, sectors, cfg_tc, strategy="benchmark")
        r_e = res_e["returns"]
        r_b = res_b["returns"]
        t_over = res_e.get("turnover")
        ex_te = res_e.get("ex_ante_te")
        n = len(r_e.dropna())
        years = n / 252.0
        ann_to = (
            float(t_over.sum()) / max(years, 1.0 / 252.0)
            if t_over is not None and len(t_over) > 0
            else 0.0
        )
        cost_drag = (tc / 10000.0) * ann_to

        m = compute_metrics(
            r_e, r_b, rf=rf, turnover=t_over, ex_ante_te=ex_te, cost_drag_annual=cost_drag
        )
        records.append({
            "tc_bps": tc,
            "IR": m["IR"],
            "excess_return": m["excess_return"],
            "ann_turnover": m["ann_turnover"],
            "cost_drag": m["cost_drag"],
            "gross_excess_return": m["gross_excess_return"],
            "ann_return": m["ann_return"],
            "realized_TE": m["realized_TE"],
        })

    return pd.DataFrame(records)


def main() -> None:
    """Run full backtest suite and save metrics, returns, weights, and plots."""
    parser = argparse.ArgumentParser(description="Run Enhanced Indexing Backtests")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic dataset")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    if args.synthetic:
        logger.info("Using synthetic dataset with seed %d...", seed)
        prices, shares, sectors = make_synthetic(n_assets=40, n_days=2500, seed=seed)
    else:
        logger.info("Loading real market data...")
        prices = load_prices(cfg)
        tickers = list(prices.columns)
        shares = load_shares(tickers, cfg)
        sectors = load_sectors(tickers, cfg)

    logger.info("Running all strategies (benchmark, enhanced_lw, enhanced_sample, mv)...")
    results = run_all(prices, shares, sectors, cfg)

    # Save outputs
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = metrics_table(
        results,
        rf=float(cfg.get("risk_free_annual", 0.065)),
        tc_bps=float(cfg.get("costs", {}).get("tc_bps", 10.0)),
    )
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info("Saved metrics table to %s", metrics_path)

    # Solver status tracking and audit
    solver_rows = []
    for strat_name in ["enhanced_lw", "enhanced_sample", "mv"]:
        if "solver_status" in results[strat_name]:
            sdf = results[strat_name]["solver_status"].copy()
            for dt, row in sdf.iterrows():
                solver_rows.append({
                    "strategy": strat_name,
                    "date": dt,
                    "status": row["status"],
                    "solver": row["solver"],
                })
    if solver_rows:
        solver_df = pd.DataFrame(solver_rows)
        solver_status_path = results_dir / "solver_status.csv"
        solver_df.to_csv(solver_status_path, index=False)
        non_optimal = solver_df[solver_df["status"] != "optimal"]
        if len(non_optimal) > 0:
            logger.warning(
                "Found %d non-optimal / fallback rebalances across strategies: %s",
                len(non_optimal),
                dict(non_optimal["status"].value_counts()),
            )
        else:
            logger.info("All rebalances solved to optimal across all backtested strategies.")

    # Cost sensitivity analysis across tc_bps in [0, 5, 10, 20, 50]
    cost_sens_df = compute_cost_sensitivity(prices, shares, sectors, cfg)
    cost_sens_path = results_dir / "cost_sensitivity.csv"
    cost_sens_df.to_csv(cost_sens_path, index=False)
    logger.info("Saved cost sensitivity to %s", cost_sens_path)

    returns_df = pd.DataFrame({name: res["returns"] for name, res in results.items()})
    returns_path = results_dir / "returns.csv"
    returns_df.to_csv(returns_path)
    logger.info("Saved returns series to %s", returns_path)

    weights_path = results_dir / "weights_enhanced.csv"
    results["enhanced_lw"]["weights"].to_csv(weights_path)
    logger.info("Saved enhanced weights to %s", weights_path)

    # Generate 5 backtest plots
    plot_cumulative_returns(results, str(results_dir / "cumulative_returns.png"))
    plot_rolling_te(
        results["enhanced_lw"]["returns"],
        results["benchmark"]["returns"],
        te_max=float(cfg["optimizer"]["te_max"]),
        window=252,
        out_path=str(results_dir / "rolling_te.png"),
    )
    plot_active_weights_last(
        results["enhanced_lw"]["weights"].iloc[-1],
        results["benchmark"]["weights"].iloc[-1],
        str(results_dir / "active_weights_last.png"),
    )
    plot_drawdown(results, str(results_dir / "drawdown.png"))
    plot_turnover(results, str(results_dir / "turnover.png"))

    logger.info("Backtest execution completed successfully.")


if __name__ == "__main__":
    main()
