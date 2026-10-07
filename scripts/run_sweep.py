"""Run tracking-error parameter sweep."""

import argparse
import copy
import logging
from pathlib import Path
import random
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from ei.backtest import run_backtest
from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares
from ei.metrics import compute_metrics
from ei.plots import plot_ir_vs_te
from ei.synthetic import make_synthetic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run tracking error parameter sweep and generate sweep.csv and ir_vs_te.png."""
    parser = argparse.ArgumentParser(description="Run Tracking-Error Parameter Sweep")
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

    logger.info("Running benchmark backtest as reference...")
    res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
    r_b = res_b["returns"]

    te_targets = cfg["sweep"]["te_max"]
    rf = float(cfg.get("risk_free_annual", 0.065))
    sweep_records = []

    for te in te_targets:
        logger.info("Evaluating enhanced strategy with te_max = %.4f...", te)
        cfg_i = copy.deepcopy(cfg)
        cfg_i["optimizer"]["te_max"] = float(te)
        res_i = run_backtest(prices, shares, sectors, cfg_i, strategy="enhanced_lw")

        m = compute_metrics(
            res_i["returns"],
            r_b,
            rf=rf,
            turnover=res_i.get("turnover"),
            ex_ante_te=res_i.get("ex_ante_te"),
        )
        sweep_records.append(
            {
                "te_max": float(te),
                "ann_return": m["ann_return"],
                "excess": m["excess_return"],
                "realized_TE": m["realized_TE"],
                "IR": m["IR"],
                "turnover": m["ann_turnover"],
            }
        )

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_df = pd.DataFrame(sweep_records)
    sweep_path = results_dir / "sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    logger.info("Saved sweep results to %s", sweep_path)

    plot_ir_vs_te(sweep_df, str(results_dir / "ir_vs_te.png"))
    logger.info("Sweep execution completed successfully.")


if __name__ == "__main__":
    main()
