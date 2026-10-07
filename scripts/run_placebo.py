"""Placebo test: run enhanced strategy with random alpha (seeds 1-20).

Replaces the true alpha vector with random z-scores (same IC scaling)
to verify the optimizer does not generate alpha from noise.
Expected result: IR centred near 0, slightly negative from transaction costs.

Outputs: results/placebo.csv
"""

import copy
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from ei.backtest import run_backtest
from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares
from ei.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _make_random_alpha_backtest(prices, shares, sectors, cfg, seed):
    """Run enhanced backtest with random-noise alpha (same IC scaling) for one seed."""
    import ei.signals as _signals
    import ei.backtest as _bt

    rng = np.random.default_rng(seed)
    original_compute_alpha = _signals.compute_alpha

    def random_alpha(prices_upto_t, Sigma_ann, cfg_inner):
        n = Sigma_ann.shape[0]
        ic = float(cfg_inner["signals"]["ic"])
        sigma_i = np.sqrt(np.diag(Sigma_ann))
        z = rng.standard_normal(n)
        z = (z - z.mean()) / (z.std() + 1e-12)
        alpha_values = ic * sigma_i * z
        return pd.Series(alpha_values, index=prices_upto_t.columns, name="alpha")

    _signals.compute_alpha = random_alpha
    _bt.compute_alpha = random_alpha

    try:
        res_e = run_backtest(prices, shares, sectors, cfg, strategy="enhanced_lw")
        res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
    finally:
        _signals.compute_alpha = original_compute_alpha
        _bt.compute_alpha = original_compute_alpha

    m = compute_metrics(
        res_e["returns"],
        res_b["returns"],
        rf=float(cfg.get("risk_free_annual", 0.065)),
        turnover=res_e.get("turnover"),
        ex_ante_te=res_e.get("ex_ante_te"),
    )
    return m


def main():
    """Run placebo test over seeds 1..20 and save results/placebo.csv."""
    cfg = load_config("config.yaml")
    np.random.seed(42)

    logger.info("Loading real market data...")
    prices = load_prices(cfg)
    tickers = list(prices.columns)
    shares = load_shares(tickers, cfg)
    sectors = load_sectors(tickers, cfg)

    records = []
    n_seeds = 20
    for seed in range(1, n_seeds + 1):
        logger.info("Placebo seed %d / %d ...", seed, n_seeds)
        m = _make_random_alpha_backtest(prices, shares, sectors, cfg, seed)
        records.append({
            "seed": seed,
            "IR": m["IR"],
            "excess_return": m["excess_return"],
            "realized_TE": m["realized_TE"],
            "ann_turnover": m["ann_turnover"],
        })

    df = pd.DataFrame(records)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "placebo.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved placebo results to %s", out_path)

    mean_ir = df["IR"].mean()
    std_ir = df["IR"].std()
    print(df.to_string(index=False))
    print(f"\nMean IR = {mean_ir:.4f}   Std IR = {std_ir:.4f}")


if __name__ == "__main__":
    main()
