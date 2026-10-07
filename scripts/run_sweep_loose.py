"""Run a second TE sweep with loose constraints (max_active=0.10, sector_dev=0.15).

te_max values: [0.005, 0.01, 0.02, 0.03, 0.04, 0.06]
Saves: results/sweep_loose.csv
Plots: results/ir_vs_te_loose.png (IR and excess return vs te_max)
"""

import copy
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ei.backtest import run_backtest
from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares
from ei.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TE_MAX_VALUES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06]


def main():
    cfg = load_config("config.yaml")
    np.random.seed(int(cfg.get("seed", 42)))

    logger.info("Loading real market data...")
    prices = load_prices(cfg)
    tickers = list(prices.columns)
    shares = load_shares(tickers, cfg)
    sectors = load_sectors(tickers, cfg)

    logger.info("Running benchmark as reference...")
    res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
    r_b = res_b["returns"]

    rf = float(cfg.get("risk_free_annual", 0.065))
    records = []

    for te in TE_MAX_VALUES:
        logger.info("Loose sweep te_max=%.4f ...", te)
        cfg_i = copy.deepcopy(cfg)
        cfg_i["optimizer"]["te_max"] = float(te)
        cfg_i["optimizer"]["max_active_weight"] = 0.10
        cfg_i["optimizer"]["sector_dev_max"] = 0.15
        res_i = run_backtest(prices, shares, sectors, cfg_i, strategy="enhanced_lw")

        m = compute_metrics(
            res_i["returns"],
            r_b,
            rf=rf,
            turnover=res_i.get("turnover"),
            ex_ante_te=res_i.get("ex_ante_te"),
        )
        records.append({
            "te_max": float(te),
            "ann_return": m["ann_return"],
            "excess": m["excess_return"],
            "realized_TE": m["realized_TE"],
            "IR": m["IR"],
            "turnover": m["ann_turnover"],
        })
        logger.info(
            "  IR=%.3f  excess=%.4f  realized_TE=%.4f  turnover=%.2f",
            m["IR"], m["excess_return"], m["realized_TE"], m["ann_turnover"],
        )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(results_dir / "sweep_loose.csv", index=False)
    logger.info("Saved results/sweep_loose.csv")

    # Plot IR and excess return vs te_max
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(df["te_max"] * 100, df["IR"], marker="o", color="#1f77b4")
    ax1.set_xlabel("te_max (%)")
    ax1.set_ylabel("Information Ratio")
    ax1.set_title("IR vs te_max (loose constraints)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["te_max"] * 100, df["excess"] * 100, marker="s", color="#ff7f0e")
    ax2.set_xlabel("te_max (%)")
    ax2.set_ylabel("Excess Return (%)")
    ax2.set_title("Excess Return vs te_max (loose constraints)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "ir_vs_te_loose.png", dpi=150)
    plt.close()
    logger.info("Saved results/ir_vs_te_loose.png")

    print(df.to_string(index=False))
    # Check monotonicity of realized TE
    te_vals = df["realized_TE"].values
    print("\nRealized TE rises with te_max?", all(te_vals[i] <= te_vals[i+1] + 0.001 for i in range(len(te_vals)-1)))


if __name__ == "__main__":
    main()
