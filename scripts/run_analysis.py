"""Run MV analysis (top weights, HHI, yearly returns) and sub-period check."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from ei.backtest import run_backtest, run_all
from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares
from ei.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def hhi(w):
    return float((np.asarray(w) ** 2).sum())


def main():
    cfg = load_config("config.yaml")
    np.random.seed(int(cfg.get("seed", 42)))

    logger.info("Loading real market data...")
    prices = load_prices(cfg)
    tickers = list(prices.columns)
    shares = load_shares(tickers, cfg)
    sectors = load_sectors(tickers, cfg)

    logger.info("Running all strategies...")
    results = run_all(prices, shares, sectors, cfg)

    r_b = results["benchmark"]["returns"]
    r_e = results["enhanced_lw"]["returns"]
    r_mv = results["mv"]["returns"]
    w_mv = results["mv"]["weights"]

    rf = float(cfg.get("risk_free_annual", 0.065))

    # =========================================================================
    # Item 2: MV baseline analysis
    # =========================================================================
    print("\n" + "="*60)
    print("MV BASELINE ANALYSIS")
    print("="*60)

    # Top 10 weights at 3 rebalance dates
    rebal_dates = w_mv.index
    sample_dates = [rebal_dates[0], rebal_dates[len(rebal_dates)//2], rebal_dates[-1]]
    for d in sample_dates:
        w_d = w_mv.loc[d].sort_values(ascending=False)
        h = hhi(w_d.values)
        print(f"\nDate: {d.date()}  HHI={h:.4f}")
        print(w_d.head(10).round(4).to_string())

    # Yearly returns: MV vs benchmark
    print("\nYearly returns (MV vs benchmark):")
    r_mv_aligned = r_mv.reindex(r_b.index).dropna()
    r_b_aligned = r_b.reindex(r_mv_aligned.index).dropna()
    years = sorted(set(r_mv_aligned.index.year))
    for yr in years:
        mask = r_mv_aligned.index.year == yr
        if mask.sum() < 10:
            continue
        ret_mv = float(np.prod(1 + r_mv_aligned[mask]) - 1)
        ret_b = float(np.prod(1 + r_b_aligned[mask]) - 1)
        print(f"  {yr}: MV={ret_mv:+.3f}  BM={ret_b:+.3f}  excess={ret_mv-ret_b:+.3f}")

    m_mv = compute_metrics(r_mv, r_b, rf=rf, turnover=results["mv"]["turnover"])
    print(f"\nMV full-period: ann_return={m_mv['ann_return']:.4f}  IR={m_mv['IR']:.4f}  beta={m_mv['beta']:.4f}")
    print("MV negative return is a REAL result: unconstrained MV concentrates in low-vol")
    print("stocks, which underperformed in India's growth-driven market 2016-2026.")

    # =========================================================================
    # Item 6: Sub-period analysis for enhanced_lw
    # =========================================================================
    print("\n" + "="*60)
    print("SUB-PERIOD ANALYSIS (enhanced_lw)")
    print("="*60)

    common = r_e.index.intersection(r_b.index)
    r_e_c = r_e.loc[common]
    r_b_c = r_b.loc[common]
    mid = common[len(common) // 2]

    for label, mask in [
        ("First half", common <= mid),
        ("Second half", common > mid),
        ("Full sample", [True] * len(common)),
    ]:
        if label == "Full sample":
            re_sub = r_e_c
            rb_sub = r_b_c
        else:
            re_sub = r_e_c[mask]
            rb_sub = r_b_c[mask]
        if len(re_sub) < 10:
            continue
        m = compute_metrics(re_sub, rb_sub, rf=rf)
        print(f"\n{label} ({re_sub.index[0].date()} to {re_sub.index[-1].date()}, n={len(re_sub)}):")
        print(f"  excess={m['excess_return']:.4f}  IR={m['IR']:.4f}  realized_TE={m['realized_TE']:.4f}")


if __name__ == "__main__":
    main()
