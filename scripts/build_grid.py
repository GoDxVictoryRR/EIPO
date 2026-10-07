"""scripts/build_grid.py - Precompute results/grid.csv over parameter combinations.

Grid:
  te_max: [0.005, 0.01, 0.02, 0.03, 0.04]
  covariance: ['ledoit_wolf', 'sample']
  tc_bps: [0, 10, 20, 50]

Columns:
  te_max, covariance, tc_bps, ann_return, excess_return, gross_excess_return, cost_drag, realized_TE, IR, ann_turnover
"""

import copy
import logging
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ei.backtest import run_backtest
from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares
from ei.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global cache for worker processes
_worker_data: dict[str, Any] = {}


def _init_worker(prices: pd.DataFrame, shares: pd.Series, sectors: pd.Series, r_b: pd.Series, cfg: dict) -> None:
    _worker_data["prices"] = prices
    _worker_data["shares"] = shares
    _worker_data["sectors"] = sectors
    _worker_data["r_b"] = r_b
    _worker_data["cfg"] = cfg


def _eval_combo(params: tuple[float, str, float]) -> dict[str, Any]:
    te_max, cov_method, tc_bps = params
    prices = _worker_data["prices"]
    shares = _worker_data["shares"]
    sectors = _worker_data["sectors"]
    r_b = _worker_data["r_b"]
    base_cfg = _worker_data["cfg"]

    c = copy.deepcopy(base_cfg)
    c["optimizer"]["te_max"] = float(te_max)
    c["covariance"] = str(cov_method)
    c["costs"]["tc_bps"] = float(tc_bps)

    res = run_backtest(prices, shares, sectors, c, strategy="enhanced")
    r = res["returns"]
    t_over = res.get("turnover")
    ex_te = res.get("ex_ante_te")

    # Annual turnover & cost drag
    n = len(r.dropna())
    years = n / 252.0
    ann_to = float(t_over.sum()) / max(years, 1.0 / 252.0) if t_over is not None else 0.0
    cost_drag = (float(tc_bps) / 10000.0) * ann_to

    m = compute_metrics(
        r,
        r_b,
        rf=float(c.get("risk_free_annual", 0.065)),
        turnover=t_over,
        ex_ante_te=ex_te,
        cost_drag_annual=cost_drag,
    )

    return {
        "te_max": float(te_max),
        "covariance": str(cov_method),
        "tc_bps": int(tc_bps) if tc_bps == int(tc_bps) else float(tc_bps),
        "ann_return": float(m["ann_return"]),
        "excess_return": float(m["excess_return"]),
        "gross_excess_return": float(m["gross_excess_return"]),
        "cost_drag": float(m["cost_drag"]),
        "realized_TE": float(m["realized_TE"]),
        "IR": float(m["IR"]),
        "ann_turnover": float(m["ann_turnover"]),
    }


def build_grid(output_dir: Path | None = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(ROOT / "config.yaml"))
    np.random.seed(int(cfg.get("seed", 42)))

    logger.info("Loading market data for grid build...")
    prices = load_prices(cfg)
    tickers = list(prices.columns)
    shares = load_shares(tickers, cfg)
    sectors = load_sectors(tickers, cfg)

    logger.info("Running baseline benchmark backtest...")
    bm_res = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
    r_b = bm_res["returns"]

    te_max_list = [0.005, 0.01, 0.02, 0.03, 0.04]
    cov_list = ["ledoit_wolf", "sample"]
    tc_bps_list = [0, 10, 20, 50]

    combos = [
        (te, cov, tc)
        for te in te_max_list
        for cov in cov_list
        for tc in tc_bps_list
    ]
    logger.info("Evaluating %d parameter combinations...", len(combos))

    n_workers = min(max(1, mp.cpu_count() - 1), 8)
    ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(prices, shares, sectors, r_b, cfg),
    ) as pool:
        rows = pool.map(_eval_combo, combos)

    df = pd.DataFrame(rows)
    # Order columns as required by spec
    cols = [
        "te_max",
        "covariance",
        "tc_bps",
        "ann_return",
        "excess_return",
        "gross_excess_return",
        "cost_drag",
        "realized_TE",
        "IR",
        "ann_turnover",
    ]
    df = df[cols]

    out_csv = output_dir / "grid.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Successfully wrote %d rows to %s", len(df), out_csv)
    return df


if __name__ == "__main__":
    build_grid()
