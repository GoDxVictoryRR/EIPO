"""Integration tests for full pipeline execution, scripts, and convergence criteria."""

import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ei.backtest import run_all, run_backtest
from ei.metrics import compute_metrics
from ei.synthetic import make_synthetic


def test_run_all_and_ex_ante_te_limits(synth, cfg):
    """Verify run_all finishes in < 120s and all ex-ante tracking errors obey te_max."""
    prices, shares, sectors = synth

    start_t = time.time()
    results = run_all(prices, shares, sectors, cfg)
    elapsed = time.time() - start_t

    assert elapsed < 120.0
    for key in ["benchmark", "enhanced_lw", "enhanced_sample", "mv"]:
        assert key in results

    te_max = float(cfg["optimizer"]["te_max"])
    for strat in ["enhanced_lw", "enhanced_sample"]:
        ex_te = results[strat]["ex_ante_te"]
        assert (ex_te <= te_max + 1e-6).all()


def test_pipeline_runs_on_planted_alpha_synthetic(cfg):
    """Verify planted alpha on seed 42 achieves positive IR and reasonable turnover < 800%.

    NOTE: IR > 0 is GUARANTEED by construction of the synthetic data (planted factor drift
    and factor structure ensure the signals pick up real cross-sectional variation).
    This test validates pipeline integration — no crashes, correct output shapes — NOT
    real-world statistical edge. For significance, see scripts/run_significance.py.
    """
    prices, shares, sectors = make_synthetic(n_assets=40, n_days=2500, seed=42)

    res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
    res_e = run_backtest(prices, shares, sectors, cfg, strategy="enhanced_lw")

    m = compute_metrics(
        res_e["returns"],
        res_b["returns"],
        rf=float(cfg.get("risk_free_annual", 0.065)),
        turnover=res_e["turnover"],
        ex_ante_te=res_e["ex_ante_te"],
    )

    assert np.isfinite(m["IR"]), "IR must be finite (NaN indicates zero tracking error)"
    assert m["IR"] > 0.0, (
        "IR > 0 by construction of synthetic data; validates integration only"
    )
    assert m["ann_turnover"] < 8.0


def test_cli_scripts_synthetic(tmp_path):
    """Verify run_backtest.py and run_sweep.py generate expected CSV and PNG outputs without polluting results/."""
    # 1. Run backtest script
    cmd_backtest = [
        sys.executable,
        "scripts/run_backtest.py",
        "--synthetic",
        "--output-dir",
        str(tmp_path),
    ]
    res_bt = subprocess.run(cmd_backtest, capture_output=True, text=True)
    assert res_bt.returncode == 0, f"run_backtest failed: {res_bt.stderr}"

    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "returns.csv").is_file()
    assert (tmp_path / "weights_enhanced.csv").is_file()
    assert (tmp_path / "cumulative_returns.png").is_file()
    assert (tmp_path / "rolling_te.png").is_file()
    assert (tmp_path / "active_weights_last.png").is_file()
    assert (tmp_path / "drawdown.png").is_file()
    assert (tmp_path / "turnover.png").is_file()

    # 2. Run sweep script
    cmd_sweep = [
        sys.executable,
        "scripts/run_sweep.py",
        "--synthetic",
        "--output-dir",
        str(tmp_path),
    ]
    res_sw = subprocess.run(cmd_sweep, capture_output=True, text=True)
    assert res_sw.returncode == 0, f"run_sweep failed: {res_sw.stderr}"

    assert (tmp_path / "sweep.csv").is_file()
    assert (tmp_path / "ir_vs_te.png").is_file()

    # Verify sweep realized TE monotonicity (allow at most 1 violation of < 0.5%)
    sweep_df = pd.read_csv(tmp_path / "sweep.csv")
    te_vals = sweep_df["realized_TE"].values
    violations = 0
    for k in range(len(te_vals) - 1):
        if te_vals[k + 1] < te_vals[k] - 0.005:
            violations += 1
    assert violations <= 1
