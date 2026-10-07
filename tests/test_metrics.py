"""Unit tests for performance and risk metrics with hand-checkable inputs."""

import numpy as np
import pandas as pd
import pytest

from ei.metrics import compute_metrics, metrics_table


def test_constant_daily_return():
    """Verify constant daily return of 0.001 over 252 days yields theoretical values."""
    dates = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series([0.001] * 252, index=dates)
    r_b = pd.Series([0.0005] * 252, index=dates)

    m = compute_metrics(r, r_b, rf=0.0)
    expected_ann_return = (1.001 ** 252) - 1.0

    assert np.isclose(m["ann_return"], expected_ann_return, atol=1e-6)
    assert np.isclose(m["ann_vol"], 0.0, atol=1e-6)
    assert np.isclose(m["max_drawdown"], 0.0, atol=1e-6)


def test_known_drawdown_series():
    """Verify known sequence [+10%, -20%, +5%] produces exactly -20% max drawdown."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    r = pd.Series([0.10, -0.20, 0.05], index=dates)

    m = compute_metrics(r, r, rf=0.0)
    assert np.isclose(m["max_drawdown"], -0.20, atol=1e-6)


def test_identical_returns_benchmark():
    """Verify r == r_b yields TE=0, beta=1, and IR=NaN (not inf)."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0005, 0.01, size=100), index=dates)

    m = compute_metrics(r, r, rf=0.05)
    assert np.isclose(m["realized_TE"], 0.0, atol=1e-6)
    assert np.isnan(m["IR"])
    assert not np.isinf(m["IR"])
    assert np.isclose(m["beta"], 1.0, atol=1e-6)


def test_scaled_returns_beta():
    """Verify r = 2 * r_b yields beta = 2.0."""
    dates = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(42)
    r_b = pd.Series(rng.normal(0.0005, 0.01, size=252), index=dates)
    r = 2.0 * r_b

    m = compute_metrics(r, r_b, rf=0.0)
    assert np.isclose(m["beta"], 2.0, atol=1e-6)


def test_ir_exact_formula():
    """Verify IR matches mean(active)*252 / (std(active, ddof=1)*sqrt(252)) on seeded random series."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(123)
    r_b = pd.Series(rng.normal(0.0005, 0.01, size=500), index=dates)
    r = pd.Series(rng.normal(0.0008, 0.012, size=500), index=dates)

    m = compute_metrics(r, r_b, rf=0.065)
    active = r - r_b
    expected_ir = float(active.mean() * 252.0 / (active.std(ddof=1) * np.sqrt(252.0)))

    assert np.isclose(m["IR"], expected_ir, atol=1e-6)


def test_metrics_table_structure():
    """Verify metrics_table builds dataframe with all expected columns."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    r = pd.Series(0.001, index=dates)
    results = {
        "benchmark": {"returns": r},
        "enhanced_lw": {"returns": r + 0.0002},
    }
    df = metrics_table(results)
    assert "ann_return" in df.columns
    assert "realized_TE" in df.columns
    assert "IR" in df.columns
    assert len(df) == 2
