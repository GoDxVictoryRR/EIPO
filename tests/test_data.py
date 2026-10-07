"""Unit tests for universe, data cleaning, market cap weights, and synthetic generator."""

import numpy as np
import pandas as pd
import pytest

from ei.data import cap_weights, clean_prices
from ei.synthetic import make_synthetic
from ei.universe import TICKERS, get_universe


def test_universe():
    """Verify that universe contains at least 30 NSE tickers ending in .NS."""
    univ = get_universe()
    assert len(univ) >= 30
    assert len(TICKERS) == len(univ)
    assert all(t.endswith(".NS") for t in univ)


def test_cap_weights(synth):
    """Verify cap_weights sums to 1 and has all non-negative weights."""
    prices, shares, _ = synth
    for date in [prices.index[0], prices.index[len(prices) // 2], prices.index[-1]]:
        w = cap_weights(prices, shares, date)
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert (w >= -1e-8).all()
        assert len(w) == prices.shape[1]


def test_make_synthetic_reproducibility():
    """Verify make_synthetic produces identical outputs for same seed and differs for different seeds."""
    p1, s1, sec1 = make_synthetic(n_assets=15, n_days=60, seed=42)
    p2, s2, sec2 = make_synthetic(n_assets=15, n_days=60, seed=42)
    p3, s3, sec3 = make_synthetic(n_assets=15, n_days=60, seed=99)

    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_series_equal(s1, s2)
    pd.testing.assert_series_equal(sec1, sec2)

    assert not p1.equals(p3)


def test_cleaning_rules():
    """Verify ticker with >5% NaN is dropped, gap <= 3 days is forward-filled, and gap > 3 days dropped."""
    dates = pd.bdate_range("2020-01-01", periods=100)

    # 1. Clean column
    col_clean = pd.Series(100.0, index=dates)

    # 2. Gap of 2 days (<= 3 days)
    col_gap2 = pd.Series(100.0, index=dates)
    col_gap2.iloc[10:12] = np.nan

    # 3. High NaN column: 10 out of 100 (> 5%)
    col_high_nan = pd.Series(100.0, index=dates)
    col_high_nan.iloc[20:30] = np.nan

    # 4. Long gap: 5 days (> 3 days) with total NaN <= 5%
    col_gap5 = pd.Series(100.0, index=dates)
    col_gap5.iloc[40:45] = np.nan

    raw_df = pd.DataFrame(
        {
            "CLEAN": col_clean,
            "GAP2": col_gap2,
            "HIGH_NAN": col_high_nan,
            "GAP5": col_gap5,
        }
    )

    cleaned = clean_prices(raw_df, max_missing_frac=0.05)

    assert "HIGH_NAN" not in cleaned.columns
    assert "GAP5" not in cleaned.columns
    assert "CLEAN" in cleaned.columns
    assert "GAP2" in cleaned.columns

    # Verify forward-fill for GAP2
    assert not cleaned["GAP2"].isna().any()
    assert cleaned.loc[dates[10], "GAP2"] == 100.0
    assert cleaned.loc[dates[11], "GAP2"] == 100.0
