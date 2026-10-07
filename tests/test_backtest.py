"""Unit tests for walk-forward backtest mechanics, drift, costs, and determinism."""

import copy
import numpy as np
import pandas as pd
import pytest

from ei.backtest import get_rebalance_dates, run_all, run_backtest



# test_benchmark_vs_itself was deleted: it subtracted the returns series from
# itself (always zero) and could not detect any bug. Replaced by
# test_benchmark_returns_mechanical in test_backtest_hardened.py.



def test_enhanced_zero_alpha_matches_benchmark(synth, cfg, monkeypatch):
    """Verify enhanced with forced alpha=0 and tc_bps=0 produces active return ~0."""
    prices, shares, sectors = synth
    cfg_zero_tc = copy.deepcopy(cfg)
    cfg_zero_tc["costs"]["tc_bps"] = 0.0

    monkeypatch.setattr(
        "ei.backtest.compute_alpha",
        lambda *args, **kwargs: pd.Series(0.0, index=prices.columns),
    )

    res_b = run_backtest(prices, shares, sectors, cfg_zero_tc, strategy="benchmark")
    res_e = run_backtest(prices, shares, sectors, cfg_zero_tc, strategy="enhanced")

    active = res_e["returns"] - res_b["returns"]
    # Assert per-day max, not just the mean (mean can cancel out alternating errors)
    assert active.abs().max() < 1e-5


def test_no_lookahead_backtest(synth, cfg):
    """Verify future price modifications after date D do not affect prior weights or returns."""
    prices, shares, sectors = synth
    rebal_dates = get_rebalance_dates(prices.index, cfg["warmup_days"])
    assert len(rebal_dates) >= 3
    date_d = rebal_dates[2]

    res_base = run_backtest(prices, shares, sectors, cfg, strategy="enhanced")

    prices_altered = prices.copy()
    prices_altered.loc[prices.index > date_d] *= 3.0
    res_altered = run_backtest(prices_altered, shares, sectors, cfg, strategy="enhanced")

    # Weights for rebalances < date_d must be identical
    w_base_prior = res_base["weights"].loc[res_base["weights"].index < date_d]
    w_alt_prior = res_altered["weights"].loc[res_altered["weights"].index < date_d]
    pd.testing.assert_frame_equal(w_base_prior, w_alt_prior)

    # Returns before date_d + 1 (i.e. <= date_d) must be identical
    r_base_prior = res_base["returns"].loc[res_base["returns"].index <= date_d]
    r_alt_prior = res_altered["returns"].loc[res_altered["returns"].index <= date_d]
    pd.testing.assert_series_equal(r_base_prior, r_alt_prior)


def test_rebalance_dates_properties(synth, cfg):
    """Verify all rebalance dates are month-end trading days, strictly increasing, and after warmup."""
    prices, _, _ = synth
    rebal_dates = get_rebalance_dates(prices.index, cfg["warmup_days"])

    assert len(rebal_dates) > 0
    # Strictly increasing
    for i in range(len(rebal_dates) - 1):
        assert rebal_dates[i] < rebal_dates[i + 1]

    # After warmup
    min_date = prices.index[cfg["warmup_days"]]
    assert all(d >= min_date for d in rebal_dates)

    # Month-end trading day
    for d in rebal_dates:
        pos = prices.index.get_loc(d)
        if pos < len(prices.index) - 1:
            next_day = prices.index[pos + 1]
            assert next_day.month != d.month


def test_weights_sum_to_one(synth, cfg):
    """Verify portfolio weights sum to 1 at all rebalance dates."""
    prices, shares, sectors = synth
    for strat in ["benchmark", "enhanced"]:
        res = run_backtest(prices, shares, sectors, cfg, strategy=strat)
        sums = res["weights"].sum(axis=1)
        assert np.allclose(sums.values, 1.0, atol=1e-6)


def test_cost_accounting(synth, cfg):
    """Verify tc_bps=0 has higher return than tc_bps=50, with difference matching turnover cost within 5%."""
    prices, shares, sectors = synth

    cfg_0 = copy.deepcopy(cfg)
    cfg_0["costs"]["tc_bps"] = 0.0
    res_0 = run_backtest(prices, shares, sectors, cfg_0, strategy="benchmark")

    cfg_50 = copy.deepcopy(cfg)
    cfg_50["costs"]["tc_bps"] = 50.0
    res_50 = run_backtest(prices, shares, sectors, cfg_50, strategy="benchmark")

    tot_ret_0 = float(np.prod(1.0 + res_0["returns"].values) - 1.0)
    tot_ret_50 = float(np.prod(1.0 + res_50["returns"].values) - 1.0)
    assert tot_ret_0 > tot_ret_50

    # Total cost diff in sum of daily returns equals tc * total turnover
    tot_turnover = float(res_50["turnover"].sum())
    expected_tc_diff = (50.0 / 10000.0) * tot_turnover
    actual_return_diff = float((res_0["returns"] - res_50["returns"]).sum())

    assert np.isclose(actual_return_diff, expected_tc_diff, rtol=0.05)


def test_output_shapes_and_determinism(synth, cfg):
    """Verify outputs have no NaNs, weights sum to 1, and runs are deterministic."""
    prices, shares, sectors = synth
    res1 = run_backtest(prices, shares, sectors, cfg, strategy="enhanced")
    res2 = run_backtest(prices, shares, sectors, cfg, strategy="enhanced")

    assert not res1["returns"].isna().any()
    assert np.allclose(res1["weights"].sum(axis=1), 1.0, atol=1e-6)

    pd.testing.assert_series_equal(res1["returns"], res2["returns"])
    pd.testing.assert_frame_equal(res1["weights"], res2["weights"])


def test_run_all(synth, cfg):
    """Verify run_all returns all four required strategies."""
    prices, shares, sectors = synth
    results = run_all(prices, shares, sectors, cfg)
    for expected_key in ["benchmark", "enhanced_lw", "enhanced_sample", "mv"]:
        assert expected_key in results
        assert "returns" in results[expected_key]
        assert "weights" in results[expected_key]
        assert len(results[expected_key]["returns"]) > 0
