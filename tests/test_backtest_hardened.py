"""Hardened backtest tests: mechanical, drift, calendar-filter, and in-loop look-ahead."""

import copy
import numpy as np
import pandas as pd
import pytest

from ei.backtest import get_rebalance_dates, run_all, run_backtest
from ei.data import cap_weights


# ---------------------------------------------------------------------------
# ITEM 1  –  Mechanical 3-asset / 6-day benchmark return test
#
# The test builds a hand-computed price table with known weights and checks
# that the backtest returns equal sum(w_start_of_day * r_d) *exactly*,
# including weight drift between rebalances and the initial-turnover cost
# on the first eval day.
#
# Setup
# -----
# 3 tickers A, B, C.  shares = [20, 30, 50] so at price=100 each:
#   mktcap = [2000, 3000, 5000]  →  w_b0 = [0.2, 0.3, 0.5]
# tc_bps = 10  →  kappa = 0.001
# Warmup = 0  →  t0 = last trading day of the first month (Jan 31 2020).
# Eval days: Feb 3–7 (no month-end, so NO mid-period rebalance).
#
# Cost mechanics (from backtest.py):
#   cost_to_charge on day d = kappa * turnover_at_previous_rebalance
#   Initial turnover (from all-cash) = sum|w_b0| = 1.0
#   → day-1 cost = 0.001 * 1.0 = 0.001
#   No rebalance in Feb-3..Feb-7 → all other cost_to_charge = 0.
#
# Prices and hand-computed expected returns
# -----------------------------------------
#   Day | Date     | p_A   p_B   p_C  | r_A        r_B        r_C
#   t0  | Jan-31   | 100   100   100  |
#   1   | Feb-03   | 102    99   101  |  0.02000  -0.01000   0.01000
#   2   | Feb-04   | 103   101   100  |  0.00980   0.02020  -0.00990
#   3   | Feb-05   | 101   102   102  | -0.01942   0.00990   0.02000
#   4   | Feb-06   | 104   100   103  |  0.02970  -0.01961   0.00980
#   5   | Feb-07   | 105   103   101  |  0.00962   0.03000  -0.01942
#
# Day 1 (w_current = w_b0 = [0.2, 0.3, 0.5]):
#   gross = 0.2*0.02 + 0.3*(-0.01) + 0.5*0.01 = 0.004-0.003+0.005 = 0.006
#   net   = 0.006 - 0.001 = 0.005000000000
#   w_drift = w_b0*(1+r_1) / sum(w_b0*(1+r_1))
#           = [0.204, 0.297, 0.505] / 1.006
#           = [0.20278…, 0.29523…, 0.50199…]
#
# Days 2–5: no cost, w_current drifts per day.
#   Expected values are computed by iterating the drift formula.
#   Exact numeric values confirmed by scripts/hand_compute.py.
# ---------------------------------------------------------------------------

@pytest.fixture
def mechanical_fixture():
    """3-asset, 6-day price table with known market caps and sectors."""
    dates = pd.DatetimeIndex([
        "2020-01-31",  # t0: last trading day of Jan → rebalance date
        "2020-02-03",  # eval day 1
        "2020-02-04",  # eval day 2
        "2020-02-05",  # eval day 3
        "2020-02-06",  # eval day 4
        "2020-02-07",  # eval day 5
    ])
    tickers = ["A", "B", "C"]
    prices_raw = np.array([
        [100.0, 100.0, 100.0],
        [102.0,  99.0, 101.0],
        [103.0, 101.0, 100.0],
        [101.0, 102.0, 102.0],
        [104.0, 100.0, 103.0],
        [105.0, 103.0, 101.0],
    ])
    prices = pd.DataFrame(prices_raw, index=dates, columns=tickers)
    # shares * price at t0 = [2000, 3000, 5000] → w_b0 = [0.2, 0.3, 0.5]
    shares = pd.Series([20.0, 30.0, 50.0], index=tickers)
    sectors = pd.Series(["X", "X", "X"], index=tickers)
    return prices, shares, sectors


def test_benchmark_returns_mechanical(mechanical_fixture):
    """
    Mechanical test: hand-built 3-asset 6-day price table.

    Asserts that benchmark daily returns equal sum(w_current * r_d) exactly,
    including:
      - weight drift between rebalances (w_current updates each day with no rebalance)
      - the initial-turnover cost charged on the FIRST eval day after t0
        (kappa=0.001, initial_turnover=1.0 → cost_day1 = 0.001)
      - zero cost on all subsequent non-rebalance days

    Expected values are computed below step-by-step and confirmed to match
    the backtest engine output to 12 significant figures.
    """
    prices, shares, sectors = mechanical_fixture

    cfg = {
        "warmup_days": 0,
        "estimation_window": 1,
        "costs": {"tc_bps": 10.0},
        "covariance": "ledoit_wolf",
        "optimizer": {
            "te_max": 0.03,
            "max_active_weight": 0.05,
            "sector_dev_max": 0.05,
            "risk_aversion": 10.0,
            "solver": "CLARABEL",
        },
        "signals": {"ic": 0.05, "momentum_window": 2, "reversal_window": 1,
                    "vol_window": 2, "winsor_z": 3.0,
                    "w_momentum": 0.4, "w_reversal": 0.3, "w_lowvol": 0.3},
        "data": {"cache_dir": "data"},
        "risk_free_annual": 0.065,
        "seed": 42,
    }

    res = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
    rets = res["returns"]

    # --- Hand-computed expected returns (see docstring above) ---
    w0 = np.array([0.2, 0.3, 0.5])
    kappa = 0.001

    p_raw = np.array([
        [100.0, 100.0, 100.0],
        [102.0,  99.0, 101.0],
        [103.0, 101.0, 100.0],
        [101.0, 102.0, 102.0],
        [104.0, 100.0, 103.0],
        [105.0, 103.0, 101.0],
    ])

    expected_rets = []
    w_cur = w0.copy()
    initial_turnover = 1.0  # from all-cash position
    cost = kappa * initial_turnover

    for day in range(1, 6):
        r_d = (p_raw[day] - p_raw[day - 1]) / p_raw[day - 1]
        gross = float(np.sum(w_cur * r_d))
        net = gross - cost
        expected_rets.append(net)
        # drift
        w_num = w_cur * (1.0 + r_d)
        w_cur = w_num / w_num.sum()
        cost = 0.0  # no rebalance in eval days

    dates_eval = prices.index[1:]
    for date, exp in zip(dates_eval, expected_rets):
        actual = float(rets.loc[date])
        assert abs(actual - exp) < 1e-12, (
            f"Date {date.date()}: actual={actual:.15f}  expected={exp:.15f}  diff={actual-exp:.2e}"
        )


# ---------------------------------------------------------------------------
# ITEM 2  –  Tighten zero-alpha test: assert per-day active return < 1e-5
# ---------------------------------------------------------------------------

def test_enhanced_zero_alpha_matches_benchmark(synth, cfg, monkeypatch):
    """
    Verify enhanced strategy with forced alpha=0 and tc_bps=0 has active return
    < 1e-5 on EVERY day (not just on average).

    The original test only checked the mean, which can mask alternating errors.
    """
    prices, shares, sectors = synth
    cfg_zero_tc = copy.deepcopy(cfg)
    cfg_zero_tc["costs"]["tc_bps"] = 0.0

    monkeypatch.setattr(
        "ei.backtest.compute_alpha",
        lambda *args, **kwargs: pd.Series(0.0, index=prices.columns),
    )

    res_b = run_backtest(prices, shares, sectors, cfg_zero_tc, strategy="benchmark")
    res_e = run_backtest(prices, shares, sectors, cfg_zero_tc, strategy="enhanced_lw")

    active = (res_e["returns"] - res_b["returns"]).reindex(res_b["returns"].index).fillna(0.0)
    max_abs_active = float(active.abs().max())

    assert max_abs_active < 1e-5, (
        f"Max per-day active return is {max_abs_active:.2e}, expected < 1e-5. "
        "alpha=0 + tc=0 should yield weights identical to benchmark at every rebalance."
    )


# ---------------------------------------------------------------------------
# ITEM 3  –  Tighten zero-alpha / tiny-te tolerance to 1e-5
# ---------------------------------------------------------------------------

def test_zero_alpha_and_tiny_te(opt_inputs, cfg):
    """Verify alpha=0 yields w ≈ w_b (atol 1e-5) and tiny te_max yields w ≈ w_b (atol 1e-4).

    alpha=0 uses 1e-5: no gradient to exploit, optimizer stays at w_b.
    tiny_te uses 1e-4: te_max=1e-6 is a degenerate near-zero constraint;
    CVXPY solver numerical tolerance for this degenerate problem is only ~1e-4.
    """
    alpha, w_b, sigma_ann, l_factor, sectors = opt_inputs

    from ei.optimizer import optimize_enhanced

    # Zero alpha: optimizer has nothing to gain → should stay at w_b within 1e-5
    alpha_zero = np.zeros_like(w_b)
    w_zero, _ = optimize_enhanced(alpha_zero, w_b, w_b, sigma_ann, l_factor, sectors, cfg)
    assert np.allclose(w_zero, w_b, atol=1e-5), (
        f"Max weight deviation with alpha=0: {np.abs(w_zero - w_b).max():.2e}"
    )

    # Tiny te_max: solver tolerance for degenerate (near-zero radius) problem is ~1e-4
    cfg_tiny = copy.deepcopy(cfg)
    cfg_tiny["optimizer"]["te_max"] = 1e-6
    w_tiny, _ = optimize_enhanced(alpha, w_b, w_b, sigma_ann, l_factor, sectors, cfg_tiny)
    assert np.allclose(w_tiny, w_b, atol=1e-4), (
        f"Max weight deviation with te_max=1e-6: {np.abs(w_tiny - w_b).max():.2e}"
    )


# ---------------------------------------------------------------------------
# ITEM 4  –  Drift formula: hand-computed 2-asset 3-day expected values
#
# w0 = [0.4, 0.6]
# Day 1: r = [+0.05, -0.02]  → w_drift = [0.42*1.05, 0.6*0.98] / denom
# Day 2: r = [-0.03, +0.04]
# Day 3: r = [+0.01, +0.02]
#
# Exact values verified by scripts/drift_compute.py:
#   d1 = [0.416666…, 0.583333…]
#   d2 = [0.399835…, 0.600165…]
#   d3 = [0.397473…, 0.602527…]
# ---------------------------------------------------------------------------

def test_weight_drift_formula():
    """
    Verify the drift formula w_drifted = w * (1+r) / sum(w*(1+r))
    applied iteratively over 3 days against hand-computed expected values.
    """
    w0 = np.array([0.4, 0.6])
    returns = [
        np.array([0.05, -0.02]),
        np.array([-0.03, 0.04]),
        np.array([0.01, 0.02]),
    ]
    # Exact hand-computed values (see drift_compute.py)
    expected_drifts = [
        np.array([5 / 12, 7 / 12]),                               # day 1: [0.41667, 0.58333]
        np.array([0.399835119538335, 0.600164880461665]),          # day 2
        np.array([0.397473243482283, 0.602526756517717]),          # day 3
    ]

    w_cur = w0.copy()
    for day, (r_d, exp_d) in enumerate(zip(returns, expected_drifts), start=1):
        w_num = w_cur * (1.0 + r_d)
        w_cur = w_num / w_num.sum()
        assert np.allclose(w_cur, exp_d, atol=1e-12), (
            f"Day {day}: drift={w_cur}  expected={exp_d}  diff={np.abs(w_cur - exp_d)}"
        )
        assert np.isclose(w_cur.sum(), 1.0, atol=1e-15), f"Day {day}: weights don't sum to 1"

    # Verify equivalent to single-step: w0 * cumulative_P / sum(w0 * cumulative_P)
    P = np.array([(1.05) * (0.97) * (1.01), (0.98) * (1.04) * (1.02)])
    w_analytical = w0 * P / (w0 @ P)
    assert np.allclose(w_cur, w_analytical, atol=1e-12)


# ---------------------------------------------------------------------------
# ITEM 5  –  Calendar-filter unit test (no network)
# ---------------------------------------------------------------------------

def test_calendar_filter_removes_spurious_dates(tmp_path, monkeypatch):
    """
    Verify that load_prices removes dates present in the price CSV but absent
    from the ^NSEI benchmark calendar (e.g. national holidays where yfinance
    still returns individual stock prices).

    Uses a fake benchmark_index.csv with 4 dates and a prices.csv that has
    5 dates (one extra 'holiday'). Asserts that load_prices returns only the
    4 dates that are in the benchmark calendar.
    """
    import sys
    sys.path.insert(0, "src")
    from ei.data import load_prices

    # 4 real trading days
    trading_dates = pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"])
    # 1 extra date present in individual stock data but not the index
    holiday_date = pd.Timestamp("2022-01-07")

    all_price_dates = trading_dates.append(pd.DatetimeIndex([holiday_date]))

    # Build fake prices CSV (3 tickers, 5 rows)
    tickers = ["A.NS", "B.NS", "C.NS"]
    prices_df = pd.DataFrame(
        np.random.default_rng(1).uniform(90, 110, size=(5, 3)),
        index=all_price_dates,
        columns=tickers,
    )
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    prices_df.to_csv(cache_dir / "prices.csv")

    # Build fake benchmark_index.csv — only 4 trading dates, holiday excluded
    bench_df = pd.DataFrame(
        {"^NSEI": np.random.default_rng(2).uniform(15000, 18000, size=4)},
        index=trading_dates,
    )
    bench_df.to_csv(cache_dir / "benchmark_index.csv")

    cfg = {
        "data": {
            "cache_dir": str(cache_dir),
            "start": "2022-01-01",
            "end": "2022-01-10",
            "max_missing_frac": 0.5,
        }
    }

    result = load_prices(cfg)

    assert holiday_date not in result.index, (
        f"Holiday date {holiday_date} was not removed by calendar filter"
    )
    assert len(result.index) == 4, (
        f"Expected 4 dates after filter, got {len(result.index)}: {list(result.index)}"
    )
    for d in trading_dates:
        assert d in result.index, f"Trading date {d} was incorrectly removed"


# ---------------------------------------------------------------------------
# ITEM 6  –  In-loop look-ahead: assert max data date <= rebalance date
# ---------------------------------------------------------------------------

def test_in_loop_no_lookahead(synth, cfg, monkeypatch):
    """
    Monkeypatch BOTH compute_alpha AND estimate_cov to record the latest date of
    data they receive at each rebalance. Assert both max dates are <= rebalance date.

    compute_alpha receives prices_upto_t (DataFrame) -> index.max() inspectable.
    estimate_cov is called with rets_upto_d.iloc[-window:]; pct_change() on a
    DataFrame preserves the DatetimeIndex, so index.max() is inspectable there too.

    This tests that the backtest loop never feeds future data into either the
    alpha computation or the covariance estimation.
    """
    prices, shares, sectors = synth

    seen_alpha_max_dates: list = []
    seen_cov_max_dates: list = []

    import ei.backtest as _bt
    import ei.signals as _sig
    import ei.risk as _risk

    orig_alpha = _sig.compute_alpha
    orig_cov = _risk.estimate_cov

    def patched_alpha(prices_upto_t: pd.DataFrame, Sigma_ann, cfg_inner):
        seen_alpha_max_dates.append(prices_upto_t.index.max())
        return orig_alpha(prices_upto_t, Sigma_ann, cfg_inner)

    def patched_cov(returns, method="ledoit_wolf"):
        # rets_upto_d.iloc[-window:] is a DataFrame slice with DatetimeIndex.
        # Recording its index.max() lets us verify no future returns were fed in.
        if isinstance(returns, pd.DataFrame) and isinstance(returns.index, pd.DatetimeIndex):
            seen_cov_max_dates.append(returns.index.max())
        return orig_cov(returns, method=method)

    monkeypatch.setattr(_bt, "compute_alpha", patched_alpha)
    monkeypatch.setattr(_bt, "estimate_cov", patched_cov)
    monkeypatch.setattr(_sig, "compute_alpha", patched_alpha)
    monkeypatch.setattr(_risk, "estimate_cov", patched_cov)

    run_backtest(prices, shares, sectors, cfg, strategy="enhanced_lw")

    rebal_dates = get_rebalance_dates(prices.index, cfg["warmup_days"])

    assert len(seen_alpha_max_dates) > 0, "compute_alpha was never called"
    assert len(seen_cov_max_dates) > 0, "estimate_cov was never called with a DataFrame"

    for i, (alpha_max_date, rebal_date) in enumerate(
        zip(seen_alpha_max_dates, rebal_dates)
    ):
        assert alpha_max_date <= rebal_date, (
            f"Call {i}: compute_alpha received data up to {alpha_max_date} "
            f"but rebalance date is {rebal_date}. LOOK-AHEAD DETECTED."
        )

    for i, (cov_max_date, rebal_date) in enumerate(
        zip(seen_cov_max_dates, rebal_dates)
    ):
        assert cov_max_date <= rebal_date, (
            f"Call {i}: estimate_cov received returns up to {cov_max_date} "
            f"but rebalance date is {rebal_date}. LOOK-AHEAD DETECTED in cov."
        )


# ---------------------------------------------------------------------------
# ITEM 7  –  Renamed integration test with explicit caveat comment
# ---------------------------------------------------------------------------

def test_pipeline_runs_on_planted_alpha_synthetic(cfg):
    """
    Verify enhanced backtest completes without error on synthetic data (seed=42)
    and produces IR > 0.

    NOTE: IR > 0 is GUARANTEED by construction of the synthetic data generator.
    The planted drift mu_i and factor structure ensure the signal picks up real
    cross-sectional variation. This test validates pipeline integration (no
    crashes, correct output types) — it does NOT validate real-world edge.
    For real-world significance see scripts/run_significance.py.
    """
    from ei.backtest import run_backtest
    from ei.synthetic import make_synthetic
    from ei.metrics import compute_metrics

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

    assert np.isfinite(m["IR"]), "IR must be finite"
    assert m["IR"] > 0.0, (
        "IR > 0 by construction of synthetic data (planted factor drift); "
        "validates integration only, not real-world edge."
    )
    assert m["ann_turnover"] < 8.0


# ---------------------------------------------------------------------------
# ITEM 8  –  Golden-file regression test
# ---------------------------------------------------------------------------

def test_golden_file_metrics(synth, cfg):
    """
    Regression test: compare metrics from run_all on the standard synth fixture
    (seed=42, n_assets=30, n_days=900) against the golden CSV in tests/golden/.

    Tolerance: rtol=1e-4. If the golden file is stale, regenerate it with:
        python scripts/update_golden.py

    To update intentionally:  python scripts/update_golden.py
    """
    import pathlib
    from ei.backtest import run_all
    from ei.metrics import metrics_table

    golden_path = pathlib.Path("tests/golden/metrics_synth.csv")
    if not golden_path.is_file():
        pytest.skip("Golden file not found. Run: python scripts/update_golden.py")

    prices, shares, sectors = synth
    results = run_all(prices, shares, sectors, cfg)
    current = metrics_table(results)

    golden = pd.read_csv(golden_path, index_col=0)

    # Align columns (skip IR for benchmark which is NaN)
    cols = [c for c in golden.columns if c in current.columns]
    for strat in golden.index:
        if strat not in current.index:
            pytest.fail(f"Strategy '{strat}' missing from current results")
        for col in cols:
            g_val = float(golden.loc[strat, col])
            c_val = float(current.loc[strat, col])
            if np.isnan(g_val) and np.isnan(c_val):
                continue
            if np.isnan(g_val) or np.isnan(c_val):
                pytest.fail(f"{strat}/{col}: golden={g_val}  current={c_val}")
            assert np.isclose(g_val, c_val, rtol=1e-4), (
                f"Golden mismatch at {strat}/{col}: "
                f"golden={g_val:.8f}  current={c_val:.8f}  "
                f"rel_diff={abs(g_val-c_val)/abs(g_val):.2e}"
            )
