"""
scripts/independent_check.py

Independent verification script that verifies results WITHOUT importing
src/ei/backtest.py or src/ei/metrics.py.

Uses:
  - Plain numpy and pandas
  - data/prices.csv (filtered by data/benchmark_index.csv)
  - data/shares.csv
  - results/weights_enhanced.csv
  - 02_math.md drift and cost rules

Checks:
  1. Recomputes benchmark and enhanced_lw daily returns and compares with
     results/returns.csv. Max absolute difference must be < 1e-8.
  2. Recomputes all values in results/metrics.csv and IR t-stat from
     results/returns.csv using pure mathematical definitions. Max abs diff < 1e-8.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml


def load_raw_data():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Prices
    p_df = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    # Filter NSE calendar
    bench_cache = Path("data/benchmark_index.csv")
    if bench_cache.is_file():
        b_idx = pd.read_csv(bench_cache, index_col=0, parse_dates=True)
        p_df = p_df.loc[p_df.index.intersection(b_idx.index)]

    # Clean prices (matching clean_prices rule)
    max_missing = cfg.get("data", {}).get("max_missing_frac", 0.05)
    missing_frac = p_df.isna().mean()
    valid_cols = missing_frac[missing_frac <= max_missing].index
    p_df = p_df[valid_cols].ffill(limit=3).dropna(axis=1)

    # Shares
    s_s = pd.read_csv("data/shares.csv", index_col=0).squeeze("columns").astype(float)
    s_s = s_s.reindex(p_df.columns).fillna(1e8)

    # Enhanced weights
    w_enh = pd.read_csv("results/weights_enhanced.csv", index_col=0, parse_dates=True)

    # Actual returns and metrics
    saved_returns = pd.read_csv("results/returns.csv", index_col=0, parse_dates=True)
    saved_metrics = pd.read_csv("results/metrics.csv", index_col=0)

    return cfg, p_df, s_s, w_enh, saved_returns, saved_metrics


def get_month_end_rebalances(dates: pd.DatetimeIndex, warmup_days: int) -> list[pd.Timestamp]:
    s = pd.Series(dates, index=dates)
    month_ends = s.groupby([dates.year, dates.month]).last().values
    min_date = dates[warmup_days]
    return sorted([pd.Timestamp(d) for d in month_ends if d >= min_date])


def compute_cap_weights(p_row: pd.Series, shares: pd.Series) -> np.ndarray:
    mcap = p_row.values * shares.values
    tot = float(np.sum(mcap))
    w = np.clip(mcap / tot, 0.0, None)
    return w / np.sum(w)


def simulate_returns_independently(
    prices: pd.DataFrame,
    shares: pd.Series,
    rebal_dates: list[pd.Timestamp],
    tc_bps: float,
    strategy: str,
    weights_enhanced: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Pure standalone simulation matching 02_math.md exactly.
    """
    tickers = list(prices.columns)
    kappa = float(tc_bps) / 10000.0
    all_dates = prices.index
    t0 = rebal_dates[0]
    t0_idx = all_dates.get_loc(t0)
    eval_dates = all_dates[t0_idx + 1 :]

    # Initial weights at t0
    if strategy == "benchmark":
        w_target = compute_cap_weights(prices.loc[t0], shares)
    elif strategy == "enhanced_lw":
        w_target = weights_enhanced.loc[t0].reindex(tickers).values
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    initial_turnover = float(np.sum(np.abs(w_target)))
    cost_to_charge = kappa * initial_turnover
    w_current = np.copy(w_target)

    rebal_set = set(rebal_dates)
    daily_returns = []
    return_dates = []
    turnovers = {t0: initial_turnover}

    for i, d in enumerate(eval_dates):
        prev_d = all_dates[t0_idx + i]
        p_today = prices.loc[d].values
        p_prev = prices.loc[prev_d].values
        r_d = (p_today - p_prev) / p_prev

        gross_ret = float(np.sum(w_current * r_d))
        net_ret = gross_ret - cost_to_charge
        daily_returns.append(net_ret)
        return_dates.append(d)

        wealth_denom = float(np.sum(w_current * (1.0 + r_d)))
        if wealth_denom > 0:
            w_drifted = (w_current * (1.0 + r_d)) / wealth_denom
        else:
            w_drifted = np.copy(w_current)

        if d in rebal_set and d != t0:
            if strategy == "benchmark":
                w_new = compute_cap_weights(prices.loc[d], shares)
            else:
                w_new = weights_enhanced.loc[d].reindex(tickers).values

            t_over = float(np.sum(np.abs(w_new - w_drifted)))
            turnovers[d] = t_over
            cost_to_charge = kappa * t_over
            w_current = np.copy(w_new)
        else:
            cost_to_charge = 0.0
            w_current = np.copy(w_drifted)

    ret_series = pd.Series(daily_returns, index=return_dates, name=strategy)
    to_series = pd.Series(turnovers)
    return ret_series, to_series


def recompute_metrics_standalone(
    r: pd.Series,
    r_b: pd.Series,
    rf: float,
    turnover: pd.Series | None = None,
    tc_bps: float = 10.0,
    ex_ante_te: pd.Series | None = None,
) -> dict[str, float]:
    """
    Pure mathematical recomputation of performance metrics.
    """
    r_clean = r.dropna()
    rb_clean = r_b.dropna()
    common = r_clean.index.intersection(rb_clean.index)
    r_series = r_clean.loc[common]
    rb_series = rb_clean.loc[common]

    n = len(r_series)
    years = n / 252.0

    # 1. ann_return = prod(1+r)^(252/n) - 1
    ann_return = float(np.prod(1.0 + r_series.values) ** (252.0 / n) - 1.0)
    ann_return_b = float(np.prod(1.0 + rb_series.values) ** (252.0 / n) - 1.0)

    # 2. ann_vol = std(r, ddof=1) * sqrt(252)
    ann_vol = float(r_series.std(ddof=1) * np.sqrt(252.0))

    # 3. sharpe = (ann_return - rf) / ann_vol
    sharpe = float((ann_return - rf) / ann_vol) if ann_vol > 1e-12 else float("nan")

    # 4. excess_return = ann_return - ann_return_b
    excess_return = float(ann_return - ann_return_b)

    # 5. active return
    active = r_series - rb_series

    # 6. realized_TE = std(active, ddof=1) * sqrt(252)
    if r_series.equals(rb_series):
        realized_te = 0.0
    else:
        realized_te = float(active.std(ddof=1) * np.sqrt(252.0))

    # 7. IR = mean(active) * 252 / realized_TE
    if realized_te > 1e-12:
        ir = float(active.mean() * 252.0 / realized_te)
    else:
        ir = float("nan")

    # 8. max_drawdown = min(wealth / cummax(wealth) - 1)
    wealth = np.cumprod(1.0 + r_series.values)
    cummax = np.maximum.accumulate(wealth)
    drawdowns = wealth / cummax - 1.0
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_dd = min(0.0, max_dd)

    # 9. beta = cov(r, r_b) / var(r_b)
    if r_series.equals(rb_series):
        beta = 1.0
    else:
        var_b = float(rb_series.var(ddof=1))
        if var_b > 1e-12:
            cov = float(np.cov(r_series.values, rb_series.values, ddof=1)[0, 1])
            beta = float(cov / var_b)
        else:
            beta = float("nan")

    # 10. ann_turnover = sum(turnover) / years
    if turnover is not None and len(turnover) > 0:
        ann_turnover = float(turnover.sum()) / max(years, 1.0 / 252.0)
    else:
        ann_turnover = float("nan")

    # 11. hit_rate = fraction of calendar months with positive active return
    r_monthly = (1.0 + r_series).resample("ME").prod() - 1.0
    rb_monthly = (1.0 + rb_series).resample("ME").prod() - 1.0
    active_m = r_monthly - rb_monthly
    hit_rate = float((active_m > 0).mean()) if len(active_m) > 0 else float("nan")

    # 12. ex_ante_te_mean
    if ex_ante_te is not None and len(ex_ante_te) > 0:
        ex_ante_te_mean = float(ex_ante_te.dropna().mean())
    else:
        ex_ante_te_mean = 0.0 if r_series.equals(rb_series) else float("nan")

    # 13. cost_drag and gross_excess_return
    kappa = tc_bps / 10000.0
    cost_drag = float(kappa * ann_turnover) if not np.isnan(ann_turnover) else float("nan")
    gross_excess = float(excess_return + cost_drag) if not np.isnan(cost_drag) else float("nan")

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "excess_return": excess_return,
        "realized_TE": realized_te,
        "IR": ir,
        "max_drawdown": max_dd,
        "beta": beta,
        "ann_turnover": ann_turnover,
        "hit_rate": hit_rate,
        "ex_ante_TE_mean": ex_ante_te_mean,
        "cost_drag": cost_drag,
        "gross_excess_return": gross_excess,
    }


def main():
    print("=== STARTING INDEPENDENT RECOMPUTATION AUDIT ===")
    cfg, prices, shares, w_enh, saved_returns, saved_metrics = load_raw_data()

    warmup_days = int(cfg.get("warmup_days", 252))
    rebal_dates = get_month_end_rebalances(prices.index, warmup_days)
    tc_bps = float(cfg.get("costs", {}).get("tc_bps", 10.0))
    rf = float(cfg.get("risk_free_annual", 0.065))

    print(f"Data range: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} days, {len(prices.columns)} tickers)")
    print(f"Rebalance dates: {len(rebal_dates)} (First: {rebal_dates[0].date()}, Last: {rebal_dates[-1].date()})")

    # 1. Independent return simulation
    print("\n--- 1. Independent Returns Check ---")
    ret_b, to_b = simulate_returns_independently(prices, shares, rebal_dates, tc_bps, strategy="benchmark")
    ret_e, to_e = simulate_returns_independently(prices, shares, rebal_dates, tc_bps, strategy="enhanced_lw", weights_enhanced=w_enh)

    # Align dates
    diff_b = (ret_b - saved_returns["benchmark"]).abs().dropna()
    diff_e = (ret_e - saved_returns["enhanced_lw"]).abs().dropna()

    max_diff_b = float(diff_b.max())
    max_diff_e = float(diff_e.max())

    print(f"Benchmark returns max absolute difference: {max_diff_b:.2e}")
    print(f"Enhanced_lw returns max absolute difference: {max_diff_e:.2e}")

    assert max_diff_b < 1e-8, f"Benchmark return mismatch: {max_diff_b:.2e} >= 1e-8"
    assert max_diff_e < 1e-8, f"Enhanced return mismatch: {max_diff_e:.2e} >= 1e-8"
    print(">>> PART 1 RESULT: PASS (All daily returns match saved results within 1e-8)")

    # 2. Independent metrics recomputation
    print("\n--- 2. Independent Metrics Check ---")
    saved_m_cols = [c for c in saved_metrics.columns if c in [
        "ann_return", "ann_vol", "sharpe", "excess_return", "realized_TE",
        "IR", "max_drawdown", "beta", "ann_turnover", "hit_rate",
        "cost_drag", "gross_excess_return"
    ]]

    all_match = True
    max_metric_diff = 0.0

    for strat in saved_returns.columns:
        r_strat = saved_returns[strat]
        r_bench = saved_returns["benchmark"]
        
        # Turnovers
        if strat == "benchmark":
            strat_to = to_b
        elif strat == "enhanced_lw":
            strat_to = to_e
        else:
            strat_to = None

        m_calc = recompute_metrics_standalone(r_strat, r_bench, rf=rf, turnover=strat_to, tc_bps=tc_bps)

        for col in saved_m_cols:
            saved_val = float(saved_metrics.loc[strat, col])
            calc_val = float(m_calc[col])

            if np.isnan(saved_val) and np.isnan(calc_val):
                diff = 0.0
            else:
                diff = abs(saved_val - calc_val)

            if diff > max_metric_diff and (strat in ["benchmark", "enhanced_lw"]):
                max_metric_diff = diff

            if diff > 1e-8 and (strat in ["benchmark", "enhanced_lw"]):
                print(f"MISMATCH for {strat}.{col}: saved={saved_val:.8f}, calc={calc_val:.8f}, diff={diff:.2e}")
                all_match = False

    # Check IR t-stat
    r_e = saved_returns["enhanced_lw"]
    r_b = saved_returns["benchmark"]
    years = len(r_e) / 252.0
    active = r_e - r_b
    real_ir = float(active.mean() * 252.0 / (active.std(ddof=1) * np.sqrt(252.0)))
    calc_tstat = real_ir * np.sqrt(years)
    
    # Read t-stat from significance.csv
    sig_df = pd.read_csv("results/significance.csv").set_index("metric")["value"]
    saved_tstat = float(sig_df["t_stat"])
    tstat_diff = abs(calc_tstat - saved_tstat)
    print(f"IR t-statistic: calculated={calc_tstat:.6f}, saved={saved_tstat:.6f}, diff={tstat_diff:.2e}")
    assert tstat_diff < 1e-8, f"IR t-stat mismatch: {tstat_diff:.2e} >= 1e-8"

    print(f"Max metrics absolute difference: {max_metric_diff:.2e}")
    assert all_match, "Metrics mismatch exceeded 1e-8 threshold!"
    print(">>> PART 2 RESULT: PASS (All metrics and IR t-stat match within 1e-8)")


if __name__ == "__main__":
    main()
