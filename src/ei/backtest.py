"""Backtesting engine for enhanced indexing and benchmark strategies."""

import copy
import logging
from typing import Any
import numpy as np
import pandas as pd

from ei.data import cap_weights
from ei.optimizer import optimize_enhanced, optimize_mv
from ei.risk import estimate_cov
from ei.signals import compute_alpha

logger = logging.getLogger(__name__)


def get_rebalance_dates(index: pd.DatetimeIndex, warmup_days: int) -> list[pd.Timestamp]:
    """Return strictly increasing list of month-end trading days occurring after warmup_days."""
    if len(index) <= warmup_days:
        return []

    s = pd.Series(index, index=index)
    month_ends = s.groupby([index.year, index.month]).last().values

    min_date = index[warmup_days]
    rebal_dates = [pd.Timestamp(d) for d in month_ends if d >= min_date]
    return sorted(rebal_dates)


def run_backtest(
    prices: pd.DataFrame,
    shares: pd.Series,
    sectors: pd.Series,
    cfg: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    """Run walk-forward monthly rebalanced portfolio backtest."""
    warmup_days = int(cfg.get("warmup_days", 252))
    rebal_dates = get_rebalance_dates(prices.index, warmup_days)
    if len(rebal_dates) == 0:
        raise ValueError("No rebalance dates available after warmup_days.")

    cov_method = cfg.get("covariance", "ledoit_wolf")
    if strategy == "enhanced_sample":
        strat_type = "enhanced"
        cov_method = "sample"
    elif strategy == "enhanced_lw":
        strat_type = "enhanced"
        cov_method = "ledoit_wolf"
    else:
        strat_type = strategy

    tickers = list(prices.columns)
    window = int(cfg.get("estimation_window", 252))
    tc_bps = float(cfg["costs"].get("tc_bps", 10.0))
    kappa = tc_bps / 10000.0

    weights_dict: dict[pd.Timestamp, pd.Series] = {}
    turnover_dict: dict[pd.Timestamp, float] = {}
    ex_ante_te_dict: dict[pd.Timestamp, float] = {}
    solver_status_dict: dict[pd.Timestamp, dict[str, Any]] = {}

    # Initial rebalance at t0
    t0 = rebal_dates[0]
    p_upto_t0 = prices.loc[:t0]
    rets_upto_t0 = p_upto_t0.pct_change().dropna()
    w_b0 = cap_weights(prices, shares, t0).reindex(tickers).fillna(0.0)

    if strat_type == "benchmark":
        w_target = w_b0.values
        ex_ante_te = 0.0
        status_curr = "optimal"
        solver_curr = "analytical"
    elif strat_type == "enhanced":
        sigma_ann, l_factor = estimate_cov(rets_upto_t0.iloc[-window:], method=cov_method)
        alpha = compute_alpha(p_upto_t0, sigma_ann, cfg).reindex(tickers).values
        w_target, info = optimize_enhanced(
            alpha, w_b0.values, w_b0.values, sigma_ann, l_factor, sectors.reindex(tickers).values, cfg
        )
        ex_ante_te = info.get("ex_ante_te", 0.0)
        status_curr = info.get("status", "unknown")
        solver_curr = info.get("solver", "unknown")
    elif strat_type == "mv":
        sigma_ann, _ = estimate_cov(rets_upto_t0.iloc[-window:], method=cov_method)
        alpha = compute_alpha(p_upto_t0, sigma_ann, cfg).reindex(tickers).values
        w_target, info = optimize_mv(alpha, w_b0.values, sigma_ann, cfg)
        a = w_target - w_b0.values
        ex_ante_te = float(np.sqrt(np.maximum(0.0, a @ sigma_ann @ a)))
        status_curr = info.get("status", "unknown")
        solver_curr = info.get("solver", "unknown")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if np.any(np.isnan(w_target)) or np.any(np.isinf(w_target)):
        raise ValueError(f"NaN or Inf in initial target weights at date {t0}")

    initial_turnover = float(np.sum(np.abs(w_target)))  # Measured vs all-cash (w_prev=0)
    weights_dict[t0] = pd.Series(w_target, index=tickers)
    turnover_dict[t0] = initial_turnover
    ex_ante_te_dict[t0] = ex_ante_te
    solver_status_dict[t0] = {"status": status_curr, "solver": solver_curr}

    # Daily simulation starting from trading day after t0
    all_dates = prices.index
    t0_idx = all_dates.get_loc(t0)
    eval_dates = all_dates[t0_idx + 1 :]

    daily_returns = []
    return_dates = []

    w_current = np.copy(w_target)
    cost_to_charge = kappa * initial_turnover

    rebal_set = set(rebal_dates)

    for i, d in enumerate(eval_dates):
        prev_date = all_dates[t0_idx + i]
        p_today = prices.loc[d].values
        p_yesterday = prices.loc[prev_date].values
        r_d = (p_today - p_yesterday) / p_yesterday

        gross_ret = float(np.sum(w_current * r_d))
        net_ret = gross_ret - cost_to_charge
        daily_returns.append(net_ret)
        return_dates.append(d)

        # Drift weights to end of day d
        wealth_denom = np.sum(w_current * (1.0 + r_d))
        if wealth_denom > 0:
            w_drifted = (w_current * (1.0 + r_d)) / wealth_denom
        else:
            w_drifted = np.copy(w_current)

        # Check if d is a rebalance date
        if d in rebal_set and d != t0:
            p_upto_d = prices.loc[:d]
            rets_upto_d = p_upto_d.pct_change().dropna()
            w_bd = cap_weights(prices, shares, d).reindex(tickers).fillna(0.0)

            if strat_type == "benchmark":
                w_new_target = w_bd.values
                te_curr = 0.0
                status_curr = "optimal"
                solver_curr = "analytical"
            elif strat_type == "enhanced":
                sigma_ann, l_factor = estimate_cov(rets_upto_d.iloc[-window:], method=cov_method)
                alpha = compute_alpha(p_upto_d, sigma_ann, cfg).reindex(tickers).values
                w_new_target, info = optimize_enhanced(
                    alpha, w_bd.values, w_drifted, sigma_ann, l_factor, sectors.reindex(tickers).values, cfg
                )
                te_curr = info.get("ex_ante_te", 0.0)
                status_curr = info.get("status", "unknown")
                solver_curr = info.get("solver", "unknown")
            elif strat_type == "mv":
                sigma_ann, _ = estimate_cov(rets_upto_d.iloc[-window:], method=cov_method)
                alpha = compute_alpha(p_upto_d, sigma_ann, cfg).reindex(tickers).values
                w_new_target, info = optimize_mv(alpha, w_drifted, sigma_ann, cfg)
                a = w_new_target - w_bd.values
                te_curr = float(np.sqrt(np.maximum(0.0, a @ sigma_ann @ a)))
                status_curr = info.get("status", "unknown")
                solver_curr = info.get("solver", "unknown")

            if np.any(np.isnan(w_new_target)) or np.any(np.isinf(w_new_target)):
                raise ValueError(f"NaN or Inf in target weights at date {d}")

            t_over = float(np.sum(np.abs(w_new_target - w_drifted)))
            weights_dict[d] = pd.Series(w_new_target, index=tickers)
            turnover_dict[d] = t_over
            ex_ante_te_dict[d] = te_curr
            solver_status_dict[d] = {"status": status_curr, "solver": solver_curr}

            cost_to_charge = kappa * t_over
            w_current = np.copy(w_new_target)
        else:
            cost_to_charge = 0.0
            w_current = np.copy(w_drifted)

    returns_series = pd.Series(daily_returns, index=return_dates, name=strategy)
    weights_df = pd.DataFrame.from_dict(weights_dict, orient="index")
    turnover_series = pd.Series(turnover_dict)
    ex_ante_te_series = pd.Series(ex_ante_te_dict)
    solver_status_df = pd.DataFrame.from_dict(solver_status_dict, orient="index")

    return {
        "returns": returns_series,
        "weights": weights_df,
        "turnover": turnover_series,
        "ex_ante_te": ex_ante_te_series,
        "solver_status": solver_status_df,
    }


def run_all(
    prices: pd.DataFrame,
    shares: pd.Series,
    sectors: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run backtests for benchmark, enhanced (Ledoit-Wolf & sample), and mean-variance baseline."""
    logger.info("Running benchmark backtest...")
    res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")

    logger.info("Running enhanced_lw backtest...")
    res_elw = run_backtest(prices, shares, sectors, cfg, strategy="enhanced_lw")

    logger.info("Running enhanced_sample backtest...")
    res_esample = run_backtest(prices, shares, sectors, cfg, strategy="enhanced_sample")

    logger.info("Running mean-variance baseline backtest...")
    res_mv = run_backtest(prices, shares, sectors, cfg, strategy="mv")

    return {
        "benchmark": res_b,
        "enhanced_lw": res_elw,
        "enhanced_sample": res_esample,
        "mv": res_mv,
    }
