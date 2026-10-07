"""Portfolio performance and risk metrics module."""

import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    r: pd.Series,
    r_b: pd.Series,
    rf: float = 0.065,
    turnover: Optional[pd.Series] = None,
    ex_ante_te: Optional[pd.Series] = None,
    cost_drag_annual: Optional[float] = None,
) -> dict[str, float]:
    """Compute standard portfolio performance and risk metrics per specification.

    Args:
        r: Strategy daily returns.
        r_b: Benchmark daily returns.
        rf: Annual risk-free rate.
        turnover: Series of one-way turnover at each rebalance (for ann_turnover).
        ex_ante_te: Series of ex-ante TE at each rebalance (for ex_ante_TE_mean).
        cost_drag_annual: Pre-computed annualized transaction cost drag (kappa * ann_turnover).
            If provided, cost_drag and gross_excess_return are populated; otherwise NaN.
    """
    r_clean = r.dropna()
    r_b_clean = r_b.reindex(r_clean.index).dropna()
    common_idx = r_clean.index.intersection(r_b_clean.index)
    r_series = r_clean.loc[common_idx]
    rb_series = r_b_clean.loc[common_idx]

    n = len(r_series)
    if n == 0:
        return {
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "excess_return": float("nan"),
            "realized_TE": float("nan"),
            "IR": float("nan"),
            "max_drawdown": float("nan"),
            "beta": float("nan"),
            "ann_turnover": float("nan"),
            "hit_rate": float("nan"),
            "ex_ante_TE_mean": float("nan"),
        }

    years = n / 252.0

    # 1. Annualized return: (prod(1+r))^(252/n) - 1
    total_wealth = float(np.prod(1.0 + r_series.values))
    ann_return = float(total_wealth ** (252.0 / n) - 1.0)

    # 2. Benchmark annualized return
    total_wealth_b = float(np.prod(1.0 + rb_series.values))
    ann_return_b = float(total_wealth_b ** (252.0 / n) - 1.0)

    # 3. Annualized volatility: std(r, ddof=1) * sqrt(252)
    daily_vol = float(r_series.std(ddof=1)) if n > 1 else 0.0
    ann_vol = daily_vol * np.sqrt(252.0)

    # 4. Sharpe ratio: (ann_return - rf) / ann_vol
    if ann_vol > 1e-12:
        sharpe = (ann_return - rf) / ann_vol
    else:
        sharpe = float("nan")

    # 5. Excess return: ann_return(strategy) - ann_return(benchmark)
    excess_return = ann_return - ann_return_b

    # 6. Active series and realized tracking error
    active = r_series - rb_series
    active_std = float(active.std(ddof=1)) if n > 1 else 0.0
    realized_te = active_std * np.sqrt(252.0)

    # 7. Information ratio: mean(active) * 252 / realized_TE
    if realized_te > 1e-12:
        ir = float(active.mean() * 252.0 / realized_te)
    else:
        ir = float("nan")

    # 8. Maximum drawdown: min(wealth / cummax(wealth) - 1)
    wealth_path = np.cumprod(1.0 + r_series.values)
    peaks = np.maximum.accumulate(wealth_path)
    drawdowns = wealth_path / peaks - 1.0
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_drawdown = min(0.0, max_drawdown)

    # 9. Beta: cov(r_p, r_b) / var(r_b)
    if r_series.equals(rb_series):
        beta = 1.0
    else:
        var_b = float(rb_series.var(ddof=1)) if n > 1 else 0.0
        if var_b > 1e-12:
            cov_mat = np.cov(r_series.values, rb_series.values, ddof=1)
            beta = float(cov_mat[0, 1] / var_b)
        else:
            beta = float("nan")

    # 10. Annualized turnover: sum(turnover) / years
    if turnover is not None and len(turnover) > 0:
        total_turnover = float(turnover.sum())
        ann_turnover = total_turnover / max(years, 1.0 / 252.0)
    else:
        ann_turnover = float("nan")

    # 11. Hit rate: fraction of months with positive active return
    if isinstance(r_series.index, pd.DatetimeIndex):
        r_monthly = (1.0 + r_series).resample("ME").prod() - 1.0
        rb_monthly = (1.0 + rb_series).resample("ME").prod() - 1.0
        active_monthly = r_monthly - rb_monthly
        hit_rate = float((active_monthly > 0).mean()) if len(active_monthly) > 0 else float("nan")
    else:
        hit_rate = float("nan")

    # 12. Mean ex-ante tracking error
    if ex_ante_te is not None and len(ex_ante_te) > 0:
        ex_ante_te_clean = ex_ante_te.dropna()
        ex_ante_te_mean = float(ex_ante_te_clean.mean()) if len(ex_ante_te_clean) > 0 else float("nan")
    else:
        ex_ante_te_mean = float("nan")

    # 13. Cost drag: annualized sum of transaction costs = sum(kappa * turnover) / years
    #     kappa = tc_bps / 10000; the backtest charges cost_to_charge on each day after
    #     a rebalance. We cannot recover the exact daily charges here, but we can compute
    #     gross vs net return difference: gross_excess = (gross_ann_return - ann_return_b),
    #     net_excess = excess_return (already computed above).  Instead, estimate cost_drag
    #     from the turnover series: cost_drag = kappa * ann_turnover (approximate, assumes
    #     10 bps default; caller should pass tc_bps for accuracy).
    #     A simpler and exact approach: cost_drag = net_annual_excess - gross_annual_excess
    #     is not available without a gross run. We therefore expose cost_drag as a parameter.
    if cost_drag_annual is not None:
        c_drag = float(cost_drag_annual)
    else:
        c_drag = float("nan")
    gross_excess_return = excess_return + c_drag if not np.isnan(c_drag) else float("nan")

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "excess_return": excess_return,
        "realized_TE": realized_te,
        "IR": ir,
        "max_drawdown": max_drawdown,
        "beta": beta,
        "ann_turnover": ann_turnover,
        "hit_rate": hit_rate,
        "ex_ante_TE_mean": ex_ante_te_mean,
        "cost_drag": c_drag,
        "gross_excess_return": gross_excess_return,
    }


def metrics_table(
    results: dict[str, dict[str, Any]],
    rf: float = 0.065,
    tc_bps: float = 10.0,
) -> pd.DataFrame:
    """Create a formatted metrics table comparing multiple backtest strategies.

    Args:
        results: Dict of strategy name -> backtest result dict.
        rf: Annual risk-free rate.
        tc_bps: Transaction cost in basis points (used to compute cost_drag).
            Defaults to 10 bps. Pass the actual config value for accuracy.
    """
    if "benchmark" not in results:
        raise KeyError("Results dictionary must contain 'benchmark' strategy.")

    kappa = tc_bps / 10000.0
    r_b = results["benchmark"]["returns"]
    rows = {}

    for strat_name, res in results.items():
        r = res["returns"]
        t_over = res.get("turnover")
        ex_te = res.get("ex_ante_te")
        # Compute annualized turnover to derive cost_drag
        n = len(r.dropna())
        years = n / 252.0
        if t_over is not None and len(t_over) > 0:
            ann_to = float(t_over.sum()) / max(years, 1.0 / 252.0)
            cost_drag_annual = kappa * ann_to
        else:
            cost_drag_annual = None
        rows[strat_name] = compute_metrics(
            r, r_b, rf=rf, turnover=t_over, ex_ante_te=ex_te,
            cost_drag_annual=cost_drag_annual,
        )

    cols = [
        "ann_return",
        "ann_vol",
        "sharpe",
        "excess_return",
        "realized_TE",
        "IR",
        "max_drawdown",
        "beta",
        "ann_turnover",
        "hit_rate",
        "ex_ante_TE_mean",
        "cost_drag",
        "gross_excess_return",
    ]
    df = pd.DataFrame.from_dict(rows, orient="index")[cols]
    return df
