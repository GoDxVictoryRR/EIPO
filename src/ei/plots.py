"""Plotting routines using Matplotlib."""

import logging
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_cumulative_returns(
    results: dict[str, dict[str, Any]],
    out_path: str = "results/cumulative_returns.png",
) -> None:
    """Plot cumulative growth of wealth across strategies vs benchmark."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for strat_name, res in results.items():
        r = res["returns"]
        wealth = (1.0 + r).cumprod()
        ax.plot(wealth.index, wealth.values, label=strat_name, linewidth=1.5)

    ax.set_title("Cumulative Returns Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved cumulative returns plot to %s", out_path)


def plot_rolling_te(
    r_enhanced: pd.Series,
    r_benchmark: pd.Series,
    te_max: float,
    window: int = 252,
    out_path: str = "results/rolling_te.png",
) -> None:
    """Plot rolling realized tracking error vs ex-ante te_max limit."""
    common = r_enhanced.index.intersection(r_benchmark.index)
    active = r_enhanced.loc[common] - r_benchmark.loc[common]

    min_p = min(len(active), max(20, window // 4))
    rolling_std = active.rolling(window, min_periods=min_p).std(ddof=1)
    rolling_te = rolling_std * np.sqrt(252.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rolling_te.index, rolling_te.values, label=f"Rolling {window}d Realized TE", color="tab:blue")
    ax.axhline(te_max, color="tab:red", linestyle="--", linewidth=1.5, label=f"te_max Limit ({te_max:.1%})")

    ax.set_title(f"Rolling Realized Tracking Error (Window: {window} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Tracking Error")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved rolling TE plot to %s", out_path)


def plot_active_weights_last(
    w_enhanced_last: pd.Series,
    w_b_last: pd.Series,
    out_path: str = "results/active_weights_last.png",
) -> None:
    """Plot bar chart of top 15 active weights at the final rebalance date."""
    common = w_enhanced_last.index.intersection(w_b_last.index)
    active = w_enhanced_last.loc[common] - w_b_last.loc[common]

    top_idx = active.abs().nlargest(min(15, len(active))).index
    top_active = active.loc[top_idx].sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:green" if x >= 0 else "tab:red" for x in top_active.values]
    ax.barh(top_active.index, top_active.values, color=colors, alpha=0.85)

    ax.set_title("Top Active Weights at Final Rebalance (w - w_b)")
    ax.set_xlabel("Active Weight")
    ax.grid(True, linestyle="--", alpha=0.6)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved active weights plot to %s", out_path)


def plot_drawdown(
    results: dict[str, dict[str, Any]],
    out_path: str = "results/drawdown.png",
) -> None:
    """Plot drawdown curves across strategies."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for strat_name, res in results.items():
        r = res["returns"]
        wealth = (1.0 + r).cumprod()
        peak = wealth.cummax()
        dd = wealth / peak - 1.0
        ax.plot(dd.index, dd.values, label=strat_name, linewidth=1.2)

    ax.set_title("Historical Underwater Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved drawdown plot to %s", out_path)


def plot_turnover(
    results: dict[str, dict[str, Any]],
    out_path: str = "results/turnover.png",
) -> None:
    """Plot rebalance turnover series across strategies."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for strat_name, res in results.items():
        turnover = res.get("turnover")
        if turnover is not None and len(turnover) > 0:
            ax.plot(turnover.index, turnover.values, marker="o", markersize=3, label=strat_name)

    ax.set_title("Turnover per Rebalance Date")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Traded Turnover")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved turnover plot to %s", out_path)


def plot_ir_vs_te(
    sweep_df: pd.DataFrame,
    out_path: str = "results/ir_vs_te.png",
) -> None:
    """Plot Information Ratio (IR) vs Tracking Error limit from parameter sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sweep_df["te_max"], sweep_df["IR"], marker="s", color="darkblue", linewidth=1.8, label="IR vs te_max")

    ax.set_title("Information Ratio vs Tracking-Error Budget")
    ax.set_xlabel("Ex-Ante Tracking-Error Limit (te_max)")
    ax.set_ylabel("Information Ratio (IR)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved IR vs TE plot to %s", out_path)
