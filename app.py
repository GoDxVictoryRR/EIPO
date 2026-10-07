"""Interactive Streamlit dashboard for Enhanced Indexing Portfolio Optimization."""

from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ei.backtest import run_all
from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares
from ei.metrics import metrics_table
from ei.synthetic import make_synthetic

st.set_page_config(page_title="Enhanced Indexing Optimizer", layout="wide")

st.title("Enhanced Indexing Portfolio Optimization")
st.markdown(
    "A constrained active management framework tracking cap-weighted benchmarks "
    "within an ex-ante tracking-error budget."
)

cfg = load_config("config.yaml")

# Sidebar parameter controls
st.sidebar.header("Strategy Settings")
te_max = st.sidebar.slider("Ex-Ante TE Limit (te_max)", 0.005, 0.06, float(cfg["optimizer"]["te_max"]), 0.005)
max_active = st.sidebar.slider(
    "Max Active Weight (|w - w_b|)", 0.01, 0.08, float(cfg["optimizer"]["max_active_weight"]), 0.005
)
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 50, int(cfg["costs"]["tc_bps"]), 5)
cov_method = st.sidebar.selectbox("Covariance Estimator", ["ledoit_wolf", "sample"])
use_synthetic = st.sidebar.checkbox("Use Synthetic Data", value=False)

if st.sidebar.button("Run Portfolio Backtest", type="primary"):
    cfg_run = cfg.copy()
    cfg_run["optimizer"]["te_max"] = te_max
    cfg_run["optimizer"]["max_active_weight"] = max_active
    cfg_run["costs"]["tc_bps"] = tc_bps
    cfg_run["covariance"] = cov_method

    with st.spinner("Executing walk-forward monthly rebalancing backtest..."):
        if use_synthetic or not (Path("data") / "prices.csv").is_file():
            prices, shares, sectors = make_synthetic(40, 2500, seed=int(cfg_run["seed"]))
        else:
            prices = load_prices(cfg_run)
            tickers = list(prices.columns)
            shares = load_shares(tickers, cfg_run)
            sectors = load_sectors(tickers, cfg_run)

        results = run_all(prices, shares, sectors, cfg_run)
        table = metrics_table(results, rf=float(cfg_run["risk_free_annual"]))

    st.subheader("Performance and Risk Metrics")
    st.dataframe(table.style.format("{:.4f}"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cumulative Growth of Wealth")
        fig, ax = plt.subplots(figsize=(8, 5))
        for strat, res in results.items():
            wealth = (1.0 + res["returns"]).cumprod()
            ax.plot(wealth.index, wealth.values, label=strat)
        ax.set_ylabel("Growth of $1")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Historical Drawdown")
        fig, ax = plt.subplots(figsize=(8, 5))
        for strat, res in results.items():
            wealth = (1.0 + res["returns"]).cumprod()
            peak = wealth.cummax()
            dd = wealth / peak - 1.0
            ax.plot(dd.index, dd.values, label=strat)
        ax.set_ylabel("Drawdown")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Adjust parameters in the sidebar and click **Run Portfolio Backtest** to explore results.")
