"""app.py - Lightweight, production-ready Streamlit dashboard for Enhanced Indexing.

Zero runtime optimization or network calls: reads purely precomputed artifacts from results/.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Enhanced Indexing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Inject Custom CSS: White clean background with colorful glowing cycling borders
st.markdown(
    """
    <style>
    /* Clean white background and high-contrast professional typography */
    .stApp {
        background-color: #ffffff !important;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Keyframes for rainbow glow cycle */
    @keyframes rainbowGlow {
        0% {
            border-color: #6366f1;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.40);
        }
        20% {
            border-color: #06b6d4;
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.40);
        }
        40% {
            border-color: #10b981;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.40);
        }
        60% {
            border-color: #f59e0b;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.40);
        }
        80% {
            border-color: #ec4899;
            box-shadow: 0 0 10px rgba(236, 72, 153, 0.40);
        }
        100% {
            border-color: #6366f1;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.40);
        }
    }

    /* Professional glowing card container */
    .glowing-card {
        background-color: #ffffff !important;
        border: 2px solid #6366f1;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        animation: rainbowGlow 6s linear infinite;
    }

    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #6366f1 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        animation: rainbowGlow 6s linear infinite !important;
    }

    /* Dataframe and table containers */
    div[data-testid="stDataFrame"] {
        border: 2px solid #6366f1 !important;
        border-radius: 12px !important;
        padding: 4px !important;
        background-color: #ffffff !important;
        animation: rainbowGlow 6s linear infinite !important;
    }

    /* Plots & image containers */
    div[data-testid="stImage"], .stPlot {
        border: 2px solid #6366f1 !important;
        border-radius: 12px !important;
        padding: 8px !important;
        background-color: #ffffff !important;
        animation: rainbowGlow 6s linear infinite !important;
    }

    /* Selectbox & controls container */
    div[data-baseweb="select"] {
        border: 1.5px solid #6366f1 !important;
        border-radius: 8px !important;
        animation: rainbowGlow 6s linear infinite !important;
    }

    /* Sidebar container styling */
    section[data-testid="stSidebar"] {
        background-color: #fafbfc !important;
        border-right: 2px solid #6366f1 !important;
        animation: rainbowGlow 6s linear infinite !important;
    }

    /* Notification / Badge pill */
    .status-pill {
        display: inline-block;
        padding: 5px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 9999px;
        border: 1.5px solid #6366f1;
        animation: rainbowGlow 6s linear infinite;
        background-color: #ffffff;
        color: #1e293b;
        margin-bottom: 12px;
    }

    /* Professional typography */
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    p, span, label {
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv_artifact(filename: str) -> pd.DataFrame:
    """Load a CSV file from results/ with caching and graceful error handling."""
    filepath = RESULTS_DIR / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing required artifact: {filepath.name}")
    return pd.read_csv(filepath)


@st.cache_data
def load_metadata_artifact() -> dict[str, Any]:
    """Load run_metadata.json with caching and graceful error handling."""
    filepath = RESULTS_DIR / "run_metadata.json"
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing required artifact: {filepath.name}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select View",
    ["Overview", "Explore", "Sweeps & Costs", "Statistical Significance", "About"],
    key="nav_page",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 0.8rem; color: #64748b;'>EIPO Production Deployment<br>Educational project, not investment advice.</div>",
    unsafe_allow_html=True,
)


# ==============================================================================
# Page 1: Overview
# ==============================================================================
if page == "Overview":
    st.title("Enhanced Indexing Portfolio Optimization (EIPO)")
    st.markdown(
        "<div class='status-pill'>Production Benchmark & Enhanced Index Performance (2016–2026)</div>",
        unsafe_allow_html=True,
    )

    try:
        metrics_df = load_csv_artifact("metrics.csv")
        returns_df = load_csv_artifact("returns.csv")
    except FileNotFoundError as err:
        st.error(f"Error loading required results files: {err}")
        st.stop()

    # Index by strategy name
    if "Unnamed: 0" in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={"Unnamed: 0": "strategy"})
    if "strategy" in metrics_df.columns:
        metrics_df = metrics_df.set_index("strategy")

    # Headline KPIs
    col1, col2, col3, col4 = st.columns(4)

    if "benchmark" in metrics_df.index:
        bm = metrics_df.loc["benchmark"]
        with col1:
            st.metric(
                label="Benchmark (Cap-Weighted)",
                value=f"{bm['ann_return'] * 100:.2f}%",
                help="Cap-weighted Nifty 50 proxy annual return",
            )
            st.caption(f"Vol: {bm['ann_vol'] * 100:.2f}% | Sharpe: {bm['sharpe']:.2f}")

    if "enhanced_lw" in metrics_df.index:
        lw = metrics_df.loc["enhanced_lw"]
        with col2:
            st.metric(
                label="Enhanced (Ledoit-Wolf)",
                value=f"{lw['ann_return'] * 100:.2f}%",
                delta=f"+{lw['excess_return'] * 100:.2f}% vs BM",
                help="Annual return and net excess return after 10 bps trading costs",
            )
            st.caption(f"IR: {lw['IR']:.3f} | Realized TE: {lw['realized_TE'] * 100:.2f}%")

    if "enhanced_sample" in metrics_df.index:
        smp = metrics_df.loc["enhanced_sample"]
        with col3:
            st.metric(
                label="Enhanced (Sample Cov)",
                value=f"{smp['ann_return'] * 100:.2f}%",
                delta=f"+{smp['excess_return'] * 100:.2f}% vs BM",
            )
            st.caption(f"IR: {smp['IR']:.3f} | Realized TE: {smp['realized_TE'] * 100:.2f}%")

    if "mv" in metrics_df.index:
        mv = metrics_df.loc["mv"]
        with col4:
            st.metric(
                label="Mean-Variance (Unconstrained)",
                value=f"{mv['ann_return'] * 100:.2f}%",
                delta=f"+{mv['excess_return'] * 100:.2f}% vs BM",
            )
            st.caption(f"Vol: {mv['ann_vol'] * 100:.2f}% | Beta: {mv['beta']:.2f}")

    st.markdown("### Performance & Risk Metrics Table")
    display_df = metrics_df.copy()
    display_cols = [
        "ann_return",
        "ann_vol",
        "sharpe",
        "excess_return",
        "gross_excess_return",
        "cost_drag",
        "realized_TE",
        "IR",
        "ann_turnover",
        "max_drawdown",
        "beta",
        "hit_rate",
    ]
    present_cols = [c for c in display_cols if c in display_df.columns]
    formatted_df = display_df[present_cols].copy()

    # Format percentages and decimals
    pct_cols = ["ann_return", "ann_vol", "excess_return", "gross_excess_return", "cost_drag", "realized_TE", "max_drawdown", "hit_rate"]
    for c in pct_cols:
        if c in formatted_df.columns:
            formatted_df[c] = formatted_df[c].apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—")

    dec_cols = ["sharpe", "IR", "ann_turnover", "beta"]
    for c in dec_cols:
        if c in formatted_df.columns:
            formatted_df[c] = formatted_df[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    st.dataframe(formatted_df, width="stretch")

    # Visualizations
    st.markdown("### Cumulative Wealth Growth & Rolling Risk")
    c1, c2 = st.columns(2)

    with c1:
        if "date" in returns_df.columns:
            returns_df["date"] = pd.to_datetime(returns_df["date"])
            r_indexed = returns_df.set_index("date")
        else:
            r_indexed = returns_df.copy()

        fig1, ax1 = plt.subplots(figsize=(7, 4.2), facecolor="white")
        for col in ["benchmark", "enhanced_lw", "enhanced_sample", "mv"]:
            if col in r_indexed.columns:
                wealth = (1.0 + r_indexed[col].dropna()).cumprod()
                ax1.plot(wealth.index, wealth.values, label=col, linewidth=1.8)

        ax1.set_title("Growth of ₹1.00 Invested (2016–2026)", fontsize=11, fontweight="bold", pad=10)
        ax1.set_ylabel("Wealth Index", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(7, 4.2), facecolor="white")
        if "enhanced_lw" in r_indexed.columns and "benchmark" in r_indexed.columns:
            active = r_indexed["enhanced_lw"] - r_indexed["benchmark"]
            rolling_te = active.rolling(252).std(ddof=1) * np.sqrt(252)
            ax2.plot(rolling_te.index, rolling_te.values, color="#4f46e5", label="252d Realized TE", linewidth=1.8)
            ax2.axhline(0.03, color="#ef4444", linestyle="--", linewidth=1.5, label="Target Budget (3%)")

        ax2.set_title("Rolling 1-Year Realized Tracking Error", fontsize=11, fontweight="bold", pad=10)
        ax2.set_ylabel("Annualized TE", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


# ==============================================================================
# Page 2: Explore Parameter Grid
# ==============================================================================
elif page == "Explore":
    st.title("Explore Parameter Sensitivity")
    st.markdown(
        "<div class='status-pill'>Precomputed from the backtest; not a live run.</div>",
        unsafe_allow_html=True,
    )

    try:
        grid_df = load_csv_artifact("grid.csv")
    except FileNotFoundError as err:
        st.error(f"Error loading parameter grid: {err}")
        st.stop()

    st.markdown(
        "Explore how varying the tracking error limit ($\\text{TE}_{\\max}$), covariance shrinkage method, "
        "and transaction cost assumptions affect portfolio returns and turnover. "
        "All 40 parameter permutations are precomputed deterministically."
    )

    col_sel1, col_sel2, col_sel3 = st.columns(3)

    te_options = sorted(grid_df["te_max"].unique())
    cov_options = sorted(grid_df["covariance"].unique())
    tc_options = sorted(grid_df["tc_bps"].unique())

    with col_sel1:
        sel_te = st.selectbox(
            "Tracking Error Budget (te_max)",
            te_options,
            index=te_options.index(0.03) if 0.03 in te_options else 0,
            help="Annual ex-ante tracking error limit",
        )

    with col_sel2:
        sel_cov = st.selectbox(
            "Covariance Estimator",
            cov_options,
            index=cov_options.index("ledoit_wolf") if "ledoit_wolf" in cov_options else 0,
            help="Ledoit-Wolf shrinkage vs Sample covariance",
        )

    with col_sel3:
        sel_tc = st.selectbox(
            "Transaction Cost (bps)",
            tc_options,
            index=tc_options.index(10) if 10 in tc_options else 0,
            help="One-way turnover trading cost in basis points",
        )

    # Lookup matching row
    mask = (grid_df["te_max"] == sel_te) & (grid_df["covariance"] == sel_cov) & (grid_df["tc_bps"] == sel_tc)
    matched = grid_df[mask]

    if matched.empty:
        st.warning("No precomputed record found for this combination.")
    else:
        row = matched.iloc[0]

        st.markdown("#### Selected Configuration Performance")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Net Annual Return", f"{row['ann_return'] * 100:.2f}%")
            st.caption(f"Gross: {row['gross_excess_return'] * 100:.2f}%")
        with m2:
            st.metric("Net Excess Return", f"{row['excess_return'] * 100:.2f}%")
            st.caption(f"Cost Drag: {row['cost_drag'] * 100:.2f}%")
        with m3:
            st.metric("Information Ratio", f"{row['IR']:.3f}")
            st.caption(f"Realized TE: {row['realized_TE'] * 100:.2f}%")
        with m4:
            st.metric("Annual Turnover", f"{row['ann_turnover']:.2f}x")
            st.caption(f"Cost: {row['tc_bps']} bps")

        st.markdown("---")
        st.markdown(f"#### Cost Sensitivity at $\\text{{TE}}_{{\\max}} = {sel_te:.3f}$, Covariance = `{sel_cov}`")

        # Slice grid across all tc_bps for the selected te_max and covariance
        sub_grid = grid_df[(grid_df["te_max"] == sel_te) & (grid_df["covariance"] == sel_cov)].sort_values("tc_bps")

        chart_col, table_col = st.columns([1, 1])

        with chart_col:
            fig, ax = plt.subplots(figsize=(6, 3.8), facecolor="white")
            ax.plot(sub_grid["tc_bps"], sub_grid["excess_return"] * 100, marker="o", color="#4f46e5", label="Net Excess (%)", linewidth=2)
            ax.plot(sub_grid["tc_bps"], sub_grid["gross_excess_return"] * 100, linestyle="--", marker="s", color="#10b981", label="Gross Excess (%)", linewidth=1.5)
            ax.set_xlabel("Transaction Cost (bps)", fontsize=10)
            ax.set_ylabel("Excess Return (%)", fontsize=10)
            ax.set_title("Net vs Gross Return Degradation", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with table_col:
            display_sub = sub_grid.copy()
            for col in ["ann_return", "excess_return", "gross_excess_return", "cost_drag", "realized_TE"]:
                display_sub[col] = display_sub[col].apply(lambda x: f"{x * 100:.2f}%")
            display_sub["IR"] = display_sub["IR"].apply(lambda x: f"{x:.3f}")
            display_sub["ann_turnover"] = display_sub["ann_turnover"].apply(lambda x: f"{x:.2f}")
            st.dataframe(display_sub, width="stretch")


# ==============================================================================
# Page 3: Sweeps & Costs
# ==============================================================================
elif page == "Sweeps & Costs":
    st.title("Tracking Error Sweeps & Execution Friction Analysis")
    st.markdown(
        "<div class='status-pill'>Ex-Ante Risk Budgets & Constraint Saturation</div>",
        unsafe_allow_html=True,
    )

    try:
        sweep_df = load_csv_artifact("sweep.csv")
        sweep_loose_df = load_csv_artifact("sweep_loose.csv")
        cost_df = load_csv_artifact("cost_sensitivity.csv")
    except FileNotFoundError as err:
        st.error(f"Error loading sweep data: {err}")
        st.stop()

    st.markdown("### 1. Tracking Error Budget Sweep")
    st.markdown(
        "Demonstrates how realized tracking error and information ratios scale with $\\text{TE}_{\\max}$. "
        "In the standard setup (`max_active=0.03`, `sector_dev=0.05`), linear constraints cause risk saturation at $\\text{TE}_{\\max} \\approx 0.032$. "
        "When limits are loosened (`max_active=0.10`, `sector_dev=0.15`), risk saturation moves to $\\text{TE}_{\\max} \\approx 0.038$."
    )

    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown("#### Standard Constraints (`sweep.csv`)")
        st.dataframe(
            sweep_df.style.format({
                "ann_return": "{:.2%}",
                "excess": "{:.2%}",
                "realized_TE": "{:.2%}",
                "IR": "{:.3f}",
                "turnover": "{:.2f}",
            }),
            width="stretch",
        )

    with sc2:
        st.markdown("#### Loose Constraints (`sweep_loose.csv`)")
        st.dataframe(
            sweep_loose_df.style.format({
                "ann_return": "{:.2%}",
                "excess": "{:.2%}",
                "realized_TE": "{:.2%}",
                "IR": "{:.3f}",
                "turnover": "{:.2f}",
            }),
            width="stretch",
        )

    # Plot Sweeps
    fig_sw, (ax_sw1, ax_sw2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="white")

    ax_sw1.plot(sweep_df["te_max"] * 100, sweep_df["realized_TE"] * 100, marker="o", color="#4f46e5", label="Standard Limits", linewidth=2)
    ax_sw1.plot(sweep_loose_df["te_max"] * 100, sweep_loose_df["realized_TE"] * 100, marker="s", color="#06b6d4", label="Loose Limits", linewidth=2)
    ax_sw1.plot([0.5, 6.0], [0.5, 6.0], linestyle=":", color="#94a3b8", label="1:1 Budget Line")
    ax_sw1.set_xlabel("Ex-Ante TE Limit (%)", fontsize=10)
    ax_sw1.set_ylabel("Realized TE (%)", fontsize=10)
    ax_sw1.set_title("Realized vs Ex-Ante Tracking Error", fontsize=11, fontweight="bold")
    ax_sw1.grid(True, linestyle="--", alpha=0.35)
    ax_sw1.legend(fontsize=9)

    ax_sw2.plot(sweep_df["te_max"] * 100, sweep_df["IR"], marker="o", color="#4f46e5", label="Standard Limits", linewidth=2)
    ax_sw2.plot(sweep_loose_df["te_max"] * 100, sweep_loose_df["IR"], marker="s", color="#06b6d4", label="Loose Limits", linewidth=2)
    ax_sw2.set_xlabel("Ex-Ante TE Limit (%)", fontsize=10)
    ax_sw2.set_ylabel("Information Ratio (IR)", fontsize=10)
    ax_sw2.set_title("Information Ratio Across TE Budgets", fontsize=11, fontweight="bold")
    ax_sw2.grid(True, linestyle="--", alpha=0.35)
    ax_sw2.legend(fontsize=9)

    fig_sw.tight_layout()
    st.pyplot(fig_sw)
    plt.close(fig_sw)

    st.markdown("### 2. Transaction Cost Sensitivity (`cost_sensitivity.csv`)")
    st.dataframe(
        cost_df.style.format({
            "IR": "{:.3f}",
            "excess_return": "{:.2%}",
            "ann_turnover": "{:.2f}",
            "cost_drag": "{:.2%}",
            "gross_excess_return": "{:.2%}",
            "ann_return": "{:.2%}",
            "realized_TE": "{:.2%}",
        }),
        width="stretch",
    )


# ==============================================================================
# Page 4: Statistical Significance
# ==============================================================================
elif page == "Statistical Significance":
    st.title("Statistical Significance & Hypothesis Testing")
    st.markdown(
        "<div class='status-pill'>Rigorous Significance Tests & Placebo Monte Carlo</div>",
        unsafe_allow_html=True,
    )

    try:
        sig_df = load_csv_artifact("significance.csv")
        sub_df = load_csv_artifact("significance_subperiod.csv")
    except FileNotFoundError as err:
        st.error(f"Error loading significance results: {err}")
        st.stop()

    sig_map = dict(zip(sig_df["metric"], sig_df["value"]))

    t_stat = float(sig_map.get("t_stat", 0.0))
    ir_boot_lo = float(sig_map.get("ir_boot_CI_lo", 0.0))
    ir_boot_hi = float(sig_map.get("ir_boot_CI_hi", 0.0))
    noise_p = float(sig_map.get("placebo_p_value", 0.0))
    perm_p = float(sig_map.get("perm_p_value", 0.0))
    real_ir = float(sig_map.get("real_IR", 0.0))

    # Metric KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Information Ratio", f"{real_ir:.3f}")
        st.caption(f"Sample: {float(sig_map.get('years', 10.4)):.1f} Years")
    with c2:
        st.metric("IR t-Statistic", f"{t_stat:.2f}")
        st.caption("Required: |t| > 2.0")
    with c3:
        st.metric("Bootstrap 95% CI", f"[{ir_boot_lo:.3f}, {ir_boot_hi:.3f}]")
        st.caption("Includes 0: Non-significant")
    with c4:
        st.metric("Permutation Placebo p", f"{perm_p:.3f}")
        st.caption(f"Noise Placebo p: {noise_p:.3f}")

    # Required Verdict Text Callout
    ci_includes_zero = ir_boot_lo <= 0 <= ir_boot_hi
    verdict_text = (
        "Verdict: not statistically significant at 5% when the bootstrap CI includes 0."
        if ci_includes_zero
        else "Verdict: statistically significant at 5%."
    )

    st.markdown(
        f"""
        <div class='glowing-card'>
            <h3 style='margin-top:0;'>Statistical Assessment</h3>
            <p><strong>{verdict_text}</strong></p>
            <p>The noise-alpha placebo is biased in favor of the real strategy because random alphas trade more and pay more costs; the label-permutation placebo is the primary test.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Sub-Period Stability Breakdown")
    st.markdown(
        "Performance split into equal 5-year chronological halves and individual calendar years. "
        "While full-sample excess returns are positive, outperformance was concentrated during 2017–2021."
    )
    st.dataframe(
        sub_df.style.format({
            "excess": "{:.2%}",
            "IR": "{:.3f}",
            "realized_TE": "{:.2%}",
        }),
        width="stretch",
    )


# ==============================================================================
# Page 5: About & Metadata
# ==============================================================================
elif page == "About":
    st.title("About Enhanced Indexing & System Metadata")
    st.markdown(
        "<div class='status-pill'>System Architecture & Methodology Disclosures</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Methodology
        **Enhanced Indexing** is a disciplined quantitative strategy that aims to outperform a cap-weighted benchmark 
        while strictly controlling active risk. Rather than taking unconstrained concentrated bets, the portfolio optimizer solves 
        a Second-Order Cone Program (SOCP) at every monthly rebalance:
        - **Objective**: Maximize alpha exposure penalized for active variance and turnover frictions.
        - **Constraints**: Long-only ($w_i \\ge 0$, $\\sum w_i = 1$), individual active stock bounds ($|w_i - w_{b,i}| \\le 3\\%$), 
          sector deviation bounds ($|\\sum_{i \\in s} (w_i - w_{b,i})| \\le 5\\%$), and an annual ex-ante tracking error budget ($\\le 3\\%$).

        ---
        ### Known Model Limitations
        1. **Survivorship Bias**: The asset universe is constructed from modern constituents of the Nifty 50 large-cap proxy. 
           Firms that were dropped, merged, or liquidated prior to 2026 are not captured.
        2. **Static Shares Outstanding**: Market-capitalization proxy weights utilize static `sharesOutstanding` cached from Yahoo Finance, 
           omitting historical float adjustments and corporate share count modifications.
        3. **Execution Cost Model**: Linear 10 bps trading cost assumes constant liquidity. Real executions incur non-uniform bid-ask spreads, 
           square-root market impact for large AUM, and statutory taxes (Securities Transaction Tax / Stamp Duty).
        4. **Yahoo Adjusted-Price Snapshot Date**: Historical dividend and split adjustments reflect Yahoo Finance total return series as of the download date.
        """
    )

    st.markdown("### Run Metadata")
    try:
        meta = load_metadata_artifact()
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.json({
                "generated_at": meta.get("generated_at"),
                "config_hash_md5": meta.get("config_hash_md5_8"),
                "data_end_date": meta.get("data_end_date"),
                "n_tickers": meta.get("n_tickers"),
            })
        with m_col2:
            st.write(f"**Constituent Universe ({meta.get('n_tickers', 41)} Tickers):**")
            st.write(", ".join(meta.get("tickers", [])))
    except FileNotFoundError:
        st.info("run_metadata.json not found in results directory.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; font-weight: 600; padding: 12px;'>"
        "Educational project, not investment advice."
        "</div>",
        unsafe_allow_html=True,
    )
