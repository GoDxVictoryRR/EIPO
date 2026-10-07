"""app.py - Lightweight, production-ready Streamlit dashboard for Enhanced Indexing.

Zero runtime optimization or network calls: reads purely precomputed artifacts from results/.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Enhanced Indexing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
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

    /* Completely hide sidebar and collapse button since navigation is at top */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* Keyframes for rainbow glow cycle (GPU optimized) */
    @keyframes rainbowGlow {
        0% {
            border-color: #6366f1;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.35);
        }
        20% {
            border-color: #06b6d4;
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.35);
        }
        40% {
            border-color: #10b981;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.35);
        }
        60% {
            border-color: #f59e0b;
            box-shadow: 0 0 12px rgba(245, 158, 11, 0.35);
        }
        80% {
            border-color: #ec4899;
            box-shadow: 0 0 12px rgba(236, 72, 153, 0.35);
        }
        100% {
            border-color: #6366f1;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.35);
        }
    }

    /* Professional glowing card container */
    .glowing-card {
        background-color: #ffffff !important;
        border: 2px solid #6366f1;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 24px;
        animation: rainbowGlow 6s linear infinite;
        will-change: border-color, box-shadow;
    }

    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #6366f1 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        animation: rainbowGlow 6s linear infinite !important;
        will-change: border-color, box-shadow;
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }

    /* Static crisp styling for Dataframe - NEVER animate to prevent canvas re-rasterization lag */
    div[data-testid="stDataFrame"] {
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        overflow: hidden !important;
    }

    /* Plots & image containers */
    div[data-testid="stImage"] {
        border: 2px solid #6366f1 !important;
        border-radius: 14px !important;
        padding: 8px !important;
        background-color: #ffffff !important;
        animation: rainbowGlow 6s linear infinite !important;
        will-change: border-color, box-shadow;
    }

    /* Top Horizontal Navigation Bar Styling */
    div[data-testid="stRadio"] {
        width: 100%;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 12px;
        background-color: #ffffff;
        padding: 8px 16px;
        border: 2px solid #6366f1;
        border-radius: 50px;
        animation: rainbowGlow 6s linear infinite;
        will-change: border-color, box-shadow;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 30px;
        padding: 8px 22px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #1e293b;
        font-weight: 600;
        font-size: 0.95rem;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background-color: #ede9fe;
        border-color: #818cf8;
        color: #4338ca;
        transform: translateY(-1px);
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked),
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background: #0f172a !important;
        color: #ffffff !important;
        border-color: #0f172a !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.25) !important;
    }
    /* Hide radio circle inside pill */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child:not([data-testid="stMarkdownContainer"]) {
        display: none !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] {
        padding-left: 0 !important;
    }

    /* Selectbox styling */
    div[data-baseweb="select"] {
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        transition: border-color 0.2s ease;
    }
    div[data-baseweb="select"]:hover, div[data-baseweb="select"]:focus-within {
        border-color: #6366f1 !important;
    }

    /* Notification / Badge pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        font-size: 0.88rem;
        font-weight: 600;
        border-radius: 9999px;
        border: 1.5px solid #6366f1;
        animation: rainbowGlow 6s linear infinite;
        will-change: border-color, box-shadow;
        background-color: #ffffff;
        color: #1e293b;
        margin-bottom: 16px;
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


# ==============================================================================
# Top Project Heading & Navigation Bar
# ==============================================================================
st.title("Enhanced Indexing Portfolio Optimization (EIPO)")
st.markdown(
    """
    <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-top: -12px; margin-bottom: 20px; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px;'>
        <span style='font-size: 1.02rem; color: #475569; font-weight: 500;'>
            Constrained SOCP Active Risk Budgeting &bull; Nifty 50 Large-Cap Benchmark (2016–2026)
        </span>
        <span class='status-pill' style='margin-bottom: 0;'>
            ⚡ Production Engine &bull; Zero Delay Artifacts
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Horizontal Navigation Bar
page = st.radio(
    "Navigation",
    ["Overview", "Explore", "Sweeps & Costs", "Statistical Significance", "About"],
    key="nav_page",
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<div style='margin-bottom: 18px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# Page 1: Overview
# ==============================================================================
if page == "Overview":
    st.subheader("Performance Overview")
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

    # Visualizations: Precomputed high-resolution artifacts for 0ms lag
    st.markdown("### Cumulative Wealth Growth & Rolling Risk")
    c1, c2 = st.columns(2)

    cum_img_path = RESULTS_DIR / "cumulative_returns.png"
    rolling_img_path = RESULTS_DIR / "rolling_te.png"

    with c1:
        if cum_img_path.is_file():
            st.image(str(cum_img_path), caption="Cumulative Wealth Index: Enhanced vs Benchmark (2016–2026)", use_container_width=True)
        else:
            st.info("Cumulative returns image not found.")

    with c2:
        if rolling_img_path.is_file():
            st.image(str(rolling_img_path), caption="Rolling 1-Year Realized Tracking Error (Target Budget: 3.0%)", use_container_width=True)
        else:
            st.info("Rolling tracking error image not found.")

    with st.expander("📊 Additional Risk Analytics: Drawdowns & Rebalance Turnover"):
        e1, e2 = st.columns(2)
        dd_img_path = RESULTS_DIR / "drawdown.png"
        turnover_img_path = RESULTS_DIR / "turnover.png"
        with e1:
            if dd_img_path.is_file():
                st.image(str(dd_img_path), caption="Historical Underwater Drawdown", use_container_width=True)
        with e2:
            if turnover_img_path.is_file():
                st.image(str(turnover_img_path), caption="Turnover per Monthly Rebalance", use_container_width=True)


# ==============================================================================
# Page 2: Explore Parameter Grid
# ==============================================================================
elif page == "Explore":
    st.subheader("Explore Parameter Sensitivity")
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
            st.markdown("**Net vs Gross Return Degradation**")
            chart_df = sub_grid.set_index("tc_bps")[["excess_return", "gross_excess_return"]].copy() * 100
            chart_df.columns = ["Net Excess (%)", "Gross Excess (%)"]
            st.line_chart(chart_df, height=300)

        with table_col:
            st.markdown("**Sensitivity Grid (tc_bps)**")
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
    st.subheader("Tracking Error Sweeps & Execution Friction Analysis")
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

    # Plot Sweeps: Precomputed high-resolution artifacts for instant loading
    st.markdown("### Information Ratio vs Tracking-Error Budget")
    sw_img1 = RESULTS_DIR / "ir_vs_te.png"
    sw_img2 = RESULTS_DIR / "ir_vs_te_loose.png"
    sw_col1, sw_col2 = st.columns(2)

    with sw_col1:
        if sw_img1.is_file():
            st.image(str(sw_img1), caption="Standard Constraints: IR vs Tracking-Error Budget", use_container_width=True)
        else:
            st.info("Sweep chart (standard) not found.")

    with sw_col2:
        if sw_img2.is_file():
            st.image(str(sw_img2), caption="Loose Constraints: IR vs Tracking-Error Budget", use_container_width=True)
        else:
            st.info("Sweep chart (loose) not found.")

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
    st.subheader("Statistical Significance & Hypothesis Testing")
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
    st.subheader("About Enhanced Indexing & System Metadata")
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


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; font-weight: 500; padding: 12px 0 24px 0;'>"
    "Enhanced Indexing Portfolio Optimization (EIPO) &bull; Production Deployment on Render &bull; Educational project, not investment advice."
    "</div>",
    unsafe_allow_html=True,
)
