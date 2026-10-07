# Enhanced-Indexing Portfolio Optimization (EIPO)

An end-to-end, mathematically rigorous enhanced indexing engine and backtester for Indian large-cap equities (Nifty 50 proxy).

The system constructs a monthly-rebalanced, long-only enhanced index portfolio designed to track a cap-weighted benchmark within an annual tracking-error budget ($\text{TE}_{\max} = 3\%$) while tilting active weights toward multi-factor alpha signals (momentum, short-term reversal, and low volatility).

> **Disclaimer**: Educational project, not investment advice.

---

## 1. Project Overview & Architecture

### Core Components
- **Data Ingestion & Cleaning (`src/ei/data.py`, `src/ei/universe.py`)**: Fetches adjusted daily closes for 41 Indian large-cap equities, intersects trading dates with the `^NSEI` index calendar (removing holiday mismatches), and computes cap-weighted benchmark weights.
- **Factor Signals (`src/ei/signals.py`)**: Computes 12-1 price momentum, 1-month short-term reversal, and 1-year low volatility signals. Standardizes via winsorized cross-sectional z-scores ($|z| \le 3.0$), combines them, and scales by $\text{IC} = 0.05$ and asset volatility.
- **Risk Model (`src/ei/risk.py`)**: Computes annualized covariance matrices using Ledoit-Wolf shrinkage (with sample covariance comparison) and verifies positive semi-definiteness via Cholesky decomposition.
- **Convex Portfolio Optimizer (`src/ei/optimizer.py`)**: Formulates and solves a second-order cone problem (SOCP) via CVXPY (`Clarabel` solver with `SCS` fallback):
  $$\max_{w} \quad \alpha^T (w - w_b) - \frac{\lambda}{2} (w - w_b)^T \Sigma (w - w_b) - \kappa \|w - w_{\text{drifted}}\|_1$$
  subject to:
  $$\mathbf{1}^T w = 1, \quad w \ge 0$$
  $$|w_i - w_{b,i}| \le \text{active\_weight\_max} \quad (\le 3\%)$$
  $$\left|\sum_{i \in s} (w_i - w_{b,i})\right| \le \text{sector\_dev\_max} \quad (\le 5\%)$$
  $$\sqrt{(w - w_b)^T \Sigma (w - w_b)} \le \text{te\_max} \quad (\le 3\%)$$
- **Backtesting Engine (`src/ei/backtest.py`)**: Realistic monthly rebalancing engine with exact daily price drift between rebalance dates and transaction cost accounting on rebalance transition days.
- **Statistical Significance & Placebos (`scripts/run_significance.py`, `scripts/run_placebo.py`)**: Computes IR t-statistics, block bootstrap confidence intervals, noise alpha placebos (100 seeds), and label permutation placebos (100 seeds).
- **Interactive Dashboard (`app.py`)**: Streamlit application allowing real-time exploration of tracking error budgets, sector constraints, and transaction cost sensitivities.

---

## 2. Setup & Installation

### Requirements
- Python 3.10+ (tested and pinned on Python 3.13.7)

### Installation
```bash
# 1. Clone the repository
git clone <repo_url>
cd EIPO

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. How to Run

### Complete Production Pipeline
To run the full end-to-end backtest, parameter sweeps, placebo simulations, and report generation:
```bash
python scripts/run_all.py
```
*Note: All production artifacts in `results/` are generated deterministically by `run_all.py`.*

### Running Tests
The test suite consists of 45 offline unit/integration tests (including headless Streamlit UI tests) and 2 network validation tests:
```bash
# Run offline test suite (mocked & synthetic data, fast, does not pollute results/)
pytest -q

# Run live network tests (validates data downloads and ^NSEI correlation)
pytest -q -m network
```

### Verification & Auditing Scripts
```bash
# Standalone math audit (recomputes returns and metrics without importing src/ei/backtest.py)
python scripts/independent_check.py

# Real-data constraint and solver integrity verification
python scripts/check_integrity.py
```

### Live Web Application
- **Live Deployment URL**: [https://eipo-txjz.onrender.com/](https://eipo-txjz.onrender.com/)
- **Local Run**:
  ```bash
  streamlit run app.py
  ```
- **Render Production Specs**: Configured via `render.yaml` with Python 3.13.7 and `requirements-app.txt`. The web app reads exclusively from precomputed artifacts in `results/` with zero runtime optimization solvers (`cvxpy`) or network dependencies.

---

## 4. Backtest Results (2016-01-29 to 2026-09-29)

### Strategy Performance Summary
Source: `results/metrics.csv`

| Strategy | Ann. Return | Ann. Vol | Sharpe (Rf=6.5%) | Excess Return | Realized TE | Info Ratio (IR) | Max Drawdown | Beta | Ann. Turnover | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benchmark** (Cap-Weighted) | 14.51% | 15.58% | 0.515 | 0.00% | 0.00% | — | -35.93% | 1.000 | 0.10 | 0.0% |
| **Enhanced Index** (Ledoit-Wolf) | **15.36%** | **15.75%** | **0.562** | **+0.84%** | **2.38%** | **0.319** | **-35.97%** | **0.999** | **4.66** | **53.9%** |
| **Enhanced Index** (Sample Cov) | 15.49% | 15.75% | 0.570 | +0.97% | 2.44% | 0.358 | -35.91% | 0.999 | 4.85 | 53.1% |
| **Mean-Variance** (Long-Only) | 15.13% | 13.38% | 0.645 | +0.62% | 7.87% | 0.027 | -25.51% | 0.741 | 4.54 | 44.5% |

### Statistical Significance (Enhanced LW vs. Benchmark)
Source: `results/significance.csv`

- **Annual Excess Return**: +0.84% (Gross: +1.31%, Cost Drag: 0.47% at 10 bps)
- **Sample Length**: 10.43 years (2628 trading days, 129 monthly rebalances)
- **Information Ratio (IR)**: 0.319
- **IR t-statistic**: $1.031$ ($|t| > 2$ required for two-sided 5% significance)
- **Stationary Bootstrap 95% Confidence Interval**:
  - IR: $[-0.250, +0.894]$ (spans zero)
  - Annual Excess Return: $[-0.60\%, +2.08\%]$
- **Noise Alpha Placebo (100 seeds)**: Mean IR = $-0.213$, Empirical $p\text{-value} = 0.020$
- **Label Permutation Placebo (100 seeds)**: Mean IR = $+0.022$, Empirical $p\text{-value} = 0.130$
- **Conclusion**: Not statistically significant at 5% (t-stat = 1.03, bootstrap 95% CI includes 0 [-0.250, 0.894], permutation placebo p = 0.130).

### Transaction Cost Sensitivity
Source: `results/cost_sensitivity.csv`

| Transaction Cost (bps) | Net Annual Return | Net Excess Return | Realized TE | Information Ratio | Annual Turnover | Cost Drag | Gross Excess Return |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 bps** | 15.93% | +1.41% | 2.41% | 0.516 | 5.57 | 0.00% | 1.41% |
| **5 bps** | 15.63% | +1.11% | 2.40% | 0.414 | 5.10 | 0.25% | 1.37% |
| **10 bps** (Base) | **15.36%** | **+0.84%** | **2.38%** | **0.319** | **4.66** | **0.47%** | **1.31%** |
| **20 bps** | 15.06% | +0.56% | 2.35% | 0.219 | 3.91 | 0.78% | 1.34% |
| **50 bps** | 14.52% | +0.05% | 2.24% | 0.031 | 2.33 | 1.16% | 1.22% |

---

## 5. Limitations & Caveats

1. **Survivorship Bias**: The asset universe is defined using current large-cap constituents of the Nifty 50 rather than historical point-in-time constituent lists. Companies that were removed or went bankrupt over 2015-2026 are not captured.
2. **Static Shares Outstanding**: Market-capitalization proxy weights use static `sharesOutstanding` reported by Yahoo Finance, rather than dynamic historical float-adjusted shares.
3. **Execution Cost Model**: The 10 bps transaction cost model assumes linear proportional trading costs. In practice, large rebalances incur market impact, non-uniform bid-ask spreads, STT (Securities Transaction Tax), and exchange fees.
4. **Calendar Frictions & Dividends**: Daily adjusted close prices reflect corporate actions and dividends, but do not model dividend withholding taxes or intra-day execution timing differences.
5. **Placebo Test Interpretation**: The noise-alpha placebo is biased in favor of the real strategy because random alphas trade more and pay more costs; the label-permutation placebo is the primary test.

---

## 6. Repository Structure

```
EIPO/
├── .agents/                    # Specification documents and build logs
├── config.yaml                 # Backtest and optimization parameters
├── requirements.txt            # Exact pinned package dependencies
├── app.py                      # Interactive Streamlit dashboard
├── src/ei/                     # Core library
│   ├── __init__.py
│   ├── config.py               # YAML configuration loader and schema validator
│   ├── universe.py             # 41-ticker Indian large-cap universe
│   ├── data.py                 # Price downloader, calendar alignment & cap weights
│   ├── synthetic.py            # Reproducible multi-factor synthetic market generator
│   ├── risk.py                 # Covariance estimation (Ledoit-Wolf & sample)
│   ├── signals.py              # Multi-factor signal computation & cross-sectional z-scores
│   ├── optimizer.py            # CVXPY SOCP portfolio optimizer
│   ├── backtest.py             # Daily drift backtesting engine
│   ├── metrics.py              # Performance metrics and annualization
│   └── plots.py                # Publication-quality matplotlib charts
├── scripts/                    # Automation and audit CLI scripts
│   ├── run_all.py              # Master pipeline runner
│   ├── run_backtest.py         # Real/synthetic backtest driver
│   ├── run_sweep.py            # Tracking error budget sweep
│   ├── run_sweep_loose.py      # Loose-constraint active sweep
│   ├── run_placebo.py          # Noise alpha Monte Carlo simulation
│   ├── run_significance.py     # Bootstrap, t-stat, permutation & cost tests
│   ├── run_analysis.py         # Sub-period analysis & summary markdown generator
│   ├── independent_check.py    # Zero-import independent math audit
│   └── check_integrity.py     # Constraint and solver status verification
├── tests/                      # Automated test suite
│   ├── conftest.py             # Pytest fixtures and synthetic market generator
│   ├── test_data.py            # Data loading, cleaning, and weights tests
│   ├── test_risk.py            # Covariance PSD and condition number tests
│   ├── test_signals.py         # Factor signal properties and lookahead tests
│   ├── test_optimizer.py       # Constraint feasibility and boundary tests
│   ├── test_backtest.py        # Rebalance dates, cash-flow and determinism tests
│   ├── test_backtest_hardened.py# Hand-calculated mechanical math & drift tests
│   ├── test_metrics.py         # Closed-form financial metric tests
│   ├── test_integration.py     # End-to-end synthetic pipeline integration tests
│   └── test_network.py         # Live Yahoo Finance & ^NSEI correlation tests
├── data/                       # Cached market prices, sectors, and shares
├── results/                    # Production backtest CSVs, PNGs, and solver status
└── report/                     # Generated results summary and audit reports
```
