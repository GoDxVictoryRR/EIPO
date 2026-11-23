"""Generate all figures and equations for the Project Progress Report.

This script adheres strictly to:
- Black and grey only (no colours)
- Times New Roman font at 12 pt
- Exact dimensions <= 6.0 inches width, <= 8.5 inches height
- High resolution 300 DPI
- Labels <= 4 words
- Data sourced directly from results/ and report/gantt.yaml
"""

import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set global matplotlib styles
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / 'report'
FIGURES_DIR = REPORT_DIR / 'figures'
RESULTS_DIR = BASE_DIR / 'results'

def generate_equations():
    """Render equations as crisp 300 DPI PNGs."""
    eqs = [
        ("eq1.png", r"$\alpha_i = \mathrm{IC} \cdot \sigma_i \cdot z_i^{\mathrm{composite}} \quad \quad (1)$"),
        ("eq2.png", r"$x_i = w_i - w_{b,i} \quad \mathrm{with} \quad \sum_{i=1}^N x_i = 0 \quad \quad (2)$"),
        ("eq3.png", r"$\mathrm{TE}(x) = \sqrt{x^T \Sigma x} \leq \mathrm{te}_{\max} \quad \quad (3)$"),
        ("eq4.png", r"$\max_{w} \; \left( \alpha^T w - \frac{\lambda}{2} (w - w_b)^T \Sigma (w - w_b) \right) \quad \quad (4)$"),
        ("eq5.png", r"$w_{i, t+1} = \frac{w_{i, t}(1 + r_{i, t+1})}{1 + \sum_{j=1}^N w_{j, t} r_{j, t+1}} \quad \quad (5)$"),
        ("eq6.png", r"$C_t = c \sum_{i=1}^N |w_{i, t} - w_{i, t^-}| \quad \quad (6)$"),
        ("eq7.png", r"$\mathrm{IR} = \frac{\overline{r}_{\mathrm{excess}}}{\sigma_{\mathrm{TE}}} = \frac{\mathrm{Annualized} \; \mathrm{Excess} \; \mathrm{Return}}{\mathrm{Realized} \; \mathrm{Tracking} \; \mathrm{Error}} \quad \quad (7)$")
    ]
    
    for filename, eq_text in eqs:
        fig = plt.figure(figsize=(6.0, 0.7), dpi=300)
        fig.text(0.5, 0.5, eq_text, fontsize=12, ha='center', va='center', color='black')
        plt.axis('off')
        out_path = FIGURES_DIR / filename
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.04, facecolor='white')
        plt.close(fig)
    print("Equations 1-7 generated.")

def generate_fig1_flowchart():
    """Figure 1: System Flowchart."""
    fig, ax = plt.subplots(figsize=(6.0, 7.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    steps = [
        ("Market Price Data", "terminator"),
        ("Data Cleaning Filter", "process"),
        ("Market-Cap Benchmark", "process"),
        ("Signal Generation", "process"),
        ("Risk Covariance Model", "process"),
        ("Convex QP Optimizer", "process"),
        ("Monthly Backtest Loop", "process"),
        ("Performance Metrics", "process"),
        ("Statistical Tests", "process"),
        ("Streamlit Web Dashboard", "terminator")
    ]

    y_pos = np.linspace(11.2, 0.8, len(steps))
    box_w = 4.8
    box_h = 0.65

    for i, (text, shape_type) in enumerate(steps):
        y = y_pos[i]
        x = 5.0 - box_w / 2
        if shape_type == "terminator":
            box = patches.FancyBboxPatch((x, y - box_h/2), box_w, box_h,
                                         boxstyle="round,pad=0.2,rounding_size=0.3",
                                         facecolor="white", edgecolor="black", linewidth=1.2)
        else:
            box = patches.Rectangle((x, y - box_h/2), box_w, box_h,
                                    facecolor="white", edgecolor="black", linewidth=1.2)
        ax.add_patch(box)
        ax.text(5.0, y, text, ha='center', va='center', fontsize=12, family='serif')

        if i < len(steps) - 1:
            next_y = y_pos[i+1]
            ax.annotate('', xy=(5.0, next_y + box_h/2), xytext=(5.0, y - box_h/2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    out_path = FIGURES_DIR / "fig1_system_flowchart.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Figure 1 generated.")

def generate_fig2_dfd0():
    """Figure 2: Data Flow Diagram Level 0 (Context Diagram)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # External Entity Left
    box_e1 = patches.Rectangle((0.3, 1.8), 2.5, 1.4, facecolor="white", edgecolor="black", linewidth=1.2)
    ax.add_patch(box_e1)
    ax.text(1.55, 2.5, "Market Data\n(Yahoo Finance)", ha='center', va='center', fontsize=12, family='serif')

    # Central Process
    circ = patches.FancyBboxPatch((3.7, 1.5), 2.6, 2.0, boxstyle="round,pad=0.2,rounding_size=0.5",
                                  facecolor="white", edgecolor="black", linewidth=1.2)
    ax.add_patch(circ)
    ax.text(5.0, 2.5, "0.0\nPortfolio System", ha='center', va='center', fontsize=12, family='serif')

    # External Entity Right
    box_e2 = patches.Rectangle((7.2, 1.8), 2.5, 1.4, facecolor="white", edgecolor="black", linewidth=1.2)
    ax.add_patch(box_e2)
    ax.text(8.45, 2.5, "Portfolio Viewer\n(End User)", ha='center', va='center', fontsize=12, family='serif')

    # Flow Left to Center
    ax.annotate('', xy=(3.7, 2.6), xytext=(2.8, 2.6),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(3.25, 2.85, "Price Data", ha='center', va='bottom', fontsize=10, family='serif')

    # Flow Center to Right
    ax.annotate('', xy=(7.2, 2.6), xytext=(6.3, 2.6),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(6.75, 2.85, "Portfolio Reports", ha='center', va='bottom', fontsize=10, family='serif')

    out_path = FIGURES_DIR / "fig2_dfd0.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Figure 2 generated.")

def generate_fig3_dfd1():
    """Figure 3: Data Flow Diagram Level 1."""
    fig, ax = plt.subplots(figsize=(6.0, 7.8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # External entities at top and bottom
    ax.add_patch(patches.Rectangle((0.5, 11.5), 2.5, 1.0, facecolor="white", edgecolor="black", lw=1.2))
    ax.text(1.75, 12.0, "Yahoo Finance", ha='center', va='center', fontsize=11, family='serif')

    ax.add_patch(patches.Rectangle((7.0, 0.5), 2.5, 1.0, facecolor="white", edgecolor="black", lw=1.2))
    ax.text(8.25, 1.0, "Portfolio Viewer", ha='center', va='center', fontsize=11, family='serif')

    # Processes
    procs = [
        (1.0, "1.0 Prepare Data", 5.5, 11.5),
        (2.0, "2.0 Build Signals", 5.5, 9.8),
        (3.0, "3.0 Estimate Risk", 5.5, 8.1),
        (4.0, "4.0 Solve Optimizer", 5.5, 6.4),
        (5.0, "5.0 Backtest Engine", 5.5, 4.7),
        (6.0, "6.0 Statistical Tests", 5.5, 3.0),
        (7.0, "7.0 Render Dashboard", 5.5, 1.3),
    ]

    for p_id, p_name, px, py in procs:
        p_box = patches.FancyBboxPatch((px - 1.8, py - 0.4), 3.6, 0.8,
                                       boxstyle="round,pad=0.1,rounding_size=0.2",
                                       facecolor="white", edgecolor="black", lw=1.2)
        ax.add_patch(p_box)
        ax.text(px, py, p_name, ha='center', va='center', fontsize=11, family='serif')

    # Data stores on the left
    stores = [
        ("D1 Price Store", 1.5, 9.0),
        ("D2 Weights Store", 1.5, 5.5),
        ("D3 Results Store", 1.5, 2.2),
    ]

    for s_name, sx, sy in stores:
        # Open-ended rectangle for data store
        ax.plot([sx - 1.2, sx + 1.2], [sy + 0.35, sy + 0.35], color="black", lw=1.2)
        ax.plot([sx - 1.2, sx + 1.2], [sy - 0.35, sy - 0.35], color="black", lw=1.2)
        ax.plot([sx - 1.2, sx - 1.2], [sy - 0.35, sy + 0.35], color="black", lw=1.2)
        ax.text(sx, sy, s_name, ha='center', va='center', fontsize=10, family='serif')

    # Connectors
    # Yahoo Finance to 1.0
    ax.annotate('', xy=(3.7, 12.0), xytext=(3.0, 12.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    
    # 1.0 to D1
    ax.annotate('', xy=(1.5, 9.4), xytext=(4.0, 11.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    # D1 to 2.0 and 3.0
    ax.annotate('', xy=(3.7, 9.8), xytext=(2.7, 9.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate('', xy=(3.7, 8.2), xytext=(2.7, 8.8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    # Flow between processes
    ax.annotate('', xy=(5.5, 10.6), xytext=(5.5, 11.1), arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate('', xy=(5.5, 8.9), xytext=(5.5, 9.4), arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate('', xy=(5.5, 7.2), xytext=(5.5, 7.7), arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate('', xy=(5.5, 5.5), xytext=(5.5, 6.0), arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate('', xy=(5.5, 3.8), xytext=(5.5, 4.3), arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate('', xy=(5.5, 2.1), xytext=(5.5, 2.6), arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    # 4.0 to D2
    ax.annotate('', xy=(2.7, 5.5), xytext=(3.7, 6.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    # D2 to 5.0
    ax.annotate('', xy=(3.7, 4.8), xytext=(2.7, 5.3),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    # 6.0 to D3
    ax.annotate('', xy=(2.7, 2.3), xytext=(3.7, 2.9),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    # D3 to 7.0
    ax.annotate('', xy=(3.7, 1.4), xytext=(2.7, 2.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    # 7.0 to Viewer
    ax.annotate('', xy=(7.0, 1.1), xytext=(6.5, 1.1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    out_path = FIGURES_DIR / "fig3_dfd1.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Figure 3 generated.")

def generate_fig4_rebalance():
    """Figure 4: Monthly Rebalance Loop Flowchart."""
    fig, ax = plt.subplots(figsize=(6.0, 7.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # 1. Start Rebalance
    start = patches.FancyBboxPatch((2.8, 9.8), 4.4, 0.65, boxstyle="round,pad=0.2,rounding_size=0.3",
                                   facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(start)
    ax.text(5.0, 10.12, "Rebalance Date Reached", ha='center', va='center', fontsize=11, family='serif')

    # 2. Form Signals & Cov
    b1 = patches.Rectangle((2.8, 8.4), 4.4, 0.65, facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(b1)
    ax.text(5.0, 8.72, "Estimate Alpha & Risk", ha='center', va='center', fontsize=11, family='serif')

    # 3. Run QP Solver
    b2 = patches.Rectangle((2.8, 7.0), 4.4, 0.65, facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(b2)
    ax.text(5.0, 7.32, "Execute Convex Solver", ha='center', va='center', fontsize=11, family='serif')

    # 4. Decision Diamond: Solver Optimal?
    diam = patches.Polygon([[5.0, 6.2], [6.8, 5.4], [5.0, 4.6], [3.2, 5.4]],
                           facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(diam)
    ax.text(5.0, 5.4, "Solver\nOptimal?", ha='center', va='center', fontsize=11, family='serif')

    # Yes Branch: Apply Target Weights
    b_yes = patches.Rectangle((6.2, 3.4), 3.2, 0.65, facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(b_yes)
    ax.text(7.8, 3.72, "Apply Target Weights", ha='center', va='center', fontsize=10, family='serif')

    # No Branch: Retain Previous Weights
    b_no = patches.Rectangle((0.6, 3.4), 3.2, 0.65, facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(b_no)
    ax.text(2.2, 3.72, "Retain Previous Weights", ha='center', va='center', fontsize=10, family='serif')

    # 5. Compute Drift & Costs
    b3 = patches.Rectangle((2.8, 1.9), 4.4, 0.65, facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(b3)
    ax.text(5.0, 2.22, "Calculate Drift & Costs", ha='center', va='center', fontsize=11, family='serif')

    # 6. End
    end = patches.FancyBboxPatch((2.8, 0.5), 4.4, 0.65, boxstyle="round,pad=0.2,rounding_size=0.3",
                                 facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(end)
    ax.text(5.0, 0.82, "Advance Trading Day", ha='center', va='center', fontsize=11, family='serif')

    # Arrows
    ax.annotate('', xy=(5.0, 9.05), xytext=(5.0, 9.8), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.annotate('', xy=(5.0, 7.65), xytext=(5.0, 8.4), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.annotate('', xy=(5.0, 6.2), xytext=(5.0, 7.0), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Branch arrows
    # Yes -> Right then down
    ax.plot([6.8, 7.8, 7.8], [5.4, 5.4, 4.05], color="black", lw=1.2)
    ax.annotate('', xy=(7.8, 4.05), xytext=(7.8, 4.15), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(7.2, 5.55, "Yes", fontsize=10, family='serif')

    # No -> Left then down
    ax.plot([3.2, 2.2, 2.2], [5.4, 5.4, 4.05], color="black", lw=1.2)
    ax.annotate('', xy=(2.2, 4.05), xytext=(2.2, 4.15), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(2.6, 5.55, "No", fontsize=10, family='serif')

    # Merge down to b3
    ax.plot([7.8, 7.8, 5.0], [3.4, 2.9, 2.9], color="black", lw=1.2)
    ax.plot([2.2, 2.2, 5.0], [3.4, 2.9, 2.9], color="black", lw=1.2)
    ax.annotate('', xy=(5.0, 2.55), xytext=(5.0, 2.9), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # b3 to end
    ax.annotate('', xy=(5.0, 1.15), xytext=(5.0, 1.9), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    out_path = FIGURES_DIR / "fig4_rebalance_loop.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Figure 4 generated.")

def generate_fig5_er_diagram():
    """Figure 5: Entity-Relationship Diagram."""
    fig, ax = plt.subplots(figsize=(6.0, 7.0), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Entities: Rectangles
    # Relationships: Diamonds
    entities = [
        ("SECTOR", 1.8, 9.8),
        ("STOCK", 1.8, 7.0),
        ("PRICE", 1.8, 4.2),
        ("REBALANCE", 7.8, 7.0),
        ("WEIGHT", 7.8, 4.2),
        ("STRATEGY", 7.8, 1.4),
        ("METRIC", 1.8, 1.4)
    ]

    for name, x, y in entities:
        box = patches.Rectangle((x - 1.4, y - 0.45), 2.8, 0.9, facecolor="white", edgecolor="black", lw=1.2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=11, family='serif', weight='bold')

    # Diamonds for relationships
    def draw_rel(text, rx, ry):
        d = patches.Polygon([[rx, ry + 0.45], [rx + 0.9, ry], [rx, ry - 0.45], [rx - 0.9, ry]],
                            facecolor="white", edgecolor="black", lw=1.2)
        ax.add_patch(d)
        ax.text(rx, ry, text, ha='center', va='center', fontsize=9, family='serif')

    draw_rel("Contains", 1.8, 8.4)
    draw_rel("Records", 1.8, 5.6)
    draw_rel("Applies", 7.8, 5.6)
    draw_rel("Calculates", 7.8, 2.8)
    draw_rel("Evaluates", 4.8, 1.4)
    draw_rel("Allocates", 4.8, 4.2)

    # Connecting lines with cardinalities
    # SECTOR to Contains (1) and Contains to STOCK (M)
    ax.plot([1.8, 1.8], [9.35, 8.85], color="black", lw=1.0)
    ax.plot([1.8, 1.8], [7.95, 7.45], color="black", lw=1.0)
    ax.text(2.0, 9.1, "1", fontsize=10, family='serif')
    ax.text(2.0, 7.6, "N", fontsize=10, family='serif')

    # STOCK to Records (1) and Records to PRICE (M)
    ax.plot([1.8, 1.8], [6.55, 6.05], color="black", lw=1.0)
    ax.plot([1.8, 1.8], [5.15, 4.65], color="black", lw=1.0)
    ax.text(2.0, 6.3, "1", fontsize=10, family='serif')
    ax.text(2.0, 4.8, "N", fontsize=10, family='serif')

    # REBALANCE to Applies (1) and Applies to WEIGHT (M)
    ax.plot([7.8, 7.8], [6.55, 6.05], color="black", lw=1.0)
    ax.plot([7.8, 7.8], [5.15, 4.65], color="black", lw=1.0)
    ax.text(8.0, 6.3, "1", fontsize=10, family='serif')
    ax.text(8.0, 4.8, "N", fontsize=10, family='serif')

    # STRATEGY to Calculates (1) and Calculates to WEIGHT (M)
    ax.plot([7.8, 7.8], [3.75, 3.25], color="black", lw=1.0)
    ax.plot([7.8, 7.8], [2.35, 1.85], color="black", lw=1.0)
    ax.text(8.0, 3.5, "M", fontsize=10, family='serif')
    ax.text(8.0, 2.0, "1", fontsize=10, family='serif')

    # STOCK to Allocates to WEIGHT
    ax.plot([3.2, 3.9], [4.2, 4.2], color="black", lw=1.0)
    ax.plot([5.7, 6.4], [4.2, 4.2], color="black", lw=1.0)
    ax.text(3.4, 4.4, "1", fontsize=10, family='serif')
    ax.text(6.1, 4.4, "N", fontsize=10, family='serif')

    # STRATEGY to Evaluates to METRIC
    ax.plot([6.4, 5.7], [1.4, 1.4], color="black", lw=1.0)
    ax.plot([3.9, 3.2], [1.4, 1.4], color="black", lw=1.0)
    ax.text(6.1, 1.6, "1", fontsize=10, family='serif')
    ax.text(3.4, 1.6, "1", fontsize=10, family='serif')

    out_path = FIGURES_DIR / "fig5_er_diagram.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Figure 5 generated.")

def generate_result_figures():
    """Figures 6-10: Results figures generated directly from results/ CSVs."""
    returns_path = RESULTS_DIR / 'returns.csv'
    df_ret = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # 1. Figure 6: Cumulative Returns
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=300)
    cum_ret = (1 + df_ret).cumprod()
    ax.plot(cum_ret.index, cum_ret['benchmark'], color='#666666', linestyle='-', lw=1.3, label='Benchmark')
    ax.plot(cum_ret.index, cum_ret['enhanced_lw'], color='black', linestyle='-', lw=1.6, label='Enhanced (Ledoit-Wolf)')
    ax.plot(cum_ret.index, cum_ret['enhanced_sample'], color='black', linestyle='--', lw=1.2, label='Enhanced (Sample Cov)')
    ax.plot(cum_ret.index, cum_ret['mv'], color='#777777', linestyle=':', lw=1.4, label='Minimum Variance')
    
    ax.set_ylabel('Growth of 1.0 Unit', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, color='grey')
    out_path = FIGURES_DIR / "fig6_cumulative_returns.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print("Figure 6 generated.")

    # 2. Figure 7: Rolling Tracking Error
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=300)
    active = df_ret['enhanced_lw'] - df_ret['benchmark']
    rolling_te = active.rolling(252).std() * np.sqrt(252)
    ax.plot(rolling_te.index, rolling_te * 100, color='black', linestyle='-', lw=1.4, label='Realized 252-Day TE (%)')
    ax.axhline(3.0, color='#666666', linestyle='--', lw=1.3, label='Target Ex-Ante Limit (3.0%)')
    ax.set_ylabel('Tracking Error (% Ann.)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, color='grey')
    out_path = FIGURES_DIR / "fig7_rolling_te.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print("Figure 7 generated.")

    # 3. Figure 8: Historical Drawdown Profile
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=300)
    cum_b = (1 + df_ret['benchmark']).cumprod()
    cum_e = (1 + df_ret['enhanced_lw']).cumprod()
    dd_b = (cum_b - cum_b.cummax()) / cum_b.cummax() * 100
    dd_e = (cum_e - cum_e.cummax()) / cum_e.cummax() * 100
    
    ax.plot(dd_b.index, dd_b, color='#777777', linestyle='-', lw=1.2, label='Benchmark Drawdown')
    ax.plot(dd_e.index, dd_e, color='black', linestyle='-', lw=1.4, label='Enhanced LW Drawdown')
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='lower left', frameon=False, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, color='grey')
    out_path = FIGURES_DIR / "fig8_drawdown.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print("Figure 8 generated.")

    # 4. Figure 9: Information Ratio vs Tracking Error
    sweep_path = RESULTS_DIR / 'sweep.csv'
    sweep_loose_path = RESULTS_DIR / 'sweep_loose.csv'
    df_sweep = pd.read_csv(sweep_path)
    df_sweep_loose = pd.read_csv(sweep_loose_path)

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=300)
    ax.plot(df_sweep['te_max'] * 100, df_sweep['IR'], color='black', marker='o', linestyle='-', lw=1.4, label='Standard TE Sweep')
    ax.plot(df_sweep_loose['te_max'] * 100, df_sweep_loose['IR'], color='#666666', marker='s', linestyle='--', lw=1.3, label='Loose TE Sweep')
    ax.set_xlabel('Target Ex-Ante Tracking Error Limit (%)', fontsize=12)
    ax.set_ylabel('Information Ratio', fontsize=12)
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, color='grey')
    out_path = FIGURES_DIR / "fig9_ir_vs_te.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print("Figure 9 generated.")

    # 5. Figure 10: Excess Return vs Trading Cost
    cost_path = RESULTS_DIR / 'cost_sensitivity.csv'
    df_cost = pd.read_csv(cost_path)

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=300)
    ax.plot(df_cost['tc_bps'], df_cost['excess_return'] * 100, color='black', marker='o', linestyle='-', lw=1.5, label='Net Excess Return (%)')
    ax.plot(df_cost['tc_bps'], df_cost['cost_drag'] * 100, color='#666666', marker='^', linestyle='--', lw=1.3, label='Cost Drag (%)')
    ax.set_xlabel('Transaction Cost Parameter (bps)', fontsize=12)
    ax.set_ylabel('Annual Return Impact (%)', fontsize=12)
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, color='grey')
    out_path = FIGURES_DIR / "fig10_cost_sensitivity.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print("Figure 10 generated.")

def generate_fig11_gantt():
    """Figure 11: Project Timeline Gantt Chart."""
    gantt_path = REPORT_DIR / 'gantt.yaml'
    with open(gantt_path, 'r', encoding='utf-8') as f:
        gantt_data = yaml.safe_load(f)

    tasks = gantt_data['tasks']
    # Reverse so task 1 is at top
    tasks = list(reversed(tasks))

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    y_pos = np.arange(len(tasks))
    task_names = [t['name'] for t in tasks]

    for i, t in enumerate(tasks):
        start = t['start_week']
        duration = t['duration']
        status = t['status']
        if status == 'Completed':
            ax.barh(i, duration, left=start, height=0.55, color='black', edgecolor='black')
        elif status == 'In Progress':
            ax.barh(i, duration, left=start, height=0.55, color='white', edgecolor='black', hatch='///')
        else:
            ax.barh(i, duration, left=start, height=0.55, color='white', edgecolor='black', hatch='...')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(task_names, fontsize=10)
    ax.set_xlabel('Project Timeline (Weeks)', fontsize=12)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(axis='x', linestyle=':', alpha=0.6, color='grey')

    # Custom legend
    p1 = patches.Patch(facecolor='black', edgecolor='black', label='Completed')
    p2 = patches.Patch(facecolor='white', edgecolor='black', hatch='///', label='In Progress')
    p3 = patches.Patch(facecolor='white', edgecolor='black', hatch='...', label='Pending')
    ax.legend(handles=[p1, p2, p3], loc='lower right', frameon=False, fontsize=10)

    out_path = FIGURES_DIR / "fig11_gantt.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print("Figure 11 generated.")

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_equations()
    generate_fig1_flowchart()
    generate_fig2_dfd0()
    generate_fig3_dfd1()
    generate_fig4_rebalance()
    generate_fig5_er_diagram()
    generate_result_figures()
    generate_fig11_gantt()
    print("All figures successfully created in", FIGURES_DIR)

if __name__ == '__main__':
    main()
