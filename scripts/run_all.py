"""
run_all.py  -  Single command that wipes results/ and regenerates everything.

Usage:
    python scripts/run_all.py

Steps:
  1. Delete results/
  2. Run backtest  -> results/metrics.csv, returns.csv, weights, plots
  3. Run sweep     -> results/sweep.csv
  4. Run loose sweep -> results/sweep_loose.csv
  5. Run placebo (20 seeds fast) -> results/placebo.csv
  6. Run significance (full 100 seeds) -> results/significance.csv
  7. Generate report/results_summary.md

Writes results/run_metadata.json with timestamp, config hash, data end date,
and ticker list.
"""

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS = Path("scripts")
RESULTS = Path("results")


def run_script(name: str) -> None:
    logger.info("Running %s ...", name)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("%s failed with exit code %d", name, result.returncode)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    logger.info("%s done.", name)


def write_metadata(cfg: dict, prices: pd.DataFrame, tickers: list[str]) -> None:
    cfg_str = json.dumps(cfg, sort_keys=True, default=str)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    data_end = str(prices.index.max().date())
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_hash_md5_8": cfg_hash,
        "data_end_date": data_end,
        "n_tickers": len(tickers),
        "tickers": tickers,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote results/run_metadata.json")


def generate_report() -> None:
    """Generate report/results_summary.md from CSV outputs."""
    metrics_path = RESULTS / "metrics.csv"
    sig_path = RESULTS / "significance.csv"
    sub_path = RESULTS / "significance_subperiod.csv"

    if not metrics_path.is_file():
        logger.warning("metrics.csv missing; skipping report")
        return

    metrics = pd.read_csv(metrics_path, index_col=0)
    sig = pd.read_csv(sig_path).set_index("metric")["value"] if sig_path.is_file() else None
    sub = pd.read_csv(sub_path) if sub_path.is_file() else None
    meta_raw = json.loads((RESULTS / "run_metadata.json").read_text())

    lines = []
    lines.append("# Results Summary")
    lines.append(f"\nGenerated: {meta_raw['generated_at']}")
    lines.append(f"Config hash: `{meta_raw['config_hash_md5_8']}`  "
                 f"Data end: `{meta_raw['data_end_date']}`  "
                 f"Tickers: {meta_raw['n_tickers']}\n")

    lines.append("## Strategy Metrics\n")
    lines.append(metrics.round(4).to_markdown())

    if sig is not None:
        real_ir = float(sig["real_IR"])
        real_exc = float(sig["real_excess_return"])
        years = float(sig["years"])
        t_stat = float(sig["t_stat"])
        ir_lo = float(sig["ir_boot_CI_lo"])
        ir_hi = float(sig["ir_boot_CI_hi"])
        exc_lo = float(sig["excess_boot_CI_lo"])
        exc_hi = float(sig["excess_boot_CI_hi"])
        p_val = float(sig["placebo_p_value"])
        n_plac = int(sig["placebo_n"])
        plac_mean = float(sig["placebo_mean_IR"])
        plac_std = float(sig["placebo_std_IR"])
        perm_mean = float(sig["perm_mean_IR"]) if "perm_mean_IR" in sig else float("nan")
        perm_std = float(sig["perm_std_IR"]) if "perm_std_IR" in sig else float("nan")
        perm_p = float(sig["perm_p_value"]) if "perm_p_value" in sig else float("nan")
        sig_flag = bool(float(sig["significant_5pct"]))

        lines.append("\n## Statistical Significance (enhanced_lw vs benchmark)\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| IR | {real_ir:.4f} |")
        lines.append(f"| Annual excess return | {real_exc:.4f} ({real_exc*100:.2f}%) |")
        lines.append(f"| Sample length (years) | {years:.1f} |")
        lines.append(f"| **IR t-stat** (IR x sqrt(years)) | **{t_stat:.3f}** |")
        lines.append(f"| Bootstrap 95% CI - IR | [{ir_lo:.4f}, {ir_hi:.4f}] |")
        lines.append(f"| Bootstrap 95% CI - Ann. excess return | [{exc_lo:.4f}, {exc_hi:.4f}] |")
        lines.append(f"| Placebo seeds | {n_plac} |")
        lines.append(f"| Noise placebo mean IR | {plac_mean:.4f} |")
        lines.append(f"| Noise placebo std IR | {plac_std:.4f} |")
        lines.append(f"| **Noise placebo empirical p-value** (share >= real IR) | **{p_val:.3f}** |")
        lines.append(f"| Label perm placebo mean IR | {perm_mean:.4f} |")
        lines.append(f"| Label perm placebo std IR | {perm_std:.4f} |")
        lines.append(f"| **Label perm empirical p-value** (share >= real IR) | **{perm_p:.3f}** |")
        lines.append("")

        if sig_flag:
            lines.append(f"> **Verdict: Statistically significant at 5%** "
                         f"(t-stat = {t_stat:.2f}, bootstrap 95% CI excludes 0, "
                         f"permutation placebo p = {perm_p:.3f}).")
        else:
            lines.append(f"> **Verdict: Not statistically significant at 5%** "
                         f"(t-stat = {t_stat:.2f}, bootstrap 95% CI includes 0 [{ir_lo:.3f}, {ir_hi:.3f}], "
                         f"permutation placebo p = {perm_p:.3f}).")

    cost_sens_path = RESULTS / "cost_sensitivity.csv"
    if cost_sens_path.is_file():
        cost_sens = pd.read_csv(cost_sens_path)
        lines.append("\n## Cost Sensitivity (enhanced_lw vs benchmark)\n")
        lines.append(cost_sens.round(4).to_markdown(index=False))

    if sub is not None:
        lines.append("\n## Sub-Period Analysis\n")
        lines.append(sub.round(4).to_markdown(index=False))

    lines.append("\n## Limitations\n")
    lines.append("1. **Survivorship Bias**: Asset universe consists of current large-cap constituents; delisted or demoted companies over 2015-2026 are excluded.")
    lines.append("2. **Static Shares Outstanding**: Market-cap weights are constructed using static shares outstanding.")
    lines.append("3. **Execution Cost Model**: Linear 10 bps model does not model variable bid-ask spreads, market impact, or statutory transaction taxes.")
    lines.append("4. **Dividend Reinvestment**: Adjusted closes reflect gross dividend reinvestment without tax frictions.")
    lines.append("5. **Placebo Interpretation**: The noise-alpha placebo is biased in favor of the real strategy because random alphas trade more and pay more costs; the label-permutation placebo is the primary test.")

    report_dir = Path("report")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "results_summary.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", report_path)


def main() -> None:
    # 1. Wipe results/
    if RESULTS.exists():
        logger.info("Deleting results/ ...")
        shutil.rmtree(RESULTS)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 2-4. Backtest, sweeps
    run_script("run_backtest.py")
    run_script("run_sweep.py")
    run_script("run_sweep_loose.py")

    # 5. Placebo (20 seeds, fast)
    run_script("run_placebo.py")

    # 6. Significance (full 100 seeds)
    run_script("run_significance.py")

    # 7. Metadata
    cfg = load_config("config.yaml")
    prices = load_prices(cfg)
    tickers = list(prices.columns)
    write_metadata(cfg, prices, tickers)

    # 8. Report
    generate_report()
    logger.info("run_all.py completed successfully.")


if __name__ == "__main__":
    main()
