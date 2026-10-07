"""
run_significance.py    Statistical significance analysis for enhanced_lw.

Outputs: results/significance.csv, results/significance_subperiod.csv
Methodology:
  1. IR t-stat = IR * sqrt(years)
  2. Monthly block bootstrap (2000 resamples, block=21 days, seed from config)
     of the active return series -> 95% CI for IR and annual excess return
  3a. Placebo "noise alpha" (100 seeds):
       At EVERY rebalance, the random z-scores are REDRAWN from the RNG associated
       with that seed. This means each seed uses a different stream of random numbers
       but the same RNG state is used for ALL rebalance dates within one seed run.
       The RNG is created once per seed outside the alpha function and called
       sequentially, so draws are consistent (not i.i.d. across seeds) but cannot
       be correlated with any real cross-sectional signal.
       Metric saved: placebo_mean_IR, placebo_std_IR, placebo_p_value
  3b. Placebo "label permutation" (100 seeds):
       For each seed, draw ONE fixed random permutation of ticker indices.
       Apply that SAME permutation at EVERY rebalance date to the cross-sectional
       z-score vector returned by compute_alpha, then rescale by sigma and ic.
       Because the permutation is fixed across dates, the ranking structure is
       preserved but assigned to the wrong stocks; any persistence of the real
       alpha signal is destroyed while the cross-sectional distribution is kept
       identical to the real strategy.
       Metric saved: perm_mean_IR, perm_std_IR, perm_p_value
  4. Sub-period IR for halves and by calendar year
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from ei.backtest import run_backtest
from ei.config import load_config
from ei.data import load_prices, load_sectors, load_shares
from ei.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ir(active_returns, n_boot=2000, block_size=21, seed=42):
    """
    Stationary block bootstrap of the IR using monthly blocks (21 trading days).
    Returns (IR_estimate, CI_lower_95, CI_upper_95).
    """
    rng = np.random.default_rng(seed)
    arr = active_returns.values
    n = len(arr)
    boot_irs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=(n // block_size + 2))
        sample = np.concatenate([arr[s: s + block_size] for s in starts])[:n]
        ir = float(sample.mean() * 252 / (sample.std(ddof=1) * np.sqrt(252) + 1e-15))
        boot_irs.append(ir)
    boot_irs = np.array(boot_irs)
    return (
        float(np.mean(boot_irs)),
        float(np.percentile(boot_irs, 2.5)),
        float(np.percentile(boot_irs, 97.5)),
    )


def block_bootstrap_excess(active_returns, n_boot=2000, block_size=21, seed=42):
    """Bootstrap annual excess return; returns (mean, CI_lo, CI_hi)."""
    rng = np.random.default_rng(seed)
    arr = active_returns.values
    n = len(arr)
    boot_excess = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=(n // block_size + 2))
        sample = np.concatenate([arr[s: s + block_size] for s in starts])[:n]
        boot_excess.append(float(sample.mean() * 252))
    boot_excess = np.array(boot_excess)
    return (
        float(np.mean(boot_excess)),
        float(np.percentile(boot_excess, 2.5)),
        float(np.percentile(boot_excess, 97.5)),
    )


# ---------------------------------------------------------------------------
# Placebo 3a: noise alpha (z redrawn each rebalance from per-seed RNG)
# ---------------------------------------------------------------------------

def run_noise_placebo_ir(prices, shares, sectors, cfg, seed, r_b=None):
    """
    At every rebalance, draw fresh z-scores from the RNG for this seed.
    The RNG is constructed once per seed (outside the alpha closure) and
    called sequentially, so draws are deterministic given the seed but
    independent of the real cross-sectional signal.
    Returns IR vs benchmark.
    """
    import ei.backtest as _bt
    import ei.signals as _sig

    rng = np.random.default_rng(seed)
    orig = _sig.compute_alpha

    def random_alpha(prices_upto_t, Sigma_ann, cfg_inner):
        n = Sigma_ann.shape[0]
        ic = float(cfg_inner["signals"]["ic"])
        sigma_i = np.sqrt(np.diag(Sigma_ann))
        # Redraw z each rebalance from the same per-seed RNG stream
        z = rng.standard_normal(n)
        z = (z - z.mean()) / (z.std() + 1e-12)
        return pd.Series(ic * sigma_i * z, index=prices_upto_t.columns)

    _sig.compute_alpha = random_alpha
    _bt.compute_alpha = random_alpha
    try:
        res_e = run_backtest(prices, shares, sectors, cfg, strategy="enhanced_lw")
        if r_b is None:
            res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
            r_b = res_b["returns"]
    finally:
        _sig.compute_alpha = orig
        _bt.compute_alpha = orig

    m = compute_metrics(res_e["returns"], r_b,
                        rf=float(cfg.get("risk_free_annual", 0.065)),
                        turnover=res_e.get("turnover"))
    return float(m["IR"])


# ---------------------------------------------------------------------------
# Placebo 3b: label permutation (one fixed permutation per seed, applied at
#             every rebalance; ranking structure preserved, labels scrambled)
# ---------------------------------------------------------------------------

def run_perm_placebo_ir(prices, shares, sectors, cfg, seed, r_b=None):
    """
    Draw ONE fixed permutation of ticker indices (determined by seed).
    At every rebalance, apply that SAME permutation to the cross-sectional
    z-score vector returned by compute_alpha, then rescale by sigma_i and ic.
    The permutation is constant across dates: the ranking is reassigned to
    wrong stocks but the distribution of z-scores is unchanged.
    Returns IR vs benchmark.
    """
    import ei.backtest as _bt
    import ei.signals as _sig

    tickers = list(prices.columns)
    n = len(tickers)

    # Draw one fixed permutation for this seed
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)  # shape (n,), same for all rebalance dates

    orig = _sig.compute_alpha

    def perm_alpha(prices_upto_t, Sigma_ann, cfg_inner):
        components = _sig.compute_signal_components(prices_upto_t, cfg_inner)
        z = components["composite"].reindex(tickers).values
        z_perm = z[perm]
        ic = float(cfg_inner["signals"]["ic"])
        sigma_i = np.sqrt(np.diag(Sigma_ann))
        alpha_perm = ic * sigma_i * z_perm
        return pd.Series(alpha_perm, index=tickers, name="alpha")

    _sig.compute_alpha = perm_alpha
    _bt.compute_alpha = perm_alpha
    try:
        res_e = run_backtest(prices, shares, sectors, cfg, strategy="enhanced_lw")
        if r_b is None:
            res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
            r_b = res_b["returns"]
    finally:
        _sig.compute_alpha = orig
        _bt.compute_alpha = orig

    m = compute_metrics(res_e["returns"], r_b,
                        rf=float(cfg.get("risk_free_annual", 0.065)),
                        turnover=res_e.get("turnover"))
    return float(m["IR"])


# Worker state and callbacks for multiprocessing pool
_worker_state = {}

def _worker_init(prices, shares, sectors, cfg, r_b):
    _worker_state["prices"] = prices
    _worker_state["shares"] = shares
    _worker_state["sectors"] = sectors
    _worker_state["cfg"] = cfg
    _worker_state["r_b"] = r_b

def _run_noise_worker(seed: int) -> float:
    return run_noise_placebo_ir(
        _worker_state["prices"],
        _worker_state["shares"],
        _worker_state["sectors"],
        _worker_state["cfg"],
        seed,
        r_b=_worker_state["r_b"],
    )

def _run_perm_worker(seed: int) -> float:
    return run_perm_placebo_ir(
        _worker_state["prices"],
        _worker_state["shares"],
        _worker_state["sectors"],
        _worker_state["cfg"],
        seed,
        r_b=_worker_state["r_b"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config("config.yaml")
    seed = int(cfg.get("seed", 42))
    rf = float(cfg.get("risk_free_annual", 0.065))

    logger.info("Loading real market data...")
    prices = load_prices(cfg)
    tickers = list(prices.columns)
    shares = load_shares(tickers, cfg)
    sectors = load_sectors(tickers, cfg)

    # Real strategy
    logger.info("Running enhanced_lw and benchmark...")
    res_e = run_backtest(prices, shares, sectors, cfg, strategy="enhanced_lw")
    res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")

    r_e = res_e["returns"]
    r_b = res_b["returns"]
    common = r_e.index.intersection(r_b.index)
    r_e = r_e.loc[common]
    r_b = r_b.loc[common]
    active = r_e - r_b

    m = compute_metrics(r_e, r_b, rf=rf, turnover=res_e["turnover"],
                        ex_ante_te=res_e["ex_ante_te"])
    real_ir = float(m["IR"])
    real_excess = float(m["excess_return"])

    # 1. T-stat
    years = len(active) / 252.0
    t_stat = real_ir * np.sqrt(years)
    logger.info("IR=%.4f  t-stat=%.4f  years=%.1f", real_ir, t_stat, years)

    # 2. Block bootstrap
    logger.info("Block bootstrap (2000 resamples)...")
    ir_boot, ir_lo, ir_hi = block_bootstrap_ir(active, n_boot=2000,
                                                block_size=21, seed=seed)
    exc_boot, exc_lo, exc_hi = block_bootstrap_excess(active, n_boot=2000,
                                                       block_size=21, seed=seed)
    logger.info("IR bootstrap: mean=%.4f  95%%CI=[%.4f, %.4f]", ir_boot, ir_lo, ir_hi)
    logger.info("Excess bootstrap 95%%CI=[%.4f, %.4f]", exc_lo, exc_hi)

    # Parallel execution for placebos
    import os
    from concurrent.futures import ProcessPoolExecutor

    n_placebo = 100
    max_workers = min(8, max(1, (os.cpu_count() or 4) - 2))

    # 3a. Noise placebo (100 seeds, z redrawn each rebalance)
    logger.info("Running %d noise-alpha placebo seeds with %d workers...", n_placebo, max_workers)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init,
                            initargs=(prices, shares, sectors, cfg, r_b)) as pool:
        placebo_irs = list(pool.map(_run_noise_worker, range(1, n_placebo + 1)))
    placebo_irs = np.array(placebo_irs)
    placebo_mean = float(placebo_irs.mean())
    placebo_std = float(placebo_irs.std())
    empirical_p = float((placebo_irs >= real_ir).mean())
    logger.info("Noise placebo: mean=%.4f  std=%.4f  p=%.4f",
                placebo_mean, placebo_std, empirical_p)

    # 3b. Label-permutation placebo (100 seeds, fixed permutation per seed)
    logger.info("Running %d label-permutation placebo seeds with %d workers...", n_placebo, max_workers)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init,
                            initargs=(prices, shares, sectors, cfg, r_b)) as pool:
        perm_irs = list(pool.map(_run_perm_worker, range(1, n_placebo + 1)))
    perm_irs = np.array(perm_irs)
    perm_mean = float(perm_irs.mean())
    perm_std = float(perm_irs.std())
    perm_p = float((perm_irs >= real_ir).mean())
    logger.info("Perm placebo: mean=%.4f  std=%.4f  p=%.4f", perm_mean, perm_std, perm_p)

    # 4. Sub-period by year and by half
    mid = common[len(common) // 2]
    sub_records = []
    for label, mask in [("first_half", common <= mid), ("second_half", common > mid)]:
        re_sub = r_e[mask]
        rb_sub = r_b[mask]
        if len(re_sub) < 60:
            continue
        ms = compute_metrics(re_sub, rb_sub, rf=rf)
        sub_records.append({
            "period": label,
            "start": str(re_sub.index[0].date()),
            "end": str(re_sub.index[-1].date()),
            "n_days": len(re_sub),
            "excess": ms["excess_return"],
            "IR": ms["IR"],
            "realized_TE": ms["realized_TE"],
        })

    for yr in sorted(set(common.year)):
        mask = common.year == yr
        re_sub = r_e[mask]
        rb_sub = r_b[mask]
        if len(re_sub) < 10:
            continue
        ms = compute_metrics(re_sub, rb_sub, rf=rf)
        sub_records.append({
            "period": str(yr),
            "start": str(re_sub.index[0].date()),
            "end": str(re_sub.index[-1].date()),
            "n_days": len(re_sub),
            "excess": ms["excess_return"],
            "IR": ms["IR"],
            "realized_TE": ms["realized_TE"],
        })

    sub_df = pd.DataFrame(sub_records)

    # Significance flag: requires BOTH bootstrap CI to exclude 0 AND both p-values < 0.05
    sig_flag = float(ir_lo > 0 and empirical_p < 0.05 and perm_p < 0.05)

    # Assemble summary
    summary_rows = [
        {"metric": "real_IR",             "value": real_ir},
        {"metric": "real_excess_return",  "value": real_excess},
        {"metric": "years",               "value": round(years, 2)},
        {"metric": "t_stat",              "value": t_stat},
        {"metric": "ir_boot_mean",        "value": ir_boot},
        {"metric": "ir_boot_CI_lo",       "value": ir_lo},
        {"metric": "ir_boot_CI_hi",       "value": ir_hi},
        {"metric": "excess_boot_CI_lo",   "value": exc_lo},
        {"metric": "excess_boot_CI_hi",   "value": exc_hi},
        {"metric": "placebo_n",           "value": n_placebo},
        {"metric": "placebo_mean_IR",     "value": placebo_mean},
        {"metric": "placebo_std_IR",      "value": placebo_std},
        {"metric": "placebo_p_value",     "value": empirical_p},
        {"metric": "perm_mean_IR",        "value": perm_mean},
        {"metric": "perm_std_IR",         "value": perm_std},
        {"metric": "perm_p_value",        "value": perm_p},
        {"metric": "significant_5pct",    "value": sig_flag},
    ]
    summary = pd.DataFrame(summary_rows)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "significance.csv", index=False)
    sub_df.to_csv(results_dir / "significance_subperiod.csv", index=False)

    logger.info("Saved results/significance.csv and results/significance_subperiod.csv")

    print("\n=== SIGNIFICANCE SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== SUB-PERIOD ===")
    print(sub_df.to_string(index=False))

    boot_ci_excludes_zero = ir_lo > 0
    print(f"\nNoise-alpha placebo p={empirical_p:.3f}  Label-perm placebo p={perm_p:.3f}")
    print(f"Bootstrap 95% CI for IR: [{ir_lo:.3f}, {ir_hi:.3f}]  "
          f"({'EXCLUDES' if boot_ci_excludes_zero else 'INCLUDES'} zero)")
    print(f"Statistically significant at 5%: {'YES' if sig_flag else 'NO'}")


if __name__ == "__main__":
    main()
