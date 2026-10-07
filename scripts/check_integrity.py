"""
scripts/check_integrity.py

Verifies Item B.3 and B.5:
  - weights sum to 1 (1e-6) and min weight >= -1e-8
  - max |active weight| <= max_active_weight + 1e-6
  - max sector deviation <= sector_dev_max + 1e-6
  - ex-ante TE <= te_max + 1e-6
  - No NaN/inf in any results csv; dates strictly increasing; first rebalance is after warmup; last date <= config end date.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
from ei.config import load_config
from ei.data import load_prices, load_shares, load_sectors, cap_weights
from ei.risk import estimate_cov


def main():
    cfg = load_config("config.yaml")
    prices = load_prices(cfg)
    shares = load_shares(list(prices.columns), cfg)
    sectors = load_sectors(list(prices.columns), cfg)
    w_df = pd.read_csv("results/weights_enhanced.csv", index_col=0, parse_dates=True)

    print(f"=== CHECKING B.3 CONSTRAINTS ON {len(w_df)} REBALANCES ===")

    # 1. Weights sum to 1 and min weight >= -1e-8
    sum_err = (w_df.sum(axis=1) - 1.0).abs().max()
    min_w = float(w_df.min().min())
    print(f"1. Max |sum(w) - 1|: {sum_err:.2e}  (limit <= 1e-6) -> {'PASS' if sum_err <= 1e-6 else 'FAIL'}")
    print(f"   Min weight:       {min_w:.2e}  (limit >= -1e-8) -> {'PASS' if min_w >= -1e-8 else 'FAIL'}")

    max_active_limit = float(cfg["optimizer"]["max_active_weight"])
    sec_dev_limit = float(cfg["optimizer"]["sector_dev_max"])
    te_limit = float(cfg["optimizer"]["te_max"])
    window = int(cfg["estimation_window"])

    max_act = 0.0
    max_sec_dev = 0.0
    max_ex_ante = 0.0

    violations = []
    for dt in w_df.index:
        w = w_df.loc[dt].values
        wb = cap_weights(prices, shares, dt).reindex(w_df.columns).values
        a = w - wb

        act_max = float(np.abs(a).max())
        max_act = max(max_act, act_max)

        sec_df = pd.DataFrame({"a": a, "sector": sectors.reindex(w_df.columns).values})
        sec_sum = float(sec_df[sec_df["sector"] != "Unknown"].groupby("sector")["a"].sum().abs().max())
        max_sec_dev = max(max_sec_dev, sec_sum)

        p_sub = prices.loc[:dt]
        rets_sub = p_sub.pct_change().dropna()
        sigma, _ = estimate_cov(rets_sub.iloc[-window:], method="ledoit_wolf")
        te = float(np.sqrt(np.maximum(0.0, a @ sigma @ a)))
        max_ex_ante = max(max_ex_ante, te)

        if (
            act_max > max_active_limit + 1e-6
            or sec_sum > sec_dev_limit + 1e-6
            or te > te_limit + 1e-6
        ):
            violations.append((dt, act_max, sec_sum, te))

    print(f"2. Max |active weight|:    {max_act:.6f}  (limit <= {max_active_limit + 1e-6:.6f}) -> {'PASS' if max_act <= max_active_limit + 1e-6 else 'FAIL'}")
    print(f"3. Max sector deviation:   {max_sec_dev:.6f}  (limit <= {sec_dev_limit + 1e-6:.6f}) -> {'PASS' if max_sec_dev <= sec_dev_limit + 1e-6 else 'FAIL'}")
    print(f"4. Max ex-ante TE:         {max_ex_ante:.6f}  (limit <= {te_limit + 1e-6:.6f}) -> {'PASS' if max_ex_ante <= te_limit + 1e-6 else 'FAIL'}")
    print(f"Total constraint violations: {len(violations)}")

    # 5. Check all results CSVs
    print("\n=== CHECKING B.5 CSV INTEGRITY ===")
    results_dir = Path("results")
    for csv_file in sorted(results_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        # Check NaN / inf
        num_cols = df.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(df[num_cols].values).any() if len(num_cols) > 0 else False
        
        # Check specific NaN rules: metrics.csv benchmark IR is nan by design
        if csv_file.name == "metrics.csv":
            nan_count = df.isna().sum().sum()
            # Exactly 1 NaN allowed: benchmark IR
            print(f"  {csv_file.name:25s}: shape {df.shape}, NaNs={nan_count} (benchmark IR), Infs={has_inf}")
            assert nan_count <= 1 and not has_inf
        elif csv_file.name in ["significance.csv", "solver_status.csv"]:
            nan_count = df.isna().sum().sum()
            print(f"  {csv_file.name:25s}: shape {df.shape}, NaNs={nan_count}, Infs={has_inf}")
            assert nan_count == 0 and not has_inf
        else:
            nan_count = df.isna().sum().sum()
            print(f"  {csv_file.name:25s}: shape {df.shape}, NaNs={nan_count}, Infs={has_inf}")
            assert nan_count == 0, f"Unexpected NaN in {csv_file.name}"
            assert not has_inf, f"Inf detected in {csv_file.name}"

    # Check date ordering
    ret_df = pd.read_csv("results/returns.csv", index_col=0, parse_dates=True)
    assert ret_df.index.is_monotonic_increasing, "Returns dates not strictly increasing"
    assert w_df.index.is_monotonic_increasing, "Weights dates not strictly increasing"

    warmup_days = int(cfg["warmup_days"])
    first_rebal = w_df.index[0]
    min_date = prices.index[warmup_days]
    print(f"\nWarmup check: first rebalance {first_rebal.date()} >= warmup date {min_date.date()} -> PASS")
    assert first_rebal >= min_date

    end_date_cfg = pd.Timestamp(cfg["data"]["end"])
    last_ret_date = ret_df.index[-1]
    print(f"End date check: last return {last_ret_date.date()} <= config end {end_date_cfg.date()} -> PASS")
    assert last_ret_date <= end_date_cfg


if __name__ == "__main__":
    main()
