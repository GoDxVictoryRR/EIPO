"""
update_golden.py    Regenerate tests/golden/metrics_synth.csv.

Run MANUALLY when you intentionally change the strategy or config and want
to accept the new numbers as the new baseline:

    python scripts/update_golden.py

Do NOT run this in CI or as part of run_all.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from ei.backtest import run_all
from ei.config import load_config
from ei.metrics import metrics_table
from ei.synthetic import make_synthetic

cfg = load_config("config.yaml")
cfg["estimation_window"] = 120
cfg["warmup_days"] = 120

prices, shares, sectors = make_synthetic(n_assets=30, n_days=900, seed=42)
results = run_all(prices, shares, sectors, cfg)
mt = metrics_table(results)

out = Path("tests/golden/metrics_synth.csv")
out.parent.mkdir(parents=True, exist_ok=True)
mt.to_csv(out)
print("Golden file updated:")
print(mt.round(8).to_string())
print(f"\nSaved to {out}")
