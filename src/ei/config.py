"""Configuration loading and validation module."""

from pathlib import Path
from typing import Any, Union
import yaml

REQUIRED_KEYS = [
    "seed",
    "data",
    "estimation_window",
    "rebalance",
    "warmup_days",
    "optimizer",
    "costs",
    "signals",
    "covariance",
    "risk_free_annual",
    "sweep",
]

REQUIRED_DATA_KEYS = [
    "start",
    "end",
    "cache_dir",
    "benchmark_index",
    "max_missing_frac",
]

REQUIRED_OPTIMIZER_KEYS = [
    "risk_aversion",
    "te_max",
    "max_active_weight",
    "sector_dev_max",
    "solver",
]

REQUIRED_COSTS_KEYS = [
    "tc_bps",
]

REQUIRED_SIGNALS_KEYS = [
    "ic",
    "weights",
    "winsor_z",
]

REQUIRED_SWEEP_KEYS = [
    "te_max",
]


def load_config(path: Union[str, Path] = "config.yaml") -> dict[str, Any]:
    """Load and validate the YAML configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a dictionary")

    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required configuration key: {key}")

    for key in REQUIRED_DATA_KEYS:
        if key not in cfg["data"]:
            raise KeyError(f"Missing required key in config['data']: {key}")

    for key in REQUIRED_OPTIMIZER_KEYS:
        if key not in cfg["optimizer"]:
            raise KeyError(f"Missing required key in config['optimizer']: {key}")

    for key in REQUIRED_COSTS_KEYS:
        if key not in cfg["costs"]:
            raise KeyError(f"Missing required key in config['costs']: {key}")

    for key in REQUIRED_SIGNALS_KEYS:
        if key not in cfg["signals"]:
            raise KeyError(f"Missing required key in config['signals']: {key}")

    for key in REQUIRED_SWEEP_KEYS:
        if key not in cfg["sweep"]:
            raise KeyError(f"Missing required key in config['sweep']: {key}")

    return cfg
