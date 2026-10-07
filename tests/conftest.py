"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest
from ei.config import load_config
from ei.data import cap_weights
from ei.risk import estimate_cov
from ei.signals import compute_alpha
from ei.synthetic import make_synthetic


@pytest.fixture
def cfg():
    """Load configuration with test overrides for fast testing."""
    c = load_config("config.yaml")
    c["estimation_window"] = 120
    c["warmup_days"] = 120
    return c


@pytest.fixture
def synth():
    """Generate synthetic market dataset for tests."""
    return make_synthetic(n_assets=30, n_days=900, seed=42)


@pytest.fixture
def opt_inputs(synth, cfg):
    """Generate optimization inputs from synthetic fixture (shared across optimizer and backtest tests)."""
    prices, shares, sectors = synth
    w_b = cap_weights(prices, shares, prices.index[-1]).values
    returns = prices.pct_change().dropna().values
    sigma_ann, l_factor = estimate_cov(
        returns[-cfg["estimation_window"]:], method="ledoit_wolf"
    )
    alpha = compute_alpha(prices, sigma_ann, cfg).values
    return alpha, w_b, sigma_ann, l_factor, sectors
