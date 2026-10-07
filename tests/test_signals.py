"""Unit tests for signal components and alpha computation."""

import copy
import numpy as np
import pandas as pd
import pytest

from ei.risk import estimate_cov
from ei.signals import compute_alpha, compute_signal_components


def test_zscore_properties_and_winsorization(synth, cfg):
    """Verify each component z-score mean ~0, std ~1, and winsorization bounds."""
    prices, _, _ = synth
    winsor_z = float(cfg["signals"]["winsor_z"])

    components = compute_signal_components(prices, cfg)

    # Check component and composite z-scores
    for key in ["momentum", "reversal", "lowvol", "composite"]:
        z = components[key]
        assert abs(z.mean()) < 1e-9
        assert abs(z.std(ddof=0) - 1.0) < 1e-6

    # Winsorization checks: clamped before re-z-score, and bounded after
    for key in ["z_mom_winsor", "z_rev_winsor", "z_vol_winsor"]:
        z_win = components[key]
        assert z_win.abs().max() <= winsor_z + 1e-8

    for key in ["momentum", "reversal", "lowvol"]:
        z_final = components[key]
        assert z_final.abs().max() <= winsor_z * 1.5 + 1e-6


def test_no_lookahead(synth, cfg):
    """Verify compute_alpha produces identical results regardless of future price changes."""
    prices, _, _ = synth
    t = prices.index[400]

    prices_upto_t = prices.loc[:t]
    returns_upto_t = prices_upto_t.pct_change().dropna().values
    sigma_ann, _ = estimate_cov(returns_upto_t[-cfg["estimation_window"]:], method="ledoit_wolf")

    alpha_base = compute_alpha(prices_upto_t, sigma_ann, cfg)

    prices_altered = prices.copy()
    prices_altered.loc[prices.index > t] *= 5.0
    alpha_altered = compute_alpha(prices_altered.loc[:t], sigma_ann, cfg)

    pd.testing.assert_series_equal(alpha_base, alpha_altered)


def test_alpha_scales_linearly_with_ic(synth, cfg):
    """Verify alpha scales strictly linearly with information coefficient."""
    prices, _, _ = synth
    returns = prices.pct_change().dropna().values
    sigma_ann, _ = estimate_cov(returns[-cfg["estimation_window"]:], method="ledoit_wolf")

    cfg1 = copy.deepcopy(cfg)
    cfg1["signals"]["ic"] = 0.05
    cfg2 = copy.deepcopy(cfg)
    cfg2["signals"]["ic"] = 0.10

    alpha1 = compute_alpha(prices, sigma_ann, cfg1)
    alpha2 = compute_alpha(prices, sigma_ann, cfg2)

    np.testing.assert_allclose(alpha2.values, 2.0 * alpha1.values, atol=1e-6)
