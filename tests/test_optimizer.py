"""Unit tests for portfolio optimization routines."""

import copy
import cvxpy as cp
import numpy as np
import pytest

from ei.data import cap_weights
from ei.optimizer import optimize_enhanced, optimize_mv
from ei.risk import estimate_cov
from ei.signals import compute_alpha



# opt_inputs fixture is defined in conftest.py and shared with test_backtest_hardened.py


def test_basic_enhanced_constraints(opt_inputs, cfg):
    """Verify weights sum to 1, long-only, active weights, sector limits, and ex-ante TE."""
    alpha, w_b, sigma_ann, l_factor, sectors = opt_inputs
    w, info = optimize_enhanced(alpha, w_b, w_b, sigma_ann, l_factor, sectors, cfg)

    # Weights sum to 1 and long-only
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    assert np.min(w) >= -1e-8

    # Active weight limits
    a = w - w_b
    max_active = cfg["optimizer"]["max_active_weight"]
    assert np.all(np.abs(a) <= max_active + 1e-6)

    # Sector active deviation limits
    sector_dev_max = cfg["optimizer"]["sector_dev_max"]
    for sec in set(sectors):
        if sec == "Unknown":
            continue
        sec_idx = [i for i, s in enumerate(sectors) if s == sec]
        assert abs(np.sum(a[sec_idx])) <= sector_dev_max + 1e-6

    # Ex-ante TE independently computed with numpy
    te_np = np.sqrt(max(0.0, a @ sigma_ann @ a))
    assert te_np <= cfg["optimizer"]["te_max"] + 1e-6
    assert np.isclose(te_np, info["ex_ante_te"], atol=1e-5)


def test_zero_alpha_and_tiny_te(opt_inputs, cfg):
    """Verify alpha=0 yields w ~ w_b (atol 1e-5) and tiny te_max yields w ~ w_b (atol 1e-4).

    alpha=0 check uses 1e-5: optimizer has no gradient to exploit, should stay at w_b.
    tiny_te check uses 1e-4: te_max=1e-6 is a degenerate near-zero constraint; CVXPY
    numerical tolerance for this degenerate case is only ~1e-4.
    """
    alpha, w_b, sigma_ann, l_factor, sectors = opt_inputs

    # Zero alpha: optimizer has nothing to gain → should stay at w_b within 1e-5
    alpha_zero = np.zeros_like(w_b)
    w_zero, _ = optimize_enhanced(alpha_zero, w_b, w_b, sigma_ann, l_factor, sectors, cfg)
    assert np.allclose(w_zero, w_b, atol=1e-5)

    # Tiny te_max: constraint forces w ≈ w_b; solver tolerance for degenerate problem ~1e-4
    cfg_tiny = copy.deepcopy(cfg)
    cfg_tiny["optimizer"]["te_max"] = 1e-6
    w_tiny, _ = optimize_enhanced(alpha, w_b, w_b, sigma_ann, l_factor, sectors, cfg_tiny)
    assert np.allclose(w_tiny, w_b, atol=1e-4)


def test_monotone_te_max_feasibility(opt_inputs, cfg):
    """Verify increasing te_max expands the feasibility set, yielding non-decreasing alpha^T a."""
    alpha, w_b, sigma_ann, l_factor, sectors = opt_inputs

    te_list = [0.005, 0.01, 0.02, 0.03, 0.04]
    alpha_active_vals = []

    for te in te_list:
        cfg_sweep = copy.deepcopy(cfg)
        cfg_sweep["optimizer"]["te_max"] = te
        w, _ = optimize_enhanced(alpha, w_b, w_b, sigma_ann, l_factor, sectors, cfg_sweep)
        a = w - w_b
        alpha_active_vals.append(float(alpha @ a))

    for k in range(len(alpha_active_vals) - 1):
        assert alpha_active_vals[k + 1] >= alpha_active_vals[k] - 1e-6


def test_transaction_cost_penalization(opt_inputs, cfg):
    """Verify high tc_bps reduces turnover relative to zero tc_bps."""
    alpha, w_b, sigma_ann, l_factor, sectors = opt_inputs
    w_prev = np.ones_like(w_b) / len(w_b)

    cfg_zero = copy.deepcopy(cfg)
    cfg_zero["costs"]["tc_bps"] = 0.0
    w_zero, _ = optimize_enhanced(alpha, w_b, w_prev, sigma_ann, l_factor, sectors, cfg_zero)
    turnover_zero = np.sum(np.abs(w_zero - w_prev))

    cfg_high = copy.deepcopy(cfg)
    cfg_high["costs"]["tc_bps"] = 500.0
    w_high, _ = optimize_enhanced(alpha, w_b, w_prev, sigma_ann, l_factor, sectors, cfg_high)
    turnover_high = np.sum(np.abs(w_high - w_prev))

    assert turnover_high < turnover_zero


def test_failed_solve_fallback(opt_inputs, cfg, monkeypatch):
    """Verify fallback to w_prev and graceful failure status when solver fails."""
    alpha, w_b, sigma_ann, l_factor, sectors = opt_inputs
    w_prev = np.ones_like(w_b) / len(w_b)

    def mock_solve(*args, **kwargs):
        raise RuntimeError("Simulated solver failure")

    monkeypatch.setattr(cp.Problem, "solve", mock_solve)

    w_res, info = optimize_enhanced(alpha, w_b, w_prev, sigma_ann, l_factor, sectors, cfg)
    assert np.allclose(w_res, w_prev, atol=1e-6)
    assert info["status"] == "failed_fallback"


def test_mv_baseline(opt_inputs, cfg):
    """Verify mean-variance baseline output is long-only and sums to 1."""
    alpha, w_b, sigma_ann, _, _ = opt_inputs
    w_mv, info = optimize_mv(alpha, w_b, sigma_ann, cfg)

    assert np.isclose(np.sum(w_mv), 1.0, atol=1e-6)
    assert np.min(w_mv) >= -1e-8
