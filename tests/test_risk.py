"""Unit tests for risk and covariance estimation."""

import numpy as np
import pytest

from ei.risk import estimate_cov


def test_sigma_properties_and_cholesky(synth):
    """Verify Sigma is symmetric, strictly positive definite, and L @ L.T equals Sigma."""
    prices, _, _ = synth
    returns = prices.pct_change().dropna().values

    for method in ["sample", "ledoit_wolf"]:
        sigma_ann, l_factor = estimate_cov(returns, method=method)

        # Symmetry check
        assert np.allclose(sigma_ann, sigma_ann.T, atol=1e-8)

        # Positive definiteness (min eigenvalue > 0)
        eigvals = np.linalg.eigvalsh(sigma_ann)
        assert np.min(eigvals) > 0.0

        # Cholesky factor reconstruction
        reconstructed = l_factor @ l_factor.T
        assert np.allclose(reconstructed, sigma_ann, atol=1e-6)


def test_ledoit_wolf_condition_number(synth):
    """Verify Ledoit-Wolf has smaller condition number than sample covariance."""
    prices, _, _ = synth
    returns = prices.pct_change().dropna().values

    sigma_sample, _ = estimate_cov(returns, method="sample")
    sigma_lw, _ = estimate_cov(returns, method="ledoit_wolf")

    cond_sample = np.linalg.cond(sigma_sample)
    cond_lw = np.linalg.cond(sigma_lw)

    assert cond_lw < cond_sample


def test_invalid_covariance_inputs():
    """Verify error handling on invalid method or NaN returns."""
    with pytest.raises(ValueError):
        estimate_cov(np.ones((20, 5)), method="invalid_method")

    bad_returns = np.ones((20, 5))
    bad_returns[0, 0] = np.nan
    with pytest.raises(ValueError):
        estimate_cov(bad_returns)
