"""Risk and covariance estimation module."""

import logging
from typing import Union
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def estimate_cov(
    returns: Union[np.ndarray, pd.DataFrame], method: str = "ledoit_wolf"
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate annualized covariance matrix and its Cholesky factor L."""
    if isinstance(returns, pd.DataFrame):
        r_arr = returns.values
    else:
        r_arr = np.asarray(returns)

    if np.any(np.isnan(r_arr)) or np.any(np.isinf(r_arr)):
        raise ValueError("Returns contain NaN or Inf values")

    n_assets = r_arr.shape[1]

    if method == "sample":
        s_daily = np.cov(r_arr, rowvar=False)
    elif method == "ledoit_wolf":
        lw = LedoitWolf().fit(r_arr)
        s_daily = lw.covariance_
    else:
        raise ValueError(f"Unknown covariance estimation method: {method}")

    s_daily = (s_daily + s_daily.T) / 2.0
    sigma_ann = 252.0 * s_daily
    sigma_ann = (sigma_ann + sigma_ann.T) / 2.0 + 1e-8 * np.eye(n_assets)

    eigvals = np.linalg.eigvalsh(sigma_ann)
    min_eig = np.min(eigvals)
    if min_eig <= 0:
        raise ValueError(
            f"Covariance matrix is not positive definite: min eigenvalue = {min_eig}"
        )

    l_factor = np.linalg.cholesky(sigma_ann)

    return sigma_ann, l_factor
