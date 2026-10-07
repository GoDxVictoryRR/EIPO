"""Synthetic data generator."""

import numpy as np
import pandas as pd


def make_synthetic(
    n_assets: int = 40, n_days: int = 2500, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate reproducible synthetic prices, shares, and sectors."""
    rng = np.random.default_rng(seed)

    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]
    dates = pd.bdate_range(start="2015-01-01", periods=n_days)

    sector_names = ["Technology", "Financials", "Consumer", "Healthcare", "Industrials"]
    asset_sectors = [sector_names[i % len(sector_names)] for i in range(n_assets)]
    sectors = pd.Series(asset_sectors, index=tickers, name="sector")

    log_shares = rng.normal(loc=18.0, scale=1.0, size=n_assets)
    shares = pd.Series(np.exp(log_shares), index=tickers, name="shares")

    # Factors: market factor (18% ann vol), 5 sector factors (10% ann vol)
    daily_mkt_vol = 0.18 / np.sqrt(252)
    daily_sector_vol = 0.10 / np.sqrt(252)

    r_m = rng.normal(loc=0.08 / 252, scale=daily_mkt_vol, size=n_days)
    f_s = rng.normal(loc=0.0, scale=daily_sector_vol, size=(n_days, len(sector_names)))

    betas = rng.uniform(0.7, 1.3, size=n_assets)
    idio_vols = rng.uniform(0.15, 0.35, size=n_assets) / np.sqrt(252)
    idio = rng.normal(loc=0.0, scale=1.0, size=(n_days, n_assets)) * idio_vols

    # Planted drift mu_i ~ N(0, (0.05/252 * 5.0)^2)
    mu = rng.normal(loc=0.0, scale=(0.05 / 252) * 5.0, size=n_assets)

    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        sec_idx = i % len(sector_names)
        returns[:, i] = mu[i] + betas[i] * r_m + f_s[:, sec_idx] + idio[:, i]

    # Clip large single-day losses to maintain positive prices
    returns = np.clip(returns, -0.5, 0.5)

    price_matrix = 100.0 * np.cumprod(1.0 + returns, axis=0)
    prices = pd.DataFrame(price_matrix, index=dates, columns=tickers)

    return prices, shares, sectors
