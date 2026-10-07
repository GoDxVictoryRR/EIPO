"""Network-dependent tests for real data download and index correlation verification."""

import logging
import numpy as np
import pytest

from ei.backtest import run_backtest
from ei.data import load_benchmark_index, load_prices, load_sectors, load_shares

logger = logging.getLogger(__name__)


@pytest.mark.network
def test_downloaded_universe_cleaning(cfg):
    """Verify real data download yields >= 30 assets with no NaNs after cleaning."""
    prices = load_prices(cfg)
    assert len(prices.columns) >= 30
    assert not prices.isna().any().any()


@pytest.mark.network
def test_benchmark_correlation_with_nsei(cfg):
    """Verify built cap-weighted benchmark has > 0.95 daily correlation with ^NSEI."""
    prices = load_prices(cfg)
    tickers = list(prices.columns)
    shares = load_shares(tickers, cfg)
    sectors = load_sectors(tickers, cfg)
    nsei = load_benchmark_index(cfg)

    res_b = run_backtest(prices, shares, sectors, cfg, strategy="benchmark")
    nsei_rets = nsei.pct_change().dropna()

    common = res_b["returns"].index.intersection(nsei_rets.index)
    corr = float(np.corrcoef(res_b["returns"].loc[common], nsei_rets.loc[common])[0, 1])

    logger.info("Daily return correlation between built benchmark and ^NSEI: %.4f", corr)
    assert corr > 0.95  # calendar-filter fix achieves ~0.988
