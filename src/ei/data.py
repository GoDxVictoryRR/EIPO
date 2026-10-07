"""Data loading, caching, cleaning, and market-cap weighting."""

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yfinance as yf

from ei.universe import get_universe

logger = logging.getLogger(__name__)


def clean_prices(prices: pd.DataFrame, max_missing_frac: float = 0.05) -> pd.DataFrame:
    """Clean prices by dropping high-missing tickers, forward-filling gaps <= 3 days, and dropping remaining NaNs."""
    if prices.empty:
        return prices.copy()

    # Drop any ticker with > max_missing_frac missing prices over the period
    missing_frac = prices.isna().mean()
    valid_cols = missing_frac[missing_frac <= max_missing_frac].index
    df = prices[valid_cols].copy()

    # Forward-fill gaps <= 3 days
    df = df.ffill(limit=3)

    # Drop any remaining columns with NaN
    df = df.dropna(axis=1)

    if len(df.columns) < 30:
        logger.warning(
            "Cleaned prices have %d assets (< 30 assets recommended).", len(df.columns)
        )

    return df


def load_prices(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load daily adjusted close prices, caching to disk and applying cleaning.

    After cleaning, the price index is intersected with the ^NSEI benchmark
    trading calendar so that spurious NSE-holiday dates (where yfinance returns
    individual-stock prices but the index doesn't trade) are removed.  This
    ensures the portfolio return series lives on exactly the same dates as the
    benchmark index, giving a correct daily-return correlation.
    """
    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "prices.csv"

    if cache_file.is_file():
        logger.info("Loading cached prices from %s", cache_file)
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        tickers = get_universe()
        logger.info(
            "Downloading prices for %d tickers from %s to %s",
            len(tickers),
            cfg["data"]["start"],
            cfg["data"]["end"],
        )
        downloaded = yf.download(
            tickers,
            start=cfg["data"]["start"],
            end=cfg["data"]["end"],
            auto_adjust=True,
            progress=False,
        )
        if isinstance(downloaded.columns, pd.MultiIndex):
            prices = downloaded["Close"]
        else:
            prices = downloaded
        prices.to_csv(cache_file)

    max_missing = cfg["data"].get("max_missing_frac", 0.05)
    cleaned = clean_prices(prices, max_missing_frac=max_missing)

    # Intersect to the ^NSEI trading calendar to eliminate ~13 spurious dates
    # (national holidays) where yfinance returns individual-stock prices but the
    # index does not trade.  Without this filter those dates split a single
    # overnight return into two portfolio entries, destroying benchmark correlation.
    bench_cache = cache_dir / "benchmark_index.csv"
    if bench_cache.is_file():
        bench_index = pd.read_csv(bench_cache, index_col=0, parse_dates=True)
        nsei_dates = bench_index.index
        common_dates = cleaned.index.intersection(nsei_dates)
        n_removed = len(cleaned.index) - len(common_dates)
        if n_removed > 0:
            logger.info(
                "Removing %d price dates not in ^NSEI calendar (holiday dates with stale yfinance data)",
                n_removed,
            )
        cleaned = cleaned.loc[common_dates]

    return cleaned


def load_shares(tickers: list[str], cfg: dict[str, Any]) -> pd.Series:
    """Load static shares outstanding per ticker with disk caching."""
    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "shares.csv"

    if cache_file.is_file():
        logger.info("Loading cached shares from %s", cache_file)
        series = pd.read_csv(cache_file, index_col=0).squeeze("columns")
        return series.astype(float).reindex(tickers).fillna(1e8)

    shares_dict: dict[str, float] = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            val = t.info.get("sharesOutstanding")
            shares_dict[ticker] = float(val) if val is not None else 1e8
        except Exception as e:
            logger.warning("Error fetching shares for %s: %s", ticker, e)
            shares_dict[ticker] = 1e8

    series = pd.Series(shares_dict, name="shares")
    series.to_csv(cache_file)
    return series.astype(float)


def load_sectors(tickers: list[str], cfg: dict[str, Any]) -> pd.Series:
    """Load sector classification per ticker with disk caching."""
    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "sectors.csv"

    if cache_file.is_file():
        logger.info("Loading cached sectors from %s", cache_file)
        series = pd.read_csv(cache_file, index_col=0).squeeze("columns")
        return series.astype(str).reindex(tickers).fillna("Unknown")

    sectors_dict: dict[str, str] = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            sec = t.info.get("sector")
            sectors_dict[ticker] = str(sec) if sec is not None else "Unknown"
        except Exception as e:
            logger.warning("Error fetching sector for %s: %s", ticker, e)
            sectors_dict[ticker] = "Unknown"

    series = pd.Series(sectors_dict, name="sector")
    series.to_csv(cache_file)
    return series.astype(str)


def load_benchmark_index(cfg: dict[str, Any]) -> pd.Series:
    """Load benchmark index close prices with disk caching."""
    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "benchmark_index.csv"

    if cache_file.is_file():
        logger.info("Loading cached benchmark index from %s", cache_file)
        series = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze("columns")
        return series.astype(float)

    b_ticker = cfg["data"].get("benchmark_index", "^NSEI")
    logger.info("Downloading benchmark index %s", b_ticker)
    downloaded = yf.download(
        b_ticker,
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        auto_adjust=True,
        progress=False,
    )
    if isinstance(downloaded, pd.DataFrame) and "Close" in downloaded.columns:
        series = downloaded["Close"].squeeze("columns")
    else:
        series = downloaded.squeeze("columns")

    series.name = b_ticker
    series.to_csv(cache_file)
    return series.astype(float)


def cap_weights(prices: pd.DataFrame, shares: pd.Series, date: Any) -> pd.Series:
    """Compute market-cap benchmark weights summing to 1 at the specified date."""
    if date not in prices.index:
        available = prices.index[prices.index <= date]
        if len(available) == 0:
            raise KeyError(f"Date {date} is earlier than first available price date.")
        date = available[-1]

    row = prices.loc[date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]

    common = row.index.intersection(shares.index)
    if len(common) == 0:
        raise ValueError(f"No common assets between prices and shares at date {date}")

    p = row[common].astype(float)
    s = shares[common].astype(float)
    mcap = p * s

    total = mcap.sum()
    if total <= 0 or np.isnan(total):
        raise ValueError(f"Invalid total market cap at date {date}: {total}")

    w = (mcap / total).clip(lower=0.0)
    w = w / w.sum()
    return w
