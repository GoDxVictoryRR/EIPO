"""Alpha signals calculation module."""

import logging
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def zscore(x: pd.Series, ddof: int = 0) -> pd.Series:
    """Compute cross-sectional z-score with mean 0 and std 1."""
    std_val = float(x.std(ddof=ddof))
    if std_val < 1e-12 or np.isnan(std_val):
        return pd.Series(0.0, index=x.index)
    return (x - float(x.mean())) / std_val


def compute_signal_components(
    prices_upto_t: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, pd.Series]:
    """Compute raw, winsorized, and z-scored signal components."""
    winsor_z = float(cfg["signals"]["winsor_z"])
    window = int(cfg.get("estimation_window", 252))

    # 1. Momentum: P[t-21] / P[t-252] - 1
    if len(prices_upto_t) >= 253:
        p_t21 = prices_upto_t.iloc[-22]
        p_t252 = prices_upto_t.iloc[-253]
    else:
        p_t21 = prices_upto_t.iloc[-22] if len(prices_upto_t) >= 22 else prices_upto_t.iloc[0]
        p_t252 = prices_upto_t.iloc[0]
    raw_mom = p_t21 / p_t252 - 1.0

    # 2. Reversal: -(P[t] / P[t-21] - 1)
    p_t = prices_upto_t.iloc[-1]
    raw_rev = -(p_t / p_t21 - 1.0)

    # 3. Low volatility: -std(daily returns over window)
    lookback_slice = prices_upto_t.iloc[-(window + 1) :]
    daily_rets = lookback_slice.pct_change().dropna()
    raw_lowvol = -daily_rets.std(ddof=1)

    # Process each component: z-score -> winsorize -> re-z-score
    z_mom_initial = zscore(raw_mom, ddof=0)
    z_mom_winsor = z_mom_initial.clip(lower=-winsor_z, upper=winsor_z)
    z_mom = zscore(z_mom_winsor, ddof=0)

    z_rev_initial = zscore(raw_rev, ddof=0)
    z_rev_winsor = z_rev_initial.clip(lower=-winsor_z, upper=winsor_z)
    z_rev = zscore(z_rev_winsor, ddof=0)

    z_vol_initial = zscore(raw_lowvol, ddof=0)
    z_vol_winsor = z_vol_initial.clip(lower=-winsor_z, upper=winsor_z)
    z_vol = zscore(z_vol_winsor, ddof=0)

    # Composite combination
    weights = cfg["signals"]["weights"]
    w_mom = float(weights.get("momentum", 0.5))
    w_rev = float(weights.get("reversal", 0.25))
    w_vol = float(weights.get("lowvol", 0.25))

    composite_raw = w_mom * z_mom + w_rev * z_rev + w_vol * z_vol
    composite_z = zscore(composite_raw, ddof=0)

    return {
        "raw_momentum": raw_mom,
        "z_mom_initial": z_mom_initial,
        "z_mom_winsor": z_mom_winsor,
        "momentum": z_mom,
        "raw_reversal": raw_rev,
        "z_rev_initial": z_rev_initial,
        "z_rev_winsor": z_rev_winsor,
        "reversal": z_rev,
        "raw_lowvol": raw_lowvol,
        "z_vol_initial": z_vol_initial,
        "z_vol_winsor": z_vol_winsor,
        "lowvol": z_vol,
        "composite": composite_z,
    }


def compute_alpha(
    prices_upto_t: pd.DataFrame, Sigma_ann: np.ndarray, cfg: dict[str, Any]
) -> pd.Series:
    """Compute expected annualized alpha vector aligned by ticker."""
    components = compute_signal_components(prices_upto_t, cfg)
    composite_z = components["composite"]

    ic = float(cfg["signals"]["ic"])
    sigma_i = np.sqrt(np.diag(Sigma_ann))

    alpha_values = ic * sigma_i * composite_z.values
    alpha_series = pd.Series(alpha_values, index=prices_upto_t.columns, name="alpha")

    if alpha_series.isna().any() or np.isinf(alpha_series).any():
        raise ValueError("NaN or Inf detected in expected alpha scores")

    return alpha_series
