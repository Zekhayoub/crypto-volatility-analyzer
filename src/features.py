"""
Feature engineering for crypto market analysis.

Computes volatility estimators, funding rate analytics, volume profile,
correlation metrics, and drawdown indicators. All rolling windows are
in 8-hour periods. The trading_mask from ingestion is used to exclude
maintenance periods from volatility calculations.
"""

import logging

import numpy as np
import pandas as pd

from src.config import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)

ASSETS = ["btc", "eth"]


def compute_log_returns(
    close: pd.Series,
    windows: list[int],
    asset: str,
) -> pd.DataFrame:
    """
    Compute log returns over multiple horizons.

    Note: Forward-filled prices during maintenance produce returns of 0.0.
    These artificial zeros will be handled by the trading_mask when
    computing volatility (commit 25). The returns themselves are kept
    as-is because a flat price IS the correct observation during a gap.

    Args:
        close: Close price series.
        windows: List of lookback periods in 8h periods.
        asset: Asset prefix ("btc" or "eth").

    Returns:
        DataFrame with columns: {asset}_return_{N}p
    """
    result = pd.DataFrame(index=close.index)
    for w in windows:
        result[f"{asset}_return_{w}p"] = np.log(close / close.shift(w))
    return result




def compute_realized_volatility(
    returns_1p: pd.Series,
    windows: list[int],
    asset: str,
    ann_factor: int = 1095,
    trading_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Rolling realized volatility (annualized).

    If trading_mask is provided, artificial zero returns from
    forward-filled maintenance periods are excluded. Without this,
    zero returns compress the rolling std and underestimate risk.

    Annualization uses sqrt(1095) = sqrt(365 * 3), which assumes
    i.i.d. returns. Given intraday autocorrelation in crypto, this
    is a standard underestimation of true annualized risk.

    Args:
        returns_1p: 1-period log returns.
        windows: Rolling windows in 8h periods.
        asset: Asset prefix.
        ann_factor: Periods per year (1095 for 8h).
        trading_mask: Boolean — True for real trading periods.

    Returns:
        DataFrame with columns: {asset}_realized_vol_{N}p
    """
    returns = returns_1p.copy()

    if trading_mask is not None:
        # Exclude maintenance periods (returns = 0.0 from forward-fill)
        returns = returns.where(trading_mask)
        logger.debug("Volatility: excluded %d non-trading periods", (~trading_mask).sum())

    result = pd.DataFrame(index=returns_1p.index)
    for w in windows:
        vol = returns.rolling(w, min_periods=w // 2).std() * np.sqrt(ann_factor)
        result[f"{asset}_realized_vol_{w}p"] = vol

    return result


