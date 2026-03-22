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



