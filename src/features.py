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



def sanitize_ohlc(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Drop or fix corrupted klines where high < low.

    In crypto there are no stock splits or dividend adjustments.
    If high < low, the data is corrupted (API anomaly).

    Args:
        df: DataFrame with OHLC columns.
        asset: Asset prefix ("btc" or "eth").

    Returns:
        DataFrame with corrupted rows handled.
    """
    h = f"{asset}_high"
    l = f"{asset}_low"

    if h in df.columns and l in df.columns:
        corrupted = df[h] < df[l]
        n = corrupted.sum()
        if n > 0:
            logger.warning("  %s: %d corrupted klines (high < low) — setting to NaN", asset.upper(), n)
            for col in [f"{asset}_open", h, l, f"{asset}_close"]:
                if col in df.columns:
                    df.loc[corrupted, col] = np.nan
            # Forward-fill the NaN (short gap)
            df[[f"{asset}_open", h, l, f"{asset}_close"]] = \
                df[[f"{asset}_open", h, l, f"{asset}_close"]].ffill(limit=1)

    return df



def compute_log_returns(
    close: pd.Series,
    windows: list[int],
    asset: str,
) -> pd.DataFrame:
    """
    Compute log returns over multiple horizons.

    Note: trading_mask : Forward-filled prices during maintenance produce returns of 0.0.
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



def compute_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int,
    asset: str,
    ann_factor: int = 1095,
) -> pd.Series:
    """
    Parkinson range-based volatility estimator (annualized).

    Uses high-low range assuming continuous Brownian motion.
    WARNING: In crypto, liquidation wicks create discontinuous jumps
    that inflate the high-low range. This estimator OVERESTIMATES
    volatility during wick events. See Garman-Klass note below.

    Args:
        high, low: High and low price series.
        window: Rolling window in 8h periods.
        asset: Asset prefix.
        ann_factor: Periods per year.

    Returns:
        Series: {asset}_parkinson_vol_{window}p
    """
    hl_log = np.log(high / low)
    pk_var = (hl_log ** 2) / (4 * np.log(2))
    rolling_pk = pk_var.rolling(window, min_periods=window // 2).mean()
    vol = np.sqrt(rolling_pk) * np.sqrt(ann_factor)
    vol.name = f"{asset}_parkinson_vol_{window}p"
    return vol


def compute_garman_klass_volatility(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    asset: str,
    ann_factor: int = 1095,
) -> pd.Series:
    """
    Garman-Klass volatility estimator using full OHLC data (annualized).

    Optionally accelerated via Rust (PyO3). Falls back to Python if
    the Rust module is not built.

    WARNING: Like Parkinson, GK assumes continuous diffusion. In crypto,
    liquidation cascades create jump discontinuities where price
    teleports through empty order book levels. GK is a LOWER-BOUND
    estimator during extreme stress (jumps are not captured).
    Future work: Bipower Variation to isolate continuous vs jump vol.

    Args:
        open_price, high, low, close: OHLC series.
        window: Rolling window.
        asset: Asset prefix.
        ann_factor: Periods per year.

    Returns:
        Series: {asset}_gk_vol_{window}p
    """
    # Try Rust implementation first
    try:
        from crypto_volatility_rust import garman_klass_volatility as _gk_rust
        result = _gk_rust(
            np.ascontiguousarray(open_price.values, dtype=np.float64),
            np.ascontiguousarray(high.values, dtype=np.float64),
            np.ascontiguousarray(low.values, dtype=np.float64),
            np.ascontiguousarray(close.values, dtype=np.float64),
            window,
            float(ann_factor),
        )
        logger.info("  Using Rust Garman-Klass (zero-copy FFI)")
        vol = pd.Series(result, index=close.index, name=f"{asset}_gk_vol_{window}p")
        return vol
    except ImportError:
        logger.info("  Rust module not available — using Python Garman-Klass fallback")

    # Python fallback
    hl = np.log(high / low)
    co = np.log(close / open_price)
    gk_var = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    rolling_gk = gk_var.rolling(window, min_periods=window // 2).mean()
    vol = np.sqrt(rolling_gk.clip(lower=0)) * np.sqrt(ann_factor)
    vol.name = f"{asset}_gk_vol_{window}p"
    return vol


# def compute_funding_zscore(
#     funding: pd.Series,
#     window: int,
#     asset: str,
# ) -> pd.Series:
#     """
#     Rolling z-score on funding rate.

#     First version — will be replaced by empirical percentiles
#     because the funding rate is CAPPED by exchange rules (~0.03%).
#     When funding hits the cap for weeks, std crushes to zero and
#     z-score explodes to +15σ artificially.

#     Args:
#         funding: Funding rate series.
#         window: Rolling window.
#         asset: Asset prefix.

#     Returns:
#         Series: {asset}_funding_zscore_{window}p
#     """
#     roll_mean = funding.rolling(window, min_periods=window // 2).mean()
#     roll_std = funding.rolling(window, min_periods=window // 2).std()

#     zscore = np.where(roll_std > 1e-10, (funding - roll_mean) / roll_std, 0.0)

#     return pd.Series(zscore, index=funding.index, name=f"{asset}_funding_zscore_{window}p")



def compute_funding_percentile(
    funding: pd.Series,
    window: int,
    asset: str,
) -> pd.Series:
    """
    Rolling empirical percentile rank for funding rate.

    Used instead of z-score because the funding rate is CAPPED by
    exchange rules. When funding hits the cap, std → 0 and z-score
    explodes artificially. Percentiles make no distributional assumptions.

    CRITICAL: Uses ROLLING window, not full-sample rank.
    Full-sample rank() would be look-ahead bias (the percentile at
    date t would include future information).

    Args:
        funding: Funding rate series.
        window: Rolling lookback window.
        asset: Asset prefix.

    Returns:
        Series between 0 and 1: {asset}_funding_pctile_{window}p
    """
    pctile = funding.rolling(window, min_periods=window // 2).rank() / window
    pctile.name = f"{asset}_funding_pctile_{window}p"
    return pctile


def compute_funding_zscore_clipped(
    funding: pd.Series,
    window: int,
    asset: str,
    clip_value: float = 5.0,
) -> pd.Series:
    """
    Rolling z-score on funding rate, clipped to ±clip_value.

    SECONDARY indicator only. The primary metric is the rolling percentile.
    Clipping prevents the +15σ artifacts when funding is capped.

    Args:
        funding: Funding rate series.
        window: Rolling window.
        asset: Asset prefix.
        clip_value: Max absolute z-score value.

    Returns:
        Series: {asset}_funding_zscore_clipped_{window}p
    """
    roll_mean = funding.rolling(window, min_periods=window // 2).mean()
    roll_std = funding.rolling(window, min_periods=window // 2).std()

    zscore = np.where(roll_std > 1e-10, (funding - roll_mean) / roll_std, 0.0)
    zscore = np.clip(zscore, -clip_value, clip_value)

    return pd.Series(zscore, index=funding.index,
                     name=f"{asset}_funding_zscore_clipped_{window}p")

def compute_volume_profile(
    volume: pd.Series,
    windows: list[int],
    asset: str,
) -> pd.DataFrame:
    """
    Rolling volume moving averages and short/long ratio.

    Args:
        volume: Volume series.
        windows: List of MA windows (e.g., [21, 90]).
        asset: Asset prefix.

    Returns:
        DataFrame with MA columns and ratio.
    """
    result = pd.DataFrame(index=volume.index)

    for w in windows:
        result[f"{asset}_volume_ma_{w}p"] = volume.rolling(w, min_periods=w // 2).mean()

    if len(windows) >= 2:
        short_w, long_w = windows[0], windows[-1]
        short_ma = result[f"{asset}_volume_ma_{short_w}p"]
        long_ma = result[f"{asset}_volume_ma_{long_w}p"]
        result[f"{asset}_volume_ratio_{short_w}p_{long_w}p"] = np.where(
            long_ma > 0, short_ma / long_ma, 1.0
        )

    return result


def compute_oi_volume_ratio(
    oi_coins: pd.Series,
    volume: pd.Series,
    asset: str,
) -> pd.Series:
    """
    Open Interest (in coins) / Volume ratio — leverage proxy.

    Uses coin-denominated OI (not USD) to avoid the nominal illusion
    where price appreciation inflates OI mechanically.

    Args:
        oi_coins: Open interest in coins (normalized in ingestion).
        volume: Trading volume.
        asset: Asset prefix.

    Returns:
        Series: {asset}_oi_volume_ratio
    """
    ratio = np.where(volume > 0, oi_coins / volume, np.nan)
    return pd.Series(ratio, index=volume.index, name=f"{asset}_oi_volume_ratio")




def compute_drawdown(
    close: pd.Series,
    asset: str,
    local_window: int = 90,
) -> pd.DataFrame:
    """
    Compute drawdown from ATH (global) and from rolling high (local).

    Global ATH drawdown: useful for EDA visualization.
    Local rolling drawdown (30d = 90 periods): useful for market makers.
    A market maker doesn't care if BTC is -60% from ATH two years ago.
    They care if the market is liquidating longs RIGHT NOW.

    Args:
        close: Close price series.
        asset: Asset prefix.
        local_window: Rolling window for local drawdown (default 90p = 30d).

    Returns:
        DataFrame with global and local drawdown columns.
    """
    result = pd.DataFrame(index=close.index)

    # Global drawdown from ATH
    running_max = close.cummax()
    result[f"{asset}_drawdown_global"] = (close - running_max) / running_max

    # Local rolling drawdown (what market makers watch)
    rolling_max = close.rolling(local_window, min_periods=1).max()
    result[f"{asset}_drawdown_local_{local_window}p"] = (close - rolling_max) / rolling_max

    # Drawdown duration (consecutive periods below previous high)
    is_dd = close < running_max
    dd_groups = (~is_dd).cumsum()
    result[f"{asset}_drawdown_duration"] = is_dd.groupby(dd_groups).cumsum()

    return result




