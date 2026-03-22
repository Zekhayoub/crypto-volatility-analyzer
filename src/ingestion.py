"""
Multi-source data ingestion for crypto markets.

Sources:
    - Binance Spot: 8h OHLCV klines with taker buy volume
    - Binance Futures: funding rates (8h native), open interest (daily)
    - Hyperliquid: DEX funding rates (1h native, scaled to 8h)

All timestamps are UTC-aware. No hardcoded defaults in function signatures.
Volume is zero-filled (not forward-filled) during missing periods.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


def request_with_retry(
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs,
) -> requests.Response:
    """
    HTTP request with exponential backoff. Handles 429 rate limiting.

    Args:
        method: "GET" or "POST".
        url: Request URL.
        max_retries: Attempts before raising.
        **kwargs: Passed to requests.get/post.

    Returns:
        Response object.
    """
    resp = None
    for attempt in range(1, max_retries + 1):
        try:
            if method.upper() == "POST":
                resp = requests.post(url, timeout=30, **kwargs)
            else:
                resp = requests.get(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            status = getattr(resp, "status_code", None)
            if status == 429:
                wait = 2**attempt
                logger.warning("  Rate limited (429). Waiting %ds...", wait)
                time.sleep(wait)
            elif attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "  Attempt %d/%d failed (HTTP %s). Retrying in %ds...",
                    attempt,
                    max_retries,
                    status,
                    wait,
                )
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "  Attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"All {max_retries} attempts failed for {url}")


KLINE_COLUMNS = {
    0: "open_time_ms",
    1: "open",
    2: "high",
    3: "low",
    4: "close",
    5: "volume",
    6: "close_time_ms",
    7: "quote_volume",
    8: "n_trades",
    9: "taker_buy_volume",
    10: "taker_buy_quote_volume",
    11: "ignore",
}


def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_date: str,
    base_url: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance Spot API with pagination.

    Includes taker_buy_volume for Order Flow Imbalance.
    No default values — all params from config.yaml (DRY principle).

    Args:
        symbol: Trading pair (e.g., "BTCUSDT").
        interval: Candle interval (e.g., "8h").
        start_date: Start date YYYY-MM-DD (from config).
        base_url: Binance API URL (from config).

    Returns:
        DataFrame with columns: open, high, low, close, volume, taker_buy_volume.
        Index is NOT yet parsed (raw open_time_ms column present).
    """
    endpoint = f"{base_url}/api/v3/klines"
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    all_klines = []

    logger.info("Fetching %s klines (%s) from %s...", symbol, interval, start_date)

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": 1000,
        }

        resp = request_with_retry("GET", endpoint, params=params)
        data = resp.json()

        if not data:
            break

        all_klines.extend(data)
        start_ts = data[-1][6] + 1  # past close_time_ms
        time.sleep(0.1)

        if len(data) < 1000:
            break

    logger.info("  -> %d raw klines for %s", len(all_klines), symbol)

    if not all_klines:
        raise ValueError(f"No klines returned for {symbol}")

    df = pd.DataFrame(all_klines)
    df.columns = list(KLINE_COLUMNS.values())

    for col in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    # Binance timestamps are MILLISECONDS. Without unit="ms", dates are year 50000+.
    # utc=True prevents tz-naive/tz-aware crash when merging with Hyperliquid.
    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume", "taker_buy_volume"]]
    df = df[~df.index.duplicated(keep="first")]

    logger.info(
        "  -> %d klines parsed, %s to %s", len(df), df.index.min(), df.index.max()
    )

    return df


def fetch_binance_funding_rate(
    symbol: str,
    start_date: str,
    base_url: str,
) -> pd.DataFrame:
    """
    Fetch perpetual funding rates from Binance Futures (8h native).

    Args:
        symbol: Futures pair (from config).
        start_date: Start date (from config).
        base_url: Binance Futures URL (from config).

    Returns:
        DataFrame with UTC DatetimeIndex and column: funding_rate.
    """
    endpoint = f"{base_url}/fapi/v1/fundingRate"
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    all_records = []

    logger.info("Fetching %s funding rate from %s...", symbol, start_date)

    while True:
        params = {"symbol": symbol, "startTime": start_ts, "limit": 1000}

        resp = request_with_retry("GET", endpoint, params=params)
        data = resp.json()

        if not data:
            break

        all_records.extend(data)
        start_ts = data[-1]["fundingTime"] + 1
        time.sleep(0.1)

        if len(data) < 1000:
            break

    logger.info("  -> %d funding records for %s", len(all_records), symbol)

    if not all_records:
        raise ValueError(f"No funding rate data for {symbol}")

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["funding_rate"]].sort_index()

    df = df[~df.index.duplicated(keep="first")]
    logger.info("  -> %d unique records after dedup", len(df))

    return df


def fetch_hyperliquid_funding(
    coin: str,
    start_date: str,
    base_url: str,
) -> pd.DataFrame:
    """
    Fetch DEX funding rates from Hyperliquid (1h native).

    Note: Hyperliquid funding is HOURLY, not 8-hourly like Binance.
    Normalization to 8h equivalent happens in a subsequent commit.

    Args:
        coin: Asset name ("BTC", "ETH") — from config.
        start_date: Start date — from config.
        base_url: Hyperliquid URL — from config.

    Returns:
        DataFrame with UTC DatetimeIndex and column: funding_rate_dex.
    """
    endpoint = f"{base_url}/info"
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    all_records = []

    logger.info("Fetching %s funding from Hyperliquid...", coin)

    while True:
        body = {"type": "fundingHistory", "coin": coin, "startTime": start_ts}

        resp = request_with_retry("POST", endpoint, json=body)
        data = resp.json()

        if not data:
            break

        all_records.extend(data)

        last_ts = data[-1].get("time", data[-1].get("timestamp", 0))
        if isinstance(last_ts, str):
            last_ts = int(pd.Timestamp(last_ts).timestamp() * 1000)
        start_ts = last_ts + 1
        time.sleep(0.1)

        if len(data) < 500:
            break

    logger.info("  -> %d Hyperliquid records for %s", len(all_records), coin)

    if not all_records:
        logger.warning("No Hyperliquid data for %s", coin)
        return pd.DataFrame(columns=["funding_rate_dex"])

    df = pd.DataFrame(all_records)

    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.set_index("timestamp")
    df["funding_rate_dex"] = pd.to_numeric(
        df.get("fundingRate", df.get("rate", 0)), errors="coerce"
    )
    df = df[["funding_rate_dex"]].sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df

def normalize_funding_rate(
    df: pd.DataFrame,
    col: str,
    source: str,
) -> pd.DataFrame:
    """
    Normalize funding rate to Binance convention (decimal per period).

    Handles two issues:
    1. Scale: some exchanges return percentages instead of decimals
    2. Maturity: Hyperliquid is 1h, Binance is 8h (fixed in next commit)

    Args:
        df: DataFrame with funding rate column.
        col: Column name.
        source: "binance" or "hyperliquid".

    Returns:
        DataFrame with normalized funding rate.
    """
    if col not in df.columns or df[col].isna().all():
        return df

    median_abs = df[col].abs().median()

    if source == "hyperliquid":
        if median_abs > 0.01:
            df[col] = df[col] / 100
            logger.info("  Normalized %s: divided by 100 (percentage → decimal)", source)
        elif median_abs > 1:
            df[col] = df[col] / (365 * 3)
            logger.info("  Normalized %s: de-annualized", source)
        else:
            logger.info("  %s funding already in decimal format", source)
        # CRITICAL: Hyperliquid funding is HOURLY. Binance is 8-HOURLY.
        # Without this scaling, CEX vs DEX comparison shows a phantom 8x gap.
        # 0.00125% per hour × 8 = 0.01% per 8h = same as Binance.
        df[col] = df[col] * 8
        logger.info("  Scaled %s: 1h rate × 8 → 8h equivalent", source)

    return df




