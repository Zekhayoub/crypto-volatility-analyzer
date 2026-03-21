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
                wait = 2 ** attempt
                logger.warning("  Rate limited (429). Waiting %ds...", wait)
                time.sleep(wait)
            elif attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning("  Attempt %d/%d failed (HTTP %s). Retrying in %ds...",
                               attempt, max_retries, status, wait)
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning("  Attempt %d/%d failed: %s. Retrying in %ds...",
                               attempt, max_retries, e, wait)
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"All {max_retries} attempts failed for {url}")



KLINE_COLUMNS = {
    0: "open_time_ms", 1: "open", 2: "high", 3: "low", 4: "close",
    5: "volume", 6: "close_time_ms", 7: "quote_volume", 8: "n_trades",
    9: "taker_buy_volume", 10: "taker_buy_quote_volume", 11: "ignore",
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
            "symbol": symbol, "interval": interval,
            "startTime": start_ts, "limit": 1000,
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

    return df






