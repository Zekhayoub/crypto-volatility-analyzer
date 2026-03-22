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



def fetch_binance_open_interest(
    symbol: str,
    start_date: str,
    base_url: str,
) -> pd.DataFrame:
    """
    Fetch historical open interest from Binance Futures.

    Returns OI in COINS (not USD) to avoid the nominal illusion:
    BTC $50k→$60k = USD OI +20% with zero new contracts.
    Coin-margined OI measures true leverage.

    Args:
        symbol, start_date, base_url: all from config.

    Returns:
        DataFrame with open_interest_usd and open_interest_coins.
    """
    endpoint = f"{base_url}/futures/data/openInterestHist"
    all_records = []

    logger.info("Fetching %s open interest...", symbol)

    current = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp.now(tz="UTC")

    while current < end:
        chunk_end = min(current + pd.Timedelta(days=30), end)

        params = {
            "symbol": symbol, "period": "1d",
            "startTime": int(current.timestamp() * 1000),
            "endTime": int(chunk_end.timestamp() * 1000),
            "limit": 500,
        }

        resp = request_with_retry("GET", endpoint, params=params)
        data = resp.json()
        if data:
            all_records.extend(data)

        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.1)

    logger.info("  -> %d OI records for %s", len(all_records), symbol)

    if not all_records:
        logger.warning("No OI data for %s", symbol)
        return pd.DataFrame(columns=["open_interest_usd", "open_interest_coins"])

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    df["open_interest_usd"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    if "sumOpenInterest" in df.columns:
        df["open_interest_coins"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    else:
        df["open_interest_coins"] = np.nan

    df = df[["open_interest_usd", "open_interest_coins"]]
    df = df[~df.index.duplicated(keep="first")]

    return df



def _enforce_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Force UTC on a DataFrame index. Safety net for merge."""
    if not df.empty and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        logger.warning("  Forced UTC on a tz-naive source")
    return df




def merge_all_sources(
    spot_btc: pd.DataFrame,
    spot_eth: pd.DataFrame,
    funding_btc: pd.DataFrame,
    funding_eth: pd.DataFrame,
    oi_btc: pd.DataFrame,
    funding_btc_dex: pd.DataFrame,
    funding_eth_dex: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all sources into a single master DataFrame at 8h UTC.

    Inner join on spot + CEX funding. Left join on OI and DEX.
    UTC enforced on all inputs as safety net.
    OI resampled from daily to 8h. OI coins normalized if missing.
    """
    logger.info("Merging all sources...")

    # UTC safety net
    for src in [spot_btc, spot_eth, funding_btc, funding_eth,
                oi_btc, funding_btc_dex, funding_eth_dex]:
        _enforce_utc(src)

    # Prefix columns
    btc_s = spot_btc.rename(columns={
        "open": "btc_open", "high": "btc_high", "low": "btc_low",
        "close": "btc_close", "volume": "btc_volume",
        "taker_buy_volume": "btc_taker_buy_volume",
    })
    eth_s = spot_eth.rename(columns={
        "open": "eth_open", "high": "eth_high", "low": "eth_low",
        "close": "eth_close", "volume": "eth_volume",
        "taker_buy_volume": "eth_taker_buy_volume",
    })
    btc_f = funding_btc.rename(columns={"funding_rate": "btc_funding_rate"})
    eth_f = funding_eth.rename(columns={"funding_rate": "eth_funding_rate"})
    btc_fd = funding_btc_dex.rename(columns={"funding_rate_dex": "btc_funding_rate_dex"})
    eth_fd = funding_eth_dex.rename(columns={"funding_rate_dex": "eth_funding_rate_dex"})

    # Resample OI daily → 8h
    if not oi_btc.empty:
        oi_8h = oi_btc.resample("8h").ffill().rename(columns={
            "open_interest_usd": "btc_oi_usd",
            "open_interest_coins": "btc_oi_coins",
        })
    else:
        oi_8h = pd.DataFrame()

    # Merge
    df = btc_s.join(eth_s, how="inner")
    df = df.join(btc_f, how="inner")
    df = df.join(eth_f, how="left")
    if not oi_8h.empty:
        df = df.join(oi_8h, how="left")
    if not btc_fd.empty:
        df = df.join(btc_fd, how="left")
    if not eth_fd.empty:
        df = df.join(eth_fd, how="left")

    # OI coins normalization
    if "btc_oi_usd" in df.columns and "btc_oi_coins" in df.columns:
        mask = df["btc_oi_coins"].isna() & df["btc_oi_usd"].notna()
        df.loc[mask, "btc_oi_coins"] = df.loc[mask, "btc_oi_usd"] / df.loc[mask, "btc_close"]

    logger.info("Merged: %d rows x %d cols, %s to %s",
                len(df), len(df.columns), df.index.min(), df.index.max())

    return df


def fill_timestamp_gaps(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """
    Reindex to complete 8h UTC grid. Forward-fill short gaps only.
    Gaps > max_gap periods left as NaN.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="8h", tz="UTC")
    n_missing = len(full_idx) - len(df)
    if n_missing > 0:
        logger.info("  Filling %d missing 8h periods (max_gap=%d)", n_missing, max_gap)
    df = df.reindex(full_idx)
    df.index.name = "timestamp"
    df = df.ffill(limit=max_gap)
    return df





def compute_raw_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Order Flow Imbalance from taker buy/sell volume.

    OFI = (taker_buy - taker_sell) / total_volume ∈ [-1, +1].
    Positive = buyers sweeping (bullish). Negative = toxic flow.
    """
    for asset in ["btc", "eth"]:
        vol = f"{asset}_volume"
        taker = f"{asset}_taker_buy_volume"
        if vol in df.columns and taker in df.columns:
            sell = df[vol] - df[taker]
            df[f"{asset}_ofi"] = np.where(df[vol] != 0, (df[taker] - sell) / df[vol], 0.0)
            logger.info("Computed %s OFI", asset.upper())
    return df


def clean_master(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean master DataFrame. Separate treatment for prices and volumes.

    CRITICAL: Missing volume = ZERO, not forward-fill.
    Forward-filling volume during exchange maintenance creates phantom
    liquidity (e.g., 15k BTC of fake volume). This corrupts OFI,
    volume moving averages, and all volume-based features.

    Rule: price stays flat during gaps, volume drops to zero.
    """
    logger.info("Cleaning master DataFrame (%d rows)...", len(df))

    # 1. Trading mask (True where real trading activity existed)
    trading_mask = (df["btc_volume"] > 0) & (df["btc_close"].notna())

    # 2a. Forward-fill CLOSE prices (price stays flat during gaps)
    close_cols = [c for c in df.columns if "close" in c]
    df[close_cols] = df[close_cols].ffill(limit=1)

    # 2b. Open, High, Low during missing periods = flat at Close
    for asset in ["btc", "eth"]:
        close_col = f"{asset}_close"
        if close_col not in df.columns:
            continue
        for suffix in ["open", "high", "low"]:
            col = f"{asset}_{suffix}"
            if col in df.columns:
                df[col] = df[col].fillna(df[close_col])

    # 2c. CRITICAL: Missing volume = ZERO (not forward-fill!)
    vol_cols = [c for c in df.columns if "volume" in c or "taker" in c]
    df[vol_cols] = df[vol_cols].fillna(0.0)

    # 2d. Forward-fill funding and OI (non-price, non-volume)
    fill_cols = [c for c in df.columns if "funding" in c or "oi" in c]
    df[fill_cols] = df[fill_cols].ffill(limit=3)

    # 3. Drop rows without BTC close
    rows_before = len(df)
    df = df.dropna(subset=["btc_close"])
    logger.info("  Dropped %d rows without BTC price", rows_before - len(df))

    # 4. Recompute OFI after volume fix (OFI was computed before clean)
    df = compute_raw_ofi(df)

    # 5. Align mask
    trading_mask = trading_mask.reindex(df.index).fillna(False)

    # 6. Assertions
    assert df["btc_close"].isna().sum() == 0, "BTC close has NaN"
    assert len(df) > 1000, f"Only {len(df)} rows"

    logger.info("  Cleaned: %d rows, trading periods: %d (%.1f%%)",
                len(df), trading_mask.sum(), trading_mask.mean() * 100)

    return df, trading_mask






def save_raw(df: pd.DataFrame, name: str, config: dict = CONFIG) -> None:
    """Save raw DataFrame to data/raw/."""
    raw_dir = PROJECT_ROOT / config["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}.csv"
    df.to_csv(path)
    logger.info("Saved raw: %s (%d rows)", path, len(df))




def save_master(df: pd.DataFrame, config: dict = CONFIG) -> Path:
    """Save master DataFrame to data/processed/master.csv."""
    proc_dir = PROJECT_ROOT / config["paths"]["processed"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    path = proc_dir / "master.csv"
    df.to_csv(path)
    logger.info("Saved master: %s (%d x %d)", path, len(df), len(df.columns))
    return path




def run_ingestion(config: dict = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point. Fetches all sources, merges, cleans, saves.

    DEX calls are wrapped in try/except — if Hyperliquid is down,
    the pipeline continues with CEX data only (production resilience).
    """
    d = config["data"]
    start = d["start_date"]
    b_url = d["binance_base_url"]
    f_url = d["binance_futures_url"]
    h_url = d["hyperliquid_url"]

    # --- Binance Spot ---
    btc_spot = fetch_binance_klines(d["spot"]["btc"], "8h", start, b_url)
    save_raw(btc_spot, "btc_spot_8h", config)
    eth_spot = fetch_binance_klines(d["spot"]["eth"], "8h", start, b_url)
    save_raw(eth_spot, "eth_spot_8h", config)

    # --- Binance Funding ---
    btc_fund = fetch_binance_funding_rate(d["futures"]["btc"], start, f_url)
    save_raw(btc_fund, "btc_funding_cex", config)
    eth_fund = fetch_binance_funding_rate(d["futures"]["eth"], start, f_url)
    save_raw(eth_fund, "eth_funding_cex", config)

    # --- Binance OI ---
    btc_oi = fetch_binance_open_interest(d["futures"]["btc"], start, f_url)
    save_raw(btc_oi, "btc_open_interest", config)

    # --- Hyperliquid (resilient) ---
    try:
        btc_fd = fetch_hyperliquid_funding(d["hyperliquid"]["btc"], start, h_url)
        btc_fd = normalize_funding_rate(btc_fd, "funding_rate_dex", "hyperliquid")
        save_raw(btc_fd, "btc_funding_dex", config)
    except Exception as e:
        logger.warning("Hyperliquid BTC failed: %s — continuing without DEX", e)
        btc_fd = pd.DataFrame(columns=["funding_rate_dex"])

    try:
        eth_fd = fetch_hyperliquid_funding(d["hyperliquid"]["eth"], start, h_url)
        eth_fd = normalize_funding_rate(eth_fd, "funding_rate_dex", "hyperliquid")
        save_raw(eth_fd, "eth_funding_dex", config)
    except Exception as e:
        logger.warning("Hyperliquid ETH failed: %s — continuing without DEX", e)
        eth_fd = pd.DataFrame(columns=["funding_rate_dex"])

    # --- Merge + OFI + Clean ---
    df = merge_all_sources(btc_spot, eth_spot, btc_fund, eth_fund, btc_oi, btc_fd, eth_fd)
    df = compute_raw_ofi(df)
    df = fill_timestamp_gaps(df)
    df, trading_mask = clean_master(df)

    # --- Save ---
    save_master(df, config)
    mask_path = PROJECT_ROOT / config["paths"]["processed"] / "trading_mask.csv"
    trading_mask.to_csv(mask_path)
    logger.info("Saved trading mask: %s", mask_path)

    return df, trading_mask


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    df, mask = run_ingestion()
    print(f"\nMaster: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Range: {df.index.min()} to {df.index.max()}")
    print(f"Trading periods: {mask.sum()} ({mask.mean()*100:.1f}%)")
    print(f"NaN: {df.isna().sum().sum()}")


































