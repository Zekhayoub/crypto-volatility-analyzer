"""
Centralized configuration loader.

All modules import CONFIG and PROJECT_ROOT from here.
No function in the codebase should have hardcoded default values
for dates, URLs, or parameters — everything comes from config.yaml.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    """Load config.yaml and return as dict."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", CONFIG_PATH)
    return config


def periods(days: int) -> int:
    """
    Convert calendar days to 8-hour periods.

    This project uses 8h candles throughout because the funding rate —
    the most reactive signal in crypto — is natively 8-hourly.
    Aggregating to daily destroys information.

    Args:
        days: Number of calendar days.

    Returns:
        Number of 8-hour periods (days * 3).
    """
    return days * 3


CONFIG = _load_config()