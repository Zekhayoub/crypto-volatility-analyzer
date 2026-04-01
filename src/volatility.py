"""
Volatility modeling: GARCH and comparison with realized estimators.

Uses the arch library for GARCH fitting. Returns are rescaled to
percentage (* 100) before fitting — the arch optimizer fails on
very small floats (e.g., 0.0001 instead of 0.01).
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from src.config import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


def fit_garch(
    returns: pd.Series,
    p: int,
    q: int,
    o: int = 0,
    dist: str = "t",
    rescale: bool = True,
) -> object:
    """
    Fit a GJR-GARCH(p,o,q) model on return series.

    The GJR variant adds an asymmetry term (gamma) that penalizes
    negative shocks more than positive ones. In crypto, -10% triggers
    liquidation cascades that amplify volatility far more than +10%.

    σ²_t = ω + α*ε²_{t-1} + γ*ε²_{t-1}*I(ε<0) + β*σ²_{t-1}

    If gamma > 0 (expected in crypto), negative shocks increase vol more.

    Args:
        returns: Log return series.
        p: GARCH lag. q: ARCH lag. o: Asymmetry lag (1 for GJR).
        dist: Error distribution.
        rescale: Multiply returns by 100.

    Returns:
        Fitted ARCHModelResult.
    """
    clean = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    if len(clean) < 500:
        raise ValueError(f"Not enough data: {len(clean)}")

    if rescale:
        clean = clean * 100

    vol_type = "GJR-Garch" if o > 0 else "Garch"
    model = arch_model(clean, vol=vol_type, p=p, o=o, q=q, dist=dist, rescale=False)

    result = model.fit(disp="off", show_warning=False)

    if result.convergence_flag != 0:
        logger.warning("  %s did not converge (flag=%d)", vol_type, result.convergence_flag)

    logger.info("  %s(%d,%d,%d) fitted: AIC=%.2f", vol_type, p, o, q, result.aic)

    return result


def compute_ewma_volatility(
    returns: pd.Series,
    span: int = 42,
    ann_factor: int = 1095,
) -> pd.Series:
    """
    EWMA volatility as fallback when GARCH fails to converge.

    Args:
        returns: 1-period returns.
        span: EWMA span.
        ann_factor: Annualization factor.

    Returns:
        Annualized EWMA volatility series.
    """
    ewma_var = returns.ewm(span=span).var()
    vol = np.sqrt(ewma_var) * np.sqrt(ann_factor)
    return vol


def fit_garch_safe(
    returns: pd.Series,
    config: dict = CONFIG,
    asset: str = "btc",
) -> tuple[object | None, pd.Series]:
    """
    Fit GJR-GARCH with fallback to EWMA on convergence failure.

    Args:
        returns: 1-period returns.
        config: Configuration.
        asset: Asset name for logging.

    Returns:
        Tuple of (ARCHModelResult or None, conditional volatility series).
    """
    vol_cfg = config["volatility"]

    try:
        result = fit_garch(
            returns, p=vol_cfg["garch_p"], q=vol_cfg["garch_q"],
            o=vol_cfg["garch_o"], dist=vol_cfg["distribution"],
            rescale=vol_cfg["rescale"],
        )
        # Extract conditional volatility
        cond_vol = result.conditional_volatility
        if vol_cfg["rescale"]:
            cond_vol = cond_vol / 100  # back to decimal
        cond_vol = cond_vol * np.sqrt(1095)  # annualize

        logger.info("  %s: GJR-GARCH converged", asset.upper())
        return result, cond_vol

    except Exception as e:
        logger.warning("  %s: GJR-GARCH failed (%s) — falling back to EWMA", asset.upper(), e)
        ewma_vol = compute_ewma_volatility(returns)
        return None, ewma_vol
    

    