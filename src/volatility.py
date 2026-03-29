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
    dist: str = "t",
    rescale: bool = True,
) -> object:
    """
    Fit a GARCH(p,q) model on return series.

    First version: symmetric GARCH. Will be upgraded to GJR-GARCH
    in a subsequent commit when we discover the leverage asymmetry.

    Args:
        returns: Log return series (1-period).
        p: GARCH lag order.
        q: ARCH lag order.
        dist: Error distribution ("normal", "t", "skewt").
        rescale: If True, multiply returns by 100 before fitting.

    Returns:
        Fitted ARCHModelResult.
    """
    clean = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    if len(clean) < 500:
        raise ValueError(f"Not enough observations for GARCH: {len(clean)} (need 500+)")

    if rescale:
        clean = clean * 100
        logger.info("  Rescaled returns to percentage (× 100) for optimizer stability")

    model = arch_model(clean, vol="Garch", p=p, q=q, dist=dist, rescale=False)

    result = model.fit(disp="off", show_warning=False)

    if result.convergence_flag != 0:
        logger.warning("  GARCH did not converge (flag=%d)", result.convergence_flag)

    logger.info("  GARCH(%d,%d) fitted: AIC=%.2f, BIC=%.2f",
                p, q, result.aic, result.bic)

    return result



