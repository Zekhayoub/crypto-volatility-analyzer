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


