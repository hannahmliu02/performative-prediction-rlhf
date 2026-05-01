"""DAG validity checks for the multi-round RLHF feedback loop.

Runtime assertions that the temporal-unrolling assumptions of the causal graph
are satisfied before each training round:

1. Prior-round and current-round RM checkpoints are distinct paths (no
   in-place parameter mutation across rounds).
2. Propensity model residuals are uncorrelated with cell membership, indicating
   the propensity model has absorbed the observation mechanism.

These checks run inside the loop simulator, not as standalone experiments.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from llm.utils.logging import get_logger

log = get_logger(__name__)


def assert_distinct_checkpoints(
    prev_checkpoint: str | pathlib.Path,
    curr_checkpoint: str | pathlib.Path,
) -> None:
    """Assert that consecutive RM checkpoints are different filesystem paths.

    A violation means the same checkpoint was used in two rounds, which breaks
    the temporal ordering assumption of the causal graph (the RM cannot be
    both prior-round and current-round simultaneously).

    Raises:
        AssertionError: if prev and curr resolve to the same path.
    """
    prev = pathlib.Path(prev_checkpoint).resolve()
    curr = pathlib.Path(curr_checkpoint).resolve()
    if prev == curr:
        raise AssertionError(
            f"DAG violation: prior-round and current-round RM are the same checkpoint "
            f"({curr}). This means in-place parameter mutation occurred, breaking the "
            "temporal ordering assumption. Each round must save to a distinct directory."
        )
    log.debug("DAG check: checkpoints are distinct", prev=str(prev), curr=str(curr))


def assert_propensity_residuals_uncorrelated(
    propensity_model: Any,
    df: pd.DataFrame,
    threshold: float = 0.05,
) -> bool:
    """Check that propensity model residuals are uncorrelated with demographic group.

    If the propensity model fully explains the observation mechanism, its
    residuals (observed − predicted) should be uncorrelated with demographic
    group membership. A significant correlation indicates residual confounding
    that the propensity model has not captured — the causal interpretation
    may then be invalid.

    Args:
        propensity_model: fitted model with ``predict_proba(df) -> np.ndarray``.
        df: DataFrame with ``demographic_signal`` column (A/B) and optionally
            ``is_observed`` binary column. If absent, all rows are treated as
            observed (residual = 1 − predicted).
        threshold: p-value threshold for flagging correlation as significant.

    Returns:
        True if the check passes (residuals uncorrelated), False otherwise.
        Does not raise — logs a warning on failure so the loop can continue.
    """
    predicted = propensity_model.predict_proba(df)

    if "is_observed" in df.columns:
        observed = df["is_observed"].to_numpy(dtype=float)
    else:
        observed = np.ones(len(df), dtype=float)

    residuals = observed - predicted
    group_indicator = (df["demographic_signal"] == "A").to_numpy(dtype=float)

    r, p_value = stats.pearsonr(residuals, group_indicator)

    if p_value < threshold:
        log.warning(
            "DAG check FAILED: propensity residuals correlated with demographic group; "
            "residual confounding present. Causal debiasing interpretation is weakened.",
            pearson_r=f"{r:.3f}",
            p_value=f"{p_value:.4f}",
            threshold=threshold,
        )
        return False

    log.debug(
        "DAG check passed: propensity residuals uncorrelated with group",
        pearson_r=f"{r:.3f}",
        p_value=f"{p_value:.4f}",
    )
    return True
