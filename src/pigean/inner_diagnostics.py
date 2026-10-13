"""Numerical helpers for inner Gibbs stopping; raw R-hat remains unchanged."""
import numpy as np


def stabilize_rhat_mean_boundary(rhat):
    """Copy R-hat, snapping roundoff near one for thresholded-mean membership.

    The float64 stopping statistic includes values >= 1. A one-step rounding
    difference at that boundary can change its denominator and trigger an extra
    Gibbs sweep. Sixteen spacings at one cover the observed cancellation noise;
    this is a narrow numerical tolerance, not a convergence tolerance or a
    guarantee that different arithmetic backends follow identical trajectories.
    Never use this copy for raw diagnostics or outlier detection.
    """
    result = np.array(rhat, copy=True)
    near_one = np.isfinite(result) & (np.abs(result - 1.0) <= 16 * np.spacing(1.0))
    result[near_one] = 1.0
    return result
