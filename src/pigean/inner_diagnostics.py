"""Numerical helpers for inner Gibbs stopping."""
import numpy as np


class CenteredChainMoments:
    """Online centered moments of inner posterior means, without retaining draws.

    Subtract each chain's first observation before Welford accumulation. This
    preserves tiny fluctuations around a nonzero coefficient without forming
    sum(x**2) - sum(x)**2/n. Counts follow active parallel slots; outlier chain
    replacement must copy the accumulated history along with sampled state.
    """

    def __init__(self, shape):
        self.origin = np.zeros(shape)
        self.mean_delta = np.zeros(shape)
        self.m2 = np.zeros(shape)
        self.count = np.zeros(shape[1], dtype=np.int64)

    def update(self, values, active):
        first = active & (self.count == 0)
        self.origin[:, first, :] = values[:, first, :]
        self.count[active] += 1
        shifted = values[:, active, :] - self.origin[:, active, :]
        delta = shifted - self.mean_delta[:, active, :]
        updated = self.mean_delta[:, active, :] + delta / self.count[active][None, :, None]
        self.m2[:, active, :] += delta * (shifted - updated)
        self.mean_delta[:, active, :] = updated

    def replace_chains(self, mask, parallel, replacements):
        for tensor in (self.origin, self.mean_delta, self.m2):
            tensor[mask, parallel, :] = tensor[replacements, parallel, :].copy()

    def statistics(self):
        num_chains, num_parallel, num_genes = self.origin.shape
        n = np.maximum(self.count, 1).astype(float)[:, None]
        mean = self.origin + self.mean_delta
        # Center across chains before adding back a potentially large origin.
        relative = (self.origin - self.origin[0:1]) + self.mean_delta
        between = relative - relative.mean(axis=0)
        B = n / max(num_chains - 1, 1) * np.square(between).sum(axis=0)
        variance = self.m2 / np.maximum(n - 1, 1)[None, :, :]
        W = variance.mean(axis=0)
        total = (n - 1) / n * W + B / n
        R = np.ones((num_parallel, num_genes))
        positive = W > 0
        R[positive] = np.sqrt(np.maximum(total[positive], 0) / W[positive])
        avg_W = variance.mean(axis=2)
        # Preserve the existing short-history and zero-within-variance rules.
        short = self.count <= 1
        B[short] = W[short] = R[short] = 1
        avg_W[:, short] = 1
        return B, W, R, avg_W, mean


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
