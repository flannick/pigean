"""Single-pass grouped moment sums; preserve per-output accumulation order."""
import numpy as np
try:
    from numba import njit
except ImportError as error:
    raise ImportError('compiled-moments requires the optional numba dependency') from error


@njit(cache=True, fastmath=False, error_model='numpy')
def grouped_moment_sums(local, values, counts, p, v, width):
    result = np.zeros((5, width), dtype=np.float64)
    for i in range(len(local)):
        j = local[i]
        x, count, probability, variance = values[i], counts[i], p[i], v[i]
        result[0, j] += count * probability
        result[1, j] += count * variance
        result[2, j] += count * (x * probability)
        result[3, j] += count * (x * variance)
        result[4, j] += count * ((x * x) * variance)
    return result
