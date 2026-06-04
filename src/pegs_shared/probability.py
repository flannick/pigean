from __future__ import annotations

import numpy as np


DEFAULT_MAX_PROBABILITY = 0.99
DEFAULT_MIN_PROBABILITY = 0.0


def validate_max_probability(max_probability, *, option_name="--max-probability", bail_fn=None):
    if max_probability is None:
        return DEFAULT_MAX_PROBABILITY
    max_probability = float(max_probability)
    if not np.isfinite(max_probability) or max_probability <= 0.0 or max_probability >= 1.0:
        message = "%s must be finite and strictly between 0 and 1" % option_name
        if bail_fn is not None:
            bail_fn(message)
        raise ValueError(message)
    return max_probability


def cap_probability(probability, *, max_probability=DEFAULT_MAX_PROBABILITY):
    max_probability = validate_max_probability(max_probability)
    return min(float(probability), max_probability)
