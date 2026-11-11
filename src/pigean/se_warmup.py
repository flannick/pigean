"""Scoped correlation-SE initialization; never used for retained inference."""
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral

_requested = ContextVar("pigean_se_warmup_iterations", default=0)
_bypass = ContextVar("pigean_bypass_correlation_se", default=False)


def validate_warmup(iterations, min_burn_in, gauss_seidel=False):
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 0:
        raise ValueError("--gibbs-se-warmup-iterations must be a nonnegative integer")
    if iterations and gauss_seidel:
        raise ValueError("SE warmup is not supported with outer Gauss-Seidel updates")
    # Burn-in ends before accumulating the iteration at min_burn_in.
    if iterations and min_burn_in < 2 * iterations + 1:
        raise ValueError("SE warmup requires --min-num-burn-in >= 2 * warmup iterations + 1")


@contextmanager
def warmup_schedule(iterations):
    token = _requested.set(iterations)
    try:
        yield
    finally:
        _requested.reset(token)


def epoch_warmup_length(min_burn_in):
    # A shortened final epoch may have a smaller burn-in allocation. Preserve
    # at least as many fully corrected discarded updates as bypassed updates.
    return min(_requested.get(), max(0, (min_burn_in - 1) // 2))


@contextmanager
def correlation_se_bypass(enabled):
    token = _bypass.set(bool(enabled))
    try:
        yield
    finally:
        _bypass.reset(token)


def bypass_correlation_se():
    return _bypass.get()
