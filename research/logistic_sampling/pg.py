"""PG(1, eta) draws from the alternating-series rejection algorithm.

Independent implementation of Algorithms 1--3 in the supplement to Polson,
Scott & Windle (2013), https://arxiv.org/abs/1205.0310. No truncated gamma sum
or deterministic replacement for the latent variable is used. Floating-point
tail calculations and distributional checks are documented in the experiments.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _log_phi(x):
    if x > -10.:
        return math.log(.5 * math.erfc(-x / math.sqrt(2.)))
    inv = 1. / (x*x)
    term = 1.
    series = 1.
    for k in range(1, 12):
        term *= -(2*k-1)*inv
        series += term
    return -.5*x*x - math.log(-x) - .5*math.log(2*math.pi) + math.log(series)


@njit(cache=True)
def _logadd(a, b):
    m = max(a, b)
    return m + math.log1p(math.exp(min(a,b)-m))


@njit(cache=True)
def draw_pg1(eta, rng):
    """One PG(1, eta) draw. Reject invalid/extreme unsupported inputs loudly."""
    if not math.isfinite(eta) or abs(eta) > 1000:
        raise ValueError('PG argument must be finite and within [-1000,1000]')
    z = abs(eta)*.5
    t = .64
    rate = math.pi**2/8 + z*z/2
    lp = math.log(math.pi/(2*rate)) - rate*t
    lq = math.log(2.) + _logadd(-z + _log_phi(z*math.sqrt(t)-1/math.sqrt(t)),
                                z + _log_phi(-z*math.sqrt(t)-1/math.sqrt(t)))
    p = math.exp(lp - _logadd(lp,lq))
    for attempt in range(10000):
        if rng.random() < p:
            x = t + rng.exponential()/rate
        elif z < 1/t:
            for trial in range(10000):
                e = rng.exponential()
                while e*e > 2*rng.exponential()/t:
                    e = rng.exponential()
                x = t/(1+t*e)**2
                if rng.random() <= math.exp(-.5*z*z*x):
                    break
            else:
                raise RuntimeError('PG truncated inverse Gaussian exhausted')
        else:
            mu = 1/z
            for trial in range(10000):
                v = rng.normal()**2
                a = .5*mu*v
                # Reciprocal form avoids cancellation in the smaller root.
                x = mu/(1+a+math.sqrt(a*(2+a)))
                if rng.random() > mu/(mu+x):
                    x = mu*mu/x
                if x <= t:
                    break
            else:
                raise RuntimeError('PG inverse Gaussian exhausted')
        s = 1.
        y = rng.random()
        for n in range(1,1000):
            if x > t:
                ratio = (2*n+1)*math.exp(-.5*math.pi**2*n*(n+1)*x)
            else:
                ratio = (2*n+1)*math.exp(-2*n*(n+1)/x)
            if n % 2:
                s -= ratio
                if y <= s:
                    return x/4
            else:
                s += ratio
                if y > s:
                    break
        else:
            raise RuntimeError('PG alternating series exhausted')
    raise RuntimeError('PG proposal budget exhausted')


@njit(cache=True)
def draw_pg(eta, rng):
    out = np.empty(eta.size)
    for i in range(eta.size):
        out[i] = draw_pg1(eta[i],rng)
    return out
