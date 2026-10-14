"""Independent symmetric support-exchange kernel for the augmented target."""
import numpy as np
from numba import njit
from .collapse import support_logmass


@njit(cache=True)
def support_swaps(Q,h,pi,tau,support,rng,attempts,probability=.5):
    p=len(support)
    if p<2: return support
    score=support_logmass(Q,h,pi,tau,support)
    # Uniform pairs selected independently of state: no proposal-count ratio.
    for k in range(attempts):
        if rng.random()>=probability: continue
        a=rng.integers(0,p);b=rng.integers(0,p-1)
        if b>=a: b+=1
        if support[a]==support[b]: continue
        proposal=support.copy();proposal[a]=not proposal[a];proposal[b]=not proposal[b]
        proposed=support_logmass(Q,h,pi,tau,proposal)
        if np.log(rng.random())<min(0.,proposed-score):
            support=proposal;score=proposed
    return support
