"""Pólya--Gamma single-coordinate control for a fixed logistic conditional."""
import numpy as np
from numba import njit
from .pg import draw_pg
from .collapse import collapsed_sweep, draw_coefficients


@njit(cache=True)
def _coordinate(Q,h,pi,tau,beta,rng):
    for j in range(len(beta)):
        v=1/(Q[j,j]+1/tau[j])
        r=h[j]-Q[j]@beta+Q[j,j]*beta[j]
        odds=np.log(pi[j]/(1-pi[j]))+.5*(np.log(v/tau[j])+r*r*v)
        prob=1/(1+np.exp(-min(700.,max(-700.,odds))))
        beta[j]=r*v+np.sqrt(v)*rng.normal() if rng.random()<prob else 0.
    return beta


def sample(target,*,sweeps=4000,burn=1000,chains=4,seed=117,initial=None,collapse=False):
    if not 0<=burn<sweeps or chains<2: raise ValueError('Invalid sampling budget')
    n,p=target.X.shape
    if p>64: raise ValueError('Research implementation capped at 64 annotations')
    rng=np.random.default_rng(seed)
    beta=np.zeros((chains,p)) if initial is None else np.array(initial,dtype=float,copy=True)
    if beta.shape!=(chains,p): raise ValueError('Initial state shape mismatch')
    draws=np.empty((sweeps-burn,chains,p))
    for it in range(sweeps):
        for c in range(chains):
            omega=draw_pg(target.offset+target.X@beta[c],rng)
            Q=target.X.T@(omega[:,None]*target.X)
            h=target.X.T@(target.y-.5-omega*target.offset)
            if collapse:
                support=collapsed_sweep(Q,h,target.inclusion,target.variance,beta[c]!=0,rng)
                beta[c]=draw_coefficients(Q,h,target.variance,support,rng)
            else:
                beta[c]=_coordinate(Q,h,target.inclusion,target.variance,beta[c],rng)
        if it>=burn: draws[it-burn]=beta
    return draws
