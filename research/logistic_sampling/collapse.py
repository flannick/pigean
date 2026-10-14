"""Integrate Gaussian coefficients conditional on PG variables, not D."""
import numpy as np
from numba import njit


@njit(cache=True)
def support_logmass(Q,h,pi,tau,support):
    ids=np.flatnonzero(support)
    prior=np.sum(np.where(support,np.log(pi),np.log1p(-pi)))
    if len(ids)==0: return prior
    A=Q[ids][:,ids]+np.diag(1/tau[ids])
    L=np.linalg.cholesky(A)
    v=np.linalg.solve(L,h[ids])
    return prior+.5*(v@v-np.log(tau[ids]).sum())-np.log(np.diag(L)).sum()


@njit(cache=True)
def collapsed_sweep(Q,h,pi,tau,support,rng):
    score=support_logmass(Q,h,pi,tau,support)
    for j in range(len(support)):
        proposal=support.copy();proposal[j]=not proposal[j]
        proposed=support_logmass(Q,h,pi,tau,proposal)
        prob=1/(1+np.exp(-min(700.,max(-700.,proposed-score))))
        if rng.random()<prob:
            support=proposal;score=proposed
    return support


@njit(cache=True)
def draw_coefficients(Q,h,tau,support,rng):
    ids=np.flatnonzero(support);beta=np.zeros(len(support))
    if len(ids):
        L=np.linalg.cholesky(Q[ids][:,ids]+np.diag(1/tau[ids]))
        mu=np.linalg.solve(L.T,np.linalg.solve(L,h[ids]))
        beta[ids]=mu+np.linalg.solve(L.T,rng.standard_normal(len(ids)))
    return beta
