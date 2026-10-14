"""Explicit quadratic-score target and exact small-support reference inference.

This is a MODEL ALTERNATIVE, not the joint posterior of current PIGEAN Gibbs.
For support S: log p(beta,S) = h beta - beta Q beta / 2 + log prior(beta,S).
The prior is independent Bernoulli inclusion with zero-mean Gaussian slabs.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_solve
from scipy.special import logsumexp


@dataclass(frozen=True)
class QuadraticTarget:
    precision: np.ndarray
    score: np.ndarray
    inclusion: np.ndarray
    slab_variance: np.ndarray

    def __post_init__(self):
        for key in ('precision', 'score', 'inclusion', 'slab_variance'):
            value = np.array(getattr(self, key), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError('Target inputs must be finite')
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        p = len(self.score)
        if self.score.shape != (p,) or self.precision.shape != (p, p):
            raise ValueError('Incompatible target dimensions')
        if self.inclusion.shape != (p,) or self.slab_variance.shape != (p,):
            raise ValueError('Priors must have one entry per coefficient')
        if np.any((self.inclusion <= 0) | (self.inclusion >= 1)) or np.any(self.slab_variance <= 0):
            raise ValueError('Require 0 < inclusion < 1 and positive slab variance')
        if not np.allclose(self.precision, self.precision.T, rtol=1e-10, atol=1e-12):
            raise ValueError('Precision must be symmetric; no implicit symmetrization')
        scale = max(1., float(np.max(np.abs(self.precision))))
        if np.linalg.eigvalsh(self.precision).min() < -1e-10 * scale:
            raise ValueError('Precision must be positive semidefinite')

    @classmethod
    def from_marginals(cls, beta_tilde, se, correlation, inclusion, slab_variance):
        """Q = S^-1 V S^-1 and h = beta_tilde / se^2, explicitly changing the model."""
        beta_tilde, se = np.asarray(beta_tilde), np.asarray(se)
        if se.shape != beta_tilde.shape or np.any(se <= 0):
            raise ValueError('Require matching beta/SE vectors with positive SEs')
        return cls(np.asarray(correlation) / se[:, None] / se[None, :],
                   beta_tilde / se**2, inclusion, slab_variance)

    @property
    def size(self):
        return len(self.score)

    def support_moments(self, support):
        support = np.asarray(support, dtype=bool)
        if support.shape != (self.size,):
            raise ValueError('Support has incorrect dimensions')
        idx = np.flatnonzero(support)
        mean = np.zeros(self.size)
        covariance = np.zeros((self.size, self.size))
        log_weight = np.log(self.inclusion[support]).sum() + np.log1p(-self.inclusion[~support]).sum()
        if len(idx):
            tau = np.sqrt(self.slab_variance[idx])
            a = np.eye(len(idx)) + tau[:, None] * self.precision[np.ix_(idx, idx)] * tau[None, :]
            chol = np.linalg.cholesky(a)
            c = self.score[idx] * tau
            solved = cho_solve((chol, True), c)
            mean[idx] = tau * solved
            covariance[np.ix_(idx, idx)] = tau[:, None] * cho_solve((chol, True), np.eye(len(idx))) * tau[None, :]
            log_weight += .5 * c.dot(solved) - np.log(np.diag(chol)).sum()
        return float(log_weight), mean, covariance

    def log_density(self, beta, support):
        beta, support = np.asarray(beta), np.asarray(support, dtype=bool)
        if beta.shape != (self.size,) or support.shape != (self.size,):
            raise ValueError('Invalid coefficient/support dimensions')
        if np.any(beta[~support] != 0):
            return -np.inf
        slab = self.slab_variance[support]
        return float(self.score.dot(beta) - .5 * beta.dot(self.precision @ beta)
                     + np.log(self.inclusion[support]).sum() + np.log1p(-self.inclusion[~support]).sum()
                     - .5 * np.sum(np.log(2*np.pi*slab) + beta[support]**2/slab))


def enumerate_posterior(target, max_variables=16):
    """Exact finite support sum, with analytic Gaussian integration within each support."""
    if target.size > max_variables:
        raise ValueError('Exact enumeration exceeds explicit variable cap')
    supports = (np.arange(2**target.size)[:, None] >> np.arange(target.size)) & 1
    supports = supports.astype(bool)
    log_weights = np.array([target.support_moments(s)[0] for s in supports])
    log_evidence = logsumexp(log_weights)
    weights = np.exp(log_weights - log_evidence)
    mean = np.zeros(target.size)
    second = np.zeros((target.size, target.size))
    for weight, support in zip(weights, supports):
        _, mu, cov = target.support_moments(support)
        mean += weight * mu
        second += weight * (cov + np.outer(mu, mu))
    return dict(supports=supports, weights=weights, log_evidence=float(log_evidence),
                inclusion=weights @ supports, mean=mean, covariance=second-np.outer(mean, mean))
