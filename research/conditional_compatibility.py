"""Necessary joint-density check for a frozen production all-slab conditional.

This diagnostic does not define an alternative model or change any sampler.
Passing symmetry and positive definiteness is not a full spike/slab audit.
"""
import numpy as np


def audit_slab_precision(V, shrink, variance, *, rtol=1e-8, atol=1e-10):
    V, shrink, variance = map(np.asarray, (V, shrink, variance))
    n = len(shrink)
    if V.shape != (n, n) or variance.shape != (n,) or shrink.shape != (n,):
        raise ValueError('Inconsistent dimensions')
    if not all(np.all(np.isfinite(a)) for a in (V, shrink, variance)):
        raise ValueError('Nonfinite conditional inputs')
    if np.any(variance <= 0) or np.any(shrink <= 0):
        raise ValueError('Require positive slab variances and shrinkage')
    # Production mean_j = shrink_j * (btilde_j - sum_{k != j} V[k,j]*b_k).
    # Hence -d(mean_j)/d(b_k) / variance_j must be a symmetric precision.
    q = V.T * (shrink / variance)[:, None]
    np.fill_diagonal(q, 1 / variance)
    delta = np.abs(q-q.T)
    bad = ~np.isclose(q, q.T, rtol=rtol, atol=atol)
    i, j = np.unravel_index(np.argmax(delta), delta.shape)
    symmetric = not bool(bad.any())
    eigmin = float(np.linalg.eigvalsh((q+q.T)/2)[0]) if symmetric else None
    return dict(n=n, symmetric=symmetric, positive_definite=None if not symmetric else eigmin > 0,
                min_eigenvalue=eigmin, max_relative_asymmetry=float(delta.max()/np.abs(q).max()),
                asymmetric_directed_pair_fraction=float(bad.sum()/max(n*(n-1),1)),
                witness=dict(i=int(i), j=int(j), q_ij=float(q[i,j]), q_ji=float(q[j,i])),
                interpretation='Necessary all-slab check only; a failure forbids a joint density with these fixed conditionals.')
