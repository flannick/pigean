# Frozen requirements-v4 selection, qualified on six archived trait inputs.
"""Research-only profile diagnostics; no inferred biological probabilities."""
import numpy as np
from scipy.optimize import nnls, linear_sum_assignment

def norm(w):
    return np.divide(w, np.linalg.norm(w, axis=0), out=np.zeros_like(w), where=np.linalg.norm(w, axis=0) > 0)

def pairgram(a, b):
    return 0.5 * ((a.T @ b) ** 2 - (a * a).T @ (b * b))

def pair_residual(b, a):
    aa = pairgram(a, a).diagonal()
    if b.shape[1] == 0:
        return np.where(aa > 1e-30, 1.0, 0.0)
    g = pairgram(b, b)
    q = pairgram(b, a)
    (e, v) = np.linalg.eigh((g + g.T) / 2)
    keep = e > max(e.max(initial=0) * 1e-12, 1e-24)
    if not keep.any():
        return np.where(aa > 1e-30, 1.0, 0.0)
    c = np.sqrt(e[keep])[:, None] * v[:, keep].T
    d = v[:, keep].T @ q / np.sqrt(e[keep])[:, None]
    r = []
    for j in range(a.shape[1]):
        (beta, _) = nnls(c, d[:, j], maxiter=2000)
        r.append(np.sqrt(max(0.0, aa[j] - 2 * q[:, j] @ beta + beta @ g @ beta) / max(aa[j], 1e-30)))
    return np.array(r)

def support(raw, w, cosine):
    counts = np.zeros(w.shape[1], int)
    for a in raw:
        if not a.shape[1] or not w.shape[1]:
            continue
        sim = norm(a).T @ norm(w)
        (i, j) = linear_sum_assignment(-sim)
        counts[j] += sim[i, j] >= cosine
    return counts
