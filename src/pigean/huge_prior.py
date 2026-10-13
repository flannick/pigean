"""Equivalent per-signal normalization of gene prior odds."""
import numpy as np
from scipy import sparse
from numba import njit


@njit(cache=True,fastmath=False)
def _normalize(indptr,indices,log_odds):
    result=np.empty(len(indices),dtype=np.float64)
    for row in range(len(indptr)-1):
        start,end=indptr[row],indptr[row+1]
        if start==end:
            continue
        largest=-np.inf
        for i in range(start,end):
            largest=max(largest,log_odds[indices[i]])
        total=0.0
        for i in range(start,end):
            result[i]=np.exp(log_odds[indices[i]]-largest)
            total+=result[i]
        for i in range(start,end):
            result[i]/=total
    return result


def signal_prior_weights(pattern,prior):
    # Given exactly one associated gene, factors for all other genes' absence
    # are common to every candidate and cancel. Preserve the caller's clipped
    # probabilities instead of substituting its pre-clipping prior odds.
    log_odds=np.log(prior)-np.log(1-prior)
    values=_normalize(pattern.indptr,pattern.indices,log_odds)
    return sparse.csr_matrix((values,pattern.indices,pattern.indptr),shape=pattern.shape).T
