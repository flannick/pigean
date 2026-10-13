"""Optional direct sparse quadratic forms for the linear prefilter."""
import numpy as np
from scipy import sparse
from numba import njit


@njit(cache=True, fastmath=False, nogil=True)
def linear_terms(xptr, xrows, xdata, rptr, rcols, rdata, r1, total_r, n, pseudo):
    k = len(xptr)-1
    real_n = len(r1)
    lookup = np.zeros(real_n, dtype=np.float64)
    independent = np.empty(k)
    correlated = np.empty(k)
    ind_scale = np.empty(k)
    cor_scale = np.empty(k)
    for j in range(k):
        start, end = xptr[j], xptr[j+1]
        sx = sx2 = dot_row_sums = pseudo_x = 0.0
        for pos in range(start,end):
            row, value = xrows[pos], xdata[pos]
            if row < real_n:
                lookup[row] = value
                sx += value
                sx2 += value*value
                dot_row_sums += value*r1[row]
            else:
                pseudo_x = value
        mean = (sx+pseudo_x)/n
        constant = -mean
        quadratic = 0.0
        for pos in range(start,end):
            row = xrows[pos]
            if row >= real_n:
                continue
            rd = 0.0
            for edge in range(rptr[row],rptr[row+1]):
                rd += rdata[edge]*lookup[rcols[edge]]
            quadratic += xdata[pos]*rd
        for pos in range(start,end):
            if xrows[pos] < real_n:
                lookup[xrows[pos]] = 0.0
        pseudo_sq = (pseudo_x+constant)**2 if pseudo else 0.0
        cross = 2*constant*dot_row_sums
        background = constant**2*total_r
        independent[j] = sx2+2*constant*sx+real_n*constant**2+pseudo_sq
        correlated[j] = quadratic+cross+background+pseudo_sq
        ind_scale[j] = sx2+abs(2*constant*sx)+real_n*constant**2+pseudo_sq
        cor_scale[j] = abs(quadratic)+abs(cross)+abs(background)+pseudo_sq
    return independent, correlated, ind_scale, cor_scale


def linear_inflation(X, r, beta, alpha, pseudo, fallback, block_size):
    a = sparse.csc_matrix(X, dtype=float, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    r = r.tocsr()
    r1 = np.asarray(r.sum(axis=1)).ravel()
    independent, correlated, ind_scale, scale = linear_terms(
        a.indptr,a.indices,a.data,r.indptr,r.indices,r.data,r1,r1.sum(),a.shape[0],bool(pseudo))
    unsafe = ((independent <= 1e-7*ind_scale) | (correlated <= 1e-7*scale)
              | ~np.isfinite(independent) | ~np.isfinite(correlated))
    ratio = np.divide(correlated,independent,out=np.ones(a.shape[1]),where=independent>0)
    ratio[unsafe] = 1
    result = np.sqrt(ratio)
    if np.any(unsafe):
        # Delegate potentially cancellation-sensitive cases to the exact prior
        # sparse implementation, including its own dense fallback decision.
        from .sparse_regression import _sparse_inflation_serial
        result[unsafe] = _sparse_inflation_serial(a[:,unsafe],r,None,None,pseudo,fallback,block_size)
    return result
