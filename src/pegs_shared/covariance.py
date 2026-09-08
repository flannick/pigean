"""Correlation quadratic forms for marginal regressions, including the intercept."""
import numpy as np
from scipy import sparse
from scipy.special import expit
from scipy.sparse.linalg import LinearOperator


def validate_correlation(matrix, n):
    if matrix.shape != (n, n):
        raise ValueError('Residual correlation shape must match the gene observations')
    if isinstance(matrix, AnnotationCorrelation):
        return matrix  # PSD by its factorized construction.
    if isinstance(matrix, LinearOperator):
        raise ValueError('Unsupported correlation operator')
    r = sparse.csr_matrix(matrix, dtype=float)
    if not np.all(np.isfinite(r.data)) or not np.allclose(r.diagonal(), 1, atol=1e-8, rtol=0):
        raise ValueError('Residual correlation must be finite with unit diagonal')
    delta = r - r.T
    if (delta.nnz and np.max(np.abs(delta.data)) > 1e-8) or (r.nnz and np.max(np.abs(r.data)) > 1 + 1e-8):
        raise ValueError('Residual correlation must be symmetric and bounded by one')
    return r


def correlation_ratio(r, influence):
    independent = np.sum(influence * influence, axis=0)
    dependent = np.sum(influence * r.dot(influence), axis=0)
    if np.any(dependent < -1e-9 * np.maximum(independent, 1)):
        raise ValueError('Residual correlation gives a negative variance')
    result = np.ones_like(independent)
    keep = independent > 0
    result[keep] = np.maximum(dependent[keep], 0) / independent[keep]
    if np.any(result[keep] <= 0):
        raise ValueError('Residual correlation gives zero slope variance')
    return np.sqrt(result)


def marginal_inflation(X, correlation, *, beta=None, alpha=None, pseudo=False, block_size=64):
    """SE multiplier; beta/alpha select a logistic rather than linear model.

    Logistic slope influence is sqrt(v_i)*(x_i-weighted_mean(x)),
    where v_i=p_i*(1-p_i). Centering eliminates the fitted intercept.
    The independent pseudocount convention matches the existing penalized fit.
    """
    n, k = X.shape
    r = validate_correlation(correlation, n - int(pseudo))
    output = np.ones(k)
    for begin in range(0, k, block_size):
        end = min(k, begin + block_size)
        a = X[:, begin:end].toarray() if sparse.issparse(X) else np.asarray(X[:, begin:end], dtype=float)
        if beta is None:
            influence = a - a.mean(axis=0)
        else:
            p = expit(np.clip(a * beta[begin:end] + alpha[begin:end], -100, 100))
            v = p * (1 - p)
            total = v.sum(axis=0)
            mean = np.divide((v * a).sum(axis=0), total, out=np.zeros(end-begin), where=total > 0)
            influence = np.sqrt(v) * (a - mean)
        if pseudo:
            independent = np.sum(influence**2, axis=0)
            correlated = np.sum(influence[:-1] * r.dot(influence[:-1]), axis=0) + influence[-1]**2
            if np.any(correlated < -1e-9 * np.maximum(independent, 1)):
                raise ValueError('Residual correlation gives a negative variance')
            ratio = np.divide(correlated, independent, out=np.ones_like(independent), where=independent > 0)
            if np.any(ratio <= 0):
                raise ValueError('Residual correlation gives zero slope variance')
            output[begin:end] = np.sqrt(ratio)
        else:
            output[begin:end] = correlation_ratio(r, influence)
    return output


class AnnotationCorrelation(LinearOperator):
    """PSD correlation of centered annotation contributions plus independent Direct variance.

    Stores X and one coefficient vector, never a dense gene-by-gene matrix.
    Pairwise sparsification is deliberately avoided: it need not preserve PSD.
    """
    def __init__(self, X, beta, direct=None):
        self.X = sparse.csr_matrix(X, dtype=float)
        self.beta = np.asarray(beta, dtype=float).reshape(-1)
        n, m = self.X.shape
        if m == 0 or self.beta.shape != (m,):
            raise ValueError('Annotation coefficient shape mismatch')
        d = np.zeros(n) if direct is None else np.asarray(direct, dtype=float).reshape(-1)
        if d.shape != (n,) or not all(np.all(np.isfinite(a)) for a in [self.X.data, self.beta, d]):
            raise ValueError('Invalid annotation or Direct values')
        self.m = m
        prior = self.X.dot(self.beta)
        # Preserve the existing inclusion gates on annotation-derived support.
        self.keep = (prior > .1) & (prior**2 / (d**2 + prior**2 + 1e-20) > .05)
        self.mean = prior * self.keep / m
        second = self.X.multiply(self.X).dot(self.beta**2) * self.keep / m
        var = np.maximum(second - self.mean**2, 0) + d**2
        self.residual = d**2 + (var == 0)
        var = var + (var == 0)
        self.scale = 1 / np.sqrt(var)
        self.storage_nnz = self.X.nnz + 4*n + m
        super().__init__(dtype=np.dtype(float), shape=(n, n))

    def _matmat(self, z):
        z = np.asarray(z, dtype=float)
        q = self.scale[:, None] * z
        t = self.X.T.dot(self.keep[:, None] * q)
        value = self.keep[:, None] * self.X.dot(self.beta[:, None]**2 * t) / self.m
        value -= self.mean[:, None] * (self.mean @ q)[None, :]
        value += self.residual[:, None] * q
        # Existing estimator blended normalized annotation correlation with identity.
        return .5 * z + .5 * self.scale[:, None] * value

    def _matvec(self, z):
        return self._matmat(np.asarray(z).reshape(-1, 1))[:, 0]

    def _rmatvec(self, z):
        return self._matvec(z)


def annotation_correlations(X, beta, direct=None):
    b = beta.toarray() if sparse.issparse(beta) else np.asarray(beta)
    if b.ndim == 1:
        b = b[None, :]
    d = None if direct is None else np.asarray(direct)
    if d is not None and d.shape != (len(b), X.shape[0]):
        raise ValueError('Direct values must have trait-by-gene shape')
    matrices = [AnnotationCorrelation(X, row, None if d is None else d[i]) for i, row in enumerate(b)]
    return matrices[0] if len(matrices) == 1 else matrices
