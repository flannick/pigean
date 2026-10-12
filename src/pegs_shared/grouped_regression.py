"""Exact sufficient statistics for repeated annotation values; no discretization."""
import numpy as np
from scipy import sparse
from scipy.special import expit
from .sparse_regression import annotation_structure_cache


class GroupedDesign:
    def __init__(self, X, num_chains=1):
        self.X = sparse.csc_matrix(X, dtype=float, copy=True)
        self.X.sum_duplicates()
        self.X.eliminate_zeros()
        self.n, self.k = self.X.shape
        cols, values, counts, groups = [], [], [], []
        offset = 0
        cache = annotation_structure_cache()
        for j in range(self.k):
            start, end = self.X.indptr[j:j+2]
            if cache is None:
                val, inverse, count = np.unique(self.X.data[start:end], return_inverse=True, return_counts=True)
                cols.extend([j]*len(val))
                values.extend(val)
                counts.extend(count)
                groups.extend(inverse+offset)
            else:
                val, inverse, count = cache.column_groups(
                    self.n, self.X.indices[start:end], self.X.data[start:end])
                cols.append(np.full(len(val), j, dtype=np.int64))
                values.append(val)
                counts.append(count)
                groups.append(inverse+offset)
            offset += len(val)
        def join(parts, dtype):
            if cache is not None and parts:
                return np.concatenate(parts).astype(dtype, copy=False)
            return np.asarray(parts, dtype=dtype)
        self.columns = join(cols, np.int64)
        self.values = join(values, float)
        self.counts = join(counts, float)
        self.groups = join(groups, np.int64)
        self.zeros = self.n - np.bincount(self.columns, weights=self.counts, minlength=self.k)
        self.chain_columns = np.concatenate([self.columns+i*self.k for i in range(num_chains)])
        self.chain_values = np.tile(self.values, num_chains)
        self.chain_counts = np.tile(self.counts, num_chains)
        self.num_coefficients = self.k*num_chains
        self._correlations = {}

    def moments(self, beta, alpha, max_probability, mask=None):
        if mask is None:
            local = self.chain_columns
            values, counts = self.chain_values, self.chain_counts
        else:
            lookup = np.full(self.num_coefficients, -1, dtype=np.int64)
            lookup[mask] = np.arange(np.sum(mask))
            local = lookup[self.chain_columns]
            keep = local >= 0
            local = local[keep]
            values, counts = self.chain_values[keep], self.chain_counts[keep]
        ex = np.exp(np.minimum(values*beta[local]+alpha[local], 100))
        p = np.minimum(ex/(1+ex), max_probability)
        v = p*(1-p)
        return tuple(np.bincount(local, weights=counts*w, minlength=len(beta))
                     for w in (p, v, values*p, values*v, values*values*v))

    def _prepare_correlation(self, correlation, pseudo):
        from .covariance import validate_correlation
        key = (id(correlation), pseudo)
        if key in self._correlations:
            return self._correlations[key][1]
        r = validate_correlation(correlation, self.n-int(pseudo))
        if not sparse.issparse(r):
            return None
        r = r.tocsr()
        row_sums = np.asarray(r.sum(axis=1)).ravel()
        col_sums = np.asarray(r.sum(axis=0)).ravel()
        diag = r.diagonal()
        real = self.X.indices < r.shape[0]
        all_rows, all_groups = self.X.indices[real], self.groups[real]
        diag_weights = np.bincount(all_groups, weights=diag[all_rows], minlength=len(self.values))
        cross_weights = np.bincount(all_groups, weights=row_sums[all_rows]+col_sums[all_rows], minlength=len(self.values))
        left, right, weights = [], [], []
        for j in range(self.k):
            start, end = self.X.indptr[j:j+2]
            rows = self.X.indices[start:end]
            group = self.groups[start:end]
            if pseudo:
                real = rows < self.n-1
                rows, group = rows[real], group[real]
            if not len(rows):
                continue
            # Read CSR entries directly instead of allocating two submatrices
            # per annotation. X's canonical CSC rows are sorted and unique.
            lengths = r.indptr[rows+1]-r.indptr[rows]
            local_rows = np.repeat(np.arange(len(rows)), lengths)
            starts = r.indptr[rows] - np.r_[0, np.cumsum(lengths[:-1])]
            entries = np.repeat(starts, lengths)+np.arange(len(local_rows))
            targets = r.indices[entries]
            positions = np.searchsorted(rows, targets)
            kept = np.flatnonzero(positions < len(rows))
            kept = kept[(rows[positions[kept]] == targets[kept]) & (positions[kept] != local_rows[kept])]
            if len(kept):
                left.append(group[local_rows[kept]])
                right.append(group[positions[kept]])
                weights.append(r.data[entries[kept]])
        left = np.concatenate(left) if left else np.array([], dtype=np.int64)
        right = np.concatenate(right) if right else np.array([], dtype=np.int64)
        weights = np.concatenate(weights) if weights else np.array([], dtype=float)
        prepared = (r, diag_weights, cross_weights, left, right, weights, row_sums.sum())
        # This cache lives only for one regression call; keep the original alive
        # so an unrelated matrix cannot reuse its identity. Never cache across Gibbs updates.
        self._correlations[key] = (correlation, prepared)
        return prepared

    def inflation(self, correlation, beta=None, alpha=None, pseudo=False):
        from .covariance import marginal_inflation
        prepared = self._prepare_correlation(correlation, pseudo)
        if prepared is None:
            return marginal_inflation(self.X, correlation, beta=beta, alpha=alpha,
                                      pseudo=pseudo, backend='sparse')
        r, diag, cross, left, right, weights, total_r = prepared
        def total(values):
            return np.bincount(self.columns, weights=values, minlength=self.k)
        if beta is None:
            mean = total(self.counts*self.values)/self.n
            constant = -mean
            influence = self.values - mean[self.columns]
        else:
            p0 = expit(np.clip(alpha, -100, 100))
            v0 = p0*(1-p0)
            p = expit(np.clip(self.values*beta[self.columns]+alpha[self.columns], -100, 100))
            v = p*(1-p)
            total_v = self.zeros*v0+total(self.counts*v)
            mean = np.divide(total(self.counts*v*self.values), total_v,
                             out=np.zeros(self.k), where=total_v > 0)
            constant = -np.sqrt(v0)*mean
            influence = np.sqrt(v)*(self.values-mean[self.columns])
        delta = influence-constant[self.columns]
        independent = total(self.counts*influence**2)+self.zeros*constant**2
        quadratic = total(diag*delta**2)
        edge_values = weights*delta[left]*delta[right]
        quadratic += np.bincount(self.columns[left], weights=edge_values, minlength=self.k)
        cross_term = constant*total(cross*delta)
        background = constant**2*total_r
        if pseudo:
            x = self.X[-1, :].toarray().ravel()
            if beta is None:
                pseudo_sq = (x-mean)**2
            else:
                pp = expit(np.clip(x*beta+alpha, -100, 100))
                pseudo_sq = pp*(1-pp)*(x-mean)**2
        else:
            pseudo_sq = 0
        correlated = quadratic+cross_term+background+pseudo_sq
        scale = (total(np.abs(diag)*delta**2)
                 +np.bincount(self.columns[left], weights=np.abs(edge_values), minlength=self.k)
                 +np.abs(cross_term)+np.abs(background)+pseudo_sq)
        unsafe = ((correlated <= 1e-10*scale) | (independent <= 0)
                  | ~np.isfinite(correlated) | ~np.isfinite(independent))
        ratio = np.divide(correlated, independent, out=np.ones(self.k), where=independent > 0)
        ratio[unsafe] = 1
        result = np.sqrt(ratio)
        if np.any(unsafe):
            result[unsafe] = marginal_inflation(
                self.X[:, unsafe], r, beta=None if beta is None else beta[unsafe],
                alpha=None if alpha is None else alpha[unsafe], pseudo=pseudo)
        return result
