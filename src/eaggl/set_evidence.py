"""Describe gene-set overlap on the same gene universe as the factor basis."""
import numpy as np
from scipy import sparse


METRICS = ("Coverage", "Enrichment", "MeanMemberLoading", "SupportingGenes")
COUNTS = ("BasisGenes", "MemberGenes")


class SetEvidence:
    def __init__(self, matrix, basis, genes):
        self.weights = np.asarray(basis, dtype=float)
        self.genes = np.asarray(genes)
        if matrix.shape[0] != len(genes) or self.weights.shape[0] != len(genes):
            raise ValueError("Gene-set membership and factor basis must share the gene axis")
        binary = sparse.csc_matrix(matrix > 0)
        binary.eliminate_zeros()
        binary.sort_indices()
        self.membership = binary
        self.mass = self.weights.sum(axis=0)
        self.shared = binary.T @ self.weights

    def cumulative_coverage(self, coefficients, set_ids):
        """Mass covered by the union through each positive set in loading order."""
        memberships = self.membership.tocsr()
        represented = np.flatnonzero(np.diff(memberships.indptr))
        n = len(set_ids)
        result = np.full(coefficients.shape, np.nan)
        if not len(represented):
            result[(coefficients > 0) & (self.mass > 0)] = 0.
            return result
        starts = memberships.indptr[represented]
        # A gene contributes once, at the rank of its first containing set.
        alphabetic = np.argsort(set_ids, kind="stable")
        ranks = np.empty(n, dtype=int)
        for j, mass in enumerate(self.mass):
            if mass <= 0:
                continue
            order = alphabetic[np.argsort(-coefficients[alphabetic, j], kind="stable")]
            ranks[order] = np.arange(n)
            first = np.minimum.reduceat(ranks[memberships.indices], starts)
            gain = np.bincount(first, weights=self.weights[represented, j], minlength=n)
            count = np.count_nonzero(coefficients[order, j] > 0)
            result[order[:count], j] = np.cumsum(gain[:count]) / mass
        return result

    def describe(self, row, columns):
        start, stop = self.membership.indptr[row:row+2]
        members = self.membership.indices[start:stop]
        count, total = len(members), len(self.genes)
        shared = self.shared[row, columns]
        mass = self.mass[columns]
        coverage = np.divide(shared, mass, out=np.full_like(shared, np.nan), where=mass > 0)
        mean = shared / count if count else np.full_like(shared, np.nan)
        enrichment = coverage * (total / count) if count else np.full_like(shared, np.nan)
        # Stable membership order resolves tied loadings. Sort only expressed columns.
        supporters = [""] * len(columns)
        for j in np.flatnonzero(shared > 0):
            w = self.weights[members, columns[j]]
            best = np.argsort(-w, kind="stable")[:3]
            supporters[j] = ",".join(self.genes[members[best[w[best] > 0]]])
        return (total, count), (coverage, enrichment, mean, supporters)


def from_state(state):
    supplied = getattr(state, "supplied_set_evidence", None)
    if supplied is not None:
        return supplied
    x, w = getattr(state, "X_orig", None), getattr(state, "exp_gene_factors", None)
    if x is None or w is None:
        return None
    genes = np.asarray(state.genes)
    mask = getattr(state, "gene_in_discovery_mask", None)
    if mask is None:
        mask = np.ones(len(genes), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    if x.shape != (len(genes), len(state.gene_sets)) or mask.shape != (len(genes),):
        raise ValueError("Cannot align retained genes and gene sets for overlap metrics")
    basis = w[mask] if len(w) == len(genes) else w
    return SetEvidence(x[mask], basis, genes[mask])


def metric_columns(factors):
    return list(COUNTS) + [f"{name}_{factor}" for name in (*METRICS, "CumulativeCoverage") for factor in factors]


def display(value):
    if isinstance(value, str):
        return value
    return f"{value:.4g}" if np.isfinite(value) else "NA"


def metric_values(evidence, row, columns):
    if evidence is None:
        return ["NA"] * (2 + len(METRICS)*len(columns))
    counts, metrics = evidence.describe(row, columns)
    return [str(x) for x in counts] + [display(x) for values in metrics for x in values]
