from __future__ import annotations

import gzip

import numpy as np
from scipy import sparse

from . import phenotype_annotation as eaggl_phenotype_annotation


def _as_dense_2d(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2:
        raise ValueError("Trait linkage expects 2D inputs")
    return matrix


def _sanitize_nonfinite(matrix):
    if matrix is None:
        return None
    return np.nan_to_num(_as_dense_2d(matrix), nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_nonfinite_preserve_sparse(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        sanitized = matrix.tocsr(copy=True).astype(float)
        if sanitized.nnz > 0:
            sanitized.data = np.nan_to_num(sanitized.data, nan=0.0, posinf=0.0, neginf=0.0)
            sanitized.eliminate_zeros()
        return sanitized
    return np.nan_to_num(_as_dense_2d(matrix), nan=0.0, posinf=0.0, neginf=0.0)


def _compute_effective_feature_count(matrix, *, eps=1e-12):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        support_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
        support_square_sums = np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(float)
        n_eff = np.square(support_sums) / np.maximum(support_square_sums, eps)
        n_eff = np.where(support_sums > 0.0, n_eff, 0.0)
        return np.asarray(n_eff, dtype=float)
    dense = _as_dense_2d(matrix)
    support_sums = np.sum(dense, axis=0, dtype=float)
    support_square_sums = np.sum(np.square(dense), axis=0, dtype=float)
    n_eff = np.square(support_sums) / np.maximum(support_square_sums, eps)
    n_eff = np.where(support_sums > 0.0, n_eff, 0.0)
    return np.asarray(n_eff, dtype=float)

def resolve_trait_linkage_source(
    requested_source,
    *,
    combined=None,
    log_bf=None,
    prior=None,
    log_fn=None,
    info_level=1,
    context_label="trait linkage",
):
    allowed = {"auto", "combined", "log_bf", "prior"}
    if requested_source not in allowed:
        raise ValueError("Unknown trait linkage source: %s" % requested_source)

    candidates = [
        ("combined", combined),
        ("log_bf", log_bf),
        ("prior", prior),
    ]
    if requested_source != "auto":
        candidates = [candidate for candidate in candidates if candidate[0] == requested_source]

    for label, matrix in candidates:
        if matrix is None:
            continue
        dense_matrix = _sanitize_nonfinite(matrix)
        if log_fn is not None and requested_source == "auto":
            log_fn("Using %s support surface for %s" % (label, context_label), info_level)
        return dense_matrix, label
    return None, None


def threshold_factor_gene_basis(basis, threshold=0.05):
    """Return nonnegative factor loadings with small gene weights zeroed."""
    dense = np.maximum(_sanitize_nonfinite(basis), 0.0)
    threshold = 0.0 if threshold is None else float(threshold)
    if threshold > 0.0:
        dense = np.where(dense >= threshold, dense, 0.0)
    return dense


def factor_basis_cosines(basis, *, eps=1e-12):
    """Cosine of each row-loading vector against each factor indicator."""
    dense = np.maximum(_sanitize_nonfinite(basis), 0.0)
    row_norms = np.linalg.norm(dense, axis=1, keepdims=True)
    return np.divide(dense, np.maximum(row_norms, eps), out=np.zeros_like(dense), where=row_norms > eps)


def write_factor_gmt(path, genes, factor_names, basis, threshold=0.05):
    """Export weighted EAGGL factors as a GMT-like factor gene-set file."""
    if path is None:
        return
    W = threshold_factor_gene_basis(basis, threshold)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as out:
        for factor_index, factor_name in enumerate(factor_names):
            keep = np.flatnonzero(W[:, factor_index] > 0.0)
            fields = [str(factor_name), "EAGGL_factor"]
            fields.extend(str(genes[i]) for i in keep)
            out.write("%s\n" % "\t".join(fields))



def compute_factor_trait_links(
    nnls_project_fn,
    basis,
    feature_by_trait,
    *,
    basis_mask=None,
    threshold_mode="weighted_thresholded",
    threshold_value=1.0,
    trait_response_source_name="combined",
    factor_loading_threshold=0.05,
    nnls_loading_threshold=0.0,
    nnls_max_value=1.0,
    computation_mode="sparse_full",
):
    """Compute native EAGGL factor-trait links by NNLS projection only.

    Factor-trait beta statistics are intentionally not computed here. To obtain
    PIGEAN beta, beta_uncorrected, beta_tilde, SE, and p-value columns, export
    factors with ``write_factor_gmt`` / ``--factor-gmt-out`` and run the PIGEAN
    multi-Y beta workflow on that GMT. Keeping this function NNLS-only avoids a
    parallel OLS/ridge implementation with semantics that differ from PIGEAN.
    """
    if computation_mode not in {"dense_full", "sparse_full"}:
        raise ValueError("Unknown trait linkage computation mode: %s" % computation_mode)
    dense_basis = threshold_factor_gene_basis(basis, factor_loading_threshold)
    target_feature_by_trait = _sanitize_nonfinite_preserve_sparse(feature_by_trait)
    if dense_basis is None or target_feature_by_trait is None:
        return None
    expected_num_rows = target_feature_by_trait.shape[0]
    if basis_mask is None:
        retained_mask = np.full(expected_num_rows, True, dtype=bool)
        if dense_basis.shape[0] != expected_num_rows:
            raise ValueError(
                "Trait linkage basis rows %s must match feature rows %s when no basis mask is provided"
                % (dense_basis.shape[0], expected_num_rows)
            )
        retained_basis = dense_basis
    else:
        retained_mask = np.asarray(basis_mask, dtype=bool)
        if retained_mask.shape[0] != expected_num_rows:
            raise ValueError("Trait linkage basis mask length must match feature rows")
        if dense_basis.shape[0] == expected_num_rows:
            retained_basis = dense_basis[retained_mask, :]
        elif dense_basis.shape[0] == int(np.sum(retained_mask)):
            retained_basis = dense_basis
        else:
            raise ValueError("Trait linkage basis rows do not match mask length or kept rows")

    full_support = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
        target_feature_by_trait,
        threshold_mode,
        threshold_value=threshold_value,
        strict_threshold=True,
    )
    retained_support = full_support[retained_mask, :]
    retained_dense = retained_support.toarray() if sparse.issparse(retained_support) else np.asarray(retained_support, dtype=float)
    retained_dense = np.nan_to_num(retained_dense, nan=0.0, posinf=0.0, neginf=0.0)

    trait_total_support = eaggl_phenotype_annotation.compute_profile_strengths(full_support)
    trait_n_eff = _compute_effective_feature_count(full_support)
    retained_n_eff = _compute_effective_feature_count(retained_support)
    nnls_max_value = None if nnls_max_value is None or float(nnls_max_value) <= 0.0 else float(nnls_max_value)
    nnls_loadings = np.asarray(
        nnls_project_fn(retained_basis, retained_dense.T, max_sum=None, max_value=nnls_max_value),
        dtype=float,
    )
    nnls_loadings = np.maximum(nnls_loadings, 0.0)
    nnls_loading_threshold = 0.0 if nnls_loading_threshold is None else float(nnls_loading_threshold)
    if nnls_loading_threshold > 0.0:
        nnls_loadings = np.where(nnls_loadings >= nnls_loading_threshold, nnls_loadings, 0.0)
    factor_weight_sum = np.sum(retained_basis, axis=0)
    factor_num_genes = np.sum(retained_basis > 0.0, axis=0).astype(int)
    return {
        "nnls": nnls_loadings,
        "trait_total_support": trait_total_support,
        "trait_n_eff": trait_n_eff,
        "retained_n_eff": retained_n_eff,
        "factor_weight_sum": factor_weight_sum,
        "factor_num_genes": factor_num_genes,
        "factor_loading_threshold": float(factor_loading_threshold),
        "trait_response_source": trait_response_source_name,
    }
