from __future__ import annotations

import numpy as np
from scipy import sparse


def _as_dense_feature_matrix(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix, dtype=float)


def compute_profile_strengths(feature_by_profile):
    dense = _as_dense_feature_matrix(feature_by_profile)
    if dense is None:
        return None
    if dense.ndim == 1:
        dense = dense[:, np.newaxis]
    return np.asarray(np.sum(dense, axis=0), dtype=float)


def prepare_thresholded_profile_input(feature_by_profile, mode):
    dense = _as_dense_feature_matrix(feature_by_profile)
    if dense is None:
        return None
    if dense.ndim == 1:
        dense = dense[:, np.newaxis]
    if mode == "weighted_thresholded":
        return np.asarray(np.maximum(dense, 0.0), dtype=float)
    if mode == "binary_thresholded":
        return np.asarray(dense > 0, dtype=float)
    raise ValueError("Unknown phenotype capture input mode: %s" % mode)


def project_phenotype_capture(nnls_project_fn, basis, feature_by_pheno, *, eps=1e-12, max_sum=1.0):
    dense_basis = _as_dense_feature_matrix(basis)
    dense_feature_by_pheno = _as_dense_feature_matrix(feature_by_pheno)
    if dense_basis is None or dense_feature_by_pheno is None:
        return None, None
    if dense_basis.ndim != 2 or dense_feature_by_pheno.ndim != 2:
        raise ValueError("Phenotype capture projection expects 2D basis and target matrices")
    if dense_basis.shape[0] != dense_feature_by_pheno.shape[0]:
        raise ValueError(
            "Phenotype capture projection basis/target mismatch: %s vs %s"
            % (dense_basis.shape, dense_feature_by_pheno.shape)
        )

    strengths = compute_profile_strengths(dense_feature_by_pheno)
    normalized_basis = dense_basis / np.maximum(np.sum(dense_basis, axis=0, keepdims=True), eps)
    normalized_targets = dense_feature_by_pheno / np.maximum(strengths[np.newaxis, :], eps)
    capture_weights = nnls_project_fn(normalized_basis, normalized_targets.T, max_sum=max_sum)
    return np.asarray(capture_weights, dtype=float), strengths


def rank_top_capture_indices(capture_matrix, strengths, num_top):
    if capture_matrix is None:
        return None
    capture = _as_dense_feature_matrix(capture_matrix)
    if capture.ndim != 2:
        raise ValueError("Capture matrix must be 2D")
    if strengths is None:
        strengths = np.zeros(capture.shape[0], dtype=float)
    else:
        strengths = np.asarray(strengths, dtype=float)

    top_by_factor = []
    for factor_index in range(capture.shape[1]):
        ordered = np.lexsort((-strengths, -capture[:, factor_index]))
        top_by_factor.append(ordered[:num_top])
    return np.array(top_by_factor, dtype=int).T
