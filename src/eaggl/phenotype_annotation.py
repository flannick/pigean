from __future__ import annotations

import numpy as np
from scipy import sparse


def _as_dense_feature_matrix(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix, dtype=float)


def _sanitize_nonfinite_feature_matrix(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        sanitized = matrix.tocsr(copy=True).astype(float)
        if sanitized.nnz > 0:
            sanitized.data = np.nan_to_num(sanitized.data, nan=0.0, posinf=0.0, neginf=0.0)
            sanitized.eliminate_zeros()
        return sanitized
    return np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def compute_profile_strengths(feature_by_profile):
    matrix = _sanitize_nonfinite_feature_matrix(feature_by_profile)
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        if matrix.ndim != 2:
            raise ValueError("Profile strengths expect a 2D sparse matrix")
        return np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    return np.asarray(np.sum(matrix, axis=0), dtype=float)


def prepare_thresholded_profile_input(
    feature_by_profile,
    mode,
    *,
    threshold_value=0.0,
    strict_threshold=True,
):
    matrix = _sanitize_nonfinite_feature_matrix(feature_by_profile)
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        thresholded = matrix.tocsr(copy=True)
        if thresholded.ndim != 2:
            raise ValueError("Thresholded profile input expects a 2D sparse matrix")
        if thresholded.nnz > 0:
            if strict_threshold:
                keep_mask = thresholded.data > threshold_value
            else:
                keep_mask = thresholded.data >= threshold_value
            thresholded.data = np.where(keep_mask, thresholded.data, 0.0)
            if mode == "binary_thresholded":
                thresholded.data = np.where(keep_mask, 1.0, 0.0)
            thresholded.eliminate_zeros()
        return thresholded
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if strict_threshold:
        support_mask = matrix > threshold_value
    else:
        support_mask = matrix >= threshold_value
    if mode == "weighted_thresholded":
        return np.asarray(matrix * support_mask, dtype=float)
    if mode == "binary_thresholded":
        return np.asarray(support_mask, dtype=float)
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
