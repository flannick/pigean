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
