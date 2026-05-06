from __future__ import annotations

import numpy as np
import scipy.sparse as sparse


def as_dense_anchor_matrix(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    return np.clip(np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def compute_anchor_weight_matrix(row_probabilities, column_probabilities, num_rows, num_cols, *, anchor_aggregation="multi"):
    anchor_aggregation = str(anchor_aggregation)
    if anchor_aggregation not in {"multi", "any"}:
        raise ValueError("anchor aggregation must be one of: multi, any")
    row_probabilities = as_dense_anchor_matrix(row_probabilities)
    column_probabilities = as_dense_anchor_matrix(column_probabilities)
    if row_probabilities is None and column_probabilities is None:
        return np.ones((int(num_rows), int(num_cols)), dtype=float)
    if row_probabilities is None:
        row_probabilities = np.ones((int(num_rows), int(column_probabilities.shape[1])), dtype=float)
    if column_probabilities is None:
        column_probabilities = np.ones((int(num_cols), int(row_probabilities.shape[1])), dtype=float)
    if row_probabilities.shape[1] != column_probabilities.shape[1]:
        raise ValueError("row and column anchor probability matrices must have the same number of anchor traits")
    if anchor_aggregation == "multi":
        return np.asarray(row_probabilities @ column_probabilities.T, dtype=float)
    weights = np.ones((int(num_rows), int(num_cols)), dtype=float)
    for anchor_index in range(row_probabilities.shape[1]):
        same_trait_support = np.outer(row_probabilities[:, anchor_index], column_probabilities[:, anchor_index])
        weights *= 1.0 - np.clip(same_trait_support, 0.0, 1.0)
    return 1.0 - weights


def compute_anchor_weight_row_scale(row_probabilities, column_probabilities, num_rows, num_cols, *, anchor_aggregation="multi"):
    row_probabilities = as_dense_anchor_matrix(row_probabilities)
    column_probabilities = as_dense_anchor_matrix(column_probabilities)
    if row_probabilities is None and column_probabilities is None:
        return np.ones(int(num_rows), dtype=float)
    if row_probabilities is None:
        row_probabilities = np.ones((int(num_rows), int(column_probabilities.shape[1])), dtype=float)
    if column_probabilities is None:
        column_probabilities = np.ones((int(num_cols), int(row_probabilities.shape[1])), dtype=float)
    if row_probabilities.shape[1] != column_probabilities.shape[1]:
        raise ValueError("row and column anchor probability matrices must have the same number of anchor traits")
    if str(anchor_aggregation) == "multi":
        return np.asarray(row_probabilities @ np.mean(column_probabilities, axis=0), dtype=float).ravel()
    out = np.zeros(int(num_rows), dtype=float)
    chunk_size = 1024
    for start in range(0, int(num_rows), chunk_size):
        stop = min(int(num_rows), start + chunk_size)
        prod = np.ones((stop - start, int(num_cols)), dtype=float)
        for anchor_index in range(row_probabilities.shape[1]):
            prod *= 1.0 - np.outer(row_probabilities[start:stop, anchor_index], column_probabilities[:, anchor_index])
        out[start:stop] = np.mean(1.0 - prod, axis=1)
    return out


def compute_noisy_or_anchor_summary_for_projection(matrix):
    if matrix is None:
        return None
    matrix = as_dense_anchor_matrix(matrix)
    return np.hstack((matrix, 1.0 - np.prod(1.0 - matrix, axis=1)[:, np.newaxis]))
