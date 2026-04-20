from __future__ import annotations

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


def compute_trait_linkage(
    nnls_project_fn,
    basis,
    feature_by_trait,
    *,
    threshold_mode="weighted_thresholded",
    eps=1e-12,
):
    dense_basis = _as_dense_2d(basis)
    dense_feature_by_trait = _sanitize_nonfinite(feature_by_trait)

    if dense_basis is None or dense_feature_by_trait is None:
        return None
    if dense_basis.shape[0] != dense_feature_by_trait.shape[0]:
        raise ValueError(
            "Trait linkage basis/target mismatch: %s vs %s"
            % (dense_basis.shape, dense_feature_by_trait.shape)
        )

    prepared = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
        dense_feature_by_trait,
        threshold_mode,
    )
    strengths = eaggl_phenotype_annotation.compute_profile_strengths(prepared)
    normalized_basis = dense_basis / np.maximum(
        np.sum(dense_basis, axis=0, keepdims=True),
        eps,
    )
    normalized_targets = prepared / np.maximum(strengths[np.newaxis, :], eps)

    joint = np.asarray(
        nnls_project_fn(normalized_basis, normalized_targets.T, max_sum=1.0),
        dtype=float,
    )
    marginal = np.zeros_like(joint, dtype=float)
    for factor_index in range(normalized_basis.shape[1]):
        factor_basis = normalized_basis[:, factor_index : factor_index + 1]
        factor_scores = np.asarray(
            nnls_project_fn(factor_basis, normalized_targets.T, max_value=1.0),
            dtype=float,
        )
        if factor_scores.ndim == 1:
            marginal[:, factor_index] = factor_scores
        else:
            marginal[:, factor_index] = factor_scores[:, 0]

    residual = np.maximum(0.0, 1.0 - np.sum(joint, axis=1))

    return {
        "prepared_feature_by_trait": prepared,
        "strength": np.asarray(strengths, dtype=float),
        "joint": joint,
        "marginal": marginal,
        "residual": residual,
    }
