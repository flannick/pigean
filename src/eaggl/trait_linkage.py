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


def _compute_positive_counts(matrix):
    dense = _as_dense_2d(matrix)
    if dense is None:
        return None
    return np.asarray(np.sum(dense > 0, axis=0), dtype=int)


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
    full_feature_by_trait=None,
    threshold_mode="weighted_thresholded",
    eps=1e-12,
):
    dense_basis = _as_dense_2d(basis)
    dense_feature_by_trait = _sanitize_nonfinite(feature_by_trait)
    dense_full_feature_by_trait = _sanitize_nonfinite(
        full_feature_by_trait if full_feature_by_trait is not None else feature_by_trait
    )

    if dense_basis is None or dense_feature_by_trait is None or dense_full_feature_by_trait is None:
        return None
    if dense_basis.shape[0] != dense_feature_by_trait.shape[0]:
        raise ValueError(
            "Trait linkage basis/target mismatch: %s vs %s"
            % (dense_basis.shape, dense_feature_by_trait.shape)
        )
    if dense_full_feature_by_trait.shape[1] != dense_feature_by_trait.shape[1]:
        raise ValueError(
            "Trait linkage full/retained target mismatch: %s vs %s"
            % (dense_full_feature_by_trait.shape, dense_feature_by_trait.shape)
        )

    masked_trait_support = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
        dense_feature_by_trait,
        threshold_mode,
    )
    full_trait_support = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
        dense_full_feature_by_trait,
        threshold_mode,
    )
    total_trait_support = eaggl_phenotype_annotation.compute_profile_strengths(full_trait_support)
    retained_trait_support = eaggl_phenotype_annotation.compute_profile_strengths(masked_trait_support)
    total_feature_counts = _compute_positive_counts(full_trait_support)
    retained_feature_counts = _compute_positive_counts(masked_trait_support)
    factor_total_mass = np.sum(dense_basis, axis=0, keepdims=True)
    normalized_factor_basis = dense_basis / np.maximum(
        factor_total_mass,
        eps,
    )
    normalized_trait_support = masked_trait_support / np.maximum(total_trait_support[np.newaxis, :], eps)

    joint = np.asarray(
        nnls_project_fn(normalized_factor_basis, normalized_trait_support.T, max_sum=1.0),
        dtype=float,
    )
    marginal = np.zeros_like(joint, dtype=float)
    for factor_index in range(normalized_factor_basis.shape[1]):
        factor_basis = normalized_factor_basis[:, factor_index : factor_index + 1]
        factor_scores = np.asarray(
            nnls_project_fn(factor_basis, normalized_trait_support.T, max_value=1.0),
            dtype=float,
        )
        if factor_scores.ndim == 1:
            marginal[:, factor_index] = factor_scores
        else:
            marginal[:, factor_index] = factor_scores[:, 0]

    residual = np.maximum(0.0, 1.0 - np.sum(joint, axis=1))
    retained_fraction = retained_trait_support / np.maximum(total_trait_support, eps)
    low_retention_flag = np.logical_or(
        retained_feature_counts < 5,
        retained_fraction < 0.1,
    )

    return {
        "prepared_feature_by_trait": masked_trait_support,
        "masked_trait_support": masked_trait_support,
        "full_trait_support": full_trait_support,
        "normalized_trait_support": np.asarray(normalized_trait_support, dtype=float),
        "normalized_factor_basis": np.asarray(normalized_factor_basis, dtype=float),
        "trait_total_support": np.asarray(total_trait_support, dtype=float),
        "retained_trait_support": np.asarray(retained_trait_support, dtype=float),
        "retained_fraction": np.asarray(retained_fraction, dtype=float),
        "total_feature_count": np.asarray(total_feature_counts, dtype=int),
        "retained_feature_count": np.asarray(retained_feature_counts, dtype=int),
        "low_retention_flag": np.asarray(low_retention_flag, dtype=bool),
        "factor_total_mass": np.asarray(np.ravel(factor_total_mass), dtype=float),
        "joint": joint,
        "marginal": marginal,
        "residual": residual,
    }
