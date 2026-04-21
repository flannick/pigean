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


def _compute_positive_counts(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        indicator = matrix.copy()
        if indicator.nnz > 0:
            indicator.data = np.where(indicator.data > 0.0, 1.0, 0.0)
            indicator.eliminate_zeros()
        return np.asarray(indicator.sum(axis=0)).ravel().astype(int)
    dense = _as_dense_2d(matrix)
    return np.asarray(np.sum(dense > 0, axis=0), dtype=int)


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


def _normalize_profiles_by_strength(feature_by_profile, strengths, *, eps=1e-12):
    if feature_by_profile is None:
        return None
    inv_strengths = 1.0 / np.maximum(np.asarray(strengths, dtype=float), eps)
    if sparse.issparse(feature_by_profile):
        return feature_by_profile @ sparse.diags(inv_strengths, format="csr")
    return np.asarray(feature_by_profile, dtype=float) * inv_strengths[np.newaxis, :]


def _mask_full_profile_rows(feature_by_profile, row_mask):
    if feature_by_profile is None:
        return None
    mask = np.asarray(row_mask, dtype=bool)
    if sparse.issparse(feature_by_profile):
        masked = feature_by_profile.tocsr(copy=True)
        drop_rows = np.flatnonzero(~mask)
        if len(drop_rows) > 0:
            masked[drop_rows, :] = 0.0
            masked.eliminate_zeros()
        return masked
    masked = np.asarray(feature_by_profile, dtype=float).copy()
    masked[~mask, :] = 0.0
    return masked


def _build_full_basis_matrix(dense_basis, basis_mask, *, expected_num_rows):
    if basis_mask is None:
        if dense_basis.shape[0] != expected_num_rows:
            raise ValueError(
                "Trait linkage basis rows %s must match feature rows %s when no basis mask is provided"
                % (dense_basis.shape[0], expected_num_rows)
            )
        return np.asarray(dense_basis, dtype=float), np.full(expected_num_rows, True, dtype=bool)

    mask = np.asarray(basis_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("Trait linkage basis mask must be 1D")
    if mask.shape[0] != expected_num_rows:
        raise ValueError(
            "Trait linkage basis mask length %s must match feature rows %s"
            % (mask.shape[0], expected_num_rows)
        )

    if dense_basis.shape[0] == mask.shape[0]:
        full_basis = np.asarray(dense_basis, dtype=float).copy()
    elif dense_basis.shape[0] == int(np.sum(mask)):
        full_basis = np.zeros((mask.shape[0], dense_basis.shape[1]), dtype=float)
        full_basis[mask, :] = np.asarray(dense_basis, dtype=float)
    else:
        raise ValueError(
            "Trait linkage basis rows %s must match mask length %s or kept rows %s"
            % (dense_basis.shape[0], mask.shape[0], int(np.sum(mask)))
        )

    # Keep linkage on the retained factorized universe while still solving in the full objective space.
    full_basis[~mask, :] = 0.0
    return full_basis, mask


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
    basis_mask=None,
    threshold_mode="weighted_thresholded",
    threshold_value=1.0,
    strict_threshold=True,
    computation_mode="sparse_full",
    eps=1e-12,
):
    if computation_mode not in {"dense_full", "sparse_full"}:
        raise ValueError("Unknown trait linkage computation mode: %s" % computation_mode)
    dense_basis = _sanitize_nonfinite(basis)
    target_feature_by_trait = _sanitize_nonfinite_preserve_sparse(feature_by_trait)
    full_target_feature_by_trait = _sanitize_nonfinite_preserve_sparse(
        full_feature_by_trait if full_feature_by_trait is not None else feature_by_trait
    )

    if dense_basis is None or target_feature_by_trait is None or full_target_feature_by_trait is None:
        return None
    if full_target_feature_by_trait.shape[1] != target_feature_by_trait.shape[1]:
        raise ValueError(
            "Trait linkage full/target column mismatch: %s vs %s"
            % (full_target_feature_by_trait.shape, target_feature_by_trait.shape)
        )

    expected_num_rows = full_target_feature_by_trait.shape[0]
    if basis_mask is None:
        retained_basis_mask = np.full(expected_num_rows, True, dtype=bool)
        if dense_basis.shape[0] != expected_num_rows:
            raise ValueError(
                "Trait linkage basis rows %s must match feature rows %s when no basis mask is provided"
                % (dense_basis.shape[0], expected_num_rows)
            )
        retained_basis = np.asarray(dense_basis, dtype=float)
        full_basis = np.asarray(dense_basis, dtype=float)
    else:
        retained_basis_mask = np.asarray(basis_mask, dtype=bool)
        if retained_basis_mask.ndim != 1 or retained_basis_mask.shape[0] != expected_num_rows:
            raise ValueError(
                "Trait linkage basis mask length %s must match feature rows %s"
                % (retained_basis_mask.shape[0] if retained_basis_mask.ndim == 1 else "invalid", expected_num_rows)
            )
        if dense_basis.shape[0] == expected_num_rows:
            retained_basis = np.asarray(dense_basis[retained_basis_mask, :], dtype=float)
        elif dense_basis.shape[0] == int(np.sum(retained_basis_mask)):
            retained_basis = np.asarray(dense_basis, dtype=float)
        else:
            raise ValueError(
                "Trait linkage basis rows %s must match mask length %s or kept rows %s"
                % (dense_basis.shape[0], expected_num_rows, int(np.sum(retained_basis_mask)))
            )
        full_basis, retained_basis_mask = _build_full_basis_matrix(
            dense_basis,
            retained_basis_mask,
            expected_num_rows=expected_num_rows,
        )

    full_trait_support = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
        full_target_feature_by_trait,
        threshold_mode,
        threshold_value=threshold_value,
        strict_threshold=strict_threshold,
    )
    if computation_mode == "dense_full":
        masked_full_trait_support = np.asarray(full_trait_support, dtype=float).copy()
        masked_full_trait_support[~retained_basis_mask, :] = 0.0
        masked_trait_support = np.asarray(masked_full_trait_support[retained_basis_mask, :], dtype=float)
    else:
        masked_full_trait_support = _mask_full_profile_rows(full_trait_support, retained_basis_mask)
        masked_trait_support = masked_full_trait_support[retained_basis_mask, :]

    total_trait_support = eaggl_phenotype_annotation.compute_profile_strengths(full_trait_support)
    retained_trait_support = eaggl_phenotype_annotation.compute_profile_strengths(masked_trait_support)
    total_feature_counts = _compute_positive_counts(full_trait_support)
    retained_feature_counts = _compute_positive_counts(masked_trait_support)
    trait_n_eff = _compute_effective_feature_count(full_trait_support, eps=eps)
    retained_n_eff = _compute_effective_feature_count(masked_trait_support, eps=eps)

    factor_total_mass = np.sum(retained_basis, axis=0, keepdims=True)
    normalized_factor_basis_retained = retained_basis / np.maximum(factor_total_mass, eps)
    normalized_trait_support_for_projection = _normalize_profiles_by_strength(
        masked_trait_support,
        total_trait_support,
        eps=eps,
    )
    normalized_factor_basis = full_basis / np.maximum(factor_total_mass, eps)
    if computation_mode == "dense_full":
        normalized_trait_support = masked_full_trait_support / np.maximum(total_trait_support[np.newaxis, :], eps)
    else:
        normalized_trait_support = _normalize_profiles_by_strength(
            masked_full_trait_support,
            total_trait_support,
            eps=eps,
        )

    joint = np.asarray(
        nnls_project_fn(normalized_factor_basis_retained, normalized_trait_support_for_projection.T, max_sum=1.0),
        dtype=float,
    )
    marginal = np.zeros_like(joint, dtype=float)
    for factor_index in range(normalized_factor_basis_retained.shape[1]):
        factor_basis = normalized_factor_basis_retained[:, factor_index : factor_index + 1]
        factor_scores = np.asarray(
            nnls_project_fn(factor_basis, normalized_trait_support_for_projection.T, max_value=1.0),
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
        "normalized_trait_support": normalized_trait_support,
        "normalized_factor_basis": np.asarray(normalized_factor_basis, dtype=float),
        "trait_total_support": np.asarray(total_trait_support, dtype=float),
        "retained_trait_support": np.asarray(retained_trait_support, dtype=float),
        "retained_fraction": np.asarray(retained_fraction, dtype=float),
        "total_feature_count": np.asarray(total_feature_counts, dtype=int),
        "retained_feature_count": np.asarray(retained_feature_counts, dtype=int),
        "trait_n_eff": np.asarray(trait_n_eff, dtype=float),
        "retained_n_eff": np.asarray(retained_n_eff, dtype=float),
        "low_retention_flag": np.asarray(low_retention_flag, dtype=bool),
        "factor_total_mass": np.asarray(np.ravel(factor_total_mass), dtype=float),
        "joint": joint,
        "marginal": marginal,
        "residual": residual,
    }
