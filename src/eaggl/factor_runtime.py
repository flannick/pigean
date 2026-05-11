from __future__ import annotations

import copy
import gzip
import math
import os
import tempfile
import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy
import scipy.sparse as sparse

from . import anchor_aggregation as eaggl_anchor_aggregation
from . import phenotype_annotation as eaggl_phenotype_annotation
from . import trait_linkage as eaggl_trait_linkage


@dataclass
class DiscoveryPlan:
    retained_gene_set_mask_full: object
    in_discovery_mask_full: object
    discovery_family_id_full: object
    discovery_representative_mask_full: object
    discovery_family_size_full: object
    discovery_family_mean_similarity_full: object
    discovery_family_effective_size_full: object
    discovery_weight_full: object
    discovery_prob_vector: object
    discovery_row_indices_full: object
    retained_row_indices_full: object


_DEFAULT_GENE_GENE_PAIR_EFFECTIVE_SIZE = 50.0


def _state_discovery_model(state):
    return str(getattr(state, "discovery_model", "gene_by_annotation") or "gene_by_annotation")


def _gene_gene_full_annotation_counts(state, gene_indices):
    X_full = state.X_orig[gene_indices, :]
    if sparse.issparse(X_full):
        counts = np.asarray(X_full.getnnz(axis=1), dtype=float).ravel()
    else:
        counts = np.sum(np.asarray(X_full) != 0, axis=1).astype(float)
    return counts


def _gene_gene_retained_annotation_count_exposure(X_retained):
    if sparse.issparse(X_retained):
        counts = np.asarray(X_retained.getnnz(axis=1), dtype=float).ravel()
    else:
        counts = np.sum(np.asarray(X_retained) != 0, axis=1).astype(float)
    return np.log1p(counts), counts


def _fit_linear_log_count_pair_correction(
    L_anchor,
    exposure,
):
    L_anchor = np.asarray(L_anchor, dtype=float)
    n = L_anchor.shape[0]
    diagnostics = {
        "mode": "linear",
        "fit_pair_count": 0,
        "total_pair_count": int(max(0, n * (n - 1) // 2)),
        "fit_applied": False,
        "message": "",
        "intercept": 0.0,
        "slope": 0.0,
    }
    if n < 2:
        return np.asarray(L_anchor, dtype=float), np.zeros_like(L_anchor, dtype=float), diagnostics

    tri_i, tri_j = np.triu_indices(n, k=1)
    total_pairs = int(tri_i.size)
    if total_pairs == 0:
        return np.asarray(L_anchor, dtype=float), np.zeros_like(L_anchor, dtype=float), diagnostics

    exposure = np.asarray(exposure, dtype=float)
    fit_x = np.nan_to_num(exposure[tri_i] + exposure[tri_j], nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(L_anchor[tri_i, tri_j], nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        return np.asarray(L_anchor, dtype=float), np.zeros_like(L_anchor, dtype=float), diagnostics

    x_mean = float(np.mean(fit_x))
    y_mean = float(np.mean(y))
    x_var = float(np.mean((fit_x - x_mean) ** 2))
    if x_var <= 1e-12:
        slope = 0.0
        intercept = y_mean
        message = "constant annotation-count feature; fitted intercept only"
    else:
        slope = float(np.mean((fit_x - x_mean) * (y - y_mean)) / x_var)
        intercept = float(y_mean - slope * x_mean)
        message = "ok"

    correction = intercept + slope * (exposure[:, np.newaxis] + exposure[np.newaxis, :])
    correction = 0.5 * (correction + correction.T)
    correction = np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    corrected = np.asarray(L_anchor, dtype=float) - correction
    corrected = 0.5 * (corrected + corrected.T)
    diagnostics.update(
        {
            "fit_pair_count": int(y.size),
            "fit_applied": True,
            "message": message,
            "intercept": float(intercept),
            "slope": float(slope),
            "correction_mean": float(np.mean(correction)) if correction.size > 0 else 0.0,
            "correction_std": float(np.std(correction)) if correction.size > 0 else 0.0,
            "corrected_L_mean": float(np.mean(corrected)) if corrected.size > 0 else 0.0,
            "corrected_L_std": float(np.std(corrected)) if corrected.size > 0 else 0.0,
        }
    )
    return corrected, correction, diagnostics


def _discovery_model_prefers_gene_basis(state):
    return _state_discovery_model(state) == "gene_by_gene"


def _resolve_gene_gene_pair_prior(
    num_genes,
    explicit_pair_prior=None,
    effective_size=None,
    background_prior=None,
):
    del background_prior
    if explicit_pair_prior is not None:
        pair_prior = float(explicit_pair_prior)
        if not (0.0 < pair_prior < 1.0):
            raise ValueError("gene-gene pair prior must be in (0, 1)")
        return pair_prior, "explicit_prior"

    if effective_size is None:
        effective_size = _DEFAULT_GENE_GENE_PAIR_EFFECTIVE_SIZE
        source = "default_effective_size"
    else:
        source = "effective_size"
    effective_size = float(effective_size)
    if effective_size <= 1.0:
        raise ValueError("gene-gene pair prior effective size must be > 1")
    denom = max(1.0, float(int(num_genes)) - 1.0)
    pair_prior = (effective_size - 1.0) / denom
    pair_prior = min(max(pair_prior, 1e-8), 0.95)
    return float(pair_prior), source


_as_dense_anchor_matrix = eaggl_anchor_aggregation.as_dense_anchor_matrix
_compute_anchor_weight_matrix = eaggl_anchor_aggregation.compute_anchor_weight_matrix
_compute_anchor_weight_row_scale = eaggl_anchor_aggregation.compute_anchor_weight_row_scale


def _canonical_gene_gene_active_indices(state, gene_mask):
    active_gene_indices = np.where(np.asarray(gene_mask, dtype=bool))[0]
    if active_gene_indices.size == 0:
        return active_gene_indices
    active_gene_order = np.argsort(
        np.asarray([str(state.genes[index]) for index in active_gene_indices], dtype=object),
        kind="mergesort",
    )
    return active_gene_indices[active_gene_order]


def _build_gene_gene_pair_matrix(
    state,
    *,
    gene_mask,
    gene_indices=None,
    gene_set_mask,
    beta_source,
    beta_values=None,
    beta_values_scaled=False,
    gene_prob_values=None,
    pair_prior,
    logbf_base,
    matrix_floor,
    excess_probability,
    diagonal_weight,
    log_fn,
    info_level,
    anchor_aggregation="multi",
    profligate_correction="none",
):
    anchor_aggregation = str(anchor_aggregation)
    if anchor_aggregation not in {"multi", "any"}:
        raise ValueError("anchor aggregation must be one of: multi, any")
    profligate_correction = str(profligate_correction)
    if profligate_correction not in {"none", "linear"}:
        raise ValueError("gene-gene profligate correction must be one of: none, linear")
    beta_source_label = str(beta_source)
    if beta_source_label != "beta":
        raise ValueError("gene-by-gene discovery requires corrected beta values; beta_uncorrected is not supported as a pair-weight source")
    if beta_values is None:
        if getattr(state, "betas", None) is None:
            raise ValueError("Corrected betas are unavailable; rerun with corrected gene-set statistics")
        beta_values = np.asarray(state.betas, dtype=float)
        beta_source_label = "beta"
    elif sparse.issparse(beta_values):
        beta_values = beta_values.toarray()
    else:
        beta_values = np.asarray(beta_values, dtype=float)

    beta_matrix = np.asarray(beta_values, dtype=float)
    if beta_matrix.ndim == 1:
        beta_matrix = beta_matrix[:, np.newaxis]

    scale_factors = np.asarray(getattr(state, "scale_factors", np.ones(beta_matrix.shape[0], dtype=float)), dtype=float)
    scale_factors = np.where(scale_factors == 0, 1.0, scale_factors)
    if not bool(beta_values_scaled):
        beta_matrix = beta_matrix / scale_factors[:, np.newaxis]
    beta_matrix = np.nan_to_num(beta_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    beta_matrix = beta_matrix[np.asarray(gene_set_mask, dtype=bool), :]

    if gene_indices is None:
        gene_indices = np.where(np.asarray(gene_mask, dtype=bool))[0]
    gene_indices = np.asarray(gene_indices, dtype=int)

    X_retained = state.X_orig[gene_indices, :][:, np.asarray(gene_set_mask, dtype=bool)]
    if not sparse.issparse(X_retained):
        X_retained = sparse.csr_matrix(np.asarray(X_retained, dtype=float))
    X_retained = X_retained.tocsr()
    retained_annotation_exposure, retained_annotation_counts = _gene_gene_retained_annotation_count_exposure(X_retained)
    full_annotation_counts = _gene_gene_full_annotation_counts(state, gene_indices)

    pair_logit = scipy.special.logit(float(pair_prior))
    gene_prob_matrix = _as_dense_anchor_matrix(gene_prob_values)
    if gene_prob_matrix is not None:
        gene_prob_matrix = gene_prob_matrix[gene_indices, :]
        if gene_prob_matrix.shape[1] != beta_matrix.shape[1]:
            raise ValueError(
                "gene-by-gene requires the same number of gene and gene-set anchor traits"
            )

    per_anchor_targets = []
    per_anchor_L = []
    profligate_diagnostics = []
    for anchor_index in range(beta_matrix.shape[1]):
        beta_vector = beta_matrix[:, anchor_index]
        X_beta = X_retained.multiply(beta_vector.reshape((1, -1)))
        L_anchor = (X_beta @ X_retained.T).toarray()
        L_anchor = 0.5 * (L_anchor + L_anchor.T)
        if str(logbf_base) == "log10":
            L_anchor = float(math.log(10.0)) * L_anchor
        if profligate_correction == "linear":
            L_anchor, _correction, correction_diag = _fit_linear_log_count_pair_correction(
                L_anchor,
                retained_annotation_exposure,
            )
            profligate_diagnostics.append(correction_diag)
            if not correction_diag.get("fit_applied", False) and log_fn is not None:
                log_fn(
                    "Gene-gene profligate correction was not fit for anchor %d: %s"
                    % (anchor_index + 1, correction_diag.get("message", "")),
                    info_level,
                )
        else:
            profligate_diagnostics.append({"mode": "none"})

        P_anchor = scipy.special.expit(pair_logit + L_anchor)
        if excess_probability:
            R_anchor = np.clip((P_anchor - float(pair_prior)) / max(1.0 - float(pair_prior), 1e-12), 0.0, 1.0)
        else:
            R_anchor = np.clip(P_anchor, 0.0, 1.0)

        if gene_prob_matrix is not None:
            prob_col = gene_prob_matrix[:, anchor_index]
            gate_anchor = np.clip(np.outer(prob_col, prob_col), 0.0, 1.0)
        else:
            gate_anchor = np.ones_like(R_anchor, dtype=float)
        T_anchor = np.asarray(R_anchor * gate_anchor, dtype=float)
        if float(matrix_floor) > 0.0:
            T_anchor[T_anchor < float(matrix_floor)] = 0.0
        if float(diagonal_weight) == 0.0:
            np.fill_diagonal(T_anchor, 0.0)
        elif float(diagonal_weight) != 1.0:
            diag = np.diag(T_anchor).copy() * float(diagonal_weight)
            np.fill_diagonal(T_anchor, diag)
        per_anchor_targets.append(T_anchor)
        per_anchor_L.append(L_anchor)

    if len(per_anchor_targets) == 1:
        M = np.asarray(per_anchor_targets[0], dtype=float)
        target_mode = "single_trait_target"
    elif anchor_aggregation == "multi":
        target_stack = np.stack(per_anchor_targets, axis=0)
        M = np.mean(target_stack, axis=0)
        target_mode = "equal_weight_multi_view_single_trait_targets"
    else:
        M = 1.0 - np.prod([1.0 - np.clip(target, 0.0, 1.0) for target in per_anchor_targets], axis=0)
        target_mode = "noisy_or_of_single_trait_targets"

    M = np.nan_to_num(np.asarray(M, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    M = np.clip(M, 0.0, 1.0)
    if float(diagonal_weight) == 0.0:
        np.fill_diagonal(M, 0.0)

    L = np.mean(per_anchor_L, axis=0) if len(per_anchor_L) > 0 else np.zeros_like(M)
    P = scipy.special.expit(pair_logit + L)
    M_row_sums = np.sum(M, axis=1) if M.size > 0 else np.array([], dtype=float)
    per_anchor_nonzero_edge_counts = [int(np.sum(target > 0.0)) for target in per_anchor_targets]
    per_anchor_mean_target = [float(np.mean(target)) if target.size > 0 else 0.0 for target in per_anchor_targets]
    retained_annotation_count_summary = {
        "mean": float(np.mean(retained_annotation_counts)) if retained_annotation_counts.size > 0 else 0.0,
        "median": float(np.median(retained_annotation_counts)) if retained_annotation_counts.size > 0 else 0.0,
        "max": float(np.max(retained_annotation_counts)) if retained_annotation_counts.size > 0 else 0.0,
    }
    full_annotation_count_summary = {
        "mean": float(np.mean(full_annotation_counts)) if full_annotation_counts.size > 0 else 0.0,
        "median": float(np.median(full_annotation_counts)) if full_annotation_counts.size > 0 else 0.0,
        "max": float(np.max(full_annotation_counts)) if full_annotation_counts.size > 0 else 0.0,
    }
    profligate_intercepts = ",".join("%.8g" % float(diag.get("intercept", 0.0)) for diag in profligate_diagnostics)
    profligate_slopes = ",".join("%.8g" % float(diag.get("slope", 0.0)) for diag in profligate_diagnostics)
    profligate_fit_pair_counts = ",".join(str(diag.get("fit_pair_count", 0)) for diag in profligate_diagnostics)
    profligate_fit_applied = ",".join(str(bool(diag.get("fit_applied", profligate_correction == "none"))) for diag in profligate_diagnostics)

    diagnostics = {
        "num_active_genes": int(gene_indices.size),
        "num_retained_gene_sets": int(np.sum(gene_set_mask)),
        "beta_source": beta_source_label,
        "beta_positive_count": int(np.sum(beta_matrix > 0)),
        "beta_negative_count": int(np.sum(beta_matrix < 0)),
        "beta_anchor_count": int(beta_matrix.shape[1]),
        "num_anchor_traits": int(beta_matrix.shape[1]),
        "anchor_aggregation": anchor_aggregation,
        "gene_gene_target_mode": target_mode,
        "gene_gene_pair_weights_used": False,
        "pair_prior": float(pair_prior),
        "pair_logbf_base": str(logbf_base),
        "pair_matrix_floor": float(matrix_floor),
        "pair_excess_probability": bool(excess_probability),
        "diagonal_weight": float(diagonal_weight),
        "gene_gene_profligate_correction": profligate_correction,
        "gene_gene_profligate_correction_feature_names": "intercept,retained_log1p_annotation_count_sum",
        "gene_gene_profligate_correction_fit_pair_counts": profligate_fit_pair_counts,
        "gene_gene_profligate_correction_fit_applied": profligate_fit_applied,
        "gene_gene_profligate_correction_intercepts": profligate_intercepts,
        "gene_gene_profligate_correction_slopes": profligate_slopes,
        "gene_gene_retained_annotation_count_mean": retained_annotation_count_summary["mean"],
        "gene_gene_retained_annotation_count_median": retained_annotation_count_summary["median"],
        "gene_gene_retained_annotation_count_max": retained_annotation_count_summary["max"],
        "gene_gene_full_annotation_count_mean": full_annotation_count_summary["mean"],
        "gene_gene_full_annotation_count_median": full_annotation_count_summary["median"],
        "gene_gene_full_annotation_count_max": full_annotation_count_summary["max"],
        "gene_gene_retained_annotation_count_source_columns": int(X_retained.shape[1]),
        "gene_gene_full_annotation_count_source_columns": int(state.X_orig.shape[1]),
        "per_anchor_nonzero_edge_counts": ",".join(str(value) for value in per_anchor_nonzero_edge_counts),
        "per_anchor_mean_target": ",".join("%.8g" % value for value in per_anchor_mean_target),
        "L_mean": float(np.mean(L)) if L.size > 0 else 0.0,
        "L_std": float(np.std(L)) if L.size > 0 else 0.0,
        "L_min": float(np.min(L)) if L.size > 0 else 0.0,
        "L_max": float(np.max(L)) if L.size > 0 else 0.0,
        "P_mean": float(np.mean(P)) if P.size > 0 else 0.0,
        "P_std": float(np.std(P)) if P.size > 0 else 0.0,
        "P_min": float(np.min(P)) if P.size > 0 else 0.0,
        "P_max": float(np.max(P)) if P.size > 0 else 0.0,
        "M_density": float(np.mean(M > 0)) if M.size > 0 else 0.0,
        "M_mean": float(np.mean(M)) if M.size > 0 else 0.0,
        "M_std": float(np.std(M)) if M.size > 0 else 0.0,
        "M_min": float(np.min(M)) if M.size > 0 else 0.0,
        "M_max": float(np.max(M)) if M.size > 0 else 0.0,
        "M_row_sum_mean": float(np.mean(M_row_sums)) if M_row_sums.size > 0 else 0.0,
        "M_row_sum_median": float(np.median(M_row_sums)) if M_row_sums.size > 0 else 0.0,
        "M_row_sum_q90": float(np.quantile(M_row_sums, 0.9)) if M_row_sums.size > 0 else 0.0,
        "M_row_sum_max": float(np.max(M_row_sums)) if M_row_sums.size > 0 else 0.0,
    }
    if log_fn is not None:
        log_fn(
            "Built gene-gene pair matrix: genes=%d retained_gene_sets=%d beta_source=%s anchor_aggregation=%s target_mode=%s pair_prior=%.6g M_density=%.6g M_mean=%.6g matrix_floor=%.6g diagonal_weight=%.6g"
            % (
                diagnostics["num_active_genes"],
                diagnostics["num_retained_gene_sets"],
                diagnostics["beta_source"],
                diagnostics["anchor_aggregation"],
                diagnostics["gene_gene_target_mode"],
                diagnostics["pair_prior"],
                diagnostics["M_density"],
                diagnostics["M_mean"],
                diagnostics["pair_matrix_floor"],
                diagnostics["diagonal_weight"],
            ),
            info_level,
        )
    return M, diagnostics, None

def _clone_runtime_value(value):
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _clone_runtime_state(source):
    cloned = copy.copy(source)
    new_values = {key: _clone_runtime_value(value) for key, value in source.__dict__.items()}
    if hasattr(cloned, "__dict__"):
        try:
            cloned.__dict__.clear()
            cloned.__dict__.update(new_values)
        except Exception:
            for key in list(getattr(cloned, "__dict__", {}).keys()):
                try:
                    delattr(cloned, key)
                except Exception:
                    pass
            for key, value in new_values.items():
                setattr(cloned, key, value)
    return cloned


def _replace_runtime_state(target, source):
    target.__dict__.clear()
    target.__dict__.update({key: _clone_runtime_value(value) for key, value in source.__dict__.items()})


def _derive_factor_run_seeds(seed, factor_runs):
    if factor_runs <= 1:
        return [seed]
    base_seed = seed if seed is not None else int(np.random.randint(0, np.iinfo(np.uint32).max))
    children = np.random.SeedSequence(base_seed).spawn(factor_runs)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def _run_with_numpy_seed(seed, fn):
    if seed is None:
        return fn()
    old_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return fn()
    finally:
        np.random.set_state(old_state)


def _normalize_factor_columns(matrix):
    if matrix is None or matrix.size == 0:
        return None
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1.0
    return matrix / norms[np.newaxis, :]


def _aggregate_consensus_stack(stacked_values, aggregation):
    if aggregation == "mean":
        return np.mean(stacked_values, axis=0)
    return np.median(stacked_values, axis=0)


def _compute_any_anchor_relevance(factor_anchor_relevance):
    anchor_relevance = np.asarray(factor_anchor_relevance, dtype=float)
    if anchor_relevance.ndim != 2:
        raise ValueError("factor_anchor_relevance must be 2D")
    if anchor_relevance.shape[1] == 0:
        return np.zeros(anchor_relevance.shape[0], dtype=float)
    clipped = np.clip(anchor_relevance, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - clipped, axis=1)


def _matrix_nonfinite_fraction(matrix):
    if matrix is None:
        return 1.0
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data, dtype=float)
    else:
        values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(~np.isfinite(values)))


def _row_max_rank_values(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        values = np.asarray(matrix.max(axis=1).toarray(), dtype=float).ravel()
    else:
        values = np.asarray(matrix, dtype=float)
        if values.ndim > 1:
            values = np.max(values, axis=1)
        else:
            values = np.ravel(values)
    return np.nan_to_num(values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)


def _sanitize_dense_or_sparse_nonnegative_probabilities(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        sanitized = matrix.copy()
        sanitized.data = np.nan_to_num(
            np.asarray(sanitized.data, dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        return sanitized
    return np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)


def _blockwise_block_ranges(num_rows, block_size, *, shuffle_blocks, rng, max_blocks=None):
    if block_size is None or block_size <= 0:
        block_size = num_rows
    num_blocks = int(math.ceil(float(num_rows) / float(max(1, block_size))))
    block_indices = np.arange(num_blocks, dtype=int)
    if shuffle_blocks and num_blocks > 1:
        rng.shuffle(block_indices)
    if max_blocks is not None:
        block_indices = block_indices[: int(max_blocks)]
    ranges = []
    for block_index in block_indices:
        start = int(block_index * block_size)
        stop = int(min(num_rows, start + block_size))
        if start < stop:
            ranges.append((int(block_index), start, stop))
    return ranges


def _dense_block(matrix, start, stop):
    block = matrix[start:stop, :]
    if sparse.issparse(block):
        block = block.toarray()
    return np.asarray(block, dtype=float)


def _dense_probabilities(prob_matrix, start=None, stop=None):
    if prob_matrix is None:
        return None
    block = prob_matrix if start is None else prob_matrix[start:stop, :]
    if sparse.issparse(block):
        block = block.toarray()
    block = np.asarray(block, dtype=float)
    if block.ndim == 1:
        block = block[:, np.newaxis]
    return block



def _summarize_anchor_weight_surface(row_probabilities, column_probabilities, num_rows, num_cols, *, anchor_aggregation):
    row_scale = _compute_anchor_weight_row_scale(
        row_probabilities,
        column_probabilities,
        int(num_rows),
        int(num_cols),
        anchor_aggregation=anchor_aggregation,
    )
    stats = {
        "annotation_anchor_weight_mode": "sum_same_trait_cosupport" if str(anchor_aggregation) == "multi" else "noisy_or_same_trait_cosupport",
        "annotation_anchor_weight_mean": float(np.mean(row_scale)) if row_scale.size > 0 else 0.0,
        "annotation_anchor_weight_row_mean_q90": float(np.quantile(row_scale, 0.9)) if row_scale.size > 0 else 0.0,
    }
    # Exact density/max are cheap for unit tests and small workflows. Avoid forcing a
    # dense diagnostic matrix for very large blockwise runs.
    if int(num_rows) * int(num_cols) <= 10_000_000:
        weights = _compute_anchor_weight_matrix(
            row_probabilities,
            column_probabilities,
            int(num_rows),
            int(num_cols),
            anchor_aggregation=anchor_aggregation,
        )
        stats.update(
            {
                "annotation_anchor_weight_density": float(np.mean(weights > 0.0)) if weights.size > 0 else 0.0,
                "annotation_anchor_weight_max": float(np.max(weights)) if weights.size > 0 else 0.0,
            }
        )
    else:
        stats.update(
            {
                "annotation_anchor_weight_density": "not_materialized_large_matrix",
                "annotation_anchor_weight_max": "not_materialized_large_matrix",
            }
        )
    return stats

def _compute_weight_matrix_for_block(block_probabilities, global_probabilities, num_rows, num_cols, *, anchor_aggregation="multi"):
    return _compute_anchor_weight_matrix(
        block_probabilities,
        global_probabilities,
        num_rows,
        num_cols,
        anchor_aggregation=anchor_aggregation,
    )


def _compute_weighted_discovery_scale_details(V, row_probabilities=None, column_probabilities=None, *, anchor_aggregation="multi", eps=1e-50):
    if sparse.issparse(V):
        matrix = V.tocsr()
    else:
        matrix = np.asarray(V, dtype=float)

    N, M = matrix.shape
    total_entries = float(max(1, N * M))

    row_scale = _compute_anchor_weight_row_scale(
        row_probabilities,
        column_probabilities,
        N,
        M,
        anchor_aggregation=anchor_aggregation,
    )
    row_scale = np.maximum(np.asarray(row_scale, dtype=float).ravel(), eps)

    sqrt_row_scale = np.sqrt(row_scale)
    if sparse.issparse(matrix):
        row_sums = np.asarray(matrix.sum(axis=1), dtype=float).ravel()
        row_sq_sums = np.asarray(matrix.power(2).sum(axis=1), dtype=float).ravel()
    else:
        row_sums = np.sum(matrix, axis=1)
        row_sq_sums = np.sum(matrix ** 2, axis=1)

    raw_mean = float(np.sum(row_sums) / total_entries)
    raw_second_moment = float(np.sum(row_sq_sums) / total_entries)
    weighted_mean = float(np.sum(sqrt_row_scale * row_sums) / total_entries)
    weighted_second_moment = float(np.sum(row_scale * row_sq_sums) / total_entries)
    raw_std = math.sqrt(max(0.0, raw_second_moment - raw_mean ** 2))
    weighted_std = math.sqrt(max(0.0, weighted_second_moment - weighted_mean ** 2))

    return {
        "row_scale": np.asarray(row_scale, dtype=float),
        "raw_mean": float(raw_mean),
        "raw_std": float(raw_std),
        "weighted_mean": float(weighted_mean),
        "weighted_std": float(weighted_std),
        "row_scale_min": float(np.min(row_scale)) if row_scale.size > 0 else 0.0,
        "row_scale_median": float(np.median(row_scale)) if row_scale.size > 0 else 0.0,
        "row_scale_mean": float(np.mean(row_scale)) if row_scale.size > 0 else 0.0,
        "row_scale_max": float(np.max(row_scale)) if row_scale.size > 0 else 0.0,
    }


def _initialize_blockwise_gene_factors(num_factors, num_columns, vmax):
    scale = max(float(vmax), 1e-6)
    return np.random.random((int(num_factors), int(num_columns))) * scale


def _initialize_blockwise_gene_set_factors(num_rows, num_factors, vmax):
    scale = max(float(vmax), 1e-6)
    return np.random.random((int(num_rows), int(num_factors))) * scale


def _fit_blockwise_global_w(
    state,
    matrix,
    *,
    gene_set_prob_vector,
    gene_or_pheno_prob_vector,
    anchor_aggregation="multi",
    max_num_factors,
    max_num_iterations,
    alpha0,
    phi,
    rel_tol,
    min_lambda_threshold,
    block_size,
    epochs,
    shuffle_blocks,
    max_blocks,
    warm_start_state,
    warm_start_enabled,
    cap_genes=False,
    cap_gene_sets=False,
    report_out,
    log_fn,
    info_level,
    pass_metrics_dir=None,
):
    started_at = time.time()
    if not hasattr(state, "last_factorization_blockwise_report"):
        state.last_factorization_blockwise_report = []
    eps = 1e-50
    V = matrix
    if sparse.issparse(V):
        V = V.tocsr()
    else:
        V = np.asarray(V, dtype=float)
    N, M = V.shape
    vmax = float(np.max(V)) if N > 0 and M > 0 else 1.0
    K = int(max_num_factors)
    K0 = int(_DEFAULT_BLOCKWISE_K0)
    a0 = float(alpha0)
    scale_details = _compute_weighted_discovery_scale_details(
        V,
        row_probabilities=gene_set_prob_vector,
        column_probabilities=gene_or_pheno_prob_vector,
        anchor_aggregation=anchor_aggregation,
        eps=eps,
    )
    phi_scaled = (float(scale_details["raw_std"]) ** 2) * float(phi)
    C = (N + M) / 2.0 + a0 + 1.0
    b0 = 3.14 * (a0 - 1.0) * float(scale_details["raw_mean"]) / (2.0 * max(1, K0))
    lambda_bound = b0 / C if C != 0 else 0.0
    lambda_cut = lambda_bound * 1.5

    shared_gene_factors = None
    lambdak = None
    block_rows = {}
    warm_started = False
    if warm_start_enabled and warm_start_state is not None:
        candidate_H = np.asarray(warm_start_state.get("gene_factors"), dtype=float)
        candidate_lambda = np.asarray(warm_start_state.get("lambdak"), dtype=float)
        if candidate_H.shape == (K, M) and candidate_lambda.shape == (K,):
            shared_gene_factors = np.maximum(candidate_H.copy(), eps)
            lambdak = np.maximum(candidate_lambda.copy(), eps)
            warm_started = True
    if shared_gene_factors is None:
        shared_gene_factors = _initialize_blockwise_gene_factors(K, M, vmax)
    if lambdak is None:
        initial_w_sq_sum = np.zeros(K, dtype=float)
        for block_number, start, stop in _blockwise_block_ranges(
            N,
            int(block_size),
            shuffle_blocks=False,
            rng=np.random.RandomState(0),
            max_blocks=None,
        ):
            W_block = _initialize_blockwise_gene_set_factors(stop - start, K, vmax)
            block_rows[block_number] = W_block
            initial_w_sq_sum += np.sum(W_block ** 2, axis=0)
        lambdak = (0.5 * (initial_w_sq_sum + np.sum(shared_gene_factors ** 2, axis=1)) + b0) / C
        lambdak = np.maximum(lambdak, eps)

    rng = np.random.RandomState(int(np.random.randint(0, np.iinfo(np.uint32).max)))
    block_reports = []
    like = None
    evid = None
    error = None
    delambda = 1.0
    total_columns_evaluated = 0
    epoch_error_trace = []

    gene_probabilities = _dense_probabilities(gene_or_pheno_prob_vector)
    gene_set_probabilities = _dense_probabilities(gene_set_prob_vector)

    def _materialize_gene_set_factors(current_block_rows, current_shared_gene_factors):
        ordered_blocks = []
        current_k = int(current_shared_gene_factors.shape[0])
        for block_number, start, stop in _blockwise_block_ranges(
            N,
            int(block_size),
            shuffle_blocks=False,
            rng=np.random.RandomState(0),
            max_blocks=None,
        ):
            W_block = current_block_rows.get(block_number)
            if W_block is None or W_block.shape != (stop - start, current_k):
                W_block = _initialize_blockwise_gene_set_factors(stop - start, current_k, vmax)
            ordered_blocks.append(np.asarray(W_block, dtype=float))
        return np.vstack(ordered_blocks) if len(ordered_blocks) > 0 else np.zeros((N, current_shared_gene_factors.shape[0]), dtype=float)

    def _run_blockwise_pass(current_shared_gene_factors, current_lambdak, current_block_rows, *, shuffle_this_pass, lambda_update_mode="normal", lambda_damping=1.0):
        current_k = int(current_shared_gene_factors.shape[0])
        ranges = _blockwise_block_ranges(
            N,
            int(block_size),
            shuffle_blocks=bool(shuffle_this_pass),
            rng=rng,
            max_blocks=max_blocks,
        )
        numerator_H = np.zeros_like(current_shared_gene_factors, dtype=float)
        denominator_H_linear = np.zeros_like(current_shared_gene_factors, dtype=float)
        w_sq_sum = np.zeros(current_k, dtype=float)
        pass_like = 0.0
        pass_error = 0.0
        pass_columns_evaluated = 0

        for block_number, start, stop in ranges:
            V_block = _dense_block(V, start, stop)
            V_block = np.maximum(V_block, 0.0)
            S_block = _compute_weight_matrix_for_block(
                None if gene_set_probabilities is None else gene_set_probabilities[start:stop, :],
                gene_probabilities,
                stop - start,
                M,
                anchor_aggregation=anchor_aggregation,
            )
            W_block = current_block_rows.get(block_number)
            if W_block is None or W_block.shape != (stop - start, current_k):
                W_block = _initialize_blockwise_gene_set_factors(stop - start, current_k, vmax)
            V_ap_block = W_block @ current_shared_gene_factors + eps
            numerator_W = (V_block * S_block) @ current_shared_gene_factors.T
            denominator_W = (V_ap_block * S_block) @ current_shared_gene_factors.T + phi_scaled * W_block * (1.0 / np.maximum(current_lambdak, eps))[np.newaxis, :] + eps
            W_block *= numerator_W / denominator_W
            W_block = np.maximum(W_block, 0.0)
            if cap_gene_sets:
                W_block = np.clip(W_block, 0.0, 1.0)
            V_ap_block = W_block @ current_shared_gene_factors + eps
            numerator_H += W_block.T @ (V_block * S_block)
            denominator_H_linear += W_block.T @ (V_ap_block * S_block)
            w_sq_sum += np.sum(W_block ** 2, axis=0)
            pass_like += float(np.sum(0.5 * S_block * (V_block - V_ap_block) ** 2))
            pass_error += float(np.sum(S_block * (V_block - V_ap_block) ** 2))
            pass_columns_evaluated += int(stop - start)
            current_block_rows[block_number] = W_block

        denominator_H = denominator_H_linear + phi_scaled * current_shared_gene_factors * (1.0 / np.maximum(current_lambdak, eps))[:, np.newaxis] + eps
        updated_shared_gene_factors = current_shared_gene_factors * (numerator_H / denominator_H)
        updated_shared_gene_factors = np.maximum(updated_shared_gene_factors, 0.0)
        if cap_genes:
            updated_shared_gene_factors = np.clip(updated_shared_gene_factors, 0.0, 1.0)
        lambdak_new = np.maximum((0.5 * (w_sq_sum + np.sum(updated_shared_gene_factors ** 2, axis=1)) + b0) / C, eps)
        if lambda_update_mode == "freeze":
            updated_lambdak = np.asarray(current_lambdak, dtype=float).copy()
        elif lambda_update_mode == "damped":
            damping = float(np.clip(lambda_damping, 0.0, 1.0))
            updated_lambdak = np.maximum((1.0 - damping) * np.asarray(current_lambdak, dtype=float) + damping * lambdak_new, eps)
        else:
            updated_lambdak = lambdak_new
        if lambda_update_mode == "freeze":
            pass_delambda = float("nan")
        else:
            pass_delambda = float(np.max(np.abs(updated_lambdak - current_lambdak) / (np.maximum(current_lambdak, eps)))) if current_lambdak.size > 0 else 0.0
        regularization = phi_scaled * np.sum((0.5 * (w_sq_sum + np.sum(updated_shared_gene_factors ** 2, axis=1)) + b0) / np.maximum(updated_lambdak, eps) + C * np.log(np.maximum(updated_lambdak, eps)))
        active_lambda = updated_lambdak[updated_lambdak >= lambda_cut]
        active_factors = int(np.sum(updated_lambdak >= lambda_cut))
        return (
            updated_shared_gene_factors,
            updated_lambdak,
            current_block_rows,
            {
                "num_blocks": int(len(ranges)),
                "columns_evaluated": int(pass_columns_evaluated),
                "error": float(pass_error),
                "evidence": float(pass_like + regularization),
                "likelihood": float(pass_like),
                "delambda": float(pass_delambda),
                "active_factors": int(active_factors),
                "retained_factors": int(updated_shared_gene_factors.shape[0]),
                "lambda_q10": float(np.quantile(active_lambda if active_lambda.size > 0 else updated_lambdak, 0.1)) if updated_lambdak.size > 0 else 0.0,
                "lambda_median": float(np.quantile(active_lambda if active_lambda.size > 0 else updated_lambdak, 0.5)) if updated_lambdak.size > 0 else 0.0,
                "lambda_q90": float(np.quantile(active_lambda if active_lambda.size > 0 else updated_lambdak, 0.9)) if updated_lambdak.size > 0 else 0.0,
                "lambda_update_mode": str(lambda_update_mode),
                "lambda_damping": float(lambda_damping),
            },
        )

    max_total_passes = int(max(1, max_num_iterations))
    initial_epoch_passes = int(min(max_total_passes, max(1, epochs)))
    total_passes_completed = 0
    collapse_guard_triggered = False
    collapse_guard_stop_pass = None
    lambda_freeze_passes = int(max(0, _DEFAULT_BLOCKWISE_REFINEMENT_LAMBDA_FREEZE_PASSES))
    lambda_damping = float(np.clip(_DEFAULT_BLOCKWISE_REFINEMENT_LAMBDA_DAMPING, 0.0, 1.0))
    min_refinement_passes = int(max(0, _DEFAULT_BLOCKWISE_GLOBAL_REFINEMENT_PASSES))
    min_shrinkage_refinement_passes = int(max(0, _DEFAULT_BLOCKWISE_MIN_SHRINKAGE_REFINEMENT_PASSES))
    min_total_refinement_passes = int(max(min_refinement_passes, lambda_freeze_passes + min_shrinkage_refinement_passes))
    family_merge_applied = False
    family_merge_components = None
    family_merge_retained_factors_before = None
    family_merge_retained_factors_after = None

    def _maybe_write_pass_checkpoint(pass_number, current_shared_gene_factors, current_lambdak, current_block_rows):
        if pass_metrics_dir is None:
            return
        if pass_number < int(_DEFAULT_BLOCKWISE_PASS_CHECKPOINT_START) or pass_number > int(_DEFAULT_BLOCKWISE_PASS_CHECKPOINT_END):
            return
        os.makedirs(pass_metrics_dir, exist_ok=True)
        checkpoint_file = os.path.join(pass_metrics_dir, "pass_%03d.factor_metrics.out.gz" % int(pass_number))
        _write_blockwise_pass_factor_metrics_checkpoint(
            state,
            output_file=checkpoint_file,
            gene_set_factors=_materialize_gene_set_factors(current_block_rows, current_shared_gene_factors),
            gene_factors=current_shared_gene_factors.T,
            lambdak=current_lambdak,
        )

    def _current_active_factor_count(current_lambdak):
        current_lambdak = np.asarray(current_lambdak, dtype=float)
        if current_lambdak.size == 0:
            return 0
        return int(np.sum(current_lambdak >= lambda_cut))

    def _apply_family_merge(current_shared_gene_factors, current_lambdak, current_block_rows):
        gene_set_factors = _materialize_gene_set_factors(current_block_rows, current_shared_gene_factors)
        records = _collect_blockwise_factor_metric_records(
            state,
            gene_set_factors=gene_set_factors,
            gene_factors=current_shared_gene_factors.T,
            lambdak=current_lambdak,
        )
        if not records:
            return (
                current_shared_gene_factors,
                current_lambdak,
                current_block_rows,
                {
                    "applied": False,
                    "components": None,
                    "retained_factors_before": int(current_shared_gene_factors.shape[0]),
                    "retained_factors_after": int(current_shared_gene_factors.shape[0]),
                },
            )
        family_summary = _build_blockwise_family_keep_indices_from_records(records)
        keep_indices = [int(index) for index in family_summary.get("keep_indices", [])]
        if len(keep_indices) == 0 or len(keep_indices) >= int(current_shared_gene_factors.shape[0]):
            return (
                current_shared_gene_factors,
                current_lambdak,
                current_block_rows,
                {
                    "applied": False,
                    "components": len(family_summary.get("components", [])),
                    "retained_factors_before": int(current_shared_gene_factors.shape[0]),
                    "retained_factors_after": int(current_shared_gene_factors.shape[0]),
                },
            )
        merged_shared_gene_factors = np.asarray(current_shared_gene_factors, dtype=float)[keep_indices, :]
        merged_lambdak = np.asarray(current_lambdak, dtype=float)[keep_indices]
        return (
            merged_shared_gene_factors,
            merged_lambdak,
            {},
            {
                "applied": True,
                "components": len(family_summary.get("components", [])),
                "retained_factors_before": int(current_shared_gene_factors.shape[0]),
                "retained_factors_after": int(len(keep_indices)),
            },
        )

    for epoch_index in range(initial_epoch_passes):
        shared_gene_factors, lambdak, block_rows, pass_stats = _run_blockwise_pass(
            shared_gene_factors,
            lambdak,
            block_rows,
            shuffle_this_pass=shuffle_blocks,
        )
        like = float(pass_stats["likelihood"])
        evid = float(pass_stats["evidence"])
        error = float(pass_stats["error"])
        delambda = float(pass_stats["delambda"])
        total_columns_evaluated = int(pass_stats["columns_evaluated"])
        epoch_error_trace.append(error)
        log_fn(
            "Blockwise epoch %d/%d; backend=blockwise_global_w; blocks=%d; rows=%d; err=%.6g; delambda=%.6g; active_factors=%d; warm_start=%s"
            % (
                epoch_index + 1,
                int(initial_epoch_passes),
                int(pass_stats["num_blocks"]),
                int(total_columns_evaluated),
                error,
                delambda,
                int(pass_stats["active_factors"]),
                str(bool(warm_started)),
            ),
            info_level,
        )
        report_stats = dict(pass_stats)
        report_stats["phase"] = "epoch"
        report_stats["epoch"] = int(epoch_index + 1)
        block_reports.append(report_stats)
        total_passes_completed += 1
        _maybe_write_pass_checkpoint(total_passes_completed, shared_gene_factors, lambdak, block_rows)
        if delambda < rel_tol and total_passes_completed >= initial_epoch_passes:
            break

    refinement_passes_completed = 0
    shrinkage_refinement_passes_completed = 0
    previous_shrinkage_delambda = None
    current_active_factors = _current_active_factor_count(lambdak)
    while total_passes_completed < max_total_passes:
        previous_shared_gene_factors = np.asarray(shared_gene_factors, dtype=float).copy()
        previous_lambdak = np.asarray(lambdak, dtype=float).copy()
        previous_active_factors = int(current_active_factors)
        current_lambda_update_mode = "freeze" if refinement_passes_completed < lambda_freeze_passes else "damped"
        current_lambda_damping = 0.0 if current_lambda_update_mode == "freeze" else lambda_damping
        candidate_shared_gene_factors, candidate_lambdak, block_rows, pass_stats = _run_blockwise_pass(
            shared_gene_factors,
            lambdak,
            block_rows,
            shuffle_this_pass=shuffle_blocks,
            lambda_update_mode=current_lambda_update_mode,
            lambda_damping=current_lambda_damping,
        )
        candidate_active_factors = int(pass_stats["active_factors"])
        active_drop_frac = float(max(0.0, previous_active_factors - candidate_active_factors) / max(1, previous_active_factors))
        current_pass_delambda = float(pass_stats["delambda"])
        delambda_spike = 1.0
        if previous_shrinkage_delambda is not None and np.isfinite(current_pass_delambda) and previous_shrinkage_delambda > 0:
            delambda_spike = float(current_pass_delambda / max(previous_shrinkage_delambda, eps))
        guard_triggered_this_pass = (
            current_lambda_update_mode != "freeze" and (
                active_drop_frac >= float(_DEFAULT_BLOCKWISE_REFINEMENT_COLLAPSE_DROP_FRAC)
                or (
                    previous_shrinkage_delambda is not None
                    and active_drop_frac >= 0.1
                    and delambda_spike >= float(_DEFAULT_BLOCKWISE_REFINEMENT_DELAMBDA_SPIKE_MULT)
                )
            )
        )
        if guard_triggered_this_pass:
            collapse_guard_triggered = True
            collapse_guard_stop_pass = int(total_passes_completed + 1)
            shared_gene_factors = previous_shared_gene_factors
            lambdak = previous_lambdak
            current_active_factors = int(previous_active_factors)
            delambda = float(previous_shrinkage_delambda) if previous_shrinkage_delambda is not None else float("nan")
            log_fn(
                "Stopping blockwise refinement at pass %d after guardrail rejected a collapse step; active_factors would have changed %d -> %d (drop_frac=%.6g) with delambda spike %.6g"
                % (
                    int(collapse_guard_stop_pass),
                    int(previous_active_factors),
                    int(candidate_active_factors),
                    float(active_drop_frac),
                    float(delambda_spike),
                ),
                info_level,
            )
            break
        shared_gene_factors = candidate_shared_gene_factors
        lambdak = candidate_lambdak
        current_active_factors = int(candidate_active_factors)
        like = float(pass_stats["likelihood"])
        evid = float(pass_stats["evidence"])
        error = float(pass_stats["error"])
        delambda = float(current_pass_delambda)
        total_columns_evaluated = int(pass_stats["columns_evaluated"])
        epoch_error_trace.append(error)
        merge_summary = None
        if current_lambda_update_mode == "freeze" and not family_merge_applied:
            shared_gene_factors, lambdak, block_rows, merge_summary = _apply_family_merge(
                shared_gene_factors,
                lambdak,
                block_rows,
            )
            family_merge_applied = bool(merge_summary.get("applied", False))
            family_merge_components = merge_summary.get("components")
            family_merge_retained_factors_before = int(merge_summary.get("retained_factors_before", current_active_factors))
            family_merge_retained_factors_after = int(merge_summary.get("retained_factors_after", current_active_factors))
            current_active_factors = _current_active_factor_count(lambdak)
            if family_merge_applied:
                log_fn(
                    "Collapsed blockwise overlap families after frozen refinement pass %d: %d factors -> %d representatives across %d families"
                    % (
                        int(refinement_passes_completed + 1),
                        int(family_merge_retained_factors_before),
                        int(family_merge_retained_factors_after),
                        int(family_merge_components),
                    ),
                    info_level,
                )
        log_fn(
            "Blockwise global refinement %d/%d; backend=blockwise_global_w; blocks=%d; rows=%d; err=%.6g; delambda=%s; active_factors=%d; retained_factors=%d; lambda_mode=%s; lambda_damping=%.6g"
            % (
                refinement_passes_completed + 1,
                int(max(0, max_total_passes - initial_epoch_passes)),
                int(pass_stats["num_blocks"]),
                int(total_columns_evaluated),
                error,
                "nan" if not np.isfinite(delambda) else "%.6g" % float(delambda),
                int(current_active_factors),
                int(shared_gene_factors.shape[0]),
                str(pass_stats.get("lambda_update_mode", "normal")),
                float(pass_stats.get("lambda_damping", 1.0)),
            ),
            info_level,
        )
        report_stats = dict(pass_stats)
        report_stats["phase"] = "refinement"
        report_stats["epoch"] = int(len(block_reports) + 1)
        report_stats["active_factors"] = int(current_active_factors)
        report_stats["retained_factors"] = int(shared_gene_factors.shape[0])
        if merge_summary is not None:
            report_stats["family_merge_applied"] = "1" if family_merge_applied else "0"
            report_stats["family_merge_components"] = "" if family_merge_components is None else str(int(family_merge_components))
            report_stats["retained_factors_after_merge"] = str(int(shared_gene_factors.shape[0]))
        block_reports.append(report_stats)
        total_passes_completed += 1
        refinement_passes_completed += 1
        if current_lambda_update_mode != "freeze" and np.isfinite(delambda):
            shrinkage_refinement_passes_completed += 1
            previous_shrinkage_delambda = float(delambda)
        _maybe_write_pass_checkpoint(total_passes_completed, shared_gene_factors, lambdak, block_rows)
        if (
            refinement_passes_completed >= min_total_refinement_passes
            and shrinkage_refinement_passes_completed >= min_shrinkage_refinement_passes
            and np.isfinite(delambda)
            and delambda < rel_tol
        ):
            break

    ordered_blocks = []
    for block_number, start, stop in _blockwise_block_ranges(N, int(block_size), shuffle_blocks=False, rng=np.random.RandomState(0), max_blocks=None):
        V_block = _dense_block(V, start, stop)
        S_block = _compute_weight_matrix_for_block(
            None if gene_set_probabilities is None else gene_set_probabilities[start:stop, :],
            gene_probabilities,
            stop - start,
            M,
            anchor_aggregation=anchor_aggregation,
        )
        current_k = int(shared_gene_factors.shape[0])
        W_block = _initialize_blockwise_gene_set_factors(stop - start, current_k, vmax)
        V_ap_block = W_block @ shared_gene_factors + eps
        numerator_W = (V_block * S_block) @ shared_gene_factors.T
        denominator_W = (V_ap_block * S_block) @ shared_gene_factors.T + phi_scaled * W_block * (1.0 / np.maximum(lambdak, eps))[np.newaxis, :] + eps
        W_block *= numerator_W / denominator_W
        W_block = np.maximum(W_block, 0.0)
        if cap_gene_sets:
            W_block = np.clip(W_block, 0.0, 1.0)
        block_rows[block_number] = W_block
        ordered_blocks.append(W_block)
    gene_set_factors = np.vstack(ordered_blocks) if len(ordered_blocks) > 0 else np.zeros((N, K), dtype=float)

    final_iterations = int(len(block_reports))
    state.last_factorization_final_delambda = float(delambda)
    state.last_factorization_iterations = final_iterations
    state.last_factorization_converged = bool(delambda < rel_tol)
    state.last_factorization_hit_iteration_cap = bool(final_iterations >= int(max_total_passes) and delambda >= rel_tol)
    state.last_factorization_backend = "blockwise_global_w"
    state.last_factorization_backend_details = {
        "backend": "blockwise_global_w",
        "num_blocks": int(math.ceil(float(N) / float(max(1, block_size)))) if N > 0 else 0,
        "block_size": int(block_size),
        "epochs": int(initial_epoch_passes),
        "max_total_passes": int(max_total_passes),
        "global_refinement_passes": int(refinement_passes_completed),
        "min_refinement_passes": int(min_refinement_passes),
        "min_shrinkage_refinement_passes": int(min_shrinkage_refinement_passes),
        "refinement_lambda_freeze_passes": int(lambda_freeze_passes),
        "refinement_lambda_damping": float(lambda_damping),
        "refinement_collapse_guard_triggered": bool(collapse_guard_triggered),
        "refinement_collapse_guard_stop_pass": None if collapse_guard_stop_pass is None else int(collapse_guard_stop_pass),
        "family_merge_applied": bool(family_merge_applied),
        "family_merge_components": None if family_merge_components is None else int(family_merge_components),
        "family_merge_retained_factors_before": None if family_merge_retained_factors_before is None else int(family_merge_retained_factors_before),
        "family_merge_retained_factors_after": None if family_merge_retained_factors_after is None else int(family_merge_retained_factors_after),
        "columns_evaluated": int(total_columns_evaluated),
        "warm_started": bool(warm_started),
        "lambda_cut": float(lambda_cut),
        "raw_mean_v": float(scale_details["raw_mean"]),
        "raw_std_v": float(scale_details["raw_std"]),
        "weighted_mean_v": float(scale_details["weighted_mean"]),
        "weighted_std_v": float(scale_details["weighted_std"]),
        "row_scale_min": float(scale_details["row_scale_min"]),
        "row_scale_median": float(scale_details["row_scale_median"]),
        "row_scale_mean": float(scale_details["row_scale_mean"]),
        "row_scale_max": float(scale_details["row_scale_max"]),
        "epoch_error_trace": [float(value) for value in epoch_error_trace],
        "wall_time_sec": float(time.time() - started_at),
    }
    state.last_factorization_scale_details = {
        key: value for key, value in scale_details.items() if key != "row_scale"
    }
    state.last_factorization_blockwise_report = block_reports

    if report_out is not None:
        with _open_text_output(report_out) as output_fh:
            output_fh.write("epoch\tphase\tbackend\tnum_blocks\tblock_size\tcolumns_evaluated\terror\tevidence\tlikelihood\tdelambda\tactive_factors\tretained_factors\tlambda_update_mode\tlambda_damping\tfamily_merge_applied\tfamily_merge_components\tretained_factors_after_merge\twarm_started\n")
            for report in block_reports:
                output_fh.write(
                    "%d\t%s\t%s\t%d\t%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%d\t%d\t%s\t%.12g\t%s\t%s\t%s\t%s\n"
                    % (
                        int(report["epoch"]),
                        str(report.get("phase", "epoch")),
                        "blockwise_global_w",
                        int(report["num_blocks"]),
                        int(block_size),
                        int(report["columns_evaluated"]),
                        float(report["error"]),
                        float(report["evidence"]),
                        float(report["likelihood"]),
                        float(report["delambda"]),
                        int(report["active_factors"]),
                        int(report.get("retained_factors", report["active_factors"])),
                        str(report.get("lambda_update_mode", "normal")),
                        float(report.get("lambda_damping", 1.0)),
                        str(report.get("family_merge_applied", "")),
                        str(report.get("family_merge_components", "")),
                        str(report.get("retained_factors_after_merge", "")),
                        "1" if warm_started else "0",
                    )
                )

    warm_start_payload = {
        "gene_factors": np.asarray(shared_gene_factors, dtype=float).copy(),
        "lambdak": np.asarray(lambdak, dtype=float).copy(),
    }
    return {
        "gene_set_factors": gene_set_factors,
        "gene_or_pheno_factors": shared_gene_factors.T,
        "likelihood": like,
        "evidence": evid,
        "lambdak": lambdak,
        "reconstruction_error": error,
        "backend": "blockwise_global_w",
        "backend_details": state.last_factorization_backend_details,
        "warm_start_payload": warm_start_payload,
    }


def _choose_gene_or_pheno_anchor_source(combined_prior_Ys, priors, Y, *, preferred_label=None, log_fn=None, info_level=1):
    candidate_map = {
        "combined_prior_Ys": combined_prior_Ys,
        "Y": Y,
        "priors": priors,
    }
    candidate_order = ["combined_prior_Ys", "Y", "priors"]
    if preferred_label in candidate_map:
        candidate_order = [preferred_label] + [label for label in candidate_order if label != preferred_label]
    candidates = [(label, candidate_map[label]) for label in candidate_order]
    first_available_label = None
    fallback_choice = None
    for label, matrix in candidates:
        if matrix is None:
            continue
        if first_available_label is None:
            first_available_label = label
        nonfinite_fraction = _matrix_nonfinite_fraction(matrix)
        if nonfinite_fraction < 1.0:
            if first_available_label != label and log_fn is not None:
                log_fn(
                    "Using %s for implicit factor-anchor relevance because earlier source %s was entirely non-finite"
                    % (label, first_available_label),
                    info_level,
                )
            elif nonfinite_fraction > 0.0 and log_fn is not None:
                log_fn(
                    "Using %s for implicit factor-anchor relevance after sanitizing non-finite values (non-finite fraction %.4g)"
                    % (label, nonfinite_fraction),
                    info_level,
                )
            return (matrix, label)
        if fallback_choice is None or nonfinite_fraction < fallback_choice[2]:
            fallback_choice = (matrix, label, nonfinite_fraction)
    if fallback_choice is not None:
        if log_fn is not None:
            log_fn(
                "All implicit factor-anchor sources contained non-finite values; using %s after sanitization (non-finite fraction %.4g)"
                % (fallback_choice[1], fallback_choice[2]),
                info_level,
            )
        return (fallback_choice[0], fallback_choice[1])
    return (None, None)


def _project_pheno_capture_matrix(state, basis, feature_by_pheno, *, basis_name):
    capture_weights, capture_strength = eaggl_phenotype_annotation.project_phenotype_capture(
        state._nnls_project_matrix,
        basis,
        feature_by_pheno,
        max_sum=1.0,
    )
    state.pheno_capture_strength = capture_strength
    state.pheno_capture_basis = basis_name
    return capture_weights


def _prepare_pheno_capture_input_matrix(feature_by_pheno, mode):
    return eaggl_phenotype_annotation.prepare_thresholded_profile_input(feature_by_pheno, mode)


def _align_projection_inputs_to_mask(basis, feature_by_target, mask):
    if mask is None:
        return basis, feature_by_target
    mask = np.asarray(mask, dtype=bool)
    mask_sum = int(np.sum(mask))

    if basis.shape[0] == mask.shape[0]:
        basis = basis[mask, :]
    elif basis.shape[0] != mask_sum:
        raise ValueError(
            "Projection basis rows %s do not match mask length %s or kept count %s"
            % (basis.shape[0], mask.shape[0], mask_sum)
        )

    if feature_by_target.shape[0] == mask.shape[0]:
        feature_by_target = feature_by_target[mask, :]
    elif feature_by_target.shape[0] != mask_sum:
        raise ValueError(
            "Projection target rows %s do not match mask length %s or kept count %s"
            % (feature_by_target.shape[0], mask.shape[0], mask_sum)
        )

    return basis, feature_by_target


def _resolve_trait_linkage_inputs(
    state,
    *,
    use_gene_set_basis,
    trait_linkage_source,
    trait_linkage_threshold,
    trait_linkage_computation_mode,
    pheno_capture_input,
    log_fn,
    info_level,
):
    if use_gene_set_basis:
        basis = state.exp_gene_set_factors
        basis_mask = state.gene_set_factor_gene_set_mask
        if basis is None or state.phenos is None or state.X_phewas_beta_uncorrected is None:
            return None
        combined = state.X_phewas_beta_uncorrected.T
        log_bf = state.X_phewas_beta.T if state.X_phewas_beta is not None else None
        prior = None
        basis_name = "gene_sets"
        context_label = "gene-set trait linkage"
    else:
        basis = state.exp_gene_factors
        basis_mask = state.gene_factor_gene_mask
        if basis is None or state.phenos is None:
            return None
        combined = state.gene_pheno_combined_prior_Ys
        log_bf = state.gene_pheno_Y
        prior = state.gene_pheno_priors
        basis_name = "genes"
        context_label = "gene trait linkage"

    feature_by_trait, source_name = eaggl_trait_linkage.resolve_trait_linkage_source(
        trait_linkage_source,
        combined=combined,
        log_bf=log_bf,
        prior=prior,
        log_fn=log_fn,
        info_level=info_level,
        context_label=context_label,
    )
    if feature_by_trait is None:
        return None

    feature_by_trait_full = feature_by_trait
    return {
        "basis": basis,
        "basis_mask": basis_mask,
        "basis_name": basis_name,
        "feature_by_trait": feature_by_trait,
        "full_feature_by_trait": feature_by_trait_full,
        "score_source": source_name,
        "pheno_capture_input": pheno_capture_input,
        "trait_linkage_threshold": trait_linkage_threshold,
        "trait_linkage_computation_mode": trait_linkage_computation_mode,
    }


def _apply_canonical_trait_linkage(
    state,
    *,
    use_gene_set_basis,
    trait_linkage_source,
    trait_linkage_threshold,
    trait_linkage_computation_mode,
    pheno_capture_input,
    update_anchor_relevance,
    log_fn,
    info_level,
):
    linkage_inputs = _resolve_trait_linkage_inputs(
        state,
        use_gene_set_basis=use_gene_set_basis,
        trait_linkage_source=trait_linkage_source,
        trait_linkage_threshold=trait_linkage_threshold,
        trait_linkage_computation_mode=trait_linkage_computation_mode,
        pheno_capture_input=pheno_capture_input,
        log_fn=log_fn,
        info_level=info_level,
    )
    if linkage_inputs is None:
        return False

    linkage_result = eaggl_trait_linkage.compute_trait_linkage(
        state._nnls_project_matrix,
        linkage_inputs["basis"],
        linkage_inputs["feature_by_trait"],
        full_feature_by_trait=linkage_inputs["full_feature_by_trait"],
        basis_mask=linkage_inputs["basis_mask"],
        threshold_mode=pheno_capture_input,
        threshold_value=trait_linkage_threshold,
        strict_threshold=True,
        computation_mode=trait_linkage_computation_mode,
    )
    if linkage_result is None:
        return False

    state.exp_pheno_factors = linkage_result["joint"]
    state.pheno_capture_strength = linkage_result["trait_total_support"]
    state.pheno_capture_basis = linkage_inputs["basis_name"]
    state.pheno_capture_input = pheno_capture_input
    state.trait_linkage_factor_total_mass = linkage_result["factor_total_mass"]
    state.trait_linkage_factor_n_eff = linkage_result["factor_n_eff"]
    state.trait_linkage_factor_top_share = linkage_result["factor_top_share"]
    state.trait_linkage_factor_top10_share = linkage_result["factor_top10_share"]
    state.trait_linkage_broad_factor_flag = linkage_result["broad_factor_flag"]
    state.trait_linkage_joint = linkage_result["joint"]
    state.trait_linkage_marginal = linkage_result["marginal"]
    state.trait_linkage_marginal_overlap = linkage_result["marginal_overlap"]
    state.trait_linkage_residual = linkage_result["residual"]
    state.trait_linkage_strength = linkage_result["trait_total_support"]
    state.trait_linkage_retained_strength = linkage_result["retained_trait_support"]
    state.trait_linkage_retained_fraction = linkage_result["retained_fraction"]
    state.trait_linkage_total_feature_count = linkage_result["total_feature_count"]
    state.trait_linkage_retained_feature_count = linkage_result["retained_feature_count"]
    state.trait_linkage_n_eff = linkage_result["trait_n_eff"]
    state.trait_linkage_retained_n_eff = linkage_result["retained_n_eff"]
    state.trait_linkage_low_retention_flag = linkage_result["low_retention_flag"]
    state.trait_linkage_basis = linkage_inputs["basis_name"]
    state.trait_linkage_score_source = linkage_inputs["score_source"]
    if state.anchor_pheno_mask is not None:
        state.trait_linkage_is_anchor = np.asarray(state.anchor_pheno_mask, dtype=bool)
    else:
        state.trait_linkage_is_anchor = np.full(len(state.phenos), False, dtype=bool)

    if update_anchor_relevance and state.anchor_pheno_mask is not None:
        anchor_mask = np.asarray(state.anchor_pheno_mask, dtype=bool)
        if np.any(anchor_mask):
            anchor_joint = linkage_result["joint"][anchor_mask, :].T
            anchor_marginal = linkage_result["marginal"][anchor_mask, :].T
            state.factor_anchor_relevance = anchor_joint
            state.factor_anchor_marginal_relevance = anchor_marginal
            state.factor_relevance = _compute_any_anchor_relevance(anchor_joint)
            state.factor_marginal_relevance = _compute_any_anchor_relevance(anchor_marginal)
    return True


def project_phenos_from_loaded_factors(
    state,
    *,
    project_phenos_from_gene_sets=False,
    trait_linkage_source="combined",
    trait_linkage_threshold=1.0,
    trait_linkage_computation_mode="sparse_full",
    pheno_capture_input="weighted_thresholded",
    bail_fn,
    log_fn,
    info_level,
):
    """Compute canonical trait linkage from precomputed gene or gene-set factors."""
    if project_phenos_from_gene_sets:
        if state.exp_gene_set_factors is None:
            bail_fn("Projection-only gene-set trait linkage requires --factor-gene-set-clusters-in")
        if state.X_phewas_beta_uncorrected is None:
            bail_fn("Projection-only gene-set trait linkage requires --gene-set-phewas-stats-in")
    else:
        if state.exp_gene_factors is None:
            bail_fn("Projection-only trait linkage requires gene factor loadings from --factor-gene-clusters-in")
        if state.gene_pheno_combined_prior_Ys is None and state.gene_pheno_Y is None and state.gene_pheno_priors is None:
            bail_fn(
                "Projection-only trait linkage requires --gene-phewas-stats-in with combined, log_bf, or prior values"
            )
    if state.phenos is None or len(state.phenos) == 0:
        bail_fn("Projection-only trait linkage requires phenotype names from the corresponding PheWAS input")

    applied = _apply_canonical_trait_linkage(
        state,
        use_gene_set_basis=project_phenos_from_gene_sets,
        trait_linkage_source=trait_linkage_source,
        trait_linkage_threshold=trait_linkage_threshold,
        trait_linkage_computation_mode=trait_linkage_computation_mode,
        pheno_capture_input=pheno_capture_input,
        update_anchor_relevance=True,
        log_fn=log_fn,
        info_level=info_level,
    )
    if not applied:
        bail_fn("Projection-only trait linkage could not resolve a usable support surface")

    state.pheno_factor_pheno_mask = np.full(len(state.phenos), False, dtype=bool)
    state._record_params(
        {
            "factor_projection_only_pheno_clusters": True,
            "factor_projection_only_pheno_capture_basis": state.trait_linkage_basis,
            "factor_projection_only_pheno_capture_input": pheno_capture_input,
            "factor_projection_only_pheno_source": state.trait_linkage_score_source,
            "factor_projection_only_pheno_threshold": trait_linkage_threshold,
            "factor_projection_only_pheno_computation_mode": trait_linkage_computation_mode,
            "factor_projection_only_num_phenos": len(state.phenos),
        },
        overwrite=True,
    )
    log_fn(
        "Computed canonical trait linkage for %d phenotypes onto %d precomputed %s factors using %s"
        % (
            len(state.phenos),
            state.exp_pheno_factors.shape[1],
            state.trait_linkage_basis,
            state.trait_linkage_score_source,
        ),
        info_level,
    )
    return state.exp_pheno_factors


def project_phenos_from_loaded_gene_factors(
    state,
    *,
    trait_linkage_source="combined",
    trait_linkage_threshold=1.0,
    trait_linkage_computation_mode="sparse_full",
    pheno_capture_input="weighted_thresholded",
    bail_fn,
    log_fn,
    info_level,
):
    """Compatibility wrapper for gene-basis projection-only pheno clusters."""
    return project_phenos_from_loaded_factors(
        state,
        project_phenos_from_gene_sets=False,
        trait_linkage_source=trait_linkage_source,
        trait_linkage_threshold=trait_linkage_threshold,
        trait_linkage_computation_mode=trait_linkage_computation_mode,
        pheno_capture_input=pheno_capture_input,
        bail_fn=bail_fn,
        log_fn=log_fn,
        info_level=info_level,
    )


def _open_text_output(path):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _open_text_append(path):
    if path.endswith(".gz"):
        return gzip.open(path, "at", encoding="utf-8")
    return open(path, "a", encoding="utf-8")


def _open_text_input(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _remap_tsv_row(fields, source_header, target_header):
    mapped = {name: fields[index] if index < len(fields) else "" for index, name in enumerate(source_header)}
    return [mapped.get(name, "") for name in target_header]


def _rewrite_phi_prefixed_table_with_union(output_path, temp_path, *, phi, existing_header, current_header):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    fd, rewrite_path = tempfile.mkstemp(prefix=".phi_candidate_rewrite_", suffix=".tsv.gz" if output_path.endswith(".gz") else ".tsv", dir=output_dir)
    os.close(fd)
    union_payload_header = list(existing_header[1:])
    for column in current_header:
        if column not in union_payload_header:
            union_payload_header.append(column)
    union_header = ["phi"] + union_payload_header
    try:
        with _open_text_output(rewrite_path) as output_fh:
            output_fh.write("%s\n" % "\t".join(union_header))
            with _open_text_input(output_path) as existing_fh:
                next(existing_fh, None)
                for line in existing_fh:
                    fields = line.rstrip("\n").split("\t")
                    output_fh.write("%s\n" % "\t".join(_remap_tsv_row(fields, existing_header, union_header)))
            with open(temp_path, "r", encoding="utf-8") as current_fh:
                next(current_fh, None)
                for line in current_fh:
                    fields = line.rstrip("\n").split("\t")
                    output_fh.write(
                        "%s\n"
                        % "\t".join(
                            _remap_tsv_row(
                                ["%.12g" % float(phi)] + fields,
                                ["phi"] + current_header,
                                union_header,
                            )
                        )
                    )
        os.replace(rewrite_path, output_path)
    finally:
        if os.path.exists(rewrite_path):
            try:
                os.remove(rewrite_path)
            except OSError:
                pass


def _append_phi_prefixed_table(output_path, *, phi, write_fn):
    if output_path is None:
        return
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".phi_candidate_", suffix=".tsv", dir=output_dir)
    os.close(fd)
    try:
        write_fn(temp_path)
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return
        with open(temp_path, "r", encoding="utf-8") as input_fh:
            current_header_line = input_fh.readline().rstrip("\n")
        if current_header_line == "":
            return
        current_header = current_header_line.split("\t")
        output_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        if not output_exists:
            with _open_text_output(output_path) as output_fh, open(temp_path, "r", encoding="utf-8") as input_fh:
                output_fh.write("phi\t%s\n" % current_header_line)
                next(input_fh, None)
                for line in input_fh:
                    output_fh.write("%.12g\t%s" % (float(phi), line))
            return

        with _open_text_input(output_path) as existing_fh:
            existing_header_line = existing_fh.readline().rstrip("\n")
        existing_header = existing_header_line.split("\t")
        existing_payload_header = existing_header[1:]
        if any(column not in existing_payload_header for column in current_header):
            _rewrite_phi_prefixed_table_with_union(
                output_path,
                temp_path,
                phi=phi,
                existing_header=existing_header,
                current_header=current_header,
            )
            return

        with _open_text_append(output_path) as output_fh, open(temp_path, "r", encoding="utf-8") as input_fh:
            next(input_fh, None)
            for line in input_fh:
                fields = line.rstrip("\n").split("\t")
                row = ["%.12g" % float(phi)] + _remap_tsv_row(fields, current_header, existing_payload_header)
                output_fh.write("%s\n" % "\t".join(row))
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _reset_phi_prefixed_outputs(output_paths):
    for output_path in output_paths:
        if output_path is None:
            continue
        if os.path.exists(output_path):
            os.remove(output_path)


def _derive_blockwise_pass_metrics_dir(factor_metrics_out, blockwise_report_out):
    candidate = factor_metrics_out if factor_metrics_out is not None else blockwise_report_out
    if candidate is None:
        return None
    if candidate.endswith(".gz"):
        candidate = candidate[:-3]
    stem, ext = os.path.splitext(candidate)
    if ext:
        return f"{stem}.blockwise_pass_metrics"
    return f"{candidate}.blockwise_pass_metrics"


def _write_blockwise_pass_factor_metrics_checkpoint(
    state,
    *,
    output_file,
    gene_set_factors,
    gene_factors,
    lambdak,
):
    if output_file is None or not hasattr(state, "write_factor_metrics"):
        return
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    saved_gene_set_factors = getattr(state, "exp_gene_set_factors", None)
    saved_gene_factors = getattr(state, "exp_gene_factors", None)
    saved_lambdak = getattr(state, "exp_lambdak", None)
    try:
        state.exp_gene_set_factors = np.asarray(gene_set_factors, dtype=float)
        state.exp_gene_factors = np.asarray(gene_factors, dtype=float)
        state.exp_lambdak = np.asarray(lambdak, dtype=float)
        state.write_factor_metrics(output_file)
    finally:
        state.exp_gene_set_factors = saved_gene_set_factors
        state.exp_gene_factors = saved_gene_factors
        state.exp_lambdak = saved_lambdak


def _collect_blockwise_factor_metric_records(
    state,
    *,
    gene_set_factors,
    gene_factors,
    lambdak,
):
    if not hasattr(state, "_collect_factor_metrics_records"):
        return None
    saved_gene_set_factors = getattr(state, "exp_gene_set_factors", None)
    saved_gene_factors = getattr(state, "exp_gene_factors", None)
    saved_lambdak = getattr(state, "exp_lambdak", None)
    try:
        state.exp_gene_set_factors = np.asarray(gene_set_factors, dtype=float)
        state.exp_gene_factors = np.asarray(gene_factors, dtype=float)
        state.exp_lambdak = np.asarray(lambdak, dtype=float)
        return state._collect_factor_metrics_records()
    finally:
        state.exp_gene_set_factors = saved_gene_set_factors
        state.exp_gene_factors = saved_gene_factors
        state.exp_lambdak = saved_lambdak


def _summarize_mask(mask):
    if mask is None:
        return (False, None, None)
    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return (True, 0, 0)
    try:
        selected = int(np.sum(mask_array.astype(bool)))
    except Exception:
        selected = int(np.sum(mask_array != 0))
    return (True, int(mask_array.size), selected)


def _build_factor_param_record(
    *,
    max_num_factors,
    phi,
    alpha0,
    beta0,
    discovery_model,
    anchor_aggregation,
    gene_gene_beta_source,
    gene_gene_pair_prior,
    gene_gene_pair_prior_effective_size,
    gene_gene_logbf_base,
    gene_gene_diagonal_weight,
    gene_gene_matrix_floor,
    gene_gene_excess_probability,
    gene_gene_row_sum_cap,
    gene_gene_sparsity,
    gene_gene_profligate_correction,
    seed,
    factor_runs,
    consensus_nmf,
    consensus_min_factor_cosine,
    consensus_min_run_support,
    consensus_aggregation,
    consensus_stats_out,
    learn_phi,
    learn_phi_max_redundancy,
    learn_phi_max_redundancy_q90,
    learn_phi_runs_per_step,
    learn_phi_min_run_support,
    learn_phi_min_stability,
    learn_phi_fit_loss_warning_frac,
    learn_phi_max_severe_fit_loss_frac,
    learn_phi_target_gene_mass,
    learn_phi_target_gene_effective_support,
    learn_phi_size_tolerance_frac,
    learn_phi_min_primary_factors,
    learn_phi_max_primary_gene_max_weight_q90,
    learn_phi_max_steps,
    learn_phi_values,
    learn_phi_expand_factor,
    learn_phi_weight_floor,
    learn_phi_metric_factor_scope,
    learn_phi_mass_floor_frac,
    learn_phi_only,
    learn_phi_report_out,
    factor_phi_metrics_out,
    factor_phi_factors_out,
    factor_phi_gene_set_clusters_out,
    factor_phi_gene_clusters_out,
    cluster_row_min_max_loading,
    factor_output_scope,
    factor_backend,
    learn_phi_backend,
    blockwise_gene_set_block_size,
    blockwise_epochs,
    blockwise_shuffle_blocks,
    blockwise_warm_start,
    blockwise_max_blocks,
    blockwise_report_out,
    learn_phi_prune_genes_num,
    learn_phi_prune_gene_sets_num,
    learn_phi_max_num_iterations,
    gene_set_filter_type,
    gene_set_filter_value,
    gene_or_pheno_filter_type,
    gene_or_pheno_filter_value,
    pheno_prune_value,
    pheno_prune_number,
    gene_prune_value,
    gene_prune_number,
    gene_set_prune_value,
    gene_set_prune_number,
    max_num_discovery_gene_sets,
    auto_discovery_subset,
    discovery_redundancy_weighting,
    discovery_redundancy_weighting_mode,
    discovery_similarity_threshold,
    anchor_pheno_mask,
    anchor_gene_mask,
    anchor_any_pheno,
    anchor_any_gene,
    anchor_gene_set,
    run_transpose,
    max_num_iterations,
    rel_tol,
    min_lambda_threshold,
    lmm_auth_key,
    lmm_model,
    lmm_provider,
    label_gene_sets_only,
    label_include_phenos,
    label_individually,
    gene_sets_for_labeling,
    keep_original_loadings,
    project_phenos_from_gene_sets,
    pheno_capture_input,
    trait_linkage_source,
    trait_linkage_threshold,
    trait_linkage_computation_mode,
    no_trait_linkage,
):
    anchor_gene_mask_present, anchor_gene_mask_total, anchor_gene_mask_selected = _summarize_mask(anchor_gene_mask)
    anchor_pheno_mask_present, anchor_pheno_mask_total, anchor_pheno_mask_selected = _summarize_mask(anchor_pheno_mask)
    return {
        "max_num_factors": int(max_num_factors),
        "phi": float(phi),
        "alpha0": float(alpha0),
        "beta0": float(beta0),
        "discovery_model": str(discovery_model),
        "gene_gene_beta_source": str(gene_gene_beta_source),
        "gene_gene_pair_prior": None if gene_gene_pair_prior is None else float(gene_gene_pair_prior),
        "gene_gene_pair_prior_effective_size": None if gene_gene_pair_prior_effective_size is None else float(gene_gene_pair_prior_effective_size),
        "gene_gene_logbf_base": str(gene_gene_logbf_base),
        "anchor_aggregation": str(anchor_aggregation),
        "gene_gene_diagonal_weight": float(gene_gene_diagonal_weight),
        "gene_gene_matrix_floor": float(gene_gene_matrix_floor),
        "gene_gene_excess_probability": bool(gene_gene_excess_probability),
        "gene_gene_row_sum_cap": bool(gene_gene_row_sum_cap),
        "gene_gene_sparsity": float(gene_gene_sparsity),
        "gene_gene_profligate_correction": str(gene_gene_profligate_correction),
        "seed": int(seed) if seed is not None else None,
        "factor_runs": int(factor_runs),
        "consensus_nmf": bool(consensus_nmf),
        "consensus_min_factor_cosine": float(consensus_min_factor_cosine),
        "consensus_min_run_support": float(consensus_min_run_support),
        "consensus_aggregation": consensus_aggregation,
        "consensus_stats_out": consensus_stats_out,
        "learn_phi": bool(learn_phi),
        "learn_phi_max_redundancy": float(learn_phi_max_redundancy),
        "learn_phi_max_redundancy_q90": float(learn_phi_max_redundancy_q90),
        "learn_phi_runs_per_step": int(learn_phi_runs_per_step),
        "learn_phi_min_run_support": float(learn_phi_min_run_support),
        "learn_phi_min_stability": float(learn_phi_min_stability),
        "learn_phi_fit_loss_warning_frac": float(learn_phi_fit_loss_warning_frac),
        "learn_phi_max_severe_fit_loss_frac": float(learn_phi_max_severe_fit_loss_frac),
        "learn_phi_target_gene_mass": None if learn_phi_target_gene_mass is None else float(learn_phi_target_gene_mass),
        "learn_phi_target_gene_effective_support": None if learn_phi_target_gene_effective_support is None else float(learn_phi_target_gene_effective_support),
        "learn_phi_size_tolerance_frac": float(learn_phi_size_tolerance_frac),
        "learn_phi_min_primary_factors": int(learn_phi_min_primary_factors),
        "learn_phi_max_primary_gene_max_weight_q90": None if learn_phi_max_primary_gene_max_weight_q90 is None else float(learn_phi_max_primary_gene_max_weight_q90),
        "learn_phi_max_steps": int(learn_phi_max_steps),
        "learn_phi_values": learn_phi_values,
        "learn_phi_expand_factor": float(learn_phi_expand_factor),
        "learn_phi_weight_floor": None if learn_phi_weight_floor is None else float(learn_phi_weight_floor),
        "learn_phi_metric_factor_scope": str(learn_phi_metric_factor_scope),
        "learn_phi_mass_floor_frac": float(learn_phi_mass_floor_frac),
        "learn_phi_only": bool(learn_phi_only),
        "learn_phi_report_out": learn_phi_report_out,
        "factor_phi_metrics_out": factor_phi_metrics_out,
        "factor_phi_factors_out": factor_phi_factors_out,
        "factor_phi_gene_set_clusters_out": factor_phi_gene_set_clusters_out,
        "factor_phi_gene_clusters_out": factor_phi_gene_clusters_out,
        "cluster_row_min_max_loading": float(cluster_row_min_max_loading),
        "factor_output_scope": str(factor_output_scope),
        "factor_primary_mass_floor": float(_PRIMARY_FACTOR_MASS_FLOOR),
        "factor_secondary_mass_floor": float(_SECONDARY_FACTOR_MASS_FLOOR),
        "factor_backend": factor_backend,
        "learn_phi_backend": learn_phi_backend,
        "blockwise_gene_set_block_size": int(blockwise_gene_set_block_size),
        "blockwise_epochs": int(blockwise_epochs),
        "blockwise_shuffle_blocks": bool(blockwise_shuffle_blocks),
        "blockwise_warm_start": bool(blockwise_warm_start),
        "blockwise_max_blocks": None if blockwise_max_blocks is None else int(blockwise_max_blocks),
        "blockwise_report_out": blockwise_report_out,
        "learn_phi_prune_genes_num": None if learn_phi_prune_genes_num is None else int(learn_phi_prune_genes_num),
        "learn_phi_prune_gene_sets_num": None if learn_phi_prune_gene_sets_num is None else int(learn_phi_prune_gene_sets_num),
        "learn_phi_max_num_iterations": None if learn_phi_max_num_iterations is None else int(learn_phi_max_num_iterations),
        "learn_phi_redundancy_basis_target": "gene",
        "gene_set_filter_value": gene_set_filter_value,
        "gene_or_pheno_filter_value": gene_or_pheno_filter_value,
        "pheno_prune_value": pheno_prune_value,
        "pheno_prune_number": pheno_prune_number,
        "gene_prune_value": gene_prune_value,
        "gene_prune_number": gene_prune_number,
        "gene_set_prune_value": gene_set_prune_value,
        "gene_set_prune_number": gene_set_prune_number,
        "max_num_discovery_gene_sets": None if max_num_discovery_gene_sets is None else int(max_num_discovery_gene_sets),
        "auto_discovery_subset": bool(auto_discovery_subset),
        "discovery_redundancy_weighting": bool(discovery_redundancy_weighting),
        "discovery_redundancy_weighting_mode": str(discovery_redundancy_weighting_mode),
        "discovery_similarity_threshold": float(discovery_similarity_threshold),
        "anchor_any_pheno": bool(anchor_any_pheno),
        "anchor_any_gene": bool(anchor_any_gene),
        "anchor_gene_set": bool(anchor_gene_set),
        "anchor_gene_mask_present": bool(anchor_gene_mask_present),
        "anchor_gene_mask_total": anchor_gene_mask_total,
        "anchor_gene_mask_selected": anchor_gene_mask_selected,
        "anchor_pheno_mask_present": bool(anchor_pheno_mask_present),
        "anchor_pheno_mask_total": anchor_pheno_mask_total,
        "anchor_pheno_mask_selected": anchor_pheno_mask_selected,
        "run_transpose": bool(run_transpose),
        "max_num_iterations": int(max_num_iterations),
        "rel_tol": float(rel_tol),
        "min_lambda_threshold": float(min_lambda_threshold),
        "lmm_auth_key_present": lmm_auth_key is not None,
        "lmm_model": lmm_model,
        "lmm_provider": lmm_provider,
        "label_gene_sets_only": bool(label_gene_sets_only),
        "label_include_phenos": bool(label_include_phenos),
        "label_individually": bool(label_individually),
        "gene_sets_for_labeling_num_ids": 0 if gene_sets_for_labeling is None else int(len(gene_sets_for_labeling)),
        "keep_original_loadings": bool(keep_original_loadings),
        "project_phenos_from_gene_sets": bool(project_phenos_from_gene_sets),
        "pheno_capture_input": pheno_capture_input,
        "trait_linkage_source": trait_linkage_source,
        "trait_linkage_threshold": float(trait_linkage_threshold),
        "trait_linkage_computation_mode": trait_linkage_computation_mode,
        "no_trait_linkage": bool(no_trait_linkage),
    }


def _extract_canonical_factor_matrix(state):
    if _discovery_model_prefers_gene_basis(state):
        if state.exp_gene_factors is not None and state.exp_gene_factors.size > 0:
            return state.exp_gene_factors
        if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0:
            return state.exp_gene_set_factors
    if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0:
        return state.exp_gene_set_factors
    if state.exp_gene_factors is not None and state.exp_gene_factors.size > 0:
        return state.exp_gene_factors
    if state.exp_pheno_factors is not None and state.exp_pheno_factors.size > 0:
        return state.exp_pheno_factors
    return None


def _extract_overlap_basis_matrix(state):
    if _discovery_model_prefers_gene_basis(state):
        if state.exp_gene_factors is not None and state.exp_gene_factors.size > 0:
            return ("gene", state.exp_gene_factors)
        if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0:
            return ("gene_set", state.exp_gene_set_factors)
    if state.exp_gene_factors is not None and state.exp_gene_factors.size > 0:
        return ("gene", state.exp_gene_factors)
    if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0:
        return ("gene_set", state.exp_gene_set_factors)
    if state.exp_pheno_factors is not None and state.exp_pheno_factors.size > 0:
        return ("pheno", state.exp_pheno_factors)
    return ("none", None)


def _resolve_factor_mass_floors(primary_mass_floor):
    primary_mass_floor = float(primary_mass_floor)
    secondary_mass_floor = min(float(_SECONDARY_FACTOR_MASS_FLOOR), primary_mass_floor)
    return primary_mass_floor, secondary_mass_floor


def _metric_factor_indices_from_mass_profile(mass_profile, metric_factor_scope, *, primary_mass_floor=None):
    mass_fractions = np.asarray(mass_profile.get("mass_fractions", []), dtype=float)
    if str(metric_factor_scope) == "all":
        return np.arange(mass_fractions.size, dtype=int)
    if str(metric_factor_scope) != "primary":
        raise ValueError("metric_factor_scope must be one of: primary, all")
    if primary_mass_floor is None:
        primary_mass_floor = mass_profile.get("primary_mass_floor", _PRIMARY_FACTOR_MASS_FLOOR)
    return np.where(mass_fractions >= float(primary_mass_floor))[0].astype(int)


def _select_factor_columns(matrix, factor_indices):
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    factor_indices = np.asarray(factor_indices, dtype=int)
    if matrix.ndim != 2:
        return matrix
    if factor_indices.size == 0:
        return matrix[:, :0]
    valid_indices = factor_indices[(factor_indices >= 0) & (factor_indices < matrix.shape[1])]
    return matrix[:, valid_indices]


def _prepare_factor_vector_for_overlap(vector, weight_floor):
    clipped = np.clip(np.asarray(vector, dtype=float), 0.0, None)
    if weight_floor is not None and weight_floor > 0:
        clipped[clipped < weight_floor] = 0.0
    total = float(np.sum(clipped))
    if total <= 0:
        return clipped
    return clipped / total


def _weighted_jaccard_similarity(u, v, weight_floor):
    u_prepared = _prepare_factor_vector_for_overlap(u, weight_floor)
    v_prepared = _prepare_factor_vector_for_overlap(v, weight_floor)
    if float(np.sum(u_prepared)) <= 0 and float(np.sum(v_prepared)) <= 0:
        return 0.0
    denominator = float(np.sum(np.maximum(u_prepared, v_prepared)))
    if denominator <= 0:
        return 0.0
    return float(np.sum(np.minimum(u_prepared, v_prepared)) / denominator)


def _gene_set_sort_rank_for_pruning(state, *, betas, betas_uncorrected):
    if state.X_phewas_beta_uncorrected is not None:
        return -np.asarray(state.X_phewas_beta_uncorrected.mean(axis=0)).ravel()
    source = betas_uncorrected if betas_uncorrected is not None else betas
    if source is None:
        return np.arange(state.X_orig.shape[1], dtype=float)
    if sparse.issparse(source):
        return -np.asarray(source.mean(axis=1)).ravel()
    source_array = np.asarray(source, dtype=float)
    if source_array.ndim == 1:
        return -source_array
    if source_array.shape[1] == 0:
        return np.zeros(source_array.shape[0], dtype=float)
    return -np.mean(source_array, axis=1)


def _compute_gene_set_prune_number_masks(state, *, gene_set_mask, gene_set_sort_rank, gene_set_prune_number):
    return state._compute_gene_set_batches(
        V=None,
        X_orig=state.X_orig[:, gene_set_mask],
        mean_shifts=state.mean_shifts[gene_set_mask],
        scale_factors=state.scale_factors[gene_set_mask],
        sort_values=gene_set_sort_rank[gene_set_mask],
        stop_at=gene_set_prune_number,
        tag="gene sets",
    )


def _combine_prune_masks(prune_masks, prune_number, sort_rank, tag, *, log_fn=None, trace_level=None):
    if prune_masks is None or len(prune_masks) == 0:
        return None
    all_prune_mask = np.full(len(prune_masks[0]), False)
    for cur_prune_mask in prune_masks:
        all_prune_mask[cur_prune_mask] = True
        if log_fn is not None and trace_level is not None:
            log_fn(
                "Adding %d relatively uncorrelated %ss (total now %d)"
                % (np.sum(cur_prune_mask), tag, np.sum(all_prune_mask)),
                trace_level,
            )
        if np.sum(all_prune_mask) > prune_number:
            break
    if np.sum(all_prune_mask) > prune_number:
        threshold_value = sorted(sort_rank[all_prune_mask])[prune_number - 1]
        all_prune_mask[sort_rank > threshold_value] = False
    if np.sum(~all_prune_mask) > 0 and log_fn is not None and trace_level is not None:
        log_fn(
            "Found %d %ss remaining after pruning to max number (of %d)"
            % (np.sum(all_prune_mask), tag, len(all_prune_mask)),
            trace_level,
        )
    return all_prune_mask


def _discovery_similarity_to_leaders(
    state,
    candidate_index,
    leader_indices,
    *,
    X_orig=None,
    mean_shifts=None,
    scale_factors=None,
):
    if len(leader_indices) == 0:
        return np.zeros(0, dtype=float)
    if X_orig is None:
        X_orig = state.X_orig
    if mean_shifts is None:
        mean_shifts = getattr(state, "mean_shifts", np.zeros(X_orig.shape[1], dtype=float))
    if scale_factors is None:
        scale_factors = getattr(state, "scale_factors", np.ones(X_orig.shape[1], dtype=float))
    candidate_cols = X_orig[:, [candidate_index]]
    leader_cols = X_orig[:, leader_indices]
    if hasattr(state, "_compute_V"):
        similarities = state._compute_V(
            candidate_cols,
            mean_shifts[[candidate_index]],
            scale_factors[[candidate_index]],
            X_orig2=leader_cols,
            mean_shifts2=mean_shifts[leader_indices],
            scale_factors2=scale_factors[leader_indices],
        )
    else:
        if sparse.issparse(candidate_cols):
            dot_product = candidate_cols.T.dot(leader_cols).toarray().astype(float)
        else:
            dot_product = np.asarray(candidate_cols, dtype=float).T.dot(
                leader_cols.toarray() if sparse.issparse(leader_cols) else np.asarray(leader_cols, dtype=float)
            )
        similarities = (
            dot_product / candidate_cols.shape[0]
            - np.outer(mean_shifts[[candidate_index]], mean_shifts[leader_indices])
        ) / (np.outer(scale_factors[[candidate_index]], scale_factors[leader_indices]) + 1e-10)
    return np.asarray(similarities, dtype=float).reshape(-1)


def _compute_retained_redundancy_counts(state, retained_indices, threshold, *, block_size=256):
    if len(retained_indices) == 0:
        return np.zeros(0, dtype=int)
    counts = np.zeros(len(retained_indices), dtype=int)
    retained_indices = np.asarray(retained_indices, dtype=int)
    full_X = state.X_orig[:, retained_indices]
    mean_shifts = getattr(state, "mean_shifts", np.zeros(state.X_orig.shape[1], dtype=float))
    scale_factors = getattr(state, "scale_factors", np.ones(state.X_orig.shape[1], dtype=float))
    full_mean = mean_shifts[retained_indices]
    full_scale = scale_factors[retained_indices]
    for start in range(0, len(retained_indices), block_size):
        stop = min(len(retained_indices), start + block_size)
        block = retained_indices[start:stop]
        if hasattr(state, "_compute_V"):
            cur_V = state._compute_V(
                state.X_orig[:, block],
                mean_shifts[block],
                scale_factors[block],
                X_orig2=full_X,
                mean_shifts2=full_mean,
                scale_factors2=full_scale,
            )
        else:
            block_X = state.X_orig[:, block]
            if sparse.issparse(block_X):
                dot_product = block_X.T.dot(full_X).toarray().astype(float)
            else:
                dot_product = np.asarray(block_X, dtype=float).T.dot(
                    full_X.toarray() if sparse.issparse(full_X) else np.asarray(full_X, dtype=float)
                )
            cur_V = (
                dot_product / block_X.shape[0]
                - np.outer(mean_shifts[block], full_mean)
            ) / (np.outer(scale_factors[block], full_scale) + 1e-10)
        counts[start:stop] = np.sum(np.asarray(cur_V, dtype=float) >= float(threshold), axis=1)
    counts[counts < 1] = 1
    return counts


def _resolve_discovery_weighting_mode(discovery_redundancy_weighting, discovery_redundancy_weighting_mode):
    if discovery_redundancy_weighting_mode is not None:
        resolved = str(discovery_redundancy_weighting_mode)
    else:
        resolved = "effective_size" if discovery_redundancy_weighting else "none"
    return resolved


def _clip_similarity(value):
    return float(np.clip(float(value), 0.0, 1.0))


def _compute_discovery_effective_size(family_size, mean_similarity):
    family_size = int(family_size)
    if family_size <= 1:
        return 1.0
    mean_similarity = _clip_similarity(mean_similarity)
    return float(family_size / (1.0 + (family_size - 1) * mean_similarity))


def _compute_family_mean_similarity(member_similarities):
    if len(member_similarities) == 0:
        return np.nan
    return float(np.mean(np.asarray(member_similarities, dtype=float)))


def _build_discovery_plan(
    state,
    *,
    retained_gene_set_mask_full,
    gene_set_sort_rank,
    gene_set_prob_vector_full,
    similarity_X_orig=None,
    similarity_mean_shifts=None,
    similarity_scale_factors=None,
    max_num_discovery_gene_sets,
    auto_discovery_subset,
    discovery_redundancy_weighting,
    discovery_redundancy_weighting_mode,
    discovery_similarity_threshold,
):
    discovery_redundancy_weighting_mode = _resolve_discovery_weighting_mode(
        discovery_redundancy_weighting,
        discovery_redundancy_weighting_mode,
    )
    num_gene_sets_total = len(state.gene_sets)
    retained_row_indices_full = np.where(np.asarray(retained_gene_set_mask_full, dtype=bool))[0]
    in_discovery_mask_full = np.zeros(num_gene_sets_total, dtype=bool)
    discovery_representative_mask_full = np.zeros(num_gene_sets_total, dtype=bool)
    discovery_family_id_full = np.full(num_gene_sets_total, -1, dtype=int)
    discovery_family_size_full = np.zeros(num_gene_sets_total, dtype=int)
    discovery_family_mean_similarity_full = np.full(num_gene_sets_total, np.nan, dtype=float)
    discovery_family_effective_size_full = np.full(num_gene_sets_total, np.nan, dtype=float)
    discovery_weight_full = np.zeros(num_gene_sets_total, dtype=float)

    if len(retained_row_indices_full) == 0:
        empty_prob = np.zeros((0, gene_set_prob_vector_full.shape[1]), dtype=float)
        return DiscoveryPlan(
            retained_gene_set_mask_full=np.asarray(retained_gene_set_mask_full, dtype=bool),
            in_discovery_mask_full=in_discovery_mask_full,
            discovery_family_id_full=discovery_family_id_full,
            discovery_representative_mask_full=discovery_representative_mask_full,
            discovery_family_size_full=discovery_family_size_full,
            discovery_family_mean_similarity_full=discovery_family_mean_similarity_full,
            discovery_family_effective_size_full=discovery_family_effective_size_full,
            discovery_weight_full=discovery_weight_full,
            discovery_prob_vector=empty_prob,
            discovery_row_indices_full=np.zeros(0, dtype=int),
            retained_row_indices_full=retained_row_indices_full,
        )

    ordered_retained_indices = retained_row_indices_full[
        np.argsort(
            np.asarray(gene_set_sort_rank, dtype=float)[retained_row_indices_full],
            kind="stable",
        )
    ]
    leader_indices = []
    family_members = []
    family_member_similarities = []

    for candidate_index in ordered_retained_indices:
        if len(leader_indices) == 0:
            family_id = 0
            leader_indices.append(int(candidate_index))
            family_members.append([int(candidate_index)])
            family_member_similarities.append([])
            discovery_representative_mask_full[candidate_index] = True
            discovery_family_id_full[candidate_index] = family_id
            continue

        similarities = _discovery_similarity_to_leaders(
            state,
            int(candidate_index),
            leader_indices,
            X_orig=similarity_X_orig,
            mean_shifts=similarity_mean_shifts,
            scale_factors=similarity_scale_factors,
        )
        best_family = int(np.argmax(similarities))
        best_similarity = float(similarities[best_family]) if similarities.size > 0 else -np.inf

        allow_new_family = best_similarity < float(discovery_similarity_threshold)
        if auto_discovery_subset and max_num_discovery_gene_sets is not None:
            allow_new_family = allow_new_family and len(leader_indices) < int(max_num_discovery_gene_sets)
        if not auto_discovery_subset:
            allow_new_family = best_similarity < float(discovery_similarity_threshold)

        if allow_new_family:
            family_id = len(leader_indices)
            leader_indices.append(int(candidate_index))
            family_members.append([int(candidate_index)])
            family_member_similarities.append([])
            discovery_representative_mask_full[candidate_index] = True
        else:
            family_id = best_family
            family_members[family_id].append(int(candidate_index))
            family_member_similarities[family_id].append(_clip_similarity(best_similarity))
        discovery_family_id_full[candidate_index] = int(family_id)

    for family_id, member_indices in enumerate(family_members):
        family_size = len(member_indices)
        discovery_family_size_full[member_indices] = family_size
        family_mean_similarity = _compute_family_mean_similarity(family_member_similarities[family_id])
        family_effective_size = _compute_discovery_effective_size(family_size, family_mean_similarity)
        discovery_family_mean_similarity_full[member_indices] = family_mean_similarity
        discovery_family_effective_size_full[member_indices] = family_effective_size

    retained_prob = np.asarray(gene_set_prob_vector_full[retained_row_indices_full, :], dtype=float)
    if auto_discovery_subset:
        discovery_row_indices_full = np.asarray(leader_indices, dtype=int)
        in_discovery_mask_full[discovery_row_indices_full] = True
        representative_prob_vector = np.asarray(gene_set_prob_vector_full[discovery_row_indices_full, :], dtype=float)
        if discovery_redundancy_weighting_mode == "effective_size":
            multipliers = np.asarray(
                [
                    discovery_family_effective_size_full[int(leader_index)]
                    for leader_index in discovery_row_indices_full
                ],
                dtype=float,
            )[:, np.newaxis]
            discovery_prob_vector = representative_prob_vector * multipliers
        elif discovery_redundancy_weighting_mode == "log_effective_size":
            multipliers = np.asarray(
                [
                    1.0 + math.log(max(1.0, discovery_family_effective_size_full[int(leader_index)]))
                    for leader_index in discovery_row_indices_full
                ],
                dtype=float,
            )[:, np.newaxis]
            discovery_prob_vector = representative_prob_vector * multipliers
        elif discovery_redundancy_weighting_mode == "none":
            discovery_prob_vector = representative_prob_vector
        else:
            raise ValueError(
                "Unknown discovery redundancy weighting mode: %s"
                % discovery_redundancy_weighting_mode
            )
        for family_id, member_indices in enumerate(family_members):
            family_weight = float(np.mean(discovery_prob_vector[family_id, :])) if discovery_prob_vector.shape[1] > 0 else 0.0
            discovery_weight_full[member_indices] = family_weight
    else:
        discovery_row_indices_full = retained_row_indices_full.copy()
        in_discovery_mask_full[retained_row_indices_full] = True
        discovery_prob_vector = retained_prob
        family_lookup = {int(idx): family_id for family_id, members in enumerate(family_members) for idx in members}
        for retained_index in retained_row_indices_full:
            discovery_family_id_full[retained_index] = family_lookup[int(retained_index)]
        discovery_weight_full[retained_row_indices_full] = np.mean(discovery_prob_vector, axis=1) if discovery_prob_vector.shape[1] > 0 else 0.0

    return DiscoveryPlan(
        retained_gene_set_mask_full=np.asarray(retained_gene_set_mask_full, dtype=bool),
        in_discovery_mask_full=in_discovery_mask_full,
        discovery_family_id_full=discovery_family_id_full,
        discovery_representative_mask_full=discovery_representative_mask_full,
        discovery_family_size_full=discovery_family_size_full,
        discovery_family_mean_similarity_full=discovery_family_mean_similarity_full,
        discovery_family_effective_size_full=discovery_family_effective_size_full,
        discovery_weight_full=discovery_weight_full,
        discovery_prob_vector=np.asarray(discovery_prob_vector, dtype=float),
        discovery_row_indices_full=np.asarray(discovery_row_indices_full, dtype=int),
        retained_row_indices_full=np.asarray(retained_row_indices_full, dtype=int),
    )


def _compute_within_run_factor_redundancy_profile(
    state,
    weight_floor,
    *,
    metric_factor_scope="primary",
    mass_profile=None,
    primary_mass_floor=None,
):
    redundancy_basis, canonical = _extract_overlap_basis_matrix(state)
    if primary_mass_floor is None:
        primary_mass_floor = _DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC
    if mass_profile is None:
        mass_profile = _compute_factor_mass_profile(
            state,
            mass_floor_frac=primary_mass_floor,
        )
    factor_indices = _metric_factor_indices_from_mass_profile(
        mass_profile,
        metric_factor_scope,
        primary_mass_floor=primary_mass_floor,
    )
    canonical = _select_factor_columns(canonical, factor_indices)
    if canonical is None or canonical.shape[1] <= 1:
        return {
            "redundancy_basis": redundancy_basis,
            "metric_factor_scope": str(metric_factor_scope),
            "metric_factor_count": 0 if canonical is None else int(canonical.shape[1]),
            "redundancy_max": 0.0,
            "redundancy_q90": 0.0,
            "redundancy_mean": 0.0,
            "nearest_neighbor_overlaps": [],
        }

    nearest_neighbor_overlaps = []
    for left_index in range(canonical.shape[1]):
        nearest_overlap = 0.0
        for right_index in range(canonical.shape[1]):
            if left_index == right_index:
                continue
            nearest_overlap = max(
                nearest_overlap,
                _weighted_jaccard_similarity(
                    canonical[:, left_index],
                    canonical[:, right_index],
                    weight_floor,
                ),
            )
        nearest_neighbor_overlaps.append(float(nearest_overlap))

    overlap_array = np.asarray(nearest_neighbor_overlaps, dtype=float)
    return {
        "redundancy_basis": redundancy_basis,
        "metric_factor_scope": str(metric_factor_scope),
        "metric_factor_count": int(canonical.shape[1]),
        "redundancy_max": float(np.max(overlap_array)) if overlap_array.size > 0 else 0.0,
        "redundancy_q90": float(np.quantile(overlap_array, 0.9)) if overlap_array.size > 0 else 0.0,
        "redundancy_mean": float(np.mean(overlap_array)) if overlap_array.size > 0 else 0.0,
        "nearest_neighbor_overlaps": nearest_neighbor_overlaps,
    }


def _compute_factor_mass_profile(state, *, mass_floor_frac):
    component_masses = []
    if getattr(state, "exp_gene_set_factors", None) is not None and state.exp_gene_set_factors.size > 0:
        gene_set_factors = np.asarray(state.exp_gene_set_factors, dtype=float)
        gene_set_factors = np.nan_to_num(gene_set_factors, nan=0.0, posinf=0.0, neginf=0.0)
        component_masses.append(np.sum(np.maximum(gene_set_factors, 0.0), axis=0))
    if getattr(state, "exp_gene_factors", None) is not None and state.exp_gene_factors is not None and state.exp_gene_factors.size > 0:
        gene_factors = np.asarray(state.exp_gene_factors, dtype=float)
        gene_factors = np.nan_to_num(gene_factors, nan=0.0, posinf=0.0, neginf=0.0)
        component_masses.append(np.sum(np.maximum(gene_factors, 0.0), axis=0))
    elif getattr(state, "exp_pheno_factors", None) is not None and state.exp_pheno_factors is not None and state.exp_pheno_factors.size > 0:
        pheno_factors = np.asarray(state.exp_pheno_factors, dtype=float)
        pheno_factors = np.nan_to_num(pheno_factors, nan=0.0, posinf=0.0, neginf=0.0)
        component_masses.append(np.sum(np.maximum(pheno_factors, 0.0), axis=0))

    if len(component_masses) == 0:
        return {
            "factor_masses": np.asarray([], dtype=float),
            "mass_fractions": np.asarray([], dtype=float),
            "effective_factor_count": 0.0,
            "mass_ge_floor_factor_count": 0,
            "primary_factor_count": 0,
            "secondary_factor_count": 0,
            "filtered_factor_count": 0,
            "tail_fraction": 0.0,
            "filtered_fraction": 0.0,
            "max_mass_fraction": 0.0,
            "top5_mass_fraction": 0.0,
            "mass_floor_frac": float(mass_floor_frac),
            "primary_mass_floor": float(mass_floor_frac),
            "secondary_mass_floor": float(min(float(_SECONDARY_FACTOR_MASS_FLOOR), float(mass_floor_frac))),
        }

    if len(component_masses) == 1:
        factor_masses = np.asarray(component_masses[0], dtype=float)
    else:
        clipped = [np.maximum(np.asarray(component, dtype=float), 1e-50) for component in component_masses]
        factor_masses = np.exp(np.mean(np.stack([np.log(component) for component in clipped], axis=0), axis=0))

    factor_masses = np.asarray(factor_masses, dtype=float)
    factor_masses = np.nan_to_num(factor_masses, nan=0.0, posinf=0.0, neginf=0.0)
    factor_masses = np.maximum(factor_masses, 0.0)
    total_mass = float(np.sum(factor_masses))
    if total_mass <= 0:
        mass_fractions = np.zeros_like(factor_masses, dtype=float)
    else:
        mass_fractions = factor_masses / total_mass

    denom = float(np.sum(np.square(factor_masses)))
    if denom <= 0:
        effective_factor_count = 0.0
    else:
        effective_factor_count = float((total_mass ** 2) / denom)

    primary_mass_floor, secondary_mass_floor = _resolve_factor_mass_floors(mass_floor_frac)
    sorted_mass_fractions = np.sort(mass_fractions)[::-1]
    return {
        "factor_masses": factor_masses,
        "mass_fractions": mass_fractions,
        "effective_factor_count": effective_factor_count,
        "mass_ge_floor_factor_count": int(np.sum(mass_fractions >= float(mass_floor_frac))),
        "primary_factor_count": int(np.sum(mass_fractions >= primary_mass_floor)),
        "secondary_factor_count": int(np.sum((mass_fractions >= secondary_mass_floor) & (mass_fractions < primary_mass_floor))),
        "filtered_factor_count": int(np.sum(mass_fractions < secondary_mass_floor)),
        "tail_fraction": float(np.sum(mass_fractions[mass_fractions < secondary_mass_floor])) if mass_fractions.size > 0 else 0.0,
        "filtered_fraction": float(np.mean(mass_fractions < secondary_mass_floor)) if mass_fractions.size > 0 else 0.0,
        "max_mass_fraction": float(sorted_mass_fractions[0]) if sorted_mass_fractions.size > 0 else 0.0,
        "top5_mass_fraction": float(np.sum(sorted_mass_fractions[:5])) if sorted_mass_fractions.size > 0 else 0.0,
        "mass_floor_frac": float(mass_floor_frac),
        "primary_mass_floor": float(primary_mass_floor),
        "secondary_mass_floor": float(secondary_mass_floor),
    }


def _factor_label_to_index(label):
    if label is None:
        return None
    label_str = str(label).strip()
    if not label_str.startswith("Factor"):
        return None
    try:
        index = int(label_str.replace("Factor", "")) - 1
    except ValueError:
        return None
    return index if index >= 0 else None


def _build_blockwise_family_keep_indices_from_records(
    records,
    *,
    gene_overlap_threshold=None,
    gene_set_overlap_threshold=None,
    unique_fraction_threshold=None,
):
    if gene_overlap_threshold is None:
        gene_overlap_threshold = _DEFAULT_BLOCKWISE_FAMILY_MERGE_GENE_OVERLAP
    if gene_set_overlap_threshold is None:
        gene_set_overlap_threshold = _DEFAULT_BLOCKWISE_FAMILY_MERGE_GENE_SET_OVERLAP
    if unique_fraction_threshold is None:
        unique_fraction_threshold = _DEFAULT_BLOCKWISE_FAMILY_MERGE_UNIQUE_FRACTION
    num_factors = int(len(records))
    if num_factors <= 1:
        return {
            "keep_indices": list(range(num_factors)),
            "components": [list(range(num_factors))] if num_factors == 1 else [],
        }

    parents = list(range(num_factors))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for factor_index, record in enumerate(records):
        gene_overlap = float(record.get("gene_max_jaccard", 0.0))
        gene_set_overlap = float(record.get("gene_set_max_jaccard", 0.0))
        combined_unique = float(record.get("combined_unique_fraction", 1.0))
        gene_neighbor = _factor_label_to_index(record.get("gene_nearest_factor"))
        gene_set_neighbor = _factor_label_to_index(record.get("gene_set_nearest_factor"))

        if gene_neighbor is not None and gene_neighbor < num_factors and gene_overlap >= float(gene_overlap_threshold):
            union(factor_index, gene_neighbor)
        if gene_set_neighbor is not None and gene_set_neighbor < num_factors and gene_set_overlap >= float(gene_set_overlap_threshold):
            union(factor_index, gene_set_neighbor)

        if combined_unique <= float(unique_fraction_threshold):
            candidate_neighbors = []
            if gene_neighbor is not None and gene_neighbor < num_factors:
                candidate_neighbors.append((gene_overlap, gene_neighbor))
            if gene_set_neighbor is not None and gene_set_neighbor < num_factors:
                candidate_neighbors.append((gene_set_overlap, gene_set_neighbor))
            if len(candidate_neighbors) > 0:
                _, best_neighbor = max(candidate_neighbors, key=lambda item: (item[0], -item[1]))
                union(factor_index, best_neighbor)

    component_map = {}
    for factor_index in range(num_factors):
        component_map.setdefault(find(factor_index), []).append(factor_index)
    components = [sorted(indices) for indices in component_map.values()]
    components.sort(key=lambda indices: (len(indices), indices[0]), reverse=True)

    keep_indices = []
    for indices in components:
        representative = max(
            indices,
            key=lambda index: (
                float(records[index].get("combined_mass_fraction", 0.0)),
                float(records[index].get("gene_effective_support", 0.0)),
                float(records[index].get("gene_set_effective_support", 0.0)),
                -int(index),
            ),
        )
        keep_indices.append(int(representative))

    keep_indices = sorted(set(keep_indices))
    return {
        "keep_indices": keep_indices,
        "components": components,
    }


_DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC = 0.005
_DEFAULT_BLOCKWISE_GLOBAL_REFINEMENT_PASSES = 2
_DEFAULT_BLOCKWISE_K0 = 15
_DEFAULT_BLOCKWISE_REFINEMENT_LAMBDA_FREEZE_PASSES = 1
_DEFAULT_BLOCKWISE_REFINEMENT_LAMBDA_DAMPING = 0.25
_DEFAULT_BLOCKWISE_REFINEMENT_COLLAPSE_DROP_FRAC = 0.35
_DEFAULT_BLOCKWISE_REFINEMENT_DELAMBDA_SPIKE_MULT = 4.0
_DEFAULT_BLOCKWISE_MIN_SHRINKAGE_REFINEMENT_PASSES = 2
_DEFAULT_BLOCKWISE_FAMILY_MERGE_GENE_OVERLAP = 0.8
_DEFAULT_BLOCKWISE_FAMILY_MERGE_GENE_SET_OVERLAP = 0.6
_DEFAULT_BLOCKWISE_FAMILY_MERGE_UNIQUE_FRACTION = 0.2
_DEFAULT_BLOCKWISE_PASS_CHECKPOINT_START = 5
_DEFAULT_BLOCKWISE_PASS_CHECKPOINT_END = 12
_PRIMARY_FACTOR_MASS_FLOOR = 0.005
_SECONDARY_FACTOR_MASS_FLOOR = 0.001


def _coerce_candidate_float(value, default=None):
    if value in (None, "", "NA"):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _summarize_primary_factor_size_from_records(records, *, primary_mass_floor=_PRIMARY_FACTOR_MASS_FLOOR):
    gene_masses = []
    gene_supports = []
    gene_max_weights = []
    gene_max_jaccards = []
    gene_top5_weight_fractions = []
    for record in records or []:
        mass_fraction = _coerce_candidate_float(record.get("combined_mass_fraction"), 0.0)
        if mass_fraction is None or mass_fraction < float(primary_mass_floor):
            continue
        gene_mass = _coerce_candidate_float(record.get("gene_mass"))
        if gene_mass is not None and gene_mass > 0:
            gene_masses.append(float(gene_mass))
        gene_support = _coerce_candidate_float(record.get("gene_effective_support"))
        if gene_support is not None and gene_support > 0:
            gene_supports.append(float(gene_support))
        gene_max_weight = _coerce_candidate_float(record.get("gene_max_weight"))
        if gene_max_weight is not None:
            gene_max_weights.append(float(gene_max_weight))
        gene_max_jaccard = _coerce_candidate_float(record.get("gene_max_jaccard"))
        if gene_max_jaccard is not None:
            gene_max_jaccards.append(float(gene_max_jaccard))
        top5_weight_fraction = _coerce_candidate_float(record.get("gene_top5_weight_fraction"))
        if top5_weight_fraction is not None:
            gene_top5_weight_fractions.append(float(top5_weight_fraction))

    mass_array = np.asarray(gene_masses, dtype=float)
    support_array = np.asarray(gene_supports, dtype=float)
    max_weight_array = np.asarray(gene_max_weights, dtype=float)
    max_jaccard_array = np.asarray(gene_max_jaccards, dtype=float)
    top5_array = np.asarray(gene_top5_weight_fractions, dtype=float)
    return {
        "primary_gene_mass_median": float(np.median(mass_array)) if mass_array.size > 0 else None,
        "primary_gene_mass_q25": float(np.quantile(mass_array, 0.25)) if mass_array.size > 0 else None,
        "primary_gene_mass_q75": float(np.quantile(mass_array, 0.75)) if mass_array.size > 0 else None,
        "primary_gene_effective_support_median": float(np.median(support_array)) if support_array.size > 0 else None,
        "primary_gene_effective_support_q25": float(np.quantile(support_array, 0.25)) if support_array.size > 0 else None,
        "primary_gene_effective_support_q75": float(np.quantile(support_array, 0.75)) if support_array.size > 0 else None,
        "primary_gene_max_jaccard_vs_all_q90": float(np.quantile(max_jaccard_array, 0.9)) if max_jaccard_array.size > 0 else None,
        "primary_gene_max_weight_q90": float(np.quantile(max_weight_array, 0.9)) if max_weight_array.size > 0 else None,
        "primary_gene_top5_weight_fraction_median": float(np.median(top5_array)) if top5_array.size > 0 else None,
    }


def _median_optional(values):
    finite_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(finite_values) == 0:
        return None
    return float(np.median(np.asarray(finite_values, dtype=float)))


def _collect_run_indices_by_modal_factor_count(run_states, run_summaries):
    factor_count_to_indices = {}
    for index, run_state in enumerate(run_states):
        factor_count_to_indices.setdefault(int(run_summaries[index].get("num_factors", 0)), []).append(index)
    max_support = max(len(indices) for indices in factor_count_to_indices.values())
    supported_counts = [key for key, indices in factor_count_to_indices.items() if len(indices) == max_support]

    def _median_error_for_count(count_value):
        errors = [
            float(run_summaries[index]["reconstruction_error"])
            for index in factor_count_to_indices[count_value]
            if run_summaries[index].get("reconstruction_error") is not None
        ]
        if len(errors) == 0:
            return float("inf")
        return float(np.median(np.asarray(errors, dtype=float)))

    modal_factor_count = min(
        supported_counts,
        key=lambda key: (_median_error_for_count(key), int(key)),
    )
    return modal_factor_count, factor_count_to_indices[modal_factor_count]


def _match_factor_cosines(reference_matrix, other_matrix):
    if reference_matrix is None or other_matrix is None:
        return []
    if reference_matrix.shape[1] == 0 or other_matrix.shape[1] == 0:
        return []
    reference_norm = _normalize_factor_columns(reference_matrix)
    other_norm = _normalize_factor_columns(other_matrix)
    similarity = reference_norm.T @ other_norm
    ref_inds, other_inds = scipy.optimize.linear_sum_assignment(1.0 - similarity)
    return [float(similarity[ref_index, other_index]) for ref_index, other_index in zip(ref_inds, other_inds)]


def _summarize_phi_candidate(run_states, run_summaries, *, phi, weight_floor, mass_floor_frac, max_num_factors, metric_factor_scope="primary"):
    modal_factor_count, modal_indices = _collect_run_indices_by_modal_factor_count(run_states, run_summaries)
    reference_index = min(modal_indices, key=lambda idx: _best_run_sort_key(run_summaries[idx]))
    run_mass_profiles = [
        _compute_factor_mass_profile(run_states[index], mass_floor_frac=mass_floor_frac)
        for index in modal_indices
    ]
    run_metric_factor_indices = {
        int(index): _metric_factor_indices_from_mass_profile(
            run_mass_profiles[offset],
            metric_factor_scope,
            primary_mass_floor=mass_floor_frac,
        )
        for offset, index in enumerate(modal_indices)
    }

    best_error = None
    best_evidence = None
    for index in modal_indices:
        reconstruction_error = run_summaries[index].get("reconstruction_error")
        if reconstruction_error is not None:
            best_error = reconstruction_error if best_error is None else min(best_error, reconstruction_error)
        evidence = run_summaries[index].get("evidence")
        if evidence is not None:
            best_evidence = evidence if best_evidence is None else min(best_evidence, evidence)

    matched_cosines = []
    reference_matrix = _select_factor_columns(
        _extract_canonical_factor_matrix(run_states[reference_index]),
        run_metric_factor_indices[int(reference_index)],
    )
    for index in modal_indices:
        if index == reference_index:
            continue
        matched_cosines.extend(
            _match_factor_cosines(
                reference_matrix,
                _select_factor_columns(
                    _extract_canonical_factor_matrix(run_states[index]),
                    run_metric_factor_indices[int(index)],
                ),
            )
        )

    if len(modal_indices) <= 1:
        stability = None
        stability_defined = False
    elif len(matched_cosines) == 0:
        stability = 0.0
        stability_defined = True
    else:
        stability = float(np.mean(np.asarray(matched_cosines, dtype=float)))
        stability_defined = True

    run_redundancy_profiles = [
        _compute_within_run_factor_redundancy_profile(
            run_states[index],
            weight_floor,
            metric_factor_scope=metric_factor_scope,
            mass_profile=run_mass_profiles[offset],
            primary_mass_floor=mass_floor_frac,
        )
        for offset, index in enumerate(modal_indices)
    ]
    redundancy_basis_values = [str(profile.get("redundancy_basis", "unknown")) for profile in run_redundancy_profiles]
    redundancy_basis = redundancy_basis_values[0] if len(set(redundancy_basis_values)) == 1 else "mixed"
    redundancy_max_values = [float(profile["redundancy_max"]) for profile in run_redundancy_profiles]
    redundancy_q90_values = [float(profile["redundancy_q90"]) for profile in run_redundancy_profiles]
    redundancy_mean_values = [float(profile["redundancy_mean"]) for profile in run_redundancy_profiles]
    effective_factor_counts = [float(profile["effective_factor_count"]) for profile in run_mass_profiles]
    mass_ge_floor_factor_counts = [int(profile["mass_ge_floor_factor_count"]) for profile in run_mass_profiles]
    primary_factor_counts = [int(profile["primary_factor_count"]) for profile in run_mass_profiles]
    secondary_factor_counts = [int(profile["secondary_factor_count"]) for profile in run_mass_profiles]
    filtered_factor_counts = [int(profile["filtered_factor_count"]) for profile in run_mass_profiles]
    tail_fractions = [float(profile["tail_fraction"]) for profile in run_mass_profiles]
    filtered_fractions = [float(profile["filtered_fraction"]) for profile in run_mass_profiles]
    max_mass_fractions = [float(profile["max_mass_fraction"]) for profile in run_mass_profiles]
    top5_mass_fractions = [float(profile["top5_mass_fraction"]) for profile in run_mass_profiles]
    run_primary_size_profiles = []
    for index in modal_indices:
        if hasattr(run_states[index], "_collect_factor_metrics_records"):
            run_primary_size_profiles.append(
                _summarize_primary_factor_size_from_records(
                    run_states[index]._collect_factor_metrics_records(),
                    primary_mass_floor=mass_floor_frac,
                )
            )
    primary_gene_effective_support_medians = [
        profile.get("primary_gene_effective_support_median")
        for profile in run_primary_size_profiles
    ]
    primary_gene_mass_medians = [
        profile.get("primary_gene_mass_median")
        for profile in run_primary_size_profiles
    ]
    primary_gene_mass_q25s = [
        profile.get("primary_gene_mass_q25")
        for profile in run_primary_size_profiles
    ]
    primary_gene_mass_q75s = [
        profile.get("primary_gene_mass_q75")
        for profile in run_primary_size_profiles
    ]
    primary_gene_effective_support_q25s = [
        profile.get("primary_gene_effective_support_q25")
        for profile in run_primary_size_profiles
    ]
    primary_gene_effective_support_q75s = [
        profile.get("primary_gene_effective_support_q75")
        for profile in run_primary_size_profiles
    ]
    primary_gene_max_jaccard_q90s = [
        profile.get("primary_gene_max_jaccard_vs_all_q90")
        for profile in run_primary_size_profiles
    ]
    primary_gene_max_weight_q90s = [
        profile.get("primary_gene_max_weight_q90")
        for profile in run_primary_size_profiles
    ]
    primary_gene_top5_weight_fraction_medians = [
        profile.get("primary_gene_top5_weight_fraction_median")
        for profile in run_primary_size_profiles
    ]
    final_delambdas = [
        float(run_summaries[index]["final_delambda"])
        for index in modal_indices
        if run_summaries[index].get("final_delambda") is not None
    ]
    final_iterations = [
        int(run_summaries[index]["final_iterations"])
        for index in modal_indices
        if run_summaries[index].get("final_iterations") is not None
    ]
    converged_values = [
        1.0 if bool(run_summaries[index].get("converged", False)) else 0.0
        for index in modal_indices
        if run_summaries[index].get("converged") is not None
    ]
    hit_iteration_cap_values = [
        1.0 if bool(run_summaries[index].get("hit_iteration_cap", False)) else 0.0
        for index in modal_indices
        if run_summaries[index].get("hit_iteration_cap") is not None
    ]
    backend_values = [str(run_summaries[index].get("backend", "full")) for index in modal_indices]
    backend = backend_values[0] if len(set(backend_values)) == 1 else "mixed"
    backend_details = [run_summaries[index].get("backend_details") or {} for index in modal_indices]
    num_blocks = [
        int(detail["num_blocks"])
        for detail in backend_details
        if detail.get("num_blocks") is not None
    ]
    block_sizes = [
        int(detail["block_size"])
        for detail in backend_details
        if detail.get("block_size") is not None
    ]
    epoch_counts = [
        int(detail["epochs"])
        for detail in backend_details
        if detail.get("epochs") is not None
    ]
    columns_evaluated = [
        int(detail["columns_evaluated"])
        for detail in backend_details
        if detail.get("columns_evaluated") is not None
    ]
    warm_started_values = [
        1.0 if bool(detail.get("warm_started", False)) else 0.0
        for detail in backend_details
        if detail.get("warm_started") is not None
    ]
    wall_times = [
        float(detail["wall_time_sec"])
        for detail in backend_details
        if detail.get("wall_time_sec") is not None
    ]
    epoch_error_traces = [detail.get("epoch_error_trace") or [] for detail in backend_details]

    return {
        "phi": float(phi),
        "modal_factor_count": int(modal_factor_count),
        "run_support": float(len(modal_indices)) / float(max(1, len(run_states))),
        "metric_factor_scope": str(metric_factor_scope),
        "stability": None if stability is None else float(stability),
        "stability_defined": bool(stability_defined),
        "num_modal_runs": int(len(modal_indices)),
        "redundancy": float(np.median(np.asarray(redundancy_max_values, dtype=float))) if len(redundancy_max_values) > 0 else 0.0,
        "redundancy_basis": redundancy_basis,
        "redundancy_max": float(np.median(np.asarray(redundancy_max_values, dtype=float))) if len(redundancy_max_values) > 0 else 0.0,
        "redundancy_q90": float(np.median(np.asarray(redundancy_q90_values, dtype=float))) if len(redundancy_q90_values) > 0 else 0.0,
        "redundancy_mean": float(np.median(np.asarray(redundancy_mean_values, dtype=float))) if len(redundancy_mean_values) > 0 else 0.0,
        "redundancy_max_worst": float(np.max(np.asarray(redundancy_max_values, dtype=float))) if len(redundancy_max_values) > 0 else 0.0,
        "effective_factor_count": float(np.median(np.asarray(effective_factor_counts, dtype=float))) if len(effective_factor_counts) > 0 else 0.0,
        "mass_ge_floor_factor_count": int(round(float(np.median(np.asarray(mass_ge_floor_factor_counts, dtype=float))))) if len(mass_ge_floor_factor_counts) > 0 else 0,
        "primary_factor_count": int(round(float(np.median(np.asarray(primary_factor_counts, dtype=float))))) if len(primary_factor_counts) > 0 else 0,
        "secondary_factor_count": int(round(float(np.median(np.asarray(secondary_factor_counts, dtype=float))))) if len(secondary_factor_counts) > 0 else 0,
        "filtered_factor_count": int(round(float(np.median(np.asarray(filtered_factor_counts, dtype=float))))) if len(filtered_factor_counts) > 0 else 0,
        "tail_fraction": float(np.median(np.asarray(tail_fractions, dtype=float))) if len(tail_fractions) > 0 else 0.0,
        "filtered_fraction": float(np.median(np.asarray(filtered_fractions, dtype=float))) if len(filtered_fractions) > 0 else 0.0,
        "max_mass_fraction": float(np.median(np.asarray(max_mass_fractions, dtype=float))) if len(max_mass_fractions) > 0 else 0.0,
        "top5_mass_fraction": float(np.median(np.asarray(top5_mass_fractions, dtype=float))) if len(top5_mass_fractions) > 0 else 0.0,
        "primary_mass_floor": float(mass_floor_frac),
        "secondary_mass_floor": float(min(float(_SECONDARY_FACTOR_MASS_FLOOR), float(mass_floor_frac))),
        "primary_gene_mass_median": _median_optional(primary_gene_mass_medians),
        "primary_gene_mass_q25": _median_optional(primary_gene_mass_q25s),
        "primary_gene_mass_q75": _median_optional(primary_gene_mass_q75s),
        "primary_gene_effective_support_median": _median_optional(primary_gene_effective_support_medians),
        "primary_gene_effective_support_q25": _median_optional(primary_gene_effective_support_q25s),
        "primary_gene_effective_support_q75": _median_optional(primary_gene_effective_support_q75s),
        "primary_gene_max_jaccard_vs_all_q90": _median_optional(primary_gene_max_jaccard_q90s),
        "primary_gene_max_weight_q90": _median_optional(primary_gene_max_weight_q90s),
        "primary_gene_top5_weight_fraction_median": _median_optional(primary_gene_top5_weight_fraction_medians),
        "mass_floor_frac": float(mass_floor_frac),
        "best_error": None if best_error is None else float(best_error),
        "best_evidence": None if best_evidence is None else float(best_evidence),
        "final_delambda": float(np.median(np.asarray(final_delambdas, dtype=float))) if len(final_delambdas) > 0 else None,
        "final_iterations": int(round(float(np.median(np.asarray(final_iterations, dtype=float))))) if len(final_iterations) > 0 else None,
        "converged_fraction": float(np.mean(np.asarray(converged_values, dtype=float))) if len(converged_values) > 0 else None,
        "hit_iteration_cap_fraction": float(np.mean(np.asarray(hit_iteration_cap_values, dtype=float))) if len(hit_iteration_cap_values) > 0 else None,
        "reference_run_index": int(reference_index),
        "modal_run_indices": [int(index) for index in modal_indices],
        "matched_cosines": [float(value) for value in matched_cosines],
        "run_redundancy_max": redundancy_max_values,
        "run_redundancy_q90": redundancy_q90_values,
        "run_redundancy_mean": redundancy_mean_values,
        "capped": bool(int(modal_factor_count) >= int(max_num_factors)),
        "backend": backend,
        "blockwise_num_blocks": int(round(float(np.median(np.asarray(num_blocks, dtype=float))))) if len(num_blocks) > 0 else None,
        "blockwise_block_size": int(round(float(np.median(np.asarray(block_sizes, dtype=float))))) if len(block_sizes) > 0 else None,
        "blockwise_epochs": int(round(float(np.median(np.asarray(epoch_counts, dtype=float))))) if len(epoch_counts) > 0 else None,
        "blockwise_columns_evaluated": int(round(float(np.median(np.asarray(columns_evaluated, dtype=float))))) if len(columns_evaluated) > 0 else None,
        "blockwise_warm_started": bool(np.mean(np.asarray(warm_started_values, dtype=float)) >= 0.5) if len(warm_started_values) > 0 else False,
        "blockwise_wall_time_sec": float(np.median(np.asarray(wall_times, dtype=float))) if len(wall_times) > 0 else None,
        "blockwise_epoch_error_trace": epoch_error_traces[0] if len(epoch_error_traces) > 0 else [],
    }


def _find_candidate_by_phi(candidates_by_phi, phi_value):
    for existing_phi, candidate in candidates_by_phi.items():
        if math.isclose(existing_phi, phi_value, rel_tol=1e-12, abs_tol=1e-15):
            return candidate
    return None


def _evaluate_phi_candidate(
    template_state,
    *,
    phi,
    seed,
    runs_per_step,
    factor_kwargs,
    learn_phi_backend="sentinel_pruned",
    weight_floor,
    mass_floor_frac=_DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC,
    metric_factor_scope="primary",
    prune_genes_num,
    prune_gene_sets_num,
    max_num_iterations,
    warm_start_payload=None,
    factor_phi_factors_out=None,
    factor_phi_gene_set_clusters_out=None,
    factor_phi_gene_clusters_out=None,
    cluster_row_min_max_loading=0.01,
    factor_output_scope="primary",
    log_fn,
    info_level,
):
    log_fn(
        "Evaluating automatic phi candidate %.6g with %d restart(s) [backend=%s]"
        % (phi, runs_per_step, learn_phi_backend),
        info_level,
    )
    search_factor_kwargs = dict(factor_kwargs)
    search_factor_kwargs["phi"] = phi
    search_factor_kwargs["lmm_auth_key"] = None
    search_factor_kwargs["label_gene_sets_only"] = False
    search_factor_kwargs["label_include_phenos"] = False
    search_factor_kwargs["label_individually"] = False
    if learn_phi_backend == "blockwise_global_w":
        search_factor_kwargs["factor_backend"] = "blockwise_global_w"
        search_factor_kwargs["blockwise_warm_start_state"] = warm_start_payload
    else:
        if prune_genes_num is not None:
            search_factor_kwargs["gene_prune_number"] = int(prune_genes_num)
    if max_num_iterations is not None:
        search_factor_kwargs["max_num_iterations"] = int(max_num_iterations)

    child_seeds = _derive_factor_run_seeds(seed, runs_per_step)
    run_states = []
    run_summaries = []
    for run_index, child_seed in enumerate(child_seeds):
        run_state = _clone_runtime_state(template_state)
        run_summary = _run_factor_with_seed(
            run_state,
            seed=child_seed,
            run_index=run_index,
            factor_kwargs=search_factor_kwargs,
        )
        run_states.append(run_state)
        run_summaries.append(run_summary)

    candidate = _summarize_phi_candidate(
        run_states,
        run_summaries,
        phi=phi,
        weight_floor=weight_floor,
        mass_floor_frac=mass_floor_frac,
        max_num_factors=int(factor_kwargs.get("max_num_factors", 0)),
        metric_factor_scope=metric_factor_scope,
    )
    candidate["run_summaries"] = copy.deepcopy(run_summaries)
    candidate["backend"] = str(learn_phi_backend if learn_phi_backend == "blockwise_global_w" else factor_kwargs.get("factor_backend", "full"))
    reference_state = run_states[int(candidate["reference_run_index"])]
    reference_summary = run_summaries[int(candidate["reference_run_index"])]
    if learn_phi_backend == "blockwise_global_w":
        candidate["warm_start_payload"] = reference_summary.get("blockwise_warm_start_payload")
    if hasattr(reference_state, "_collect_factor_metrics_records"):
        candidate["factor_metric_rows"] = reference_state._collect_factor_metrics_records()
    else:
        candidate["factor_metric_rows"] = []
    if reference_state.num_factors() > 0:
        _append_phi_prefixed_table(
            factor_phi_factors_out,
            phi=phi,
            write_fn=lambda path: reference_state.write_matrix_factors(path, factor_output_scope=factor_output_scope),
        )
        _append_phi_prefixed_table(
            factor_phi_gene_set_clusters_out,
            phi=phi,
            write_fn=lambda path: reference_state.write_clusters(
                gene_set_clusters_output_file=path,
                cluster_row_min_max_loading=cluster_row_min_max_loading,
                factor_output_scope=factor_output_scope,
            ),
        )
        _append_phi_prefixed_table(
            factor_phi_gene_clusters_out,
            phi=phi,
            write_fn=lambda path: reference_state.write_clusters(
                gene_clusters_output_file=path,
                cluster_row_min_max_loading=cluster_row_min_max_loading,
                factor_output_scope=factor_output_scope,
            ),
        )
    best_error = candidate.get("best_error")
    best_evidence = candidate.get("best_evidence")
    log_fn(
        "Automatic phi candidate %.6g summary: K_eff=%d, K_mass=%.3g, K_mass_ge_floor=%d, capped=%s, metric_factor_scope=%s, redundancy_max[%s]=%.3g, redundancy_q90=%.3g, redundancy_mean=%.3g, stability=%s, run_support=%.3g, best_error=%s, best_evidence=%s, final_delambda=%s, final_iterations=%s, converged_fraction=%s, hit_iteration_cap_fraction=%s"
        % (
            float(candidate["phi"]),
            int(candidate["modal_factor_count"]),
            float(candidate.get("effective_factor_count", 0.0)),
            int(candidate.get("mass_ge_floor_factor_count", 0)),
            bool(candidate.get("capped", False)),
            str(candidate.get("metric_factor_scope", metric_factor_scope)),
            str(candidate.get("redundancy_basis", "unknown")),
            float(candidate["redundancy_max"]),
            float(candidate["redundancy_q90"]),
            float(candidate["redundancy_mean"]),
            "NA" if candidate.get("stability") is None else "%.3g" % float(candidate["stability"]),
            float(candidate["run_support"]),
            "NA" if best_error is None else "%.6g" % float(best_error),
            "NA" if best_evidence is None else "%.6g" % float(best_evidence),
            "NA" if candidate.get("final_delambda") is None else "%.6g" % float(candidate["final_delambda"]),
            "NA" if candidate.get("final_iterations") is None else str(int(candidate["final_iterations"])),
            "NA" if candidate.get("converged_fraction") is None else "%.3g" % float(candidate["converged_fraction"]),
            "NA" if candidate.get("hit_iteration_cap_fraction") is None else "%.3g" % float(candidate["hit_iteration_cap_fraction"]),
        ),
        info_level,
    )
    return candidate


def _candidate_target_size_value(candidate, target_metric="gene_effective_support"):
    if target_metric == "gene_mass":
        value = _coerce_candidate_float(candidate.get("primary_gene_mass_median"))
    else:
        value = _coerce_candidate_float(candidate.get("primary_gene_effective_support_median"))
    return value if value is not None and value > 0 else None


def _candidate_target_size_error(candidate, target_value, *, target_metric="gene_effective_support"):
    size_value = _candidate_target_size_value(candidate, target_metric=target_metric)
    if size_value is None:
        return None
    target = float(target_value)
    if target <= 0:
        return None
    return abs(math.log(float(size_value)) - math.log(target))


def _candidate_fit_limit(candidates, fit_loss_frac):
    finite_errors = [
        float(candidate["best_error"])
        for candidate in candidates
        if candidate.get("best_error") is not None and math.isfinite(float(candidate["best_error"]))
    ]
    if len(finite_errors) == 0:
        return None
    return min(finite_errors) * (1.0 + float(fit_loss_frac))


def _candidate_target_acceptability_violations(
    candidate,
    *,
    max_redundancy,
    max_redundancy_q90,
    min_run_support,
    min_stability,
    runs_per_step,
    severe_fit_limit,
    target_metric,
    target_gene_effective_support,
    min_primary_factors,
    max_primary_gene_max_weight_q90,
    redundancy_hard_filter=True,
    gene_by_gene_guardrails=False,
):
    violations = []
    if int(candidate.get("modal_factor_count", 0)) <= 0:
        violations.append("zero_factor")
    if int(candidate.get("primary_factor_count", 0)) < int(min_primary_factors):
        violations.append("too_few_primary_factors")
    if _candidate_target_size_value(candidate, target_metric=target_metric) is None:
        violations.append("undefined_target_size")
    if redundancy_hard_filter:
        if float(candidate.get("redundancy_max", 0.0)) > float(max_redundancy):
            violations.append("redundancy_max")
        if float(candidate.get("redundancy_q90", 0.0)) > float(max_redundancy_q90):
            violations.append("redundancy_q90")
    if float(candidate.get("run_support", 0.0)) < float(min_run_support):
        violations.append("run_support")
    if runs_per_step > 1:
        if int(candidate.get("num_modal_runs", 0)) < 2:
            violations.append("modal_run_count")
        elif not bool(candidate.get("stability_defined", False)):
            violations.append("stability_undefined")
        elif candidate.get("stability") is None or float(candidate["stability"]) < float(min_stability):
            violations.append("stability")
    if severe_fit_limit is not None:
        best_error = _coerce_candidate_float(candidate.get("best_error"))
        if best_error is None:
            violations.append("fit_missing")
        elif best_error > float(severe_fit_limit) + 1e-12:
            violations.append("severe_fit_loss")
    if max_primary_gene_max_weight_q90 is not None:
        spike_value = _coerce_candidate_float(candidate.get("primary_gene_max_weight_q90"))
        if spike_value is None or spike_value > float(max_primary_gene_max_weight_q90):
            violations.append("primary_gene_spike")
    if gene_by_gene_guardrails:
        active_genes = _coerce_candidate_float(candidate.get("num_active_genes"))
        median_support = _coerce_candidate_float(candidate.get("primary_gene_effective_support_median"))
        broad_limit_parts = []
        if target_gene_effective_support is not None:
            broad_limit_parts.append(2.5 * float(target_gene_effective_support))
        if active_genes is not None:
            broad_limit_parts.append(0.5 * float(active_genes))
        if median_support is not None and broad_limit_parts and median_support > max(broad_limit_parts):
            violations.append("gene_gene_severe_broadness")
        if float(candidate.get("redundancy_q90", 0.0)) > 0.70:
            violations.append("gene_gene_severe_redundancy_q90")
        if float(candidate.get("redundancy_max", 0.0)) > 0.90:
            violations.append("gene_gene_severe_redundancy_max")
        max_weight_q90 = _coerce_candidate_float(candidate.get("primary_gene_max_weight_q90"))
        if (
            max_weight_q90 is not None
            and median_support is not None
            and active_genes is not None
            and max_weight_q90 < 1e-4
            and median_support > 0.5 * float(active_genes)
        ):
            violations.append("gene_gene_near_uniform_factors")
    return violations


def _candidate_selection_warnings(
    candidate,
    *,
    fit_warning_limit,
    max_redundancy=None,
    max_redundancy_q90=None,
    redundancy_hard_filter=True,
):
    warnings = []
    if fit_warning_limit is not None:
        best_error = _coerce_candidate_float(candidate.get("best_error"))
        if best_error is None:
            warnings.append("fit_missing")
        elif best_error > float(fit_warning_limit) + 1e-12:
            warnings.append("fit_loss")
    if not redundancy_hard_filter:
        if max_redundancy is not None and float(candidate.get("redundancy_max", 0.0)) > float(max_redundancy):
            warnings.append("redundancy_max")
        if max_redundancy_q90 is not None and float(candidate.get("redundancy_q90", 0.0)) > float(max_redundancy_q90):
            warnings.append("redundancy_q90")
    return warnings


def _select_phi_candidate(
    candidates,
    *,
    max_redundancy,
    max_redundancy_q90,
    min_run_support,
    min_stability,
    fit_loss_warning_frac,
    max_severe_fit_loss_frac,
    target_gene_mass,
    target_gene_effective_support,
    size_tolerance_frac,
    min_primary_factors,
    max_primary_gene_max_weight_q90,
    runs_per_step,
    redundancy_hard_filter=True,
    gene_by_gene_guardrails=False,
):
    if target_gene_mass is not None:
        target_metric = "gene_mass"
        target_value = float(target_gene_mass)
        target_value_field = "target_gene_mass"
        target_error_field = "target_gene_mass_error_log"
        target_ratio_field = "target_gene_mass_ratio"
        target_reason_in_tolerance = "target_gene_mass_in_tolerance"
        target_reason_closest = "target_gene_mass_closest"
    else:
        target_metric = "gene_effective_support"
        target_value = float(target_gene_effective_support)
        target_value_field = "target_gene_effective_support"
        target_error_field = "target_gene_effective_support_error_log"
        target_ratio_field = "target_gene_effective_support_ratio"
        target_reason_in_tolerance = "target_gene_effective_support_in_tolerance"
        target_reason_closest = "target_gene_effective_support_closest"
    fit_warning_limit = _candidate_fit_limit(candidates, fit_loss_warning_frac)
    severe_fit_limit = _candidate_fit_limit(candidates, max_severe_fit_loss_frac)
    for candidate in candidates:
        size_error = _candidate_target_size_error(candidate, target_value, target_metric=target_metric)
        candidate[target_error_field] = size_error
        candidate[target_value_field] = target_value
        candidate[target_ratio_field] = (
            None
            if _candidate_target_size_value(candidate, target_metric=target_metric) is None
            else float(_candidate_target_size_value(candidate, target_metric=target_metric)) / target_value
        )
        candidate["selection_target_metric"] = target_metric
        candidate["selection_target_value"] = target_value
        candidate["selection_target_error_log"] = size_error
        candidate["selection_target_ratio"] = candidate[target_ratio_field]
        candidate["selection_fit_warning_limit"] = fit_warning_limit
        candidate["selection_severe_fit_limit"] = severe_fit_limit
        candidate["selection_redundancy_hard_filter"] = bool(redundancy_hard_filter)
        candidate["selection_warnings"] = ",".join(
            _candidate_selection_warnings(
                candidate,
                fit_warning_limit=fit_warning_limit,
                max_redundancy=max_redundancy,
                max_redundancy_q90=max_redundancy_q90,
                redundancy_hard_filter=redundancy_hard_filter,
            )
        )

    acceptable = []
    for candidate in candidates:
        violations = _candidate_target_acceptability_violations(
            candidate,
            max_redundancy=max_redundancy,
            max_redundancy_q90=max_redundancy_q90,
            min_run_support=min_run_support,
            min_stability=min_stability,
            runs_per_step=runs_per_step,
            severe_fit_limit=severe_fit_limit,
            target_metric=target_metric,
            target_gene_effective_support=target_gene_effective_support,
            min_primary_factors=min_primary_factors,
            max_primary_gene_max_weight_q90=max_primary_gene_max_weight_q90,
            redundancy_hard_filter=redundancy_hard_filter,
            gene_by_gene_guardrails=gene_by_gene_guardrails,
        )
        candidate["selection_violations"] = ",".join(violations)
        if len(violations) == 0:
            acceptable.append(candidate)

    if len(acceptable) > 0:
        acceptable_uncapped = [candidate for candidate in acceptable if not bool(candidate.get("capped", False))]
        selection_pool = acceptable_uncapped if len(acceptable_uncapped) > 0 else acceptable
        selection_pool_name = "uncapped" if len(acceptable_uncapped) > 0 else "capped"
        tolerance_log = math.log1p(float(size_tolerance_frac))
        in_tolerance = [
            candidate
            for candidate in selection_pool
            if candidate.get(target_error_field) is not None
            and float(candidate[target_error_field]) <= tolerance_log + 1e-12
        ]
        if len(in_tolerance) > 0:
            selected = min(
                in_tolerance,
                key=lambda candidate: (
                    -float(candidate.get("phi", 0.0)),
                    float(candidate.get(target_error_field, float("inf"))),
                    float(candidate.get("tail_fraction", 0.0)),
                    float(candidate.get("filtered_fraction", 0.0)),
                    float(candidate.get("redundancy_q90", 0.0)),
                    float("inf") if candidate.get("primary_gene_max_weight_q90") is None else float(candidate["primary_gene_max_weight_q90"]),
                    float("inf") if candidate.get("best_error") is None else float(candidate["best_error"]),
                ),
            )
            selected["selection_target_tolerance_log"] = float(tolerance_log)
            selected["selection_target_in_tolerance_size"] = int(len(in_tolerance))
            selected["selection_pool"] = selection_pool_name
            selected["selection_frontier_size"] = int(len(in_tolerance))
            selected["selection_marginal_gain"] = None
            return selected, target_reason_in_tolerance

        selected = min(
            selection_pool,
            key=lambda candidate: (
                float(candidate.get(target_error_field, float("inf"))),
                -float(candidate.get("phi", 0.0)),
                float(candidate.get("tail_fraction", 0.0)),
                float(candidate.get("filtered_fraction", 0.0)),
                float(candidate.get("redundancy_q90", 0.0)),
                float("inf") if candidate.get("primary_gene_max_weight_q90") is None else float(candidate["primary_gene_max_weight_q90"]),
                float("inf") if candidate.get("best_error") is None else float(candidate["best_error"]),
            ),
        )
        selected["selection_target_tolerance_log"] = float(tolerance_log)
        selected["selection_target_in_tolerance_size"] = 0
        selected["selection_pool"] = selection_pool_name
        selected["selection_frontier_size"] = int(len(selection_pool))
        selected["selection_marginal_gain"] = None
        return selected, target_reason_closest

    if gene_by_gene_guardrails:
        guardrail_prefix = "gene_gene_"
        guardrail_failed = [
            candidate
            for candidate in candidates
            if any(str(value).startswith(guardrail_prefix) for value in str(candidate.get("selection_violations", "")).split(",") if value)
        ]
        if len(guardrail_failed) == len(candidates):
            for candidate in candidates:
                candidate["selection_pool"] = "failed_gene_by_gene_guardrails"
                candidate["selection_frontier_size"] = 0
                candidate["selection_marginal_gain"] = None
            raise RuntimeError("All gene_by_gene phi candidates failed hard guardrails; inspect learn-phi diagnostics or adjust the target/search range")

    def _fallback_sort_key(candidate):
        violations = candidate.get("selection_violations", "")
        violation_count = len([value for value in str(violations).split(",") if value])
        size_error = candidate.get(target_error_field)
        size_error = float("inf") if size_error is None else float(size_error)
        fit_violation = 0.0
        if fit_warning_limit is not None and candidate.get("best_error") is not None:
            fit_violation = max(0.0, float(candidate["best_error"]) - float(fit_warning_limit))
        return (
            violation_count,
            1 if bool(candidate.get("capped", False)) else 0,
            max(0.0, float(candidate.get("redundancy_q90", 0.0)) - float(max_redundancy_q90)),
            max(0.0, float(candidate.get("redundancy_max", 0.0)) - float(max_redundancy)),
            max(0.0, float(min_run_support) - float(candidate.get("run_support", 0.0))),
            size_error,
            -float(candidate.get("phi", 0.0)),
            fit_violation,
        )

    selected = min(candidates, key=_fallback_sort_key)
    selected["selection_pool"] = "fallback"
    selected["selection_frontier_size"] = 0
    selected["selection_marginal_gain"] = None
    return selected, "fallback_min_target_constraint_violation"

def _write_phi_search_report(report_path, candidates, *, selected_phi, selection_reason):
    if report_path is None:
        return
    columns = [
        "selected",
        "selection_reason",
        "phi",
        "modal_factor_count",
        "effective_factor_count",
        "mass_ge_floor_factor_count",
        "primary_factor_count",
        "secondary_factor_count",
        "filtered_factor_count",
        "tail_fraction",
        "filtered_fraction",
        "mass_floor_frac",
        "primary_mass_floor",
        "secondary_mass_floor",
        "metric_factor_scope",
        "max_mass_fraction",
        "top5_mass_fraction",
        "primary_gene_mass_median",
        "primary_gene_mass_q25",
        "primary_gene_mass_q75",
        "primary_gene_effective_support_median",
        "primary_gene_effective_support_q25",
        "primary_gene_effective_support_q75",
        "primary_gene_max_jaccard_vs_all_q90",
        "primary_gene_max_weight_q90",
        "primary_gene_top5_weight_fraction_median",
        "selection_target_metric",
        "selection_target_value",
        "selection_target_error_log",
        "selection_target_ratio",
        "target_gene_effective_support",
        "target_gene_effective_support_error_log",
        "target_gene_effective_support_ratio",
        "selection_fit_warning_limit",
        "selection_severe_fit_limit",
        "selection_redundancy_hard_filter",
        "selection_violations",
        "selection_warnings",
        "selection_target_tolerance_log",
        "selection_target_in_tolerance_size",
        "capped",
        "num_modal_runs",
        "run_support",
        "stability",
        "stability_defined",
        "redundancy_basis",
        "redundancy_max",
        "redundancy_q90",
        "redundancy_mean",
        "redundancy_max_worst",
        "best_error",
        "best_evidence",
        "final_delambda",
        "final_iterations",
        "converged_fraction",
        "hit_iteration_cap_fraction",
        "backend",
        "blockwise_num_blocks",
        "blockwise_block_size",
        "blockwise_epochs",
        "blockwise_columns_evaluated",
        "blockwise_warm_started",
        "blockwise_wall_time_sec",
        "blockwise_epoch_error_trace",
        "reference_run_index",
        "modal_run_indices",
        "matched_cosines",
    ]

    def _format_value(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (list, tuple)):
            return ",".join([_format_value(item) for item in value])
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            return "%.12g" % value
        return str(value)

    with _open_text_output(report_path) as output_fh:
        output_fh.write("%s\n" % "\t".join(columns))
        for candidate in sorted(candidates, key=lambda row: float(row["phi"])):
            row_values = {
                "selected": math.isclose(float(candidate["phi"]), float(selected_phi), rel_tol=1e-12, abs_tol=1e-15),
                "selection_reason": selection_reason,
                "phi": float(candidate["phi"]),
                "modal_factor_count": int(candidate["modal_factor_count"]),
                "effective_factor_count": float(candidate.get("effective_factor_count", 0.0)),
                "mass_ge_floor_factor_count": int(candidate.get("mass_ge_floor_factor_count", 0)),
                "primary_factor_count": int(candidate.get("primary_factor_count", 0)),
                "secondary_factor_count": int(candidate.get("secondary_factor_count", 0)),
                "filtered_factor_count": int(candidate.get("filtered_factor_count", 0)),
                "tail_fraction": float(candidate.get("tail_fraction", 0.0)),
                "filtered_fraction": float(candidate.get("filtered_fraction", 0.0)),
                "mass_floor_frac": float(candidate.get("mass_floor_frac", _DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC)),
                "primary_mass_floor": candidate.get("primary_mass_floor"),
                "secondary_mass_floor": candidate.get("secondary_mass_floor"),
                "metric_factor_scope": str(candidate.get("metric_factor_scope", "primary")),
                "max_mass_fraction": float(candidate.get("max_mass_fraction", 0.0)),
                "top5_mass_fraction": float(candidate.get("top5_mass_fraction", 0.0)),
                "primary_gene_mass_median": candidate.get("primary_gene_mass_median"),
                "primary_gene_mass_q25": candidate.get("primary_gene_mass_q25"),
                "primary_gene_mass_q75": candidate.get("primary_gene_mass_q75"),
                "primary_gene_effective_support_median": candidate.get("primary_gene_effective_support_median"),
                "primary_gene_effective_support_q25": candidate.get("primary_gene_effective_support_q25"),
                "primary_gene_effective_support_q75": candidate.get("primary_gene_effective_support_q75"),
                "primary_gene_max_jaccard_vs_all_q90": candidate.get("primary_gene_max_jaccard_vs_all_q90"),
                "primary_gene_max_weight_q90": candidate.get("primary_gene_max_weight_q90"),
                "primary_gene_top5_weight_fraction_median": candidate.get("primary_gene_top5_weight_fraction_median"),
                "selection_target_metric": candidate.get("selection_target_metric"),
                "selection_target_value": candidate.get("selection_target_value"),
                "selection_target_error_log": candidate.get("selection_target_error_log"),
                "selection_target_ratio": candidate.get("selection_target_ratio"),
                "target_gene_effective_support": candidate.get("target_gene_effective_support"),
                "target_gene_effective_support_error_log": candidate.get("target_gene_effective_support_error_log"),
                "target_gene_effective_support_ratio": candidate.get("target_gene_effective_support_ratio"),
                "selection_fit_warning_limit": candidate.get("selection_fit_warning_limit"),
                "selection_severe_fit_limit": candidate.get("selection_severe_fit_limit"),
                "selection_violations": candidate.get("selection_violations", ""),
                "selection_warnings": candidate.get("selection_warnings", ""),
                "selection_target_tolerance_log": candidate.get("selection_target_tolerance_log"),
                "selection_target_in_tolerance_size": candidate.get("selection_target_in_tolerance_size"),
                "capped": bool(candidate.get("capped", False)),
                "num_modal_runs": int(candidate.get("num_modal_runs", 0)),
                "run_support": float(candidate["run_support"]),
                "stability": candidate.get("stability"),
                "stability_defined": bool(candidate.get("stability_defined", False)),
                "redundancy_basis": str(candidate.get("redundancy_basis", "unknown")),
                "redundancy_max": float(candidate.get("redundancy_max", candidate.get("redundancy", 0.0))),
                "redundancy_q90": float(candidate.get("redundancy_q90", 0.0)),
                "redundancy_mean": float(candidate.get("redundancy_mean", 0.0)),
                "redundancy_max_worst": float(candidate.get("redundancy_max_worst", 0.0)),
                "best_error": candidate.get("best_error"),
                "best_evidence": candidate.get("best_evidence"),
                "final_delambda": candidate.get("final_delambda"),
                "final_iterations": candidate.get("final_iterations"),
                "converged_fraction": candidate.get("converged_fraction"),
                "hit_iteration_cap_fraction": candidate.get("hit_iteration_cap_fraction"),
                "backend": str(candidate.get("backend", "full")),
                "blockwise_num_blocks": candidate.get("blockwise_num_blocks"),
                "blockwise_block_size": candidate.get("blockwise_block_size"),
                "blockwise_epochs": candidate.get("blockwise_epochs"),
                "blockwise_columns_evaluated": candidate.get("blockwise_columns_evaluated"),
                "blockwise_warm_started": bool(candidate.get("blockwise_warm_started", False)),
                "blockwise_wall_time_sec": candidate.get("blockwise_wall_time_sec"),
                "blockwise_epoch_error_trace": candidate.get("blockwise_epoch_error_trace", []),
                "reference_run_index": int(candidate["reference_run_index"]),
                "modal_run_indices": candidate.get("modal_run_indices", []),
                "matched_cosines": candidate.get("matched_cosines", []),
            }
            output_fh.write("%s\n" % "\t".join([_format_value(row_values.get(column)) for column in columns]))


def _write_phi_factor_metrics_report(report_path, candidates, *, selected_phi):
    if report_path is None:
        return
    metric_columns = None
    for candidate in candidates:
        factor_metric_rows = candidate.get("factor_metric_rows") or []
        if len(factor_metric_rows) > 0:
            metric_columns = list(factor_metric_rows[0].keys())
            break
    if metric_columns is None:
        metric_columns = []
    candidate_columns = [
        "candidate_primary_gene_mass_median",
        "candidate_primary_gene_mass_q25",
        "candidate_primary_gene_mass_q75",
        "candidate_primary_gene_effective_support_median",
        "candidate_primary_gene_effective_support_q25",
        "candidate_primary_gene_effective_support_q75",
        "candidate_primary_gene_max_jaccard_vs_all_q90",
        "candidate_primary_gene_max_weight_q90",
        "candidate_primary_gene_top5_weight_fraction_median",
        "candidate_primary_mass_floor",
        "candidate_secondary_mass_floor",
        "candidate_selection_target_metric",
        "candidate_selection_target_value",
        "candidate_selection_target_error_log",
        "candidate_selection_target_ratio",
        "candidate_target_gene_effective_support_error_log",
        "candidate_target_gene_effective_support_ratio",
    ]
    with _open_text_output(report_path) as output_fh:
        output_fh.write("%s\n" % "\t".join(["phi", "selected", "reference_run_index"] + candidate_columns + metric_columns))
        for candidate in sorted(candidates, key=lambda row: float(row["phi"])):
            selected_flag = "1" if math.isclose(float(candidate["phi"]), float(selected_phi), rel_tol=1e-12, abs_tol=1e-15) else "0"
            candidate_values = [
                "" if candidate.get("primary_gene_mass_median") is None else "%.12g" % float(candidate["primary_gene_mass_median"]),
                "" if candidate.get("primary_gene_mass_q25") is None else "%.12g" % float(candidate["primary_gene_mass_q25"]),
                "" if candidate.get("primary_gene_mass_q75") is None else "%.12g" % float(candidate["primary_gene_mass_q75"]),
                "" if candidate.get("primary_gene_effective_support_median") is None else "%.12g" % float(candidate["primary_gene_effective_support_median"]),
                "" if candidate.get("primary_gene_effective_support_q25") is None else "%.12g" % float(candidate["primary_gene_effective_support_q25"]),
                "" if candidate.get("primary_gene_effective_support_q75") is None else "%.12g" % float(candidate["primary_gene_effective_support_q75"]),
                "" if candidate.get("primary_gene_max_jaccard_vs_all_q90") is None else "%.12g" % float(candidate["primary_gene_max_jaccard_vs_all_q90"]),
                "" if candidate.get("primary_gene_max_weight_q90") is None else "%.12g" % float(candidate["primary_gene_max_weight_q90"]),
                "" if candidate.get("primary_gene_top5_weight_fraction_median") is None else "%.12g" % float(candidate["primary_gene_top5_weight_fraction_median"]),
                "" if candidate.get("primary_mass_floor") is None else "%.12g" % float(candidate["primary_mass_floor"]),
                "" if candidate.get("secondary_mass_floor") is None else "%.12g" % float(candidate["secondary_mass_floor"]),
                "" if candidate.get("selection_target_metric") is None else str(candidate["selection_target_metric"]),
                "" if candidate.get("selection_target_value") is None else "%.12g" % float(candidate["selection_target_value"]),
                "" if candidate.get("selection_target_error_log") is None else "%.12g" % float(candidate["selection_target_error_log"]),
                "" if candidate.get("selection_target_ratio") is None else "%.12g" % float(candidate["selection_target_ratio"]),
                "" if candidate.get("target_gene_effective_support_error_log") is None else "%.12g" % float(candidate["target_gene_effective_support_error_log"]),
                "" if candidate.get("target_gene_effective_support_ratio") is None else "%.12g" % float(candidate["target_gene_effective_support_ratio"]),
            ]
            for record in candidate.get("factor_metric_rows") or []:
                output_fh.write(
                    "%s\n"
                    % "\t".join(
                        [
                            "%.12g" % float(candidate["phi"]),
                            selected_flag,
                            str(int(candidate.get("reference_run_index", 0))),
                        ] + candidate_values + [str(record.get(column, "")) for column in metric_columns]
                    )
                )


def _parse_explicit_phi_values(values):
    if values is None:
        return None
    if isinstance(values, str):
        raw_values = [part.strip() for part in values.split(",")]
    else:
        raw_values = list(values)
    parsed = []
    for raw_value in raw_values:
        if raw_value in (None, ""):
            continue
        value = float(raw_value)
        if value <= 0:
            raise ValueError("explicit phi values must be positive")
        if not any(math.isclose(value, existing, rel_tol=1e-12, abs_tol=1e-15) for existing in parsed):
            parsed.append(value)
    return sorted(parsed)


def _record_phi_search_params(
    state,
    *,
    initial_phi,
    selected_candidate,
    selection_reason,
    candidates,
    weight_floor,
    metric_factor_scope,
    max_redundancy,
    max_redundancy_q90,
    runs_per_step,
    min_run_support,
    min_stability,
    fit_loss_warning_frac,
    max_severe_fit_loss_frac,
    target_gene_mass,
    target_gene_effective_support,
    size_tolerance_frac,
    min_primary_factors,
    max_primary_gene_max_weight_q90,
    max_steps,
    explicit_phi_values,
    expand_factor,
    mass_floor_frac,
    learn_phi_backend,
    prune_genes_num,
    prune_gene_sets_num,
    max_num_iterations,
    redundancy_hard_filter=True,
):
    state._record_params(
        {
            "learn_phi": True,
            "learn_phi_initial_phi": float(initial_phi),
            "learn_phi_selected_phi": float(selected_candidate["phi"]),
            "learn_phi_selection_reason": selection_reason,
            "learn_phi_max_redundancy": float(max_redundancy),
            "learn_phi_max_redundancy_q90": float(max_redundancy_q90),
            "learn_phi_redundancy_hard_filter": bool(redundancy_hard_filter),
            "learn_phi_runs_per_step": int(runs_per_step),
            "learn_phi_min_run_support": float(min_run_support),
            "learn_phi_min_stability": float(min_stability),
            "learn_phi_fit_loss_warning_frac": float(fit_loss_warning_frac),
            "learn_phi_max_severe_fit_loss_frac": float(max_severe_fit_loss_frac),
            "learn_phi_target_metric": str(selected_candidate.get("selection_target_metric", "gene_effective_support")),
            "learn_phi_target_gene_mass": None if target_gene_mass is None else float(target_gene_mass),
            "learn_phi_target_gene_effective_support": None if target_gene_effective_support is None else float(target_gene_effective_support),
            "learn_phi_size_tolerance_frac": float(size_tolerance_frac),
            "learn_phi_min_primary_factors": int(min_primary_factors),
            "learn_phi_max_primary_gene_max_weight_q90": (
                None if max_primary_gene_max_weight_q90 is None else float(max_primary_gene_max_weight_q90)
            ),
            "learn_phi_mass_floor_frac": float(mass_floor_frac),
            "learn_phi_backend": str(learn_phi_backend),
            "learn_phi_max_steps": int(max_steps),
            "learn_phi_values": None if explicit_phi_values is None else ",".join("%.12g" % float(value) for value in explicit_phi_values),
            "learn_phi_expand_factor": float(expand_factor),
            "learn_phi_weight_floor": float(weight_floor),
            "learn_phi_metric_factor_scope": str(metric_factor_scope),
            "learn_phi_prune_genes_num": None if prune_genes_num is None else int(prune_genes_num),
            "learn_phi_prune_gene_sets_num": None if prune_gene_sets_num is None else int(prune_gene_sets_num),
            "learn_phi_max_num_iterations": None if max_num_iterations is None else int(max_num_iterations),
            "learn_phi_redundancy_basis": str(selected_candidate.get("redundancy_basis", "unknown")),
            "learn_phi_selection_pool": str(selected_candidate.get("selection_pool", "unknown")),
            "learn_phi_selection_frontier_size": int(selected_candidate.get("selection_frontier_size", 0)),
            "learn_phi_selection_marginal_gain": selected_candidate.get("selection_marginal_gain"),
            "learn_phi_selected_primary_gene_effective_support_median": selected_candidate.get("primary_gene_effective_support_median"),
            "learn_phi_selected_primary_gene_effective_support_q25": selected_candidate.get("primary_gene_effective_support_q25"),
            "learn_phi_selected_primary_gene_effective_support_q75": selected_candidate.get("primary_gene_effective_support_q75"),
            "learn_phi_selected_primary_gene_max_jaccard_vs_all_q90": selected_candidate.get("primary_gene_max_jaccard_vs_all_q90"),
            "learn_phi_selected_primary_gene_max_weight_q90": selected_candidate.get("primary_gene_max_weight_q90"),
            "learn_phi_selected_primary_gene_top5_weight_fraction_median": selected_candidate.get("primary_gene_top5_weight_fraction_median"),
            "learn_phi_selected_target_gene_effective_support_error_log": selected_candidate.get("target_gene_effective_support_error_log"),
            "learn_phi_selected_target_gene_effective_support_ratio": selected_candidate.get("target_gene_effective_support_ratio"),
            "learn_phi_selected_primary_gene_mass_median": selected_candidate.get("primary_gene_mass_median"),
            "learn_phi_selected_primary_gene_mass_q25": selected_candidate.get("primary_gene_mass_q25"),
            "learn_phi_selected_primary_gene_mass_q75": selected_candidate.get("primary_gene_mass_q75"),
            "learn_phi_selected_selection_target_value": selected_candidate.get("selection_target_value"),
            "learn_phi_selected_selection_target_error_log": selected_candidate.get("selection_target_error_log"),
            "learn_phi_selected_selection_target_ratio": selected_candidate.get("selection_target_ratio"),
            "learn_phi_selected_selection_fit_warning_limit": selected_candidate.get("selection_fit_warning_limit"),
            "learn_phi_selected_selection_severe_fit_limit": selected_candidate.get("selection_severe_fit_limit"),
            "learn_phi_selected_selection_warnings": selected_candidate.get("selection_warnings", ""),
            "learn_phi_selected_primary_mass_floor": selected_candidate.get("primary_mass_floor"),
            "learn_phi_selected_secondary_mass_floor": selected_candidate.get("secondary_mass_floor"),
        },
        overwrite=True,
    )
    metric_map = {
        "learn_phi_candidate_phi": "phi",
        "learn_phi_candidate_metric_factor_scope": "metric_factor_scope",
        "learn_phi_candidate_modal_factor_count": "modal_factor_count",
        "learn_phi_candidate_effective_factor_count": "effective_factor_count",
        "learn_phi_candidate_mass_ge_floor_factor_count": "mass_ge_floor_factor_count",
        "learn_phi_candidate_primary_factor_count": "primary_factor_count",
        "learn_phi_candidate_secondary_factor_count": "secondary_factor_count",
        "learn_phi_candidate_filtered_factor_count": "filtered_factor_count",
        "learn_phi_candidate_tail_fraction": "tail_fraction",
        "learn_phi_candidate_filtered_fraction": "filtered_fraction",
        "learn_phi_candidate_max_mass_fraction": "max_mass_fraction",
        "learn_phi_candidate_top5_mass_fraction": "top5_mass_fraction",
        "learn_phi_candidate_primary_gene_mass_median": "primary_gene_mass_median",
        "learn_phi_candidate_primary_gene_mass_q25": "primary_gene_mass_q25",
        "learn_phi_candidate_primary_gene_mass_q75": "primary_gene_mass_q75",
        "learn_phi_candidate_primary_gene_effective_support_median": "primary_gene_effective_support_median",
        "learn_phi_candidate_primary_gene_effective_support_q25": "primary_gene_effective_support_q25",
        "learn_phi_candidate_primary_gene_effective_support_q75": "primary_gene_effective_support_q75",
        "learn_phi_candidate_primary_gene_max_jaccard_vs_all_q90": "primary_gene_max_jaccard_vs_all_q90",
        "learn_phi_candidate_primary_gene_max_weight_q90": "primary_gene_max_weight_q90",
        "learn_phi_candidate_primary_gene_top5_weight_fraction_median": "primary_gene_top5_weight_fraction_median",
        "learn_phi_candidate_primary_mass_floor": "primary_mass_floor",
        "learn_phi_candidate_secondary_mass_floor": "secondary_mass_floor",
        "learn_phi_candidate_selection_target_metric": "selection_target_metric",
        "learn_phi_candidate_selection_target_value": "selection_target_value",
        "learn_phi_candidate_selection_target_error_log": "selection_target_error_log",
        "learn_phi_candidate_selection_target_ratio": "selection_target_ratio",
        "learn_phi_candidate_target_gene_effective_support": "target_gene_effective_support",
        "learn_phi_candidate_target_gene_effective_support_error_log": "target_gene_effective_support_error_log",
        "learn_phi_candidate_target_gene_effective_support_ratio": "target_gene_effective_support_ratio",
        "learn_phi_candidate_selection_fit_warning_limit": "selection_fit_warning_limit",
        "learn_phi_candidate_selection_severe_fit_limit": "selection_severe_fit_limit",
        "learn_phi_candidate_selection_violations": "selection_violations",
        "learn_phi_candidate_selection_warnings": "selection_warnings",
        "learn_phi_candidate_capped": "capped",
        "learn_phi_candidate_num_modal_runs": "num_modal_runs",
        "learn_phi_candidate_run_support": "run_support",
        "learn_phi_candidate_stability": "stability",
        "learn_phi_candidate_stability_defined": "stability_defined",
        "learn_phi_candidate_redundancy_basis": "redundancy_basis",
        "learn_phi_candidate_redundancy": "redundancy",
        "learn_phi_candidate_redundancy_max": "redundancy_max",
        "learn_phi_candidate_redundancy_q90": "redundancy_q90",
        "learn_phi_candidate_redundancy_mean": "redundancy_mean",
        "learn_phi_candidate_redundancy_max_worst": "redundancy_max_worst",
        "learn_phi_candidate_best_error": "best_error",
        "learn_phi_candidate_best_evidence": "best_evidence",
        "learn_phi_candidate_final_delambda": "final_delambda",
        "learn_phi_candidate_final_iterations": "final_iterations",
        "learn_phi_candidate_converged_fraction": "converged_fraction",
        "learn_phi_candidate_hit_iteration_cap_fraction": "hit_iteration_cap_fraction",
        "learn_phi_candidate_backend": "backend",
        "learn_phi_candidate_blockwise_num_blocks": "blockwise_num_blocks",
        "learn_phi_candidate_blockwise_block_size": "blockwise_block_size",
        "learn_phi_candidate_blockwise_epochs": "blockwise_epochs",
        "learn_phi_candidate_blockwise_columns_evaluated": "blockwise_columns_evaluated",
        "learn_phi_candidate_blockwise_warm_started": "blockwise_warm_started",
        "learn_phi_candidate_blockwise_wall_time_sec": "blockwise_wall_time_sec",
    }
    for candidate in sorted(candidates, key=lambda row: float(row["phi"])):
        for param_name, candidate_key in metric_map.items():
            value = candidate.get(candidate_key)
            if value is not None:
                state._record_param(param_name, value)


def _learn_phi(
    state,
    *,
    initial_phi,
    seed,
    runs_per_step,
    max_redundancy,
    max_redundancy_q90,
    min_run_support,
    min_stability,
    fit_loss_warning_frac,
    max_severe_fit_loss_frac,
    target_gene_mass,
    target_gene_effective_support,
    size_tolerance_frac,
    min_primary_factors,
    max_primary_gene_max_weight_q90,
    max_steps,
    explicit_phi_values=None,
    expand_factor,
    weight_floor,
    metric_factor_scope="primary",
    mass_floor_frac=_DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC,
    learn_phi_backend="sentinel_pruned",
    blockwise_warm_start=True,
    report_out,
    factor_phi_metrics_out=None,
    factor_phi_factors_out=None,
    factor_phi_gene_set_clusters_out=None,
    factor_phi_gene_clusters_out=None,
    cluster_row_min_max_loading=0.01,
    factor_output_scope="primary",
    prune_genes_num,
    prune_gene_sets_num,
    max_num_iterations,
    factor_kwargs,
    log_fn,
    info_level,
    redundancy_hard_filter=True,
):
    candidates_by_phi = {}
    explicit_phi_values = _parse_explicit_phi_values(explicit_phi_values)
    _reset_phi_prefixed_outputs(
        [
            factor_phi_factors_out,
            factor_phi_gene_set_clusters_out,
            factor_phi_gene_clusters_out,
        ]
    )

    remaining_evaluations = int(max_steps)
    target_metric = "gene_mass" if target_gene_mass is not None else "gene_effective_support"
    target_value = float(target_gene_mass) if target_gene_mass is not None else float(target_gene_effective_support)

    def _evaluate(phi_value, *, consume_budget=True):
        nonlocal remaining_evaluations
        existing = _find_candidate_by_phi(candidates_by_phi, float(phi_value))
        if existing is not None:
            return existing
        if consume_budget and remaining_evaluations <= 0:
            return None
        warm_start_payload = None
        if learn_phi_backend == "blockwise_global_w" and blockwise_warm_start and len(candidates_by_phi) > 0:
            prior_candidates = [
                candidate
                for candidate in candidates_by_phi.values()
                if candidate.get("warm_start_payload") is not None
            ]
            if len(prior_candidates) > 0:
                warm_start_payload = min(
                    prior_candidates,
                    key=lambda candidate: abs(math.log(float(candidate["phi"])) - math.log(float(phi_value))),
                ).get("warm_start_payload")
        evaluate_kwargs = {
            "phi": float(phi_value),
            "seed": seed,
            "runs_per_step": runs_per_step,
            "factor_kwargs": factor_kwargs,
            "weight_floor": weight_floor,
            "metric_factor_scope": metric_factor_scope,
            "mass_floor_frac": mass_floor_frac,
            "prune_genes_num": prune_genes_num,
            "prune_gene_sets_num": prune_gene_sets_num,
            "max_num_iterations": max_num_iterations,
            "factor_phi_factors_out": factor_phi_factors_out,
            "factor_phi_gene_set_clusters_out": factor_phi_gene_set_clusters_out,
            "factor_phi_gene_clusters_out": factor_phi_gene_clusters_out,
            "cluster_row_min_max_loading": cluster_row_min_max_loading,
            "factor_output_scope": factor_output_scope,
            "log_fn": log_fn,
            "info_level": info_level,
        }
        if learn_phi_backend != "sentinel_pruned" or warm_start_payload is not None:
            evaluate_kwargs["learn_phi_backend"] = learn_phi_backend
            evaluate_kwargs["warm_start_payload"] = warm_start_payload
        candidate = _evaluate_phi_candidate(state, **evaluate_kwargs)
        candidates_by_phi[float(candidate["phi"])] = candidate
        if consume_budget:
            remaining_evaluations -= 1
        return candidate

    initial_phi = float(initial_phi)
    min_phi = initial_phi / 1e4
    max_phi = initial_phi * 1e4
    expand_factor = float(expand_factor)

    def _clip_phi(phi_value):
        return min(max(float(phi_value), min_phi), max_phi)

    def _new_phi(proposal):
        proposal = _clip_phi(proposal)
        existing = _find_candidate_by_phi(candidates_by_phi, proposal)
        if existing is not None:
            return None
        for phi_value in candidates_by_phi.keys():
            if math.isclose(float(proposal), float(phi_value), rel_tol=1e-12, abs_tol=1e-15):
                return None
        return proposal

    def _candidate_size_side(candidate):
        size_value = _candidate_target_size_value(candidate, target_metric=target_metric)
        if size_value is not None:
            if math.isclose(math.log(float(size_value)), math.log(target_value), rel_tol=0.0, abs_tol=1e-12):
                return 0
            return -1 if float(size_value) < target_value else 1
        if int(candidate.get("modal_factor_count", 0)) <= 0:
            return 1
        if bool(candidate.get("capped", False)):
            return -1
        return None

    def _size_bracket():
        sized_candidates = [
            candidate
            for candidate in sorted(candidates_by_phi.values(), key=lambda row: float(row["phi"]))
            if _candidate_target_size_value(candidate, target_metric=target_metric) is not None
        ]
        if len(sized_candidates) < 2:
            return None
        brackets = []
        for lower_candidate, upper_candidate in zip(sized_candidates[:-1], sized_candidates[1:]):
            lower_size = float(_candidate_target_size_value(lower_candidate, target_metric=target_metric))
            upper_size = float(_candidate_target_size_value(upper_candidate, target_metric=target_metric))
            lower_delta = lower_size - target_value
            upper_delta = upper_size - target_value
            if lower_delta == 0.0 or upper_delta == 0.0 or lower_delta * upper_delta < 0:
                brackets.append((abs(math.log(float(upper_candidate["phi"])) - math.log(float(lower_candidate["phi"]))), lower_candidate, upper_candidate))
        if len(brackets) == 0:
            return None
        _, first_candidate, second_candidate = min(brackets, key=lambda item: item[0])
        if float(first_candidate["phi"]) < float(second_candidate["phi"]):
            return first_candidate, second_candidate
        return second_candidate, first_candidate

    def _propose_expansion():
        sorted_candidates = sorted(candidates_by_phi.values(), key=lambda row: float(row["phi"]))
        sized_candidates = [
            candidate
            for candidate in sorted_candidates
            if _candidate_target_size_value(candidate, target_metric=target_metric) is not None
        ]
        if len(sized_candidates) > 0:
            below = [candidate for candidate in sized_candidates if float(_candidate_target_size_value(candidate, target_metric=target_metric)) < target_value]
            above = [candidate for candidate in sized_candidates if float(_candidate_target_size_value(candidate, target_metric=target_metric)) > target_value]
            if len(below) > 0 and len(above) == 0:
                return _new_phi(max(float(candidate["phi"]) for candidate in sorted_candidates) * expand_factor)
            if len(above) > 0 and len(below) == 0:
                return _new_phi(min(float(candidate["phi"]) for candidate in sorted_candidates) / expand_factor)
            return None

        sides = [_candidate_size_side(candidate) for candidate in sorted_candidates]
        if any(side is not None and side < 0 for side in sides):
            return _new_phi(max(float(candidate["phi"]) for candidate in sorted_candidates) * expand_factor)
        if any(side is not None and side > 0 for side in sides):
            return _new_phi(min(float(candidate["phi"]) for candidate in sorted_candidates) / expand_factor)
        if len(sorted_candidates) % 2 == 1:
            return _new_phi(min(float(candidate["phi"]) for candidate in sorted_candidates) / expand_factor)
        return _new_phi(max(float(candidate["phi"]) for candidate in sorted_candidates) * expand_factor)

    if explicit_phi_values is not None:
        log_fn(
            "Evaluating explicit phi candidate set: %s"
            % ",".join("%.6g" % float(value) for value in explicit_phi_values),
            info_level,
        )
        for phi_value in explicit_phi_values:
            _evaluate(phi_value, consume_budget=False)
    else:
        _evaluate(initial_phi, consume_budget=False)

        while remaining_evaluations > 0 and _size_bracket() is None:
            proposal = _propose_expansion()
            if proposal is None:
                sorted_phis = sorted(float(candidate["phi"]) for candidate in candidates_by_phi.values())
                lower_proposal = _new_phi(sorted_phis[0] / expand_factor)
                upper_proposal = _new_phi(sorted_phis[-1] * expand_factor)
                proposal = lower_proposal if lower_proposal is not None else upper_proposal
            if proposal is None:
                break
            _evaluate(proposal)

        while remaining_evaluations > 0:
            bracket = _size_bracket()
            if bracket is None:
                break
            low_candidate, high_candidate = bracket
            low_phi = float(low_candidate["phi"])
            high_phi = float(high_candidate["phi"])
            if not (low_phi < high_phi):
                break
            mid_phi = math.sqrt(low_phi * high_phi)
            if math.isclose(mid_phi, low_phi, rel_tol=1e-12, abs_tol=1e-15) or math.isclose(mid_phi, high_phi, rel_tol=1e-12, abs_tol=1e-15):
                break
            proposal = _new_phi(mid_phi)
            if proposal is None:
                break
            _evaluate(proposal)

    candidates = list(candidates_by_phi.values())
    selected_candidate, selection_reason = _select_phi_candidate(
        candidates,
        max_redundancy=max_redundancy,
        max_redundancy_q90=max_redundancy_q90,
        min_run_support=min_run_support,
        min_stability=min_stability,
        fit_loss_warning_frac=fit_loss_warning_frac,
        max_severe_fit_loss_frac=max_severe_fit_loss_frac,
        target_gene_mass=target_gene_mass,
        target_gene_effective_support=target_gene_effective_support,
        size_tolerance_frac=size_tolerance_frac,
        min_primary_factors=min_primary_factors,
        max_primary_gene_max_weight_q90=max_primary_gene_max_weight_q90,
        runs_per_step=runs_per_step,
        redundancy_hard_filter=redundancy_hard_filter,
        gene_by_gene_guardrails=str(factor_kwargs.get("discovery_model", "gene_by_annotation")) == "gene_by_gene",
    )
    _record_phi_search_params(
        state,
        initial_phi=initial_phi,
        selected_candidate=selected_candidate,
        selection_reason=selection_reason,
        candidates=candidates,
        weight_floor=weight_floor,
        metric_factor_scope=metric_factor_scope,
        max_redundancy=max_redundancy,
        max_redundancy_q90=max_redundancy_q90,
        runs_per_step=runs_per_step,
        min_run_support=min_run_support,
        min_stability=min_stability,
        fit_loss_warning_frac=fit_loss_warning_frac,
        max_severe_fit_loss_frac=max_severe_fit_loss_frac,
        target_gene_mass=target_gene_mass,
        target_gene_effective_support=target_gene_effective_support,
        size_tolerance_frac=size_tolerance_frac,
        min_primary_factors=min_primary_factors,
        max_primary_gene_max_weight_q90=max_primary_gene_max_weight_q90,
        max_steps=max_steps,
        explicit_phi_values=explicit_phi_values,
        expand_factor=expand_factor,
        mass_floor_frac=mass_floor_frac,
        learn_phi_backend=learn_phi_backend,
        prune_genes_num=prune_genes_num,
        prune_gene_sets_num=prune_gene_sets_num,
        max_num_iterations=max_num_iterations,
        redundancy_hard_filter=redundancy_hard_filter,
    )
    _write_phi_search_report(
        report_out,
        candidates,
        selected_phi=selected_candidate["phi"],
        selection_reason=selection_reason,
    )
    _write_phi_factor_metrics_report(
        factor_phi_metrics_out,
        candidates,
        selected_phi=selected_candidate["phi"],
    )
    if target_metric == "gene_mass":
        selected_target_summary = selected_candidate.get("primary_gene_mass_median")
    else:
        selected_target_summary = selected_candidate.get("primary_gene_effective_support_median")
    log_fn(
        "Selected phi %.6g by target-size automatic tuning [%s]: target_%s=%.6g, primary_%s_median=%s, target_error_log=%s, backend=%s, K_eff=%d, K_mass=%.3g, primary_factors=%d, capped=%s, pool=%s, metric_factor_scope=%s, redundancy_max[%s]=%.3g, redundancy_q90=%.3g, gene_max_jaccard_q90=%s, primary_gene_max_weight_q90=%s, stability=%s, run_support=%.3g, selection_warnings=%s"
        % (
            float(selected_candidate["phi"]),
            selection_reason,
            str(target_metric),
            float(target_value),
            str(target_metric),
            "NA" if selected_target_summary is None else "%.6g" % float(selected_target_summary),
            "NA" if selected_candidate.get("selection_target_error_log") is None else "%.6g" % float(selected_candidate["selection_target_error_log"]),
            str(selected_candidate.get("backend", learn_phi_backend)),
            int(selected_candidate["modal_factor_count"]),
            float(selected_candidate.get("effective_factor_count", 0.0)),
            int(selected_candidate.get("primary_factor_count", 0)),
            bool(selected_candidate.get("capped", False)),
            str(selected_candidate.get("selection_pool", "unknown")),
            str(selected_candidate.get("metric_factor_scope", metric_factor_scope)),
            str(selected_candidate.get("redundancy_basis", "unknown")),
            float(selected_candidate.get("redundancy_max", selected_candidate.get("redundancy", 0.0))),
            float(selected_candidate.get("redundancy_q90", 0.0)),
            "NA" if selected_candidate.get("primary_gene_max_jaccard_vs_all_q90") is None else "%.3g" % float(selected_candidate["primary_gene_max_jaccard_vs_all_q90"]),
            "NA" if selected_candidate.get("primary_gene_max_weight_q90") is None else "%.3g" % float(selected_candidate["primary_gene_max_weight_q90"]),
            "NA" if selected_candidate.get("stability") is None else "%.3g" % float(selected_candidate["stability"]),
            float(selected_candidate["run_support"]),
            str(selected_candidate.get("selection_warnings", "")),
        ),
        info_level,
    )
    return selected_candidate


def _build_factor_run_summary(state, *, run_index, seed, evidence, likelihood, reconstruction_error, factor_gene_set_x_pheno):
    backend_details = copy.deepcopy(getattr(state, "last_factorization_backend_details", None))
    return {
        "run_index": int(run_index),
        "seed": None if seed is None else int(seed),
        "evidence": None if evidence is None else float(evidence),
        "likelihood": None if likelihood is None else float(likelihood),
        "reconstruction_error": None if reconstruction_error is None else float(reconstruction_error),
        "num_factors": int(state.num_factors()),
        "factor_gene_set_x_pheno": bool(factor_gene_set_x_pheno),
        "backend": getattr(state, "last_factorization_backend", "full"),
        "backend_details": backend_details,
        "final_delambda": None if getattr(state, "last_factorization_final_delambda", None) is None else float(state.last_factorization_final_delambda),
        "final_iterations": None if getattr(state, "last_factorization_iterations", None) is None else int(state.last_factorization_iterations),
        "converged": None if getattr(state, "last_factorization_converged", None) is None else bool(state.last_factorization_converged),
        "hit_iteration_cap": None if getattr(state, "last_factorization_hit_iteration_cap", None) is None else bool(state.last_factorization_hit_iteration_cap),
    }


def _best_run_sort_key(run_summary):
    evidence = run_summary.get("evidence")
    return (
        float("inf") if evidence is None else float(evidence),
        -int(run_summary.get("num_factors", 0)),
        int(run_summary.get("run_index", 0)),
    )


def _top_indices_by_factor(scores, num_top, *, allowed_mask=None, gene_sets=None, warn_fn=None, context="factor"):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("expected a 2D score matrix")
    num_items, num_factors = scores.shape
    out = np.full((num_top, num_factors), -1, dtype=int)
    for factor_index in range(num_factors):
        order = np.argsort(-scores[:, factor_index])
        if allowed_mask is not None:
            filtered = [idx for idx in order if allowed_mask[idx] and scores[idx, factor_index] > 0]
            if filtered:
                chosen = filtered[:num_top]
                out[: len(chosen), factor_index] = chosen
                continue
            if warn_fn is not None:
                warn_fn(
                    "%s %d has no positive-loading gene sets matching --gene-sets-for-labeling; using unfiltered gene sets for its label"
                    % (context, factor_index + 1)
                )
        chosen = order[: min(num_top, num_items)]
        out[: len(chosen), factor_index] = chosen
    return out


def _top_anchor_gene_set_indices_by_factor(scores, num_top, *, allowed_mask=None, warn_fn=None):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 3:
        raise ValueError("expected a 3D anchor gene-set score matrix")
    num_items, num_factors, num_anchors = scores.shape
    out = np.full((num_top, num_factors, num_anchors), -1, dtype=int)
    for anchor_index in range(num_anchors):
        out[:, :, anchor_index] = _top_indices_by_factor(
            scores[:, :, anchor_index],
            num_top,
            allowed_mask=allowed_mask,
            warn_fn=warn_fn,
            context="anchor %d factor" % (anchor_index + 1),
        )
    return out


def _finalize_factor_outputs(
    state,
    *,
    factor_gene_set_x_pheno,
    factor_top_loading_type,
    lmm_auth_key,
    lmm_model,
    lmm_provider,
    label_gene_sets_only,
    label_include_phenos,
    label_individually,
    bail_fn,
    warn_fn,
    log_fn,
    info_level,
    labeling_module,
    gene_sets_for_labeling=None,
):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level

    reorder_inds = np.argsort(-state.factor_relevance)

    state.exp_lambdak = state.exp_lambdak[reorder_inds]
    state.factor_anchor_relevance = state.factor_anchor_relevance[reorder_inds, :]
    if state.factor_anchor_marginal_relevance is not None:
        state.factor_anchor_marginal_relevance = state.factor_anchor_marginal_relevance[reorder_inds, :]
    state.factor_relevance = state.factor_relevance[reorder_inds]
    if state.factor_marginal_relevance is not None:
        state.factor_marginal_relevance = state.factor_marginal_relevance[reorder_inds]
    if state.exp_gene_factors is not None:
        state.exp_gene_factors = state.exp_gene_factors[:, reorder_inds]
    full_gene_factors = getattr(state, "exp_gene_factors_full_projected", None)
    if full_gene_factors is not None:
        state.exp_gene_factors_full_projected = full_gene_factors[:, reorder_inds]
    if state.exp_pheno_factors is not None:
        state.exp_pheno_factors = state.exp_pheno_factors[:, reorder_inds]
    if state.trait_linkage_joint is not None:
        state.trait_linkage_joint = state.trait_linkage_joint[:, reorder_inds]
    if state.trait_linkage_marginal is not None:
        state.trait_linkage_marginal = state.trait_linkage_marginal[:, reorder_inds]
    if state.trait_linkage_marginal_overlap is not None:
        state.trait_linkage_marginal_overlap = state.trait_linkage_marginal_overlap[:, reorder_inds]
    if state.trait_linkage_factor_total_mass is not None:
        state.trait_linkage_factor_total_mass = state.trait_linkage_factor_total_mass[reorder_inds]
    if state.trait_linkage_factor_n_eff is not None:
        state.trait_linkage_factor_n_eff = state.trait_linkage_factor_n_eff[reorder_inds]
    if state.trait_linkage_factor_top_share is not None:
        state.trait_linkage_factor_top_share = state.trait_linkage_factor_top_share[reorder_inds]
    if state.trait_linkage_factor_top10_share is not None:
        state.trait_linkage_factor_top10_share = state.trait_linkage_factor_top10_share[reorder_inds]
    if state.trait_linkage_broad_factor_flag is not None:
        state.trait_linkage_broad_factor_flag = state.trait_linkage_broad_factor_flag[reorder_inds]
    state.exp_gene_set_factors = state.exp_gene_set_factors[:, reorder_inds]

    threshold = 1e-5
    if state.num_factors() > 0:
        if state.exp_gene_factors is not None and np.max(state.exp_gene_factors) > 0:
            state.exp_gene_factors[state.exp_gene_factors < np.max(state.exp_gene_factors) * threshold] = 0
        if state.exp_pheno_factors is not None and np.max(state.exp_pheno_factors) > 0:
            state.exp_pheno_factors[state.exp_pheno_factors < np.max(state.exp_pheno_factors) * threshold] = 0
        if np.max(state.exp_gene_set_factors) > 0:
            state.exp_gene_set_factors[state.exp_gene_set_factors < np.max(state.exp_gene_set_factors) * threshold] = 0

    num_top = 5
    exp_gene_factors_for_top = state.get_factor_loadings(state.exp_gene_factors, loading_type=factor_top_loading_type)
    exp_pheno_factors_for_top = state.get_factor_loadings(state.exp_pheno_factors, loading_type=factor_top_loading_type)
    exp_gene_set_factors_for_top = state.get_factor_loadings(state.exp_gene_set_factors, loading_type=factor_top_loading_type)
    gene_sets_for_labeling = set(gene_sets_for_labeling or [])
    gene_set_labeling_mask = None
    if gene_sets_for_labeling:
        gene_set_labeling_mask = np.array([gene_set in gene_sets_for_labeling for gene_set in state.gene_sets], dtype=bool)
        matched = int(np.sum(gene_set_labeling_mask))
        if matched == 0:
            warn("--gene-sets-for-labeling has no overlap with retained EAGGL gene sets; using unfiltered gene sets for labels")
            gene_set_labeling_mask = None
        state._record_params(
            {
                "gene_sets_for_labeling_num_ids": int(len(gene_sets_for_labeling)),
                "gene_sets_for_labeling_num_matched": int(matched),
            }
        )

    top_anchor_gene_or_pheno_inds = None
    top_anchor_pheno_or_gene_inds = None

    if factor_gene_set_x_pheno:
        top_anchor_gene_or_pheno_inds = np.swapaxes(
            np.argsort(
                -(exp_pheno_factors_for_top).T[:, :, np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis, :, :],
                axis=1,
            )[:, :num_top, :],
            0,
            1,
        )
        if exp_gene_factors_for_top is not None:
            top_anchor_pheno_or_gene_inds = np.swapaxes(
                np.argsort(
                    -(exp_gene_factors_for_top).T[:, :, np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis, :, :],
                    axis=1,
                )[:, :num_top, :],
                0,
                1,
            )
    else:
        top_anchor_gene_or_pheno_inds = np.swapaxes(
            np.argsort(
                -(exp_gene_factors_for_top).T[:, :, np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis, :, :],
                axis=1,
            )[:, :num_top, :],
            0,
            1,
        )
        if exp_pheno_factors_for_top is not None:
            top_anchor_pheno_or_gene_inds = np.swapaxes(
                np.argsort(
                    -(exp_pheno_factors_for_top).T[:, :, np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis, :, :],
                    axis=1,
                )[:, :num_top, :],
                0,
                1,
            )

    top_anchor_gene_set_inds = _top_anchor_gene_set_indices_by_factor(
        exp_gene_set_factors_for_top[:, :, np.newaxis] * state.gene_set_prob_factor_vector[:, np.newaxis, :],
        num_top,
        allowed_mask=gene_set_labeling_mask,
        warn_fn=warn,
    )

    top_gene_or_pheno_inds = None
    top_pheno_or_gene_inds = None
    top_capture_inds = eaggl_phenotype_annotation.rank_top_capture_indices(
        exp_pheno_factors_for_top,
        state.pheno_capture_strength,
        num_top,
    )

    if factor_gene_set_x_pheno:
        top_gene_or_pheno_inds = top_capture_inds
        if exp_gene_factors_for_top is not None:
            top_pheno_or_gene_inds = np.swapaxes(
                np.argsort(-(exp_gene_factors_for_top).T, axis=1)[:, :num_top],
                0,
                1,
            )
    else:
        top_gene_or_pheno_inds = np.swapaxes(
            np.argsort(-(exp_gene_factors_for_top).T, axis=1)[:, :num_top],
            0,
            1,
        )
        if exp_pheno_factors_for_top is not None:
            top_pheno_or_gene_inds = top_capture_inds

    top_gene_set_inds = _top_indices_by_factor(
        exp_gene_set_factors_for_top,
        num_top,
        allowed_mask=gene_set_labeling_mask,
        warn_fn=warn,
    )

    labeling_module.populate_factor_labels(
        state,
        factor_gene_set_x_pheno=factor_gene_set_x_pheno,
        top_gene_set_inds=top_gene_set_inds,
        top_anchor_gene_set_inds=top_anchor_gene_set_inds,
        top_gene_or_pheno_inds=top_gene_or_pheno_inds,
        top_anchor_gene_or_pheno_inds=top_anchor_gene_or_pheno_inds,
        top_pheno_or_gene_inds=top_pheno_or_gene_inds,
        lmm_auth_key=lmm_auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        label_gene_sets_only=label_gene_sets_only,
        label_include_phenos=label_include_phenos,
        label_individually=label_individually,
        log_fn=log,
        bail_fn=bail,
        warn_fn=warn,
    )

    log("Found %d factors" % state.num_factors(), INFO)
    return reorder_inds


def _run_factor_single(state, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, seed=None, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, max_num_discovery_gene_sets=None, auto_discovery_subset=True, discovery_redundancy_weighting=True, discovery_redundancy_weighting_mode="effective_size", discovery_similarity_threshold=0.35, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, lmm_model=None, lmm_provider="openai", label_gene_sets_only=False, label_include_phenos=False, label_individually=False, gene_sets_for_labeling=None, factor_top_loading_type="combined", keep_original_loadings=False, project_phenos_from_gene_sets=False, pheno_capture_input="weighted_thresholded", trait_linkage_source="combined", trait_linkage_threshold=1.0, trait_linkage_computation_mode="sparse_full", no_trait_linkage=False, factor_backend="full", blockwise_gene_set_block_size=5000, blockwise_epochs=3, blockwise_shuffle_blocks=True, blockwise_warm_start=True, blockwise_max_blocks=None, blockwise_report_out=None, blockwise_warm_start_state=None, factors_out=None, factor_metrics_out=None, gene_set_clusters_out=None, gene_clusters_out=None, cluster_row_min_max_loading=0.01, factor_output_scope="primary", discovery_model="gene_by_annotation", anchor_aggregation="multi", gene_gene_beta_source="beta", gene_gene_pair_prior=None, gene_gene_pair_prior_effective_size=None, gene_gene_logbf_base="natural", gene_gene_diagonal_weight=0.0, gene_gene_matrix_floor=1e-3, gene_gene_excess_probability=True, gene_gene_row_sum_cap=True, gene_gene_sparsity=0.0, gene_gene_profligate_correction="none", learn_phi_target_gene_mass=None, learn_phi_target_gene_effective_support=None, *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, labeling_module):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    DEBUG = debug_level
    TRACE = trace_level

    if state.X_orig is None:
        bail("Cannot run factoring without X")
    state.discovery_model = str(discovery_model)
    anchor_aggregation = str(anchor_aggregation)
    if anchor_aggregation not in {"multi", "any"}:
        bail("--anchor-aggregation must be one of: multi, any")

    # Persist explicit anchor masks for downstream output writers.
    state.anchor_pheno_mask = np.copy(anchor_pheno_mask) if anchor_pheno_mask is not None else None
    state.anchor_gene_mask = np.copy(anchor_gene_mask) if anchor_gene_mask is not None else None

    if (anchor_any_gene or anchor_any_pheno or anchor_gene_set or anchor_gene_mask is not None or anchor_pheno_mask is not None or pheno_prune_value is not None or pheno_prune_number is not None) and state.X_phewas_beta is None:
        bail("Cannot run factoring without X phewas")

    if anchor_any_gene:
        if anchor_any_pheno:
            warn("Ignoring anchor any pheno since anchor any gene was specified")
        if anchor_gene_mask:
            warn("Ignoring anchor gene since anchor any gene was specified")
        if anchor_pheno_mask:
            warn("Ignoring anchor pheno since anchor any gene was specified")
        if anchor_gene_set:
            warn("Ignoring anchor gene set since anchor any gene was specified")

        state._record_params({"anchor": "any_gene"})
        anchor_any_pheno = False
        anchor_pheno_mask = None
        anchor_gene_mask = np.full(state.X_orig.shape[0], True)
        anchor_gene_set = False

    elif anchor_any_pheno:
        if anchor_gene_mask:
            warn("Ignoring anchor gene since anchor any pheno was specified")
        if anchor_pheno_mask:
            warn("Ignoring anchor pheno since anchor any pheno was specified")
        if anchor_gene_set:
            warn("Ignoring anchor gene set since anchor any pheno was specified")
        anchor_gene_mask = None
        anchor_pheno_mask = np.full(state.X_phewas_beta.shape[0], True)
        anchor_gene_set = False
        state._record_params({"anchor": "any_pheno"})
    elif anchor_gene_set:
        if anchor_gene_mask:
            warn("Ignoring anchor gene since anchor gene set was specified")
        if anchor_pheno_mask:
            warn("Ignoring anchor pheno since anchor gene set was specified")
        anchor_gene_mask = None
        anchor_pheno_mask = None
        state._record_params({"anchor": "gene set"})

    # Record the effective anchor masks after option precedence is resolved.
    state.anchor_pheno_mask = np.copy(anchor_pheno_mask) if anchor_pheno_mask is not None else None
    state.anchor_gene_mask = np.copy(anchor_gene_mask) if anchor_gene_mask is not None else None

    #ensure at most one anchor mask, and initialize the matrix mask accordingly
    #remember that single pheno anchoring mode is implicit and doesn't have the anchor mask defined
    num_users = 1
    anchor_mask = None
    factor_gene_set_x_pheno = False
    pheno_Y = None

    if anchor_gene_mask is not None or anchor_gene_set:
        if anchor_pheno_mask is not None:
            warn("Ignoring anchor pheno since anchor gene or anchor gene set was specified")
            anchor_pheno_mask = None
        gene_or_pheno_mask = np.full(state.X_phewas_beta.shape[0], True)
        gene_set_mask = np.full(state.X_phewas_beta.shape[1], True)
        factor_gene_set_x_pheno = True

        combined_prior_Ys = state.gene_pheno_combined_prior_Ys.T if state.gene_pheno_combined_prior_Ys is not None else None
        priors = state.gene_pheno_priors.T if state.gene_pheno_priors is not None else None
        Y = state.gene_pheno_Y.T if state.gene_pheno_Y is not None else None

        state._record_params({"factor_gene_vectors": "gene_pheno.T"})

        if anchor_gene_mask is not None:
            betas = None
            betas_uncorrected = None

            anchor_mask = anchor_gene_mask
            num_users = np.sum(anchor_mask)
            state._record_params({"factor_gene_set_vectors": "None"})

        else:
            #we need to set things up below
            #we are going to construct a pheno x gene set matrix, using the X_phewas as input
            #we need to have weights for the rows (phenos) and columns (gene sets)
            #the column weights need to be the betas

            anchor_gene_mask = np.full(1, True)
            anchor_mask = anchor_gene_mask
            num_users = 1

            #for the gene set mode, we use the pheno_Y for weights, and do a special setting below
            #we need to keep combined_prior_Y for projecting, but use pheno_Y for weighting
            pheno_Y = state.pheno_Y_vs_input_combined_prior_Ys_beta if state.pheno_Y_vs_input_combined_prior_Ys_beta is not None else state.pheno_Y_vs_input_Y_beta if state.pheno_Y_vs_input_Y_beta is not None else state.pheno_Y_vs_input_priors_beta
            if pheno_Y is not None:
                pheno_Y = pheno_Y[:,np.newaxis]
            
            #betas are in external units
            betas = (state.betas / state.scale_factors)[:,np.newaxis] if state.betas is not None else None
            betas_uncorrected = (state.betas_uncorrected / state.scale_factors)[:,np.newaxis] if state.betas_uncorrected is not None else None
            state._record_params({"factor_gene_set_vectors": "betas"})

    else:
        if anchor_pheno_mask is not None and anchor_gene_mask is not None:
            warn("Ignoring anchor gene since anchor pheno was specified")
        anchor_gene_mask = None
        gene_or_pheno_mask = np.full(state.X_orig.shape[0], True)
        gene_set_mask = np.full(state.X_orig.shape[1], True)
        if anchor_pheno_mask is not None:

            anchor_mask = anchor_pheno_mask

            combined_prior_Ys = state.gene_pheno_combined_prior_Ys
            priors = state.gene_pheno_priors
            Y = state.gene_pheno_Y

            state._record_params({"factor_gene_vectors": "gene_pheno"})
            betas = state.X_phewas_beta.T if state.X_phewas_beta is not None else None
            betas_uncorrected = state.X_phewas_beta_uncorrected.T if state.X_phewas_beta_uncorrected is not None else None
            state._record_params({"factor_gene_set_vectors": "X_phewas"})

        else:

            combined_prior_Ys = state.combined_prior_Ys[:,np.newaxis] if state.combined_prior_Ys is not None else None
            priors = state.priors[:,np.newaxis] if state.priors is not None else None
            Y = state.Y[:,np.newaxis] if state.Y is not None else None

            state._record_params({"factor_gene_vectors": "Y"})

            betas = (state.betas / state.scale_factors)[:,np.newaxis] if state.betas is not None else None
            betas_uncorrected = (state.betas_uncorrected / state.scale_factors)[:,np.newaxis] if state.betas_uncorrected is not None else None

            state._record_params({"factor_gene_set_vectors": "betas"})


            #when running the original factoring based off the internal betas and gene scores, we are going to emulate the phewas-like behavior by appending these as the only anchor alongside any gene/pheno loaded values
            #this will allow projection to other phenotypes to happen naturally below
            anchor_mask = np.full(1, True)

            have_phewas = False
            if combined_prior_Ys is not None and state.gene_pheno_combined_prior_Ys is not None:
                combined_prior_Ys = sparse.hstack((state.gene_pheno_combined_prior_Ys, sparse.csc_matrix(combined_prior_Ys))).tocsc()
                have_phewas = True
            if priors is not None and state.gene_pheno_priors is not None:
                priors = sparse.hstack((state.gene_pheno_priors, sparse.csc_matrix(priors))).tocsc()
                have_phewas = True
            if Y is not None and state.gene_pheno_Y is not None:
                Y = sparse.hstack((state.gene_pheno_Y, sparse.csc_matrix(Y))).tocsc()
                have_phewas = True

            if betas is not None and state.X_phewas_beta is not None:
                betas = sparse.hstack((state.X_phewas_beta.T, sparse.csc_matrix(betas))).tocsc()
                have_phewas = True
            if betas_uncorrected is not None and state.X_phewas_beta_uncorrected is not None:
                betas_uncorrected = sparse.hstack((state.X_phewas_beta_uncorrected.T, sparse.csc_matrix(betas_uncorrected))).tocsc()
                have_phewas = True

            if have_phewas:
                #we have phewas for at least one of combined, prior, or Y
                #set those that don't to None
                #otherwise update the internal structures
                if combined_prior_Ys is not None and combined_prior_Ys.shape[1] == 1:
                    combined_prior_Ys = None
                else:
                    state.gene_pheno_combined_prior_Ys = combined_prior_Ys
                    
                if priors is not None and priors.shape[1] == 1:
                    priors = None
                else:
                    state.gene_pheno_priors = priors

                if Y is not None and Y.shape[1] == 1:
                    Y = None
                else:
                    state.gene_pheno_Y = Y
                if betas is not None and betas.shape[1] == 1:
                    betas = None
                else:
                    state.X_phewas_beta = betas.T
                if betas_uncorrected is not None and betas_uncorrected.shape[1] == 1:
                    betas_uncorrected = None
                else:
                    state.X_phewas_beta_uncorrected = betas_uncorrected.T

                state.default_pheno = "input_gene_stats"
                state.phenos.append(state.default_pheno)
                state.default_pheno_mask = np.full(len(state.phenos), False, dtype=bool)
                state.default_pheno_mask[-1] = True
                anchor_pheno_mask = state.default_pheno_mask
                state.anchor_pheno_mask = np.copy(anchor_pheno_mask)
                state._record_params(
                    {
                        "anchor_mode": "default_stats",
                        "anchor_pheno_mask_present": True,
                        "anchor_pheno_mask_total": int(len(anchor_pheno_mask)),
                        "anchor_pheno_mask_selected": int(np.sum(anchor_pheno_mask)),
                    },
                    overwrite=True,
                )

                #we need to update these as well
                state.pheno_Y_vs_input_Y_beta = np.append(state.pheno_Y_vs_input_Y_beta, 0) if state.pheno_Y_vs_input_Y_beta is not None else None
                state.pheno_Y_vs_input_Y_beta_tilde = np.append(state.pheno_Y_vs_input_Y_beta_tilde, 0) if state.pheno_Y_vs_input_Y_beta_tilde is not None else None
                state.pheno_Y_vs_input_Y_se = np.append(state.pheno_Y_vs_input_Y_se, 0) if state.pheno_Y_vs_input_Y_se is not None else None
                state.pheno_Y_vs_input_Y_Z = np.append(state.pheno_Y_vs_input_Y_Z, 0) if state.pheno_Y_vs_input_Y_Z is not None else None
                state.pheno_Y_vs_input_Y_p_value = np.append(state.pheno_Y_vs_input_Y_p_value, 1) if state.pheno_Y_vs_input_Y_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_Y_beta = np.append(state.pheno_combined_prior_Ys_vs_input_Y_beta, 0) if state.pheno_combined_prior_Ys_vs_input_Y_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_se = np.append(state.pheno_combined_prior_Ys_vs_input_Y_se, 0) if state.pheno_combined_prior_Ys_vs_input_Y_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_Z = np.append(state.pheno_combined_prior_Ys_vs_input_Y_Z, 0) if state.pheno_combined_prior_Ys_vs_input_Y_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_Y_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_Y_p_value is not None else None

                state.pheno_Y_vs_input_combined_prior_Ys_beta = np.append(state.pheno_Y_vs_input_combined_prior_Ys_beta, 0) if state.pheno_Y_vs_input_combined_prior_Ys_beta is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde = np.append(state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde, 0) if state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_se = np.append(state.pheno_Y_vs_input_combined_prior_Ys_se, 0) if state.pheno_Y_vs_input_combined_prior_Ys_se is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_Z = np.append(state.pheno_Y_vs_input_combined_prior_Ys_Z, 0) if state.pheno_Y_vs_input_combined_prior_Ys_Z is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_p_value = np.append(state.pheno_Y_vs_input_combined_prior_Ys_p_value, 1) if state.pheno_Y_vs_input_combined_prior_Ys_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value is not None else None

                state.pheno_Y_vs_input_priors_beta = np.append(state.pheno_Y_vs_input_priors_beta, 0) if state.pheno_Y_vs_input_priors_beta is not None else None
                state.pheno_Y_vs_input_priors_beta_tilde = np.append(state.pheno_Y_vs_input_priors_beta_tilde, 0) if state.pheno_Y_vs_input_priors_beta_tilde is not None else None
                state.pheno_Y_vs_input_priors_se = np.append(state.pheno_Y_vs_input_priors_se, 0) if state.pheno_Y_vs_input_priors_se is not None else None
                state.pheno_Y_vs_input_priors_Z = np.append(state.pheno_Y_vs_input_priors_Z, 0) if state.pheno_Y_vs_input_priors_Z is not None else None
                state.pheno_Y_vs_input_priors_p_value = np.append(state.pheno_Y_vs_input_priors_p_value, 1) if state.pheno_Y_vs_input_priors_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_priors_beta = np.append(state.pheno_combined_prior_Ys_vs_input_priors_beta, 0) if state.pheno_combined_prior_Ys_vs_input_priors_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_se = np.append(state.pheno_combined_prior_Ys_vs_input_priors_se, 0) if state.pheno_combined_prior_Ys_vs_input_priors_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_Z = np.append(state.pheno_combined_prior_Ys_vs_input_priors_Z, 0) if state.pheno_combined_prior_Ys_vs_input_priors_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_priors_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_priors_p_value is not None else None

                if combined_prior_Ys is None and priors is None and Y is None:
                    bail("Need to load gene phewas stats if you are loading gene set phewas stats")
                if betas is None and betas_uncorrected is None:
                    bail("Need to load gene set phewas stats if you are loading gene phewas stats")
                
            #the newly appended ones are not anchors
            anchor_mask = np.append(np.full((combined_prior_Ys.shape[1] if combined_prior_Ys is not None else priors.shape[1] if priors is not None else Y.shape[1] if Y is not None else 1) - 1, False), anchor_mask)


        num_users = np.sum(anchor_pheno_mask)

    #get one dimensional vectors with probabilities
    preferred_anchor_source = None
    if gene_or_pheno_filter_type in set(["combined_prior_Ys", "gene_phewas_combined"]):
        preferred_anchor_source = "combined_prior_Ys"
    elif gene_or_pheno_filter_type == "priors":
        preferred_anchor_source = "priors"
    elif gene_or_pheno_filter_type == "Y":
        preferred_anchor_source = "Y"
    gene_or_pheno_full_vector, gene_or_pheno_filter_type = _choose_gene_or_pheno_anchor_source(
        combined_prior_Ys,
        priors,
        Y,
        preferred_label=preferred_anchor_source,
        log_fn=log,
        info_level=INFO,
    )

    gene_or_pheno_vector = None
    if anchor_gene_set:
        gene_or_pheno_vector = pheno_Y
    else:
        if gene_or_pheno_full_vector is not None:
            gene_or_pheno_vector = gene_or_pheno_full_vector[:,anchor_mask]

    if gene_or_pheno_vector is not None:
        if sparse.issparse(gene_or_pheno_vector):
            gene_or_pheno_vector = gene_or_pheno_vector.toarray()

    #now get the aggregations and masks
    gene_or_pheno_max_vector = np.max(gene_or_pheno_vector, axis=1) if gene_or_pheno_vector is not None else None

    if gene_or_pheno_max_vector is not None and gene_or_pheno_filter_value is not None:
        gene_or_pheno_mask = gene_or_pheno_max_vector > gene_or_pheno_filter_value

    if pheno_prune_value is not None or pheno_prune_number is not None:
        mask_for_pruning = gene_or_pheno_mask if factor_gene_set_x_pheno else anchor_pheno_mask
        if mask_for_pruning is not None:
        
            if factor_gene_set_x_pheno:
                log("Pruning phenos to reduce matrix size", DEBUG)
            else:
                log("Pruning phenos to reduce number of anchors", DEBUG)                    

            pheno_sort_rank = -state.X_phewas_beta.mean(axis=1).A1 if state.X_phewas_beta is not None else np.arange(len(mask_for_pruning))
            #now if we request pruning
            if pheno_prune_value is not None:
                pheno_prune_mask = state._prune_gene_sets(pheno_prune_value, X_orig=state.X_phewas_beta_uncorrected[mask_for_pruning,:].T, gene_sets=[state.phenos[i] for i in np.where(mask_for_pruning)[0]], rank_vector=pheno_sort_rank[mask_for_pruning], do_internal_pruning=False)
                log("Found %d phenos remaining after pruning (of %d)" % (np.sum(pheno_prune_mask), len(state.phenos)))

                mask_for_pruning[np.where(mask_for_pruning)[0][~pheno_prune_mask]] = False

            if pheno_prune_number is not None:
                (mean_shifts, scale_factors) = state._calc_X_shift_scale(state.X_phewas_beta_uncorrected[mask_for_pruning,:].T)
                pheno_prune_number_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_phewas_beta_uncorrected[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=pheno_sort_rank[mask_for_pruning], stop_at=pheno_prune_number, tag="phenos")
                all_pheno_prune_mask = _combine_prune_masks(
                    pheno_prune_number_masks,
                    pheno_prune_number,
                    pheno_sort_rank[mask_for_pruning],
                    "pheno",
                    log_fn=log,
                    trace_level=TRACE,
                )
                mask_for_pruning[np.where(mask_for_pruning)[0][~all_pheno_prune_mask]] = False
            if mask_for_pruning is anchor_pheno_mask and num_users > 1:
                #in this case, we may have changed the number of users
                num_users = np.sum(anchor_pheno_mask)

    if not anchor_gene_set and (gene_prune_value is not None or gene_prune_number is not None):
        mask_for_pruning = gene_or_pheno_mask if not factor_gene_set_x_pheno else anchor_gene_mask
        if mask_for_pruning is not None:
            gene_sort_rank_source = gene_or_pheno_full_vector
            if gene_sort_rank_source is not None:
                gene_sort_rank = -_row_max_rank_values(gene_sort_rank_source)
            else:
                gene_sort_rank = np.arange(len(mask_for_pruning))
            if not factor_gene_set_x_pheno:
                log("Pruning genes to reduce matrix size", DEBUG)
            else:
                log("Pruning genes to reduce number of anchors", DEBUG)                    


            #now if we request pruning
            if gene_prune_value is not None:
                gene_prune_mask = state._prune_gene_sets(gene_prune_value, X_orig=state.X_orig[mask_for_pruning,:].T, gene_sets=[state.genes[i] for i in np.where(mask_for_pruning)[0]], rank_vector=gene_sort_rank[mask_for_pruning], do_internal_pruning=False)
                log("Found %d genes remaining after pruning (of %d)" % (np.sum(gene_prune_mask), len(state.genes)))

                mask_for_pruning[np.where(mask_for_pruning)[0][~gene_prune_mask]] = False

            if gene_prune_number is not None:
                if np.any(mask_for_pruning):
                    if str(discovery_model) == "gene_by_gene":
                        ranked_local = np.argsort(gene_sort_rank[mask_for_pruning], kind="mergesort")
                        all_gene_prune_mask = np.zeros(int(np.sum(mask_for_pruning)), dtype=bool)
                        all_gene_prune_mask[ranked_local[: int(gene_prune_number)]] = True
                    else:
                        (mean_shifts, scale_factors) = state._calc_X_shift_scale(state.X_orig[mask_for_pruning,:].T)
                        gene_prune_number_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_orig[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=gene_sort_rank[mask_for_pruning], stop_at=gene_prune_number, tag="genes")
                        all_gene_prune_mask = _combine_prune_masks(
                            gene_prune_number_masks,
                            gene_prune_number,
                            gene_sort_rank[mask_for_pruning],
                            "gene",
                            log_fn=log,
                            trace_level=TRACE,
                        )
                    if all_gene_prune_mask is not None:
                        mask_for_pruning[np.where(mask_for_pruning)[0][~all_gene_prune_mask]] = False
                else:
                    log("Skipping search-only gene pruning because no genes passed the current filter", DEBUG)

            if mask_for_pruning is anchor_gene_mask and num_users > 1:
                #in this case, we may have changed the number of users
                num_users = np.sum(anchor_gene_mask)

    #add in the any vectors
    gene_or_pheno_full_prob_vector = None
    if gene_or_pheno_full_vector is not None:
        #we are going to approximate things below the threshold as zero probability, and not fold those in the background prior
        #to get around this we would have to use a dense matrix
        if sparse.issparse(gene_or_pheno_full_vector):
            gene_or_pheno_full_prob_vector_data = np.exp(gene_or_pheno_full_vector.data + state.background_log_bf)
            gene_or_pheno_full_prob_vector_data = gene_or_pheno_full_prob_vector_data / (1 + gene_or_pheno_full_prob_vector_data)
            gene_or_pheno_full_prob_vector = copy.copy(gene_or_pheno_full_vector)
            gene_or_pheno_full_prob_vector.data = gene_or_pheno_full_prob_vector_data
        else:
            gene_or_pheno_full_prob_vector = np.exp(gene_or_pheno_full_vector + state.background_log_bf) / (1 + np.exp(gene_or_pheno_full_vector + state.background_log_bf))
        gene_or_pheno_full_prob_vector = _sanitize_dense_or_sparse_nonnegative_probabilities(gene_or_pheno_full_prob_vector)

    if anchor_gene_set:
        gene_or_pheno_prob_vector = np.exp(gene_or_pheno_vector + state.background_log_bf) / (1 + np.exp(gene_or_pheno_vector + state.background_log_bf)) if gene_or_pheno_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))
    else:
        gene_or_pheno_prob_vector = gene_or_pheno_full_prob_vector[:,anchor_mask] if gene_or_pheno_full_prob_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))
    gene_or_pheno_prob_vector = _sanitize_dense_or_sparse_nonnegative_probabilities(gene_or_pheno_prob_vector)

    if gene_or_pheno_prob_vector is not None and sparse.issparse(gene_or_pheno_prob_vector):
        gene_or_pheno_prob_vector = gene_or_pheno_prob_vector.toarray()

    if anchor_any_gene or anchor_any_pheno:
        #only have one user
        gene_or_pheno_any_prob_vector = 1 - np.prod(1 - gene_or_pheno_prob_vector, axis=1)
        gene_or_pheno_prob_vector = gene_or_pheno_any_prob_vector[:,np.newaxis]

    if factor_gene_set_x_pheno:
        state.pheno_prob_factor_vector = gene_or_pheno_prob_vector
        state.gene_prob_factor_vector = None
    else:
        state.gene_prob_factor_vector = gene_or_pheno_prob_vector
        state.pheno_prob_factor_vector = None

    #now do the gene set vectors and masks
    #normalize
    gene_set_full_vector = betas_uncorrected if betas_uncorrected is not None else betas
    gene_set_vector = None
    if gene_set_full_vector is not None:
        gene_set_vector = gene_set_full_vector[:,anchor_mask]
        if sparse.issparse(gene_set_vector):
            gene_set_vector = gene_set_vector.toarray()

    gene_set_filter_type = "betas_uncorrected" if betas_uncorrected is not None else "betas"
    gene_set_max_vector = np.max(gene_set_vector, axis=1) if gene_set_vector is not None else None

    if gene_set_max_vector is not None and gene_set_filter_value is not None:
        gene_set_mask = gene_set_max_vector > gene_set_filter_value

    if not np.any(gene_set_mask):
        bail(
            "No gene sets remained after factor gene-set filtering; relax --gene-set-filter-value or adjust the input gene-set statistics thresholds"
        )

    gene_set_sort_rank = _gene_set_sort_rank_for_pruning(
        state,
        betas=betas,
        betas_uncorrected=betas_uncorrected,
    )

    if gene_set_prune_value is not None:
        warn("--factor-prune-gene-sets-val is deprecated and ignored for discovery selection")
    if gene_set_prune_number is not None and max_num_discovery_gene_sets is None:
        warn("--factor-prune-gene-sets-num is deprecated; mapping it to --max-num-discovery-gene-sets")
        max_num_discovery_gene_sets = int(gene_set_prune_number)
    if not auto_discovery_subset and discovery_redundancy_weighting_mode != "none":
        warn(
            "discovery_redundancy_weighting_mode=%s is not supported with --no-auto-discovery-subset; falling back to none"
            % discovery_redundancy_weighting_mode
        )
        discovery_redundancy_weighting_mode = "none"
        discovery_redundancy_weighting = False
    
    gene_set_full_prob_vector = None
    if gene_set_full_vector is not None:
        if sparse.issparse(gene_set_full_vector):
            gene_set_full_prob_vector_data = np.exp(gene_set_full_vector.data + state.background_log_bf)
            gene_set_full_prob_vector_data = gene_set_full_prob_vector_data / (1 + gene_set_full_prob_vector_data)
            gene_set_full_prob_vector = copy.copy(gene_set_full_vector)
            gene_set_full_prob_vector.data = gene_set_full_prob_vector_data
        else:
            gene_set_full_prob_vector = np.exp(gene_set_full_vector + state.background_log_bf) / (1 + np.exp(gene_set_full_vector + state.background_log_bf))

    gene_set_prob_vector = gene_set_full_prob_vector[:,anchor_mask] if gene_set_full_prob_vector is not None else np.ones((len(gene_set_mask), num_users))

    if gene_set_prob_vector is not None and sparse.issparse(gene_set_prob_vector):
        gene_set_prob_vector = gene_set_prob_vector.toarray()

    if anchor_any_gene or anchor_any_pheno:
        #only have one user
        gene_set_any_prob_vector = 1 - np.prod(1 - gene_set_prob_vector, axis=1)
        gene_set_prob_vector = gene_set_any_prob_vector[:,np.newaxis]

    state.gene_set_prob_vector = gene_set_prob_vector

    anchor_trait_names = []
    if anchor_pheno_mask is not None and getattr(state, "phenos", None) is not None:
        anchor_trait_names = [str(state.phenos[i]) for i in np.where(np.asarray(anchor_pheno_mask, dtype=bool))[0] if i < len(state.phenos)]
    elif anchor_mask is not None:
        anchor_trait_names = ["anchor_%d" % int(i + 1) for i in range(int(np.sum(anchor_mask)))]
    state._record_params(
        {
            "anchor_aggregation": str(anchor_aggregation),
            "num_anchor_traits": int(gene_set_prob_vector.shape[1]) if gene_set_prob_vector is not None and np.ndim(gene_set_prob_vector) == 2 else int(num_users),
            "anchor_trait_names": ",".join(anchor_trait_names),
        },
        overwrite=True,
    )

    if str(discovery_model) == "gene_by_gene":
        if factor_gene_set_x_pheno:
            bail("--discovery-model gene_by_gene is currently supported only for gene-space discovery, not rectangular phenotype-input matrix factoring")
        if max_num_discovery_gene_sets is not None:
            warn("--max-num-discovery-gene-sets is ignored for --discovery-model gene_by_gene")
        if not auto_discovery_subset:
            warn("--no-auto-discovery-subset has no effect for --discovery-model gene_by_gene because all retained gene sets are used")
        if discovery_redundancy_weighting_mode != "effective_size" or not discovery_redundancy_weighting:
            warn("Discovery-family redundancy weighting controls are ignored for --discovery-model gene_by_gene")

        retained_row_indices = np.where(gene_set_mask)[0]
        discovery_plan = DiscoveryPlan(
            retained_gene_set_mask_full=np.asarray(gene_set_mask, dtype=bool),
            in_discovery_mask_full=np.asarray(gene_set_mask, dtype=bool),
            discovery_family_id_full=np.full(len(gene_set_mask), -1, dtype=int),
            discovery_representative_mask_full=np.asarray(gene_set_mask, dtype=bool),
            discovery_family_size_full=np.where(np.asarray(gene_set_mask, dtype=bool), 1, 0),
            discovery_family_mean_similarity_full=np.zeros(len(gene_set_mask), dtype=float),
            discovery_family_effective_size_full=np.where(np.asarray(gene_set_mask, dtype=bool), 1.0, 0.0),
            discovery_weight_full=np.where(np.asarray(gene_set_mask, dtype=bool), 1.0, 0.0),
            discovery_prob_vector=np.asarray(gene_set_prob_vector[gene_set_mask, :], dtype=float),
            discovery_row_indices_full=np.asarray(retained_row_indices, dtype=int),
            retained_row_indices_full=np.asarray(retained_row_indices, dtype=int),
        )
        pair_effective_size = (
            gene_gene_pair_prior_effective_size
            if gene_gene_pair_prior_effective_size is not None
            else learn_phi_target_gene_effective_support
        )
        ordered_active_gene_indices = _canonical_gene_gene_active_indices(state, gene_or_pheno_mask)
        pair_prior, pair_prior_source = _resolve_gene_gene_pair_prior(
            int(np.sum(gene_or_pheno_mask)),
            explicit_pair_prior=gene_gene_pair_prior,
            effective_size=pair_effective_size,
            background_prior=getattr(state, "background_prior", None),
        )
        pair_beta_values = None
        if anchor_pheno_mask is not None:
            pair_beta_values = betas
            if pair_beta_values is None:
                bail("Requested gene-by-gene discovery but corrected beta gene-set statistics are unavailable")
            pair_beta_values = pair_beta_values[:, anchor_mask]
            if sparse.issparse(pair_beta_values):
                pair_beta_values = pair_beta_values.toarray()
        pair_matrix, pair_diagnostics, pair_weights = _build_gene_gene_pair_matrix(
            state,
            gene_mask=gene_or_pheno_mask,
            gene_indices=ordered_active_gene_indices,
            gene_set_mask=gene_set_mask,
            beta_source=gene_gene_beta_source,
            beta_values=pair_beta_values,
            beta_values_scaled=pair_beta_values is not None,
            gene_prob_values=None,
            pair_prior=pair_prior,
            logbf_base=gene_gene_logbf_base,
            anchor_aggregation=anchor_aggregation,
            matrix_floor=gene_gene_matrix_floor,
            excess_probability=gene_gene_excess_probability,
            diagonal_weight=gene_gene_diagonal_weight,
            profligate_correction=gene_gene_profligate_correction,
            log_fn=log,
            info_level=INFO,
        )
        state._record_params(
            {
                "max_num_factors": max_num_factors,
                "alpha0": alpha0,
                "phi": phi,
                "discovery_model": str(discovery_model),
                "gene_set_filter_type": gene_set_filter_type,
                "gene_set_filter_value": gene_set_filter_value,
                "gene_or_pheno_filter_type": gene_or_pheno_filter_type,
                "gene_or_pheno_filter_value": gene_or_pheno_filter_value,
                "pheno_prune_value": pheno_prune_value,
                "pheno_prune_number": pheno_prune_number,
                "gene_set_prune_value": gene_set_prune_value,
                "gene_set_prune_number": gene_set_prune_number,
                "max_num_discovery_gene_sets": None,
                "auto_discovery_subset": False,
                "discovery_redundancy_weighting": False,
                "discovery_redundancy_weighting_mode": "none",
                "discovery_similarity_threshold": discovery_similarity_threshold,
                "factor_output_scope": factor_output_scope,
                "factor_primary_mass_floor": _PRIMARY_FACTOR_MASS_FLOOR,
                "factor_secondary_mass_floor": _SECONDARY_FACTOR_MASS_FLOOR,
                "num_retained_gene_sets": int(np.sum(gene_set_mask)),
                "num_discovery_gene_sets": int(np.sum(gene_set_mask)),
                "run_transpose": run_transpose,
                "gene_gene_beta_source": str(gene_gene_beta_source),
                "gene_gene_pair_prior": float(pair_prior),
                "gene_gene_pair_prior_source": str(pair_prior_source),
                "gene_gene_pair_prior_effective_size": None if pair_effective_size is None else float(pair_effective_size),
                "gene_gene_canonical_gene_order": "gene_id",
                "gene_gene_logbf_base": str(gene_gene_logbf_base),
                "anchor_aggregation": str(anchor_aggregation),
                "gene_gene_diagonal_weight": float(gene_gene_diagonal_weight),
                "gene_gene_matrix_floor": float(gene_gene_matrix_floor),
                "gene_gene_excess_probability": bool(gene_gene_excess_probability),
                "gene_gene_row_sum_cap": bool(gene_gene_row_sum_cap),
                "gene_gene_sparsity": float(gene_gene_sparsity),
                                "gene_gene_active_genes": int(pair_diagnostics["num_active_genes"]),
                "gene_gene_retained_gene_sets": int(pair_diagnostics["num_retained_gene_sets"]),
                "gene_gene_beta_positive_count": int(pair_diagnostics["beta_positive_count"]),
                "gene_gene_beta_negative_count": int(pair_diagnostics["beta_negative_count"]),
                "gene_gene_beta_anchor_count": int(pair_diagnostics["beta_anchor_count"]),
                "gene_gene_L_mean": float(pair_diagnostics["L_mean"]),
                "gene_gene_L_std": float(pair_diagnostics["L_std"]),
                "gene_gene_L_min": float(pair_diagnostics["L_min"]),
                "gene_gene_L_max": float(pair_diagnostics["L_max"]),
                "gene_gene_P_mean": float(pair_diagnostics["P_mean"]),
                "gene_gene_P_std": float(pair_diagnostics["P_std"]),
                "gene_gene_P_min": float(pair_diagnostics["P_min"]),
                "gene_gene_P_max": float(pair_diagnostics["P_max"]),
                "gene_gene_M_density": float(pair_diagnostics["M_density"]),
                "gene_gene_M_mean": float(pair_diagnostics["M_mean"]),
                "gene_gene_M_std": float(pair_diagnostics["M_std"]),
                "gene_gene_M_min": float(pair_diagnostics["M_min"]),
                "gene_gene_M_max": float(pair_diagnostics["M_max"]),
                "gene_gene_target_mode": str(pair_diagnostics["gene_gene_target_mode"]),
                "gene_gene_pair_weights_used": bool(pair_diagnostics["gene_gene_pair_weights_used"]),
                "gene_gene_pair_gate": "none",
                "num_anchor_traits": int(pair_diagnostics["num_anchor_traits"]),
                "gene_gene_per_anchor_nonzero_edge_counts": str(pair_diagnostics["per_anchor_nonzero_edge_counts"]),
                "gene_gene_per_anchor_mean_target": str(pair_diagnostics["per_anchor_mean_target"]),
                "gene_gene_M_row_sum_median": float(pair_diagnostics["M_row_sum_median"]),
                "gene_gene_M_row_sum_q90": float(pair_diagnostics["M_row_sum_q90"]),
                "gene_gene_M_row_sum_max": float(pair_diagnostics["M_row_sum_max"]),
                "gene_gene_M_row_sum_mean": float(pair_diagnostics["M_row_sum_mean"]),
            }
        )
        log("Running symmetric gene-by-gene factorization", INFO)
        log("Pair matrix to factor shape: (%s, %s)" % (pair_matrix.shape), DEBUG)
        if np.max(pair_matrix.shape) == 0 or np.min(pair_matrix.shape) == 0:
            log("Skipping factoring since there aren't enough retained genes or gene sets")
            return

        normalize_genes = False
        normalize_gene_sets = False
        cap = True
        result = state._bayes_sym_nmf_l2_extension(
            pair_matrix,
            K0=max_num_factors,
            phi=phi,
            a0=alpha0,
            max_iter=max_num_iterations,
            rel_tol=rel_tol,
            min_lambda_threshold=min_lambda_threshold,
            seed=seed,
            row_sum_cap=gene_gene_row_sum_cap,
            sparsity=gene_gene_sparsity,
            diagonal_weight=gene_gene_diagonal_weight,
            pair_weights=None,
            log_fn=log,
        )
        state.last_factorization_backend = "symnmf_gene_by_gene"
        state.last_factorization_backend_details = {
            "backend": "symnmf_gene_by_gene",
            "delegated_to_full_solver": False,
            "num_blocks": 1,
            "block_size": int(pair_matrix.shape[0]),
            "epochs": int(max_num_iterations),
            "columns_evaluated": int(pair_matrix.shape[0]),
            "warm_started": False,
            "gene_gene_pair_weights_used": False,
            "gene_gene_target_mode": str(pair_diagnostics.get("gene_gene_target_mode", "")),
            "lambda_cut": getattr(state, "last_factorization_lambda_cut", None),
            "epoch_error_trace": [],
            "wall_time_sec": None,
        }
        state.exp_lambdak = np.asarray(result[3], dtype=float)
        fit_gene_factors = np.asarray(result[0], dtype=float)
        state.exp_gene_set_factors = state._project_H_with_fixed_W(
            fit_gene_factors,
            state.X_orig[ordered_active_gene_indices, :][:, gene_set_mask],
            gene_or_pheno_prob_vector[ordered_active_gene_indices, :],
            gene_set_prob_vector[gene_set_mask, :],
            phi=phi,
            tol=rel_tol,
            cap_genes=cap,
            normalize_genes=normalize_gene_sets,
        )
        exp_gene_or_pheno_factors = np.zeros((len(state.genes), fit_gene_factors.shape[1]), dtype=float)
        exp_gene_or_pheno_factors[ordered_active_gene_indices, :] = fit_gene_factors
        evidence_value = result[2]
        likelihood_value = result[1]
        reconstruction_error_value = result[4]
        blockwise_warm_start_payload = None
    else:
        similarity_X_orig = state.X_orig[gene_or_pheno_mask, :]
        if hasattr(state, "_calc_X_shift_scale"):
            (similarity_mean_shifts, similarity_scale_factors) = state._calc_X_shift_scale(similarity_X_orig)
        else:
            similarity_mean_shifts = getattr(state, "mean_shifts", np.zeros(similarity_X_orig.shape[1], dtype=float))
            similarity_scale_factors = getattr(state, "scale_factors", np.ones(similarity_X_orig.shape[1], dtype=float))

        discovery_plan = _build_discovery_plan(
            state,
            retained_gene_set_mask_full=gene_set_mask,
            gene_set_sort_rank=gene_set_sort_rank,
            gene_set_prob_vector_full=gene_set_prob_vector,
            similarity_X_orig=similarity_X_orig,
            similarity_mean_shifts=similarity_mean_shifts,
            similarity_scale_factors=similarity_scale_factors,
            max_num_discovery_gene_sets=max_num_discovery_gene_sets,
            auto_discovery_subset=auto_discovery_subset,
            discovery_redundancy_weighting=discovery_redundancy_weighting,
            discovery_redundancy_weighting_mode=discovery_redundancy_weighting_mode,
            discovery_similarity_threshold=discovery_similarity_threshold,
        )

        annotation_weight_summary = _summarize_anchor_weight_surface(
            discovery_plan.discovery_prob_vector,
            gene_or_pheno_prob_vector[gene_or_pheno_mask, :],
            int(discovery_plan.discovery_row_indices_full.size),
            int(np.sum(gene_or_pheno_mask)),
            anchor_aggregation=anchor_aggregation,
        )
        state._record_params(annotation_weight_summary, overwrite=True)

        state._record_params({"max_num_factors": max_num_factors, "alpha0": alpha0, "phi": phi, "discovery_model": str(discovery_model), "anchor_aggregation": str(anchor_aggregation), "gene_set_filter_type": gene_set_filter_type, "gene_set_filter_value": gene_set_filter_value, "gene_or_pheno_filter_type": gene_or_pheno_filter_type, "gene_or_pheno_filter_value": gene_or_pheno_filter_value, "pheno_prune_value": pheno_prune_value, "pheno_prune_number": pheno_prune_number, "gene_set_prune_value": gene_set_prune_value, "gene_set_prune_number": gene_set_prune_number, "max_num_discovery_gene_sets": max_num_discovery_gene_sets, "auto_discovery_subset": auto_discovery_subset, "discovery_redundancy_weighting": discovery_redundancy_weighting, "discovery_redundancy_weighting_mode": discovery_redundancy_weighting_mode, "discovery_similarity_threshold": discovery_similarity_threshold, "factor_output_scope": factor_output_scope, "factor_primary_mass_floor": _PRIMARY_FACTOR_MASS_FLOOR, "factor_secondary_mass_floor": _SECONDARY_FACTOR_MASS_FLOOR, "num_retained_gene_sets": int(np.sum(gene_set_mask)), "num_discovery_gene_sets": int(discovery_plan.discovery_row_indices_full.size), "run_transpose": run_transpose})

        matrix = state.X_phewas_beta_uncorrected.T if factor_gene_set_x_pheno else state.X_orig.T
        matrix = matrix[discovery_plan.discovery_row_indices_full,:][:,gene_or_pheno_mask]
        matrix[matrix < 0] = 0
        if not run_transpose:
            matrix = matrix.T

        if factor_backend not in {"full", "blockwise_global_w"}:
            bail("Unknown factor backend '%s'" % factor_backend)
        if factor_backend == "blockwise_global_w" and not run_transpose:
            bail("--factor-backend blockwise_global_w currently requires the default transposed factor matrix")

        log("Running matrix factorization with backend=%s" % factor_backend)
        if np.sum(~gene_or_pheno_mask) > 0 or np.sum(~gene_set_mask) > 0 or discovery_plan.discovery_row_indices_full.size != int(np.sum(gene_set_mask)):
            log("Filtered original matrix from (%s, %s) to (%s, %s)" % (len(gene_or_pheno_mask), len(gene_set_mask), sum(gene_or_pheno_mask), discovery_plan.discovery_row_indices_full.size))
        log("Matrix to factor shape: (%s, %s)" % (matrix.shape), DEBUG)

        if np.max(matrix.shape) == 0:
            log("Skipping factoring since there aren't enough significant genes and gene sets")
            return

        if np.min(matrix.shape) == 0:
            log("Empty genes or gene sets! Skipping factoring")
            return

        normalize_genes = False
        normalize_gene_sets = False
        cap = True

        effective_block_count = int(
            min(
                math.ceil(float(matrix.shape[0]) / float(max(1, blockwise_gene_set_block_size))),
                int(blockwise_max_blocks) if blockwise_max_blocks is not None else math.ceil(float(matrix.shape[0]) / float(max(1, blockwise_gene_set_block_size))),
            )
        ) if matrix.shape[0] > 0 else 0

        delegated_single_block_to_full = bool(
            factor_backend == "blockwise_global_w" and effective_block_count <= 1
        )
        if delegated_single_block_to_full:
            log(
                "Delegating single-block blockwise_global_w run to the full solver for exact equivalence",
                INFO,
            )

        if factor_backend == "blockwise_global_w" and not delegated_single_block_to_full:
            blockwise_started_at = time.time()
            blockwise_pass_metrics_dir = _derive_blockwise_pass_metrics_dir(factor_metrics_out, blockwise_report_out)
            result = _fit_blockwise_global_w(
                state,
                matrix,
                gene_set_prob_vector=discovery_plan.discovery_prob_vector,
                gene_or_pheno_prob_vector=gene_or_pheno_prob_vector[gene_or_pheno_mask, :],
                anchor_aggregation=anchor_aggregation,
                max_num_factors=max_num_factors,
                max_num_iterations=max_num_iterations,
                alpha0=alpha0,
                phi=phi,
                rel_tol=rel_tol,
                min_lambda_threshold=min_lambda_threshold,
                block_size=blockwise_gene_set_block_size,
                epochs=blockwise_epochs,
                shuffle_blocks=blockwise_shuffle_blocks,
                max_blocks=blockwise_max_blocks,
                warm_start_state=blockwise_warm_start_state,
                warm_start_enabled=blockwise_warm_start,
                cap_genes=cap,
                cap_gene_sets=cap,
                report_out=blockwise_report_out,
                pass_metrics_dir=blockwise_pass_metrics_dir,
                log_fn=log,
                info_level=INFO,
            )
            state.last_factorization_backend = "blockwise_global_w"
            state.last_factorization_backend_details["wall_time_sec"] = float(time.time() - blockwise_started_at)
            state.exp_lambdak = np.asarray(result["lambdak"], dtype=float)
            exp_gene_or_pheno_factors = np.asarray(result["gene_or_pheno_factors"], dtype=float)
            state.exp_gene_set_factors = np.asarray(result["gene_set_factors"], dtype=float)
            evidence_value = result["evidence"]
            likelihood_value = result["likelihood"]
            reconstruction_error_value = result["reconstruction_error"]
            blockwise_warm_start_payload = result.get("warm_start_payload")
        else:
            result = state._bayes_nmf_l2_extension(
                matrix.toarray(),
                discovery_plan.discovery_prob_vector,
                gene_or_pheno_prob_vector[gene_or_pheno_mask,:],
                anchor_aggregation=anchor_aggregation,
                n_iter=max_num_iterations,
                a0=alpha0,
                K=max_num_factors,
                tol=rel_tol,
                phi=phi,
                cap_genes=cap,
                cap_gene_sets=cap,
                normalize_genes=normalize_genes,
                normalize_gene_sets=normalize_gene_sets,
            )
            state.last_factorization_backend = "blockwise_global_w" if delegated_single_block_to_full else "full"
            state.last_factorization_backend_details = {
                "backend": "blockwise_global_w" if delegated_single_block_to_full else "full",
                "delegated_to_full_solver": bool(delegated_single_block_to_full),
                "num_blocks": 1,
                "block_size": int(matrix.shape[0]),
                "epochs": int(max_num_iterations),
                "columns_evaluated": int(matrix.shape[0]),
                "warm_started": False,
                "lambda_cut": None,
                "epoch_error_trace": [],
                "wall_time_sec": None,
            }
            scale_details = getattr(state, "last_factorization_scale_details", None)
            if scale_details is not None:
                state.last_factorization_backend_details.update(
                    {
                        "raw_mean_v": float(scale_details.get("raw_mean", 0.0)),
                        "raw_std_v": float(scale_details.get("raw_std", 0.0)),
                        "weighted_mean_v": float(scale_details.get("weighted_mean", 0.0)),
                        "weighted_std_v": float(scale_details.get("weighted_std", 0.0)),
                        "row_scale_min": float(scale_details.get("row_scale_min", 0.0)),
                        "row_scale_median": float(scale_details.get("row_scale_median", 0.0)),
                        "row_scale_mean": float(scale_details.get("row_scale_mean", 0.0)),
                        "row_scale_max": float(scale_details.get("row_scale_max", 0.0)),
                    }
                )
            state.exp_lambdak = result[4]
            exp_gene_or_pheno_factors = result[1].T
            state.exp_gene_set_factors = result[0]
            evidence_value = result[3]
            likelihood_value = result[2]
            reconstruction_error_value = result[5]
            blockwise_warm_start_payload = None

    scale_details = getattr(state, "last_factorization_scale_details", None)
    if scale_details is not None:
        state._record_params(
            {
                "discovery_scale_raw_mean_v": float(scale_details.get("raw_mean", 0.0)),
                "discovery_scale_raw_std_v": float(scale_details.get("raw_std", 0.0)),
                "discovery_scale_weighted_mean_v": float(scale_details.get("weighted_mean", 0.0)),
                "discovery_scale_weighted_std_v": float(scale_details.get("weighted_std", 0.0)),
                "discovery_scale_row_weight_min": float(scale_details.get("row_scale_min", 0.0)),
                "discovery_scale_row_weight_median": float(scale_details.get("row_scale_median", 0.0)),
                "discovery_scale_row_weight_mean": float(scale_details.get("row_scale_mean", 0.0)),
                "discovery_scale_row_weight_max": float(scale_details.get("row_scale_max", 0.0)),
            },
            overwrite=True,
        )

    #subset_out the weak factors
    lambda_keep_threshold = float(min_lambda_threshold)
    if factor_backend == "blockwise_global_w":
        lambda_cut = (state.last_factorization_backend_details or {}).get("lambda_cut")
        if lambda_cut is not None and np.isfinite(lambda_cut):
            lambda_keep_threshold = min(float(min_lambda_threshold), float(lambda_cut))
            log(
                "Using backend-scale lambda retention threshold %.6g (min_lambda_threshold %.6g, lambda_cut %.6g)"
                % (lambda_keep_threshold, float(min_lambda_threshold), float(lambda_cut))
            )
    factor_mask = (state.exp_lambdak > lambda_keep_threshold) & (np.sum(exp_gene_or_pheno_factors, axis=0) > min_lambda_threshold) & (np.sum(state.exp_gene_set_factors, axis=0) > min_lambda_threshold)
    factor_mask = factor_mask & (np.max(state.exp_gene_set_factors, axis=0) > 1e-5 * np.max(state.exp_gene_set_factors))
    if np.sum(~factor_mask) > 0:
        state.exp_lambdak = state.exp_lambdak[factor_mask]
        exp_gene_or_pheno_factors = exp_gene_or_pheno_factors[:,factor_mask]
        state.exp_gene_set_factors = state.exp_gene_set_factors[:,factor_mask]

    if factor_gene_set_x_pheno:
        state.pheno_in_discovery_mask = gene_or_pheno_mask
        state.pheno_factor_pheno_mask = state.pheno_in_discovery_mask
        state.exp_pheno_factors = exp_gene_or_pheno_factors
        state.pheno_prob_factor_vector = gene_or_pheno_prob_vector
        state.gene_prob_factor_vector = None
    else:
        state.gene_in_discovery_mask = gene_or_pheno_mask
        state.gene_factor_gene_mask = state.gene_in_discovery_mask
        state.exp_gene_factors = exp_gene_or_pheno_factors
        state.gene_prob_factor_vector = gene_or_pheno_prob_vector
        state.pheno_prob_factor_vector = None

    state.gene_set_prob_factor_vector = gene_set_prob_vector
    state.gene_set_in_discovery_mask = discovery_plan.in_discovery_mask_full
    state.gene_set_factor_gene_set_mask = state.gene_set_in_discovery_mask
    state.gene_set_discovery_family_id = discovery_plan.discovery_family_id_full
    state.gene_set_discovery_representative_mask = discovery_plan.discovery_representative_mask_full
    state.gene_set_discovery_family_size = discovery_plan.discovery_family_size_full
    state.gene_set_discovery_weight = discovery_plan.discovery_weight_full
    state.gene_set_discovery_family_mean_similarity = discovery_plan.discovery_family_mean_similarity_full
    state.gene_set_discovery_family_effective_size = discovery_plan.discovery_family_effective_size_full

    #now project the additional genes/phenos/gene sets onto the factors

    log("Projecting factors", TRACE)

    #this code gets the "relevance" values
    #first get the probabilities for either the genotypes or phenotypes (whichever we didn't use to factor)
    #these need to be specific to the anchors
    if factor_gene_set_x_pheno:
        if gene_or_pheno_full_prob_vector is not None:
            state.gene_prob_factor_vector = state._nnls_project_matrix(state.pheno_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
            state._record_params({"factor_gene_prob_from": "phenos"})
        else:
            state.gene_prob_factor_vector = state._nnls_project_matrix(state.gene_set_prob_factor_vector, state.X_orig)
            state._record_params({"factor_gene_prob_from": "gene_sets"})
    else:
        if gene_or_pheno_full_prob_vector is not None:
            state.pheno_prob_factor_vector = state._nnls_project_matrix(state.gene_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
            state._record_params({"factor_pheno_prob_from": "genes"})
        elif state.X_phewas_beta_uncorrected is not None:
            state.pheno_prob_factor_vector = state._nnls_project_matrix(state.gene_set_prob_factor_vector, state.X_phewas_beta_uncorrected)
            state._record_params({"factor_pheno_prob_from": "gene_sets"})

    if state.gene_set_prob_factor_vector is not None and sparse.issparse(state.gene_set_prob_factor_vector):
        state.gene_set_prob_factor_vector = state.gene_set_prob_factor_vector.toarray()
    if state.gene_prob_factor_vector is not None and sparse.issparse(state.gene_prob_factor_vector):
        state.gene_prob_factor_vector = state.gene_prob_factor_vector.toarray()
    if state.pheno_prob_factor_vector is not None and sparse.issparse(state.pheno_prob_factor_vector):
        state.pheno_prob_factor_vector = state.pheno_prob_factor_vector.toarray()

    gene_matrix_to_project = state.X_orig.T
    if not run_transpose:
        gene_matrix_to_project = gene_matrix_to_project.T

    #this code projects to the additional dimensions

    #all gene factor values
    if _state_discovery_model(state) == "gene_by_gene" and not factor_gene_set_x_pheno:
        full_gene_factor_values = state.project_full_gene_factors_gene_by_gene(
            retained_discovery_gene_factors=state.exp_gene_factors[state.gene_in_discovery_mask, :],
            cap_genes=cap,
        )
        state._record_params({"factor_gene_projection_from": "direct_gene_gene"})
    else:
        full_gene_factor_values = state._project_H_with_fixed_W(state.exp_gene_set_factors, gene_matrix_to_project[state.gene_set_in_discovery_mask,:], state.gene_set_prob_factor_vector[state.gene_set_in_discovery_mask,:], state.gene_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
    if not factor_gene_set_x_pheno and keep_original_loadings:
        full_gene_factor_values[state.gene_in_discovery_mask,:] = state.exp_gene_factors

    #all pheno factor values, either from the phewas used to factor or the phewas passed in to project
    full_pheno_factor_values = state.exp_pheno_factors
    state.pheno_capture_strength = None
    state.pheno_capture_basis = None
    state.pheno_capture_input = None
    pheno_matrix_to_project = None

    if state.exp_gene_factors is None and state.exp_gene_set_factors is None:
        bail("Something went wrong: both gene factors and gene set factors are empty")

    run_legacy_pheno_capture = bool(no_trait_linkage)

    if (
        run_legacy_pheno_capture
        and state.X_phewas_beta_uncorrected is not None
        and state.pheno_prob_factor_vector is not None
    ):
        if project_phenos_from_gene_sets or state.exp_gene_factors is None:
            pheno_matrix_to_project = state.X_phewas_beta_uncorrected.T
            if not run_transpose:
                pheno_matrix_to_project = pheno_matrix_to_project.T
            basis = state.exp_gene_set_factors
            feature_by_pheno = _prepare_pheno_capture_input_matrix(pheno_matrix_to_project, pheno_capture_input)
            basis, feature_by_pheno = _align_projection_inputs_to_mask(
                basis,
                feature_by_pheno,
                state.gene_set_in_discovery_mask,
            )
            full_pheno_factor_values = _project_pheno_capture_matrix(
                state,
                basis,
                feature_by_pheno,
                basis_name="gene_sets",
            )
        else:
            pheno_matrix_to_project = state.gene_pheno_combined_prior_Ys if state.gene_pheno_combined_prior_Ys is not None else state.gene_pheno_Y
            if not run_transpose:
                pheno_matrix_to_project = pheno_matrix_to_project.T
            basis = state.exp_gene_factors
            feature_by_pheno = _prepare_pheno_capture_input_matrix(pheno_matrix_to_project, pheno_capture_input)
            basis, feature_by_pheno = _align_projection_inputs_to_mask(
                basis,
                feature_by_pheno,
                state.gene_in_discovery_mask,
            )
            full_pheno_factor_values = _project_pheno_capture_matrix(
                state,
                basis,
                feature_by_pheno,
                basis_name="genes",
            )

            
        if keep_original_loadings:
            full_pheno_factor_values[state.pheno_in_discovery_mask,:] = state.exp_pheno_factors
        state.pheno_capture_input = pheno_capture_input
    elif run_legacy_pheno_capture and state.exp_pheno_factors is not None:
        state.pheno_capture_basis = "native"
        if state.X_phewas_beta_uncorrected is not None:
            feature_by_pheno = state.X_phewas_beta_uncorrected.T
            if not run_transpose:
                feature_by_pheno = feature_by_pheno.T
            feature_by_pheno = _prepare_pheno_capture_input_matrix(feature_by_pheno, pheno_capture_input)
            state.pheno_capture_strength = eaggl_phenotype_annotation.compute_profile_strengths(feature_by_pheno)
            state.pheno_capture_input = pheno_capture_input
        else:
            state.pheno_capture_strength = np.sum(np.asarray(state.exp_pheno_factors, dtype=float), axis=1)

    #now gene set factor values, projecting from either phenos or genes depending on what was used
    if factor_gene_set_x_pheno and pheno_matrix_to_project is not None:
        #we have to swap the gene sets and genes, which means transposing the matrix to project and swapping the prios
        full_gene_set_factor_values = state._project_H_with_fixed_W(state.exp_pheno_factors, pheno_matrix_to_project[:,state.pheno_in_discovery_mask].T if run_transpose else pheno_matrix_to_project[state.pheno_in_discovery_mask,:].T, state.pheno_prob_factor_vector[state.pheno_in_discovery_mask,:], state.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)
    else:
        gene_basis_for_projection = (
            state.exp_gene_factors[state.gene_in_discovery_mask, :]
            if _state_discovery_model(state) == "gene_by_gene" and not factor_gene_set_x_pheno
            else state.exp_gene_factors
        )
        full_gene_set_factor_values = state._project_H_with_fixed_W(gene_basis_for_projection, gene_matrix_to_project[:,state.gene_in_discovery_mask].T if run_transpose else gene_matrix_to_project[state.gene_in_discovery_mask,:].T, state.gene_prob_factor_vector[state.gene_in_discovery_mask,:], state.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)

    if keep_original_loadings:
        full_gene_set_factor_values[state.gene_set_in_discovery_mask,:] = state.exp_gene_set_factors

    # Keep gene-by-gene discovery loadings as the canonical factor basis.  Full
    # gene projection is a separate optional output and must not change phi
    # metrics or standard gene_clusters semantics.
    if _state_discovery_model(state) == "gene_by_gene" and not factor_gene_set_x_pheno:
        state.exp_gene_factors_full_projected = full_gene_factor_values
        state._record_params({"factor_full_gene_projection_available": True})
    else:
        state.exp_gene_factors = full_gene_factor_values
    state.exp_pheno_factors = full_pheno_factor_values
    state.exp_gene_set_factors = full_gene_set_factor_values

    if factor_gene_set_x_pheno:
        exp_gene_or_pheno_factors = state.exp_pheno_factors
    else:
        exp_gene_or_pheno_factors = state.exp_gene_factors

    #now update relevance

    matrix_to_mult = state.exp_pheno_factors if factor_gene_set_x_pheno else state.exp_gene_factors
    vector_to_mult = state.pheno_prob_factor_vector if factor_gene_set_x_pheno else state.gene_prob_factor_vector

    #matrix_to_mult: (genes, factors)
    #vector_to_mult: (users, genes)
    #want: (factors, users)

    state.factor_anchor_relevance = state._nnls_project_matrix(matrix_to_mult, vector_to_mult.T, max_value=1).T
    state.factor_relevance = _compute_any_anchor_relevance(state.factor_anchor_relevance)

    if not no_trait_linkage:
        _apply_canonical_trait_linkage(
            state,
            use_gene_set_basis=bool(project_phenos_from_gene_sets or factor_gene_set_x_pheno or state.exp_gene_factors is None),
            trait_linkage_source=trait_linkage_source,
            trait_linkage_threshold=trait_linkage_threshold,
            trait_linkage_computation_mode=trait_linkage_computation_mode,
            pheno_capture_input=pheno_capture_input,
            update_anchor_relevance=state.anchor_pheno_mask is not None,
            log_fn=log,
            info_level=INFO,
        )

    reorder_inds = _finalize_factor_outputs(
        state,
        factor_gene_set_x_pheno=factor_gene_set_x_pheno,
        factor_top_loading_type=factor_top_loading_type,
        lmm_auth_key=lmm_auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        label_gene_sets_only=label_gene_sets_only,
        label_include_phenos=label_include_phenos,
        label_individually=label_individually,
        gene_sets_for_labeling=gene_sets_for_labeling,
        bail_fn=bail,
        warn_fn=warn,
        log_fn=log,
        info_level=INFO,
        labeling_module=labeling_module,
    )

    return _build_factor_run_summary(
        state,
        run_index=0,
        seed=None,
        evidence=evidence_value,
        likelihood=likelihood_value,
        reconstruction_error=reconstruction_error_value,
        factor_gene_set_x_pheno=factor_gene_set_x_pheno,
    ) | ({
        "blockwise_warm_start_payload": blockwise_warm_start_payload,
    } if blockwise_warm_start_payload is not None else {})


def _run_factor_with_seed(state, *, seed, run_index, factor_kwargs):
    def _runner():
        return _run_factor_single(state, **factor_kwargs)

    summary = _run_with_numpy_seed(seed, _runner)
    if summary is None:
        summary = _build_factor_run_summary(
            state,
            run_index=run_index,
            seed=seed,
            evidence=None,
            likelihood=None,
            reconstruction_error=None,
            factor_gene_set_x_pheno=bool(state.exp_pheno_factors is not None and state.gene_in_discovery_mask is None),
        )
    summary["run_index"] = int(run_index)
    summary["seed"] = None if seed is None else int(seed)
    return summary


def _collect_consensus_matches(run_states, run_summaries, reference_index, min_factor_cosine):
    reference_factors = _extract_canonical_factor_matrix(run_states[reference_index])
    reference_norm = _normalize_factor_columns(reference_factors)
    factor_matches = {idx: [(reference_index, idx, 1.0)] for idx in range(reference_factors.shape[1])}

    for run_index, run_state in enumerate(run_states):
        if run_index == reference_index:
            continue
        run_factors = _extract_canonical_factor_matrix(run_state)
        if run_factors is None or run_factors.size == 0:
            continue
        run_norm = _normalize_factor_columns(run_factors)
        similarity = reference_norm.T @ run_norm
        ref_inds, run_inds = scipy.optimize.linear_sum_assignment(1.0 - similarity)
        for ref_factor_index, run_factor_index in zip(ref_inds, run_inds):
            cosine = float(similarity[ref_factor_index, run_factor_index])
            if cosine >= min_factor_cosine:
                factor_matches[ref_factor_index].append((run_index, run_factor_index, cosine))
    return factor_matches


def _apply_consensus_solution(
    reference_state,
    run_states,
    run_summaries,
    *,
    min_run_support,
    aggregation,
    min_factor_cosine,
    factor_kwargs,
):
    eligible_indices = [idx for idx, state in enumerate(run_states) if _extract_canonical_factor_matrix(state) is not None and state.num_factors() > 0]
    if len(eligible_indices) == 0:
        return _clone_runtime_state(run_states[min(range(len(run_summaries)), key=lambda idx: _best_run_sort_key(run_summaries[idx]))]), {
            "mode": "consensus",
            "reference_run_index": None,
            "run_summaries": copy.deepcopy(run_summaries),
            "factor_support": [],
        }

    factor_count_to_indices = {}
    for idx in eligible_indices:
        factor_count_to_indices.setdefault(run_summaries[idx]["num_factors"], []).append(idx)
    modal_factor_count = max(sorted(factor_count_to_indices), key=lambda key: (len(factor_count_to_indices[key]), key))
    reference_candidates = factor_count_to_indices[modal_factor_count]
    reference_index = min(reference_candidates, key=lambda idx: _best_run_sort_key(run_summaries[idx]))

    consensus_state = _clone_runtime_state(run_states[reference_index])
    factor_matches = _collect_consensus_matches(run_states, run_summaries, reference_index, min_factor_cosine)

    kept_factor_columns = []
    kept_gene_columns = []
    kept_pheno_columns = []
    kept_lambda = []
    kept_anchor_relevance = []
    kept_factor_relevance = []
    factor_support_rows = []
    total_runs = max(1, len(run_states))

    reference_factor_matrix = _extract_canonical_factor_matrix(run_states[reference_index])
    for reference_factor_index in range(reference_factor_matrix.shape[1]):
        matched = factor_matches.get(reference_factor_index, [])
        support_fraction = float(len(matched)) / float(total_runs)
        factor_support_rows.append(
            {
                "reference_factor_index": int(reference_factor_index),
                "support_runs": int(len(matched)),
                "support_fraction": support_fraction,
                "kept": bool(support_fraction >= min_run_support),
                "matched_run_indices": [int(run_index) for run_index, _, _ in matched],
                "matched_cosines": [float(cosine) for _, _, cosine in matched],
            }
        )
        if support_fraction < min_run_support:
            continue

        gene_set_stack = np.stack(
            [run_states[run_index].exp_gene_set_factors[:, factor_index] for run_index, factor_index, _ in matched],
            axis=0,
        )
        kept_factor_columns.append(_aggregate_consensus_stack(gene_set_stack, aggregation))

        gene_stack = [
            run_states[run_index].exp_gene_factors[:, factor_index]
            for run_index, factor_index, _ in matched
            if run_states[run_index].exp_gene_factors is not None
        ]
        if len(gene_stack) > 0:
            kept_gene_columns.append(_aggregate_consensus_stack(np.stack(gene_stack, axis=0), aggregation))

        pheno_stack = [
            run_states[run_index].exp_pheno_factors[:, factor_index]
            for run_index, factor_index, _ in matched
            if run_states[run_index].exp_pheno_factors is not None
        ]
        if len(pheno_stack) > 0:
            kept_pheno_columns.append(_aggregate_consensus_stack(np.stack(pheno_stack, axis=0), aggregation))

        kept_lambda.append(
            _aggregate_consensus_stack(
                np.array([run_states[run_index].exp_lambdak[factor_index] for run_index, factor_index, _ in matched], dtype=float),
                aggregation,
            )
        )
        kept_anchor_relevance.append(
            _aggregate_consensus_stack(
                np.stack([run_states[run_index].factor_anchor_relevance[factor_index, :] for run_index, factor_index, _ in matched], axis=0),
                aggregation,
            )
        )
        kept_factor_relevance.append(
            _aggregate_consensus_stack(
                np.array([run_states[run_index].factor_relevance[factor_index] for run_index, factor_index, _ in matched], dtype=float),
                aggregation,
            )
        )

    if len(kept_factor_columns) == 0:
        consensus_state = _clone_runtime_state(run_states[reference_index])
        diagnostics = {
            "mode": "consensus",
            "reference_run_index": int(reference_index),
            "run_summaries": copy.deepcopy(run_summaries),
            "factor_support": factor_support_rows,
            "fallback_to_reference": True,
        }
        return consensus_state, diagnostics

    consensus_state.exp_gene_set_factors = np.column_stack(kept_factor_columns)
    consensus_state.exp_gene_factors = np.column_stack(kept_gene_columns) if len(kept_gene_columns) > 0 else None
    consensus_state.exp_pheno_factors = np.column_stack(kept_pheno_columns) if len(kept_pheno_columns) > 0 else None
    consensus_state.exp_lambdak = np.asarray(kept_lambda, dtype=float)
    consensus_state.factor_anchor_relevance = np.vstack(kept_anchor_relevance)
    consensus_state.factor_relevance = np.asarray(kept_factor_relevance, dtype=float)

    _finalize_factor_outputs(
        consensus_state,
        factor_gene_set_x_pheno=bool(run_summaries[reference_index]["factor_gene_set_x_pheno"]),
        factor_top_loading_type=factor_kwargs["factor_top_loading_type"],
        lmm_auth_key=factor_kwargs["lmm_auth_key"],
        lmm_model=factor_kwargs["lmm_model"],
        lmm_provider=factor_kwargs["lmm_provider"],
        label_gene_sets_only=factor_kwargs["label_gene_sets_only"],
        label_include_phenos=factor_kwargs["label_include_phenos"],
        label_individually=factor_kwargs["label_individually"],
        gene_sets_for_labeling=factor_kwargs.get("gene_sets_for_labeling"),
        bail_fn=factor_kwargs["bail_fn"],
        warn_fn=factor_kwargs["warn_fn"],
        log_fn=factor_kwargs["log_fn"],
        info_level=factor_kwargs["info_level"],
        labeling_module=factor_kwargs["labeling_module"],
    )

    diagnostics = {
        "mode": "consensus",
        "reference_run_index": int(reference_index),
        "run_summaries": copy.deepcopy(run_summaries),
        "factor_support": factor_support_rows,
        "modal_factor_count": int(modal_factor_count),
        "aggregation": aggregation,
        "min_factor_cosine": float(min_factor_cosine),
        "min_run_support": float(min_run_support),
    }
    return consensus_state, diagnostics


def run_factor(state, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, seed=None, factor_runs=1, consensus_nmf=False, consensus_min_factor_cosine=0.7, consensus_min_run_support=0.5, consensus_aggregation="median", consensus_stats_out=None, learn_phi=False, learn_phi_max_redundancy=0.5, learn_phi_max_redundancy_q90=0.35, learn_phi_runs_per_step=1, learn_phi_min_run_support=0.6, learn_phi_min_stability=0.85, learn_phi_fit_loss_warning_frac=0.05, learn_phi_max_severe_fit_loss_frac=1.0, learn_phi_target_gene_mass=None, learn_phi_target_gene_effective_support=None, learn_phi_size_tolerance_frac=0.25, learn_phi_min_primary_factors=3, learn_phi_max_primary_gene_max_weight_q90=None, learn_phi_max_steps=5, learn_phi_values=None, learn_phi_expand_factor=2.0, learn_phi_weight_floor=None, learn_phi_metric_factor_scope="primary", learn_phi_mass_floor_frac=_DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC, learn_phi_only=False, learn_phi_report_out=None, factor_phi_metrics_out=None, factor_phi_factors_out=None, factor_phi_gene_set_clusters_out=None, factor_phi_gene_clusters_out=None, factor_backend="full", learn_phi_backend="sentinel_pruned", blockwise_gene_set_block_size=5000, blockwise_epochs=3, blockwise_shuffle_blocks=True, blockwise_warm_start=True, blockwise_max_blocks=None, blockwise_report_out=None, factors_out=None, factor_metrics_out=None, gene_set_clusters_out=None, gene_clusters_out=None, cluster_row_min_max_loading=0.01, factor_output_scope="primary", learn_phi_prune_genes_num=1000, learn_phi_prune_gene_sets_num=1000, learn_phi_max_num_iterations=None, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, max_num_discovery_gene_sets=None, auto_discovery_subset=True, discovery_redundancy_weighting=True, discovery_redundancy_weighting_mode="effective_size", discovery_similarity_threshold=0.35, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, lmm_model=None, lmm_provider="openai", label_gene_sets_only=False, label_include_phenos=False, label_individually=False, gene_sets_for_labeling=None, factor_top_loading_type="combined", keep_original_loadings=False, project_phenos_from_gene_sets=False, pheno_capture_input="weighted_thresholded", trait_linkage_source="combined", trait_linkage_threshold=1.0, trait_linkage_computation_mode="sparse_full", no_trait_linkage=False, discovery_model="gene_by_annotation", anchor_aggregation="multi", gene_gene_beta_source="beta", gene_gene_pair_prior=None, gene_gene_pair_prior_effective_size=None, gene_gene_logbf_base="natural", gene_gene_diagonal_weight=0.0, gene_gene_matrix_floor=1e-3, gene_gene_excess_probability=True, gene_gene_row_sum_cap=True, gene_gene_sparsity=0.0, gene_gene_profligate_correction="none", *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, labeling_module):
    bail = bail_fn
    log = log_fn
    INFO = info_level
    if (
        learn_phi
        and str(discovery_model) == "gene_by_gene"
        and learn_phi_target_gene_mass is None
        and learn_phi_target_gene_effective_support is None
    ):
        learn_phi_target_gene_mass = 40.0

    if factor_runs < 1:
        bail("--factor-runs must be at least 1")
    if factor_backend not in {"full", "blockwise_global_w"}:
        bail("--factor-backend must be one of: full, blockwise_global_w")
    if learn_phi_backend not in {"sentinel_pruned", "blockwise_global_w"}:
        bail("--learn-phi-backend must be one of: sentinel_pruned, blockwise_global_w")
    if int(blockwise_gene_set_block_size) < 1:
        bail("--blockwise-gene-set-block-size must be at least 1")
    if int(blockwise_epochs) < 1:
        bail("--blockwise-epochs must be at least 1")
    if blockwise_max_blocks is not None and int(blockwise_max_blocks) < 1:
        bail("--blockwise-max-blocks must be at least 1")
    if float(cluster_row_min_max_loading) < 0:
        bail("--cluster-row-min-max-loading must be nonnegative")
    if str(factor_output_scope) not in {"primary", "primary_secondary", "all"}:
        bail("--factor-output-scope must be one of: primary, primary_secondary, all")
    if str(learn_phi_metric_factor_scope) not in {"primary", "all"}:
        bail("--learn-phi-metric-factor-scope must be one of: primary, all")
    if consensus_aggregation not in {"median", "mean"}:
        bail("--consensus-aggregation must be one of: median, mean")
    if not (0 < consensus_min_factor_cosine <= 1):
        bail("--consensus-min-factor-cosine must be in (0, 1]")
    if not (0 < consensus_min_run_support <= 1):
        bail("--consensus-min-run-support must be in (0, 1]")
    if consensus_nmf and factor_runs < 2:
        bail("--consensus-nmf requires --factor-runs >= 2")
    if factor_top_loading_type not in {"raw", "specific", "combined"}:
        bail("--factor-top-loading-type must be one of: raw, specific, combined")
    if learn_phi:
        if phi <= 0:
            bail("--learn-phi requires --phi > 0")
        if learn_phi_target_gene_mass is None and learn_phi_target_gene_effective_support is None:
            bail("--learn-phi requires either --learn-phi-target-gene-mass or --learn-phi-target-gene-effective-support")
        if learn_phi_target_gene_mass is not None and learn_phi_target_gene_effective_support is not None:
            bail("--learn-phi-target-gene-mass and --learn-phi-target-gene-effective-support cannot both be set")
        if learn_phi_target_gene_mass is not None and float(learn_phi_target_gene_mass) <= 0:
            bail("--learn-phi-target-gene-mass must be positive")
        if learn_phi_target_gene_effective_support is not None and float(learn_phi_target_gene_effective_support) <= 0:
            bail("--learn-phi-target-gene-effective-support must be positive")
        if not (0 < learn_phi_max_redundancy <= 1):
            bail("--learn-phi-max-redundancy must be in (0, 1]")
        if not (0 < learn_phi_max_redundancy_q90 <= 1):
            bail("--learn-phi-max-redundancy-q90 must be in (0, 1]")
        if learn_phi_runs_per_step < 1:
            bail("--learn-phi-runs-per-step must be at least 1")
        if not (0 < learn_phi_min_run_support <= 1):
            bail("--learn-phi-min-run-support must be in (0, 1]")
        if not (0 < learn_phi_min_stability <= 1):
            bail("--learn-phi-min-stability must be in (0, 1]")
        if learn_phi_fit_loss_warning_frac < 0:
            bail("--learn-phi-fit-loss-warning-frac must be >= 0")
        if learn_phi_max_severe_fit_loss_frac < 0:
            bail("--learn-phi-max-severe-fit-loss-frac must be >= 0")
        if learn_phi_size_tolerance_frac < 0:
            bail("--learn-phi-size-tolerance-frac must be nonnegative")
        if learn_phi_min_primary_factors < 1:
            bail("--learn-phi-min-primary-factors must be at least 1")
        if learn_phi_max_primary_gene_max_weight_q90 is not None and not (0 < learn_phi_max_primary_gene_max_weight_q90 <= 1):
            bail("--learn-phi-max-primary-gene-max-weight-q90 must be in (0, 1]")
        if learn_phi_max_steps < 1:
            bail("--learn-phi-max-steps must be at least 1")
        try:
            parsed_learn_phi_values = _parse_explicit_phi_values(learn_phi_values)
        except (TypeError, ValueError):
            bail("--learn-phi-values must be a comma-separated list of positive numbers")
        if learn_phi_expand_factor <= 1:
            bail("--learn-phi-expand-factor must be > 1")
        if learn_phi_weight_floor is not None and learn_phi_weight_floor < 0:
            bail("--learn-phi-weight-floor must be >= 0 when set")
        if not (0 < learn_phi_mass_floor_frac <= 1):
            bail("--learn-phi-mass-floor-frac must be in (0, 1]")
        if learn_phi_prune_genes_num is not None and learn_phi_prune_genes_num < 1:
            bail("--learn-phi-prune-genes-num must be at least 1")
        if learn_phi_prune_gene_sets_num is not None and learn_phi_prune_gene_sets_num < 1:
            bail("--learn-phi-prune-gene-sets-num must be at least 1")
        if learn_phi_max_num_iterations is not None and learn_phi_max_num_iterations < 1:
            bail("--learn-phi-max-num-iterations must be at least 1")
    if max_num_discovery_gene_sets is not None and int(max_num_discovery_gene_sets) < 1:
        bail("--max-num-discovery-gene-sets must be at least 1")
    if discovery_redundancy_weighting_mode not in {"effective_size", "log_effective_size", "none"}:
        bail("--discovery-redundancy-weighting-mode must be one of: effective_size, log_effective_size, none")
    if not (0 <= float(discovery_similarity_threshold) <= 1):
        bail("--discovery-similarity-threshold must be in [0, 1]")
    if str(discovery_model) not in {"gene_by_annotation", "gene_by_gene"}:
        bail("--discovery-model must be one of: gene_by_annotation, gene_by_gene")
    if str(gene_gene_beta_source) != "beta":
        bail("--gene-gene-beta-source must be beta")
    if str(gene_gene_profligate_correction) not in {"none", "linear"}:
        bail("--gene-gene-profligate-correction must be one of: none, linear")
    if str(discovery_model) != "gene_by_gene" and str(gene_gene_profligate_correction) != "none":
        bail("--gene-gene-profligate-correction requires --discovery-model gene_by_gene")
    if str(gene_gene_logbf_base) not in {"natural", "log10"}:
        bail("--gene-gene-logbf-base must be one of: natural, log10")
    if str(anchor_aggregation) not in {"multi", "any"}:
        bail("--anchor-aggregation must be one of: multi, any")
    if gene_gene_pair_prior is not None and not (0 < float(gene_gene_pair_prior) < 1):
        bail("--gene-gene-pair-prior must be in (0, 1)")
    if gene_gene_pair_prior_effective_size is not None and float(gene_gene_pair_prior_effective_size) <= 1:
        bail("--gene-gene-pair-prior-effective-size must be > 1")
    if float(gene_gene_diagonal_weight) < 0:
        bail("--gene-gene-diagonal-weight must be >= 0")
    if float(gene_gene_matrix_floor) < 0:
        bail("--gene-gene-matrix-floor must be >= 0")
    if float(gene_gene_sparsity) < 0:
        bail("--gene-gene-sparsity must be >= 0")
    if str(discovery_model) == "gene_by_gene":
        if factor_backend != "full":
            bail("--discovery-model gene_by_gene currently requires --factor-backend full")
        if learn_phi_backend != "sentinel_pruned":
            bail("--discovery-model gene_by_gene currently requires --learn-phi-backend sentinel_pruned")
        if not run_transpose:
            bail("--discovery-model gene_by_gene currently requires the default transposed factor matrix")
    if learn_phi_prune_gene_sets_num is not None:
        warnings.warn(
            "learn_phi_prune_gene_sets_num is deprecated and ignored; learn-phi now uses the same discovery plan as the final fit",
            DeprecationWarning,
            stacklevel=2,
        )

    factor_kwargs = {
        "max_num_factors": max_num_factors,
        "phi": phi,
        "seed": seed,
        "discovery_model": discovery_model,
        "gene_gene_beta_source": gene_gene_beta_source,
        "gene_gene_pair_prior": gene_gene_pair_prior,
        "gene_gene_pair_prior_effective_size": gene_gene_pair_prior_effective_size,
        "gene_gene_logbf_base": gene_gene_logbf_base,
        "anchor_aggregation": anchor_aggregation,
        "gene_gene_diagonal_weight": gene_gene_diagonal_weight,
        "gene_gene_matrix_floor": gene_gene_matrix_floor,
        "gene_gene_excess_probability": gene_gene_excess_probability,
        "gene_gene_row_sum_cap": gene_gene_row_sum_cap,
        "gene_gene_sparsity": gene_gene_sparsity,
        "gene_gene_profligate_correction": gene_gene_profligate_correction,
        "alpha0": alpha0,
        "beta0": beta0,
        "gene_set_filter_type": gene_set_filter_type,
        "gene_set_filter_value": gene_set_filter_value,
        "gene_or_pheno_filter_type": gene_or_pheno_filter_type,
        "gene_or_pheno_filter_value": gene_or_pheno_filter_value,
        "pheno_prune_value": pheno_prune_value,
        "pheno_prune_number": pheno_prune_number,
        "gene_prune_value": gene_prune_value,
        "gene_prune_number": gene_prune_number,
        "gene_set_prune_value": gene_set_prune_value,
        "gene_set_prune_number": gene_set_prune_number,
        "max_num_discovery_gene_sets": max_num_discovery_gene_sets,
        "auto_discovery_subset": auto_discovery_subset,
        "discovery_redundancy_weighting": discovery_redundancy_weighting,
        "discovery_redundancy_weighting_mode": discovery_redundancy_weighting_mode,
        "discovery_similarity_threshold": discovery_similarity_threshold,
        "anchor_pheno_mask": anchor_pheno_mask,
        "anchor_gene_mask": anchor_gene_mask,
        "anchor_any_pheno": anchor_any_pheno,
        "anchor_any_gene": anchor_any_gene,
        "anchor_gene_set": anchor_gene_set,
        "run_transpose": run_transpose,
        "max_num_iterations": max_num_iterations,
        "rel_tol": rel_tol,
        "min_lambda_threshold": min_lambda_threshold,
        "lmm_auth_key": lmm_auth_key,
        "lmm_model": lmm_model,
        "lmm_provider": lmm_provider,
        "label_gene_sets_only": label_gene_sets_only,
        "label_include_phenos": label_include_phenos,
        "label_individually": label_individually,
        "gene_sets_for_labeling": gene_sets_for_labeling,
        "factor_top_loading_type": factor_top_loading_type,
        "keep_original_loadings": keep_original_loadings,
        "project_phenos_from_gene_sets": project_phenos_from_gene_sets,
        "pheno_capture_input": pheno_capture_input,
        "trait_linkage_source": trait_linkage_source,
        "trait_linkage_threshold": trait_linkage_threshold,
        "trait_linkage_computation_mode": trait_linkage_computation_mode,
        "no_trait_linkage": no_trait_linkage,
        "factor_backend": factor_backend,
        "blockwise_gene_set_block_size": blockwise_gene_set_block_size,
        "blockwise_epochs": blockwise_epochs,
        "blockwise_shuffle_blocks": blockwise_shuffle_blocks,
        "blockwise_warm_start": blockwise_warm_start,
        "blockwise_max_blocks": blockwise_max_blocks,
        "blockwise_report_out": blockwise_report_out,
        "factors_out": factors_out,
        "factor_metrics_out": factor_metrics_out,
        "gene_set_clusters_out": gene_set_clusters_out,
        "gene_clusters_out": gene_clusters_out,
        "cluster_row_min_max_loading": cluster_row_min_max_loading,
        "factor_output_scope": factor_output_scope,
        "learn_phi_target_gene_mass": learn_phi_target_gene_mass,
        "learn_phi_target_gene_effective_support": learn_phi_target_gene_effective_support,
        "bail_fn": bail_fn,
        "warn_fn": warn_fn,
        "log_fn": log_fn,
        "info_level": info_level,
        "debug_level": debug_level,
        "trace_level": trace_level,
        "labeling_module": labeling_module,
    }

    state._record_params(
        _build_factor_param_record(
            max_num_factors=max_num_factors,
            phi=phi,
            alpha0=alpha0,
            beta0=beta0,
            seed=seed,
            factor_runs=factor_runs,
            consensus_nmf=consensus_nmf,
            consensus_min_factor_cosine=consensus_min_factor_cosine,
            consensus_min_run_support=consensus_min_run_support,
            consensus_aggregation=consensus_aggregation,
            consensus_stats_out=consensus_stats_out,
            discovery_model=discovery_model,
            anchor_aggregation=anchor_aggregation,
            gene_gene_beta_source=gene_gene_beta_source,
            gene_gene_pair_prior=gene_gene_pair_prior,
            gene_gene_pair_prior_effective_size=gene_gene_pair_prior_effective_size,
            gene_gene_logbf_base=gene_gene_logbf_base,
                gene_gene_diagonal_weight=gene_gene_diagonal_weight,
            gene_gene_matrix_floor=gene_gene_matrix_floor,
            gene_gene_excess_probability=gene_gene_excess_probability,
            gene_gene_row_sum_cap=gene_gene_row_sum_cap,
            gene_gene_sparsity=gene_gene_sparsity,
            gene_gene_profligate_correction=gene_gene_profligate_correction,
            learn_phi=learn_phi,
            learn_phi_max_redundancy=learn_phi_max_redundancy,
            learn_phi_max_redundancy_q90=learn_phi_max_redundancy_q90,
            learn_phi_runs_per_step=learn_phi_runs_per_step,
            learn_phi_min_run_support=learn_phi_min_run_support,
            learn_phi_min_stability=learn_phi_min_stability,
            learn_phi_fit_loss_warning_frac=learn_phi_fit_loss_warning_frac,
            learn_phi_max_severe_fit_loss_frac=learn_phi_max_severe_fit_loss_frac,
            learn_phi_target_gene_mass=learn_phi_target_gene_mass,
            learn_phi_target_gene_effective_support=learn_phi_target_gene_effective_support,
            learn_phi_size_tolerance_frac=learn_phi_size_tolerance_frac,
            learn_phi_min_primary_factors=learn_phi_min_primary_factors,
            learn_phi_max_primary_gene_max_weight_q90=learn_phi_max_primary_gene_max_weight_q90,
            learn_phi_max_steps=learn_phi_max_steps,
            learn_phi_values=learn_phi_values,
            learn_phi_expand_factor=learn_phi_expand_factor,
            learn_phi_weight_floor=learn_phi_weight_floor,
            learn_phi_metric_factor_scope=learn_phi_metric_factor_scope,
            learn_phi_mass_floor_frac=learn_phi_mass_floor_frac,
            learn_phi_only=learn_phi_only,
            learn_phi_report_out=learn_phi_report_out,
            factor_phi_metrics_out=factor_phi_metrics_out,
            factor_phi_factors_out=factor_phi_factors_out,
            factor_phi_gene_set_clusters_out=factor_phi_gene_set_clusters_out,
            factor_phi_gene_clusters_out=factor_phi_gene_clusters_out,
            cluster_row_min_max_loading=cluster_row_min_max_loading,
            factor_output_scope=factor_output_scope,
            factor_backend=factor_backend,
            learn_phi_backend=learn_phi_backend,
            blockwise_gene_set_block_size=blockwise_gene_set_block_size,
            blockwise_epochs=blockwise_epochs,
            blockwise_shuffle_blocks=blockwise_shuffle_blocks,
            blockwise_warm_start=blockwise_warm_start,
            blockwise_max_blocks=blockwise_max_blocks,
            blockwise_report_out=blockwise_report_out,
            learn_phi_prune_genes_num=learn_phi_prune_genes_num,
            learn_phi_prune_gene_sets_num=learn_phi_prune_gene_sets_num,
            learn_phi_max_num_iterations=learn_phi_max_num_iterations,
            gene_set_filter_type=gene_set_filter_type,
            gene_set_filter_value=gene_set_filter_value,
            gene_or_pheno_filter_type=gene_or_pheno_filter_type,
            gene_or_pheno_filter_value=gene_or_pheno_filter_value,
            pheno_prune_value=pheno_prune_value,
            pheno_prune_number=pheno_prune_number,
            gene_prune_value=gene_prune_value,
            gene_prune_number=gene_prune_number,
            gene_set_prune_value=gene_set_prune_value,
            gene_set_prune_number=gene_set_prune_number,
            max_num_discovery_gene_sets=max_num_discovery_gene_sets,
            auto_discovery_subset=auto_discovery_subset,
            discovery_redundancy_weighting=discovery_redundancy_weighting,
            discovery_redundancy_weighting_mode=discovery_redundancy_weighting_mode,
            discovery_similarity_threshold=discovery_similarity_threshold,
            anchor_pheno_mask=anchor_pheno_mask,
            anchor_gene_mask=anchor_gene_mask,
            anchor_any_pheno=anchor_any_pheno,
            anchor_any_gene=anchor_any_gene,
            anchor_gene_set=anchor_gene_set,
            run_transpose=run_transpose,
            max_num_iterations=max_num_iterations,
            rel_tol=rel_tol,
            min_lambda_threshold=min_lambda_threshold,
            lmm_auth_key=lmm_auth_key,
            lmm_model=lmm_model,
            lmm_provider=lmm_provider,
            label_gene_sets_only=label_gene_sets_only,
            label_include_phenos=label_include_phenos,
            label_individually=label_individually,
            gene_sets_for_labeling=gene_sets_for_labeling,
            keep_original_loadings=keep_original_loadings,
            project_phenos_from_gene_sets=project_phenos_from_gene_sets,
            pheno_capture_input=pheno_capture_input,
            trait_linkage_source=trait_linkage_source,
            trait_linkage_threshold=trait_linkage_threshold,
            trait_linkage_computation_mode=trait_linkage_computation_mode,
            no_trait_linkage=no_trait_linkage,
        ),
        overwrite=True,
    )
    log(
        "Factor config: phi=%.6g alpha0=%.6g beta0=%.6g max_num_factors=%d factor_runs=%d consensus_nmf=%s learn_phi=%s factor_backend=%s learn_phi_backend=%s discovery_model=%s max_num_iterations=%d rel_tol=%.3g"
        % (
            float(phi),
            float(alpha0),
            float(beta0),
            int(max_num_factors),
            int(factor_runs),
            bool(consensus_nmf),
            bool(learn_phi),
            str(factor_backend),
            str(learn_phi_backend),
            str(discovery_model),
            int(max_num_iterations),
            float(rel_tol),
        ),
        INFO,
    )

    if learn_phi:
        weight_floor = 0.01 if learn_phi_weight_floor is None else float(learn_phi_weight_floor)
        selected_candidate = _learn_phi(
            state,
            initial_phi=phi,
            seed=seed,
            runs_per_step=learn_phi_runs_per_step,
            max_redundancy=learn_phi_max_redundancy,
            max_redundancy_q90=learn_phi_max_redundancy_q90,
            min_run_support=learn_phi_min_run_support,
            min_stability=learn_phi_min_stability,
            fit_loss_warning_frac=learn_phi_fit_loss_warning_frac,
            max_severe_fit_loss_frac=learn_phi_max_severe_fit_loss_frac,
            target_gene_mass=(
                None if learn_phi_target_gene_mass is None else float(learn_phi_target_gene_mass)
            ),
            target_gene_effective_support=(
                None
                if learn_phi_target_gene_effective_support is None
                else float(learn_phi_target_gene_effective_support)
            ),
            size_tolerance_frac=float(learn_phi_size_tolerance_frac),
            min_primary_factors=int(learn_phi_min_primary_factors),
            max_primary_gene_max_weight_q90=learn_phi_max_primary_gene_max_weight_q90,
            max_steps=learn_phi_max_steps,
            explicit_phi_values=learn_phi_values,
            expand_factor=learn_phi_expand_factor,
            weight_floor=weight_floor,
            metric_factor_scope=str(learn_phi_metric_factor_scope),
            mass_floor_frac=float(learn_phi_mass_floor_frac),
            learn_phi_backend=learn_phi_backend,
            blockwise_warm_start=blockwise_warm_start,
            report_out=learn_phi_report_out,
            factor_phi_metrics_out=factor_phi_metrics_out,
            factor_phi_factors_out=factor_phi_factors_out,
            factor_phi_gene_set_clusters_out=factor_phi_gene_set_clusters_out,
            factor_phi_gene_clusters_out=factor_phi_gene_clusters_out,
            cluster_row_min_max_loading=cluster_row_min_max_loading,
            factor_output_scope=factor_output_scope,
            prune_genes_num=learn_phi_prune_genes_num,
            prune_gene_sets_num=learn_phi_prune_gene_sets_num,
            max_num_iterations=learn_phi_max_num_iterations,
            factor_kwargs=factor_kwargs,
            log_fn=log,
            info_level=INFO,
            redundancy_hard_filter=(str(discovery_model) != "gene_by_gene"),
        )
        phi = float(selected_candidate["phi"])
        factor_kwargs["phi"] = phi
        state._record_params({"phi": phi}, overwrite=True)
        log("Using learned phi %.6g for final factorization" % float(phi), INFO)
        if learn_phi_only:
            log("Stopping after automatic phi selection because --learn-phi-only was requested", INFO)
            return
    else:
        log("Using fixed phi %.6g for factorization" % float(phi), INFO)

    if factor_runs == 1 and not consensus_nmf:
        summary = _run_factor_with_seed(state, seed=seed, run_index=0, factor_kwargs=factor_kwargs)
        state.consensus_mode = "single"
        state.consensus_reference_run = 0
        state.consensus_run_diagnostics = [summary]
        state.consensus_factor_support = []
        return

    child_seeds = _derive_factor_run_seeds(seed, factor_runs)
    run_states = []
    run_summaries = []
    for run_index, child_seed in enumerate(child_seeds):
        log("Running factor restart %d/%d%s" % (run_index + 1, factor_runs, "" if child_seed is None else " [seed=%d]" % child_seed), INFO)
        run_state = _clone_runtime_state(state)
        summary = _run_factor_with_seed(run_state, seed=child_seed, run_index=run_index, factor_kwargs=factor_kwargs)
        run_states.append(run_state)
        run_summaries.append(summary)

    if consensus_nmf:
        selected_state, diagnostics = _apply_consensus_solution(
            state,
            run_states,
            run_summaries,
            min_run_support=consensus_min_run_support,
            aggregation=consensus_aggregation,
            min_factor_cosine=consensus_min_factor_cosine,
            factor_kwargs=factor_kwargs,
        )
    else:
        best_run_index = min(range(len(run_summaries)), key=lambda idx: _best_run_sort_key(run_summaries[idx]))
        selected_state = run_states[best_run_index]
        diagnostics = {
            "mode": "best_of_n",
            "reference_run_index": int(best_run_index),
            "run_summaries": copy.deepcopy(run_summaries),
            "factor_support": [],
        }

    _replace_runtime_state(state, selected_state)
    state.consensus_mode = diagnostics["mode"]
    state.consensus_reference_run = diagnostics["reference_run_index"]
    state.consensus_run_diagnostics = diagnostics["run_summaries"]
    state.consensus_factor_support = diagnostics["factor_support"]
    state.consensus_stats_out = consensus_stats_out
