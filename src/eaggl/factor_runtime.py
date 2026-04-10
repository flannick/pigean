from __future__ import annotations

import copy
import gzip
import math
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import scipy
import scipy.sparse as sparse

from . import phenotype_annotation as eaggl_phenotype_annotation


@contextmanager
def _open_text_output(path):
    if path is None:
        raise ValueError("output path must not be None")
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            yield handle
    else:
        with open(path, "w", encoding="utf-8") as handle:
            yield handle

def _clone_runtime_value(value):
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _clone_runtime_state(source):
    cloned = copy.copy(source)
    cloned.__dict__ = {key: _clone_runtime_value(value) for key, value in source.__dict__.items()}
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


def _choose_gene_or_pheno_anchor_source(combined_prior_Ys, priors, Y, *, log_fn=None, info_level=None):
    for label, matrix in (("combined_prior_Ys", combined_prior_Ys), ("Y", Y), ("priors", priors)):
        if matrix is None:
            continue
        if _matrix_nonfinite_fraction(matrix) >= 1.0:
            continue
        if label != "combined_prior_Ys" and log_fn is not None:
            log_fn(
                "Using %s for implicit factor-anchor relevance because earlier source combined_prior_Ys had non-finite values"
                % str(label),
                info_level,
            )
        return matrix, label
    return None, None


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


def _append_with_any_user_for_blockwise(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    return np.hstack((matrix, 1 - np.prod(1 - matrix, axis=1)[:, np.newaxis]))


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


def _compute_weight_matrix_for_block(block_probabilities, global_probabilities, num_rows, num_cols):
    if block_probabilities is None and global_probabilities is None:
        return np.ones((num_rows, num_cols), dtype=float)
    block_probabilities = _append_with_any_user_for_blockwise(block_probabilities)
    global_probabilities = _append_with_any_user_for_blockwise(global_probabilities)
    if block_probabilities is None:
        block_probabilities = np.ones((num_rows, global_probabilities.shape[1]), dtype=float)
    if global_probabilities is None:
        global_probabilities = np.ones((num_cols, block_probabilities.shape[1]), dtype=float)
    return np.asarray(block_probabilities @ global_probabilities.T, dtype=float)


def _initialize_blockwise_gene_factors(num_factors, num_columns, vmax):
    scale = max(float(vmax), 1e-6)
    return np.random.random((int(num_factors), int(num_columns))) * scale


def _initialize_blockwise_gene_set_factors(num_rows, num_factors, vmax):
    scale = max(float(vmax), 1e-6)
    return np.random.random((int(num_rows), int(num_factors))) * scale


def _normalize_rows_l2(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix.reshape(matrix.shape)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return matrix / norms


def _normalize_columns_l2(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix.reshape(matrix.shape)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms[norms <= 0] = 1.0
    return matrix / norms


def _blockwise_ranges_by_id(num_rows, block_size, *, max_blocks=None):
    return {
        int(block_number): (int(start), int(stop))
        for block_number, start, stop in _blockwise_block_ranges(
            num_rows,
            int(block_size),
            shuffle_blocks=False,
            rng=np.random.RandomState(0),
            max_blocks=max_blocks,
        )
    }


def _factor_metric_record_summary(records):
    if records is None or len(records) == 0:
        return {
            "median_gene_overlap": 0.0,
            "q90_gene_overlap": 0.0,
            "median_gene_set_overlap": 0.0,
            "q90_gene_set_overlap": 0.0,
            "median_combined_unique_fraction": 1.0,
        }

    def _extract(values_key, default_value=0.0):
        values = [float(record.get(values_key, default_value) or default_value) for record in records]
        return np.asarray(values, dtype=float)

    gene_overlap = _extract("gene_max_jaccard")
    gene_set_overlap = _extract("gene_set_max_jaccard")
    combined_unique = _extract("combined_unique_fraction", 1.0)
    return {
        "median_gene_overlap": float(np.median(gene_overlap)) if gene_overlap.size > 0 else 0.0,
        "q90_gene_overlap": float(np.quantile(gene_overlap, 0.9)) if gene_overlap.size > 0 else 0.0,
        "median_gene_set_overlap": float(np.median(gene_set_overlap)) if gene_set_overlap.size > 0 else 0.0,
        "q90_gene_set_overlap": float(np.quantile(gene_set_overlap, 0.9)) if gene_set_overlap.size > 0 else 0.0,
        "median_combined_unique_fraction": float(np.median(combined_unique)) if combined_unique.size > 0 else 1.0,
    }


def _summarize_block_factor_usage(block_rows) -> np.ndarray:
    if block_rows is None or len(block_rows) == 0:
        return np.zeros((0, 0), dtype=float)
    block_ids = sorted(int(block_id) for block_id in block_rows.keys())
    first_block = np.asarray(block_rows[block_ids[0]], dtype=float)
    num_factors = int(first_block.shape[1]) if first_block.ndim == 2 else 0
    usage = np.zeros((len(block_ids), num_factors), dtype=float)
    for row_index, block_id in enumerate(block_ids):
        block = np.asarray(block_rows[block_id], dtype=float)
        if block.ndim != 2 or block.shape[1] != num_factors:
            continue
        usage[row_index, :] = np.sum(np.square(np.maximum(block, 0.0)), axis=0)
    return usage


def _block_usage_entropy_summary(block_usage) -> dict:
    usage = np.asarray(block_usage, dtype=float)
    if usage.size == 0 or usage.ndim != 2 or usage.shape[1] == 0:
        return {"median_block_usage_entropy": 0.0, "q90_block_usage_entropy": 0.0}
    normalized = np.maximum(usage, 0.0)
    normalized /= np.maximum(np.sum(normalized, axis=0, keepdims=True), 1e-50)
    entropy = -np.sum(np.where(normalized > 0, normalized * np.log(np.maximum(normalized, 1e-50)), 0.0), axis=0)
    if usage.shape[0] > 1:
        entropy /= math.log(float(usage.shape[0]))
    return {
        "median_block_usage_entropy": float(np.median(entropy)) if entropy.size > 0 else 0.0,
        "q90_block_usage_entropy": float(np.quantile(entropy, 0.9)) if entropy.size > 0 else 0.0,
    }


def _solve_blockwise_block_with_fixed_basis(
    V_block,
    S_block,
    shared_gene_factors,
    lambdak,
    *,
    phi_scaled,
    eps,
    cap_gene_sets,
    initial_W_block=None,
    vmax=1.0,
    iterations=5,
):
    current_k = int(shared_gene_factors.shape[0])
    W_block = (
        _initialize_blockwise_gene_set_factors(V_block.shape[0], current_k, vmax)
        if initial_W_block is None or np.asarray(initial_W_block).shape != (V_block.shape[0], current_k)
        else np.maximum(np.asarray(initial_W_block, dtype=float).copy(), eps)
    )
    for _ in range(int(max(1, iterations))):
        V_ap_block = W_block @ shared_gene_factors + eps
        numerator_W = (V_block * S_block) @ shared_gene_factors.T
        denominator_W = (V_ap_block * S_block) @ shared_gene_factors.T + phi_scaled * W_block * (1.0 / np.maximum(lambdak, eps))[np.newaxis, :] + eps
        W_block *= numerator_W / denominator_W
        W_block = np.maximum(W_block, 0.0)
        if cap_gene_sets:
            W_block = np.clip(W_block, 0.0, 1.0)
    V_ap_block = W_block @ shared_gene_factors + eps
    error = float(np.sum(S_block * np.square(V_block - V_ap_block)))
    likelihood = float(np.sum(0.5 * S_block * np.square(V_block - V_ap_block)))
    return W_block, error, likelihood


def _fit_online_shared_basis(
    state,
    matrix,
    *,
    gene_set_prob_vector,
    gene_or_pheno_prob_vector,
    max_num_factors,
    max_num_iterations,
    alpha0,
    phi,
    rel_tol,
    min_lambda_threshold,
    block_size,
    passes,
    min_passes_before_stopping,
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
    V = matrix.tocsr() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
    N, M = V.shape
    vmax = float(np.max(V)) if N > 0 and M > 0 else 1.0
    K = int(max_num_factors)
    K0 = int(_DEFAULT_BLOCKWISE_K0)
    a0 = float(alpha0)
    if sparse.issparse(V):
        total_entries = float(max(1, N * M))
        data = np.asarray(V.data, dtype=float)
        mean_V = float(np.sum(data) / total_entries) if data.size > 0 else 0.0
        second_moment = float(np.sum(np.square(data)) / total_entries) if data.size > 0 else 0.0
        std_V = math.sqrt(max(0.0, second_moment - mean_V ** 2))
    else:
        mean_V = float(np.mean(V)) if N > 0 and M > 0 else 0.0
        std_V = float(np.std(V)) if N > 0 and M > 0 else 0.0
    phi_scaled = (std_V ** 2) * float(phi)
    C = (N + M) / 2.0 + a0 + 1.0
    b0 = 3.14 * (a0 - 1.0) * mean_V / (2.0 * max(1, K0))
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

    block_ranges_by_id = _blockwise_ranges_by_id(N, int(block_size), max_blocks=max_blocks)
    ordered_block_ids = sorted(block_ranges_by_id.keys())
    if lambdak is None:
        initial_w_sq_sum = np.zeros(K, dtype=float)
        for block_number, (start, stop) in sorted(block_ranges_by_id.items()):
            W_block = _initialize_blockwise_gene_set_factors(stop - start, K, vmax)
            block_rows[block_number] = W_block
            initial_w_sq_sum += np.sum(W_block ** 2, axis=0)
        lambdak = np.maximum((0.5 * (initial_w_sq_sum + np.sum(shared_gene_factors ** 2, axis=1)) + b0) / C, eps)

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
        for block_number, (start, stop) in sorted(block_ranges_by_id.items()):
            W_block = current_block_rows.get(block_number)
            if W_block is None or W_block.shape != (stop - start, current_k):
                W_block = _initialize_blockwise_gene_set_factors(stop - start, current_k, vmax)
            ordered_blocks.append(np.asarray(W_block, dtype=float))
        return np.vstack(ordered_blocks) if len(ordered_blocks) > 0 else np.zeros((int(N), current_shared_gene_factors.shape[0]), dtype=float)

    def _compute_pass_diagnostics(current_shared_gene_factors, current_lambdak, current_block_rows):
        gene_set_factors = _materialize_gene_set_factors(current_block_rows, current_shared_gene_factors)
        factor_records = _collect_blockwise_factor_metric_records(
            state,
            gene_set_factors=gene_set_factors,
            gene_factors=current_shared_gene_factors.T,
            lambdak=current_lambdak,
        ) or []
        mass_profile = _compute_factor_mass_profile(
            SimpleNamespace(
                exp_gene_set_factors=gene_set_factors,
                exp_gene_factors=current_shared_gene_factors.T,
                exp_pheno_factors=None,
            ),
            mass_floor_frac=_DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC,
        )
        overlap_summary = _factor_metric_record_summary(factor_records)
        entropy_summary = _block_usage_entropy_summary(_summarize_block_factor_usage(current_block_rows))
        return {
            "effective_factor_count": float(mass_profile.get("effective_factor_count", 0.0)),
            "mass_ge_floor_factor_count": int(mass_profile.get("mass_ge_floor_factor_count", 0)),
            "median_gene_overlap": float(overlap_summary["median_gene_overlap"]),
            "q90_gene_overlap": float(overlap_summary["q90_gene_overlap"]),
            "median_gene_set_overlap": float(overlap_summary["median_gene_set_overlap"]),
            "q90_gene_set_overlap": float(overlap_summary["q90_gene_set_overlap"]),
            "median_combined_unique_fraction": float(overlap_summary["median_combined_unique_fraction"]),
            "top5_mass_fraction": float(mass_profile.get("top5_mass_fraction", 0.0)),
            "median_block_usage_entropy": float(entropy_summary["median_block_usage_entropy"]),
            "q90_block_usage_entropy": float(entropy_summary["q90_block_usage_entropy"]),
        }

    def _run_online_pass(current_shared_gene_factors, current_lambdak, current_block_rows, *, shuffle_this_pass):
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
            V_block = np.maximum(_dense_block(V, start, stop), 0.0)
            S_block = _compute_weight_matrix_for_block(
                None if gene_set_probabilities is None else gene_set_probabilities[start:stop, :],
                gene_probabilities,
                stop - start,
                M,
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
        updated_shared_gene_factors = np.maximum(current_shared_gene_factors * (numerator_H / denominator_H), 0.0)
        if cap_genes:
            updated_shared_gene_factors = np.clip(updated_shared_gene_factors, 0.0, 1.0)
        updated_lambdak = np.maximum((0.5 * (w_sq_sum + np.sum(updated_shared_gene_factors ** 2, axis=1)) + b0) / C, eps)
        pass_delambda = float(np.max(np.abs(updated_lambdak - current_lambdak) / np.maximum(current_lambdak, eps))) if current_lambdak.size > 0 else 0.0
        regularization = phi_scaled * np.sum((0.5 * (w_sq_sum + np.sum(updated_shared_gene_factors ** 2, axis=1)) + b0) / np.maximum(updated_lambdak, eps) + C * np.log(np.maximum(updated_lambdak, eps)))
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
            },
        )

    def _maybe_write_pass_checkpoint(pass_number, current_shared_gene_factors, current_lambdak, current_block_rows):
        if pass_metrics_dir is None or pass_number < int(_DEFAULT_BLOCKWISE_PASS_CHECKPOINT_START) or pass_number > int(_DEFAULT_BLOCKWISE_PASS_CHECKPOINT_END):
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

    def _reproject_all_blocks_with_fixed_basis(current_shared_gene_factors, current_lambdak, *, iterations, initial_block_rows=None):
        reprojected_rows = {}
        for block_number, (start, stop) in sorted(block_ranges_by_id.items()):
            V_block = np.maximum(_dense_block(V, start, stop), 0.0)
            S_block = _compute_weight_matrix_for_block(
                None if gene_set_probabilities is None else gene_set_probabilities[start:stop, :],
                gene_probabilities,
                stop - start,
                M,
            )
            initial_W_block = None if initial_block_rows is None else initial_block_rows.get(block_number)
            W_block, _, _ = _solve_blockwise_block_with_fixed_basis(
                V_block,
                S_block,
                current_shared_gene_factors,
                current_lambdak,
                phi_scaled=phi_scaled,
                eps=eps,
                cap_gene_sets=cap_gene_sets,
                initial_W_block=initial_W_block,
                vmax=vmax,
                iterations=iterations,
            )
            reprojected_rows[int(block_number)] = W_block
        return reprojected_rows

    max_total_passes = int(min(max(1, max_num_iterations), max(1, passes)))
    min_passes = int(min(max_total_passes, max(1, min_passes_before_stopping)))
    target_passes = int(max_total_passes)
    for pass_index in range(target_passes):
        current_shared_gene_factors = shared_gene_factors
        shared_gene_factors, lambdak, block_rows, pass_stats = _run_online_pass(
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
        report_stats = dict(pass_stats)
        report_stats.update(_compute_pass_diagnostics(shared_gene_factors, lambdak, block_rows))
        report_stats["phase"] = "online"
        report_stats["epoch"] = int(pass_index + 1)
        block_reports.append(report_stats)
        _maybe_write_pass_checkpoint(pass_index + 1, shared_gene_factors, lambdak, block_rows)
        log_fn(
            "Online shared-basis pass %d/%d; backend=online_shared_basis; blocks=%d; rows=%d; err=%.6g; delambda=%.6g; active_factors=%d; warm_start=%s"
            % (
                pass_index + 1,
                int(target_passes),
                int(pass_stats["num_blocks"]),
                int(total_columns_evaluated),
                error,
                delambda,
                int(pass_stats["active_factors"]),
                str(bool(warm_started)),
            ),
            info_level,
        )
        if (pass_index + 1) >= int(min_passes) and delambda < rel_tol:
            break

    block_rows = _reproject_all_blocks_with_fixed_basis(shared_gene_factors, lambdak, iterations=5, initial_block_rows=None)
    gene_set_factors = _materialize_gene_set_factors(block_rows, shared_gene_factors)
    final_iterations = int(len(block_reports))
    state.last_factorization_final_delambda = float(delambda)
    state.last_factorization_iterations = final_iterations
    state.last_factorization_converged = bool(np.isfinite(delambda) and delambda < rel_tol)
    state.last_factorization_hit_iteration_cap = bool(final_iterations >= int(max_total_passes) and not (np.isfinite(delambda) and delambda < rel_tol))
    state.last_factorization_backend = "online_shared_basis"
    state.last_factorization_backend_details = {
        "backend": "online_shared_basis",
        "num_blocks": int(len(ordered_block_ids)),
        "block_size": int(block_size),
        "epochs": int(target_passes),
        "passes_completed": int(final_iterations),
        "online_passes": int(target_passes),
        "online_min_passes_before_stopping": int(min_passes),
        "columns_evaluated": int(total_columns_evaluated),
        "warm_started": bool(warm_started),
        "lambda_cut": float(lambda_cut),
        "epoch_error_trace": [float(value) for value in epoch_error_trace],
        "pass_error_trace": [float(value) for value in epoch_error_trace],
        "wall_time_sec": float(time.time() - started_at),
    }
    state.last_factorization_blockwise_report = block_reports

    if report_out is not None:
        with _open_text_output(report_out) as output_fh:
            output_fh.write("epoch\tphase\tbackend\tnum_blocks\tblock_size\tcolumns_evaluated\terror\tevidence\tlikelihood\tdelambda\tactive_factors\tretained_factors\teffective_factor_count\tmass_ge_floor_factor_count\tmedian_gene_overlap\tq90_gene_overlap\tmedian_gene_set_overlap\tq90_gene_set_overlap\tmedian_combined_unique_fraction\ttop5_mass_fraction\tmedian_block_usage_entropy\tq90_block_usage_entropy\twarm_started\n")
            for report in block_reports:
                output_fh.write(
                    "%d\t%s\t%s\t%d\t%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%d\t%d\t%.12g\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%s\n"
                    % (
                        int(report["epoch"]),
                        str(report.get("phase", "online")),
                        "online_shared_basis",
                        int(report["num_blocks"]),
                        int(block_size),
                        int(report["columns_evaluated"]),
                        float(report["error"]),
                        float(report["evidence"]),
                        float(report["likelihood"]),
                        float(report["delambda"]),
                        int(report["active_factors"]),
                        int(report.get("retained_factors", report["active_factors"])),
                        float(report.get("effective_factor_count", 0.0)),
                        int(report.get("mass_ge_floor_factor_count", 0)),
                        float(report.get("median_gene_overlap", 0.0)),
                        float(report.get("q90_gene_overlap", 0.0)),
                        float(report.get("median_gene_set_overlap", 0.0)),
                        float(report.get("q90_gene_set_overlap", 0.0)),
                        float(report.get("median_combined_unique_fraction", 1.0)),
                        float(report.get("top5_mass_fraction", 0.0)),
                        float(report.get("median_block_usage_entropy", 0.0)),
                        float(report.get("q90_block_usage_entropy", 0.0)),
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
        "reconstruction_error": error,
        "lambdak": np.asarray(lambdak, dtype=float),
        "warm_start_payload": warm_start_payload,
    }


def _fit_blockwise_global_w(*args, **kwargs):
    if "epochs" in kwargs and "passes" not in kwargs:
        kwargs["passes"] = kwargs.pop("epochs")
    if "min_passes_before_stopping" not in kwargs:
        kwargs["min_passes_before_stopping"] = int(kwargs.get("passes", kwargs.get("max_num_iterations", 1)))
    return _fit_online_shared_basis(*args, **kwargs)


def _compute_sketch_row_embedding(matrix, embedding_dim, *, random_seed):
    matrix_csr = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix, dtype=float))
    num_rows, num_cols = matrix_csr.shape
    dim = int(max(1, min(int(embedding_dim), num_cols)))
    rng = np.random.RandomState(0 if random_seed is None else int(random_seed))
    projection = rng.normal(0.0, 1.0 / math.sqrt(float(dim)), size=(num_cols, dim))
    embedding = matrix_csr @ projection
    embedding = np.asarray(embedding, dtype=float)
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms <= 0] = 1.0
    return embedding / row_norms


def _select_projected_cluster_medoids_rows(embedding, sketch_size, *, importance_scores=None):
    embedding = np.asarray(embedding, dtype=float)
    num_rows = int(embedding.shape[0])
    target = int(min(max(1, sketch_size), num_rows))
    if target >= num_rows:
        return list(range(num_rows))
    importance = np.ones(num_rows, dtype=float) if importance_scores is None else np.asarray(importance_scores, dtype=float)
    if importance.shape[0] != num_rows:
        importance = np.ones(num_rows, dtype=float)
    if np.max(importance) > np.min(importance):
        importance = (importance - np.min(importance)) / max(np.max(importance) - np.min(importance), 1e-12)
    else:
        importance = np.zeros(num_rows, dtype=float)
    selected = []
    first_index = int(np.argmax(np.linalg.norm(embedding, axis=1) + 0.05 * importance))
    selected.append(first_index)
    best_similarity = embedding @ embedding[first_index]
    while len(selected) < target:
        candidate_score = (1.0 - np.clip(best_similarity, -1.0, 1.0)) * (1.0 + 0.05 * importance)
        candidate_score[np.asarray(selected, dtype=int)] = -np.inf
        next_index = int(np.argmax(candidate_score))
        if not np.isfinite(candidate_score[next_index]):
            break
        selected.append(next_index)
        best_similarity = np.maximum(best_similarity, embedding @ embedding[next_index])
    if len(selected) < target:
        remaining = [idx for idx in range(num_rows) if idx not in set(selected)]
        selected.extend(remaining[: target - len(selected)])
    return sorted(int(index) for index in selected[:target])


def _fit_structure_preserving_sketch(
    state,
    matrix,
    *,
    gene_set_prob_vector,
    gene_or_pheno_prob_vector,
    max_num_factors,
    max_num_iterations,
    alpha0,
    phi,
    rel_tol,
    sketch_size,
    sketch_embedding_dim,
    sketch_selection_method,
    sketch_random_seed,
    sketch_refinement_passes,
    projection_block_size,
    cap_genes,
    cap_gene_sets,
    log_fn,
    info_level,
    gene_set_importance_scores=None,
):
    resolved_selection_method = _resolve_sketch_selection_method(sketch_selection_method)
    if str(resolved_selection_method) != "projected_cluster_medoids":
        raise ValueError("Unsupported sketch selection method '%s'" % str(sketch_selection_method))
    started_at = time.time()
    V = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix, dtype=float))
    num_rows, num_cols = V.shape
    selected_indices = _select_projected_cluster_medoids_rows(
        _compute_sketch_row_embedding(V, int(sketch_embedding_dim), random_seed=sketch_random_seed),
        int(sketch_size),
        importance_scores=gene_set_importance_scores,
    )
    sketch_matrix = np.asarray(V[selected_indices, :].toarray(), dtype=float)
    sketch_gene_set_prob = np.asarray(gene_set_prob_vector[selected_indices, :], dtype=float)
    exact_result = state._bayes_nmf_l2_extension(
        sketch_matrix,
        sketch_gene_set_prob,
        gene_or_pheno_prob_vector,
        n_iter=max_num_iterations,
        a0=alpha0,
        K=max_num_factors,
        tol=rel_tol,
        phi=phi,
        cap_genes=cap_genes,
        cap_gene_sets=cap_gene_sets,
        normalize_genes=False,
        normalize_gene_sets=False,
    )
    shared_gene_factors = np.asarray(exact_result[1], dtype=float)
    lambdak = np.asarray(exact_result[4], dtype=float)
    phi_scaled = (float(np.std(sketch_matrix)) ** 2) * float(phi)
    eps = 1e-50
    if projection_block_size is None or int(projection_block_size) < 1:
        projection_block_size = int(sketch_size)
    block_ranges_by_id = _blockwise_ranges_by_id(num_rows, int(projection_block_size), max_blocks=None)
    gene_probabilities = _dense_probabilities(gene_or_pheno_prob_vector)
    gene_set_probabilities = _dense_probabilities(gene_set_prob_vector)
    projected_rows = {}
    projection_iterations = int(max(5, 5 + 2 * max(0, int(sketch_refinement_passes))))
    for block_number, (start, stop) in sorted(block_ranges_by_id.items()):
        V_block = np.maximum(_dense_block(V, start, stop), 0.0)
        S_block = _compute_weight_matrix_for_block(
            None if gene_set_probabilities is None else gene_set_probabilities[start:stop, :],
            gene_probabilities,
            stop - start,
            num_cols,
        )
        W_block, _, _ = _solve_blockwise_block_with_fixed_basis(
            V_block,
            S_block,
            shared_gene_factors,
            lambdak,
            phi_scaled=phi_scaled,
            eps=eps,
            cap_gene_sets=cap_gene_sets,
            initial_W_block=None,
            vmax=float(np.max(sketch_matrix)) if sketch_matrix.size > 0 else 1.0,
            iterations=projection_iterations,
        )
        projected_rows[int(block_number)] = W_block
    gene_set_factors = np.vstack([projected_rows[block_id] for block_id in sorted(projected_rows.keys())]) if len(projected_rows) > 0 else np.zeros((num_rows, shared_gene_factors.shape[0]), dtype=float)
    state.last_factorization_backend = "structure_preserving_sketch"
    state.last_factorization_backend_details = {
        "backend": "structure_preserving_sketch",
        "sketch_size": int(len(selected_indices)),
        "selected_gene_sets": int(len(selected_indices)),
        "projected_gene_sets": int(num_rows),
        "embedding_dim": int(sketch_embedding_dim),
        "selection_method": str(resolved_selection_method),
        "projection_block_size": int(projection_block_size),
        "sketch_refinement_passes": int(sketch_refinement_passes),
        "wall_time_sec": float(time.time() - started_at),
    }
    state.last_factorization_blockwise_report = []
    state.last_factorization_final_delambda = None
    state.last_factorization_iterations = int(max_num_iterations)
    state.last_factorization_converged = None
    state.last_factorization_hit_iteration_cap = None
    log_fn(
        "Structure-preserving sketch selected %d actual gene sets from %d using %s (embedding_dim=%d)"
        % (int(len(selected_indices)), int(num_rows), str(resolved_selection_method), int(sketch_embedding_dim)),
        info_level,
    )
    warm_start_payload = {
        "gene_factors": np.asarray(shared_gene_factors, dtype=float).copy(),
        "lambdak": np.asarray(lambdak, dtype=float).copy(),
        "selected_gene_set_indices": [int(index) for index in selected_indices],
        "selection_method": str(resolved_selection_method),
        "sketch_embedding_dim": int(sketch_embedding_dim),
    }
    return {
        "gene_set_factors": gene_set_factors,
        "gene_or_pheno_factors": shared_gene_factors.T,
        "likelihood": exact_result[2],
        "evidence": exact_result[3],
        "reconstruction_error": exact_result[5],
        "lambdak": lambdak,
        "selected_gene_set_indices": selected_indices,
        "warm_start_payload": warm_start_payload,
    }

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


def _derive_pre_projection_output_path(path):
    if path is None:
        return None
    if path.endswith(".gz"):
        base = path[:-3]
        root, ext = os.path.splitext(base)
        return f"{root}.pre_projection{ext}.gz" if ext else f"{base}.pre_projection.gz"
    root, ext = os.path.splitext(path)
    return f"{root}.pre_projection{ext}" if ext else f"{path}.pre_projection"


def _write_pre_projection_checkpoint(
    state,
    *,
    factor_metrics_out,
    gene_set_clusters_out,
    gene_clusters_out,
    log_fn,
    info_level,
):
    checkpoint_factor_metrics_out = _derive_pre_projection_output_path(factor_metrics_out)
    checkpoint_gene_set_clusters_out = _derive_pre_projection_output_path(gene_set_clusters_out)
    checkpoint_gene_clusters_out = _derive_pre_projection_output_path(gene_clusters_out)
    if checkpoint_factor_metrics_out is not None:
        log_fn("Writing factor metrics to %s" % checkpoint_factor_metrics_out, info_level)
        state.write_factor_metrics(checkpoint_factor_metrics_out)
    if checkpoint_gene_set_clusters_out is not None or checkpoint_gene_clusters_out is not None:
        factor_labels = getattr(state, "factor_labels", None)
        if factor_labels is None:
            log_fn("Skipping pre-projection cluster checkpoint because factor labels are unavailable", info_level)
        else:
            log_fn("Writing gene set clusters to %s" % checkpoint_gene_set_clusters_out, info_level)
            state.write_clusters(
                checkpoint_gene_set_clusters_out,
                checkpoint_gene_clusters_out,
                None,
            )


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
    learn_phi_max_fit_loss_frac,
    learn_phi_k_band_frac,
    learn_phi_max_steps,
    learn_phi_expand_factor,
    learn_phi_weight_floor,
    learn_phi_mass_floor_frac,
    learn_phi_min_error_gain_per_factor,
    learn_phi_only,
    learn_phi_report_out,
    factor_phi_metrics_out,
    max_num_gene_sets,
    gene_set_budget_mode,
    learn_phi_gene_set_budget_mode,
    factor_backend,
    learn_phi_backend,
    online_block_size,
    online_passes,
    online_min_passes_before_stopping,
    online_shuffle_blocks,
    online_warm_start,
    online_max_blocks,
    online_report_out,
    approx_projection_block_size,
    approx_projection_n_iter,
    sketch_size,
    sketch_embedding_dim,
    sketch_selection_method,
    sketch_random_seed,
    sketch_refinement_passes,
    learn_phi_scout_repeats,
    learn_phi_confirm_topk,
    learn_phi_confirm_online_passes,
    learn_phi_confirm_block_size,
    learn_phi_scout_selection_method,
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
    keep_original_loadings,
    project_phenos_from_gene_sets,
    pheno_capture_input,
):
    anchor_gene_mask_present, anchor_gene_mask_total, anchor_gene_mask_selected = _summarize_mask(anchor_gene_mask)
    anchor_pheno_mask_present, anchor_pheno_mask_total, anchor_pheno_mask_selected = _summarize_mask(anchor_pheno_mask)
    return {
        "max_num_factors": int(max_num_factors),
        "phi": float(phi),
        "alpha0": float(alpha0),
        "beta0": float(beta0),
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
        "learn_phi_max_fit_loss_frac": float(learn_phi_max_fit_loss_frac),
        "learn_phi_k_band_frac": float(learn_phi_k_band_frac),
        "learn_phi_max_steps": int(learn_phi_max_steps),
        "learn_phi_expand_factor": float(learn_phi_expand_factor),
        "learn_phi_weight_floor": None if learn_phi_weight_floor is None else float(learn_phi_weight_floor),
        "learn_phi_mass_floor_frac": float(learn_phi_mass_floor_frac),
        "learn_phi_min_error_gain_per_factor": float(learn_phi_min_error_gain_per_factor),
        "learn_phi_only": bool(learn_phi_only),
        "learn_phi_report_out": learn_phi_report_out,
        "factor_phi_metrics_out": factor_phi_metrics_out,
        "max_num_gene_sets": None if max_num_gene_sets is None else int(max_num_gene_sets),
        "gene_set_budget_mode": str(gene_set_budget_mode),
        "learn_phi_gene_set_budget_mode": None if learn_phi_gene_set_budget_mode is None else str(learn_phi_gene_set_budget_mode),
        "factor_backend": factor_backend,
        "learn_phi_backend": learn_phi_backend,
        "online_block_size": None if online_block_size is None else int(online_block_size),
        "online_passes": int(online_passes),
        "online_min_passes_before_stopping": int(online_min_passes_before_stopping),
        "online_shuffle_blocks": bool(online_shuffle_blocks),
        "online_warm_start": bool(online_warm_start),
        "online_max_blocks": None if online_max_blocks is None else int(online_max_blocks),
        "online_report_out": online_report_out,
        "approx_projection_block_size": None if approx_projection_block_size is None else int(approx_projection_block_size),
        "approx_projection_n_iter": int(approx_projection_n_iter),
        "sketch_size": None if sketch_size is None else int(sketch_size),
        "sketch_embedding_dim": int(sketch_embedding_dim),
        "sketch_selection_method": str(_resolve_sketch_selection_method(sketch_selection_method)),
        "sketch_random_seed": None if sketch_random_seed is None else int(sketch_random_seed),
        "sketch_refinement_passes": int(sketch_refinement_passes),
        "learn_phi_scout_repeats": int(learn_phi_scout_repeats),
        "learn_phi_confirm_topk": int(learn_phi_confirm_topk),
        "learn_phi_confirm_online_passes": int(learn_phi_confirm_online_passes),
        "learn_phi_confirm_block_size": None if learn_phi_confirm_block_size is None else int(learn_phi_confirm_block_size),
        "learn_phi_scout_selection_method": None if learn_phi_scout_selection_method is None else str(_resolve_sketch_selection_method(learn_phi_scout_selection_method)),
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
        "keep_original_loadings": bool(keep_original_loadings),
        "project_phenos_from_gene_sets": bool(project_phenos_from_gene_sets),
        "pheno_capture_input": pheno_capture_input,
    }


def _resolve_budget_mode_alias(mode_value, *, default_mode):
    mode_str = str(mode_value or default_mode)
    if mode_str in {"pruned", "online_shared_basis", "structure_preserving_sketch"}:
        return mode_str
    if mode_str == "sentinel_pruned":
        return "pruned"
    if mode_str == "blockwise_global_w":
        return "online_shared_basis"
    if mode_str == "full":
        return str(default_mode)
    return str(default_mode)


def _resolve_sketch_selection_method(method_value):
    method_str = str(method_value or "projected_cluster_medoids")
    if method_str == "projected_kmedoids":
        return "projected_cluster_medoids"
    return method_str


def _should_use_blockwise_projection(effective_factor_mode):
    return str(effective_factor_mode) in {"online_shared_basis", "structure_preserving_sketch"}


def _project_remaining_loadings_blockwise(
    state,
    *,
    factor_gene_set_x_pheno,
    run_transpose,
    phi,
    rel_tol,
    cap,
    normalize_genes,
    normalize_gene_sets,
    keep_original_loadings,
    project_phenos_from_gene_sets,
    pheno_capture_input,
    gene_or_pheno_full_prob_vector,
    approx_projection_block_size,
    approx_projection_n_iter,
):
    block_size = None if approx_projection_block_size is None else int(approx_projection_block_size)
    projection_n_iter = int(max(1, approx_projection_n_iter))

    if factor_gene_set_x_pheno:
        if gene_or_pheno_full_prob_vector is not None:
            state.gene_prob_factor_vector = state._nnls_project_matrix_blockwise(
                state.pheno_prob_factor_vector,
                gene_or_pheno_full_prob_vector.T,
                block_size=block_size,
            )
            state._record_params({"factor_gene_prob_from": "phenos"})
        else:
            state.gene_prob_factor_vector = state._nnls_project_matrix_blockwise(
                state.gene_set_prob_factor_vector,
                state.X_orig,
                block_size=block_size,
            )
            state._record_params({"factor_gene_prob_from": "gene_sets"})
    else:
        if gene_or_pheno_full_prob_vector is not None:
            state.pheno_prob_factor_vector = state._nnls_project_matrix_blockwise(
                state.gene_prob_factor_vector,
                gene_or_pheno_full_prob_vector.T,
                block_size=block_size,
            )
            state._record_params({"factor_pheno_prob_from": "genes"})
        elif state.X_phewas_beta_uncorrected is not None:
            state.pheno_prob_factor_vector = state._nnls_project_matrix_blockwise(
                state.gene_set_prob_factor_vector,
                state.X_phewas_beta_uncorrected,
                block_size=block_size,
            )
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

    full_gene_factor_values = state._project_H_with_fixed_W_blockwise(
        state.exp_gene_set_factors,
        gene_matrix_to_project[state.gene_set_factor_gene_set_mask, :],
        state.gene_set_prob_factor_vector[state.gene_set_factor_gene_set_mask, :],
        state.gene_prob_factor_vector,
        block_size=block_size,
        phi=phi,
        tol=rel_tol,
        n_iter=projection_n_iter,
        cap_genes=cap,
        normalize_genes=normalize_genes,
    )
    if not factor_gene_set_x_pheno and keep_original_loadings:
        full_gene_factor_values[state.gene_factor_gene_mask, :] = state.exp_gene_factors

    full_pheno_factor_values = state.exp_pheno_factors
    state.pheno_capture_strength = None
    state.pheno_capture_basis = None
    state.pheno_capture_input = None
    pheno_matrix_to_project = None

    if state.exp_gene_factors is None and state.exp_gene_set_factors is None:
        raise ValueError("Something went wrong: both gene factors and gene set factors are empty")

    if state.X_phewas_beta_uncorrected is not None and state.pheno_prob_factor_vector is not None:
        if project_phenos_from_gene_sets or state.exp_gene_factors is None:
            pheno_matrix_to_project = state.X_phewas_beta_uncorrected.T
            if not run_transpose:
                pheno_matrix_to_project = pheno_matrix_to_project.T
            basis = state.exp_gene_set_factors
            feature_by_pheno = _prepare_pheno_capture_input_matrix(pheno_matrix_to_project, pheno_capture_input)
            basis, feature_by_pheno = _align_projection_inputs_to_mask(
                basis,
                feature_by_pheno,
                state.gene_set_factor_gene_set_mask,
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
                state.gene_factor_gene_mask,
            )
            full_pheno_factor_values = _project_pheno_capture_matrix(
                state,
                basis,
                feature_by_pheno,
                basis_name="genes",
            )

        if keep_original_loadings:
            full_pheno_factor_values[state.pheno_factor_pheno_mask, :] = state.exp_pheno_factors
        state.pheno_capture_input = pheno_capture_input
    elif state.exp_pheno_factors is not None:
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

    if factor_gene_set_x_pheno and pheno_matrix_to_project is not None:
        full_gene_set_factor_values = state._project_H_with_fixed_W_blockwise(
            state.exp_pheno_factors,
            pheno_matrix_to_project[:, state.pheno_factor_pheno_mask].T if run_transpose else pheno_matrix_to_project[state.pheno_factor_pheno_mask, :].T,
            state.pheno_prob_factor_vector[state.pheno_factor_pheno_mask, :],
            state.gene_set_prob_factor_vector,
            block_size=block_size,
            phi=phi,
            tol=rel_tol,
            n_iter=projection_n_iter,
            cap_genes=cap,
            normalize_genes=normalize_gene_sets,
        )
    else:
        full_gene_set_factor_values = state._project_H_with_fixed_W_blockwise(
            state.exp_gene_factors,
            gene_matrix_to_project[:, state.gene_factor_gene_mask].T if run_transpose else gene_matrix_to_project[state.gene_factor_gene_mask, :].T,
            state.gene_prob_factor_vector[state.gene_factor_gene_mask, :],
            state.gene_set_prob_factor_vector,
            block_size=block_size,
            phi=phi,
            tol=rel_tol,
            n_iter=projection_n_iter,
            cap_genes=cap,
            normalize_genes=normalize_gene_sets,
        )

    if keep_original_loadings:
        full_gene_set_factor_values[state.gene_set_factor_gene_set_mask, :] = state.exp_gene_set_factors

    state.exp_gene_factors = full_gene_factor_values
    state.exp_pheno_factors = full_pheno_factor_values
    state.exp_gene_set_factors = full_gene_set_factor_values

    matrix_to_mult = state.exp_pheno_factors if factor_gene_set_x_pheno else state.exp_gene_factors
    vector_to_mult = state.pheno_prob_factor_vector if factor_gene_set_x_pheno else state.gene_prob_factor_vector
    state.factor_anchor_relevance = state._nnls_project_matrix_blockwise(
        matrix_to_mult,
        vector_to_mult.T,
        block_size=block_size,
        max_value=1,
    ).T
    state.factor_relevance = _compute_any_anchor_relevance(state.factor_anchor_relevance)


def _extract_canonical_factor_matrix(state):
    if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0:
        return state.exp_gene_set_factors
    if state.exp_gene_factors is not None and state.exp_gene_factors.size > 0:
        return state.exp_gene_factors
    if state.exp_pheno_factors is not None and state.exp_pheno_factors.size > 0:
        return state.exp_pheno_factors
    return None


def _extract_overlap_basis_matrix(state):
    if state.exp_gene_factors is not None and state.exp_gene_factors.size > 0:
        return ("gene", state.exp_gene_factors)
    if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0:
        return ("gene_set", state.exp_gene_set_factors)
    if state.exp_pheno_factors is not None and state.exp_pheno_factors.size > 0:
        return ("pheno", state.exp_pheno_factors)
    return ("none", None)


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


def _compute_within_run_factor_redundancy_profile(state, weight_floor):
    redundancy_basis, canonical = _extract_overlap_basis_matrix(state)
    if canonical is None or canonical.shape[1] <= 1:
        return {
            "redundancy_basis": redundancy_basis,
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

    sorted_mass_fractions = np.sort(mass_fractions)[::-1]
    return {
        "factor_masses": factor_masses,
        "mass_fractions": mass_fractions,
        "effective_factor_count": effective_factor_count,
        "mass_ge_floor_factor_count": int(np.sum(mass_fractions >= float(mass_floor_frac))),
        "primary_factor_count": int(np.sum(mass_fractions >= 0.005)),
        "secondary_factor_count": int(np.sum((mass_fractions >= 0.0025) & (mass_fractions < 0.005))),
        "filtered_factor_count": int(np.sum(mass_fractions < 0.0025)),
        "tail_fraction": float(np.sum(mass_fractions[mass_fractions < 0.0025])) if mass_fractions.size > 0 else 0.0,
        "filtered_fraction": float(np.mean(mass_fractions < 0.0025)) if mass_fractions.size > 0 else 0.0,
        "max_mass_fraction": float(sorted_mass_fractions[0]) if sorted_mass_fractions.size > 0 else 0.0,
        "top5_mass_fraction": float(np.sum(sorted_mass_fractions[:5])) if sorted_mass_fractions.size > 0 else 0.0,
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


_DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC = 0.005
_LEARN_PHI_MIN_ERROR_GAIN_PER_FACTOR = 5.0
_DEFAULT_BLOCKWISE_K0 = 15
_DEFAULT_BLOCKWISE_PASS_CHECKPOINT_START = 5
_DEFAULT_BLOCKWISE_PASS_CHECKPOINT_END = 12


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


def _summarize_phi_candidate(run_states, run_summaries, *, phi, weight_floor, mass_floor_frac, max_num_factors):
    modal_factor_count, modal_indices = _collect_run_indices_by_modal_factor_count(run_states, run_summaries)
    reference_index = min(modal_indices, key=lambda idx: _best_run_sort_key(run_summaries[idx]))

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
    reference_matrix = _extract_canonical_factor_matrix(run_states[reference_index])
    for index in modal_indices:
        if index == reference_index:
            continue
        matched_cosines.extend(
            _match_factor_cosines(reference_matrix, _extract_canonical_factor_matrix(run_states[index]))
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
        _compute_within_run_factor_redundancy_profile(run_states[index], weight_floor)
        for index in modal_indices
    ]
    redundancy_basis_values = [str(profile.get("redundancy_basis", "unknown")) for profile in run_redundancy_profiles]
    redundancy_basis = redundancy_basis_values[0] if len(set(redundancy_basis_values)) == 1 else "mixed"
    redundancy_max_values = [float(profile["redundancy_max"]) for profile in run_redundancy_profiles]
    redundancy_q90_values = [float(profile["redundancy_q90"]) for profile in run_redundancy_profiles]
    redundancy_mean_values = [float(profile["redundancy_mean"]) for profile in run_redundancy_profiles]
    run_mass_profiles = [
        _compute_factor_mass_profile(run_states[index], mass_floor_frac=mass_floor_frac)
        for index in modal_indices
    ]
    effective_factor_counts = [float(profile["effective_factor_count"]) for profile in run_mass_profiles]
    mass_ge_floor_factor_counts = [int(profile["mass_ge_floor_factor_count"]) for profile in run_mass_profiles]
    primary_factor_counts = [int(profile["primary_factor_count"]) for profile in run_mass_profiles]
    secondary_factor_counts = [int(profile["secondary_factor_count"]) for profile in run_mass_profiles]
    filtered_factor_counts = [int(profile["filtered_factor_count"]) for profile in run_mass_profiles]
    tail_fractions = [float(profile["tail_fraction"]) for profile in run_mass_profiles]
    filtered_fractions = [float(profile["filtered_fraction"]) for profile in run_mass_profiles]
    max_mass_fractions = [float(profile["max_mass_fraction"]) for profile in run_mass_profiles]
    top5_mass_fractions = [float(profile["top5_mass_fraction"]) for profile in run_mass_profiles]
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
    prune_genes_num,
    prune_gene_sets_num,
    max_num_iterations,
    warm_start_payload=None,
    log_fn,
    info_level,
):
    effective_budget_mode = _resolve_budget_mode_alias(
        learn_phi_backend,
        default_mode=str(factor_kwargs.get("gene_set_budget_mode", "pruned")),
    )
    log_fn(
        "Evaluating automatic phi candidate %.6g with %d restart(s) [backend=%s]"
        % (phi, runs_per_step, effective_budget_mode),
        info_level,
    )
    search_factor_kwargs = dict(factor_kwargs)
    search_factor_kwargs["phi"] = phi
    search_factor_kwargs["lmm_auth_key"] = None
    search_factor_kwargs["label_gene_sets_only"] = False
    search_factor_kwargs["label_include_phenos"] = False
    search_factor_kwargs["label_individually"] = False
    search_factor_kwargs["gene_set_budget_mode"] = effective_budget_mode
    if effective_budget_mode == "online_shared_basis":
        search_factor_kwargs["factor_backend"] = "online_shared_basis"
        search_factor_kwargs["blockwise_warm_start_state"] = warm_start_payload
    elif effective_budget_mode == "structure_preserving_sketch":
        search_factor_kwargs["factor_backend"] = "structure_preserving_sketch"
    else:
        search_factor_kwargs["factor_backend"] = "full"
    if prune_genes_num is not None:
        search_factor_kwargs["gene_prune_number"] = int(prune_genes_num)
    if effective_budget_mode == "pruned" and prune_gene_sets_num is not None:
        search_factor_kwargs["gene_set_prune_number"] = int(prune_gene_sets_num)
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
    )
    candidate["run_summaries"] = copy.deepcopy(run_summaries)
    candidate["backend"] = str(effective_budget_mode)
    reference_state = run_states[int(candidate["reference_run_index"])]
    reference_summary = run_summaries[int(candidate["reference_run_index"])]
    if effective_budget_mode == "online_shared_basis":
        candidate["warm_start_payload"] = reference_summary.get("blockwise_warm_start_payload")
    if hasattr(reference_state, "_collect_factor_metrics_records"):
        candidate["factor_metric_rows"] = reference_state._collect_factor_metrics_records()
    else:
        candidate["factor_metric_rows"] = []
    best_error = candidate.get("best_error")
    best_evidence = candidate.get("best_evidence")
    log_fn(
        "Automatic phi candidate %.6g summary: K_eff=%d, K_mass=%.3g, K_mass_ge_floor=%d, capped=%s, redundancy_max[%s]=%.3g, redundancy_q90=%.3g, redundancy_mean=%.3g, stability=%s, run_support=%.3g, best_error=%s, best_evidence=%s, final_delambda=%s, final_iterations=%s, converged_fraction=%s, hit_iteration_cap_fraction=%s"
        % (
            float(candidate["phi"]),
            int(candidate["modal_factor_count"]),
            float(candidate.get("effective_factor_count", 0.0)),
            int(candidate.get("mass_ge_floor_factor_count", 0)),
            bool(candidate.get("capped", False)),
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


def _candidate_complexity_value(candidate):
    effective_factor_count = float(candidate.get("effective_factor_count", 0.0) or 0.0)
    if effective_factor_count > 0:
        return float(effective_factor_count)
    mass_floor_count = int(candidate.get("mass_ge_floor_factor_count", 0) or 0)
    if mass_floor_count > 0:
        return float(mass_floor_count)
    return float(candidate.get("modal_factor_count", 0) or 0)


def _select_candidate_by_effective_k_tail(selection_pool, *, max_fit_loss_frac, k_band_frac):
    if len(selection_pool) == 0:
        return None
    finite_errors = [float(candidate["best_error"]) for candidate in selection_pool if candidate.get("best_error") is not None]
    if len(finite_errors) == 0:
        return None
    best_error = min(finite_errors)
    fit_limit = float(best_error) * (1.0 + float(max_fit_loss_frac))
    fit_eligible = [
        candidate
        for candidate in selection_pool
        if candidate.get("best_error") is not None and float(candidate["best_error"]) <= fit_limit + 1e-12
    ]
    if len(fit_eligible) == 0:
        return None
    best_effective_k = max(_candidate_complexity_value(candidate) for candidate in fit_eligible)
    k_threshold = float(best_effective_k) * float(k_band_frac)
    band_eligible = [
        candidate
        for candidate in fit_eligible
        if _candidate_complexity_value(candidate) >= k_threshold - 1e-12
    ]
    if len(band_eligible) == 0:
        band_eligible = fit_eligible
    selected = min(
        band_eligible,
        key=lambda candidate: (
            -float(candidate.get("phi", 0.0)),
            float(candidate.get("tail_fraction", 0.0)),
            float(candidate.get("filtered_fraction", 0.0)),
            float("inf") if candidate.get("best_error") is None else float(candidate["best_error"]),
            -float(candidate.get("primary_factor_count", 0)),
        ),
    )
    selected["selection_fit_limit"] = fit_limit
    selected["selection_k_threshold"] = k_threshold
    selected["selection_fit_eligible_size"] = int(len(fit_eligible))
    selected["selection_band_size"] = int(len(band_eligible))
    return selected


def _select_phi_candidate(
    candidates,
    *,
    max_redundancy,
    max_redundancy_q90,
    min_run_support,
    min_stability,
    max_fit_loss_frac,
    k_band_frac,
    runs_per_step,
    min_error_gain_per_factor,
):
    acceptable = []
    for candidate in candidates:
        if int(candidate.get("modal_factor_count", 0)) <= 0:
            continue
        if candidate["redundancy_max"] > max_redundancy:
            continue
        if candidate["redundancy_q90"] > max_redundancy_q90:
            continue
        if candidate["run_support"] < min_run_support:
            continue
        if runs_per_step > 1:
            if int(candidate.get("num_modal_runs", 0)) < 2:
                continue
            if not bool(candidate.get("stability_defined", False)):
                continue
            if candidate.get("stability") is None or float(candidate["stability"]) < min_stability:
                continue
        acceptable.append(candidate)

    if len(acceptable) > 0:
        acceptable_uncapped = [candidate for candidate in acceptable if not bool(candidate.get("capped", False))]
        selection_pool = acceptable_uncapped if len(acceptable_uncapped) > 0 else acceptable
        selection_pool_name = "uncapped" if len(acceptable_uncapped) > 0 else "capped"
        selected = _select_candidate_by_effective_k_tail(
            selection_pool,
            max_fit_loss_frac=max_fit_loss_frac,
            k_band_frac=k_band_frac,
        )
        if selected is None:
            selected = min(
                selection_pool,
                key=lambda candidate: (
                    float("inf") if candidate.get("best_error") is None else float(candidate["best_error"]),
                    _candidate_complexity_value(candidate),
                    -float(candidate.get("phi", 0.0)),
                ),
            )
        selected["selection_pool"] = selection_pool_name
        selected["k_band_threshold"] = selected.get("selection_k_threshold")
        selected["selection_frontier_size"] = int(selected.get("selection_band_size", 0))
        selected["selection_marginal_gain"] = None
        return selected, "effective_k_tail_band"

    finite_errors = [float(candidate["best_error"]) for candidate in candidates if candidate.get("best_error") is not None]
    best_global_error = min(finite_errors) if len(finite_errors) > 0 else None

    def _fallback_sort_key(candidate):
        fit_limit = None if best_global_error is None else float(best_global_error) * (1.0 + max_fit_loss_frac)
        fit_violation = 0.0
        if fit_limit is not None and candidate.get("best_error") is not None:
            fit_violation = max(0.0, float(candidate["best_error"]) - fit_limit)
        run_support_violation = max(0.0, min_run_support - float(candidate["run_support"]))
        if runs_per_step > 1:
            if int(candidate.get("num_modal_runs", 0)) < 2 or not bool(candidate.get("stability_defined", False)) or candidate.get("stability") is None:
                stability_violation = float("inf")
            else:
                stability_violation = max(0.0, min_stability - float(candidate["stability"]))
        else:
            stability_violation = 0.0
        return (
            1 if bool(candidate.get("capped", False)) else 0,
            max(0.0, float(candidate.get("redundancy_q90", 0.0)) - max_redundancy_q90),
            max(0.0, float(candidate.get("redundancy_max", 0.0)) - max_redundancy),
            run_support_violation,
            stability_violation,
            fit_violation,
            _candidate_complexity_value(candidate),
            float(candidate.get("redundancy_q90", float("inf"))),
            float(candidate.get("redundancy_max", float("inf"))),
            -float(candidate.get("phi", 0.0)),
        )

    selected = min(candidates, key=_fallback_sort_key)
    selected["selection_pool"] = "fallback"
    selected["k_band_threshold"] = None
    selected["selection_frontier_size"] = 0
    selected["selection_marginal_gain"] = None
    return selected, "fallback_min_constraint_violation"


def _write_phi_search_report(report_path, candidates, *, selected_phi, selection_reason):
    if report_path is None:
        return
    with _open_text_output(report_path) as output_fh:
        output_fh.write(
            "selected\tselection_reason\tphi\tmodal_factor_count\teffective_factor_count\tmass_ge_floor_factor_count\tprimary_factor_count\tsecondary_factor_count\tfiltered_factor_count\ttail_fraction\tfiltered_fraction\tmass_floor_frac\tmax_mass_fraction\ttop5_mass_fraction\tcapped\tnum_modal_runs\trun_support\tstability\tstability_defined\tredundancy_basis\tredundancy_max\tredundancy_q90\tredundancy_mean\tredundancy_max_worst\tbest_error\tbest_evidence\tfinal_delambda\tfinal_iterations\tconverged_fraction\thit_iteration_cap_fraction\tbackend\tblockwise_num_blocks\tblockwise_block_size\tblockwise_epochs\tblockwise_columns_evaluated\tblockwise_warm_started\tblockwise_wall_time_sec\tblockwise_epoch_error_trace\treference_run_index\tmodal_run_indices\tmatched_cosines\n"
        )
        for candidate in sorted(candidates, key=lambda row: float(row["phi"])):
            output_fh.write(
                "%s\t%s\t%.12g\t%d\t%.6g\t%d\t%d\t%d\t%d\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%s\t%d\t%.6g\t%s\t%s\t%s\t%.6g\t%.6g\t%.6g\t%.6g\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n"
                % (
                    "1" if math.isclose(float(candidate["phi"]), float(selected_phi), rel_tol=1e-12, abs_tol=1e-15) else "0",
                    selection_reason,
                    float(candidate["phi"]),
                    int(candidate["modal_factor_count"]),
                    float(candidate.get("effective_factor_count", 0.0)),
                    int(candidate.get("mass_ge_floor_factor_count", 0)),
                    int(candidate.get("primary_factor_count", 0)),
                    int(candidate.get("secondary_factor_count", 0)),
                    int(candidate.get("filtered_factor_count", 0)),
                    float(candidate.get("tail_fraction", 0.0)),
                    float(candidate.get("filtered_fraction", 0.0)),
                    float(candidate.get("mass_floor_frac", _DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC)),
                    float(candidate.get("max_mass_fraction", 0.0)),
                    float(candidate.get("top5_mass_fraction", 0.0)),
                    "1" if bool(candidate.get("capped", False)) else "0",
                    int(candidate.get("num_modal_runs", 0)),
                    float(candidate["run_support"]),
                    "" if candidate.get("stability") is None else "%.6g" % float(candidate["stability"]),
                    "1" if bool(candidate.get("stability_defined", False)) else "0",
                    str(candidate.get("redundancy_basis", "unknown")),
                    float(candidate.get("redundancy_max", candidate.get("redundancy", 0.0))),
                    float(candidate.get("redundancy_q90", 0.0)),
                    float(candidate.get("redundancy_mean", 0.0)),
                    float(candidate.get("redundancy_max_worst", 0.0)),
                    "" if candidate.get("best_error") is None else "%.12g" % float(candidate["best_error"]),
                    "" if candidate.get("best_evidence") is None else "%.12g" % float(candidate["best_evidence"]),
                    "" if candidate.get("final_delambda") is None else "%.12g" % float(candidate["final_delambda"]),
                    "" if candidate.get("final_iterations") is None else str(int(candidate["final_iterations"])),
                    "" if candidate.get("converged_fraction") is None else "%.6g" % float(candidate["converged_fraction"]),
                    "" if candidate.get("hit_iteration_cap_fraction") is None else "%.6g" % float(candidate["hit_iteration_cap_fraction"]),
                    str(candidate.get("backend", "full")),
                    "" if candidate.get("blockwise_num_blocks") is None else str(int(candidate["blockwise_num_blocks"])),
                    "" if candidate.get("blockwise_block_size") is None else str(int(candidate["blockwise_block_size"])),
                    "" if candidate.get("blockwise_epochs") is None else str(int(candidate["blockwise_epochs"])),
                    "" if candidate.get("blockwise_columns_evaluated") is None else str(int(candidate["blockwise_columns_evaluated"])),
                    "1" if bool(candidate.get("blockwise_warm_started", False)) else "0",
                    "" if candidate.get("blockwise_wall_time_sec") is None else "%.6g" % float(candidate["blockwise_wall_time_sec"]),
                    ",".join(["%.6g" % float(value) for value in candidate.get("blockwise_epoch_error_trace", [])]),
                    int(candidate["reference_run_index"]),
                    ",".join([str(index) for index in candidate.get("modal_run_indices", [])]),
                    ",".join(["%.6g" % float(value) for value in candidate.get("matched_cosines", [])]),
                )
            )


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
    with _open_text_output(report_path) as output_fh:
        output_fh.write("%s\n" % "\t".join(["phi", "selected", "reference_run_index"] + metric_columns))
        for candidate in sorted(candidates, key=lambda row: float(row["phi"])):
            selected_flag = "1" if math.isclose(float(candidate["phi"]), float(selected_phi), rel_tol=1e-12, abs_tol=1e-15) else "0"
            for record in candidate.get("factor_metric_rows") or []:
                output_fh.write(
                    "%s\n"
                    % "\t".join(
                        [
                            "%.12g" % float(candidate["phi"]),
                            selected_flag,
                            str(int(candidate.get("reference_run_index", 0))),
                        ] + [str(record.get(column, "")) for column in metric_columns]
                    )
                )


def _record_phi_search_params(
    state,
    *,
    initial_phi,
    selected_candidate,
    selection_reason,
    candidates,
    weight_floor,
    max_redundancy,
    max_redundancy_q90,
    runs_per_step,
    min_run_support,
    min_stability,
    max_fit_loss_frac,
    k_band_frac,
    max_steps,
    expand_factor,
    mass_floor_frac,
    min_error_gain_per_factor,
    learn_phi_backend,
    prune_genes_num,
    prune_gene_sets_num,
    max_num_iterations,
):
    state._record_params(
        {
            "learn_phi": True,
            "learn_phi_initial_phi": float(initial_phi),
            "learn_phi_selected_phi": float(selected_candidate["phi"]),
            "learn_phi_selection_reason": selection_reason,
            "learn_phi_max_redundancy": float(max_redundancy),
            "learn_phi_max_redundancy_q90": float(max_redundancy_q90),
            "learn_phi_runs_per_step": int(runs_per_step),
            "learn_phi_min_run_support": float(min_run_support),
            "learn_phi_min_stability": float(min_stability),
            "learn_phi_max_fit_loss_frac": float(max_fit_loss_frac),
            "learn_phi_k_band_frac": float(k_band_frac),
            "learn_phi_mass_floor_frac": float(mass_floor_frac),
            "learn_phi_min_error_gain_per_factor": float(min_error_gain_per_factor),
            "learn_phi_backend": str(learn_phi_backend),
            "learn_phi_max_steps": int(max_steps),
            "learn_phi_expand_factor": float(expand_factor),
            "learn_phi_weight_floor": float(weight_floor),
            "learn_phi_prune_genes_num": None if prune_genes_num is None else int(prune_genes_num),
            "learn_phi_prune_gene_sets_num": None if prune_gene_sets_num is None else int(prune_gene_sets_num),
            "learn_phi_max_num_iterations": None if max_num_iterations is None else int(max_num_iterations),
            "learn_phi_redundancy_basis": str(selected_candidate.get("redundancy_basis", "unknown")),
            "learn_phi_selection_pool": str(selected_candidate.get("selection_pool", "unknown")),
            "learn_phi_selection_frontier_size": int(selected_candidate.get("selection_frontier_size", 0)),
            "learn_phi_selection_marginal_gain": selected_candidate.get("selection_marginal_gain"),
        },
        overwrite=True,
    )
    metric_map = {
        "learn_phi_candidate_phi": "phi",
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
    max_fit_loss_frac,
    k_band_frac,
    max_steps,
    expand_factor,
    weight_floor,
    mass_floor_frac=_DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC,
    min_error_gain_per_factor=_LEARN_PHI_MIN_ERROR_GAIN_PER_FACTOR,
    learn_phi_backend="sentinel_pruned",
    blockwise_warm_start=True,
    report_out,
    factor_phi_metrics_out=None,
    prune_genes_num,
    prune_gene_sets_num,
    max_num_iterations,
    factor_kwargs,
    log_fn,
    info_level,
):
    effective_learn_phi_mode = _resolve_budget_mode_alias(
        learn_phi_backend,
        default_mode=str(factor_kwargs.get("gene_set_budget_mode", "pruned")),
    )
    candidates_by_phi = {}

    remaining_evaluations = int(max_steps)

    def _evaluate(phi_value, *, consume_budget=True):
        nonlocal remaining_evaluations
        existing = _find_candidate_by_phi(candidates_by_phi, float(phi_value))
        if existing is not None:
            return existing
        if consume_budget and remaining_evaluations <= 0:
            return None
        warm_start_payload = None
        if effective_learn_phi_mode == "online_shared_basis" and blockwise_warm_start and len(candidates_by_phi) > 0:
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
            "mass_floor_frac": mass_floor_frac,
            "prune_genes_num": prune_genes_num,
            "prune_gene_sets_num": prune_gene_sets_num,
            "max_num_iterations": max_num_iterations,
            "log_fn": log_fn,
            "info_level": info_level,
        }
        if effective_learn_phi_mode != "pruned" or warm_start_payload is not None:
            evaluate_kwargs["learn_phi_backend"] = effective_learn_phi_mode
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

    def _factor_count(candidate):
        return int(candidate.get("modal_factor_count", 0))

    def _is_capped(candidate):
        return bool(candidate.get("capped", False))

    def _refine_bracket(low_phi, high_phi, *, predicate):
        low_phi = _clip_phi(low_phi)
        high_phi = _clip_phi(high_phi)
        while remaining_evaluations > 0:
            if not (low_phi < high_phi):
                break
            mid_phi = math.sqrt(low_phi * high_phi)
            if math.isclose(mid_phi, low_phi, rel_tol=1e-12, abs_tol=1e-15) or math.isclose(mid_phi, high_phi, rel_tol=1e-12, abs_tol=1e-15):
                break
            candidate = _evaluate(mid_phi)
            if candidate is None:
                break
            if predicate(candidate):
                low_phi = float(candidate["phi"])
            else:
                high_phi = float(candidate["phi"])

    def _pick_better(candidates_subset):
        if len(candidates_subset) == 0:
            return None
        if len(candidates_subset) == 1:
            return candidates_subset[0]
        selected, _ = _select_phi_candidate(
            list(candidates_subset),
            max_redundancy=max_redundancy,
            max_redundancy_q90=max_redundancy_q90,
            min_run_support=min_run_support,
            min_stability=min_stability,
            max_fit_loss_frac=max_fit_loss_frac,
            k_band_frac=k_band_frac,
            runs_per_step=runs_per_step,
            min_error_gain_per_factor=min_error_gain_per_factor,
        )
        return selected

    def _adjacent_candidates(center_phi):
        sorted_phis = sorted(float(value) for value in candidates_by_phi.keys())
        center_index = None
        for index, phi_value in enumerate(sorted_phis):
            if math.isclose(phi_value, float(center_phi), rel_tol=1e-12, abs_tol=1e-15):
                center_index = index
                break
        if center_index is None:
            return None, None
        lower_candidate = None if center_index == 0 else candidates_by_phi[sorted_phis[center_index - 1]]
        upper_candidate = None if center_index >= len(sorted_phis) - 1 else candidates_by_phi[sorted_phis[center_index + 1]]
        return lower_candidate, upper_candidate

    def _proposed_phi(center_phi, neighbor_candidate, *, lower_direction):
        center_phi = float(center_phi)
        if neighbor_candidate is None:
            proposal = center_phi / expand_factor if lower_direction else center_phi * expand_factor
        else:
            proposal = math.sqrt(float(neighbor_candidate["phi"]) * center_phi)
        proposal = _clip_phi(proposal)
        existing = _find_candidate_by_phi(candidates_by_phi, proposal)
        if existing is not None:
            return None
        if math.isclose(proposal, center_phi, rel_tol=1e-12, abs_tol=1e-15):
            return None
        return proposal

    initial_candidate = _evaluate(initial_phi, consume_budget=False)
    focus_candidate = initial_candidate
    bootstrapped = False

    while remaining_evaluations > 0:
        lower_neighbor, upper_neighbor = _adjacent_candidates(float(focus_candidate["phi"]))
        if lower_neighbor is not None and upper_neighbor is not None:
            preferred_neighbor = _pick_better([lower_neighbor, upper_neighbor])
            preferred_direction = "lower" if preferred_neighbor is lower_neighbor else "upper"
        elif lower_neighbor is not None:
            preferred_direction = "upper"
        elif upper_neighbor is not None:
            preferred_direction = "lower"
        else:
            preferred_direction = "lower"

        directions = [preferred_direction, "upper" if preferred_direction == "lower" else "lower"]
        new_candidates = []
        for direction in directions:
            if remaining_evaluations <= 0:
                break
            proposal = _proposed_phi(
                float(focus_candidate["phi"]),
                lower_neighbor if direction == "lower" else upper_neighbor,
                lower_direction=(direction == "lower"),
            )
            if proposal is None:
                continue
            candidate = _evaluate(proposal)
            if candidate is not None:
                new_candidates.append(candidate)

        if len(new_candidates) == 0:
            break

        best_new = _pick_better(new_candidates)
        if not bootstrapped:
            focus_candidate = best_new
            bootstrapped = True
            continue

        better_focus = _pick_better([focus_candidate, best_new])
        if better_focus is None or math.isclose(float(better_focus["phi"]), float(focus_candidate["phi"]), rel_tol=1e-12, abs_tol=1e-15):
            break
        focus_candidate = better_focus

    candidates = list(candidates_by_phi.values())
    selected_candidate, selection_reason = _select_phi_candidate(
        candidates,
        max_redundancy=max_redundancy,
        max_redundancy_q90=max_redundancy_q90,
        min_run_support=min_run_support,
        min_stability=min_stability,
        max_fit_loss_frac=max_fit_loss_frac,
        k_band_frac=k_band_frac,
        runs_per_step=runs_per_step,
        min_error_gain_per_factor=min_error_gain_per_factor,
    )
    _record_phi_search_params(
        state,
        initial_phi=initial_phi,
        selected_candidate=selected_candidate,
        selection_reason=selection_reason,
        candidates=candidates,
        weight_floor=weight_floor,
        max_redundancy=max_redundancy,
        max_redundancy_q90=max_redundancy_q90,
        runs_per_step=runs_per_step,
        min_run_support=min_run_support,
        min_stability=min_stability,
        max_fit_loss_frac=max_fit_loss_frac,
        k_band_frac=k_band_frac,
        max_steps=max_steps,
        expand_factor=expand_factor,
        mass_floor_frac=mass_floor_frac,
        min_error_gain_per_factor=min_error_gain_per_factor,
        learn_phi_backend=effective_learn_phi_mode,
        prune_genes_num=prune_genes_num,
        prune_gene_sets_num=prune_gene_sets_num,
        max_num_iterations=max_num_iterations,
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
    log_fn(
        "Selected phi %.6g by automatic tuning [%s]: backend=%s, K_eff=%d, K_mass=%.3g, K_mass_ge_floor=%d, capped=%s, pool=%s, marginal_gain=%s, redundancy_max[%s]=%.3g, redundancy_q90=%.3g, stability=%s, run_support=%.3g"
        % (
            float(selected_candidate["phi"]),
            selection_reason,
            str(selected_candidate.get("backend", effective_learn_phi_mode)),
            int(selected_candidate["modal_factor_count"]),
            float(selected_candidate.get("effective_factor_count", 0.0)),
            int(selected_candidate.get("mass_ge_floor_factor_count", 0)),
            bool(selected_candidate.get("capped", False)),
            str(selected_candidate.get("selection_pool", "unknown")),
            "NA" if selected_candidate.get("selection_marginal_gain") is None else "%.3g" % float(selected_candidate["selection_marginal_gain"]),
            str(selected_candidate.get("redundancy_basis", "unknown")),
            float(selected_candidate.get("redundancy_max", selected_candidate.get("redundancy", 0.0))),
            float(selected_candidate.get("redundancy_q90", 0.0)),
            "NA" if selected_candidate.get("stability") is None else "%.3g" % float(selected_candidate["stability"]),
            float(selected_candidate["run_support"]),
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


def _finalize_factor_outputs(
    state,
    *,
    factor_gene_set_x_pheno,
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
):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level

    reorder_inds = np.argsort(-state.factor_relevance)

    state.exp_lambdak = state.exp_lambdak[reorder_inds]
    state.factor_anchor_relevance = state.factor_anchor_relevance[reorder_inds, :]
    state.factor_relevance = state.factor_relevance[reorder_inds]
    if state.exp_gene_factors is not None:
        state.exp_gene_factors = state.exp_gene_factors[:, reorder_inds]
    if state.exp_pheno_factors is not None:
        state.exp_pheno_factors = state.exp_pheno_factors[:, reorder_inds]
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
    exp_gene_factors_for_top = state.get_factor_loadings(state.exp_gene_factors, loading_type="combined")
    exp_pheno_factors_for_top = state.get_factor_loadings(state.exp_pheno_factors, loading_type="combined")
    exp_gene_set_factors_for_top = state.get_factor_loadings(state.exp_gene_set_factors, loading_type="combined")

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

    top_anchor_gene_set_inds = np.swapaxes(
        np.argsort(-exp_gene_set_factors_for_top.T[:, :, np.newaxis] * state.gene_set_prob_factor_vector[np.newaxis, :, :], axis=1)[:, :num_top, :],
        0,
        1,
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
                np.argsort(
                    -(1 - np.prod(1 - ((exp_gene_factors_for_top).T[:, :, np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis, :, :]), axis=2)),
                    axis=1,
                )[:, :num_top],
                0,
                1,
            )
    else:
        top_gene_or_pheno_inds = np.swapaxes(
            np.argsort(
                -(1 - np.prod(1 - ((exp_gene_factors_for_top).T[:, :, np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis, :, :]), axis=2)),
                axis=1,
            )[:, :num_top],
            0,
            1,
        )
        if exp_pheno_factors_for_top is not None:
            top_pheno_or_gene_inds = top_capture_inds

    top_gene_set_inds = np.swapaxes(
        np.argsort(-(1 - np.prod(1 - (exp_gene_set_factors_for_top.T[:, :, np.newaxis] * state.gene_set_prob_factor_vector[np.newaxis, :, :]), axis=2)), axis=1)[:, :num_top],
        0,
        1,
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


def _run_factor_single(state, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, lmm_model=None, lmm_provider="openai", label_gene_sets_only=False, label_include_phenos=False, label_individually=False, keep_original_loadings=False, project_phenos_from_gene_sets=False, pheno_capture_input="weighted_thresholded", max_num_gene_sets=None, gene_set_budget_mode="pruned", factor_backend="full", online_block_size=None, online_passes=20, online_min_passes_before_stopping=5, online_shuffle_blocks=True, online_warm_start=True, online_max_blocks=None, online_report_out=None, approx_projection_block_size=None, approx_projection_n_iter=100, sketch_size=None, sketch_embedding_dim=16, sketch_selection_method="projected_cluster_medoids", sketch_random_seed=None, sketch_refinement_passes=0, blockwise_warm_start_state=None, factors_out=None, factor_metrics_out=None, gene_set_clusters_out=None, gene_clusters_out=None, *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, labeling_module):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    DEBUG = debug_level
    TRACE = trace_level

    if state.X_orig is None:
        bail("Cannot run factoring without X")

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

                state.phenos.append(state.default_pheno)
                state.default_pheno_mask = np.append(np.full(len(state.phenos), False), True)

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
    gene_or_pheno_full_vector, gene_or_pheno_filter_type = _choose_gene_or_pheno_anchor_source(
        combined_prior_Ys,
        priors,
        Y,
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
                gene_sort_rank_source = np.nan_to_num(
                    np.asarray(gene_sort_rank_source, dtype=float),
                    nan=-np.inf,
                    posinf=np.inf,
                    neginf=-np.inf,
                )
                gene_sort_rank = -gene_sort_rank_source
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


    gene_set_sort_rank = _gene_set_sort_rank_for_pruning(
        state,
        betas=betas,
        betas_uncorrected=betas_uncorrected,
    )

    resolved_budget_mode = str(gene_set_budget_mode or "pruned")
    effective_gene_set_budget = None if max_num_gene_sets is None else int(max_num_gene_sets)
    if factor_backend in {"blockwise_global_w", "online_shared_basis"}:
        resolved_budget_mode = "online_shared_basis"
        if effective_gene_set_budget is None:
            effective_gene_set_budget = int(online_block_size) if online_block_size is not None else None
    elif factor_backend == "structure_preserving_sketch":
        resolved_budget_mode = "structure_preserving_sketch"
        if effective_gene_set_budget is None:
            effective_gene_set_budget = int(sketch_size) if sketch_size is not None else None
    effective_pruned_budget = gene_set_prune_number
    if (
        resolved_budget_mode == "pruned"
        and effective_gene_set_budget is not None
        and int(np.sum(gene_set_mask)) > int(effective_gene_set_budget)
    ):
        effective_pruned_budget = int(effective_gene_set_budget) if effective_pruned_budget is None else min(int(effective_pruned_budget), int(effective_gene_set_budget))

    if gene_set_prune_value is not None or gene_set_prune_number is not None:
        log("Pruning gene sets to reduce matrix size", DEBUG)

    if gene_set_prune_value is not None:
        gene_set_prune_mask = state._prune_gene_sets(gene_set_prune_value, X_orig=state.X_orig[:,gene_set_mask], gene_sets=[state.gene_sets[i] for i in np.where(gene_set_mask)[0]], rank_vector=gene_set_sort_rank[gene_set_mask], do_internal_pruning=False)

        log("Found %d gene_sets remaining after pruning (of %d)" % (np.sum(gene_set_prune_mask), len(state.gene_sets)))
        gene_set_mask[np.where(gene_set_mask)[0][~gene_set_prune_mask]] = False

    if effective_pruned_budget is not None:
        gene_set_prune_number_masks = _compute_gene_set_prune_number_masks(
            state,
            gene_set_mask=gene_set_mask,
            gene_set_sort_rank=gene_set_sort_rank,
            gene_set_prune_number=effective_pruned_budget,
        )

        all_gene_set_prune_mask = _combine_prune_masks(
            gene_set_prune_number_masks,
            effective_pruned_budget,
            gene_set_sort_rank[gene_set_mask],
            "gene set",
            log_fn=log,
            trace_level=TRACE,
        )

        gene_set_mask[np.where(gene_set_mask)[0][~all_gene_set_prune_mask]] = False
    
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

    state.gene_set_prob_vector = gene_set_full_prob_vector

    state._record_params({"max_num_factors": max_num_factors, "alpha0": alpha0, "phi": phi, "gene_set_filter_type": gene_set_filter_type, "gene_set_filter_value": gene_set_filter_value, "gene_or_pheno_filter_type": gene_or_pheno_filter_type, "gene_or_pheno_filter_value": gene_or_pheno_filter_value, "pheno_prune_value": pheno_prune_value, "pheno_prune_number": pheno_prune_number, "gene_set_prune_value": gene_set_prune_value, "gene_set_prune_number": gene_set_prune_number, "run_transpose": run_transpose})


    matrix = state.X_phewas_beta_uncorrected.T if factor_gene_set_x_pheno else state.X_orig.T

    matrix = matrix[gene_set_mask,:][:,gene_or_pheno_mask]
    matrix[matrix < 0] = 0
    if not run_transpose:
        matrix = matrix.T

    num_retained_gene_sets = int(matrix.shape[0])
    effective_factor_mode = "full"
    budget_is_binding = bool(effective_gene_set_budget is not None and num_retained_gene_sets > int(effective_gene_set_budget))
    if budget_is_binding and resolved_budget_mode == "online_shared_basis":
        effective_factor_mode = "online_shared_basis"
    elif budget_is_binding and resolved_budget_mode == "structure_preserving_sketch":
        effective_factor_mode = "structure_preserving_sketch"

    if effective_factor_mode in {"online_shared_basis", "structure_preserving_sketch"} and not run_transpose:
        bail("--gene-set-budget-mode %s currently requires the default transposed factor matrix" % effective_factor_mode)

    log("Running matrix factorization with mode=%s (requested_budget_mode=%s, budget=%s, retained_gene_sets=%d)" % (effective_factor_mode, resolved_budget_mode, str(effective_gene_set_budget), num_retained_gene_sets))
    if np.sum(~gene_or_pheno_mask) > 0 or np.sum(~gene_set_mask) > 0:
        log("Filtered original matrix from (%s, %s) to (%s, %s)" % (len(gene_or_pheno_mask), len(gene_set_mask), sum(gene_or_pheno_mask), sum(gene_set_mask)))
    log("Matrix to factor shape: (%s, %s)" % (matrix.shape), DEBUG)

    if np.max(matrix.shape) == 0:
        log("Skipping factoring since there aren't enough significant genes and gene sets")
        return

    if np.min(matrix.shape) == 0:
        log("Empty genes or gene sets! Skipping factoring")
        return

    #constrain loadings to be at most 1, but don't require them to sum to 1
    normalize_genes = False
    normalize_gene_sets = False
    cap = True

    effective_block_size = int(online_block_size) if online_block_size is not None else int(effective_gene_set_budget if effective_gene_set_budget is not None else max(1, num_retained_gene_sets))
    effective_block_count = int(
        min(
            math.ceil(float(matrix.shape[0]) / float(max(1, effective_block_size))),
            int(online_max_blocks) if online_max_blocks is not None else math.ceil(float(matrix.shape[0]) / float(max(1, effective_block_size))),
        )
    ) if matrix.shape[0] > 0 else 0

    delegated_single_block_to_full = bool(effective_factor_mode == "online_shared_basis" and effective_block_count <= 1)
    if delegated_single_block_to_full:
        log(
            "Delegating single-block online_shared_basis run to the full solver for exact equivalence",
            INFO,
        )

    if effective_factor_mode == "online_shared_basis" and not delegated_single_block_to_full:
        online_started_at = time.time()
        blockwise_pass_metrics_dir = _derive_blockwise_pass_metrics_dir(factor_metrics_out, online_report_out)
        result = _fit_online_shared_basis(
            state,
            matrix,
            gene_set_prob_vector=gene_set_prob_vector[gene_set_mask, :],
            gene_or_pheno_prob_vector=gene_or_pheno_prob_vector[gene_or_pheno_mask, :],
            max_num_factors=max_num_factors,
            max_num_iterations=max_num_iterations,
            alpha0=alpha0,
            phi=phi,
            rel_tol=rel_tol,
            min_lambda_threshold=min_lambda_threshold,
            block_size=effective_block_size,
            passes=online_passes,
            min_passes_before_stopping=online_min_passes_before_stopping,
            shuffle_blocks=online_shuffle_blocks,
            max_blocks=online_max_blocks,
            warm_start_state=blockwise_warm_start_state,
            warm_start_enabled=online_warm_start,
            cap_genes=cap,
            cap_gene_sets=cap,
            report_out=online_report_out,
            pass_metrics_dir=blockwise_pass_metrics_dir,
            log_fn=log,
            info_level=INFO,
        )
        state.last_factorization_backend = "online_shared_basis"
        state.last_factorization_backend_details["wall_time_sec"] = float(time.time() - online_started_at)
        state.exp_lambdak = np.asarray(result["lambdak"], dtype=float)
        exp_gene_or_pheno_factors = np.asarray(result["gene_or_pheno_factors"], dtype=float)
        state.exp_gene_set_factors = np.asarray(result["gene_set_factors"], dtype=float)
        evidence_value = result["evidence"]
        likelihood_value = result["likelihood"]
        reconstruction_error_value = result["reconstruction_error"]
        blockwise_warm_start_payload = result.get("warm_start_payload")
    elif effective_factor_mode == "structure_preserving_sketch":
        sketch_started_at = time.time()
        sketch_result = _fit_structure_preserving_sketch(
            state,
            matrix,
            gene_set_prob_vector=gene_set_prob_vector[gene_set_mask, :],
            gene_or_pheno_prob_vector=gene_or_pheno_prob_vector[gene_or_pheno_mask, :],
            max_num_factors=max_num_factors,
            max_num_iterations=max_num_iterations,
            alpha0=alpha0,
            phi=phi,
            rel_tol=rel_tol,
            sketch_size=int(sketch_size) if sketch_size is not None else int(effective_gene_set_budget),
            sketch_embedding_dim=int(sketch_embedding_dim),
            sketch_selection_method=_resolve_sketch_selection_method(sketch_selection_method),
            sketch_random_seed=sketch_random_seed,
            sketch_refinement_passes=sketch_refinement_passes,
            projection_block_size=approx_projection_block_size if approx_projection_block_size is not None else effective_block_size,
            cap_genes=cap,
            cap_gene_sets=cap,
            log_fn=log,
            info_level=INFO,
            gene_set_importance_scores=gene_set_sort_rank[gene_set_mask],
        )
        state.last_factorization_backend_details["wall_time_sec"] = float(time.time() - sketch_started_at)
        state.exp_lambdak = np.asarray(sketch_result["lambdak"], dtype=float)
        exp_gene_or_pheno_factors = np.asarray(sketch_result["gene_or_pheno_factors"], dtype=float)
        state.exp_gene_set_factors = np.asarray(sketch_result["gene_set_factors"], dtype=float)
        evidence_value = sketch_result["evidence"]
        likelihood_value = sketch_result["likelihood"]
        reconstruction_error_value = sketch_result["reconstruction_error"]
        blockwise_warm_start_payload = None
    else:
        result = state._bayes_nmf_l2_extension(
            matrix.toarray(),
            gene_set_prob_vector[gene_set_mask,:],
            gene_or_pheno_prob_vector[gene_or_pheno_mask,:],
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
        state.last_factorization_backend = "online_shared_basis" if delegated_single_block_to_full else "full"
        state.last_factorization_backend_details = {
            "backend": "online_shared_basis" if delegated_single_block_to_full else "full",
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
        state.exp_lambdak = result[4]
        exp_gene_or_pheno_factors = result[1].T
        state.exp_gene_set_factors = result[0]
        evidence_value = result[3]
        likelihood_value = result[2]
        reconstruction_error_value = result[5]
        blockwise_warm_start_payload = None

    #subset_out the weak factors
    lambda_keep_threshold = float(min_lambda_threshold)
    if effective_factor_mode == "online_shared_basis":
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
        state.pheno_factor_pheno_mask = gene_or_pheno_mask
        state.exp_pheno_factors = exp_gene_or_pheno_factors
        state.pheno_prob_factor_vector = gene_or_pheno_prob_vector
        state.gene_prob_factor_vector = None
    else:
        state.gene_factor_gene_mask = gene_or_pheno_mask            
        state.exp_gene_factors = exp_gene_or_pheno_factors
        state.gene_prob_factor_vector = gene_or_pheno_prob_vector
        state.pheno_prob_factor_vector = None

    state.gene_set_prob_factor_vector = gene_set_prob_vector
    state.gene_set_factor_gene_set_mask = gene_set_mask

    _write_pre_projection_checkpoint(
        state,
        factor_metrics_out=factor_metrics_out,
        gene_set_clusters_out=gene_set_clusters_out,
        gene_clusters_out=gene_clusters_out,
        log_fn=log,
        info_level=INFO,
    )

    #now project the additional genes/phenos/gene sets onto the factors

    log("Projecting factors", TRACE)
    if _should_use_blockwise_projection(effective_factor_mode):
        _project_remaining_loadings_blockwise(
            state,
            factor_gene_set_x_pheno=factor_gene_set_x_pheno,
            run_transpose=run_transpose,
            phi=phi,
            rel_tol=rel_tol,
            cap=cap,
            normalize_genes=normalize_genes,
            normalize_gene_sets=normalize_gene_sets,
            keep_original_loadings=keep_original_loadings,
            project_phenos_from_gene_sets=project_phenos_from_gene_sets,
            pheno_capture_input=pheno_capture_input,
            gene_or_pheno_full_prob_vector=gene_or_pheno_full_prob_vector,
            approx_projection_block_size=approx_projection_block_size if approx_projection_block_size is not None else effective_block_size,
            approx_projection_n_iter=approx_projection_n_iter,
        )
    else:
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

        full_gene_factor_values = state._project_H_with_fixed_W(state.exp_gene_set_factors, gene_matrix_to_project[state.gene_set_factor_gene_set_mask,:], state.gene_set_prob_factor_vector[state.gene_set_factor_gene_set_mask,:], state.gene_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
        if not factor_gene_set_x_pheno and keep_original_loadings:
            full_gene_factor_values[state.gene_factor_gene_mask,:] = state.exp_gene_factors

        full_pheno_factor_values = state.exp_pheno_factors
        state.pheno_capture_strength = None
        state.pheno_capture_basis = None
        state.pheno_capture_input = None
        pheno_matrix_to_project = None

        if state.exp_gene_factors is None and state.exp_gene_set_factors is None:
            bail("Something went wrong: both gene factors and gene set factors are empty")

        if state.X_phewas_beta_uncorrected is not None and state.pheno_prob_factor_vector is not None:
            if project_phenos_from_gene_sets or state.exp_gene_factors is None:
                pheno_matrix_to_project = state.X_phewas_beta_uncorrected.T
                if not run_transpose:
                    pheno_matrix_to_project = pheno_matrix_to_project.T
                basis = state.exp_gene_set_factors
                feature_by_pheno = _prepare_pheno_capture_input_matrix(pheno_matrix_to_project, pheno_capture_input)
                basis, feature_by_pheno = _align_projection_inputs_to_mask(
                    basis,
                    feature_by_pheno,
                    state.gene_set_factor_gene_set_mask,
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
                    state.gene_factor_gene_mask,
                )
                full_pheno_factor_values = _project_pheno_capture_matrix(
                    state,
                    basis,
                    feature_by_pheno,
                    basis_name="genes",
                )

            if keep_original_loadings:
                full_pheno_factor_values[state.pheno_factor_pheno_mask,:] = state.exp_pheno_factors
            state.pheno_capture_input = pheno_capture_input
        elif state.exp_pheno_factors is not None:
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

        if factor_gene_set_x_pheno and pheno_matrix_to_project is not None:
            full_gene_set_factor_values = state._project_H_with_fixed_W(state.exp_pheno_factors, pheno_matrix_to_project[:,state.pheno_factor_pheno_mask].T if run_transpose else pheno_matrix_to_project[state.pheno_factor_pheno_mask,:].T, state.pheno_prob_factor_vector[state.pheno_factor_pheno_mask,:], state.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)
        else:
            full_gene_set_factor_values = state._project_H_with_fixed_W(state.exp_gene_factors, gene_matrix_to_project[:,state.gene_factor_gene_mask].T if run_transpose else gene_matrix_to_project[state.gene_factor_gene_mask,:].T, state.gene_prob_factor_vector[state.gene_factor_gene_mask,:], state.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)

        if keep_original_loadings:
            full_gene_set_factor_values[state.gene_set_factor_gene_set_mask,:] = state.exp_gene_set_factors

        state.exp_gene_factors = full_gene_factor_values
        state.exp_pheno_factors = full_pheno_factor_values
        state.exp_gene_set_factors = full_gene_set_factor_values

        matrix_to_mult = state.exp_pheno_factors if factor_gene_set_x_pheno else state.exp_gene_factors
        vector_to_mult = state.pheno_prob_factor_vector if factor_gene_set_x_pheno else state.gene_prob_factor_vector
        state.factor_anchor_relevance = state._nnls_project_matrix(matrix_to_mult, vector_to_mult.T, max_value=1).T
        state.factor_relevance = _compute_any_anchor_relevance(state.factor_anchor_relevance)

    _finalize_factor_outputs(
        state,
        factor_gene_set_x_pheno=factor_gene_set_x_pheno,
        lmm_auth_key=lmm_auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        label_gene_sets_only=label_gene_sets_only,
        label_include_phenos=label_include_phenos,
        label_individually=label_individually,
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
            factor_gene_set_x_pheno=bool(state.exp_pheno_factors is not None and state.gene_factor_gene_mask is None),
        )
    summary["run_index"] = int(run_index)
    summary["seed"] = None if seed is None else int(seed)
    return summary


def _collect_consensus_matches(run_states, run_summaries, reference_index, min_factor_cosine):
    reference_factors = run_states[reference_index].exp_gene_set_factors
    reference_norm = _normalize_factor_columns(reference_factors)
    factor_matches = {idx: [(reference_index, idx, 1.0)] for idx in range(reference_factors.shape[1])}

    for run_index, run_state in enumerate(run_states):
        if run_index == reference_index:
            continue
        run_factors = run_state.exp_gene_set_factors
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
    eligible_indices = [idx for idx, state in enumerate(run_states) if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0 and state.num_factors() > 0]
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

    for reference_factor_index in range(run_states[reference_index].exp_gene_set_factors.shape[1]):
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
        lmm_auth_key=factor_kwargs["lmm_auth_key"],
        lmm_model=factor_kwargs["lmm_model"],
        lmm_provider=factor_kwargs["lmm_provider"],
        label_gene_sets_only=factor_kwargs["label_gene_sets_only"],
        label_include_phenos=factor_kwargs["label_include_phenos"],
        label_individually=factor_kwargs["label_individually"],
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


def run_factor(state, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, seed=None, factor_runs=1, consensus_nmf=False, consensus_min_factor_cosine=0.7, consensus_min_run_support=0.5, consensus_aggregation="median", consensus_stats_out=None, learn_phi=False, learn_phi_max_redundancy=0.5, learn_phi_max_redundancy_q90=0.35, learn_phi_runs_per_step=1, learn_phi_min_run_support=0.6, learn_phi_min_stability=0.85, learn_phi_max_fit_loss_frac=0.05, learn_phi_k_band_frac=0.9, learn_phi_max_steps=5, learn_phi_expand_factor=2.0, learn_phi_weight_floor=None, learn_phi_mass_floor_frac=_DEFAULT_LEARN_PHI_MASS_FLOOR_FRAC, learn_phi_min_error_gain_per_factor=_LEARN_PHI_MIN_ERROR_GAIN_PER_FACTOR, learn_phi_only=False, learn_phi_report_out=None, factor_phi_metrics_out=None, max_num_gene_sets=None, gene_set_budget_mode="pruned", learn_phi_gene_set_budget_mode=None, factor_backend="full", learn_phi_backend="sentinel_pruned", online_block_size=None, online_passes=20, online_min_passes_before_stopping=5, online_shuffle_blocks=True, online_warm_start=True, online_max_blocks=None, online_report_out=None, approx_projection_block_size=None, approx_projection_n_iter=100, sketch_size=None, sketch_embedding_dim=16, sketch_selection_method="projected_cluster_medoids", sketch_random_seed=None, sketch_refinement_passes=0, factors_out=None, factor_metrics_out=None, gene_set_clusters_out=None, gene_clusters_out=None, learn_phi_prune_genes_num=1000, learn_phi_prune_gene_sets_num=1000, learn_phi_max_num_iterations=None, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, lmm_model=None, lmm_provider="openai", label_gene_sets_only=False, label_include_phenos=False, label_individually=False, keep_original_loadings=False, project_phenos_from_gene_sets=False, pheno_capture_input="weighted_thresholded", learn_phi_scout_repeats=3, learn_phi_confirm_topk=3, learn_phi_confirm_online_passes=5, learn_phi_confirm_block_size=None, learn_phi_scout_selection_method=None, *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, labeling_module):
    bail = bail_fn
    log = log_fn
    INFO = info_level

    if factor_runs < 1:
        bail("--factor-runs must be at least 1")
    if max_num_gene_sets is not None and int(max_num_gene_sets) < 1:
        bail("--max-num-gene-sets must be at least 1")
    if str(gene_set_budget_mode) not in {"pruned", "online_shared_basis", "structure_preserving_sketch"}:
        bail("--gene-set-budget-mode must be one of: pruned, online_shared_basis, structure_preserving_sketch")
    if learn_phi_gene_set_budget_mode is not None and str(learn_phi_gene_set_budget_mode) not in {"pruned", "online_shared_basis", "structure_preserving_sketch"}:
        bail("--learn-phi-gene-set-budget-mode must be one of: pruned, online_shared_basis, structure_preserving_sketch")
    if factor_backend not in {"full", "blockwise_global_w", "online_shared_basis", "structure_preserving_sketch"}:
        bail("--factor-backend must be one of: full, blockwise_global_w, online_shared_basis, structure_preserving_sketch")
    if learn_phi_backend not in {"sentinel_pruned", "blockwise_global_w", "online_shared_basis", "structure_preserving_sketch"}:
        bail("--learn-phi-backend must be one of: sentinel_pruned, blockwise_global_w, online_shared_basis, structure_preserving_sketch")
    if online_block_size is not None and int(online_block_size) < 1:
        bail("--online-block-size must be at least 1")
    if int(online_passes) < 1:
        bail("--online-passes must be at least 1")
    if int(online_min_passes_before_stopping) < 1:
        bail("--online-min-passes-before-stopping must be at least 1")
    if int(online_min_passes_before_stopping) > int(online_passes):
        bail("--online-min-passes-before-stopping must be <= --online-passes")
    if online_max_blocks is not None and int(online_max_blocks) < 1:
        bail("--online-max-blocks must be at least 1")
    if approx_projection_block_size is not None and int(approx_projection_block_size) < 1:
        bail("--approx-projection-block-size must be at least 1")
    if int(approx_projection_n_iter) < 1:
        bail("--approx-projection-n-iter must be at least 1")
    if sketch_size is not None and int(sketch_size) < 1:
        bail("--sketch-size must be at least 1")
    if int(sketch_embedding_dim) < 1:
        bail("--sketch-embedding-dim must be at least 1")
    if str(_resolve_sketch_selection_method(sketch_selection_method)) not in {"projected_cluster_medoids"}:
        bail("--sketch-selection-method must be one of: projected_cluster_medoids, projected_kmedoids")
    if int(sketch_refinement_passes) < 0:
        bail("--sketch-refinement-passes must be >= 0")
    if int(learn_phi_scout_repeats) < 1:
        bail("--learn-phi-scout-repeats must be at least 1")
    if int(learn_phi_confirm_topk) < 1:
        bail("--learn-phi-confirm-topk must be at least 1")
    if int(learn_phi_confirm_online_passes) < 1:
        bail("--learn-phi-confirm-online-passes must be at least 1")
    if learn_phi_confirm_block_size is not None and int(learn_phi_confirm_block_size) < 1:
        bail("--learn-phi-confirm-block-size must be at least 1")
    if learn_phi_scout_selection_method is not None and str(_resolve_sketch_selection_method(learn_phi_scout_selection_method)) not in {"projected_cluster_medoids"}:
        bail("--learn-phi-scout-selection-method must be one of: projected_cluster_medoids, projected_kmedoids")
    effective_factor_backend = _resolve_budget_mode_alias(
        factor_backend,
        default_mode=str(gene_set_budget_mode),
    )
    effective_learn_phi_backend = _resolve_budget_mode_alias(
        learn_phi_backend,
        default_mode=str(learn_phi_gene_set_budget_mode if learn_phi_gene_set_budget_mode is not None else gene_set_budget_mode),
    )
    if consensus_aggregation not in {"median", "mean"}:
        bail("--consensus-aggregation must be one of: median, mean")
    if not (0 < consensus_min_factor_cosine <= 1):
        bail("--consensus-min-factor-cosine must be in (0, 1]")
    if not (0 < consensus_min_run_support <= 1):
        bail("--consensus-min-run-support must be in (0, 1]")
    if consensus_nmf and factor_runs < 2:
        bail("--consensus-nmf requires --factor-runs >= 2")
    if learn_phi:
        if phi <= 0:
            bail("--learn-phi requires --phi > 0")
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
        if learn_phi_max_fit_loss_frac < 0:
            bail("--learn-phi-max-fit-loss-frac must be >= 0")
        if not (0 < learn_phi_k_band_frac <= 1):
            bail("--learn-phi-k-band-frac must be in (0, 1]")
        if learn_phi_max_steps < 1:
            bail("--learn-phi-max-steps must be at least 1")
        if learn_phi_expand_factor <= 1:
            bail("--learn-phi-expand-factor must be > 1")
        if learn_phi_weight_floor is not None and learn_phi_weight_floor < 0:
            bail("--learn-phi-weight-floor must be >= 0 when set")
        if not (0 < learn_phi_mass_floor_frac <= 1):
            bail("--learn-phi-mass-floor-frac must be in (0, 1]")
        if learn_phi_min_error_gain_per_factor < 0:
            bail("--learn-phi-min-error-gain-per-factor must be >= 0")
        if learn_phi_prune_genes_num is not None and learn_phi_prune_genes_num < 1:
            bail("--learn-phi-prune-genes-num must be at least 1")
        if learn_phi_prune_gene_sets_num is not None and learn_phi_prune_gene_sets_num < 1:
            bail("--learn-phi-prune-gene-sets-num must be at least 1")
        if learn_phi_max_num_iterations is not None and learn_phi_max_num_iterations < 1:
            bail("--learn-phi-max-num-iterations must be at least 1")

    factor_kwargs = {
        "max_num_factors": max_num_factors,
        "phi": phi,
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
        "keep_original_loadings": keep_original_loadings,
        "project_phenos_from_gene_sets": project_phenos_from_gene_sets,
        "pheno_capture_input": pheno_capture_input,
        "max_num_gene_sets": max_num_gene_sets,
        "gene_set_budget_mode": effective_factor_backend,
        "factor_backend": effective_factor_backend,
        "online_block_size": online_block_size,
        "online_passes": online_passes,
        "online_min_passes_before_stopping": online_min_passes_before_stopping,
        "online_shuffle_blocks": online_shuffle_blocks,
        "online_warm_start": online_warm_start,
        "online_max_blocks": online_max_blocks,
        "online_report_out": online_report_out,
        "approx_projection_block_size": approx_projection_block_size,
        "approx_projection_n_iter": approx_projection_n_iter,
        "sketch_size": sketch_size,
        "sketch_embedding_dim": sketch_embedding_dim,
        "sketch_selection_method": _resolve_sketch_selection_method(sketch_selection_method),
        "sketch_random_seed": sketch_random_seed,
        "sketch_refinement_passes": sketch_refinement_passes,
        "factors_out": factors_out,
        "factor_metrics_out": factor_metrics_out,
        "gene_set_clusters_out": gene_set_clusters_out,
        "gene_clusters_out": gene_clusters_out,
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
            learn_phi=learn_phi,
            learn_phi_max_redundancy=learn_phi_max_redundancy,
            learn_phi_max_redundancy_q90=learn_phi_max_redundancy_q90,
            learn_phi_runs_per_step=learn_phi_runs_per_step,
            learn_phi_min_run_support=learn_phi_min_run_support,
            learn_phi_min_stability=learn_phi_min_stability,
            learn_phi_max_fit_loss_frac=learn_phi_max_fit_loss_frac,
            learn_phi_k_band_frac=learn_phi_k_band_frac,
            learn_phi_max_steps=learn_phi_max_steps,
            learn_phi_expand_factor=learn_phi_expand_factor,
            learn_phi_weight_floor=learn_phi_weight_floor,
            learn_phi_mass_floor_frac=learn_phi_mass_floor_frac,
            learn_phi_min_error_gain_per_factor=learn_phi_min_error_gain_per_factor,
            learn_phi_only=learn_phi_only,
            learn_phi_report_out=learn_phi_report_out,
            factor_phi_metrics_out=factor_phi_metrics_out,
            max_num_gene_sets=max_num_gene_sets,
            gene_set_budget_mode=effective_factor_backend,
            learn_phi_gene_set_budget_mode=effective_learn_phi_backend,
            factor_backend=effective_factor_backend,
            learn_phi_backend=effective_learn_phi_backend,
            online_block_size=online_block_size,
            online_passes=online_passes,
            online_min_passes_before_stopping=online_min_passes_before_stopping,
            online_shuffle_blocks=online_shuffle_blocks,
            online_warm_start=online_warm_start,
            online_max_blocks=online_max_blocks,
            online_report_out=online_report_out,
            approx_projection_block_size=approx_projection_block_size,
            approx_projection_n_iter=approx_projection_n_iter,
            sketch_size=sketch_size,
            sketch_embedding_dim=sketch_embedding_dim,
            sketch_selection_method=_resolve_sketch_selection_method(sketch_selection_method),
            sketch_random_seed=sketch_random_seed,
            sketch_refinement_passes=sketch_refinement_passes,
            learn_phi_scout_repeats=learn_phi_scout_repeats,
            learn_phi_confirm_topk=learn_phi_confirm_topk,
            learn_phi_confirm_online_passes=learn_phi_confirm_online_passes,
            learn_phi_confirm_block_size=learn_phi_confirm_block_size,
            learn_phi_scout_selection_method=learn_phi_scout_selection_method,
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
            keep_original_loadings=keep_original_loadings,
            project_phenos_from_gene_sets=project_phenos_from_gene_sets,
            pheno_capture_input=pheno_capture_input,
        ),
        overwrite=True,
    )
    log(
        "Factor config: phi=%.6g alpha0=%.6g beta0=%.6g max_num_factors=%d factor_runs=%d consensus_nmf=%s learn_phi=%s gene_set_budget_mode=%s learn_phi_budget_mode=%s max_num_iterations=%d rel_tol=%.3g"
        % (
            float(phi),
            float(alpha0),
            float(beta0),
            int(max_num_factors),
            int(factor_runs),
            bool(consensus_nmf),
            bool(learn_phi),
            str(effective_factor_backend),
            str(effective_learn_phi_backend),
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
            max_fit_loss_frac=learn_phi_max_fit_loss_frac,
            k_band_frac=learn_phi_k_band_frac,
            max_steps=learn_phi_max_steps,
            expand_factor=learn_phi_expand_factor,
            weight_floor=weight_floor,
            mass_floor_frac=float(learn_phi_mass_floor_frac),
            min_error_gain_per_factor=float(learn_phi_min_error_gain_per_factor),
            learn_phi_backend=effective_learn_phi_backend,
            blockwise_warm_start=online_warm_start,
            report_out=learn_phi_report_out,
            factor_phi_metrics_out=factor_phi_metrics_out,
            prune_genes_num=learn_phi_prune_genes_num,
            prune_gene_sets_num=learn_phi_prune_gene_sets_num,
            max_num_iterations=learn_phi_max_num_iterations,
            factor_kwargs=factor_kwargs,
            log_fn=log,
            info_level=INFO,
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
