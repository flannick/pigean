from __future__ import annotations

import copy

import numpy as np
import scipy.linalg
import scipy.sparse as sparse

from pegs_shared.cli import _default_bail
from pegs_shared.io_common import construct_map_to_ind
from pegs_shared.ydata import apply_hyperparameter_data_to_runtime, ensure_hyperparameter_state


def compute_banded_y_corr_cholesky(Y_corr_m, diag_add=0.05):
    Y_corr_m_copy = copy.copy(Y_corr_m)
    while True:
        try:
            Y_corr_m_copy[0, :] += diag_add
            Y_corr_m_copy /= (1 + diag_add)
            return scipy.linalg.cholesky_banded(Y_corr_m_copy, lower=True)
        except np.linalg.LinAlgError:
            pass


def whiten_matrix_with_banded_cholesky(matrix, corr_cholesky, *, whiten=True, full_whiten=False):
    if full_whiten:
        matrix = scipy.linalg.cho_solve_banded((corr_cholesky, True), matrix, overwrite_b=True)
    elif whiten:
        matrix = scipy.linalg.solve_banded((corr_cholesky.shape[0] - 1, 0), corr_cholesky, matrix, overwrite_ab=True)
    return matrix


def calc_shift_scale_for_dense_block(X_b):
    mean_shifts = []
    scale_factors = []
    for i in range(X_b.shape[1]):
        X_i = X_b[:, i]
        mean_shifts.append(np.mean(X_i))
        scale_factor = np.std(X_i)
        if scale_factor == 0:
            scale_factor = 1
        scale_factors.append(scale_factor)
    return (np.array(mean_shifts), np.array(scale_factors))


def calc_X_shift_scale(
    X,
    *,
    y_corr_cholesky=None,
    get_X_blocks_internal_fn=None,
    calc_shift_scale_fn=calc_shift_scale_for_dense_block,
):
    if y_corr_cholesky is None:
        if sparse.issparse(X):
            mean_shifts = X.sum(axis=0).A1 / X.shape[0]
            scale_factors = np.sqrt(X.power(2).sum(axis=0).A1 / X.shape[0] - np.square(mean_shifts))
        else:
            mean_shifts = np.mean(X, axis=0)
            scale_factors = np.std(X, axis=0)
        return (mean_shifts, scale_factors)

    if get_X_blocks_internal_fn is None:
        _default_bail("Expected get_X_blocks_internal_fn when y_corr_cholesky is provided")

    scale_factors = np.array([])
    mean_shifts = np.array([])
    for X_b, _begin, _end, _batch in get_X_blocks_internal_fn(X, y_corr_cholesky):
        (cur_mean_shifts, cur_scale_factors) = calc_shift_scale_fn(X_b)
        mean_shifts = np.append(mean_shifts, cur_mean_shifts)
        scale_factors = np.append(scale_factors, cur_scale_factors)
    return (mean_shifts, scale_factors)


def calculate_V_internal(
    X_orig,
    y_corr_cholesky,
    mean_shifts,
    scale_factors,
    *,
    y_corr_sparse=None,
    get_num_X_blocks_fn=None,
    get_X_blocks_internal_fn=None,
    compute_V_fn=None,
):
    if y_corr_cholesky is not None:
        if get_num_X_blocks_fn is None or get_X_blocks_internal_fn is None or compute_V_fn is None:
            _default_bail(
                "calculate_V_internal requires get_num_X_blocks_fn, get_X_blocks_internal_fn, and compute_V_fn when y_corr_cholesky is set"
            )

        if get_num_X_blocks_fn(X_orig) == 1:
            whiten1 = True
            full_whiten1 = False
            whiten2 = True
            full_whiten2 = False
        else:
            whiten1 = False
            full_whiten1 = True
            whiten2 = False
            full_whiten2 = False

        V = None
        if X_orig is not None:
            for X_b1, _begin1, _end1, _batch1 in get_X_blocks_internal_fn(
                X_orig,
                y_corr_cholesky,
                whiten=whiten1,
                full_whiten=full_whiten1,
                mean_shifts=mean_shifts,
                scale_factors=scale_factors,
            ):
                cur_V = None
                if y_corr_sparse is not None:
                    X_b1 = y_corr_sparse.dot(X_b1)

                for X_b2, _begin2, _end2, _batch2 in get_X_blocks_internal_fn(
                    X_orig,
                    y_corr_cholesky,
                    whiten=whiten2,
                    full_whiten=full_whiten2,
                    mean_shifts=mean_shifts,
                    scale_factors=scale_factors,
                ):
                    V_block = compute_V_fn(X_b1, 0, 1, X_orig2=X_b2, mean_shifts2=0, scale_factors2=1)
                    if cur_V is None:
                        cur_V = V_block
                    else:
                        cur_V = np.hstack((cur_V, V_block))
                if V is None:
                    V = cur_V
                else:
                    V = np.vstack((V, cur_V))
    else:
        if compute_V_fn is None:
            _default_bail("calculate_V_internal requires compute_V_fn when y_corr_cholesky is not set")
        V = compute_V_fn(X_orig, mean_shifts, scale_factors)

    return V


def set_runtime_x_from_inputs(
    runtime,
    X_orig,
    genes,
    gene_sets,
    *,
    skip_scale_factors=False,
    skip_N=True,
    reread_gene_phewas_bfs_fn=None,
    construct_map_to_ind_fn=None,
    get_col_sums_fn=None,
    set_scale_factors_fn=None,
    log_fn=None,
    trace_level=0,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    if construct_map_to_ind_fn is None:
        construct_map_to_ind_fn = construct_map_to_ind

    log_fn("Setting X", trace_level)

    if X_orig is not None:
        if not len(genes) == X_orig.shape[0]:
            bail_fn("Dimension mismatch when setting X: %d genes but %d rows in X" % (len(genes), X_orig.shape[0]))
        if not len(gene_sets) == X_orig.shape[1]:
            bail_fn("Dimension mismatch when setting X: %d gene sets but %d columns in X" % (len(gene_sets), X_orig.shape[1]))

    if (
        runtime.X_orig is not None
        and X_orig is runtime.X_orig
        and genes is runtime.genes
        and gene_sets is runtime.gene_sets
        and (
            (runtime.y_corr_cholesky is None and not runtime.scale_is_for_whitened)
            or (runtime.y_corr_cholesky is not None and runtime.scale_is_for_whitened)
        )
    ):
        return False

    runtime.last_X_block = None
    runtime.genes = genes

    if runtime.gene_pheno_Y is not None or runtime.gene_pheno_combined_prior_Ys is not None or runtime.gene_pheno_priors is not None:
        if len(runtime.genes) != runtime.gene_pheno_Y.shape[0] and reread_gene_phewas_bfs_fn is not None:
            reread_gene_phewas_bfs_fn(runtime)

    if runtime.genes is not None:
        runtime.gene_to_ind = construct_map_to_ind_fn(runtime.genes)
    else:
        runtime.gene_to_ind = None

    runtime.gene_sets = gene_sets
    if runtime.gene_sets is not None:
        runtime.gene_set_to_ind = construct_map_to_ind_fn(runtime.gene_sets)
    else:
        runtime.gene_set_to_ind = None

    runtime.X_orig = X_orig
    if runtime.X_orig is not None:
        runtime.X_orig.eliminate_zeros()

    if runtime.X_orig is None:
        runtime.X_orig_missing_genes = None
        runtime.X_orig_missing_genes_missing_gene_sets = None
        runtime.X_orig_missing_gene_sets = None
        runtime.last_X_block = None
        return True

    if not skip_N and get_col_sums_fn is not None:
        runtime.gene_N = get_col_sums_fn(runtime.X_orig, axis=1)

    if not skip_scale_factors and set_scale_factors_fn is not None:
        set_scale_factors_fn()

    return True


def get_num_X_blocks(X_orig, batch_size):
    return int(np.ceil(X_orig.shape[1] / batch_size))


def iterate_X_blocks_internal(
    X_orig,
    y_corr_cholesky,
    *,
    batch_size,
    log_fn=None,
    trace_level=0,
    is_missing_x=False,
    consider_cache=False,
    cache_state=None,
    whiten_fn=whiten_matrix_with_banded_cholesky,
    whiten=True,
    full_whiten=False,
    start_batch=0,
    mean_shifts=None,
    scale_factors=None,
):
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None

    if y_corr_cholesky is None:
        whiten = False
        full_whiten = False

    num_batches = get_num_X_blocks(X_orig, batch_size)
    if cache_state is None:
        cache_state = {}

    for batch in range(start_batch, num_batches):
        log_fn(
            "Getting X%s block batch %s (%s)"
            % (
                "_missing" if is_missing_x else "",
                batch,
                "fully whitened" if full_whiten else ("whitened" if whiten else "original"),
            ),
            trace_level,
        )
        begin = batch * batch_size
        end = (batch + 1) * batch_size
        if end > X_orig.shape[1]:
            end = X_orig.shape[1]

        last_X_block = cache_state.get("last_X_block")
        if last_X_block is not None and consider_cache and last_X_block[1:] == (whiten, full_whiten, begin, end, batch):
            log_fn("Using cache!", trace_level)
            yield (last_X_block[0], begin, end, batch)
            continue

        X_b = X_orig[:, begin:end].toarray()
        if mean_shifts is not None:
            X_b = X_b - mean_shifts[begin:end]
        if scale_factors is not None:
            X_b = X_b / scale_factors[begin:end]

        if whiten or full_whiten:
            X_b = whiten_fn(X_b, y_corr_cholesky, whiten=whiten, full_whiten=full_whiten)

        if consider_cache:
            cache_state["last_X_block"] = (X_b, whiten, full_whiten, begin, end, batch)
        else:
            cache_state["last_X_block"] = None

        yield (X_b, begin, end, batch)


def set_runtime_p(runtime, p):
    hyper_state = ensure_hyperparameter_state(runtime)
    hyper_state.set_p(p)
    apply_hyperparameter_data_to_runtime(runtime, hyper_state)
    return hyper_state


def set_runtime_sigma(
    runtime,
    sigma2,
    sigma_power,
    sigma2_osc=None,
    sigma2_se=None,
    sigma2_p=None,
    sigma2_scale_factors=None,
    convert_sigma_to_internal_units=False,
):
    hyper_state = ensure_hyperparameter_state(runtime)
    hyper_state.set_sigma(
        runtime,
        sigma2,
        sigma_power,
        sigma2_osc=sigma2_osc,
        sigma2_se=sigma2_se,
        sigma2_p=sigma2_p,
        sigma2_scale_factors=sigma2_scale_factors,
        convert_sigma_to_internal_units=convert_sigma_to_internal_units,
    )
    apply_hyperparameter_data_to_runtime(runtime, hyper_state)
    return hyper_state
