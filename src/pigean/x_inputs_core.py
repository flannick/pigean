from __future__ import annotations

import copy
import random

import numpy as np
import scipy.sparse as sparse

from pegs_shared.io_common import construct_map_to_ind
import pegs_utils as pegs_utils_mod
from . import runtime as pigean_runtime


def normalize_dense_gene_rows(mat_info, genes, gene_label_map):
    if gene_label_map is not None:
        genes = list(map(lambda x: gene_label_map[x] if x in gene_label_map else x, genes))

    if len(set(genes)) != len(genes):
        seen_genes = set()
        unique_mask = np.full(len(genes), True)
        for i in range(len(genes)):
            if genes[i] in seen_genes:
                unique_mask[i] = False
            else:
                seen_genes.add(genes[i])
        mat_info = mat_info[unique_mask, :]
        genes = [genes[i] for i in range(len(genes)) if unique_mask[i]]

    return (mat_info, genes)


def build_sparse_x_from_dense_input(
    runtime_state,
    mat_info,
    genes,
    gene_sets,
    x_sparsify,
    min_gene_set_size,
    add_ext,
    add_top,
    add_bottom,
    fname=None,
    *,
    log_fn,
    debug_level,
    warn_fn,
    bail_fn,
    ext_tag,
    bot_tag,
    top_tag,
):
    if len(x_sparsify) > 0:
        sparsity_threshold = 1 - np.max(x_sparsify).astype(float) / mat_info.shape[0]
    else:
        sparsity_threshold = 0.95

    orig_dense_gene_sets = gene_sets
    cur_X = None

    convert_to_sparse = np.sum(mat_info == 0, axis=0) / mat_info.shape[0] > sparsity_threshold

    abs_mat_info = np.abs(mat_info)
    max_weights = abs_mat_info.max(axis=0)
    all_non_zero_same = np.sum(abs_mat_info * (abs_mat_info != max_weights), axis=0) == 0

    convert_to_sparse = np.logical_or(convert_to_sparse, all_non_zero_same)
    if np.any(convert_to_sparse):
        log_fn(
            "Detected sparse matrix for %d of %d columns" % (np.sum(convert_to_sparse), len(convert_to_sparse)),
            debug_level,
        )
        cur_X = sparse.csc_matrix(mat_info[:, convert_to_sparse])
        gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if convert_to_sparse[i]]
        orig_dense_gene_sets = [orig_dense_gene_sets[i] for i in range(len(orig_dense_gene_sets)) if not convert_to_sparse[i]]

        mat_info = mat_info[:, ~convert_to_sparse]
        enough_genes = runtime_state.get_col_sums(cur_X, num_nonzero=True) >= min_gene_set_size
        if np.any(~enough_genes):
            log_fn("Excluded %d gene sets due to too small size" % np.sum(~enough_genes), debug_level)
            cur_X = cur_X[:, enough_genes]
            gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if enough_genes[i]]

    if mat_info.shape[1] > 0:
        mat_sd = np.std(mat_info, axis=0)
        if np.any(mat_sd == 0):
            mat_info = mat_info[:, mat_sd != 0]

        mat_info = (mat_info - np.mean(mat_info, axis=0)) / np.std(mat_info, axis=0)

        subset_mask = np.full(len(genes), True)
        x_for_stats = mat_info
        if runtime_state.Y is not None and runtime_state.genes is not None:
            subset_mask[[i for i in range(len(genes)) if genes[i] not in runtime_state.gene_to_ind]] = False
            x_for_stats = mat_info[subset_mask, :]

        if x_for_stats.shape[0] == 0:
            warn_fn(
                "No genes in --Xd-in %swere seen before so skipping; example genes: %s"
                % ("%s " % fname if fname is not None else "", ",".join(genes[:4]))
            )
            return (None, None, True)

        top_numbers = list(reversed(sorted(x_sparsify)))
        top_fractions = np.array(top_numbers, dtype=float) / x_for_stats.shape[0]

        top_fractions[top_fractions > 1] = 1
        top_fractions[top_fractions < 0] = 0

        if len(top_fractions) == 0:
            bail_fn("No --X-sparsify set so doing nothing")
            return (None, None, True)

        upper_quantiles = np.quantile(x_for_stats, 1 - top_fractions, axis=0)
        lower_quantiles = np.quantile(x_for_stats, top_fractions, axis=0)

        upper = copy.copy(mat_info)
        lower = copy.copy(mat_info)

        assert np.all(upper_quantiles[0, :] == np.min(upper_quantiles, axis=0))
        assert np.all(lower_quantiles[0, :] == np.max(lower_quantiles, axis=0))

        for i in range(len(top_numbers)):
            upper_threshold_mask = upper < upper_quantiles[i, :]
            if np.sum(upper_threshold_mask) == 0:
                upper_threshold_mask = upper <= upper_quantiles[i, :]

            lower_threshold_mask = lower > lower_quantiles[i, :]
            if np.sum(lower_threshold_mask) == 0:
                lower_threshold_mask = lower >= lower_quantiles[i, :]

            mat_info[np.logical_and(upper_threshold_mask, lower_threshold_mask)] = 0
            upper[upper_threshold_mask] = 0
            lower[lower_threshold_mask] = 0

            if add_ext:
                temp_X = sparse.csc_matrix(mat_info)
                top_gene_sets = ["%s_%s%d" % (x, ext_tag, top_numbers[i]) for x in orig_dense_gene_sets]
                if cur_X is None:
                    cur_X = temp_X
                    gene_sets = top_gene_sets
                else:
                    cur_X = sparse.hstack((cur_X, temp_X))
                    gene_sets = gene_sets + top_gene_sets

            if add_bottom:
                temp_X = sparse.csc_matrix(lower)
                top_gene_sets = ["%s_%s%d" % (x, bot_tag, top_numbers[i]) for x in orig_dense_gene_sets]
                if cur_X is None:
                    cur_X = temp_X
                    gene_sets = top_gene_sets
                else:
                    cur_X = sparse.hstack((cur_X, temp_X))
                    gene_sets = gene_sets + top_gene_sets

            if add_top or (not add_ext and not add_bottom):
                temp_X = sparse.csc_matrix(upper)
                top_gene_sets = ["%s_%s%d" % (x, top_tag, top_numbers[i]) for x in orig_dense_gene_sets]
                if cur_X is None:
                    cur_X = temp_X
                    gene_sets = top_gene_sets
                else:
                    gene_sets = gene_sets + top_gene_sets
                    cur_X = sparse.hstack((cur_X, temp_X))

            if cur_X is None:
                return (None, None, True)

            all_negative_mask = ((cur_X < 0).sum(axis=0) == cur_X.astype(bool).sum(axis=0)).A1
            cur_X[:, all_negative_mask] = -cur_X[:, all_negative_mask]
            cur_X.eliminate_zeros()

        if cur_X is None or cur_X.shape[1] == 0:
            return (None, None, True)

    return (cur_X, gene_sets, False)


def standardize_qc_metrics_after_x_read_for_runtime(runtime_state):
    pegs_utils_mod.standardize_qc_metrics_after_x_read(runtime_state)


def maybe_correct_gene_set_betas_after_x_read_for_runtime(
    runtime_state,
    filter_gene_set_p,
    correct_betas_mean,
    correct_betas_var,
    filter_using_phewas,
    *,
    log_fn,
):
    pegs_utils_mod.maybe_correct_gene_set_betas_after_x_read(
        runtime_state,
        filter_gene_set_p=filter_gene_set_p,
        correct_betas_mean=correct_betas_mean,
        correct_betas_var=correct_betas_var,
        filter_using_phewas=filter_using_phewas,
        log_fn=log_fn,
    )


def maybe_limit_initial_gene_sets_by_p_for_runtime(runtime_state, max_num_gene_sets_initial, *, log_fn):
    pegs_utils_mod.maybe_limit_initial_gene_sets_by_p(
        runtime_state,
        max_num_gene_sets_initial=max_num_gene_sets_initial,
        log_fn=log_fn,
    )


def maybe_prune_gene_sets_after_x_read_for_runtime(
    runtime_state,
    skip_betas,
    prune_gene_sets,
    prune_deterministically,
    weighted_prune_gene_sets,
):
    pegs_utils_mod.maybe_prune_gene_sets_after_x_read(
        runtime_state,
        skip_betas=skip_betas,
        prune_gene_sets=prune_gene_sets,
        prune_deterministically=prune_deterministically,
        weighted_prune_gene_sets=weighted_prune_gene_sets,
    )


def initialize_hyper_defaults_after_x_read_for_runtime(
    runtime_state,
    initial_p,
    update_hyper_p,
    sigma_power,
    initial_sigma2_cond,
    update_hyper_sigma,
    initial_sigma2,
    sigma_soft_threshold_95,
    sigma_soft_threshold_5,
    *,
    warn_fn,
    log_fn,
):
    return pegs_utils_mod.initialize_hyper_defaults_after_x_read(
        runtime_state,
        initial_p=initial_p,
        update_hyper_p=update_hyper_p,
        sigma_power=sigma_power,
        initial_sigma2_cond=initial_sigma2_cond,
        update_hyper_sigma=update_hyper_sigma,
        initial_sigma2=initial_sigma2,
        sigma_soft_threshold_95=sigma_soft_threshold_95,
        sigma_soft_threshold_5=sigma_soft_threshold_5,
        warn_fn=warn_fn,
        log_fn=log_fn,
    )


def maybe_adjust_overaggressive_p_filter_after_x_read_for_runtime(
    runtime_state,
    filter_gene_set_p,
    increase_filter_gene_set_p,
    filter_using_phewas,
    *,
    log_fn,
):
    pegs_utils_mod.maybe_adjust_overaggressive_p_filter_after_x_read(
        runtime_state,
        filter_gene_set_p=filter_gene_set_p,
        increase_filter_gene_set_p=increase_filter_gene_set_p,
        filter_using_phewas=filter_using_phewas,
        log_fn=log_fn,
    )


def apply_post_read_gene_set_size_and_qc_filters_for_runtime(
    runtime_state,
    min_gene_set_size,
    max_gene_set_size,
    filter_gene_set_metric_z,
    *,
    log_fn,
):
    pegs_utils_mod.apply_post_read_gene_set_size_and_qc_filters(
        runtime_state,
        min_gene_set_size=min_gene_set_size,
        max_gene_set_size=max_gene_set_size,
        filter_gene_set_metric_z=filter_gene_set_metric_z,
        log_fn=log_fn,
    )


def learn_hyper_for_gene_set_batch(
    runtime_state,
    gene_sets_for_hyper_mask,
    num_missing_gene_sets,
    update_hyper_p,
    update_hyper_sigma,
    first_for_sigma_cond,
    fixed_sigma_cond,
    ordered_batch_ind,
    max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    sigma_num_devs_to_top,
    p_noninf_inflate,
    sparse_solution,
    sparse_frac_betas,
    betas_trace_out,
):
    with pigean_runtime.temporary_state_fields(
        runtime_state,
        overrides={"ps": None, "sigma2s": None},
        restore_fields=("ps", "sigma2s"),
    ):
        if np.sum(gene_sets_for_hyper_mask) > runtime_state.batch_size:
            V = None
        else:
            V = runtime_state._calculate_V_internal(
                runtime_state.X_orig[:, gene_sets_for_hyper_mask],
                runtime_state.y_corr_cholesky,
                runtime_state.mean_shifts[gene_sets_for_hyper_mask],
                runtime_state.scale_factors[gene_sets_for_hyper_mask],
            )

        num_p_pseudo = min(1, np.sum(gene_sets_for_hyper_mask) / 1000)

        cur_update_hyper_p = update_hyper_p
        cur_update_hyper_sigma = update_hyper_sigma
        adjust_hyper_sigma_p = False
        if (first_for_sigma_cond and ordered_batch_ind > 0) or fixed_sigma_cond:
            adjust_hyper_sigma_p = True
            if cur_update_hyper_p:
                cur_update_hyper_sigma = False

        runtime_state._calculate_non_inf_betas(
            initial_p=None,
            beta_tildes=runtime_state.beta_tildes[gene_sets_for_hyper_mask],
            ses=runtime_state.ses[gene_sets_for_hyper_mask],
            V=V,
            X_orig=runtime_state.X_orig[:, gene_sets_for_hyper_mask],
            scale_factors=runtime_state.scale_factors[gene_sets_for_hyper_mask],
            mean_shifts=runtime_state.mean_shifts[gene_sets_for_hyper_mask],
            is_dense_gene_set=runtime_state.is_dense_gene_set[gene_sets_for_hyper_mask],
            ps=None,
            max_num_burn_in=max_num_burn_in,
            max_num_iter=max_num_iter_betas,
            min_num_iter=min_num_iter_betas,
            num_chains=num_chains_betas,
            r_threshold_burn_in=r_threshold_burn_in_betas,
            use_max_r_for_convergence=use_max_r_for_convergence_betas,
            max_frac_sem=max_frac_sem_betas,
            max_allowed_batch_correlation=max_allowed_batch_correlation,
            gauss_seidel=False,
            update_hyper_sigma=cur_update_hyper_sigma,
            update_hyper_p=cur_update_hyper_p,
            only_update_hyper=True,
            adjust_hyper_sigma_p=adjust_hyper_sigma_p,
            sigma_num_devs_to_top=sigma_num_devs_to_top,
            p_noninf_inflate=p_noninf_inflate,
            num_p_pseudo=num_p_pseudo,
            num_missing_gene_sets=num_missing_gene_sets,
            sparse_solution=sparse_solution,
            sparse_frac_betas=sparse_frac_betas,
            betas_trace_out=betas_trace_out,
            betas_trace_gene_sets=[
                runtime_state.gene_sets[j]
                for j in range(len(runtime_state.gene_sets))
                if gene_sets_for_hyper_mask[j]
            ],
        )

        return {
            "computed_p": runtime_state.p,
            "computed_sigma2": runtime_state.sigma2,
            "computed_sigma_power": runtime_state.sigma_power,
        }


def apply_learned_batch_hyper_values(
    runtime_state,
    gene_sets_in_batch_mask,
    computed_p,
    computed_sigma2,
    first_p,
    first_max_p_for_hyper,
):
    updated_first_p = first_p
    adjusted_p = computed_p
    adjusted_sigma2 = computed_sigma2

    if updated_first_p is None:
        updated_first_p = adjusted_p
    elif first_max_p_for_hyper and adjusted_p > updated_first_p:
        adjusted_sigma2 = adjusted_sigma2 / adjusted_p * updated_first_p
        adjusted_p = updated_first_p

    runtime_state.ps[gene_sets_in_batch_mask] = adjusted_p
    runtime_state.sigma2s[gene_sets_in_batch_mask] = adjusted_sigma2
    return updated_first_p


def finalize_batch_hyper_vectors(runtime_state, first_for_hyper):
    runtime_state.ps = np.array(
        [np.nan if value is None else value for value in runtime_state.ps],
        dtype=float,
    )
    runtime_state.sigma2s = np.array(
        [np.nan if value is None else value for value in runtime_state.sigma2s],
        dtype=float,
    )

    assert len(runtime_state.ps) > 0 and not np.isnan(runtime_state.ps[0])
    assert len(runtime_state.sigma2s) > 0 and not np.isnan(runtime_state.sigma2s[0])

    if first_for_hyper:
        runtime_state.ps[np.isnan(runtime_state.ps)] = runtime_state.ps[0]
        runtime_state.sigma2s[np.isnan(runtime_state.sigma2s)] = runtime_state.sigma2s[0]
    else:
        runtime_state.ps[np.isnan(runtime_state.ps)] = np.mean(runtime_state.ps[~np.isnan(runtime_state.ps)])
        runtime_state.sigma2s[np.isnan(runtime_state.sigma2s)] = np.mean(runtime_state.sigma2s[~np.isnan(runtime_state.sigma2s)])

    runtime_state.set_p(np.mean(runtime_state.ps))
    runtime_state.set_sigma(np.mean(runtime_state.sigma2s), runtime_state.sigma_power)


def maybe_learn_batch_hyper_after_x_read_for_runtime(
    runtime_state,
    skip_betas,
    update_hyper_p,
    update_hyper_sigma,
    batches,
    num_ignored_gene_sets,
    first_for_hyper,
    max_num_gene_sets_hyper,
    first_for_sigma_cond,
    fixed_sigma_cond,
    first_max_p_for_hyper,
    max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    sigma_num_devs_to_top,
    p_noninf_inflate,
    sparse_solution,
    sparse_frac_betas,
    betas_trace_out,
    *,
    log_fn,
    debug_level,
):
    if skip_betas or runtime_state.p_values is None or (not update_hyper_p and not update_hyper_sigma) or len(runtime_state.gene_set_batches) == 0:
        return

    assert runtime_state.gene_set_batches[0] is not None
    ordered_batches = [runtime_state.gene_set_batches[0]] + list(set([x for x in runtime_state.gene_set_batches if x != runtime_state.gene_set_batches[0]]))
    batches_num_ignored = {}
    for i in range(len(batches)):
        if batches[i] not in batches_num_ignored:
            batches_num_ignored[batches[i]] = 0
        batches_num_ignored[batches[i]] += num_ignored_gene_sets[i]

    if update_hyper_p:
        runtime_state.ps = np.full(len(runtime_state.gene_set_batches), np.nan)
    runtime_state.sigma2s = np.full(len(runtime_state.gene_set_batches), np.nan)

    first_p = None
    for ordered_batch_ind in range(len(ordered_batches)):
        if ordered_batches[ordered_batch_ind] is None:
            assert first_for_hyper
            continue

        gene_sets_in_batch_mask = runtime_state.gene_set_batches == ordered_batches[ordered_batch_ind]
        gene_sets_for_hyper_mask = gene_sets_in_batch_mask.copy()

        if max_num_gene_sets_hyper is not None and np.sum(gene_sets_for_hyper_mask) > max_num_gene_sets_hyper:
            drop_mask = np.random.default_rng().choice(
                np.where(gene_sets_for_hyper_mask)[0],
                size=np.sum(gene_sets_for_hyper_mask) - runtime_state.batch_size,
                replace=False,
            )
            log_fn(
                "Dropping %d gene sets to reduce gene sets used for hyper parameters to %d"
                % (len(drop_mask), max_num_gene_sets_hyper),
                debug_level,
            )
            gene_sets_for_hyper_mask[drop_mask] = False

        if ordered_batch_ind > 0 and np.sum(gene_sets_for_hyper_mask) + batches_num_ignored[ordered_batches[ordered_batch_ind]] < 100:
            log_fn("Skipping learning hyper for batch %s since not enough gene sets" % (ordered_batches[ordered_batch_ind]))
            continue

        hyper_fit = learn_hyper_for_gene_set_batch(
            runtime_state=runtime_state,
            gene_sets_for_hyper_mask=gene_sets_for_hyper_mask,
            num_missing_gene_sets=batches_num_ignored[ordered_batches[ordered_batch_ind]],
            update_hyper_p=update_hyper_p,
            update_hyper_sigma=update_hyper_sigma,
            first_for_sigma_cond=first_for_sigma_cond,
            fixed_sigma_cond=fixed_sigma_cond,
            ordered_batch_ind=ordered_batch_ind,
            max_num_burn_in=max_num_burn_in,
            max_num_iter_betas=max_num_iter_betas,
            min_num_iter_betas=min_num_iter_betas,
            num_chains_betas=num_chains_betas,
            r_threshold_burn_in_betas=r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=use_max_r_for_convergence_betas,
            max_frac_sem_betas=max_frac_sem_betas,
            max_allowed_batch_correlation=max_allowed_batch_correlation,
            sigma_num_devs_to_top=sigma_num_devs_to_top,
            p_noninf_inflate=p_noninf_inflate,
            sparse_solution=sparse_solution,
            sparse_frac_betas=sparse_frac_betas,
            betas_trace_out=betas_trace_out,
        )
        computed_p = hyper_fit["computed_p"]
        computed_sigma2 = hyper_fit["computed_sigma2"]
        computed_sigma_power = hyper_fit["computed_sigma_power"]

        log_fn("Learned p=%.4g, sigma2=%.4g (sigma2/p=%.4g)" % (computed_p, computed_sigma2, computed_sigma2 / computed_p))
        runtime_state._record_params(
            {
                "p": computed_p,
                "sigma2": computed_sigma2,
                "sigma2_cond": computed_sigma2 / computed_p,
                "sigma_power": computed_sigma_power,
                "sigma_threshold_k": runtime_state.sigma_threshold_k,
                "sigma_threshold_xo": runtime_state.sigma_threshold_xo,
            }
        )

        first_p = apply_learned_batch_hyper_values(
            runtime_state=runtime_state,
            gene_sets_in_batch_mask=gene_sets_in_batch_mask,
            computed_p=computed_p,
            computed_sigma2=computed_sigma2,
            first_p=first_p,
            first_max_p_for_hyper=first_max_p_for_hyper,
        )

    finalize_batch_hyper_vectors(runtime_state=runtime_state, first_for_hyper=first_for_hyper)


def maybe_filter_zero_uncorrected_betas_after_x_read_for_runtime(
    runtime_state,
    sort_rank,
    skip_betas,
    filter_gene_set_p,
    filter_using_phewas,
    max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    sparse_solution,
    sparse_frac_betas,
    *,
    log_fn,
):
    if skip_betas or runtime_state.p_values is None or filter_gene_set_p >= 1 or filter_using_phewas:
        return sort_rank

    betas, _avg_postp = runtime_state._calculate_non_inf_betas(
        initial_p=None,
        assume_independent=True,
        max_num_burn_in=max_num_burn_in,
        max_num_iter=max_num_iter_betas,
        min_num_iter=min_num_iter_betas,
        num_chains=num_chains_betas,
        r_threshold_burn_in=r_threshold_burn_in_betas,
        use_max_r_for_convergence=use_max_r_for_convergence_betas,
        max_frac_sem=max_frac_sem_betas,
        max_allowed_batch_correlation=max_allowed_batch_correlation,
        gauss_seidel=False,
        update_hyper_sigma=False,
        update_hyper_p=False,
        adjust_hyper_sigma_p=False,
        sparse_solution=sparse_solution,
        sparse_frac_betas=sparse_frac_betas,
    )

    log_fn("%d have betas uncorrected equal 0" % np.sum(betas == 0))
    log_fn("%d have betas uncorrected below 0.001" % np.sum(betas < 0.001))
    log_fn("%d have betas uncorrected below 0.01" % np.sum(betas < 0.01))

    beta_ignore = betas == 0
    beta_mask = ~beta_ignore
    if np.sum(beta_mask) > 0:
        log_fn("Ignoring %d gene sets due to zero uncorrected betas (kept %d)" % (np.sum(beta_ignore), np.sum(beta_mask)))
        runtime_state.subset_gene_sets(beta_mask, keep_missing=False, ignore_missing=True, skip_V=True)
    else:
        log_fn("Keeping %d gene sets with zero uncorrected betas to avoid having none" % (np.sum(beta_ignore)))

    return -np.abs(betas[beta_mask])


def maybe_reduce_gene_sets_to_max_after_x_read_for_runtime(
    runtime_state,
    skip_betas,
    max_num_gene_sets,
    sort_rank,
    *,
    log_fn,
    debug_level,
    trace_level,
):
    if skip_betas or max_num_gene_sets is None or max_num_gene_sets <= 0:
        return
    if len(runtime_state.gene_sets) <= max_num_gene_sets:
        return

    log_fn(
        "Current %d gene sets is greater than the maximum specified %d; reducing using pruning + small beta removal"
        % (len(runtime_state.gene_sets), max_num_gene_sets),
        debug_level,
    )
    gene_set_masks = runtime_state._compute_gene_set_batches(
        V=None,
        X_orig=runtime_state.X_orig,
        mean_shifts=runtime_state.mean_shifts,
        scale_factors=runtime_state.scale_factors,
        sort_values=sort_rank,
        resort_as_added=True,
        stop_at=max_num_gene_sets,
    )
    keep_mask = np.full(len(runtime_state.gene_sets), False)
    for gene_set_mask in gene_set_masks:
        keep_mask[gene_set_mask] = True
        log_fn(
            "Adding %d relatively uncorrelated gene sets (total now %d)" % (np.sum(gene_set_mask), np.sum(keep_mask)),
            trace_level,
        )
        if np.sum(keep_mask) > max_num_gene_sets:
            break
    if np.sum(keep_mask) > max_num_gene_sets:
        keep_indices = np.where(keep_mask)[0]
        if sort_rank is not None:
            keep_indices = keep_indices[np.argsort(sort_rank[keep_indices], kind="stable")]
        trimmed_keep_mask = np.full(len(runtime_state.gene_sets), False)
        trimmed_keep_mask[keep_indices[:max_num_gene_sets]] = True
        keep_mask = trimmed_keep_mask
    if np.sum(~keep_mask) > 0:
        runtime_state.subset_gene_sets(keep_mask, keep_missing=False, ignore_missing=True, skip_V=True)


def estimate_dense_chunk_size(gene_set_count, only_ids, default_chunk_size=500):
    max_num_at_once = default_chunk_size
    if only_ids and len(only_ids) < gene_set_count:
        max_num_at_once = int(max_num_at_once / (float(len(only_ids)) / gene_set_count))
    return max_num_at_once


def record_x_addition(
    runtime_state,
    num_added,
    num_ignored,
    batch_value,
    label_value,
    initial_p_value,
    num_ignored_gene_sets,
    input_index,
    fail_if_first_empty=False,
    *,
    bail_fn,
):
    if fail_if_first_empty and num_added + num_ignored == 0:
        bail_fn("--first-for-hyper was specified but first file had no gene sets")

    runtime_state.gene_set_batches = np.append(runtime_state.gene_set_batches, np.full(num_added, batch_value))
    runtime_state.gene_set_labels = np.append(runtime_state.gene_set_labels, np.full(num_added, label_value))
    if runtime_state.ps is not None and initial_p_value is not None:
        runtime_state.ps = np.append(runtime_state.ps, np.full(num_added, initial_p_value))
    runtime_state.gene_set_labels_ignored = np.append(runtime_state.gene_set_labels_ignored, np.full(num_ignored, label_value))
    num_ignored_gene_sets[input_index] += num_ignored


def process_dense_x_file(
    runtime_state,
    X_in,
    tag,
    only_ids,
    x_sparsify,
    batch_value,
    label_value,
    initial_p_value,
    num_ignored_gene_sets,
    input_index,
    add_to_x_fn,
    *,
    open_gz_fn,
    warn_fn,
    bail_fn,
    log_fn,
    debug_level,
    ext_tag,
    top_tag,
    bot_tag,
):
    with open_gz_fn(X_in) as gene_sets_fh:
        header = gene_sets_fh.readline().strip("\n")
        header = header.lstrip("# \t")
        gene_sets = header.split()
        if len(gene_sets) < 2:
            warn_fn("First line of --Xd-in %s must contain gene column followed by list of gene sets; skipping file" % X_in)
            return False

        gene_sets = gene_sets[1:]

        max_num_at_once = estimate_dense_chunk_size(
            len(gene_sets),
            only_ids=only_ids,
            default_chunk_size=500,
        )

        if len(gene_sets) > max_num_at_once:
            log_fn("Splitting reading of file into chunks to limit memory", debug_level)
        for j in range(0, len(gene_sets), max_num_at_once):
            if len(gene_sets) > max_num_at_once:
                log_fn(
                    "Reading gene sets %d-%d" % (j + 1, j + min(len(gene_sets), j + max_num_at_once + 1)),
                    debug_level,
                )

            gene_set_indices_to_load = list(range(j, min(len(gene_sets), j + max_num_at_once)))

            gene_set_indices_to_load = filter_dense_chunk_gene_set_indices(
                gene_sets,
                chunk_indices=gene_set_indices_to_load,
                only_ids=only_ids,
                x_sparsify=x_sparsify,
                ext_tag=ext_tag,
                top_tag=top_tag,
                bot_tag=bot_tag,
            )
            if only_ids is not None:
                if len(gene_set_indices_to_load) > 0:
                    log_fn("Will load %d gene sets that were requested" % len(gene_set_indices_to_load), debug_level)
                else:
                    continue

            indices_to_load = [0] + [k + 1 for k in gene_set_indices_to_load]

            cur_X = np.loadtxt(X_in, skiprows=1, dtype=str, usecols=indices_to_load)

            if len(cur_X.shape) == 1:
                cur_X = cur_X[:, np.newaxis]

            if cur_X.shape[1] != len(indices_to_load):
                bail_fn("Xd matrix %s dimensions %s do not match number of gene sets in header line (%s)" % (X_in, cur_X.shape, len(gene_sets)))
            cur_gene_sets = [gene_sets[k] for k in gene_set_indices_to_load]

            genes = cur_X[:, 0]
            if runtime_state.gene_label_map is not None:
                genes = list(map(lambda x: runtime_state.gene_label_map[x] if x in runtime_state.gene_label_map else x, genes))

            mat_info = cur_X[:, 1:].astype(float)
            num_added, num_ignored = add_to_x_fn(
                mat_info,
                genes,
                cur_gene_sets,
                tag,
                skip_scale_factors=False,
            )
            record_x_addition(
                runtime_state,
                num_added=num_added,
                num_ignored=num_ignored,
                batch_value=batch_value,
                label_value=label_value,
                initial_p_value=initial_p_value,
                num_ignored_gene_sets=num_ignored_gene_sets,
                input_index=input_index,
                fail_if_first_empty=(input_index == 0),
                bail_fn=bail_fn,
            )

    return True


def process_sparse_x_file(
    runtime_state,
    X_in,
    tag,
    only_ids,
    min_gene_set_size,
    only_inc_genes,
    fraction_inc_genes,
    ignore_genes,
    max_num_entries_at_once,
    batch_value,
    label_value,
    initial_p_value,
    num_ignored_gene_sets,
    input_index,
    add_to_x_fn,
    *,
    open_gz_fn,
    warn_fn,
    bail_fn,
    log_fn,
):
    (
        genes,
        gene_to_ind,
        new_gene_to_ind,
        gene_sets,
        data,
        row,
        col,
        num_read,
        cur_num_read,
    ) = init_sparse_x_batch_state(runtime_state)
    gene_set_to_ind = {}
    num_too_small = 0
    ignored_for_fraction_inc = 0

    with open_gz_fn(X_in) as gene_sets_fh:
        if max_num_entries_at_once is None:
            max_num_entries_at_once = 200 * 10000

        already_seen = 0
        for line in gene_sets_fh:
            line = line.strip("\n")
            cols = line.split()

            if len(cols) < 2:
                warn_fn("Line does not match format for --X-in: %s" % (line))
                continue
            gs = cols[0]

            if only_ids is not None and gs not in only_ids:
                continue

            if gs in gene_set_to_ind or (runtime_state.gene_set_to_ind is not None and gs in runtime_state.gene_set_to_ind):
                already_seen += 1
                continue

            cur_genes = set(cols[1:])
            if runtime_state.gene_label_map is not None:
                cur_genes = set(map(lambda x: runtime_state.gene_label_map[x] if x in runtime_state.gene_label_map else x, cur_genes))

            if len(cur_genes) < min_gene_set_size:
                num_too_small += 1
                continue

            gene_set_ind = len(gene_sets)
            gene_sets.append(gs)
            gene_set_to_ind[gs] = gene_set_ind

            if only_inc_genes is not None:
                fraction_match = len(only_inc_genes.intersection(cur_genes)) / float(len(only_inc_genes))
                if fraction_match < (fraction_inc_genes if fraction_inc_genes is not None else 1e-5):
                    ignored_for_fraction_inc += 1
                    continue

            for gene in cur_genes:
                gene_array = gene.split(":")
                gene = gene_array[0]
                if gene in ignore_genes:
                    continue
                if len(gene_array) == 2:
                    try:
                        weight = float(gene_array[1])
                    except ValueError:
                        warn_fn("Couldn't convert weight %s to number so skipping token: %s" % (weight, ":".join(gene_array)))
                        continue
                else:
                    weight = 1.0

                if gene_to_ind is not None and gene in gene_to_ind:
                    gene_ind = gene_to_ind[gene]
                else:
                    if gene not in new_gene_to_ind:
                        gene_ind = len(new_gene_to_ind)
                        if gene_to_ind is not None:
                            gene_ind += len(gene_to_ind)

                        new_gene_to_ind[gene] = gene_ind
                        genes.append(gene)
                    else:
                        gene_ind = new_gene_to_ind[gene]

                col.append(gene_set_ind)
                row.append(gene_ind)
                data.append(weight)
            num_read += 1
            cur_num_read += 1

            if len(data) >= max_num_entries_at_once:
                log_fn("Batching %d lines to save memory" % cur_num_read)
                num_added, num_ignored = add_to_x_fn((data, row, col), genes, gene_sets, tag, skip_scale_factors=False)
                record_x_addition(
                    runtime_state,
                    num_added=num_added,
                    num_ignored=num_ignored,
                    batch_value=batch_value,
                    label_value=label_value,
                    initial_p_value=initial_p_value,
                    num_ignored_gene_sets=num_ignored_gene_sets,
                    input_index=input_index,
                    fail_if_first_empty=(input_index == 0),
                    bail_fn=bail_fn,
                )

                (
                    genes,
                    gene_to_ind,
                    new_gene_to_ind,
                    gene_sets,
                    data,
                    row,
                    col,
                    num_read,
                    cur_num_read,
                ) = init_sparse_x_batch_state(runtime_state)
                log_fn("Continuing reading...")

        if already_seen > 0:
            warn_fn("Skipped second occurrence of %d repeated gene sets" % already_seen)

        mat_info = (data, row, col) if len(data) > 0 else None

    if mat_info is not None:
        num_added, num_ignored = add_to_x_fn(mat_info, genes, gene_sets, tag, skip_scale_factors=False)
        record_x_addition(
            runtime_state,
            num_added=num_added,
            num_ignored=num_ignored,
            batch_value=batch_value,
            label_value=label_value,
            initial_p_value=initial_p_value,
            num_ignored_gene_sets=num_ignored_gene_sets,
            input_index=input_index,
            fail_if_first_empty=(input_index == 0),
            bail_fn=bail_fn,
        )

    return (num_too_small, ignored_for_fraction_inc)


def process_x_input_file(
    runtime_state,
    X_in,
    tag,
    is_dense_input,
    only_ids,
    x_sparsify,
    batch_value,
    label_value,
    initial_p_value,
    num_ignored_gene_sets,
    input_index,
    add_to_x_fn,
    min_gene_set_size,
    only_inc_genes,
    fraction_inc_genes,
    ignore_genes,
    max_num_entries_at_once,
    *,
    open_gz_fn,
    warn_fn,
    bail_fn,
    log_fn,
    debug_level,
    ext_tag,
    top_tag,
    bot_tag,
):
    num_too_small = 0
    ignored_for_fraction_inc = 0

    if is_dense_input:
        processed_dense = process_dense_x_file(
            runtime_state,
            X_in=X_in,
            tag=tag,
            only_ids=only_ids,
            x_sparsify=x_sparsify,
            batch_value=batch_value,
            label_value=label_value,
            initial_p_value=initial_p_value,
            num_ignored_gene_sets=num_ignored_gene_sets,
            input_index=input_index,
            add_to_x_fn=add_to_x_fn,
            open_gz_fn=open_gz_fn,
            warn_fn=warn_fn,
            bail_fn=bail_fn,
            log_fn=log_fn,
            debug_level=debug_level,
            ext_tag=ext_tag,
            top_tag=top_tag,
            bot_tag=bot_tag,
        )
        if not processed_dense:
            return (num_too_small, ignored_for_fraction_inc, False)
    else:
        num_too_small, ignored_for_fraction_inc = process_sparse_x_file(
            runtime_state,
            X_in=X_in,
            tag=tag,
            only_ids=only_ids,
            min_gene_set_size=min_gene_set_size,
            only_inc_genes=only_inc_genes,
            fraction_inc_genes=fraction_inc_genes,
            ignore_genes=ignore_genes,
            max_num_entries_at_once=max_num_entries_at_once,
            batch_value=batch_value,
            label_value=label_value,
            initial_p_value=initial_p_value,
            num_ignored_gene_sets=num_ignored_gene_sets,
            input_index=input_index,
            add_to_x_fn=add_to_x_fn,
            open_gz_fn=open_gz_fn,
            warn_fn=warn_fn,
            bail_fn=bail_fn,
            log_fn=lambda message: log_fn(message, debug_level),
        )

    return (num_too_small, ignored_for_fraction_inc, True)


def init_sparse_x_batch_state(runtime_state):
    genes = []
    gene_to_ind = None
    if runtime_state.genes is not None:
        genes = copy.copy(runtime_state.genes)
        if runtime_state.genes_missing is not None:
            genes += runtime_state.genes_missing
        gene_to_ind = construct_map_to_ind(genes)

    return (
        genes,
        gene_to_ind,
        {},
        [],
        [],
        [],
        [],
        0,
        0,
    )


def filter_dense_chunk_gene_set_indices(gene_sets, chunk_indices, only_ids, x_sparsify, *, ext_tag, top_tag, bot_tag):
    if only_ids is None:
        return chunk_indices

    keep_mask = np.full(len(chunk_indices), False)
    for k in range(len(keep_mask)):
        gs = gene_sets[chunk_indices[k]]
        if gs in only_ids:
            keep_mask[k] = True
        elif x_sparsify is not None:
            for top_number in x_sparsify:
                matched = False
                for sparse_tag in [ext_tag, top_tag, bot_tag]:
                    if "%s_%s%d" % (gs, sparse_tag, top_number) in only_ids:
                        keep_mask[k] = True
                        matched = True
                        break
                if matched:
                    break

    if np.any(keep_mask):
        return [chunk_indices[i] for i in range(len(keep_mask)) if keep_mask[i]]
    return []


def normalize_gene_set_weights(runtime_state, cur_X, threshold_weights, cap_weights):
    denom = runtime_state.get_col_sums(cur_X, num_nonzero=True)
    denom[denom == 0] = 1
    avg_weights = np.abs(cur_X).sum(axis=0) / denom
    if np.sum(avg_weights != 1) > 0:
        max_weight_devs = None
        if max_weight_devs is not None:
            dev_weights = np.sqrt(np.abs(cur_X).power(2).sum(axis=0) / denom - np.power(avg_weights, 2))
            temp_X = copy.copy(np.abs(cur_X))
            temp_X[temp_X > avg_weights + max_weight_devs * dev_weights] = 0
            weight_norm = temp_X.max(axis=0).todense().A1
        else:
            weight_norm = avg_weights.A1

        weight_norm = np.round(weight_norm, 10)
        weight_norm[weight_norm == 0] = 1

        normalize_mask = (np.abs(cur_X) > 1).sum(axis=0).A1 > 0
        if threshold_weights is not None and threshold_weights > 0:
            normalize_mask = np.logical_or(
                normalize_mask,
                (np.abs(cur_X) >= threshold_weights).sum(axis=0).A1 != (np.abs(cur_X) > 0).sum(axis=0).A1,
            )

        weight_norm[~normalize_mask] = 1.0
        cur_X = sparse.csc_matrix(cur_X.multiply(1.0 / weight_norm))

        if threshold_weights is not None and threshold_weights > 0:
            cur_X.data[np.abs(cur_X.data) < threshold_weights] = 0
            if cap_weights:
                cur_X.data[cur_X.data > 1] = 1
                cur_X.data[cur_X.data < -1] = -1
        cur_X.eliminate_zeros()

    return cur_X


def maybe_permute_gene_set_rows(runtime_state, cur_X, permute_gene_sets):
    if not permute_gene_sets:
        return cur_X

    if runtime_state.Y is not None:
        assert len(runtime_state.Y) == len(runtime_state.genes)
        orig_indices = list(range(len(runtime_state.Y)))
        new_indices = random.sample(orig_indices, len(orig_indices))
        if cur_X.shape[0] > len(orig_indices):
            num_to_add = cur_X.shape[0] - len(orig_indices)
            to_add = list(range(len(orig_indices), len(orig_indices) + num_to_add))
            orig_indices += to_add
            new_indices += random.sample(to_add, len(to_add))
    else:
        orig_indices = list(range(cur_X.shape[0]))
        new_indices = random.sample(orig_indices, len(orig_indices))

    index_map = dict(zip(orig_indices, new_indices))
    cur_X = sparse.csc_matrix(cur_X)
    return sparse.csc_matrix(
        (cur_X.data, [index_map[x] for x in cur_X.indices], cur_X.indptr),
        shape=(cur_X.shape[0], cur_X.shape[1]),
    )


def align_prefilter_gene_set_signs(cur_X, beta_tildes, z_scores, *, log_fn, debug_level):
    negative_weights_mask = (cur_X < 0).sum(axis=0).A1 > 0
    if np.sum(negative_weights_mask) > 0:
        flip_mask = np.logical_and(beta_tildes < 0, negative_weights_mask)
        if np.sum(flip_mask) > 0:
            log_fn("Flipped %d gene sets" % np.sum(flip_mask), debug_level)
            beta_tildes[flip_mask] = -beta_tildes[flip_mask]
            z_scores[flip_mask] = -z_scores[flip_mask]
            cur_X[:, flip_mask] = -cur_X[:, flip_mask]
    return (cur_X, beta_tildes, z_scores)


def build_prefilter_keep_mask(
    p_values,
    beta_tildes,
    filter_gene_set_p,
    filter_using_phewas=False,
    p_values_phewas=None,
    beta_tildes_phewas=None,
    increase_filter_gene_set_p=None,
    filter_negative=True,
    *,
    log_fn,
    debug_level,
):
    p_value_mask = p_values <= filter_gene_set_p
    if filter_using_phewas:
        p_value_mask = np.logical_or(p_value_mask, np.any(p_values_phewas <= filter_gene_set_p, axis=0))

    if increase_filter_gene_set_p is not None and np.mean(p_value_mask) < increase_filter_gene_set_p:
        p_from_quantile = np.quantile(p_values, increase_filter_gene_set_p)
        log_fn(
            "Choosing revised p threshold %.3g to ensure keeping %.3g fraction of gene sets"
            % (p_from_quantile, increase_filter_gene_set_p),
            debug_level,
        )
        p_value_mask = p_values <= p_from_quantile
        if filter_using_phewas:
            p_value_mask = np.logical_or(p_value_mask, np.any(p_values_phewas <= p_from_quantile, axis=0))

        if np.sum(~p_value_mask) > 0:
            log_fn("Ignoring %d gene sets due to p-value filters" % (np.sum(~p_value_mask)), debug_level)

    if filter_negative:
        negative_beta_tildes_mask = beta_tildes < 0
        if filter_using_phewas:
            negative_beta_tildes_mask = np.logical_and(negative_beta_tildes_mask, np.all(beta_tildes_phewas < 0, axis=0))
        p_value_mask = np.logical_and(p_value_mask, ~negative_beta_tildes_mask)
        if np.sum(negative_beta_tildes_mask) > 0:
            log_fn("Ignoring %d gene sets due to negative beta filters" % (np.sum(negative_beta_tildes_mask)), debug_level)

    return p_value_mask


def compute_prefilter_qc_metrics(runtime_state, cur_X):
    total_qc_metrics = None
    mean_qc_metrics = None
    total_qc_metrics_directions = None
    if runtime_state.gene_covariates is None:
        return (total_qc_metrics, mean_qc_metrics, total_qc_metrics_directions)

    cur_X_size = np.abs(cur_X).sum(axis=0)
    cur_X_size[cur_X_size == 0] = 1

    total_qc_metrics = (np.array(cur_X.T.dot(runtime_state.gene_covariate_zs).T / cur_X_size)).T
    total_qc_metrics = np.hstack(
        (
            total_qc_metrics[:, : runtime_state.gene_covariate_intercept_index],
            total_qc_metrics[:, runtime_state.gene_covariate_intercept_index + 1 :],
        )
    )

    total_qc_metrics_directions = np.append(
        runtime_state.gene_covariate_directions[: runtime_state.gene_covariate_intercept_index],
        runtime_state.gene_covariate_directions[runtime_state.gene_covariate_intercept_index + 1 :],
    )

    total_huge_adjustments = (np.array(cur_X.T.dot(runtime_state.gene_covariate_adjustments).T / cur_X_size)).T

    total_qc_metrics = np.hstack((total_qc_metrics, total_huge_adjustments))
    total_qc_metrics_directions = np.append(total_qc_metrics_directions, -1)

    if runtime_state.debug_only_avg_huge:
        total_qc_metrics = total_huge_adjustments
        total_qc_metrics_directions = np.array(-1)

    mean_qc_metrics = total_huge_adjustments.squeeze()
    mean_qc_metrics = total_huge_adjustments
    if len(mean_qc_metrics.shape) == 2 and mean_qc_metrics.shape[1] == 1:
        mean_qc_metrics = mean_qc_metrics.squeeze(axis=1)

    return (total_qc_metrics, mean_qc_metrics, total_qc_metrics_directions)


def compute_prefilter_assoc_stats(runtime_state, cur_X, run_logistic, filter_using_phewas, mean_shifts, scale_factors):
    Y_to_use = runtime_state.Y_for_regression
    gene_pheno_Y = runtime_state.gene_pheno_Y.T.toarray() if filter_using_phewas else None

    if run_logistic:
        Y = np.exp(Y_to_use + runtime_state.background_log_bf) / (1 + np.exp(Y_to_use + runtime_state.background_log_bf))
        (
            beta_tildes,
            ses,
            z_scores,
            p_values,
            se_inflation_factors,
            _alpha_tildes,
            _diverged,
        ) = runtime_state._compute_logistic_beta_tildes(
            cur_X,
            Y,
            scale_factors,
            mean_shifts,
            resid_correlation_matrix=runtime_state.y_corr_sparse,
        )

        beta_tildes_phewas = None
        p_values_phewas = None
        if filter_using_phewas:
            gene_pheno_Y = np.exp(np.array(gene_pheno_Y) + runtime_state.background_log_bf) / (1 + np.exp(gene_pheno_Y + runtime_state.background_log_bf))
            (
                beta_tildes_phewas,
                _ses_phewas,
                _z_scores_phewas,
                p_values_phewas,
                _se_inflation_factors_phewas,
                _alpha_tildes_phewas,
                _diverged_phewas,
            ) = runtime_state._compute_logistic_beta_tildes(
                cur_X,
                gene_pheno_Y,
                scale_factors,
                mean_shifts,
                resid_correlation_matrix=runtime_state.y_corr_sparse,
            )
    else:
        (
            beta_tildes,
            ses,
            z_scores,
            p_values,
            se_inflation_factors,
        ) = runtime_state._compute_beta_tildes(
            cur_X,
            Y_to_use,
            np.var(Y_to_use),
            scale_factors,
            mean_shifts,
            resid_correlation_matrix=runtime_state.y_corr_sparse,
        )
        beta_tildes_phewas = None
        p_values_phewas = None
        if filter_using_phewas:
            (
                beta_tildes_phewas,
                _ses_phewas,
                _z_scores_phewas,
                p_values_phewas,
                _se_inflation_factors_phewas,
            ) = runtime_state._compute_beta_tildes(
                cur_X,
                gene_pheno_Y,
                None,
                scale_factors,
                mean_shifts,
                resid_correlation_matrix=runtime_state.y_corr_sparse,
            )

    return (
        beta_tildes,
        ses,
        z_scores,
        p_values,
        se_inflation_factors,
        beta_tildes_phewas,
        p_values_phewas,
    )


def apply_prefilter_and_record(
    runtime_state,
    cur_X,
    gene_sets,
    p_value_mask,
    filter_gene_set_p,
    filter_gene_set_metric_z,
    scale_factors,
    mean_shifts,
    beta_tildes,
    p_values,
    ses,
    z_scores,
    se_inflation_factors,
    total_qc_metrics,
    mean_qc_metrics,
    cur_X_missing_genes_new,
    cur_X_missing_genes_int,
    *,
    log_fn,
):
    p_value_ignore = np.full(len(p_value_mask), False)
    gene_ignored_N = None
    gene_ignored_N_missing_new = None
    gene_ignored_N_missing_int = None

    if filter_gene_set_p < 1 or filter_gene_set_metric_z is not None:
        p_value_ignore = ~p_value_mask
        if np.sum(p_value_ignore) > 0:
            log_fn("Kept %d gene sets after p-value and beta filters" % (np.sum(p_value_mask)))

        if runtime_state.gene_sets_ignored is None:
            runtime_state.gene_sets_ignored = []
        if runtime_state.col_sums_ignored is None:
            runtime_state.col_sums_ignored = np.array([])
        if runtime_state.scale_factors_ignored is None:
            runtime_state.scale_factors_ignored = np.array([])
        if runtime_state.mean_shifts_ignored is None:
            runtime_state.mean_shifts_ignored = np.array([])
        if runtime_state.beta_tildes_ignored is None:
            runtime_state.beta_tildes_ignored = np.array([])
        if runtime_state.p_values_ignored is None:
            runtime_state.p_values_ignored = np.array([])
        if runtime_state.ses_ignored is None:
            runtime_state.ses_ignored = np.array([])
        if runtime_state.z_scores_ignored is None:
            runtime_state.z_scores_ignored = np.array([])
        if runtime_state.se_inflation_factors_ignored is None:
            runtime_state.se_inflation_factors_ignored = np.array([])

        runtime_state.gene_sets_ignored = runtime_state.gene_sets_ignored + [gene_sets[i] for i in range(len(gene_sets)) if p_value_ignore[i]]
        gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if p_value_mask[i]]

        runtime_state.col_sums_ignored = np.append(runtime_state.col_sums_ignored, runtime_state.get_col_sums(cur_X[:, p_value_ignore]))
        runtime_state.scale_factors_ignored = np.append(runtime_state.scale_factors_ignored, scale_factors[p_value_ignore])
        runtime_state.mean_shifts_ignored = np.append(runtime_state.mean_shifts_ignored, mean_shifts[p_value_ignore])
        runtime_state.beta_tildes_ignored = np.append(runtime_state.beta_tildes_ignored, beta_tildes[p_value_ignore])
        runtime_state.p_values_ignored = np.append(runtime_state.p_values_ignored, p_values[p_value_ignore])
        runtime_state.ses_ignored = np.append(runtime_state.ses_ignored, ses[p_value_ignore])
        runtime_state.z_scores_ignored = np.append(runtime_state.z_scores_ignored, z_scores[p_value_ignore])

        runtime_state.beta_tildes = np.append(runtime_state.beta_tildes, beta_tildes[p_value_mask])
        runtime_state.p_values = np.append(runtime_state.p_values, p_values[p_value_mask])
        runtime_state.ses = np.append(runtime_state.ses, ses[p_value_mask])
        runtime_state.z_scores = np.append(runtime_state.z_scores, z_scores[p_value_mask])

        if se_inflation_factors is not None:
            runtime_state.se_inflation_factors_ignored = np.append(
                runtime_state.se_inflation_factors_ignored,
                se_inflation_factors[p_value_ignore],
            )
            if runtime_state.se_inflation_factors is None:
                runtime_state.se_inflation_factors = np.array([])
            runtime_state.se_inflation_factors = np.append(
                runtime_state.se_inflation_factors,
                se_inflation_factors[p_value_mask],
            )

        if runtime_state.gene_covariates is not None:
            if runtime_state.total_qc_metrics_ignored is None:
                runtime_state.total_qc_metrics_ignored = total_qc_metrics[p_value_ignore, :]
                runtime_state.mean_qc_metrics_ignored = mean_qc_metrics[p_value_ignore]
            else:
                runtime_state.total_qc_metrics_ignored = np.vstack((runtime_state.total_qc_metrics_ignored, total_qc_metrics[p_value_ignore, :]))
                runtime_state.mean_qc_metrics_ignored = np.append(runtime_state.mean_qc_metrics_ignored, mean_qc_metrics[p_value_ignore])

            total_qc_metrics = total_qc_metrics[p_value_mask]
            mean_qc_metrics = mean_qc_metrics[p_value_mask]

        gene_ignored_N = runtime_state.get_col_sums(cur_X[:, p_value_ignore], axis=1)

        if cur_X_missing_genes_new is not None:
            gene_ignored_N_missing_new = np.array(np.abs(cur_X_missing_genes_new[:, p_value_ignore]).sum(axis=1)).flatten()
            cur_X_missing_genes_new = cur_X_missing_genes_new[:, p_value_mask]

        if cur_X_missing_genes_int is not None:
            gene_ignored_N_missing_int = np.array(np.abs(cur_X_missing_genes_int[:, p_value_ignore]).sum(axis=1)).flatten()
            cur_X_missing_genes_int = cur_X_missing_genes_int[:, p_value_mask]

        cur_X = cur_X[:, p_value_mask]

    return (
        cur_X,
        gene_sets,
        p_value_ignore,
        gene_ignored_N,
        cur_X_missing_genes_new,
        gene_ignored_N_missing_new,
        cur_X_missing_genes_int,
        gene_ignored_N_missing_int,
        total_qc_metrics,
        mean_qc_metrics,
    )


def maybe_prefilter_x_block(
    runtime_state,
    cur_X,
    gene_sets,
    run_logistic,
    filter_gene_set_p,
    filter_gene_set_metric_z,
    filter_using_phewas,
    increase_filter_gene_set_p,
    filter_negative,
    cur_X_missing_genes_new,
    gene_ignored_N_missing_new,
    cur_X_missing_genes_int,
    gene_ignored_N_missing_int,
    gene_ignored_N,
    *,
    log_fn,
    debug_level,
):
    p_value_ignore = None
    total_qc_metrics = None
    mean_qc_metrics = None
    total_qc_metrics_directions = None

    if (filter_gene_set_p < 1 or filter_gene_set_metric_z is not None) and runtime_state.Y is not None:
        log_fn("Analyzing gene sets to pre-filter")

        (mean_shifts, scale_factors) = runtime_state._calc_X_shift_scale(cur_X)

        (
            total_qc_metrics,
            mean_qc_metrics,
            total_qc_metrics_directions,
        ) = compute_prefilter_qc_metrics(runtime_state, cur_X)

        (
            beta_tildes,
            ses,
            z_scores,
            p_values,
            se_inflation_factors,
            beta_tildes_phewas,
            p_values_phewas,
        ) = compute_prefilter_assoc_stats(
            runtime_state,
            cur_X=cur_X,
            run_logistic=run_logistic,
            filter_using_phewas=filter_using_phewas,
            mean_shifts=mean_shifts,
            scale_factors=scale_factors,
        )

        cur_X, beta_tildes, z_scores = align_prefilter_gene_set_signs(
            cur_X,
            beta_tildes=beta_tildes,
            z_scores=z_scores,
            log_fn=log_fn,
            debug_level=debug_level,
        )

        p_value_mask = build_prefilter_keep_mask(
            p_values,
            beta_tildes=beta_tildes,
            filter_gene_set_p=filter_gene_set_p,
            filter_using_phewas=filter_using_phewas,
            p_values_phewas=p_values_phewas if filter_using_phewas else None,
            beta_tildes_phewas=beta_tildes_phewas if filter_using_phewas else None,
            increase_filter_gene_set_p=increase_filter_gene_set_p,
            filter_negative=filter_negative,
            log_fn=log_fn,
            debug_level=debug_level,
        )

        (
            cur_X,
            gene_sets,
            p_value_ignore,
            gene_ignored_N,
            cur_X_missing_genes_new,
            gene_ignored_N_missing_new,
            cur_X_missing_genes_int,
            gene_ignored_N_missing_int,
            total_qc_metrics,
            mean_qc_metrics,
        ) = apply_prefilter_and_record(
            runtime_state,
            cur_X=cur_X,
            gene_sets=gene_sets,
            p_value_mask=p_value_mask,
            filter_gene_set_p=filter_gene_set_p,
            filter_gene_set_metric_z=filter_gene_set_metric_z,
            scale_factors=scale_factors,
            mean_shifts=mean_shifts,
            beta_tildes=beta_tildes,
            p_values=p_values,
            ses=ses,
            z_scores=z_scores,
            se_inflation_factors=se_inflation_factors,
            total_qc_metrics=total_qc_metrics,
            mean_qc_metrics=mean_qc_metrics,
            cur_X_missing_genes_new=cur_X_missing_genes_new,
            cur_X_missing_genes_int=cur_X_missing_genes_int,
            log_fn=log_fn,
        )

    return (
        cur_X,
        gene_sets,
        p_value_ignore,
        gene_ignored_N,
        cur_X_missing_genes_new,
        gene_ignored_N_missing_new,
        cur_X_missing_genes_int,
        gene_ignored_N_missing_int,
        total_qc_metrics,
        mean_qc_metrics,
        total_qc_metrics_directions,
    )


def merge_missing_gene_rows(
    runtime_state,
    cur_X,
    genes,
    num_old_gene_sets,
    num_new_gene_sets,
    cur_X_missing_genes_int,
    gene_ignored_N_missing_int,
    cur_X_missing_genes_new,
    gene_ignored_N_missing_new,
    genes_missing_new,
):
    if runtime_state.genes_missing is not None:
        genes += runtime_state.genes_missing

        if runtime_state.X_orig_missing_genes is None:
            X_orig_missing_genes = sparse.csc_matrix(([], ([], [])), shape=(len(runtime_state.genes_missing), num_old_gene_sets))
        else:
            X_orig_missing_genes = copy.copy(runtime_state.X_orig_missing_genes)

        if cur_X_missing_genes_int is not None:
            if runtime_state.gene_ignored_N_missing is not None:
                if gene_ignored_N_missing_int is not None:
                    runtime_state.gene_ignored_N_missing += gene_ignored_N_missing_int
            else:
                runtime_state.gene_ignored_N_missing = gene_ignored_N_missing_int

            cur_X = sparse.vstack((cur_X, sparse.hstack((X_orig_missing_genes, cur_X_missing_genes_int))))
        elif X_orig_missing_genes is not None:
            X_orig_missing_genes.resize((X_orig_missing_genes.shape[0], X_orig_missing_genes.shape[1] + num_new_gene_sets))
            cur_X = sparse.vstack((cur_X, X_orig_missing_genes))

    if cur_X_missing_genes_new is not None:
        cur_X = sparse.vstack(
            (
                cur_X,
                sparse.hstack(
                    (
                        sparse.csc_matrix(([], ([], [])), shape=(cur_X_missing_genes_new.shape[0], num_old_gene_sets)),
                        cur_X_missing_genes_new,
                    )
                ),
            )
        )
        if runtime_state.gene_ignored_N_missing is not None:
            if gene_ignored_N_missing_new is not None:
                runtime_state.gene_ignored_N_missing = np.append(runtime_state.gene_ignored_N_missing, gene_ignored_N_missing_new)
        else:
            runtime_state.gene_ignored_N_missing = gene_ignored_N_missing_new

        genes += genes_missing_new

    return (cur_X, genes)


def finalize_added_x_block(
    runtime_state,
    cur_X,
    genes,
    gene_sets,
    skip_scale_factors,
    p_value_ignore,
    gene_ignored_N,
    total_qc_metrics,
    mean_qc_metrics,
    total_qc_metrics_directions,
):
    subset_mask = np.full(len(genes), True)
    if runtime_state.gene_to_ind is not None:
        subset_mask[[i for i in range(len(genes)) if genes[i] not in runtime_state.gene_to_ind]] = False

    num_added = cur_X.shape[1]
    if runtime_state.X_orig is not None:
        num_added -= runtime_state.X_orig.shape[1]
    num_ignored = np.sum(p_value_ignore) if p_value_ignore is not None else 0

    runtime_state._set_X(
        sparse.csc_matrix(cur_X, shape=cur_X.shape),
        genes,
        gene_sets,
        skip_scale_factors=skip_scale_factors,
        skip_V=True,
        skip_N=False,
    )

    if runtime_state.gene_ignored_N is not None:
        if gene_ignored_N is not None:
            runtime_state.gene_ignored_N += gene_ignored_N
    else:
        runtime_state.gene_ignored_N = gene_ignored_N

    if runtime_state.gene_ignored_N is not None and runtime_state.gene_ignored_N_missing is not None:
        runtime_state.gene_ignored_N = np.append(runtime_state.gene_ignored_N, runtime_state.gene_ignored_N_missing)

    runtime_state._subset_genes(
        subset_mask,
        skip_V=True,
        overwrite_missing=True,
        skip_scale_factors=False,
        skip_Y=True,
    )

    if runtime_state.gene_covariates is not None:
        if runtime_state.total_qc_metrics is None:
            runtime_state.total_qc_metrics = total_qc_metrics
            runtime_state.mean_qc_metrics = mean_qc_metrics
        else:
            runtime_state.total_qc_metrics = np.vstack((runtime_state.total_qc_metrics, total_qc_metrics))
            runtime_state.mean_qc_metrics = np.append(runtime_state.mean_qc_metrics, mean_qc_metrics)

        runtime_state.total_qc_metrics_directions = total_qc_metrics_directions

    return (num_added, num_ignored)


def partition_missing_gene_rows(runtime_state, cur_X, genes, gene_sets, *, bail_fn):
    gene_ignored_N = None

    cur_X_missing_genes_int = None
    gene_ignored_N_missing_int = None

    genes_missing_new = []
    cur_X_missing_genes_new = None
    gene_ignored_N_missing_new = None

    if (runtime_state.Y is not None and len(genes) > len(runtime_state.Y)) or (runtime_state.genes is not None):
        genes_missing_old = runtime_state.genes_missing if runtime_state.genes_missing is not None else []
        gene_missing_old_to_ind = construct_map_to_ind(genes_missing_old)

        genes_missing_new = [x for x in genes if x not in runtime_state.gene_to_ind and x not in gene_missing_old_to_ind]
        genes_missing_new_set = set(genes_missing_new)

        genes_missing_int_set = set([x for x in genes if x in gene_missing_old_to_ind])

        int_mask = np.full(len(genes), False)
        int_mask[[i for i in range(len(genes)) if genes[i] in genes_missing_int_set]] = True
        if np.sum(int_mask) > 0:
            cur_X_missing_genes_int = cur_X[int_mask, :]

        new_mask = np.full(len(genes), False)
        new_mask[[i for i in range(len(genes)) if genes[i] in genes_missing_new_set]] = True
        if np.sum(new_mask) > 0:
            cur_X_missing_genes_new = cur_X[new_mask, :]

        subset_mask = np.full(len(genes), True)
        subset_mask[[i for i in range(len(genes)) if genes[i] not in runtime_state.gene_to_ind]] = False

        cur_X = cur_X[subset_mask, :]
        genes = [x for x in genes if x in runtime_state.gene_to_ind]

        gene_set_nonempty_mask = runtime_state.get_col_sums(cur_X) > 0
        if np.sum(~gene_set_nonempty_mask) > 0:
            cur_X = cur_X[:, gene_set_nonempty_mask]

            if cur_X_missing_genes_int is not None:
                cur_X_missing_genes_int = cur_X_missing_genes_int[:, gene_set_nonempty_mask]
            if cur_X_missing_genes_new is not None:
                cur_X_missing_genes_new = cur_X_missing_genes_new[:, gene_set_nonempty_mask]
            gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if gene_set_nonempty_mask[i]]

        if runtime_state.Y is not None:
            assert len(genes) == len(runtime_state.Y)

        if cur_X.shape[1] == 0:
            bail_fn("Error: no genes overlapped Y and X; you may have forgotten to map gene names over to a common namespace")

    return (
        cur_X,
        genes,
        gene_sets,
        gene_ignored_N,
        cur_X_missing_genes_int,
        gene_ignored_N_missing_int,
        genes_missing_new,
        cur_X_missing_genes_new,
        gene_ignored_N_missing_new,
    )


def reindex_x_rows_to_current_genes(runtime_state, cur_X, genes):
    if runtime_state.genes is None:
        return (cur_X, genes)

    old_genes = genes
    genes = runtime_state.genes
    if runtime_state.genes_missing is not None:
        genes += runtime_state.genes_missing
    genes += [
        x
        for x in old_genes
        if (runtime_state.gene_to_ind is None or x not in runtime_state.gene_to_ind)
        and (runtime_state.gene_missing_to_ind is None or x not in runtime_state.gene_missing_to_ind)
    ]
    gene_to_ind = construct_map_to_ind(genes)
    index_map = {i: gene_to_ind[old_genes[i]] for i in range(len(old_genes))}
    cur_X = sparse.csc_matrix(
        (cur_X.data, [index_map[x] for x in cur_X.indices], cur_X.indptr),
        shape=(len(genes), cur_X.shape[1]),
    )
    return (cur_X, genes)


def ensure_gene_universe_for_x(
    runtime_state,
    X_ins,
    is_dense,
    add_all_genes,
    only_ids,
    only_inc_genes,
    fraction_inc_genes,
    *,
    open_gz_fn,
    remove_tag_from_input_fn,
    log_fn,
    debug_level,
    bail_fn,
):
    if runtime_state.genes is None or add_all_genes:
        if runtime_state.genes is None:
            log_fn("No genes initialized before reading X: constructing gene list from union of all files", debug_level)

        all_genes = []
        gene_counts = {}
        num_gene_sets = 0
        for i in range(len(X_ins)):
            X_in = X_ins[i]
            (X_in, _tag) = remove_tag_from_input_fn(X_in)

            if is_dense[i]:
                with open_gz_fn(X_in) as gene_sets_fh:
                    num_in_file = None
                    for line in gene_sets_fh:
                        line = line.strip("\n")
                        cols = line.split()
                        if num_in_file is None:
                            num_in_file = len(cols) - 1
                            num_gene_sets += num_in_file
                        elif len(cols) - 1 != num_in_file:
                            bail_fn("Not a square matrix!")

                        if len(cols) > 0:
                            all_genes += cols[0]
                        if cols[0] not in gene_counts:
                            gene_counts[cols[0]] = 0
                        gene_counts[cols[0]] += num_in_file
            else:
                with open_gz_fn(X_in) as gene_sets_fh:
                    it = 0
                    for line in gene_sets_fh:
                        line = line.strip("\n")
                        cols = line.split()
                        if len(cols) < 2:
                            continue

                        cur_genes = set(cols[1:])

                        if only_ids is not None and cols[0] not in only_ids:
                            continue

                        if ":" in line:
                            cur_genes = [gene.split(":")[0] for gene in cur_genes]
                        if runtime_state.gene_label_map is not None:
                            cur_genes = set(map(lambda x: runtime_state.gene_label_map[x] if x in runtime_state.gene_label_map else x, cur_genes))

                        if not add_all_genes and only_inc_genes is not None:
                            fraction_match = len(only_inc_genes.intersection(cur_genes)) / float(len(only_inc_genes))
                            if fraction_match < (fraction_inc_genes if fraction_inc_genes is not None else 1e-5):
                                continue

                        all_genes += cur_genes
                        for gene in cur_genes:
                            if gene not in gene_counts:
                                gene_counts[gene] = 0
                            gene_counts[gene] += 1

                        num_gene_sets += 1
                        it += 1
                        if it % 1000 == 0:
                            all_genes = list(set(all_genes))

            all_genes = list(set(all_genes))

        if runtime_state.genes is not None:
            add_genes = [x for x in all_genes if x not in runtime_state.gene_to_ind]
            log_fn("Adding an additional %d genes from gene sets not in input Y values" % len(add_genes), debug_level)
            all_genes = runtime_state.genes + add_genes
            new_Y = runtime_state.Y
            if new_Y is not None:
                assert len(new_Y) == len(runtime_state.genes)
                new_Y = np.append(new_Y, np.zeros(len(add_genes)))
            new_Y_for_regression = runtime_state.Y_for_regression
            if new_Y_for_regression is not None:
                assert len(new_Y_for_regression) == len(runtime_state.genes)
                new_Y_for_regression = np.append(new_Y_for_regression, np.zeros(len(add_genes)))
            new_Y_exomes = runtime_state.Y_exomes
            if new_Y_exomes is not None:
                assert len(new_Y_exomes) == len(runtime_state.genes)
                new_Y_exomes = np.append(new_Y_exomes, np.zeros(len(add_genes)))
            new_Y_positive_controls = runtime_state.Y_positive_controls
            if new_Y_positive_controls is not None:
                assert len(new_Y_positive_controls) == len(runtime_state.genes)
                new_Y_positive_controls = np.append(new_Y_positive_controls, np.zeros(len(add_genes)))

            new_Y_case_counts = runtime_state.Y_case_counts
            if new_Y_case_counts is not None:
                assert len(new_Y_case_counts) == len(runtime_state.genes)
                new_Y_case_counts = np.append(new_Y_case_counts, np.zeros(len(add_genes)))

            runtime_state._set_Y(new_Y, new_Y_for_regression, new_Y_exomes, new_Y_positive_controls, new_Y_case_counts)

        runtime_state._set_X(runtime_state.X_orig, list(all_genes), runtime_state.gene_sets, skip_N=False)
