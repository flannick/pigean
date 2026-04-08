from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sparse

from pegs_shared.gene_io import load_gene_ids_from_file
from pegs_shared.io_common import (
    clean_chrom_name,
    construct_map_to_ind,
    open_text_with_retry,
    parse_gene_map_file,
    read_loc_file_with_gene_map,
    resolve_column_index,
)


def set_const_Y(runtime_state, value):
    const_Y = np.full(len(runtime_state.genes), value)
    runtime_state._set_Y(const_Y, const_Y, None, None, None, skip_V=True, skip_scale_factors=True)


def init_gene_locs(runtime_state, gene_loc_file, *, warn_fn, bail_fn, log_fn):
    log_fn("Reading --gene-loc-file %s" % gene_loc_file)
    (
        runtime_state.gene_chrom_name_pos,
        runtime_state.gene_to_chrom,
        runtime_state.gene_to_pos,
    ) = read_loc_file_with_gene_map(
        gene_loc_file,
        gene_label_map=runtime_state.gene_label_map,
        clean_chrom_fn=clean_chrom_name,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
    )


def read_gene_map(
    runtime_state,
    gene_map_in,
    gene_map_orig_gene_col=1,
    gene_map_new_gene_col=2,
    allow_multi=False,
    *,
    bail_fn,
):
    runtime_state.gene_label_map = parse_gene_map_file(
        gene_map_in,
        gene_map_orig_gene_col=gene_map_orig_gene_col,
        gene_map_new_gene_col=gene_map_new_gene_col,
        allow_multi=allow_multi,
        bail_fn=bail_fn,
    )


def apply_gene_covariates_and_correct_huge(
    runtime_state,
    gene_covs_in=None,
    *,
    log_fn,
    trace_level,
    bail_fn,
    **kwargs,
):
    maybe_append_input_gene_covariates(runtime_state, gene_covs_in=gene_covs_in, **kwargs)

    if runtime_state.gene_covariates is None:
        return

    prepare_gene_covariate_regression_state(
        runtime_state,
        log_fn=log_fn,
        trace_level=trace_level,
        bail_fn=bail_fn,
    )
    apply_huge_correction_with_covariates(runtime_state)


def apply_huge_correction_with_covariates(runtime_state):
    # Recompute corrected HuGE scores.
    Y_for_regression = runtime_state.Y_for_regression
    if runtime_state.Y_for_regression is not None:
        (Y_for_regression, _, _) = runtime_state._correct_huge(
            runtime_state.Y_for_regression,
            runtime_state.gene_covariates,
            runtime_state.gene_covariates_mask,
            runtime_state.gene_covariates_mat_inv,
            runtime_state.gene_covariate_names,
            runtime_state.gene_covariate_intercept_index,
        )

    (Y, runtime_state.Y_uncorrected, _) = runtime_state._correct_huge(
        runtime_state.Y,
        runtime_state.gene_covariates,
        runtime_state.gene_covariates_mask,
        runtime_state.gene_covariates_mat_inv,
        runtime_state.gene_covariate_names,
        runtime_state.gene_covariate_intercept_index,
    )

    runtime_state._set_Y(Y, Y_for_regression, runtime_state.Y_exomes, runtime_state.Y_positive_controls, runtime_state.Y_case_counts)
    runtime_state.gene_covariate_adjustments = runtime_state.Y_for_regression - runtime_state.Y_uncorrected

    if runtime_state.gene_to_gwas_huge_score is not None:
        Y_huge = np.zeros(len(runtime_state.Y_for_regression))
        assert len(Y_huge) == len(runtime_state.genes)
        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in runtime_state.gene_to_gwas_huge_score:
                Y_huge[i] = runtime_state.gene_to_gwas_huge_score[runtime_state.genes[i]]

        (Y_huge, _, _) = runtime_state._correct_huge(
            Y_huge,
            runtime_state.gene_covariates,
            runtime_state.gene_covariates_mask,
            runtime_state.gene_covariates_mat_inv,
            runtime_state.gene_covariate_names,
            runtime_state.gene_covariate_intercept_index,
        )

        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in runtime_state.gene_to_gwas_huge_score:
                runtime_state.gene_to_gwas_huge_score[runtime_state.genes[i]] = Y_huge[i]

        runtime_state.combine_huge_scores()


def prepare_gene_covariate_regression_state(runtime_state, *, log_fn, trace_level, bail_fn):
    # Remove degenerate / highly collinear columns before linear correction.
    constant_features = np.isclose(np.var(runtime_state.gene_covariates, axis=0), 0)
    if np.sum(constant_features) > 0:
        runtime_state.gene_covariates = runtime_state.gene_covariates[:, ~constant_features]
        runtime_state.gene_covariate_names = [runtime_state.gene_covariate_names[i] for i in np.where(~constant_features)[0]]
        runtime_state.gene_covariate_directions = np.array([runtime_state.gene_covariate_directions[i] for i in np.where(~constant_features)[0]])

    prune_threshold = 0.95
    cor_mat = np.abs(np.corrcoef(runtime_state.gene_covariates.T))
    np.fill_diagonal(cor_mat, 0)

    while True:
        if np.max(cor_mat) < prune_threshold:
            try:
                np.linalg.inv(runtime_state.gene_covariates.T.dot(runtime_state.gene_covariates))
                break
            except np.linalg.LinAlgError:
                pass

        max_index = np.unravel_index(np.argmax(cor_mat), cor_mat.shape)
        if np.max(max_index) == runtime_state.gene_covariate_intercept_index:
            max_index = np.min(max_index)
        else:
            max_index = np.max(max_index)
        log_fn("Removing feature %s" % runtime_state.gene_covariate_names[max_index], trace_level)
        runtime_state.gene_covariates = np.delete(runtime_state.gene_covariates, max_index, axis=1)
        del runtime_state.gene_covariate_names[max_index]
        runtime_state.gene_covariate_directions = np.delete(runtime_state.gene_covariate_directions, max_index)

        cor_mat = np.delete(np.delete(cor_mat, max_index, axis=1), max_index, axis=0)
        if len(runtime_state.gene_covariates) == 0:
            bail_fn("Error: something went wrong with matrix inversion. Still couldn't invert after removing all but one column")

    # Ensure a single intercept is present.
    runtime_state.gene_covariate_intercept_index = np.where(np.isclose(np.var(runtime_state.gene_covariates, axis=0), 0))[0]
    if len(runtime_state.gene_covariate_intercept_index) == 0:
        runtime_state.gene_covariates = np.hstack((runtime_state.gene_covariates, np.ones(runtime_state.gene_covariates.shape[0])[:, np.newaxis]))
        runtime_state.gene_covariate_names.append("intercept")
        runtime_state.gene_covariate_directions = np.append(runtime_state.gene_covariate_directions, 0)
        runtime_state.gene_covariate_intercept_index = len(runtime_state.gene_covariate_names) - 1
    else:
        runtime_state.gene_covariate_intercept_index = runtime_state.gene_covariate_intercept_index[0]

    covariate_means = np.mean(runtime_state.gene_covariates, axis=0)
    covariate_sds = np.std(runtime_state.gene_covariates, axis=0)
    covariate_sds[covariate_sds == 0] = 1

    runtime_state.gene_covariates_mask = np.all(runtime_state.gene_covariates < covariate_means + 5 * covariate_sds, axis=1)
    runtime_state.gene_covariates_mat_inv = np.linalg.inv(
        runtime_state.gene_covariates[runtime_state.gene_covariates_mask, :].T.dot(
            runtime_state.gene_covariates[runtime_state.gene_covariates_mask, :]
        )
    )
    gene_covariate_sds = np.std(runtime_state.gene_covariates, axis=0)
    gene_covariate_sds[gene_covariate_sds == 0] = 1
    runtime_state.gene_covariate_zs = (runtime_state.gene_covariates - np.mean(runtime_state.gene_covariates, axis=0)) / gene_covariate_sds


def maybe_append_input_gene_covariates(runtime_state, gene_covs_in=None, **kwargs):
    # Load optional covariates and append to any existing covariate matrix.
    if gene_covs_in is None:
        return

    (cov_names, gene_covs, _, _) = runtime_state.read_gene_covs(gene_covs_in, **kwargs)
    cov_dirs = np.array([0] * len(cov_names))

    col_means = np.nanmean(gene_covs, axis=0)
    nan_indices = np.where(np.isnan(gene_covs))
    gene_covs[nan_indices] = np.take(col_means, nan_indices[1])

    if runtime_state.gene_covariates is not None:
        assert gene_covs.shape[0] == runtime_state.gene_covariates.shape[0]
        runtime_state.gene_covariates = np.hstack((runtime_state.gene_covariates, gene_covs))
        runtime_state.gene_covariate_names = runtime_state.gene_covariate_names + cov_names
        runtime_state.gene_covariate_directions = np.append(runtime_state.gene_covariate_directions, cov_dirs)
    else:
        runtime_state.gene_covariates = gene_covs
        runtime_state.gene_covariate_names = cov_names
        runtime_state.gene_covariate_directions = cov_dirs


def read_y_from_contract(runtime_state, y_read_contract, *, read_y_fn, bail_fn):
    if y_read_contract is None:
        bail_fn("Bug in code: y_read_contract must be non-None")
    if not hasattr(y_read_contract, "to_read_kwargs"):
        bail_fn("Bug in code: y_read_contract must provide to_read_kwargs()")
    return read_y_fn(runtime_state, **y_read_contract.to_read_kwargs())


def read_y_pipeline(
    runtime_state,
    gwas_in=None,
    huge_statistics_in=None,
    huge_statistics_out=None,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    ctrl_counts_in=None,
    gene_bfs_in=None,
    gene_universe_in=None,
    gene_universe_id_col=None,
    gene_universe_has_header=True,
    gene_universe_from_y=False,
    gene_universe_from_x=False,
    gene_loc_file=None,
    gene_covs_in=None,
    hold_out_chrom=None,
    *,
    warn_fn,
    bail_fn,
    log_fn,
    trace_level,
    apply_gene_covariates_and_correct_huge_fn,
    **kwargs,
):
    (gene_universe_mode, gene_universe_genes) = resolve_requested_gene_universe(
        runtime_state,
        gwas_in=gwas_in,
        huge_statistics_in=huge_statistics_in,
        exomes_in=exomes_in,
        positive_controls_in=positive_controls_in,
        positive_controls_list=positive_controls_list,
        case_counts_in=case_counts_in,
        gene_bfs_in=gene_bfs_in,
        gene_universe_in=gene_universe_in,
        gene_universe_id_col=gene_universe_id_col,
        gene_universe_has_header=gene_universe_has_header,
        gene_universe_from_y=gene_universe_from_y,
        gene_universe_from_x=gene_universe_from_x,
        log_fn=log_fn,
        bail_fn=bail_fn,
        warn_fn=warn_fn,
    )
    initialize_explicit_gene_universe_if_needed(
        runtime_state,
        gene_universe_mode=gene_universe_mode,
        gene_universe_genes=gene_universe_genes,
        log_fn=log_fn,
    )

    (
        Y1_exomes,
        Y1_positive_controls,
        Y1_case_counts,
        extra_genes_all,
        extra_Y_exomes,
        extra_Y_positive_controls,
        extra_Y_case_counts,
        missing_value_exomes,
        missing_value_positive_controls,
        missing_value_case_counts,
    ) = read_and_align_auxiliary_y_components(
        runtime_state,
        exomes_in=exomes_in,
        positive_controls_in=positive_controls_in,
        positive_controls_list=positive_controls_list,
        case_counts_in=case_counts_in,
        ctrl_counts_in=ctrl_counts_in,
        gene_loc_file=gene_loc_file,
        hold_out_chrom=hold_out_chrom,
        bail_fn=bail_fn,
        **kwargs,
    )

    (
        Y1,
        extra_genes,
        extra_Y,
        Y1_for_regression,
        extra_Y_for_regression,
        missing_value,
        gene_combined_map,
        gene_prior_map,
    ) = read_primary_y_source(
        runtime_state,
        gwas_in=gwas_in,
        huge_statistics_in=huge_statistics_in,
        huge_statistics_out=huge_statistics_out,
        exomes_in=exomes_in,
        positive_controls_in=positive_controls_in,
        positive_controls_list=positive_controls_list,
        case_counts_in=case_counts_in,
        gene_bfs_in=gene_bfs_in,
        hold_out_chrom=hold_out_chrom,
        gene_loc_file=gene_loc_file,
        Y1_exomes=Y1_exomes,
        Y1_positive_controls=Y1_positive_controls,
        Y1_case_counts=Y1_case_counts,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
        **kwargs,
    )

    (
        Y,
        Y_for_regression,
        Y_exomes,
        Y_positive_controls,
        Y_case_counts,
        extra_genes,
        extra_Y,
        extra_Y_for_regression,
        extra_Y_exomes,
        extra_Y_positive_controls,
        extra_Y_case_counts,
    ) = materialize_y_on_gene_universe(
        runtime_state,
        Y1=Y1,
        Y1_for_regression=Y1_for_regression,
        Y1_exomes=Y1_exomes,
        Y1_positive_controls=Y1_positive_controls,
        Y1_case_counts=Y1_case_counts,
        extra_genes=extra_genes,
        extra_Y=extra_Y,
        extra_Y_for_regression=extra_Y_for_regression,
        extra_genes_all=extra_genes_all,
        extra_Y_exomes=extra_Y_exomes,
        extra_Y_positive_controls=extra_Y_positive_controls,
        extra_Y_case_counts=extra_Y_case_counts,
        missing_value=missing_value,
        missing_value_exomes=missing_value_exomes,
        missing_value_positive_controls=missing_value_positive_controls,
        missing_value_case_counts=missing_value_case_counts,
        extend_with_extra_genes=(gene_universe_mode != "file"),
        warn_fn=warn_fn,
    )

    finalize_y_vectors_and_expand_x(
        runtime_state,
        Y=Y,
        Y_for_regression=Y_for_regression,
        Y_exomes=Y_exomes,
        Y_positive_controls=Y_positive_controls,
        Y_case_counts=Y_case_counts,
        extra_genes=extra_genes,
        extra_Y=extra_Y,
        extra_Y_for_regression=extra_Y_for_regression,
        extra_Y_exomes=extra_Y_exomes,
        extra_Y_positive_controls=extra_Y_positive_controls,
        extra_Y_case_counts=extra_Y_case_counts,
        log_fn=log_fn,
        trace_level=trace_level,
    )

    apply_gene_level_maps_after_read_y(
        runtime_state,
        gene_combined_map=gene_combined_map,
        gene_prior_map=gene_prior_map,
    )
    apply_gene_covariates_and_correct_huge_fn(runtime_state, gene_covs_in=gene_covs_in, **kwargs)


def resolve_requested_gene_universe(
    runtime_state,
    *,
    gwas_in,
    huge_statistics_in,
    exomes_in,
    positive_controls_in,
    positive_controls_list,
    case_counts_in,
    gene_bfs_in,
    gene_universe_in,
    gene_universe_id_col,
    gene_universe_has_header,
    gene_universe_from_y,
    gene_universe_from_x,
    log_fn,
    bail_fn,
    warn_fn,
):
    num_selected = int(gene_universe_in is not None) + int(bool(gene_universe_from_y)) + int(bool(gene_universe_from_x))
    if num_selected > 1:
        bail_fn("Specify at most one of --gene-universe-in, --gene-universe-from-y, or --gene-universe-from-x")

    if gene_universe_in is not None:
        universe_genes = load_gene_ids_from_file(
            gene_universe_in,
            gene_ids_id_col=gene_universe_id_col,
            gene_ids_has_header=gene_universe_has_header,
            gene_label_map=runtime_state.gene_label_map,
            open_text_fn=open_text_with_retry,
            get_col_fn=resolve_column_index,
            log_fn=log_fn,
            warn_fn=warn_fn,
            bail_fn=bail_fn,
        )
        log_fn("Using explicit gene universe from --gene-universe-in with %d genes" % len(universe_genes))
        return ("file", universe_genes)

    if gene_universe_from_y:
        log_fn("Using only genes present in the input Y values as the gene universe")
        return ("y", None)

    if gene_universe_from_x:
        log_fn("Using the union of genes across input gene sets as the gene universe")
        return ("x", None)

    if gwas_in is not None or huge_statistics_in is not None:
        log_fn("No explicit gene universe provided; defaulting to the HuGE/GWAS gene list used during gene-score construction")
        return ("y", None)

    if gene_bfs_in is not None or exomes_in is not None or positive_controls_in is not None or positive_controls_list is not None or case_counts_in is not None:
        bail_fn(
            "This input Y mode requires an explicit gene universe. Provide --gene-universe-in, "
            "or opt into --gene-universe-from-y or --gene-universe-from-x."
        )

    return ("y", None)


def initialize_explicit_gene_universe_if_needed(runtime_state, *, gene_universe_mode, gene_universe_genes, log_fn):
    if gene_universe_mode != "file":
        return
    requested_genes = list(gene_universe_genes)
    if runtime_state.genes is not None and list(runtime_state.genes) == requested_genes:
        return
    if runtime_state.genes is None:
        log_fn("Initializing analysis gene universe from explicit gene-universe file")
    else:
        log_fn("Reinitializing analysis gene universe from explicit gene-universe file after X load")
    runtime_state._set_X(runtime_state.X_orig, requested_genes, runtime_state.gene_sets, skip_N=False)


def apply_gene_level_maps_after_read_y(runtime_state, gene_combined_map=None, gene_prior_map=None):
    if gene_combined_map is not None:
        runtime_state.combined_prior_Ys = copy.copy(runtime_state.Y)
        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in gene_combined_map:
                runtime_state.combined_prior_Ys[i] = gene_combined_map[runtime_state.genes[i]]

    if gene_prior_map is not None:
        runtime_state.priors = np.zeros(len(runtime_state.genes))
        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in gene_prior_map:
                runtime_state.priors[i] = gene_prior_map[runtime_state.genes[i]]


def finalize_y_vectors_and_expand_x(
    runtime_state,
    Y,
    Y_for_regression,
    Y_exomes,
    Y_positive_controls,
    Y_case_counts,
    extra_genes,
    extra_Y,
    extra_Y_for_regression,
    extra_Y_exomes,
    extra_Y_positive_controls,
    extra_Y_case_counts,
    *,
    log_fn,
    trace_level,
):
    if len(extra_Y) > 0:
        Y = np.concatenate((Y, extra_Y))
        Y_for_regression = np.concatenate((Y_for_regression, extra_Y_for_regression))
        Y_exomes = np.concatenate((Y_exomes, extra_Y_exomes))
        Y_positive_controls = np.concatenate((Y_positive_controls, extra_Y_positive_controls))
        Y_case_counts = np.concatenate((Y_case_counts, extra_Y_case_counts))

    if runtime_state.X_orig is not None:
        log_fn("Expanding matrix", trace_level)
        runtime_state._set_X(
            sparse.csc_matrix(
                (runtime_state.X_orig.data, runtime_state.X_orig.indices, runtime_state.X_orig.indptr),
                shape=(runtime_state.X_orig.shape[0] + len(extra_Y), runtime_state.X_orig.shape[1]),
            ),
            runtime_state.genes,
            runtime_state.gene_sets,
            skip_V=True,
            skip_scale_factors=True,
            skip_N=False,
        )

    if runtime_state.genes is not None:
        runtime_state._set_X(runtime_state.X_orig, runtime_state.genes + extra_genes, runtime_state.gene_sets, skip_N=False)

    runtime_state._set_Y(
        Y,
        Y_for_regression,
        Y_exomes,
        Y_positive_controls,
        Y_case_counts,
        skip_V=True,
        skip_scale_factors=True,
    )


def read_and_align_auxiliary_y_components(
    runtime_state,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    ctrl_counts_in=None,
    gene_loc_file=None,
    hold_out_chrom=None,
    *,
    bail_fn,
    **kwargs,
):
    Y1_exomes = np.array([])
    extra_genes_all = []
    extra_Y_exomes = []

    if exomes_in is not None:
        (Y1_exomes, extra_genes_exomes, extra_Y_exomes) = runtime_state.calculate_huge_scores_exomes(
            exomes_in,
            hold_out_chrom=hold_out_chrom,
            gene_loc_file=gene_loc_file,
            **kwargs,
        )
        if runtime_state.genes is None:
            runtime_state._set_X(runtime_state.X_orig, extra_genes_exomes, runtime_state.gene_sets, skip_N=True, skip_V=True)
            runtime_state.Y_exomes = extra_Y_exomes
            Y1_exomes = extra_Y_exomes
            extra_genes_all = []
            extra_Y_exomes = np.array([])
        else:
            extra_genes_all = extra_genes_exomes

    missing_value_exomes = 0
    missing_value_positive_controls = 0
    missing_value_case_counts = 0

    Y1_positive_controls = np.array([])
    extra_Y_positive_controls = []

    if positive_controls_in is not None or positive_controls_list is not None:
        (Y1_positive_controls, extra_genes_positive_controls, extra_Y_positive_controls) = runtime_state.read_positive_controls(
            positive_controls_in,
            positive_controls_list=positive_controls_list,
            hold_out_chrom=hold_out_chrom,
            gene_loc_file=gene_loc_file,
            **kwargs,
        )
        if runtime_state.genes is None:
            assert len(Y1_exomes) == 0
            runtime_state._set_X(runtime_state.X_orig, extra_genes_positive_controls, runtime_state.gene_sets, skip_N=True, skip_V=True)
            runtime_state.Y_positive_controls = extra_Y_positive_controls
            Y1_positive_controls = extra_Y_positive_controls
            extra_genes_positive_controls = []
            extra_genes_all = extra_genes_positive_controls
            extra_Y_positive_controls = np.array([])
            Y1_exomes = np.zeros(len(Y1_positive_controls))
        else:
            extra_genes_all, aligned_existing_values, extra_Y_positive_controls = align_extra_genes_with_new_source(
                existing_extra_genes=extra_genes_all,
                existing_extra_values=[extra_Y_exomes],
                new_source_genes=extra_genes_positive_controls,
                new_source_values=extra_Y_positive_controls,
                existing_missing_values=[missing_value_exomes],
                new_source_missing_value=missing_value_positive_controls,
            )
            extra_Y_exomes = aligned_existing_values[0]
    else:
        Y1_positive_controls = np.zeros(len(Y1_exomes))
        extra_Y_positive_controls = np.zeros(len(extra_genes_all))

    if runtime_state.genes is not None and len(Y1_exomes) == 0:
        Y1_exomes = np.zeros(len(runtime_state.genes))
    if runtime_state.genes is not None and len(Y1_positive_controls) == 0:
        Y1_positive_controls = np.zeros(len(runtime_state.genes))

    assert len(extra_Y_exomes) == len(extra_genes_all)
    assert len(extra_Y_exomes) == len(extra_Y_positive_controls)
    assert len(Y1_exomes) == len(Y1_positive_controls)

    Y1_case_counts = np.array([])
    extra_Y_case_counts = []

    if case_counts_in is not None or ctrl_counts_in is not None:
        if case_counts_in is None or ctrl_counts_in is None:
            bail_fn("If specify one of --case-counts-in or --ctrl-counts-in must specify both of them")

        (Y1_case_counts, extra_genes_case_counts, extra_Y_case_counts) = runtime_state.read_count_file(
            case_counts_in,
            ctrl_counts_in,
            hold_out_chrom=hold_out_chrom,
            gene_loc_file=gene_loc_file,
            **kwargs,
        )
        if runtime_state.genes is None:
            assert len(Y1_exomes) == 0
            assert len(Y1_positive_controls) == 0
            runtime_state._set_X(runtime_state.X_orig, extra_genes_case_counts, runtime_state.gene_sets, skip_N=True, skip_V=True)
            runtime_state.Y_case_counts = extra_Y_case_counts
            Y1_case_counts = extra_Y_case_counts
            extra_genes_case_counts = []
            extra_Y_case_counts = np.array([])
            extra_genes_all = extra_genes_case_counts
            Y1_exomes = np.zeros(len(Y1_case_counts))
            Y1_positive_controls = np.zeros(len(Y1_case_counts))
        else:
            extra_genes_all, aligned_existing_values, extra_Y_case_counts = align_extra_genes_with_new_source(
                existing_extra_genes=extra_genes_all,
                existing_extra_values=[extra_Y_exomes, extra_Y_positive_controls],
                new_source_genes=extra_genes_case_counts,
                new_source_values=extra_Y_case_counts,
                existing_missing_values=[missing_value_exomes, missing_value_positive_controls],
                new_source_missing_value=missing_value_case_counts,
            )
            extra_Y_exomes = aligned_existing_values[0]
            extra_Y_positive_controls = aligned_existing_values[1]
    else:
        Y1_case_counts = np.zeros(len(Y1_exomes))
        extra_Y_case_counts = np.zeros(len(extra_genes_all))

    if runtime_state.genes is not None and len(Y1_exomes) == 0:
        Y1_exomes = np.zeros(len(runtime_state.genes))
    if runtime_state.genes is not None and len(Y1_positive_controls) == 0:
        Y1_positive_controls = np.zeros(len(runtime_state.genes))
    if runtime_state.genes is not None and len(Y1_case_counts) == 0:
        Y1_case_counts = np.zeros(len(runtime_state.genes))

    assert len(extra_Y_exomes) == len(extra_genes_all)
    assert len(extra_Y_exomes) == len(extra_Y_positive_controls)
    assert len(extra_Y_exomes) == len(extra_Y_case_counts)
    assert len(Y1_exomes) == len(Y1_positive_controls)
    assert len(Y1_exomes) == len(Y1_case_counts)

    return (
        Y1_exomes,
        Y1_positive_controls,
        Y1_case_counts,
        extra_genes_all,
        extra_Y_exomes,
        extra_Y_positive_controls,
        extra_Y_case_counts,
        missing_value_exomes,
        missing_value_positive_controls,
        missing_value_case_counts,
    )


def read_primary_y_source(
    runtime_state,
    gwas_in=None,
    huge_statistics_in=None,
    huge_statistics_out=None,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    gene_bfs_in=None,
    hold_out_chrom=None,
    gene_loc_file=None,
    Y1_exomes=None,
    Y1_positive_controls=None,
    Y1_case_counts=None,
    *,
    warn_fn,
    bail_fn,
    **kwargs,
):
    missing_value = None
    gene_combined_map = None
    gene_prior_map = None

    huge_or_gwas_source = read_primary_huge_or_gwas_source(
        runtime_state,
        huge_statistics_in=huge_statistics_in,
        gwas_in=gwas_in,
        huge_statistics_out=huge_statistics_out,
        gene_loc_file=gene_loc_file,
        hold_out_chrom=hold_out_chrom,
        warn_fn=warn_fn,
        **kwargs,
    )

    if huge_or_gwas_source is not None:
        (
            Y1,
            extra_genes,
            extra_Y,
            Y1_for_regression,
            extra_Y_for_regression,
            missing_value,
        ) = huge_or_gwas_source
    else:
        (
            Y1,
            extra_genes,
            extra_Y,
            Y1_for_regression,
            extra_Y_for_regression,
            gene_combined_map,
            gene_prior_map,
        ) = read_primary_non_huge_source(
            runtime_state,
            gene_bfs_in=gene_bfs_in,
            exomes_in=exomes_in,
            positive_controls_in=positive_controls_in,
            positive_controls_list=positive_controls_list,
            case_counts_in=case_counts_in,
            Y1_exomes=Y1_exomes,
            Y1_positive_controls=Y1_positive_controls,
            Y1_case_counts=Y1_case_counts,
            hold_out_chrom=hold_out_chrom,
            gene_loc_file=gene_loc_file,
            bail_fn=bail_fn,
            **kwargs,
        )

    return (
        Y1,
        extra_genes,
        extra_Y,
        Y1_for_regression,
        extra_Y_for_regression,
        missing_value,
        gene_combined_map,
        gene_prior_map,
    )


def read_primary_huge_or_gwas_source(
    runtime_state,
    huge_statistics_in=None,
    gwas_in=None,
    huge_statistics_out=None,
    gene_loc_file=None,
    hold_out_chrom=None,
    *,
    warn_fn,
    **kwargs,
):
    if huge_statistics_in is not None:
        if gwas_in is not None:
            warn_fn("Both --gwas-in and --huge-statistics-in were passed; using --huge-statistics-in")
        (Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression) = runtime_state.read_huge_statistics(huge_statistics_in)
        return (Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression, 0)

    if gwas_in is None:
        return None

    (Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression) = runtime_state.calculate_huge_scores_gwas(
        gwas_in,
        gene_loc_file=gene_loc_file,
        hold_out_chrom=hold_out_chrom,
        **kwargs,
    )
    if huge_statistics_out is not None:
        runtime_state.write_huge_statistics(huge_statistics_out, Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression)
    return (Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression, 0)


def read_primary_non_huge_source(
    runtime_state,
    gene_bfs_in=None,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    Y1_exomes=None,
    Y1_positive_controls=None,
    Y1_case_counts=None,
    hold_out_chrom=None,
    gene_loc_file=None,
    *,
    bail_fn,
    **kwargs,
):
    runtime_state.huge_signal_bfs = None
    runtime_state.huge_signal_bfs_for_regression = None

    gene_combined_map = None
    gene_prior_map = None
    if gene_bfs_in is not None:
        (Y1, extra_genes, extra_Y, gene_combined_map, gene_prior_map) = runtime_state.read_gene_bfs(
            gene_bfs_in,
            **kwargs,
        )
    elif exomes_in is not None:
        (Y1, extra_genes, extra_Y) = (np.zeros(Y1_exomes.shape), [], [])
    elif positive_controls_in is not None or positive_controls_list is not None:
        (Y1, extra_genes, extra_Y) = (np.zeros(Y1_positive_controls.shape), [], [])
    elif case_counts_in is not None:
        (Y1, extra_genes, extra_Y) = (np.zeros(Y1_case_counts.shape), [], [])
    else:
        bail_fn("Need to specify either gene_bfs_in or exomes_in or positive_controls_in or case_counts_in")

    (Y1, extra_genes, extra_Y) = apply_hold_out_chrom_to_y(
        runtime_state,
        Y1,
        extra_genes,
        extra_Y,
        hold_out_chrom=hold_out_chrom,
        gene_loc_file=gene_loc_file,
        bail_fn=bail_fn,
    )
    Y1_for_regression = copy.copy(Y1)
    extra_Y_for_regression = copy.copy(extra_Y)

    return (
        Y1,
        extra_genes,
        extra_Y,
        Y1_for_regression,
        extra_Y_for_regression,
        gene_combined_map,
        gene_prior_map,
    )


def materialize_y_on_gene_universe(
    runtime_state,
    Y1,
    Y1_for_regression,
    Y1_exomes,
    Y1_positive_controls,
    Y1_case_counts,
    extra_genes,
    extra_Y,
    extra_Y_for_regression,
    extra_genes_all,
    extra_Y_exomes,
    extra_Y_positive_controls,
    extra_Y_case_counts,
    missing_value,
    missing_value_exomes,
    missing_value_positive_controls,
    missing_value_case_counts,
    extend_with_extra_genes=True,
    warn_fn=None,
):
    if missing_value is None:
        if len(Y1) > 0:
            missing_value = np.nanmean(Y1)
        else:
            missing_value = 0

    if runtime_state.genes is None:
        assert len(Y1) == 0
        assert len(Y1_exomes) == 0
        assert len(Y1_positive_controls) == 0
        assert len(Y1_case_counts) == 0
        return initialize_y_from_new_gene_universe(
            runtime_state,
            extra_genes=extra_genes,
            extra_Y=extra_Y,
            extra_Y_for_regression=extra_Y_for_regression,
            extra_genes_all=extra_genes_all,
            extra_Y_exomes=extra_Y_exomes,
            extra_Y_positive_controls=extra_Y_positive_controls,
            extra_Y_case_counts=extra_Y_case_counts,
            missing_value=missing_value,
            missing_value_exomes=missing_value_exomes,
            missing_value_positive_controls=missing_value_positive_controls,
            missing_value_case_counts=missing_value_case_counts,
        )

    return merge_y_into_existing_gene_universe(
        runtime_state,
        Y1=Y1,
        Y1_for_regression=Y1_for_regression,
        Y1_exomes=Y1_exomes,
        Y1_positive_controls=Y1_positive_controls,
        Y1_case_counts=Y1_case_counts,
        extra_genes=extra_genes,
        extra_Y=extra_Y,
        extra_Y_for_regression=extra_Y_for_regression,
        extra_genes_all=extra_genes_all,
        extra_Y_exomes=extra_Y_exomes,
        extra_Y_positive_controls=extra_Y_positive_controls,
        extra_Y_case_counts=extra_Y_case_counts,
        missing_value=missing_value,
        missing_value_exomes=missing_value_exomes,
        missing_value_positive_controls=missing_value_positive_controls,
        missing_value_case_counts=missing_value_case_counts,
        extend_with_extra_genes=extend_with_extra_genes,
        warn_fn=warn_fn,
    )


def initialize_y_from_new_gene_universe(
    runtime_state,
    extra_genes,
    extra_Y,
    extra_Y_for_regression,
    extra_genes_all,
    extra_Y_exomes,
    extra_Y_positive_controls,
    extra_Y_case_counts,
    missing_value,
    missing_value_exomes,
    missing_value_positive_controls,
    missing_value_case_counts,
):
    genes_union = []
    genes_seen = set()
    for gene in extra_genes + extra_genes_all:
        if gene not in genes_seen:
            genes_union.append(gene)
            genes_seen.add(gene)

    runtime_state._set_X(runtime_state.X_orig, genes_union, runtime_state.gene_sets, skip_N=False)

    Y = np.full(len(runtime_state.genes), missing_value, dtype=float)
    Y_for_regression = np.full(len(runtime_state.genes), missing_value, dtype=float)
    Y_exomes = np.full(len(runtime_state.genes), missing_value_exomes, dtype=float)
    Y_positive_controls = np.full(len(runtime_state.genes), missing_value_positive_controls, dtype=float)
    Y_case_counts = np.full(len(runtime_state.genes), missing_value_case_counts, dtype=float)

    for i in range(len(extra_genes)):
        Y[runtime_state.gene_to_ind[extra_genes[i]]] = extra_Y[i]
        Y_for_regression[runtime_state.gene_to_ind[extra_genes[i]]] = extra_Y_for_regression[i]

    for i in range(len(extra_genes_all)):
        Y_exomes[runtime_state.gene_to_ind[extra_genes_all[i]]] = extra_Y_exomes[i]
        Y_positive_controls[runtime_state.gene_to_ind[extra_genes_all[i]]] = extra_Y_positive_controls[i]
        Y_case_counts[runtime_state.gene_to_ind[extra_genes_all[i]]] = extra_Y_case_counts[i]

    Y += Y_exomes
    Y += Y_positive_controls
    Y += Y_case_counts

    Y_for_regression += Y_exomes
    Y_for_regression += Y_positive_controls
    Y_for_regression += Y_case_counts

    if runtime_state.huge_signal_bfs is not None or runtime_state.gene_covariates is not None:
        if runtime_state.huge_signal_bfs is not None:
            index_map = {i: runtime_state.gene_to_ind[extra_genes[i]] for i in range(len(extra_genes))}
            runtime_state.huge_signal_bfs = sparse.csc_matrix(
                (
                    runtime_state.huge_signal_bfs.data,
                    [index_map[x] for x in runtime_state.huge_signal_bfs.indices],
                    runtime_state.huge_signal_bfs.indptr,
                ),
                shape=runtime_state.huge_signal_bfs.shape,
            )

        if runtime_state.huge_signal_bfs_for_regression is not None:
            index_map = {i: runtime_state.gene_to_ind[extra_genes[i]] for i in range(len(extra_genes))}
            runtime_state.huge_signal_bfs_for_regression = sparse.csc_matrix(
                (
                    runtime_state.huge_signal_bfs_for_regression.data,
                    [index_map[x] for x in runtime_state.huge_signal_bfs_for_regression.indices],
                    runtime_state.huge_signal_bfs_for_regression.indptr,
                ),
                shape=runtime_state.huge_signal_bfs_for_regression.shape,
            )

        if runtime_state.gene_covariates is not None:
            index_map_rev = {runtime_state.gene_to_ind[extra_genes[i]]: i for i in range(len(extra_genes))}
            runtime_state.gene_covariates = runtime_state.gene_covariates[
                [index_map_rev[x] for x in range(runtime_state.gene_covariates.shape[0])], :
            ]

    return (
        Y,
        Y_for_regression,
        Y_exomes,
        Y_positive_controls,
        Y_case_counts,
        [],
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )


def merge_y_into_existing_gene_universe(
    runtime_state,
    Y1,
    Y1_for_regression,
    Y1_exomes,
    Y1_positive_controls,
    Y1_case_counts,
    extra_genes,
    extra_Y,
    extra_Y_for_regression,
    extra_genes_all,
    extra_Y_exomes,
    extra_Y_positive_controls,
    extra_Y_case_counts,
    missing_value,
    missing_value_exomes,
    missing_value_positive_controls,
    missing_value_case_counts,
    extend_with_extra_genes=True,
    warn_fn=None,
):
    Y = Y1 + Y1_exomes + Y1_positive_controls + Y1_case_counts
    Y[np.isnan(Y1)] = Y1_exomes[np.isnan(Y1)] + Y1_positive_controls[np.isnan(Y1)] + Y1_case_counts[np.isnan(Y1)] + missing_value
    Y[np.isnan(Y1_exomes)] = Y1[np.isnan(Y1_exomes)] + Y1_positive_controls[np.isnan(Y1_exomes)] + Y1_case_counts[np.isnan(Y1_exomes)] + missing_value_exomes
    Y[np.isnan(Y1_positive_controls)] = Y1[np.isnan(Y1_positive_controls)] + Y1_exomes[np.isnan(Y1_positive_controls)] + Y1_case_counts[np.isnan(Y1_positive_controls)] + missing_value_positive_controls
    Y[np.isnan(Y1_case_counts)] = Y1[np.isnan(Y1_case_counts)] + Y1_exomes[np.isnan(Y1_case_counts)] + Y1_positive_controls[np.isnan(Y1_case_counts)] + missing_value_case_counts

    Y_for_regression = Y1_for_regression + Y1_exomes + Y1_positive_controls + Y1_case_counts
    Y_for_regression[np.isnan(Y1_for_regression)] = Y1_exomes[np.isnan(Y1_for_regression)] + Y1_positive_controls[np.isnan(Y1_for_regression)] + Y1_case_counts[np.isnan(Y1_for_regression)] + missing_value
    Y_for_regression[np.isnan(Y1_exomes)] = Y1_for_regression[np.isnan(Y1_exomes)] + Y1_positive_controls[np.isnan(Y1_exomes)] + Y1_case_counts[np.isnan(Y1_exomes)] + missing_value_exomes
    Y_for_regression[np.isnan(Y1_positive_controls)] = Y_for_regression[np.isnan(Y1_positive_controls)] + Y1_exomes[np.isnan(Y1_positive_controls)] + Y1_case_counts[np.isnan(Y1_positive_controls)] + missing_value_positive_controls
    Y_for_regression[np.isnan(Y1_case_counts)] = Y_for_regression[np.isnan(Y1_case_counts)] + Y1_exomes[np.isnan(Y1_case_counts)] + Y1_positive_controls[np.isnan(Y1_case_counts)] + missing_value_case_counts

    Y_exomes = Y1_exomes
    Y_exomes[np.isnan(Y1_exomes)] = missing_value_exomes

    Y_positive_controls = Y1_positive_controls
    Y_positive_controls[np.isnan(Y1_positive_controls)] = missing_value_positive_controls

    Y_case_counts = Y1_case_counts
    Y_case_counts[np.isnan(Y1_case_counts)] = missing_value_case_counts

    extra_gene_to_ind = construct_map_to_ind(extra_genes)
    extra_Y = list(extra_Y)
    extra_Y_for_regression = list(extra_Y_for_regression)
    new_extra_Y_exomes = list(np.full(len(extra_Y), missing_value_exomes))
    new_extra_Y_positive_controls = list(np.full(len(extra_Y), missing_value_positive_controls))
    new_extra_Y_case_counts = list(np.full(len(extra_Y), missing_value_case_counts))

    num_add = 0
    num_skipped_extra = 0
    for i in range(len(extra_genes_all)):
        if extra_genes_all[i] in extra_gene_to_ind:
            extra_Y[extra_gene_to_ind[extra_genes_all[i]]] += (extra_Y_exomes[i] + extra_Y_positive_controls[i] + extra_Y_case_counts[i])
            extra_Y_for_regression[extra_gene_to_ind[extra_genes_all[i]]] += (extra_Y_exomes[i] + extra_Y_positive_controls[i] + extra_Y_case_counts[i])
            new_extra_Y_exomes[extra_gene_to_ind[extra_genes_all[i]]] = extra_Y_exomes[i]
            new_extra_Y_positive_controls[extra_gene_to_ind[extra_genes_all[i]]] = extra_Y_positive_controls[i]
            new_extra_Y_case_counts[extra_gene_to_ind[extra_genes_all[i]]] = extra_Y_case_counts[i]
        elif extend_with_extra_genes:
            num_add += 1
            extra_genes.append(extra_genes_all[i])
            extra_Y.append(extra_Y_exomes[i] + extra_Y_positive_controls[i] + extra_Y_case_counts[i])
            extra_Y_for_regression.append(extra_Y_exomes[i] + extra_Y_positive_controls[i] + extra_Y_case_counts[i])
            new_extra_Y_exomes.append(extra_Y_exomes[i])
            new_extra_Y_positive_controls.append(extra_Y_positive_controls[i])
            new_extra_Y_case_counts.append(extra_Y_case_counts[i])
        else:
            num_skipped_extra += 1

    if not extend_with_extra_genes and warn_fn is not None and (len(extra_genes) > 0 or num_skipped_extra > 0):
        warn_fn(
            "Ignored %d genes from Y inputs because they were absent from the explicit gene universe"
            % (len(extra_genes) + num_skipped_extra)
        )
        extra_genes = []
        extra_Y = []
        extra_Y_for_regression = []
        new_extra_Y_exomes = []
        new_extra_Y_positive_controls = []
        new_extra_Y_case_counts = []

    extra_Y = np.array(extra_Y)
    extra_Y_for_regression = np.array(extra_Y_for_regression)
    extra_Y_exomes = np.array(new_extra_Y_exomes)
    extra_Y_positive_controls = np.array(new_extra_Y_positive_controls)
    extra_Y_case_counts = np.array(new_extra_Y_case_counts)

    if runtime_state.huge_signal_bfs is not None:
        runtime_state.huge_signal_bfs = sparse.csc_matrix(
            (runtime_state.huge_signal_bfs.data, runtime_state.huge_signal_bfs.indices, runtime_state.huge_signal_bfs.indptr),
            shape=(runtime_state.huge_signal_bfs.shape[0] + num_add, runtime_state.huge_signal_bfs.shape[1]),
        )

    if runtime_state.huge_signal_bfs_for_regression is not None:
        runtime_state.huge_signal_bfs_for_regression = sparse.csc_matrix(
            (
                runtime_state.huge_signal_bfs_for_regression.data,
                runtime_state.huge_signal_bfs_for_regression.indices,
                runtime_state.huge_signal_bfs_for_regression.indptr,
            ),
            shape=(runtime_state.huge_signal_bfs_for_regression.shape[0] + num_add, runtime_state.huge_signal_bfs_for_regression.shape[1]),
        )

    if runtime_state.gene_covariates is not None:
        add_gene_covariates = np.tile(np.mean(runtime_state.gene_covariates, axis=0), num_add).reshape((num_add, runtime_state.gene_covariates.shape[1]))
        runtime_state.gene_covariates = np.vstack((runtime_state.gene_covariates, add_gene_covariates))

    return (
        Y,
        Y_for_regression,
        Y_exomes,
        Y_positive_controls,
        Y_case_counts,
        extra_genes,
        extra_Y,
        extra_Y_for_regression,
        extra_Y_exomes,
        extra_Y_positive_controls,
        extra_Y_case_counts,
    )


def align_extra_genes_with_new_source(
    existing_extra_genes,
    existing_extra_values,
    new_source_genes,
    new_source_values,
    existing_missing_values,
    new_source_missing_value,
):
    new_source_gene_to_ind = construct_map_to_ind(new_source_genes)
    merged_extra_genes = list(new_source_genes)
    merged_new_source_values = list(new_source_values)
    aligned_existing_values = [
        list(np.full(len(merged_new_source_values), missing_value))
        for missing_value in existing_missing_values
    ]

    for i in range(len(existing_extra_genes)):
        gene = existing_extra_genes[i]
        if gene in new_source_gene_to_ind:
            source_ind = new_source_gene_to_ind[gene]
            for j in range(len(existing_extra_values)):
                aligned_existing_values[j][source_ind] = existing_extra_values[j][i]
        else:
            merged_extra_genes.append(gene)
            for j in range(len(existing_extra_values)):
                aligned_existing_values[j].append(existing_extra_values[j][i])
            merged_new_source_values.append(new_source_missing_value)

    return (
        merged_extra_genes,
        [np.array(x) for x in aligned_existing_values],
        np.array(merged_new_source_values),
    )


def apply_hold_out_chrom_to_y(
    runtime_state,
    Y,
    extra_genes,
    extra_Y,
    hold_out_chrom,
    gene_loc_file,
    *,
    bail_fn,
    warn_fn=None,
):
    if hold_out_chrom is None:
        return (Y, extra_genes, extra_Y)

    if runtime_state.gene_to_chrom is None:
        (
            runtime_state.gene_chrom_name_pos,
            runtime_state.gene_to_chrom,
            runtime_state.gene_to_pos,
        ) = read_loc_file_with_gene_map(
            gene_loc_file,
            gene_label_map=runtime_state.gene_label_map,
            clean_chrom_fn=clean_chrom_name,
            warn_fn=warn_fn,
            bail_fn=bail_fn,
        )

    extra_Y_mask = np.full(len(extra_Y), True)
    for i in range(len(extra_genes)):
        if extra_genes[i] in runtime_state.gene_to_chrom and runtime_state.gene_to_chrom[extra_genes[i]] == hold_out_chrom:
            extra_Y_mask[i] = False
    if np.sum(~extra_Y_mask) > 0:
        extra_genes = [extra_genes[i] for i in range(len(extra_genes)) if extra_Y_mask[i]]
        extra_Y = extra_Y[extra_Y_mask]

    if runtime_state.genes is not None:
        Y_nan_mask = np.full(len(Y), False)
        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in runtime_state.gene_to_chrom and runtime_state.gene_to_chrom[runtime_state.genes[i]] == hold_out_chrom:
                Y_nan_mask[i] = True
        if np.sum(Y_nan_mask) > 0:
            Y[Y_nan_mask] = np.nan

    return (Y, extra_genes, extra_Y)
