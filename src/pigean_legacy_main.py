"""Active transitional PIGEAN inner core.

This file is still on the runtime path and still owns unextracted inner
sampler/reader behavior. The name is historical, but the current policy is:

1. keep stage orchestration, CLI surface, and new shared helpers in package
   modules under `src/pigean/`
2. do not add new feature logic here unless the same change also extracts or
   shrinks existing inner-core ownership
3. treat this file as active technical debt to drain, not as dead ballast
"""

#Usage: python calc.py
#
#This script...
#
#Arguments:
# warnings-file[path]: file to output warnings to; default: standard error
#these are for debugging
#import cProfile
#import resource

#import urllib.request #used (below) only if you specify a URL as an input file
#import requests #used (below) only if you specify --lmm-auth-key

import optparse
import sys
import time
import os
import copy
import contextlib
import functools
import json
import re
from dataclasses import dataclass, field
import scipy
import scipy.sparse as sparse
import scipy.stats
import numpy as np
import itertools
import gzip
import random
from pigean import runtime as pigean_runtime
from pigean import model as pigean_model
from pigean import main_support as pigean_main_support
from pigean import state as pigean_state
from pigean import x_inputs as pigean_x_inputs
from pigean import x_inputs_core as pigean_x_inputs_core
from pigean import y_inputs as pigean_y_inputs
from pigean import y_inputs_core as pigean_y_inputs_core
from pigean import gibbs as pigean_gibbs

try:
    import pegs_shared.bundle as pegs_bundle
    import pegs_shared.cli as pegs_cli
    import pegs_shared.gene_io as pegs_gene_io
    import pegs_shared.huge_cache as pegs_huge_cache
    import pegs_shared.phewas as pegs_phewas
    import pegs_shared.regression as pegs_regression
    import pegs_shared.runtime_matrix as pegs_runtime_matrix
    import pegs_shared.types as pegs_types
    import pegs_shared.xdata as pegs_xdata
    import pegs_shared.ydata as pegs_ydata
    from . import pegs_cli_errors as pegs_cli_errors
    from . import pegs_utils as pegs_utils_mod
except ImportError:
    import pegs_shared.bundle as pegs_bundle
    import pegs_shared.cli as pegs_cli
    import pegs_shared.gene_io as pegs_gene_io
    import pegs_shared.huge_cache as pegs_huge_cache
    import pegs_shared.phewas as pegs_phewas
    import pegs_shared.regression as pegs_regression
    import pegs_shared.runtime_matrix as pegs_runtime_matrix
    import pegs_shared.types as pegs_types
    import pegs_shared.xdata as pegs_xdata
    import pegs_shared.ydata as pegs_ydata
    import pegs_cli_errors as pegs_cli_errors
    import pegs_utils as pegs_utils_mod

PegsXReadConfig = pegs_types.XReadConfig
PegsXReadCallbacks = pegs_types.XReadCallbacks
PegsXReadPostCallbacks = pegs_types.XReadPostCallbacks

DataValidationError = pegs_cli_errors.DataValidationError
PegsCliError = pegs_cli_errors.PegsCliError
pegs_handle_cli_exception = pegs_cli_errors.handle_cli_exception
pegs_handle_unexpected_exception = pegs_cli_errors.handle_unexpected_exception

pegs_apply_cli_config_overrides = pegs_cli.apply_cli_config_overrides
pegs_callback_set_comma_separated_args = pegs_cli.callback_set_comma_separated_args
pegs_callback_set_comma_separated_args_as_float = pegs_cli.callback_set_comma_separated_args_as_float
pegs_coerce_option_int_list = pegs_cli.coerce_option_int_list
pegs_configure_random_seed = pegs_cli.configure_random_seed
pegs_emit_stderr_warning = pegs_cli.emit_stderr_warning
pegs_fail_removed_cli_aliases = pegs_cli.fail_removed_cli_aliases
pegs_format_removed_option_message = pegs_cli.format_removed_option_message
pegs_harmonize_cli_mode_args = pegs_cli.harmonize_cli_mode_args
pegs_initialize_cli_logging = pegs_cli.initialize_cli_logging
pegs_is_path_like_dest = pegs_cli.is_path_like_dest
pegs_iter_parser_options = pegs_cli.iter_parser_options
pegs_json_safe = pegs_cli.json_safe
pegs_load_json_config = pegs_cli.load_json_config
pegs_merge_dicts = pegs_cli.merge_dicts
pegs_resolve_config_path_value = pegs_cli.resolve_config_path_value

pegs_xdata_from_input_plan = pegs_xdata.xdata_from_input_plan
pegs_build_read_x_ingestion_options = pegs_xdata.build_read_x_ingestion_options
pegs_build_read_x_post_options = pegs_xdata.build_read_x_post_options
pegs_initialize_matrix_and_gene_index_state = pegs_xdata.initialize_matrix_and_gene_index_state

pegs_sync_y_state = pegs_ydata.sync_y_state
pegs_sync_hyperparameter_state = pegs_ydata.sync_hyperparameter_state
pegs_sync_phewas_runtime_state = pegs_ydata.sync_phewas_runtime_state
pegs_sync_runtime_state_bundle = pegs_ydata.sync_runtime_state_bundle

pegs_load_aligned_gene_bfs = pegs_gene_io.load_aligned_gene_bfs
pegs_load_aligned_gene_covariates = pegs_gene_io.load_aligned_gene_covariates

pegs_apply_huge_statistics_meta_to_runtime = pegs_huge_cache.apply_huge_statistics_meta_to_runtime
pegs_build_huge_statistics_matrix_row_genes = pegs_huge_cache.build_huge_statistics_matrix_row_genes
pegs_build_huge_statistics_meta = pegs_huge_cache.build_huge_statistics_meta
pegs_build_huge_statistics_score_maps = pegs_huge_cache.build_huge_statistics_score_maps
pegs_coerce_runtime_state_dict = pegs_huge_cache.coerce_runtime_state_dict
pegs_combine_runtime_huge_scores = pegs_huge_cache.combine_runtime_huge_scores
pegs_get_huge_statistics_paths_for_prefix = pegs_huge_cache.get_huge_statistics_paths_for_prefix
pegs_load_huge_statistics_sparse_and_vectors = pegs_huge_cache.load_huge_statistics_sparse_and_vectors
pegs_read_huge_statistics_covariates_if_present = pegs_huge_cache.read_huge_statistics_covariates_if_present
pegs_read_huge_statistics_text_tables = pegs_huge_cache.read_huge_statistics_text_tables
pegs_read_numeric_vector_file = pegs_huge_cache.read_numeric_vector_file
pegs_resolve_huge_statistics_gene_vectors = pegs_huge_cache.resolve_huge_statistics_gene_vectors
pegs_validate_huge_statistics_loaded_shapes = pegs_huge_cache.validate_huge_statistics_loaded_shapes
pegs_write_huge_statistics_runtime_vectors = pegs_huge_cache.write_huge_statistics_runtime_vectors
pegs_write_huge_statistics_sparse_components = pegs_huge_cache.write_huge_statistics_sparse_components
pegs_write_huge_statistics_text_tables = pegs_huge_cache.write_huge_statistics_text_tables
pegs_write_numeric_vector_file = pegs_huge_cache.write_numeric_vector_file

pegs_accumulate_standard_phewas_outputs = pegs_phewas.accumulate_standard_phewas_outputs
pegs_append_phewas_metric_block = pegs_phewas.append_phewas_metric_block
pegs_prepare_phewas_phenos_from_file = pegs_phewas.prepare_phewas_phenos_from_file
pegs_read_phewas_file_batch = pegs_phewas.read_phewas_file_batch
pegs_build_phewas_stage_config = pegs_phewas.build_phewas_stage_config
pegs_resolve_gene_phewas_input_decision_for_stage = pegs_phewas.resolve_gene_phewas_input_decision_for_stage

pegs_finalize_regression_outputs = pegs_regression.finalize_regression_outputs
pegs_compute_beta_tildes = pegs_regression.compute_beta_tildes
pegs_compute_logistic_beta_tildes = pegs_regression.compute_logistic_beta_tildes
pegs_correct_beta_tildes = pegs_regression.correct_beta_tildes
pegs_compute_multivariate_beta_tildes = pegs_regression.compute_multivariate_beta_tildes

pegs_is_huge_statistics_bundle_path = pegs_utils_mod.is_huge_statistics_bundle_path
pegs_initialize_read_x_batch_seed_state = pegs_utils_mod.initialize_read_x_batch_seed_state
pegs_initialize_filtered_gene_set_state = pegs_utils_mod.initialize_filtered_gene_set_state
pegs_maybe_prepare_filtered_correlation = pegs_utils_mod.maybe_prepare_filtered_correlation
pegs_resolve_read_x_run_logistic = pegs_utils_mod.resolve_read_x_run_logistic
pegs_record_read_x_counts = pegs_utils_mod.record_read_x_counts
pegs_standardize_qc_metrics_after_x_read = pegs_utils_mod.standardize_qc_metrics_after_x_read
pegs_maybe_correct_gene_set_betas_after_x_read = pegs_utils_mod.maybe_correct_gene_set_betas_after_x_read
pegs_maybe_limit_initial_gene_sets_by_p = pegs_utils_mod.maybe_limit_initial_gene_sets_by_p
pegs_maybe_prune_gene_sets_after_x_read = pegs_utils_mod.maybe_prune_gene_sets_after_x_read
pegs_initialize_hyper_defaults_after_x_read = pegs_utils_mod.initialize_hyper_defaults_after_x_read
pegs_maybe_adjust_overaggressive_p_filter_after_x_read = pegs_utils_mod.maybe_adjust_overaggressive_p_filter_after_x_read
pegs_apply_post_read_gene_set_size_and_qc_filters = pegs_utils_mod.apply_post_read_gene_set_size_and_qc_filters
pegs_prepare_read_x_inputs = pegs_utils_mod.prepare_read_x_inputs
pegs_build_read_x_pipeline_config = pegs_utils_mod.build_read_x_pipeline_config
pegs_build_xin_to_p_noninf_index_map = pegs_utils_mod.build_xin_to_p_noninf_index_map
pegs_load_and_apply_gene_phewas_bfs_to_runtime = pegs_utils_mod.load_and_apply_gene_phewas_bfs_to_runtime
pegs_load_and_apply_gene_set_statistics_to_runtime = pegs_utils_mod.load_and_apply_gene_set_statistics_to_runtime
pegs_set_runtime_y_from_inputs = pegs_utils_mod.set_runtime_y_from_inputs
pegs_compute_banded_y_corr_cholesky = pegs_runtime_matrix.compute_banded_y_corr_cholesky
pegs_whiten_matrix_with_banded_cholesky = pegs_runtime_matrix.whiten_matrix_with_banded_cholesky
pegs_calc_shift_scale_for_dense_block = pegs_runtime_matrix.calc_shift_scale_for_dense_block
pegs_calc_X_shift_scale = pegs_runtime_matrix.calc_X_shift_scale
pegs_calculate_V_internal = pegs_runtime_matrix.calculate_V_internal
pegs_set_runtime_x_from_inputs = pegs_runtime_matrix.set_runtime_x_from_inputs
pegs_get_num_X_blocks = pegs_runtime_matrix.get_num_X_blocks
pegs_iterate_X_blocks_internal = pegs_runtime_matrix.iterate_X_blocks_internal
pegs_set_runtime_p = pegs_runtime_matrix.set_runtime_p
pegs_set_runtime_sigma = pegs_runtime_matrix.set_runtime_sigma
pegs_write_gene_set_statistics = pegs_utils_mod.write_gene_set_statistics
pegs_write_phewas_gene_set_statistics = pegs_utils_mod.write_phewas_gene_set_statistics
pegs_write_gene_statistics = pegs_utils_mod.write_gene_statistics
pegs_write_gene_gene_set_statistics = pegs_utils_mod.write_gene_gene_set_statistics
pegs_write_phewas_statistics = pegs_utils_mod.write_phewas_statistics
pegs_remove_tag_from_input = pegs_utils_mod.remove_tag_from_input
pegs_clean_chrom_name = pegs_utils_mod.clean_chrom_name
pegs_parse_gene_map_file = pegs_utils_mod.parse_gene_map_file
pegs_read_loc_file_with_gene_map = pegs_utils_mod.read_loc_file_with_gene_map
pegs_infer_columns_from_table_file = pegs_utils_mod.infer_columns_from_table_file
pegs_needs_gwas_column_detection = pegs_utils_mod.needs_gwas_column_detection
pegs_autodetect_gwas_columns = pegs_utils_mod.autodetect_gwas_columns
pegs_complete_p_beta_se = pegs_utils_mod.complete_p_beta_se
pegs_construct_map_to_ind = pegs_utils_mod.construct_map_to_ind
pegs_open_text_with_retry = pegs_utils_mod.open_text_with_retry
pegs_require_existing_nonempty_file = pegs_bundle.require_existing_nonempty_file
pegs_resolve_column_index = pegs_utils_mod.resolve_column_index
pegs_write_bundle_from_specs = pegs_bundle.write_bundle_from_specs
pegs_get_tar_write_mode_for_bundle_path = pegs_bundle.get_tar_write_mode_for_bundle_path
pegs_write_prefixed_tar_bundle = pegs_bundle.write_prefixed_tar_bundle
pegs_read_prefixed_tar_bundle = pegs_bundle.read_prefixed_tar_bundle
PEGS_EAGGL_BUNDLE_SCHEMA = pegs_bundle.EAGGL_BUNDLE_SCHEMA

# Canonical suffix tags used when expanding dense gene-set inputs into
# sparse derived sets (top/ext/bottom thresholds).
EXT_TAG = "ext"
BOT_TAG = "bot"
TOP_TAG = "top"

def bail(message):
    raise DataValidationError(message)

try:
    from pigean import cli as _pigean_cli
except ImportError:
    import pigean_cli as _pigean_cli

usage = _pigean_cli.usage
parser = _pigean_cli.parser
REMOVED_OPTION_REPLACEMENTS = _pigean_cli.REMOVED_OPTION_REPLACEMENTS
_build_mode_state = _pigean_cli._build_mode_state
_json_safe = _pigean_cli._json_safe

options = None
args = []
mode = None
config_mode = None
cli_specified_dests = set()
config_specified_dests = set()
NONE = 0
INFO = 1
DEBUG = 2
TRACE = 3
debug_level = 1
log_fh = None
warnings_fh = None


def _noop_log(*_args, **_kwargs):
    return None


log = _noop_log
warn = _noop_log

def _build_runtime_state(_options):
    pigean_state.bind_legacy_namespace(sys.modules[__name__])
    return pigean_runtime.build_runtime_state(PigeanState, _options)


_STATE_FIELDS_X_INDEXING = pigean_runtime.STATE_FIELDS_X_INDEXING
_STATE_FIELDS_Y_SOURCES = pigean_runtime.STATE_FIELDS_Y_SOURCES
_STATE_FIELDS_COVARIATE_CORRECTION = pigean_runtime.STATE_FIELDS_COVARIATE_CORRECTION
_STATE_FIELDS_SAMPLER_HYPER = pigean_runtime.STATE_FIELDS_SAMPLER_HYPER
_snapshot_state_fields = pigean_runtime.snapshot_state_fields
_restore_state_fields = pigean_runtime.restore_state_fields
_temporary_state_fields = pigean_runtime.temporary_state_fields


@contextlib.contextmanager
def _open_optional_gibbs_trace_files(gene_set_stats_trace_out, gene_stats_trace_out):
    with pigean_runtime.open_optional_gibbs_trace_files(
        gene_set_stats_trace_out,
        gene_stats_trace_out,
        open_gz=open_gz,
    ) as trace_handles:
        yield trace_handles


def _open_optional_inner_betas_trace_file(betas_trace_out):
    return pigean_runtime.open_optional_inner_betas_trace_file(betas_trace_out, open_gz=open_gz)


_close_optional_inner_betas_trace_file = pigean_runtime.close_optional_inner_betas_trace_file
_return_inner_betas_result = pigean_runtime.return_inner_betas_result
_maybe_unsubset_gene_sets = pigean_runtime.maybe_unsubset_gene_sets
_restore_subset_gene_sets = pigean_runtime.restore_subset_gene_sets
_temporary_unsubset_gene_sets = pigean_runtime.temporary_unsubset_gene_sets


def _bootstrap_cli(argv=None):
    global options, args, mode, config_mode, cli_specified_dests, config_specified_dests
    global NONE, INFO, DEBUG, TRACE, debug_level, log_fh, warnings_fh, log, warn

    should_continue = _pigean_cli._bootstrap_cli(argv)
    options = _pigean_cli.options
    args = _pigean_cli.args
    mode = _pigean_cli.mode
    config_mode = _pigean_cli.config_mode
    cli_specified_dests = _pigean_cli.cli_specified_dests
    config_specified_dests = _pigean_cli.config_specified_dests
    NONE = _pigean_cli.NONE
    INFO = _pigean_cli.INFO
    DEBUG = _pigean_cli.DEBUG
    TRACE = _pigean_cli.TRACE
    debug_level = _pigean_cli.debug_level
    log_fh = _pigean_cli.log_fh
    warnings_fh = _pigean_cli.warnings_fh
    log = _pigean_cli.log
    warn = _pigean_cli.warn
    return should_continue


def open_gz(file, flag=None):
    return pegs_open_text_with_retry(
        file,
        flag=flag,
        log_fn=lambda message: log(message, INFO),
        bail_fn=bail,
    )


_HYPERPARAMETER_PROXY_FIELDS = pigean_runtime.HYPERPARAMETER_PROXY_FIELDS
_bind_hyperparameter_properties = pigean_runtime.bind_hyperparameter_properties


from pigean.state import PigeanState


_bind_hyperparameter_properties(PigeanState)


_set_const_Y = pigean_y_inputs_core.set_const_Y

def _standardize_qc_metrics_after_x_read(runtime_state):
    pegs_standardize_qc_metrics_after_x_read(runtime_state)


def _maybe_correct_gene_set_betas_after_x_read(
    runtime_state,
    filter_gene_set_p,
    correct_betas_mean,
    correct_betas_var,
    filter_using_phewas,
):
    pegs_maybe_correct_gene_set_betas_after_x_read(
        runtime_state,
        filter_gene_set_p=filter_gene_set_p,
        correct_betas_mean=correct_betas_mean,
        correct_betas_var=correct_betas_var,
        filter_using_phewas=filter_using_phewas,
        log_fn=lambda message: log(message),
    )


def _maybe_limit_initial_gene_sets_by_p(runtime_state, max_num_gene_sets_initial):
    pegs_maybe_limit_initial_gene_sets_by_p(
        runtime_state,
        max_num_gene_sets_initial=max_num_gene_sets_initial,
        log_fn=lambda message: log(message),
    )


def _maybe_prune_gene_sets_after_x_read(
    runtime_state,
    skip_betas,
    prune_gene_sets,
    prune_deterministically,
    weighted_prune_gene_sets,
):
    pegs_maybe_prune_gene_sets_after_x_read(
        runtime_state,
        skip_betas=skip_betas,
        prune_gene_sets=prune_gene_sets,
        prune_deterministically=prune_deterministically,
        weighted_prune_gene_sets=weighted_prune_gene_sets,
    )


def _initialize_hyper_defaults_after_x_read(
    runtime_state,
    initial_p,
    update_hyper_p,
    sigma_power,
    initial_sigma2_cond,
    update_hyper_sigma,
    initial_sigma2,
    sigma_soft_threshold_95,
    sigma_soft_threshold_5,
):
    return pegs_initialize_hyper_defaults_after_x_read(
        runtime_state,
        initial_p=initial_p,
        update_hyper_p=update_hyper_p,
        sigma_power=sigma_power,
        initial_sigma2_cond=initial_sigma2_cond,
        update_hyper_sigma=update_hyper_sigma,
        initial_sigma2=initial_sigma2,
        sigma_soft_threshold_95=sigma_soft_threshold_95,
        sigma_soft_threshold_5=sigma_soft_threshold_5,
        warn_fn=lambda message: warn(message),
        log_fn=lambda message: log(message),
    )


def _learn_hyper_for_gene_set_batch(
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
    # Keep per-gene-set p/sigma vectors unset while learning a batch-level
    # hyper state, then return explicit learned values to the caller.
    with _temporary_state_fields(
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

        # Only add pseudo counts for large values.
        num_p_pseudo = min(1, np.sum(gene_sets_for_hyper_mask) / 1000)

        # Optionally keep sigma/p fixed across batches.
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


def _apply_learned_batch_hyper_values(
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
        # Keep sigma/first_p = sigma/computed_p.
        adjusted_sigma2 = adjusted_sigma2 / adjusted_p * updated_first_p
        adjusted_p = updated_first_p

    runtime_state.ps[gene_sets_in_batch_mask] = adjusted_p
    runtime_state.sigma2s[gene_sets_in_batch_mask] = adjusted_sigma2
    return updated_first_p


def _finalize_batch_hyper_vectors(runtime_state, first_for_hyper):
    assert(len(runtime_state.ps) > 0 and not np.isnan(runtime_state.ps[0]))
    assert(len(runtime_state.sigma2s) > 0 and not np.isnan(runtime_state.sigma2s[0]))

    if first_for_hyper:
        runtime_state.ps[np.isnan(runtime_state.ps)] = runtime_state.ps[0]
        runtime_state.sigma2s[np.isnan(runtime_state.sigma2s)] = runtime_state.sigma2s[0]
    else:
        runtime_state.ps[np.isnan(runtime_state.ps)] = np.mean(runtime_state.ps[~np.isnan(runtime_state.ps)])
        runtime_state.sigma2s[np.isnan(runtime_state.sigma2s)] = np.mean(runtime_state.sigma2s[~np.isnan(runtime_state.sigma2s)])

    runtime_state.set_p(np.mean(runtime_state.ps))
    runtime_state.set_sigma(np.mean(runtime_state.sigma2s), runtime_state.sigma_power)


def _maybe_learn_batch_hyper_after_x_read(
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
):
    if skip_betas or runtime_state.p_values is None or (not update_hyper_p and not update_hyper_sigma) or len(runtime_state.gene_set_batches) == 0:
        return

    # Now learn the hyper values.
    assert(runtime_state.gene_set_batches[0] is not None)
    # First order unique batches; batches has one value per file but we need one per unique batch.
    ordered_batches = [runtime_state.gene_set_batches[0]] + list(set([x for x in runtime_state.gene_set_batches if not x == runtime_state.gene_set_batches[0]]))
    # Get total ignored counts per batch.
    batches_num_ignored = {}
    for i in range(len(batches)):
        if batches[i] not in batches_num_ignored:
            batches_num_ignored[batches[i]] = 0
        batches_num_ignored[batches[i]] += num_ignored_gene_sets[i]

    if update_hyper_p:
        runtime_state.ps = np.full(len(runtime_state.gene_set_batches), np.nan)
    runtime_state.sigma2s = np.full(len(runtime_state.gene_set_batches), np.nan)

    # None batch learns from first; others learn within themselves.
    first_p = None
    for ordered_batch_ind in range(len(ordered_batches)):
        if ordered_batches[ordered_batch_ind] is None:
            assert(first_for_hyper)
            continue

        gene_sets_in_batch_mask = (runtime_state.gene_set_batches == ordered_batches[ordered_batch_ind])
        gene_sets_for_hyper_mask = gene_sets_in_batch_mask.copy()

        if max_num_gene_sets_hyper is not None:
            if np.sum(gene_sets_for_hyper_mask) > max_num_gene_sets_hyper:
                drop_mask = np.random.default_rng().choice(
                    np.where(gene_sets_for_hyper_mask)[0],
                    size=np.sum(gene_sets_for_hyper_mask) - runtime_state.batch_size,
                    replace=False,
                )
                log("Dropping %d gene sets to reduce gene sets used for hyper parameters to %d" % (len(drop_mask), max_num_gene_sets_hyper))
                gene_sets_for_hyper_mask[drop_mask] = False

        if ordered_batch_ind > 0 and np.sum(gene_sets_for_hyper_mask) + batches_num_ignored[ordered_batches[ordered_batch_ind]] < 100:
            log("Skipping learning hyper for batch %s since not enough gene sets" % (ordered_batches[ordered_batch_ind]))
            continue

        hyper_fit = _learn_hyper_for_gene_set_batch(
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

        log("Learned p=%.4g, sigma2=%.4g (sigma2/p=%.4g)" % (computed_p, computed_sigma2, computed_sigma2 / computed_p))
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

        first_p = _apply_learned_batch_hyper_values(
            runtime_state=runtime_state,
            gene_sets_in_batch_mask=gene_sets_in_batch_mask,
            computed_p=computed_p,
            computed_sigma2=computed_sigma2,
            first_p=first_p,
            first_max_p_for_hyper=first_max_p_for_hyper,
        )

    _finalize_batch_hyper_vectors(runtime_state=runtime_state, first_for_hyper=first_for_hyper)


def _maybe_adjust_overaggressive_p_filter_after_x_read(
    runtime_state,
    filter_gene_set_p,
    increase_filter_gene_set_p,
    filter_using_phewas,
):
    pegs_maybe_adjust_overaggressive_p_filter_after_x_read(
        runtime_state,
        filter_gene_set_p=filter_gene_set_p,
        increase_filter_gene_set_p=increase_filter_gene_set_p,
        filter_using_phewas=filter_using_phewas,
        log_fn=lambda message: log(message),
    )


def _apply_post_read_gene_set_size_and_qc_filters(
    runtime_state,
    min_gene_set_size,
    max_gene_set_size,
    filter_gene_set_metric_z,
):
    pegs_apply_post_read_gene_set_size_and_qc_filters(
        runtime_state,
        min_gene_set_size=min_gene_set_size,
        max_gene_set_size=max_gene_set_size,
        filter_gene_set_metric_z=filter_gene_set_metric_z,
        log_fn=lambda message: log(message),
    )


def _maybe_filter_zero_uncorrected_betas_after_x_read(
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
):
    if skip_betas or runtime_state.p_values is None or filter_gene_set_p >= 1 or filter_using_phewas:
        return sort_rank

    # Remove features with identically zero uncorrected effects.
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

    log("%d have betas uncorrected equal 0" % np.sum(betas == 0))
    log("%d have betas uncorrected below 0.001" % np.sum(betas < 0.001))
    log("%d have betas uncorrected below 0.01" % np.sum(betas < 0.01))

    beta_ignore = betas == 0
    beta_mask = ~beta_ignore
    if np.sum(beta_mask) > 0:
        log("Ignoring %d gene sets due to zero uncorrected betas (kept %d)" % (np.sum(beta_ignore), np.sum(beta_mask)))
        runtime_state.subset_gene_sets(beta_mask, keep_missing=False, ignore_missing=True, skip_V=True)
    else:
        log("Keeping %d gene sets with zero uncorrected betas to avoid having none" % (np.sum(beta_ignore)))

    return -np.abs(betas[beta_mask])


def _maybe_reduce_gene_sets_to_max_after_x_read(
    runtime_state,
    skip_betas,
    max_num_gene_sets,
    sort_rank,
):
    if skip_betas or max_num_gene_sets is None or max_num_gene_sets <= 0:
        return
    if len(runtime_state.gene_sets) <= max_num_gene_sets:
        return

    log(
        "Current %d gene sets is greater than the maximum specified %d; reducing using pruning + small beta removal"
        % (len(runtime_state.gene_sets), max_num_gene_sets),
        DEBUG,
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
        log("Adding %d relatively uncorrelated gene sets (total now %d)" % (np.sum(gene_set_mask), np.sum(keep_mask)), TRACE)
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


def _get_col(col_name_or_index, header_cols, require_match=True):
    return pegs_resolve_column_index(
        col_name_or_index,
        header_cols,
        require_match=require_match,
        bail_fn=bail,
    )


def _determine_columns_from_file(filename):
    return pegs_infer_columns_from_table_file(
        filename,
        open_gz,
        log_fn=lambda message: log(message),
        bail_fn=bail,
    )


def _needs_gwas_column_detection(
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_n_col,
    gwas_n,
):
    try:
        from pigean import huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.needs_gwas_column_detection(
        sys.modules[__name__],
        gwas_pos_col,
        gwas_chrom_col,
        gwas_locus_col,
        gwas_p_col,
        gwas_beta_col,
        gwas_se_col,
        gwas_n_col,
        gwas_n,
    )


def _autodetect_gwas_columns(
    gwas_in,
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_freq_col,
    gwas_n_col,
    gwas_n,
    debug_just_check_header=False,
):
    try:
        from pigean import huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.autodetect_gwas_columns(
        sys.modules[__name__],
        gwas_in,
        gwas_pos_col,
        gwas_chrom_col,
        gwas_locus_col,
        gwas_p_col,
        gwas_beta_col,
        gwas_se_col,
        gwas_freq_col,
        gwas_n_col,
        gwas_n,
        debug_just_check_header=debug_just_check_header,
    )


class _IntervalTree(object):
    def __new__(cls, *args, **kwargs):
        try:
            from pigean import huge as _pigean_huge
        except ImportError:
            import pigean_huge as _pigean_huge

        _pigean_huge.configure_numpy(np)
        return _pigean_huge.IntervalTree(*args, **kwargs)


def _load_huge_gene_and_exon_locations(gene_loc_file, gene_label_map, hold_out_chrom=None, exons_loc_file=None):
    try:
        from pigean import huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    _pigean_huge.configure_numpy(np)
    return _pigean_huge.load_huge_gene_and_exon_locations(
        sys.modules[__name__],
        gene_loc_file,
        gene_label_map,
        hold_out_chrom=hold_out_chrom,
        exons_loc_file=exons_loc_file,
    )


def _compute_huge_variant_thresholds(min_var_posterior, gwas_high_p_posterior, allelic_var_k, gwas_prior_odds):
    try:
        from pigean import huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.compute_huge_variant_thresholds(
        sys.modules[__name__],
        min_var_posterior,
        gwas_high_p_posterior,
        allelic_var_k,
        gwas_prior_odds,
    )


def _validate_and_normalize_huge_gwas_inputs(
    gwas_in,
    gene_loc_file,
    credible_sets_in=None,
    credible_sets_chrom_col=None,
    credible_sets_pos_col=None,
    signal_window_size=250000,
    signal_min_sep=100000,
    signal_max_logp_ratio=None,
):
    try:
        from pigean import huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.validate_and_normalize_huge_gwas_inputs(
        sys.modules[__name__],
        gwas_in,
        gene_loc_file,
        credible_sets_in=credible_sets_in,
        credible_sets_chrom_col=credible_sets_chrom_col,
        credible_sets_pos_col=credible_sets_pos_col,
        signal_window_size=signal_window_size,
        signal_min_sep=signal_min_sep,
        signal_max_logp_ratio=signal_max_logp_ratio,
    )


def _write_huge_statistics_bundle(runtime_state, prefix, gene_bf, extra_genes, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression):
    try:
        from pigean import huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.write_huge_statistics_bundle(
        sys.modules[__name__],
        runtime_state,
        prefix,
        gene_bf,
        extra_genes,
        extra_gene_bf,
        gene_bf_for_regression,
        extra_gene_bf_for_regression,
    )


def _read_huge_statistics_bundle(runtime_state, prefix):
    try:
        from pigean import huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.read_huge_statistics_bundle(
        sys.modules[__name__],
        runtime_state,
        prefix,
    )

# ==========================================================================
# Gibbs epoch aggregation helpers.
# ==========================================================================
_GIBBS_EPOCH_SUM_KEYS = (
    "sum_betas_m",
    "sum_betas2_m",
    "sum_betas_uncorrected_m",
    "sum_betas_uncorrected2_m",
    "sum_postp_m",
    "sum_beta_tildes_m",
    "sum_z_scores_m",
    "num_sum_beta_m",
    "sum_Ys_m",
    "sum_Y_raws_m",
    "sum_log_pos_m",
    "sum_log_po_raws_m",
    "sum_log_po_raws2_m",
    "sum_priors_m",
    "sum_priors2_m",
    "sum_Ds_m",
    "sum_D_raws_m",
    "sum_bf_orig_m",
    "sum_bf_orig_raw_m",
    "sum_bf_orig_raw2_m",
    "num_sum_Y_m",
)

_GIBBS_EPOCH_MISSING_SUM_KEYS = (
    "sum_priors_missing_m",
    "sum_Ds_missing_m",
    "num_sum_priors_missing_m",
)

_GIBBS_EPOCH_RUNTIME_SUM_KEYS = (
    "all_sum_betas_m",
    "all_sum_betas2_m",
    "all_num_sum_m",
)


def _has_gibbs_epoch_aggregates(epoch_aggregates):
    return len(epoch_aggregates["sum_betas_m"]) > 0


def _build_gibbs_diag_sums(epoch_aggregates, sum_betas_m, sum_betas2_m, num_sum_beta_m, sum_Ds_m, num_sum_Y_m):
    if not _has_gibbs_epoch_aggregates(epoch_aggregates):
        return (sum_betas_m, sum_betas2_m, num_sum_beta_m, sum_Ds_m, num_sum_Y_m)
    diag_sum_betas_m = np.vstack(epoch_aggregates["sum_betas_m"] + [sum_betas_m])
    diag_sum_betas2_m = np.vstack(epoch_aggregates["sum_betas2_m"] + [sum_betas2_m])
    diag_num_sum_beta_m = np.vstack(epoch_aggregates["num_sum_beta_m"] + [num_sum_beta_m])
    diag_sum_Ds_m = np.vstack(epoch_aggregates["sum_Ds_m"] + [sum_Ds_m])
    diag_num_sum_Y_m = np.vstack(epoch_aggregates["num_sum_Y_m"] + [num_sum_Y_m])
    return (diag_sum_betas_m, diag_sum_betas2_m, diag_num_sum_beta_m, diag_sum_Ds_m, diag_num_sum_Y_m)


# ========================= Outer Gibbs Control Normalization =========================
def _normalize_gibbs_epoch_iteration_controls(
    max_num_iter,
    min_num_burn_in,
    max_num_burn_in,
    min_num_post_burn_in,
    max_num_post_burn_in,
):
    if min_num_burn_in is None:
        min_num_burn_in = 0
    if min_num_burn_in < 0:
        min_num_burn_in = 0

    if min_num_post_burn_in is None:
        min_num_post_burn_in = 1
    if min_num_post_burn_in < 1:
        min_num_post_burn_in = 1

    if max_num_burn_in is None:
        if max_num_iter is not None and max_num_iter > 0:
            max_num_burn_in = int(np.ceil(max_num_iter * 0.8))
        else:
            max_num_burn_in = min_num_burn_in
        if max_num_burn_in < 1:
            max_num_burn_in = 1
    if max_num_burn_in < 1:
        max_num_burn_in = 1
    if max_num_burn_in < min_num_burn_in:
        max_num_burn_in = min_num_burn_in

    if max_num_post_burn_in is None:
        if max_num_iter is not None and max_num_iter > 0:
            max_num_post_burn_in = max_num_iter - min_num_burn_in
        else:
            max_num_post_burn_in = min_num_post_burn_in
    if max_num_post_burn_in < 1:
        max_num_post_burn_in = 1
    if max_num_post_burn_in < min_num_post_burn_in:
        max_num_post_burn_in = min_num_post_burn_in

    passed_in_max_num_burn_in = max_num_burn_in
    epoch_max_num_iter_config = max_num_burn_in + max_num_post_burn_in
    if epoch_max_num_iter_config < 2:
        epoch_max_num_iter_config = 2

    return {
        "min_num_burn_in": min_num_burn_in,
        "max_num_burn_in": max_num_burn_in,
        "min_num_post_burn_in": min_num_post_burn_in,
        "max_num_post_burn_in": max_num_post_burn_in,
        "passed_in_max_num_burn_in": passed_in_max_num_burn_in,
        "epoch_max_num_iter_config": epoch_max_num_iter_config,
    }


def _sanitize_gibbs_diagnostic_controls(
    num_chains,
    diag_every,
    burn_in_patience,
    burn_in_stall_window,
    burn_in_stall_delta,
    stop_patience,
    stop_top_gene_k,
    stop_min_gene_d,
    active_beta_top_k,
    active_beta_min_abs,
    beta_rel_mcse_denom_floor,
    stall_window,
    stall_min_burn_in,
    stall_min_post_burn_in,
    stall_delta_rhat,
    stall_delta_mcse,
    stall_recent_window,
    stall_recent_eps,
    burn_in_rhat_quantile,
    use_max_r_for_convergence,
):
    if num_chains < 2:
        num_chains = 2
    if diag_every < 1:
        diag_every = 1
    if burn_in_patience < 1:
        burn_in_patience = 1
    if burn_in_stall_window is None or burn_in_stall_window < 2:
        burn_in_stall_window = 0
    if burn_in_stall_delta is None or burn_in_stall_delta < 0:
        burn_in_stall_delta = 0
    if stop_patience < 1:
        stop_patience = 1
    if stop_top_gene_k < 1:
        stop_top_gene_k = 1
    if stop_min_gene_d is not None:
        if stop_min_gene_d < 0:
            stop_min_gene_d = 0
        if stop_min_gene_d > 1:
            stop_min_gene_d = 1
    if active_beta_top_k < 1:
        active_beta_top_k = 1
    if active_beta_min_abs < 0:
        active_beta_min_abs = 0
    if beta_rel_mcse_denom_floor <= 0:
        beta_rel_mcse_denom_floor = 1e-12
    if stall_window is None or stall_window < 2:
        stall_window = 0
    if stall_min_burn_in is None or stall_min_burn_in < 0:
        stall_min_burn_in = 0
    if stall_min_post_burn_in is None or stall_min_post_burn_in < 0:
        stall_min_post_burn_in = 0
    if stall_delta_rhat is None or stall_delta_rhat < 0:
        stall_delta_rhat = 0
    if stall_delta_mcse is None or stall_delta_mcse < 0:
        stall_delta_mcse = 0
    if stall_recent_window is None or stall_recent_window < 2:
        stall_recent_window = 0
    if stall_recent_eps is None or stall_recent_eps < 0:
        stall_recent_eps = 0
    if burn_in_rhat_quantile is None:
        burn_in_rhat_quantile = 1.0
    if use_max_r_for_convergence:
        burn_in_rhat_quantile = 1.0
    if burn_in_rhat_quantile < 0:
        burn_in_rhat_quantile = 0
    elif burn_in_rhat_quantile > 1:
        burn_in_rhat_quantile = 1

    return {
        "num_chains": num_chains,
        "diag_every": diag_every,
        "burn_in_patience": burn_in_patience,
        "burn_in_stall_window": burn_in_stall_window,
        "burn_in_stall_delta": burn_in_stall_delta,
        "stop_patience": stop_patience,
        "stop_top_gene_k": stop_top_gene_k,
        "stop_min_gene_d": stop_min_gene_d,
        "active_beta_top_k": active_beta_top_k,
        "active_beta_min_abs": active_beta_min_abs,
        "beta_rel_mcse_denom_floor": beta_rel_mcse_denom_floor,
        "stall_window": stall_window,
        "stall_min_burn_in": stall_min_burn_in,
        "stall_min_post_burn_in": stall_min_post_burn_in,
        "stall_delta_rhat": stall_delta_rhat,
        "stall_delta_mcse": stall_delta_mcse,
        "stall_recent_window": stall_recent_window,
        "stall_recent_eps": stall_recent_eps,
        "burn_in_rhat_quantile": burn_in_rhat_quantile,
    }


def _normalize_gibbs_run_controls(
    max_num_iter,
    total_num_iter,
    max_num_restarts,
    num_chains,
    min_num_burn_in,
    max_num_burn_in,
    min_num_post_burn_in,
    max_num_post_burn_in,
    diag_every,
    burn_in_patience,
    burn_in_stall_window,
    burn_in_stall_delta,
    stop_patience,
    stop_top_gene_k,
    stop_min_gene_d,
    active_beta_top_k,
    active_beta_min_abs,
    beta_rel_mcse_denom_floor,
    stall_window,
    stall_min_burn_in,
    stall_min_post_burn_in,
    stall_delta_rhat,
    stall_delta_mcse,
    stall_recent_window,
    stall_recent_eps,
    burn_in_rhat_quantile,
    use_max_r_for_convergence,
):
    if max_num_restarts is None or max_num_restarts < 0:
        max_num_restarts = 0
    target_num_epochs = max_num_restarts + 1

    normalized_epoch_controls = _normalize_gibbs_epoch_iteration_controls(
        max_num_iter=max_num_iter,
        min_num_burn_in=min_num_burn_in,
        max_num_burn_in=max_num_burn_in,
        min_num_post_burn_in=min_num_post_burn_in,
        max_num_post_burn_in=max_num_post_burn_in,
    )
    min_num_burn_in = normalized_epoch_controls["min_num_burn_in"]
    max_num_burn_in = normalized_epoch_controls["max_num_burn_in"]
    min_num_post_burn_in = normalized_epoch_controls["min_num_post_burn_in"]
    max_num_post_burn_in = normalized_epoch_controls["max_num_post_burn_in"]
    epoch_max_num_iter_config = normalized_epoch_controls["epoch_max_num_iter_config"]

    if total_num_iter is None:
        total_num_iter = epoch_max_num_iter_config
    if total_num_iter < 1:
        total_num_iter = 1
    run_state = _initialize_gibbs_run_state(total_num_iter, target_num_epochs, max_num_restarts)

    sanitized_diag_controls = _sanitize_gibbs_diagnostic_controls(
        num_chains=num_chains,
        diag_every=diag_every,
        burn_in_patience=burn_in_patience,
        burn_in_stall_window=burn_in_stall_window,
        burn_in_stall_delta=burn_in_stall_delta,
        stop_patience=stop_patience,
        stop_top_gene_k=stop_top_gene_k,
        stop_min_gene_d=stop_min_gene_d,
        active_beta_top_k=active_beta_top_k,
        active_beta_min_abs=active_beta_min_abs,
        beta_rel_mcse_denom_floor=beta_rel_mcse_denom_floor,
        stall_window=stall_window,
        stall_min_burn_in=stall_min_burn_in,
        stall_min_post_burn_in=stall_min_post_burn_in,
        stall_delta_rhat=stall_delta_rhat,
        stall_delta_mcse=stall_delta_mcse,
        stall_recent_window=stall_recent_window,
        stall_recent_eps=stall_recent_eps,
        burn_in_rhat_quantile=burn_in_rhat_quantile,
        use_max_r_for_convergence=use_max_r_for_convergence,
    )

    (
        _,
        first_min_num_burn_in,
        first_max_num_burn_in,
        first_min_num_post_burn_in,
        first_max_num_post_burn_in,
    ) = _resolve_epoch_iteration_budget(
        total_num_iter,
        epoch_max_num_iter_config,
        min_num_burn_in,
        max_num_burn_in,
        min_num_post_burn_in,
        max_num_post_burn_in,
    )

    return GibbsRunControls(
        max_num_restarts=max_num_restarts,
        target_num_epochs=target_num_epochs,
        min_num_burn_in=normalized_epoch_controls["min_num_burn_in"],
        max_num_burn_in=normalized_epoch_controls["max_num_burn_in"],
        min_num_post_burn_in=normalized_epoch_controls["min_num_post_burn_in"],
        max_num_post_burn_in=normalized_epoch_controls["max_num_post_burn_in"],
        passed_in_max_num_burn_in=normalized_epoch_controls["passed_in_max_num_burn_in"],
        epoch_max_num_iter_config=normalized_epoch_controls["epoch_max_num_iter_config"],
        total_num_iter=total_num_iter,
        run_state=run_state,
        num_chains=sanitized_diag_controls["num_chains"],
        diag_every=sanitized_diag_controls["diag_every"],
        burn_in_patience=sanitized_diag_controls["burn_in_patience"],
        burn_in_stall_window=sanitized_diag_controls["burn_in_stall_window"],
        burn_in_stall_delta=sanitized_diag_controls["burn_in_stall_delta"],
        stop_patience=sanitized_diag_controls["stop_patience"],
        stop_top_gene_k=sanitized_diag_controls["stop_top_gene_k"],
        stop_min_gene_d=sanitized_diag_controls["stop_min_gene_d"],
        active_beta_top_k=sanitized_diag_controls["active_beta_top_k"],
        active_beta_min_abs=sanitized_diag_controls["active_beta_min_abs"],
        beta_rel_mcse_denom_floor=sanitized_diag_controls["beta_rel_mcse_denom_floor"],
        stall_window=sanitized_diag_controls["stall_window"],
        stall_min_burn_in=sanitized_diag_controls["stall_min_burn_in"],
        stall_min_post_burn_in=sanitized_diag_controls["stall_min_post_burn_in"],
        stall_delta_rhat=sanitized_diag_controls["stall_delta_rhat"],
        stall_delta_mcse=sanitized_diag_controls["stall_delta_mcse"],
        stall_recent_window=sanitized_diag_controls["stall_recent_window"],
        stall_recent_eps=sanitized_diag_controls["stall_recent_eps"],
        burn_in_rhat_quantile=sanitized_diag_controls["burn_in_rhat_quantile"],
        first_min_num_burn_in=first_min_num_burn_in,
        first_max_num_burn_in=first_max_num_burn_in,
        first_min_num_post_burn_in=first_min_num_post_burn_in,
        first_max_num_post_burn_in=first_max_num_post_burn_in,
    )


# ========================= Outer Gibbs Configuration Logging =========================
def _build_gibbs_record_config(
    gibbs_controls,
    num_chains_betas,
    max_num_iter,
    use_mean_betas,
    warm_start,
    stopping_preset_name,
    r_threshold_burn_in,
    stop_mcse_quantile,
    max_abs_mcse_d,
    max_rel_mcse_beta,
    sparse_solution,
    sparse_frac_gibbs,
    sparse_max_gibbs,
    sparse_frac_betas,
    pre_filter_batch_size,
    max_allowed_batch_correlation,
    initial_linear_filter,
    correct_betas_mean,
    correct_betas_var,
    adjust_priors,
    experimental_hyper_mutation,
    increase_hyper_if_betas_below,
):
    return {
        "num_chains": gibbs_controls.num_chains,
        "max_num_restarts": gibbs_controls.max_num_restarts,
        "target_num_epochs": gibbs_controls.target_num_epochs,
        "total_num_iter": gibbs_controls.total_num_iter,
        "epoch_max_num_iter_config": gibbs_controls.epoch_max_num_iter_config,
        "burn_in_rhat_quantile": gibbs_controls.burn_in_rhat_quantile,
        "burn_in_patience": gibbs_controls.burn_in_patience,
        "first_min_num_burn_in": gibbs_controls.first_min_num_burn_in,
        "first_max_num_burn_in": gibbs_controls.first_max_num_burn_in,
        "first_min_num_post_burn_in": gibbs_controls.first_min_num_post_burn_in,
        "first_max_num_post_burn_in": gibbs_controls.first_max_num_post_burn_in,
        "burn_in_stall_window": gibbs_controls.burn_in_stall_window,
        "burn_in_stall_delta": gibbs_controls.burn_in_stall_delta,
        "active_beta_top_k": gibbs_controls.active_beta_top_k,
        "active_beta_min_abs": gibbs_controls.active_beta_min_abs,
        "stop_patience": gibbs_controls.stop_patience,
        "stop_top_gene_k": gibbs_controls.stop_top_gene_k,
        "stop_min_gene_d": gibbs_controls.stop_min_gene_d,
        "beta_rel_mcse_denom_floor": gibbs_controls.beta_rel_mcse_denom_floor,
        "stall_window": gibbs_controls.stall_window,
        "stall_min_burn_in": gibbs_controls.stall_min_burn_in,
        "stall_min_post_burn_in": gibbs_controls.stall_min_post_burn_in,
        "stall_delta_rhat": gibbs_controls.stall_delta_rhat,
        "stall_delta_mcse": gibbs_controls.stall_delta_mcse,
        "stall_recent_window": gibbs_controls.stall_recent_window,
        "stall_recent_eps": gibbs_controls.stall_recent_eps,
        "diag_every": gibbs_controls.diag_every,
        "num_chains_betas": num_chains_betas,
        "max_num_iter": max_num_iter,
        "use_mean_betas": use_mean_betas,
        "warm_start": warm_start,
        "stopping_preset_name": stopping_preset_name,
        "r_threshold_burn_in": r_threshold_burn_in,
        "stop_mcse_quantile": stop_mcse_quantile,
        "max_abs_mcse_d": max_abs_mcse_d,
        "max_rel_mcse_beta": max_rel_mcse_beta,
        "sparse_solution": sparse_solution,
        "sparse_frac_gibbs": sparse_frac_gibbs,
        "sparse_max_gibbs": sparse_max_gibbs,
        "sparse_frac_betas": sparse_frac_betas,
        "pre_filter_batch_size": pre_filter_batch_size,
        "max_allowed_batch_correlation": max_allowed_batch_correlation,
        "initial_linear_filter": initial_linear_filter,
        "correct_betas_mean": correct_betas_mean,
        "correct_betas_var": correct_betas_var,
        "adjust_priors": adjust_priors,
        "experimental_hyper_mutation": experimental_hyper_mutation,
        "increase_hyper_if_betas_below": increase_hyper_if_betas_below,
    }


def _record_gibbs_configuration_params(state, run_state, config):
    state._record_params(
        {
            "num_chains": config["num_chains"],
            "num_chains_betas": config["num_chains_betas"],
            "max_num_restarts": config["max_num_restarts"],
            "target_num_epochs": config["target_num_epochs"],
            "max_num_attempt_restarts": run_state.max_num_attempt_restarts,
            "max_num_iter": config["max_num_iter"],
            "total_num_iter": config["total_num_iter"],
            "epoch_max_num_iter": config["epoch_max_num_iter_config"],
            "use_mean_betas": config["use_mean_betas"],
            "warm_start": config["warm_start"],
            "stopping_preset_name": config["stopping_preset_name"],
            "r_threshold_burn_in": config["r_threshold_burn_in"],
            "burn_in_rhat_quantile": config["burn_in_rhat_quantile"],
            "burn_in_rhat_quantile_effective": config["burn_in_rhat_quantile"],
            "burn_in_patience": config["burn_in_patience"],
            "min_num_burn_in": config["first_min_num_burn_in"],
            "max_num_burn_in": config["first_max_num_burn_in"],
            "min_num_post_burn_in": config["first_min_num_post_burn_in"],
            "max_num_post_burn_in": config["first_max_num_post_burn_in"],
            "burn_in_stall_window": config["burn_in_stall_window"],
            "burn_in_stall_delta": config["burn_in_stall_delta"],
            "active_beta_top_k": config["active_beta_top_k"],
            "active_beta_min_abs": config["active_beta_min_abs"],
            "stop_mcse_quantile": config["stop_mcse_quantile"],
            "stop_patience": config["stop_patience"],
            "stop_top_gene_k": config["stop_top_gene_k"],
            "stop_min_gene_d": config["stop_min_gene_d"],
            "max_abs_mcse_d": config["max_abs_mcse_d"],
            "max_rel_mcse_beta": config["max_rel_mcse_beta"],
            "beta_rel_mcse_denom_floor": config["beta_rel_mcse_denom_floor"],
            "stall_window": config["stall_window"],
            "stall_min_burn_in": config["stall_min_burn_in"],
            "stall_min_post_burn_in": config["stall_min_post_burn_in"],
            "stall_delta_rhat": config["stall_delta_rhat"],
            "stall_delta_mcse": config["stall_delta_mcse"],
            "stall_recent_window": config["stall_recent_window"],
            "stall_recent_eps": config["stall_recent_eps"],
            "diag_every": config["diag_every"],
            "sparse_solution": config["sparse_solution"],
            "sparse_frac": config["sparse_frac_gibbs"],
            "sparse_max": config["sparse_max_gibbs"],
            "sparse_frac_betas": config["sparse_frac_betas"],
            "pre_filter_batch_size": config["pre_filter_batch_size"],
            "max_allowed_batch_correlation": config["max_allowed_batch_correlation"],
            "initial_linear_filter": config["initial_linear_filter"],
            "correct_betas_mean": config["correct_betas_mean"],
            "correct_betas_var": config["correct_betas_var"],
            "adjust_priors": config["adjust_priors"],
            "experimental_hyper_mutation": config["experimental_hyper_mutation"],
            "increase_hyper_if_betas_below": config["increase_hyper_if_betas_below"],
        }
    )
    state._record_param("min_num_post_burn_in_effective", config["first_min_num_post_burn_in"])
    state._record_param("stall_min_post_burn_samples", config["stall_min_post_burn_in"])


def _log_gibbs_configuration_summary(config, run_state):
    log("Running Gibbs")
    log(
        "Gibbs stopping preset=%s; burn-in: r_threshold=%.4g, rhat_q=%.3g, patience=%d; active betas: topK=%d, min_abs=%.4g"
        % (
            config["stopping_preset_name"],
            config["r_threshold_burn_in"],
            config["burn_in_rhat_quantile"],
            config["burn_in_patience"],
            config["active_beta_top_k"],
            config["active_beta_min_abs"],
        ),
        INFO,
    )
    log(
        "Gibbs restart schedule: target_epochs=%d (max_num_restarts=%d), max_attempts=%d, per-epoch max_num_iter=%d, total_num_iter=%d"
        % (
            config["target_num_epochs"],
            config["max_num_restarts"],
            run_state.max_num_attempt_restarts,
            config["epoch_max_num_iter_config"],
            config["total_num_iter"],
        ),
        INFO,
    )
    log(
        "Gibbs epoch bounds (epoch 1): burn=[%d,%d], post=[%d,%d], stall_window=%d, stall_delta=%.4g"
        % (
            config["first_min_num_burn_in"],
            config["first_max_num_burn_in"],
            config["first_min_num_post_burn_in"],
            config["first_max_num_post_burn_in"],
            config["burn_in_stall_window"],
            config["burn_in_stall_delta"],
        ),
        INFO,
    )
    log(
        "Gibbs stopping thresholds: stop_q=%.3g, stop_patience=%d, max_rel_mcse_beta=%.4g, beta_rel_mcse_denom_floor=%.4g, stop_top_gene_k=%d, stop_min_gene_d=%s, max_abs_mcse_d=%.4g, diag_every=%d"
        % (
            config["stop_mcse_quantile"],
            config["stop_patience"],
            config["max_rel_mcse_beta"],
            config["beta_rel_mcse_denom_floor"],
            config["stop_top_gene_k"],
            ("%.4g" % config["stop_min_gene_d"]) if config["stop_min_gene_d"] is not None else "None",
            config["max_abs_mcse_d"],
            config["diag_every"],
        ),
        INFO,
    )
    log(
        "Gibbs experimental hyper mutation: enabled=%s, threshold=%s"
        % (
            str(config["experimental_hyper_mutation"]),
            ("%.4g" % config["increase_hyper_if_betas_below"])
            if config["increase_hyper_if_betas_below"] is not None
            else "None",
        ),
        INFO,
    )
    log(
        "Gibbs stall controls: window=%d, min_burn=%d, min_post_for_stall=%d, delta_rhat=%.4g, delta_mcse=%.4g, recent_window=%d, recent_eps=%.4g"
        % (
            config["stall_window"],
            config["stall_min_burn_in"],
            config["stall_min_post_burn_in"],
            config["stall_delta_rhat"],
            config["stall_delta_mcse"],
            config["stall_recent_window"],
            config["stall_recent_eps"],
        ),
        INFO,
    )


# ========================= Outer Gibbs Epoch Execution =========================
def _resolve_epoch_iteration_budget(
    remaining_iter,
    epoch_max_num_iter_config,
    min_num_burn_in,
    max_num_burn_in,
    min_num_post_burn_in,
    max_num_post_burn_in,
):
    local_epoch_max_num_iter = min(epoch_max_num_iter_config, remaining_iter)
    if local_epoch_max_num_iter < 1:
        local_epoch_max_num_iter = 1

    local_max_num_burn_in = min(max_num_burn_in, max(1, local_epoch_max_num_iter - 1))
    local_min_num_burn_in = min(min_num_burn_in, local_max_num_burn_in)

    local_max_num_post_burn_in = min(max_num_post_burn_in, max(1, local_epoch_max_num_iter - local_min_num_burn_in))
    local_min_num_post_burn_in = min(min_num_post_burn_in, local_max_num_post_burn_in)

    # Ensure burn-in max still leaves room for at least the local post-burn minimum.
    local_max_num_burn_in = min(local_max_num_burn_in, max(1, local_epoch_max_num_iter - local_min_num_post_burn_in))
    local_min_num_burn_in = min(local_min_num_burn_in, local_max_num_burn_in)
    local_max_num_post_burn_in = min(local_max_num_post_burn_in, max(1, local_epoch_max_num_iter - local_min_num_burn_in))
    local_min_num_post_burn_in = min(local_min_num_post_burn_in, local_max_num_post_burn_in)

    return (
        local_epoch_max_num_iter,
        local_min_num_burn_in,
        local_max_num_burn_in,
        local_min_num_post_burn_in,
        local_max_num_post_burn_in,
    )

@dataclass
class GibbsRunState:
    target_num_epochs: int
    max_num_attempt_restarts: int
    num_p_increases: int = 0
    num_attempts: int = 0
    num_completed_epochs: int = 0
    remaining_total_iter: int = 0


@dataclass
class GibbsRunControls:
    max_num_restarts: int
    target_num_epochs: int
    min_num_burn_in: int
    max_num_burn_in: int
    min_num_post_burn_in: int
    max_num_post_burn_in: int
    passed_in_max_num_burn_in: int
    epoch_max_num_iter_config: int
    total_num_iter: int
    run_state: GibbsRunState
    num_chains: int
    diag_every: int
    burn_in_patience: int
    burn_in_stall_window: int
    burn_in_stall_delta: float
    stop_patience: int
    stop_top_gene_k: int
    stop_min_gene_d: float | None
    active_beta_top_k: int
    active_beta_min_abs: float
    beta_rel_mcse_denom_floor: float
    stall_window: int
    stall_min_burn_in: int
    stall_min_post_burn_in: int
    stall_delta_rhat: float
    stall_delta_mcse: float
    stall_recent_window: int
    stall_recent_eps: float
    burn_in_rhat_quantile: float
    first_min_num_burn_in: int
    first_max_num_burn_in: int
    first_min_num_post_burn_in: int
    first_max_num_post_burn_in: int


@dataclass
class GibbsEpochPhaseConfig:
    total_num_iter: int
    num_chains: int
    num_full_gene_sets: int
    use_mean_betas: bool
    max_mb_X_h: int
    target_num_epochs: int
    num_mad: int
    adjust_priors: bool
    epoch_max_num_iter_config: int
    min_num_burn_in: int
    max_num_burn_in: int
    min_num_post_burn_in: int
    max_num_post_burn_in: int
    increase_hyper_if_betas_below: float | None
    experimental_hyper_mutation: bool


@dataclass
class GibbsIterationUpdateConfig:
    use_mean_betas: bool
    warm_start: bool
    debug_zero_sparse: bool
    num_chains: int
    num_batches_parallel: int
    betas_trace_out: str | None
    update_huge_scores: bool
    compute_Y_raw: bool
    adjust_priors: bool


@dataclass
class GibbsEpochIterationStaticConfig:
    inner_beta_kwargs: dict
    iteration_update_config: GibbsIterationUpdateConfig
    cur_background_log_bf_v: object
    y_var_orig: float
    gauss_seidel: bool
    initial_linear_filter: bool
    sparse_frac_gibbs: float
    sparse_max_gibbs: float
    correct_betas_mean: bool
    correct_betas_var: bool
    prefilter_config: dict
    iteration_progress_config: dict


@dataclass
class GibbsEpochRuntimeConfigs:
    epoch_phase_config: GibbsEpochPhaseConfig
    epoch_iteration_static_config: GibbsEpochIterationStaticConfig


@dataclass
class GibbsIterationCorrectionConfig:
    inner_beta_kwargs: dict
    iteration_update_config: GibbsIterationUpdateConfig
    num_mad: int
    num_attempts: int
    max_num_attempt_restarts: int
    increase_hyper_if_betas_below_for_epoch: float | None
    experimental_hyper_mutation: bool
    num_before_checking_p_increase: int
    p_scale_factor: float


@dataclass
class GibbsEpochIterationLoopConfig:
    epoch_max_num_iter: int
    epoch_total_iter_offset: int
    trace_chain_offset: int
    full_betas_m_shape: tuple
    num_stack_batches: int
    stack_batch_size: int
    X_hstacked: object
    min_num_burn_in_for_epoch: int
    max_num_burn_in_for_epoch: int
    min_num_iter_for_epoch: int
    min_num_post_burn_in_for_epoch: int
    max_num_post_burn_in_for_epoch: int
    post_burn_reset_arrays: list
    post_burn_reset_missing_arrays: list
    inner_beta_kwargs: dict
    iteration_update_config: GibbsIterationUpdateConfig
    cur_background_log_bf_v: object
    y_var_orig: float
    gauss_seidel: bool
    initial_linear_filter: bool
    sparse_frac_gibbs: float
    sparse_max_gibbs: float
    correct_betas_mean: bool
    correct_betas_var: bool
    prefilter_config: dict
    iteration_progress_config: dict
    num_attempts: int
    max_num_attempt_restarts: int
    num_mad: int
    increase_hyper_if_betas_below_for_epoch: float | None
    experimental_hyper_mutation: bool
    num_before_checking_p_increase: int
    p_scale_factor: float


@dataclass
class GibbsIterationProgressRuntimeConfig:
    trace_chain_offset: int
    epoch_total_iter_offset: int
    epoch_max_num_iter: int
    max_num_burn_in_for_epoch: int
    min_num_iter_for_epoch: int
    min_num_burn_in_for_epoch: int
    max_num_post_burn_in_for_epoch: int
    min_num_post_burn_in_for_epoch: int
    post_burn_reset_arrays: list
    post_burn_reset_missing_arrays: list
    iteration_progress_config: dict


@dataclass
class GibbsIterationRuntimeConfigs:
    correction_config: GibbsIterationCorrectionConfig
    progress_runtime_config: GibbsIterationProgressRuntimeConfig
    iteration_state_config: dict


def _initialize_gibbs_run_state(total_num_iter, target_num_epochs, max_num_restarts):
    # Mutable run-level counters shared across epoch attempts.
    return GibbsRunState(
        target_num_epochs=target_num_epochs,
        max_num_attempt_restarts=target_num_epochs + max_num_restarts,
        num_p_increases=0,
        num_attempts=0,
        num_completed_epochs=0,
        remaining_total_iter=int(total_num_iter),
    )


def _prepare_gibbs_epoch_attempt(
    state,
    run_state,
    epoch_phase_config,
):
    # Resolve one epoch attempt's bounds and bookkeeping from run-level state.
    total_num_iter = epoch_phase_config.total_num_iter
    num_chains = epoch_phase_config.num_chains
    target_num_epochs = epoch_phase_config.target_num_epochs
    epoch_max_num_iter_config = epoch_phase_config.epoch_max_num_iter_config
    min_num_burn_in = epoch_phase_config.min_num_burn_in
    max_num_burn_in = epoch_phase_config.max_num_burn_in
    min_num_post_burn_in = epoch_phase_config.min_num_post_burn_in
    max_num_post_burn_in = epoch_phase_config.max_num_post_burn_in
    increase_hyper_if_betas_below = epoch_phase_config.increase_hyper_if_betas_below

    run_state.num_attempts += 1

    (
        epoch_max_num_iter,
        min_num_burn_in_for_epoch,
        max_num_burn_in_for_epoch,
        min_num_post_burn_in_for_epoch,
        max_num_post_burn_in_for_epoch,
    ) = _resolve_epoch_iteration_budget(
        run_state.remaining_total_iter,
        epoch_max_num_iter_config,
        min_num_burn_in,
        max_num_burn_in,
        min_num_post_burn_in,
        max_num_post_burn_in,
    )
    if epoch_max_num_iter < 1:
        return None

    trace_chain_offset = run_state.num_completed_epochs * num_chains
    p_scale_factor = 1 - np.log(state.p) / (2 * np.log(10))

    min_num_iter_for_epoch = min_num_burn_in_for_epoch
    increase_hyper_if_betas_below_for_epoch = increase_hyper_if_betas_below if run_state.num_completed_epochs == 0 else None
    num_before_checking_p_increase = max(min_num_iter_for_epoch, min_num_burn_in_for_epoch)
    if increase_hyper_if_betas_below_for_epoch is not None and num_before_checking_p_increase > min_num_iter_for_epoch:
        # Ensure this check runs before any early break.
        min_num_iter_for_epoch = num_before_checking_p_increase

    state._record_param("num_gibbs_restarts", run_state.num_attempts - 1, overwrite=True)
    state._record_param("num_gibbs_epochs_completed", run_state.num_completed_epochs, overwrite=True)
    if run_state.num_attempts > 1:
        log("Gibbs restart attempt %d" % (run_state.num_attempts - 1))
    log(
        "Gibbs epoch %d/%d: max_num_iter=%d, burn=[%d,%d], post=[%d,%d], remaining_total_iter=%d"
        % (
            run_state.num_completed_epochs + 1,
            target_num_epochs,
            epoch_max_num_iter,
            min_num_burn_in_for_epoch,
            max_num_burn_in_for_epoch,
            min_num_post_burn_in_for_epoch,
            max_num_post_burn_in_for_epoch,
            run_state.remaining_total_iter,
        ),
        INFO,
    )

    epoch_total_iter_offset = total_num_iter - run_state.remaining_total_iter
    return {
        "epoch_max_num_iter": epoch_max_num_iter,
        "min_num_burn_in_for_epoch": min_num_burn_in_for_epoch,
        "max_num_burn_in_for_epoch": max_num_burn_in_for_epoch,
        "min_num_post_burn_in_for_epoch": min_num_post_burn_in_for_epoch,
        "max_num_post_burn_in_for_epoch": max_num_post_burn_in_for_epoch,
        "trace_chain_offset": trace_chain_offset,
        "p_scale_factor": p_scale_factor,
        "min_num_iter_for_epoch": min_num_iter_for_epoch,
        "increase_hyper_if_betas_below_for_epoch": increase_hyper_if_betas_below_for_epoch,
        "experimental_hyper_mutation": epoch_phase_config.experimental_hyper_mutation,
        "num_before_checking_p_increase": num_before_checking_p_increase,
        "epoch_total_iter_offset": epoch_total_iter_offset,
    }


def _trim_stall_histories(
    post_stall_best_beta_rhat_history,
    post_stall_best_D_mcse_history,
    post_stall_snapshots,
    max_stall_history_len,
):
    if len(post_stall_best_beta_rhat_history) > max_stall_history_len:
        del post_stall_best_beta_rhat_history[:-max_stall_history_len]
        del post_stall_best_D_mcse_history[:-max_stall_history_len]
    if len(post_stall_snapshots) > max_stall_history_len:
        del post_stall_snapshots[:-max_stall_history_len]


def _evaluate_post_stall_status(
    num_post_burn_beta,
    num_post_burn_D,
    post_stall_beta_sum_m,
    post_stall_beta_sum2_m,
    post_stall_D_sum_m,
    post_stall_beta_indices,
    post_stall_gene_indices,
    beta_rhat_q_post,
    D_mcse_q,
    stop_mcse_quantile,
    stall_window,
    stall_min_post_burn_in,
    min_num_post_burn_in_for_epoch,
    stall_delta_rhat,
    stall_delta_mcse,
    stall_recent_window,
    stall_recent_eps,
    post_stall_best_beta_rhat_history,
    post_stall_best_D_mcse_history,
    post_stall_snapshots,
    num_chains,
):
    post_stall_plateau = False
    post_stall_recent_worse = False
    post_stall_recent_beta_rhat_q = np.nan
    post_stall_recent_D_mcse_q = np.nan
    min_post_burn_for_stall = max(stall_min_post_burn_in, min_num_post_burn_in_for_epoch)

    if stall_window > 0 and num_post_burn_D >= min_post_burn_for_stall:
        if len(post_stall_best_beta_rhat_history) >= stall_window:
            post_rhat_improvement = post_stall_best_beta_rhat_history[-stall_window] - post_stall_best_beta_rhat_history[-1]
            post_mcse_improvement = post_stall_best_D_mcse_history[-stall_window] - post_stall_best_D_mcse_history[-1]
            if post_rhat_improvement < stall_delta_rhat and post_mcse_improvement < stall_delta_mcse:
                post_stall_plateau = True

        (
            post_stall_recent_beta_rhat_q,
            post_stall_recent_D_mcse_q,
        ) = _compute_recent_post_stall_metrics(
            num_post_burn_beta=num_post_burn_beta,
            num_post_burn_D=num_post_burn_D,
            post_stall_beta_sum_m=post_stall_beta_sum_m,
            post_stall_beta_sum2_m=post_stall_beta_sum2_m,
            post_stall_D_sum_m=post_stall_D_sum_m,
            post_stall_beta_indices=post_stall_beta_indices,
            post_stall_gene_indices=post_stall_gene_indices,
            post_stall_snapshots=post_stall_snapshots,
            stall_recent_window=stall_recent_window,
            stop_mcse_quantile=stop_mcse_quantile,
            num_chains=num_chains,
        )

        if np.isfinite(post_stall_recent_beta_rhat_q) and post_stall_recent_beta_rhat_q > beta_rhat_q_post * (1 + stall_recent_eps):
            post_stall_recent_worse = True
        if np.isfinite(post_stall_recent_D_mcse_q) and post_stall_recent_D_mcse_q > D_mcse_q * (1 + stall_recent_eps):
            post_stall_recent_worse = True

    return (
        post_stall_plateau,
        post_stall_recent_worse,
        post_stall_recent_beta_rhat_q,
        post_stall_recent_D_mcse_q,
    )


def _compute_recent_post_stall_metrics(
    num_post_burn_beta,
    num_post_burn_D,
    post_stall_beta_sum_m,
    post_stall_beta_sum2_m,
    post_stall_D_sum_m,
    post_stall_beta_indices,
    post_stall_gene_indices,
    post_stall_snapshots,
    stall_recent_window,
    stop_mcse_quantile,
    num_chains,
):
    recent_beta_rhat_q = np.nan
    recent_D_mcse_q = np.nan
    if not (stall_recent_window > 0 and len(post_stall_snapshots) >= stall_recent_window + 1):
        return (recent_beta_rhat_q, recent_D_mcse_q)

    old_beta_n, old_beta_sum_m, old_beta_sum2_m, old_D_n, old_D_sum_m = post_stall_snapshots[-(stall_recent_window + 1)]
    recent_beta_n = num_post_burn_beta - old_beta_n
    recent_D_n = num_post_burn_D - old_D_n

    if recent_beta_n > 1 and post_stall_beta_indices.size > 0:
        recent_beta_sum_m = post_stall_beta_sum_m - old_beta_sum_m
        recent_beta_sum2_m = post_stall_beta_sum2_m - old_beta_sum2_m
        _, _, recent_R_beta_v, _ = _calculate_rhat_from_sums(recent_beta_sum_m, recent_beta_sum2_m, recent_beta_n)
        recent_rhat_candidates = recent_R_beta_v[np.logical_and(np.isfinite(recent_R_beta_v), recent_R_beta_v >= 1)]
        recent_beta_rhat_q = _safe_quantile(recent_rhat_candidates, stop_mcse_quantile, 1.0)

    if recent_D_n > 1 and post_stall_gene_indices.size > 0:
        recent_D_sum_m = post_stall_D_sum_m - old_D_sum_m
        recent_D_chain_means_m = recent_D_sum_m / float(recent_D_n)
        recent_D_mcse_v = np.sqrt(np.var(recent_D_chain_means_m, axis=0, ddof=1) / float(num_chains))
        recent_D_mcse_q = _safe_quantile(recent_D_mcse_v, stop_mcse_quantile, np.inf)

    return (recent_beta_rhat_q, recent_D_mcse_q)


def _zero_arrays(*arrays):
    for arr in arrays:
        if arr is not None:
            arr[:] = 0


def _reset_gibbs_post_burn_accumulators(post_burn_reset_arrays, post_burn_reset_missing_arrays):
    _zero_arrays(*(post_burn_reset_arrays + post_burn_reset_missing_arrays))


def _end_gibbs_burn_in(
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_pass_streak,
    stop_pass_streak,
    reset_burn_in_pass_streak=False,
):
    in_burn_in = False
    if reset_burn_in_pass_streak:
        burn_in_pass_streak = 0
    stop_pass_streak = 0
    _reset_gibbs_post_burn_accumulators(post_burn_reset_arrays, post_burn_reset_missing_arrays)
    return (in_burn_in, burn_in_pass_streak, stop_pass_streak)


def _evaluate_burn_in_diagnostics(
    epoch_control,
    burn_in_config,
    min_num_burn_in_for_epoch,
    epoch_runtime,
    num_samples,
):
    all_sum_betas_m = epoch_runtime["all_sum_betas_m"]
    all_sum_betas2_m = epoch_runtime["all_sum_betas2_m"]
    all_num_sum_m = epoch_runtime["all_num_sum_m"]
    burn_in_pass_streak = epoch_control["burn_in_pass_streak"]
    burn_in_rhat_history = epoch_control["burn_in_rhat_history"]
    burn_stall_best_beta_rhat_history = epoch_control["burn_stall_best_beta_rhat_history"]
    burn_stall_snapshots = epoch_control["burn_stall_snapshots"]
    burn_stall_beta_indices = epoch_control["burn_stall_beta_indices"]

    active_beta_top_k = burn_in_config["active_beta_top_k"]
    active_beta_min_abs = burn_in_config["active_beta_min_abs"]
    burn_in_rhat_quantile = burn_in_config["burn_in_rhat_quantile"]
    r_threshold_burn_in = burn_in_config["r_threshold_burn_in"]
    stall_window = burn_in_config["stall_window"]
    stall_min_burn_in = burn_in_config["stall_min_burn_in"]
    stall_delta_rhat = burn_in_config["stall_delta_rhat"]
    stall_recent_window = burn_in_config["stall_recent_window"]
    stall_recent_eps = burn_in_config["stall_recent_eps"]
    burn_in_stall_window = burn_in_config["burn_in_stall_window"]
    burn_in_stall_delta = burn_in_config["burn_in_stall_delta"]
    (
        R_beta_v,
        active_beta_mask_v,
        num_active_betas,
        beta_rhat_q,
        beta_rhat_max,
    ) = _compute_burn_in_active_beta_rhat_stats(
        all_sum_betas_m=all_sum_betas_m,
        all_sum_betas2_m=all_sum_betas2_m,
        all_num_sum_m=all_num_sum_m,
        num_samples=num_samples,
        active_beta_top_k=active_beta_top_k,
        active_beta_min_abs=active_beta_min_abs,
        burn_in_rhat_quantile=burn_in_rhat_quantile,
    )

    if beta_rhat_q <= r_threshold_burn_in:
        burn_in_pass_streak += 1
    else:
        burn_in_pass_streak = 0

    burn_in_rhat_history.append(beta_rhat_q)
    burn_best_beta_rhat_q = beta_rhat_q if len(burn_stall_best_beta_rhat_history) == 0 else min(burn_stall_best_beta_rhat_history[-1], beta_rhat_q)
    burn_stall_best_beta_rhat_history.append(burn_best_beta_rhat_q)

    if burn_stall_beta_indices is None:
        burn_stall_beta_indices = _prepare_stall_indices(active_beta_mask_v, np.mean(_means_from_sums(all_sum_betas_m, all_num_sum_m), axis=0), active_beta_top_k)

    if burn_stall_beta_indices.size > 0:
        burn_stall_sum_m = copy.copy(all_sum_betas_m[:,burn_stall_beta_indices])
        burn_stall_sum2_m = copy.copy(all_sum_betas2_m[:,burn_stall_beta_indices])
    else:
        burn_stall_sum_m = np.zeros((all_sum_betas_m.shape[0], 0))
        burn_stall_sum2_m = np.zeros((all_sum_betas2_m.shape[0], 0))
    burn_stall_snapshots.append((num_samples, burn_stall_sum_m, burn_stall_sum2_m, beta_rhat_q))

    max_stall_history_len = max(stall_window + 2, stall_recent_window + 2, 10)
    if len(burn_stall_best_beta_rhat_history) > max_stall_history_len:
        del burn_stall_best_beta_rhat_history[:-max_stall_history_len]
    if len(burn_stall_snapshots) > max_stall_history_len:
        del burn_stall_snapshots[:-max_stall_history_len]

    burn_stall_plateau = False
    burn_stall_recent_worse = False
    min_samples_for_burn_stall = max(min_num_burn_in_for_epoch, stall_min_burn_in)
    if stall_window > 0 and num_samples >= min_samples_for_burn_stall:
        if len(burn_stall_best_beta_rhat_history) >= stall_window:
            burn_rhat_improvement = burn_stall_best_beta_rhat_history[-stall_window] - burn_stall_best_beta_rhat_history[-1]
            if burn_rhat_improvement < stall_delta_rhat:
                burn_stall_plateau = True

        burn_stall_recent_beta_rhat_q = _compute_recent_burn_in_stall_beta_rhat(
            num_samples=num_samples,
            burn_stall_sum_m=burn_stall_sum_m,
            burn_stall_sum2_m=burn_stall_sum2_m,
            burn_stall_beta_indices=burn_stall_beta_indices,
            burn_stall_snapshots=burn_stall_snapshots,
            stall_recent_window=stall_recent_window,
            burn_in_rhat_quantile=burn_in_rhat_quantile,
        )
        if np.isfinite(burn_stall_recent_beta_rhat_q) and burn_stall_recent_beta_rhat_q > beta_rhat_q * (1 + stall_recent_eps):
            burn_stall_recent_worse = True

    burn_stall_detected = burn_stall_plateau or burn_stall_recent_worse
    burn_window_plateau_detected, burn_window_span = _compute_burn_in_window_plateau_status(
        burn_in_rhat_history=burn_in_rhat_history,
        burn_in_stall_window=burn_in_stall_window,
        burn_in_stall_delta=burn_in_stall_delta,
    )

    return {
        "R_beta_v": R_beta_v,
        "burn_in_pass_streak": burn_in_pass_streak,
        "burn_stall_beta_indices": burn_stall_beta_indices,
        "beta_rhat_q": beta_rhat_q,
        "beta_rhat_max": beta_rhat_max,
        "num_active_betas": num_active_betas,
        "burn_stall_plateau": burn_stall_plateau,
        "burn_stall_recent_worse": burn_stall_recent_worse,
        "burn_stall_detected": burn_stall_detected,
        "burn_window_plateau_detected": burn_window_plateau_detected,
        "burn_window_span": burn_window_span,
    }


def _compute_recent_burn_in_stall_beta_rhat(
    num_samples,
    burn_stall_sum_m,
    burn_stall_sum2_m,
    burn_stall_beta_indices,
    burn_stall_snapshots,
    stall_recent_window,
    burn_in_rhat_quantile,
):
    if not (stall_recent_window > 0 and len(burn_stall_snapshots) >= stall_recent_window + 1 and burn_stall_beta_indices.size > 0):
        return np.nan
    old_num_samples, old_sum_m, old_sum2_m, _ = burn_stall_snapshots[-(stall_recent_window + 1)]
    recent_num_samples = num_samples - old_num_samples
    if recent_num_samples <= 1:
        return np.nan
    recent_sum_m = burn_stall_sum_m - old_sum_m
    recent_sum2_m = burn_stall_sum2_m - old_sum2_m
    _, _, burn_recent_R_beta_v, _ = _calculate_rhat_from_sums(recent_sum_m, recent_sum2_m, recent_num_samples)
    burn_recent_candidates = burn_recent_R_beta_v[np.logical_and(np.isfinite(burn_recent_R_beta_v), burn_recent_R_beta_v >= 1)]
    return _safe_quantile(burn_recent_candidates, burn_in_rhat_quantile, 1.0)


def _compute_burn_in_window_plateau_status(
    burn_in_rhat_history,
    burn_in_stall_window,
    burn_in_stall_delta,
):
    if not (burn_in_stall_window > 0 and len(burn_in_rhat_history) >= burn_in_stall_window):
        return (False, np.nan)
    burn_window_v = np.array(burn_in_rhat_history[-burn_in_stall_window:], dtype=float)
    if not np.all(np.isfinite(burn_window_v)):
        return (False, np.nan)
    burn_window_span = np.max(burn_window_v) - np.min(burn_window_v)
    return (burn_window_span < burn_in_stall_delta, burn_window_span)


def _compute_burn_in_active_beta_rhat_stats(
    all_sum_betas_m,
    all_sum_betas2_m,
    all_num_sum_m,
    num_samples,
    active_beta_top_k,
    active_beta_min_abs,
    burn_in_rhat_quantile,
):
    (_, _, R_beta_v, _) = _calculate_rhat_from_sums(all_sum_betas_m, all_sum_betas2_m, num_samples)
    active_beta_mask_v, _, _ = _get_active_beta_mask(all_sum_betas_m, all_num_sum_m, active_beta_top_k, active_beta_min_abs)
    num_active_betas = int(np.sum(active_beta_mask_v))
    if num_active_betas > 0:
        R_beta_active_v = R_beta_v[active_beta_mask_v]
        rhat_candidates = R_beta_active_v[np.logical_and(np.isfinite(R_beta_active_v), R_beta_active_v >= 1)]
        beta_rhat_q = _safe_quantile(rhat_candidates, burn_in_rhat_quantile, 1.0)
        beta_rhat_max = _safe_quantile(R_beta_active_v, 1.0, 1.0)
    else:
        beta_rhat_q = 1.0
        beta_rhat_max = 1.0
    return (R_beta_v, active_beta_mask_v, num_active_betas, beta_rhat_q, beta_rhat_max)


def _handle_gibbs_burn_in_max_iter(
    num_samples,
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_pass_streak,
    stop_pass_streak,
):
    in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
        post_burn_reset_arrays=post_burn_reset_arrays,
        post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
        burn_in_pass_streak=burn_in_pass_streak,
        stop_pass_streak=stop_pass_streak,
        reset_burn_in_pass_streak=True,
    )
    log("Stopping Gibbs burn in after %d iterations (per-epoch max burn-in reached)" % (num_samples), INFO)
    return (in_burn_in, burn_in_pass_streak, stop_pass_streak)


def _handle_gibbs_burn_in_gauss_seidel_path(
    num_samples,
    epoch_total_iter_offset,
    min_num_iter_for_epoch,
    eps,
    Y_sample_m,
    prev_Ys_m,
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_pass_streak,
    stop_pass_streak,
):
    in_burn_in = True
    if prev_Ys_m is not None:
        sum_diff = np.sum(np.abs(prev_Ys_m - Y_sample_m))
        sum_prev = np.sum(np.abs(prev_Ys_m))
        max_diff_frac = np.max(np.abs((prev_Ys_m - Y_sample_m) / prev_Ys_m))

        tot_diff = sum_diff / sum_prev
        log(
            "Gibbs iteration %d (global %d): mean gauss seidel difference = %.4g / %.4g = %.4g; max frac difference = %.4g"
            % (num_samples, epoch_total_iter_offset + num_samples, sum_diff, sum_prev, tot_diff, max_diff_frac)
        )
        if num_samples > min_num_iter_for_epoch and tot_diff < eps:
            in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
                post_burn_reset_arrays=post_burn_reset_arrays,
                post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
                burn_in_pass_streak=burn_in_pass_streak,
                stop_pass_streak=stop_pass_streak,
                reset_burn_in_pass_streak=True,
            )
            log("Gibbs gauss converged after %d iterations" % num_samples, INFO)
    prev_Ys_m = Y_sample_m
    return (in_burn_in, burn_in_pass_streak, stop_pass_streak, prev_Ys_m)


def _handle_gibbs_burn_in_diag_path(
    num_samples,
    epoch_control,
    burn_in_config,
    min_num_burn_in_for_epoch,
    epoch_runtime,
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_pass_streak,
    stop_pass_streak,
):
    in_burn_in = True
    burn_diag = _evaluate_burn_in_diagnostics(
        epoch_control=epoch_control,
        burn_in_config=burn_in_config,
        min_num_burn_in_for_epoch=min_num_burn_in_for_epoch,
        epoch_runtime=epoch_runtime,
        num_samples=num_samples,
    )
    R_beta_v = burn_diag["R_beta_v"]
    burn_in_pass_streak = burn_diag["burn_in_pass_streak"]
    burn_stall_beta_indices = burn_diag["burn_stall_beta_indices"]
    beta_rhat_q = burn_diag["beta_rhat_q"]
    beta_rhat_max = burn_diag["beta_rhat_max"]
    num_active_betas = burn_diag["num_active_betas"]
    burn_stall_plateau = burn_diag["burn_stall_plateau"]
    burn_stall_recent_worse = burn_diag["burn_stall_recent_worse"]
    burn_stall_detected = burn_diag["burn_stall_detected"]
    burn_window_plateau_detected = burn_diag["burn_window_plateau_detected"]
    burn_window_span = burn_diag["burn_window_span"]

    burn_in_rhat_quantile = burn_in_config["burn_in_rhat_quantile"]
    num_full_gene_sets = burn_in_config["num_full_gene_sets"]
    burn_in_patience = burn_in_config["burn_in_patience"]
    stop_patience = burn_in_config["stop_patience"]
    r_threshold_burn_in = burn_in_config["r_threshold_burn_in"]
    burn_in_stall_window = burn_in_config["burn_in_stall_window"]
    burn_in_stall_delta = burn_in_config["burn_in_stall_delta"]

    log(
        "Gibbs burn-in iter %d: beta_Rhat_q(%.2f)=%.4g; beta_Rhat_max=%.4g; active_betas=%d/%d; burn_streak=%d/%d; stop_streak=%d/%d"
        % (
            num_samples,
            burn_in_rhat_quantile,
            beta_rhat_q,
            beta_rhat_max,
            num_active_betas,
            num_full_gene_sets,
            burn_in_pass_streak,
            burn_in_patience,
            stop_pass_streak,
            stop_patience,
        ),
        INFO,
    )
    if burn_in_pass_streak >= burn_in_patience:
        in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
        )
        log(
            "Burn-in complete at iter %d (beta R-hat q=%.2f on active betas, threshold=%.4g, patience=%d)"
            % (num_samples, burn_in_rhat_quantile, r_threshold_burn_in, burn_in_patience),
            INFO,
        )
    elif burn_stall_detected:
        in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
            reset_burn_in_pass_streak=True,
        )
        log(
            "Stopping Gibbs burn in at iter %d due to stall detectors (plateau=%s, recent_worse=%s)"
            % (num_samples, str(burn_stall_plateau), str(burn_stall_recent_worse)),
            INFO,
        )
    elif burn_window_plateau_detected:
        in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
            reset_burn_in_pass_streak=True,
        )
        log(
            "Stopping Gibbs burn in at iter %d due to R-hat plateau (window=%d, span=%.4g < delta=%.4g)"
            % (num_samples, burn_in_stall_window, burn_window_span, burn_in_stall_delta),
            INFO,
        )
    return (in_burn_in, burn_in_pass_streak, stop_pass_streak, burn_stall_beta_indices, R_beta_v)


def _update_gibbs_burn_in_state(
    epoch_control,
    iteration_num,
    epoch_total_iter_offset,
    epoch_max_num_iter,
    max_num_burn_in_for_epoch,
    min_num_iter_for_epoch,
    min_num_burn_in_for_epoch,
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_config,
    iter_state,
    epoch_runtime,
):
    in_burn_in = epoch_control["in_burn_in"]
    prev_Ys_m = epoch_control["prev_Ys_m"]
    burn_in_pass_streak = epoch_control["burn_in_pass_streak"]
    burn_stall_beta_indices = epoch_control["burn_stall_beta_indices"]
    stop_pass_streak = epoch_control["stop_pass_streak"]
    R_beta_v = epoch_control["R_beta_v"]

    gauss_seidel = burn_in_config["gauss_seidel"]
    eps = burn_in_config["eps"]
    diag_every = burn_in_config["diag_every"]

    Y_sample_m = iter_state["Y_sample_m"]

    if not in_burn_in:
        return _build_gibbs_burn_in_control_from_epoch(epoch_control)

    num_samples = iteration_num + 1
    if num_samples >= max_num_burn_in_for_epoch:
        in_burn_in, burn_in_pass_streak, stop_pass_streak = _handle_gibbs_burn_in_max_iter(
            num_samples=num_samples,
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
        )
    elif gauss_seidel:
        in_burn_in, burn_in_pass_streak, stop_pass_streak, prev_Ys_m = _handle_gibbs_burn_in_gauss_seidel_path(
            num_samples=num_samples,
            epoch_total_iter_offset=epoch_total_iter_offset,
            min_num_iter_for_epoch=min_num_iter_for_epoch,
            eps=eps,
            Y_sample_m=Y_sample_m,
            prev_Ys_m=prev_Ys_m,
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
        )
    elif num_samples >= min_num_burn_in_for_epoch and (num_samples % diag_every == 0 or num_samples == epoch_max_num_iter):
        in_burn_in, burn_in_pass_streak, stop_pass_streak, burn_stall_beta_indices, R_beta_v = _handle_gibbs_burn_in_diag_path(
            num_samples=num_samples,
            epoch_control=epoch_control,
            burn_in_config=burn_in_config,
            min_num_burn_in_for_epoch=min_num_burn_in_for_epoch,
            epoch_runtime=epoch_runtime,
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
        )

    return _build_gibbs_burn_in_control_update(
        in_burn_in=in_burn_in,
        burn_in_pass_streak=burn_in_pass_streak,
        stop_pass_streak=stop_pass_streak,
        prev_Ys_m=prev_Ys_m,
        burn_stall_beta_indices=burn_stall_beta_indices,
        R_beta_v=R_beta_v,
    )


def _record_gibbs_hyper_mutation_event(state, event):
    event_json = json.dumps(event, sort_keys=True)
    state._record_param("gibbs_hyper_mutation_event", event_json)

    prev_count = state.params.get("gibbs_hyper_mutation_event_count", 0)
    if isinstance(prev_count, list):
        prev_count = prev_count[-1]
    try:
        prev_count = int(prev_count)
    except (TypeError, ValueError):
        prev_count = 0
    state._record_param("gibbs_hyper_mutation_event_count", prev_count + 1, overwrite=True)
    log("Gibbs hyper mutation event: %s" % event_json, INFO)


def _maybe_restart_gibbs_for_low_betas(
    state,
    increase_hyper_if_betas_below_for_epoch,
    experimental_hyper_mutation,
    num_before_checking_p_increase,
    p_scale_factor,
    epoch_runtime,
    epoch_sums,
    num_mad,
    num_attempts,
    max_num_attempt_restarts,
    iteration_num,
):
    all_sum_betas_m = epoch_runtime["all_sum_betas_m"]
    all_num_sum_m = epoch_runtime["all_num_sum_m"]
    num_p_increases = epoch_runtime["num_p_increases"]

    num_sum_beta_m = epoch_sums["num_sum_beta_m"]
    sum_betas_m = epoch_sums["sum_betas_m"]

    gibbs_good = True
    should_break = False

    if increase_hyper_if_betas_below_for_epoch is None:
        return GibbsLowBetaRestartUpdate(
            gibbs_good=gibbs_good,
            num_p_increases=num_p_increases,
            should_break=should_break,
        )

    # Check to make sure that we satisfy the hyperparameter growth criteria.
    if np.any(all_num_sum_m == 0):
        gibbs_good = False
        return GibbsLowBetaRestartUpdate(
            gibbs_good=gibbs_good,
            num_p_increases=num_p_increases,
            should_break=should_break,
        )

    # Check both sums over all iterations and post-burn aggregates.
    _outlier_resistant_mean(all_sum_betas_m, all_num_sum_m, num_mad, record_param_fn=state._record_param)

    fraction_required = 0.001
    state._record_param("fraction_required_to_not_increase_hyper", fraction_required)

    (
        has_post_burn_beta_samples,
        cur_avg_betas_v,
        all_low,
    ) = _evaluate_gibbs_low_beta_condition(
        state=state,
        sum_betas_m=sum_betas_m,
        num_sum_beta_m=num_sum_beta_m,
        num_mad=num_mad,
        increase_hyper_if_betas_below_for_epoch=increase_hyper_if_betas_below_for_epoch,
        fraction_required=fraction_required,
    )

    if has_post_burn_beta_samples:
        top_gene_set = np.argmax(np.mean(sum_betas_m / num_sum_beta_m, axis=0) / state.scale_factors)
        log("Top gene set %s has value %.3g" % (state.gene_sets[top_gene_set], (np.mean(sum_betas_m / num_sum_beta_m, axis=0) / state.scale_factors)[top_gene_set]), TRACE)
        top_gene_set2 = np.argmax(cur_avg_betas_v / state.scale_factors)
        log("Top gene set %s has outlier value %.3g" % (state.gene_sets[top_gene_set2], (cur_avg_betas_v / state.scale_factors)[top_gene_set2]), TRACE)

    if all_low:
        fraction_above_threshold = float(
            np.mean(cur_avg_betas_v / state.scale_factors > increase_hyper_if_betas_below_for_epoch)
        )
        log(
            "Only %.3g of %d (%.3g) are above %.3g"
            % (
                np.sum(cur_avg_betas_v / state.scale_factors > increase_hyper_if_betas_below_for_epoch),
                len(cur_avg_betas_v),
                fraction_above_threshold,
                increase_hyper_if_betas_below_for_epoch,
            )
        )

        # At minimum, guarantee that it will restart unless it gets above this.
        gibbs_good = False
        if not experimental_hyper_mutation:
            state._record_param("gibbs_no_signal_detected", 1, overwrite=True)
            state._record_param(
                "gibbs_no_signal_fraction_above_threshold",
                fraction_above_threshold,
                overwrite=True,
            )
            state._record_param(
                "gibbs_no_signal_threshold",
                increase_hyper_if_betas_below_for_epoch,
                overwrite=True,
            )
            bail(
                "Detected no-signal Gibbs condition (fraction above threshold %.6g < %.6g at threshold %.6g). "
                "Default behavior is explicit failure without hyper mutation. "
                "To enable legacy restart/mutation heuristic, pass --experimental-hyper-mutation with --experimental-increase-hyper-if-betas-below."
                % (
                    fraction_above_threshold,
                    fraction_required,
                    increase_hyper_if_betas_below_for_epoch,
                )
            )
        # Only if above num for checking though that we increase and restart.
        num_p_increases, should_break = _maybe_increase_gibbs_hyper_and_restart(
            state=state,
            increase_hyper_if_betas_below_for_epoch=increase_hyper_if_betas_below_for_epoch,
            fraction_above_threshold=fraction_above_threshold,
            num_before_checking_p_increase=num_before_checking_p_increase,
            p_scale_factor=p_scale_factor,
            num_p_increases=num_p_increases,
            num_attempts=num_attempts,
            max_num_attempt_restarts=max_num_attempt_restarts,
            iteration_num=iteration_num,
        )
    else:
        gibbs_good = True

    return GibbsLowBetaRestartUpdate(
        gibbs_good=gibbs_good,
        num_p_increases=num_p_increases,
        should_break=should_break,
    )


def _evaluate_gibbs_low_beta_condition(
    state,
    sum_betas_m,
    num_sum_beta_m,
    num_mad,
    increase_hyper_if_betas_below_for_epoch,
    fraction_required,
):
    has_post_burn_beta_samples = np.all(num_sum_beta_m > 0)
    cur_avg_betas_v = None
    all_low = False
    if has_post_burn_beta_samples:
        _, cur_avg_betas_v = _outlier_resistant_mean(
            sum_betas_m,
            num_sum_beta_m,
            num_mad,
            record_param_fn=state._record_param,
        )
        all_low = (
            np.mean(cur_avg_betas_v / state.scale_factors > increase_hyper_if_betas_below_for_epoch)
            < fraction_required
        )
    return (has_post_burn_beta_samples, cur_avg_betas_v, all_low)


def _maybe_increase_gibbs_hyper_and_restart(
    state,
    increase_hyper_if_betas_below_for_epoch,
    fraction_above_threshold,
    num_before_checking_p_increase,
    p_scale_factor,
    num_p_increases,
    num_attempts,
    max_num_attempt_restarts,
    iteration_num,
):
    should_break = False
    if iteration_num <= num_before_checking_p_increase:
        return (num_p_increases, should_break)

    old_p = state.p
    old_sigma2 = state.sigma2
    new_p = old_p
    new_sigma2 = old_sigma2

    state._record_param("p_scale_factor", p_scale_factor)

    new_p = state.p * p_scale_factor
    num_p_increases += 1
    if new_p > 1:
        new_p = 1

    break_loop = False
    if new_p != state.p and num_attempts < max_num_attempt_restarts:
        # Update so that new_sigma2 / new_p = self.sigma2 / self.p
        new_sigma2 = state.sigma2 * new_p / state.p

        state.ps *= new_p / state.p
        state.set_p(new_p)
        state._record_param("p_adj", new_p)
        log(
            "Detected all gene set betas below %.3g; increasing p to %.3g and restarting gibbs"
            % (increase_hyper_if_betas_below_for_epoch, state.p)
        )

        # Restart.
        break_loop = True
    if new_sigma2 != state.sigma2 and num_attempts < max_num_attempt_restarts:
        state.sigma2s *= new_sigma2 / state.sigma2
        state._record_param("sigma2_adj", new_sigma2)
        state.set_sigma(new_sigma2, state.sigma_power)
        log(
            "Detected all gene set betas below %.3g; increasing sigma to %.3g and restarting gibbs"
            % (increase_hyper_if_betas_below_for_epoch, state.sigma2)
        )
        break_loop = True
    if break_loop:
        _record_gibbs_hyper_mutation_event(
            state,
            {
                "event": "gibbs_hyper_mutation_restart",
                "trigger": "all_betas_below_threshold",
                "threshold": float(increase_hyper_if_betas_below_for_epoch),
                "fraction_above_threshold": float(fraction_above_threshold),
                "iteration_num": int(iteration_num),
                "num_attempts": int(num_attempts),
                "max_num_attempt_restarts": int(max_num_attempt_restarts),
                "p_scale_factor": float(p_scale_factor),
                "old_p": float(old_p),
                "new_p": float(state.p),
                "old_sigma2": float(old_sigma2),
                "new_sigma2": float(state.sigma2),
            },
        )
        should_break = True
    return (num_p_increases, should_break)


def _safe_quantile(values, q, fallback):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return fallback
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(finite_values, q))


def _prepare_stall_indices(mask_v, fallback_scores_v, fallback_k):
    if mask_v is not None and np.any(mask_v):
        return np.where(mask_v)[0]
    if fallback_scores_v is None or len(fallback_scores_v) == 0:
        return np.array([], dtype=int)
    k = min(max(fallback_k, 1), len(fallback_scores_v))
    if k >= len(fallback_scores_v):
        return np.arange(len(fallback_scores_v))
    return np.argpartition(-np.abs(fallback_scores_v), k - 1)[:k]


def _means_from_sums(sum_m, num_sum_m):
    return np.divide(sum_m, np.maximum(num_sum_m, 1.0))


def _apply_inner_beta_sparsity_update(
    sparse_solution,
    sparse_frac_betas,
    curr_postp_t,
    ps_m,
    curr_post_means_t,
    curr_betas_t,
    compute_mask_v,
):
    if not sparse_solution:
        return

    sparse_mask_t = curr_postp_t < ps_m
    if sparse_frac_betas is not None:
        # Zero out very small values relative to the top within each chain/parallel slice.
        relative_value = np.max(np.abs(curr_post_means_t), axis=2)
        sparse_mask_t = np.logical_or(
            sparse_mask_t,
            (np.abs(curr_post_means_t).T < sparse_frac_betas * relative_value.T).T,
        )

    # Do not sparsify slices that are currently outside the compute mask.
    sparse_mask_t[:, np.logical_not(compute_mask_v), :] = False
    log(
        "Setting %d entries to zero due to sparsity"
        % np.sum(np.logical_and(sparse_mask_t, curr_betas_t > 0)),
        TRACE,
    )
    curr_betas_t[sparse_mask_t] = 0
    curr_post_means_t[sparse_mask_t] = 0


def _write_inner_beta_trace_rows(
    betas_trace_fh,
    iteration_num,
    num_parallel,
    num_chains,
    num_gene_sets,
    betas_trace_gene_sets,
    curr_post_means_t,
    curr_betas_t,
    curr_postp_t,
    res_beta_hat_t,
    scale_factors_m,
    beta_tildes_m,
    ses_m,
    sigma2_m,
    ps_m,
    R_m,
    beta_weights_m,
    sem2_m,
):
    for parallel_num in range(num_parallel):
        for chain_num in range(num_chains):
            for gene_set_idx in range(num_gene_sets):
                gene_set = gene_set_idx
                if betas_trace_gene_sets is not None and len(betas_trace_gene_sets) == num_gene_sets:
                    gene_set = betas_trace_gene_sets[gene_set_idx]

                betas_trace_fh.write(
                    "%d\t%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n"
                    % (
                        iteration_num,
                        parallel_num + 1,
                        chain_num + 1,
                        gene_set,
                        curr_post_means_t[chain_num, parallel_num, gene_set_idx]
                        / scale_factors_m[parallel_num, gene_set_idx],
                        curr_betas_t[chain_num, parallel_num, gene_set_idx]
                        / scale_factors_m[parallel_num, gene_set_idx],
                        curr_postp_t[chain_num, parallel_num, gene_set_idx],
                        res_beta_hat_t[chain_num, parallel_num, gene_set_idx]
                        / scale_factors_m[parallel_num, gene_set_idx],
                        beta_tildes_m[parallel_num, gene_set_idx]
                        / scale_factors_m[parallel_num, gene_set_idx],
                        curr_betas_t[chain_num, parallel_num, gene_set_idx],
                        res_beta_hat_t[chain_num, parallel_num, gene_set_idx],
                        beta_tildes_m[parallel_num, gene_set_idx],
                        ses_m[parallel_num, gene_set_idx],
                        sigma2_m[parallel_num, gene_set_idx]
                        if len(np.shape(sigma2_m)) > 0
                        else sigma2_m,
                        ps_m[parallel_num, gene_set_idx] if len(np.shape(ps_m)) > 0 else ps_m,
                        R_m[parallel_num, gene_set_idx],
                        R_m[parallel_num, gene_set_idx]
                        * beta_weights_m[parallel_num, gene_set_idx],
                        sem2_m[parallel_num, gene_set_idx],
                    )
                )

    betas_trace_fh.flush()


def _compute_inner_beta_hyper_update_targets(
    curr_postp_t,
    res_beta_hat_t,
    hdmp_hdmpn_m,
    se2s_m,
    curr_betas_m,
    V_diag_m,
    num_parallel,
    use_X,
    multiple_V,
    V,
    sparse_V,
    num_p_pseudo,
    curr_postp_m,
    sigma_power,
    scale_factors_m,
    num_gene_sets,
    num_missing_gene_sets,
    p_noninf_inflate,
):
    # Hyper-updates use Rao-Blackwellized moments to avoid sigma collapse.
    # Conditional slab mean m = hdmp_hdmpn * res_beta_hat.
    cond_mean_t = hdmp_hdmpn_m[np.newaxis, :, :] * res_beta_hat_t
    # Conditional slab variance v = hdmp_hdmpn * se2.
    cond_var_m = hdmp_hdmpn_m * se2s_m
    # E[beta^2] = postp * (m^2 + v).
    e_beta2_m = np.mean(curr_postp_t * (np.square(cond_mean_t) + cond_var_m[np.newaxis, :, :]), axis=0)
    # mu = E[beta].
    mu_m = curr_betas_m
    # Var(beta) = E[beta^2] - (E[beta])^2.
    var_m = e_beta2_m - np.square(mu_m)
    var_m[var_m < 0] = 0.0

    if V_diag_m is None:
        diag_m = np.ones(mu_m.shape)
    else:
        diag_m = V_diag_m

    h2 = 0.0
    for i in range(num_parallel):
        if use_X:
            # In X-implicit mode, use diagonal approximation directly from E[beta^2].
            h2 += float(np.sum(e_beta2_m[i, :]))
        else:
            if multiple_V:
                cur_V = V[i, :, :]
            else:
                cur_V = V

            mu_v = mu_m[i, :]
            if sparse_V:
                muVmu = float(mu_v.dot(cur_V.dot(mu_v)))
            else:
                muVmu = float(mu_v.dot(cur_V).dot(mu_v))
            h2 += muVmu + float(np.sum(diag_m[i, :] * var_m[i, :]))
    h2 /= float(num_parallel)

    # Rao-Blackwellized p update from posterior inclusion probabilities.
    if num_p_pseudo is not None and num_p_pseudo > 0:
        a0 = float(num_p_pseudo)
        b0 = float(num_p_pseudo)
        sum_r = float(np.sum(curr_postp_m))
        m_tot = float(curr_postp_m.size)
        new_p = (a0 + sum_r) / (a0 + b0 + m_tot)
    else:
        new_p = float(np.mean(curr_postp_m))

    if sigma_power is not None:
        new_sigma2 = h2 / np.mean(np.sum(np.power(scale_factors_m, sigma_power), axis=1))
    else:
        new_sigma2 = h2 / num_gene_sets

    if num_missing_gene_sets:
        missing_scale_factor = num_gene_sets / (num_gene_sets + num_missing_gene_sets)
        new_sigma2 *= missing_scale_factor
        new_p *= missing_scale_factor

    if p_noninf_inflate != 1:
        log("Inflating p by %.3g" % p_noninf_inflate, DEBUG)
        new_p *= p_noninf_inflate

    return (new_p, new_sigma2)


def _update_inner_beta_gene_set_batch(
    compute_mask_m,
    compute_mask_v,
    alpha_shrink,
    rand_ps_t,
    rand_norms_t,
    hdmp_hdmpn_m,
    c_const_m,
    d_const_m,
    hdmpn_m,
    se2s_m,
    norm_scale_m,
    assume_independent,
    beta_tildes_m,
    curr_betas_t,
    V,
    multiple_V,
    sparse_V,
    use_X,
    X_orig,
    scale_factors_m,
    mean_shifts_m,
    betas_trace_out,
    betas_trace_gene_sets,
    account_for_V_diag_m,
    V_diag_m,
    curr_postp_t,
    curr_post_means_t,
    gauss_seidel,
    res_beta_hat_t,
):
    # 1) Build residualized beta_tilde for the active batch.
    compute_mask_union = np.any(compute_mask_m, axis=0)
    compute_mask_union_filter_m = compute_mask_m[:, compute_mask_union]

    if assume_independent:
        res_beta_hat_t_flat = beta_tildes_m[compute_mask_m]
    else:
        current_num_parallel = sum(compute_mask_v)

        if multiple_V:
            # Pointwise matmul across active parallels while preserving chain dimension.
            res_beta_hat_union_t = np.einsum(
                "hij,ijk->hik",
                curr_betas_t[:, compute_mask_v, :],
                V[compute_mask_v, :, :][:, :, compute_mask_union],
            ).reshape((curr_betas_t.shape[0], current_num_parallel, np.sum(compute_mask_union)))
        elif sparse_V:
            res_beta_hat_union_t = V[compute_mask_union, :].dot(
                curr_betas_t[:, compute_mask_v, :].T.reshape(
                    (curr_betas_t.shape[2], np.sum(compute_mask_v) * curr_betas_t.shape[0])
                )
            ).reshape((np.sum(compute_mask_union), np.sum(compute_mask_v), curr_betas_t.shape[0])).T
        elif use_X:
            if len(compute_mask_union.shape) == 2:
                assert compute_mask_union.shape[0] == 1
                compute_mask_union = np.squeeze(compute_mask_union)

            curr_betas_filtered_t = curr_betas_t[:, compute_mask_v, :] / scale_factors_m[compute_mask_v, :]
            interm = X_orig.dot(
                curr_betas_filtered_t.T.reshape(
                    (curr_betas_filtered_t.shape[2], curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1])
                )
            ).reshape((X_orig.shape[0], curr_betas_filtered_t.shape[1], curr_betas_filtered_t.shape[0])) - np.sum(
                mean_shifts_m[compute_mask_v, :] * curr_betas_filtered_t, axis=2
            ).T

            # This path can be sensitive when some parallels converge earlier.
            res_beta_hat_union_t = (
                X_orig[:, compute_mask_union]
                .T.dot(interm.reshape((interm.shape[0], interm.shape[1] * interm.shape[2])))
                .reshape((np.sum(compute_mask_union), interm.shape[1], interm.shape[2]))
                - mean_shifts_m.T[compute_mask_union, :][:, compute_mask_v, np.newaxis] * np.sum(interm, axis=0)
            ).T
            res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[compute_mask_v, :][:, compute_mask_union])
        else:
            res_beta_hat_union_t = curr_betas_t[:, compute_mask_v, :].dot(V[:, compute_mask_union])

        if betas_trace_out is not None and betas_trace_gene_sets is not None:
            cur_sets = [betas_trace_gene_sets[x] for x in range(len(betas_trace_gene_sets)) if compute_mask_union[x]]
            pegs_construct_map_to_ind(betas_trace_gene_sets)
            pegs_construct_map_to_ind(cur_sets)

        res_beta_hat_t_flat = res_beta_hat_union_t[:, compute_mask_union_filter_m[compute_mask_v, :]]
        assert res_beta_hat_t_flat.shape[1] == np.sum(compute_mask_m)
        res_beta_hat_t_flat = beta_tildes_m[compute_mask_m] - res_beta_hat_t_flat

        if account_for_V_diag_m:
            res_beta_hat_t_flat = res_beta_hat_t_flat + V_diag_m[compute_mask_m] * curr_betas_t[:, compute_mask_m]
        else:
            res_beta_hat_t_flat = res_beta_hat_t_flat + curr_betas_t[:, compute_mask_m]

    # 2) Convert residualized effect to inclusion probabilities.
    b2_t_flat = np.power(res_beta_hat_t_flat, 2)
    d_const_b2_exp_t_flat = d_const_m[compute_mask_m] * np.exp(-b2_t_flat / (se2s_m[compute_mask_m] * 2.0))
    numerator_t_flat = c_const_m[compute_mask_m] * np.exp(-b2_t_flat / (2.0 * hdmpn_m[compute_mask_m]))
    numerator_zero_mask_t_flat = numerator_t_flat == 0
    denominator_t_flat = numerator_t_flat + d_const_b2_exp_t_flat
    denominator_t_flat[numerator_zero_mask_t_flat] = 1

    d_imaginary_mask_t_flat = ~np.isreal(d_const_b2_exp_t_flat)
    numerator_imaginary_mask_t_flat = ~np.isreal(numerator_t_flat)

    if np.any(np.logical_or(d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)):
        warn("Detected imaginary numbers!")
        denominator_t_flat[d_imaginary_mask_t_flat] = numerator_t_flat[d_imaginary_mask_t_flat]
        numerator_t_flat[np.logical_and(~d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)] = 0

    curr_postp_t[:, compute_mask_m] = numerator_t_flat / denominator_t_flat

    # 3) Update conditional means and sampled betas for this batch.
    curr_post_means_t[:, compute_mask_m] = hdmp_hdmpn_m[compute_mask_m] * (
        curr_postp_t[:, compute_mask_m] * res_beta_hat_t_flat
    )

    if gauss_seidel:
        proposed_beta_t_flat = curr_post_means_t[:, compute_mask_m]
    else:
        norm_mean_t_flat = hdmp_hdmpn_m[compute_mask_m] * res_beta_hat_t_flat
        proposed_beta_t_flat = norm_mean_t_flat + norm_scale_m[compute_mask_m] * rand_norms_t[:, compute_mask_m]
        zero_mask_t_flat = rand_ps_t[:, compute_mask_m] >= curr_postp_t[:, compute_mask_m] * alpha_shrink
        proposed_beta_t_flat[zero_mask_t_flat] = 0

    curr_betas_t[:, compute_mask_m] = proposed_beta_t_flat
    res_beta_hat_t[:, compute_mask_m] = res_beta_hat_t_flat


def _update_inner_beta_rhat_and_outliers(
    sum_betas_t,
    sum_betas2_t,
    iteration_num,
    compute_mask_v,
    use_max_r_for_convergence,
    beta_outlier_iqr_threshold,
    curr_betas_t,
    curr_postp_t,
    curr_post_means_t,
    burn_in_phase_v,
    r_threshold_burn_in,
    num_parallel,
):
    # These matrices have convergence statistics in format (num_parallel, num_gene_sets).
    # WARNING: only results for compute_mask_v are valid.
    (B_m, W_m, R_m, avg_W_m, mean_t) = _calculate_r_tensor_from_chain_sums(sum_betas_t, sum_betas2_t, iteration_num)

    beta_weights_m = np.zeros((sum_betas_t.shape[1], sum_betas_t.shape[2]))
    sum_betas_t_mean = np.mean(sum_betas_t)
    if sum_betas_t_mean > 0:
        np.mean(sum_betas_t, axis=0) / sum_betas_t_mean

    # Calculate thresholded/scaled R summary.
    num_R_above_1_v = np.sum(R_m >= 1, axis=1)
    num_R_above_1_v[num_R_above_1_v == 0] = 1

    R_m_above_1 = copy.copy(R_m)
    R_m_above_1[R_m_above_1 < 1] = 0
    mean_thresholded_R_v = np.sum(R_m_above_1, axis=1) / num_R_above_1_v

    # Max R for each parallel run.
    max_index_v = np.argmax(R_m, axis=1)
    max_index_parallel = None
    max_val = None
    for i in range(len(max_index_v)):
        if compute_mask_v[i] and (max_val is None or R_m[i, max_index_v[i]] > max_val):
            max_val = R_m[i, max_index_v[i]]
            max_index_parallel = i
    max_R_v = np.max(R_m, axis=1)

    if use_max_r_for_convergence:
        convergence_statistic_v = max_R_v
    else:
        convergence_statistic_v = mean_thresholded_R_v

    outlier_mask_m = np.full(avg_W_m.shape, False)
    if avg_W_m.shape[0] > 10:
        # Check per-chain oscillation via variance IQR.
        q3, median, q1 = np.percentile(avg_W_m, [75, 50, 25], axis=0)
        iqr_mask = q3 > q1
        chain_iqr_m = np.zeros(avg_W_m.shape)
        chain_iqr_m[:, iqr_mask] = (avg_W_m[:, iqr_mask] - median[iqr_mask]) / (q3 - q1)[iqr_mask]
        # dimensions chain x parallel
        outlier_mask_m = beta_outlier_iqr_threshold
        if np.sum(outlier_mask_m) > 0:
            log("Detected %d outlier chains due to oscillations" % np.sum(outlier_mask_m), DEBUG)

    if np.sum(R_m > 1) > 10:
        # Check extreme high-R outliers.
        q3, median, q1 = np.percentile(R_m[R_m > 1], [75, 50, 25])
        if q3 > q1:
            R_iqr_m = (R_m - median) / (q3 - q1)
            bad_gene_sets_m = np.logical_and(R_iqr_m > 100, R_m > 2.5)
            if np.sum(bad_gene_sets_m) > 0:
                bad_chains = np.argmax(np.abs(mean_t - np.mean(mean_t, axis=0)), axis=0)[bad_gene_sets_m]

                cur_outlier_mask_m = np.zeros(outlier_mask_m.shape)
                cur_outlier_mask_m[bad_chains, np.where(bad_gene_sets_m)[0]] = True

                log(
                    "Found %d outlier chains across %d parallel runs due to %d gene sets with high R (%.4g - %.4g; %.4g - %.4g)"
                    % (
                        np.sum(cur_outlier_mask_m),
                        np.sum(np.any(cur_outlier_mask_m, axis=0)),
                        np.sum(bad_gene_sets_m),
                        np.min(R_m[bad_gene_sets_m]),
                        np.max(R_m[bad_gene_sets_m]),
                        np.min(R_iqr_m[bad_gene_sets_m]),
                        np.max(R_iqr_m[bad_gene_sets_m]),
                    ),
                    DEBUG,
                )
                outlier_mask_m = np.logical_or(outlier_mask_m, cur_outlier_mask_m)

    non_outliers_m = ~outlier_mask_m
    if np.sum(outlier_mask_m) > 0:
        log("Detected %d total outlier chains" % np.sum(outlier_mask_m), DEBUG)
        # Dimensions are num_chains x num_parallel.
        for outlier_parallel in np.where(np.any(outlier_mask_m, axis=0))[0]:
            if np.sum(outlier_mask_m[:, outlier_parallel]) > 0:
                if np.sum(non_outliers_m[:, outlier_parallel]) > 0:
                    replacement_chains = np.random.choice(
                        np.where(non_outliers_m[:, outlier_parallel])[0],
                        size=np.sum(outlier_mask_m[:, outlier_parallel]),
                    )
                    log(
                        "Replaced chains %s with chains %s in parallel %d"
                        % (
                            np.where(outlier_mask_m[:, outlier_parallel])[0],
                            replacement_chains,
                            outlier_parallel,
                        ),
                        DEBUG,
                    )

                    for tensor in [curr_betas_t, curr_postp_t, curr_post_means_t, sum_betas_t, sum_betas2_t]:
                        tensor[outlier_mask_m[:, outlier_parallel], outlier_parallel, :] = copy.copy(
                            tensor[replacement_chains, outlier_parallel, :]
                        )
                else:
                    log("Every chain was an outlier so doing nothing", TRACE)

    log(
        "Iteration %d: max ind=%s; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;"
        % (
            iteration_num,
            (max_index_parallel, max_index_v[max_index_parallel]) if num_parallel > 1 else max_index_v[max_index_parallel],
            B_m[max_index_parallel, max_index_v[max_index_parallel]],
            W_m[max_index_parallel, max_index_v[max_index_parallel]],
            R_m[max_index_parallel, max_index_v[max_index_parallel]],
            np.mean(mean_thresholded_R_v),
            np.sum(R_m > r_threshold_burn_in),
        ),
        TRACE,
    )

    converged_v = convergence_statistic_v < r_threshold_burn_in
    newly_converged_v = np.logical_and(burn_in_phase_v, converged_v)
    if np.sum(newly_converged_v) > 0:
        if num_parallel == 1:
            log("Converged after %d iterations" % iteration_num, INFO)
        else:
            log(
                "Parallel %s converged after %d iterations"
                % (",".join([str(p) for p in np.nditer(np.where(newly_converged_v))]), iteration_num),
                INFO,
            )
        burn_in_phase_v = np.logical_and(burn_in_phase_v, np.logical_not(converged_v))

    return (R_m, beta_weights_m, burn_in_phase_v)


def _calculate_r_tensor_from_chain_sums(sum_betas_t, sum_betas2_t, num):
    num_chains = sum_betas_t.shape[0]
    num_parallel = sum_betas_t.shape[1]
    num_gene_sets = sum_betas_t.shape[2]

    mean_t = sum_betas_t / float(max(num, 1))
    if num <= 1:
        ones_m = np.ones((num_parallel, num_gene_sets))
        avg_W_m = np.ones((num_chains, num_parallel))
        return (ones_m, ones_m, ones_m, avg_W_m, mean_t)

    mean_m = np.mean(mean_t, axis=0)
    var_t = (sum_betas2_t - float(num) * np.square(mean_t)) / (float(num) - 1)
    b_denom = max(num_chains - 1, 1)
    B_m = (float(num) / float(b_denom)) * np.sum(np.square(mean_t - mean_m), axis=0)
    W_m = np.mean(var_t, axis=0)
    var_given_y_m = np.add((float(num) - 1) / float(num) * W_m, (1.0 / float(num)) * B_m)
    var_given_y_m[var_given_y_m < 0] = 0
    R_m = np.ones((num_parallel, num_gene_sets))
    R_non_zero_mask = W_m > 0
    R_m[R_non_zero_mask] = np.sqrt(var_given_y_m[R_non_zero_mask] / W_m[R_non_zero_mask])
    avg_W_m = np.mean(var_t, axis=2)
    return (B_m, W_m, R_m, avg_W_m, mean_t)


def _calculate_rhat_from_sums(sum_m, sum2_m, num):
    if num <= 1:
        default_v = np.ones(sum_m.shape[1])
        return (default_v, default_v, default_v, default_v)
    mean_m = sum_m / float(num)
    mean_v = np.mean(mean_m, axis=0)
    var_m = (sum2_m - float(num) * np.power(mean_m, 2)) / (float(num) - 1)
    B_v = (float(num) / (mean_m.shape[0] - 1)) * np.sum(np.power(mean_m - mean_v, 2), axis=0)
    W_v = (1.0 / float(mean_m.shape[0])) * np.sum(var_m, axis=0)
    var_given_y_v = np.add((float(num) - 1) / float(num) * W_v, (1.0 / float(num)) * B_v)
    var_given_y_v[var_given_y_v < 0] = 0
    R_v = np.ones(len(W_v))
    R_non_zero_mask = W_v > 0
    R_v[R_non_zero_mask] = np.sqrt(var_given_y_v[R_non_zero_mask] / W_v[R_non_zero_mask])
    return (B_v, W_v, R_v, var_given_y_v)


def _get_active_beta_mask(sum_betas_for_diag_m, num_sum_beta_for_diag_m, active_beta_top_k, active_beta_min_abs):
    beta_chain_means_m = _means_from_sums(sum_betas_for_diag_m, num_sum_beta_for_diag_m)
    beta_mean_v = np.mean(beta_chain_means_m, axis=0)
    abs_beta_mean_v = np.abs(beta_mean_v)

    num_beta = len(beta_mean_v)
    active_mask_v = np.zeros(num_beta, dtype=bool)
    if num_beta == 0:
        return (active_mask_v, beta_chain_means_m, beta_mean_v)

    top_k = min(max(active_beta_top_k, 1), num_beta)
    if top_k >= num_beta:
        active_mask_v[:] = True
    else:
        active_idx = np.argpartition(-abs_beta_mean_v, top_k - 1)[:top_k]
        active_mask_v[active_idx] = True

    if active_beta_min_abs > 0:
        filtered_mask_v = np.logical_and(active_mask_v, abs_beta_mean_v >= active_beta_min_abs)
        if np.any(filtered_mask_v):
            active_mask_v = filtered_mask_v

    return (active_mask_v, beta_chain_means_m, beta_mean_v)


def _initialize_gibbs_epoch_state(state, num_chains, num_full_gene_sets, use_mean_betas, max_mb_X_h, log_fun):
    full_betas_m_shape = (num_chains, num_full_gene_sets)
    prev_warm_start_betas_m = None
    prev_warm_start_postp_m = None
    sum_betas_m = np.zeros(full_betas_m_shape)
    sum_betas2_m = np.zeros(full_betas_m_shape)
    sum_betas_uncorrected_m = np.zeros(full_betas_m_shape)
    sum_betas_uncorrected2_m = np.zeros(full_betas_m_shape)
    sum_postp_m = np.zeros(full_betas_m_shape)
    sum_beta_tildes_m = np.zeros(full_betas_m_shape)
    sum_z_scores_m = np.zeros(full_betas_m_shape)
    num_sum_beta_m = np.zeros(full_betas_m_shape)

    Y_m_shape = (num_chains, len(state.Y_for_regression))
    sum_Ys_m = np.zeros(Y_m_shape)
    sum_Y_raws_m = np.zeros(Y_m_shape)
    sum_log_pos_m = np.zeros(Y_m_shape)
    sum_log_po_raws_m = np.zeros(Y_m_shape)
    sum_log_po_raws2_m = np.zeros(Y_m_shape)
    sum_priors_m = np.zeros(Y_m_shape)
    sum_priors2_m = np.zeros(Y_m_shape)
    sum_Ds_m = np.zeros(Y_m_shape)
    sum_D_raws_m = np.zeros(Y_m_shape)
    sum_bf_orig_m = np.zeros(Y_m_shape)
    sum_bf_orig_raw_m = np.zeros(Y_m_shape)
    sum_bf_orig_raw2_m = np.zeros(Y_m_shape)
    num_sum_Y_m = np.zeros(Y_m_shape)

    # Sums across all iterations, not just converged.
    all_sum_betas_m = np.zeros(full_betas_m_shape)
    all_sum_betas2_m = np.zeros(full_betas_m_shape)
    all_num_sum_m = np.zeros(full_betas_m_shape)

    # Initialize per-chain priors.
    priors_sample_m = np.zeros(Y_m_shape)
    priors_mean_m = np.zeros(Y_m_shape)
    priors_percentage_max_sample_m = np.zeros(Y_m_shape)
    priors_percentage_max_mean_m = np.zeros(Y_m_shape)
    priors_adjustment_sample_m = np.zeros(Y_m_shape)
    priors_adjustment_mean_m = np.zeros(Y_m_shape)

    priors_for_Y_m = priors_sample_m
    priors_percentage_max_for_Y_m = priors_percentage_max_sample_m
    priors_adjustment_for_Y_m = priors_adjustment_sample_m
    if use_mean_betas:
        priors_for_Y_m = priors_mean_m
        priors_percentage_max_for_Y_m = priors_percentage_max_mean_m
        priors_adjustment_for_Y_m = priors_adjustment_mean_m

    num_genes_missing = 0
    if state.genes_missing is not None:
        num_genes_missing = len(state.genes_missing)

    sum_priors_missing_m = np.zeros((num_chains, num_genes_missing))
    sum_Ds_missing_m = np.zeros((num_chains, num_genes_missing))
    priors_missing_sample_m = np.zeros(sum_priors_missing_m.shape)
    priors_missing_mean_m = np.zeros(sum_priors_missing_m.shape)
    num_sum_priors_missing_m = np.zeros(sum_priors_missing_m.shape)

    post_burn_reset_arrays = (
        sum_Ys_m,
        sum_Y_raws_m,
        sum_log_pos_m,
        sum_log_po_raws_m,
        sum_log_po_raws2_m,
        sum_priors_m,
        sum_priors2_m,
        sum_Ds_m,
        sum_D_raws_m,
        sum_bf_orig_m,
        sum_bf_orig_raw_m,
        sum_bf_orig_raw2_m,
        num_sum_Y_m,
        sum_betas_m,
        sum_betas2_m,
        sum_betas_uncorrected_m,
        sum_betas_uncorrected2_m,
        sum_postp_m,
        sum_beta_tildes_m,
        sum_z_scores_m,
        num_sum_beta_m,
    )
    post_burn_reset_missing_arrays = ()
    if state.genes_missing is not None:
        post_burn_reset_missing_arrays = (
            sum_priors_missing_m,
            sum_Ds_missing_m,
            num_sum_priors_missing_m,
        )

    _maybe_unsubset_gene_sets(state, state.gene_sets_missing is not None, skip_V=True)

    stack_batch_size = num_chains + 1
    if num_chains > 1:
        X_size_mb = state._get_X_size_mb()
        X_h_size_mb = num_chains * X_size_mb
        if X_h_size_mb <= max_mb_X_h:
            X_hstacked = sparse.hstack([state.X_orig] * num_chains)
        else:
            stack_batch_size = int(max_mb_X_h / X_size_mb)
            if stack_batch_size == 0:
                stack_batch_size = 1
            log_fun(
                "Not building X_hstacked, size would be %d > %d; will instead run %d chains at a time"
                % (X_h_size_mb, max_mb_X_h, stack_batch_size)
            )
            X_hstacked = sparse.hstack([state.X_orig] * stack_batch_size)
    else:
        X_hstacked = state.X_orig

    num_stack_batches = int(np.ceil(num_chains / float(stack_batch_size)))

    return {
        "full_betas_m_shape": full_betas_m_shape,
        "prev_warm_start_betas_m": prev_warm_start_betas_m,
        "prev_warm_start_postp_m": prev_warm_start_postp_m,
        "sum_betas_m": sum_betas_m,
        "sum_betas2_m": sum_betas2_m,
        "sum_betas_uncorrected_m": sum_betas_uncorrected_m,
        "sum_betas_uncorrected2_m": sum_betas_uncorrected2_m,
        "sum_postp_m": sum_postp_m,
        "sum_beta_tildes_m": sum_beta_tildes_m,
        "sum_z_scores_m": sum_z_scores_m,
        "num_sum_beta_m": num_sum_beta_m,
        "sum_Ys_m": sum_Ys_m,
        "sum_Y_raws_m": sum_Y_raws_m,
        "sum_log_pos_m": sum_log_pos_m,
        "sum_log_po_raws_m": sum_log_po_raws_m,
        "sum_log_po_raws2_m": sum_log_po_raws2_m,
        "sum_priors_m": sum_priors_m,
        "sum_priors2_m": sum_priors2_m,
        "sum_Ds_m": sum_Ds_m,
        "sum_D_raws_m": sum_D_raws_m,
        "sum_bf_orig_m": sum_bf_orig_m,
        "sum_bf_orig_raw_m": sum_bf_orig_raw_m,
        "sum_bf_orig_raw2_m": sum_bf_orig_raw2_m,
        "num_sum_Y_m": num_sum_Y_m,
        "all_sum_betas_m": all_sum_betas_m,
        "all_sum_betas2_m": all_sum_betas2_m,
        "all_num_sum_m": all_num_sum_m,
        "priors_sample_m": priors_sample_m,
        "priors_mean_m": priors_mean_m,
        "priors_percentage_max_sample_m": priors_percentage_max_sample_m,
        "priors_percentage_max_mean_m": priors_percentage_max_mean_m,
        "priors_adjustment_sample_m": priors_adjustment_sample_m,
        "priors_adjustment_mean_m": priors_adjustment_mean_m,
        "priors_for_Y_m": priors_for_Y_m,
        "priors_percentage_max_for_Y_m": priors_percentage_max_for_Y_m,
        "priors_adjustment_for_Y_m": priors_adjustment_for_Y_m,
        "sum_priors_missing_m": sum_priors_missing_m,
        "sum_Ds_missing_m": sum_Ds_missing_m,
        "priors_missing_sample_m": priors_missing_sample_m,
        "priors_missing_mean_m": priors_missing_mean_m,
        "num_sum_priors_missing_m": num_sum_priors_missing_m,
        "post_burn_reset_arrays": post_burn_reset_arrays,
        "post_burn_reset_missing_arrays": post_burn_reset_missing_arrays,
        "X_hstacked": X_hstacked,
        "stack_batch_size": stack_batch_size,
        "num_stack_batches": num_stack_batches,
    }


def _initialize_gibbs_epoch_control_state(epoch_state):
    # Mutable per-epoch control/diagnostic state that evolves each iteration.
    num_gene_sets = epoch_state["sum_betas_m"].shape[1]
    num_genes = epoch_state["sum_Ys_m"].shape[1]
    return {
        "in_burn_in": True,
        "burn_in_pass_streak": 0,
        "burn_in_rhat_history": [],
        "stop_pass_streak": 0,
        "R_beta_v": np.ones(num_gene_sets),
        "betas_sem2_v": np.zeros(num_gene_sets),
        "sem2_v": np.zeros(num_genes),
        "stop_due_to_stall": False,
        "stop_due_to_precision": False,
        "restart_due_to_stall": False,
        "burn_stall_best_beta_rhat_history": [],
        "burn_stall_snapshots": [],
        "burn_stall_beta_indices": None,
        "post_stall_best_beta_rhat_history": [],
        "post_stall_best_D_mcse_history": [],
        "post_stall_snapshots": [],
        "post_stall_beta_indices": None,
        "post_stall_gene_indices": None,
        "prev_Ys_m": None,
    }


def _initialize_gibbs_epoch_sums_state(epoch_state, epoch_aggregates):
    # Mutable per-epoch posterior sum arrays accumulated after burn-in.
    epoch_sums = {key: epoch_state[key] for key in _GIBBS_EPOCH_SUM_KEYS}
    for key in _GIBBS_EPOCH_MISSING_SUM_KEYS:
        epoch_sums[key] = epoch_state[key]
    epoch_sums["epoch_aggregates"] = epoch_aggregates
    return epoch_sums


def _initialize_gibbs_epoch_priors_state(epoch_state):
    # Mutable per-epoch priors/warm-start state updated each iteration.
    return {
        "prev_warm_start_betas_m": epoch_state["prev_warm_start_betas_m"],
        "prev_warm_start_postp_m": epoch_state["prev_warm_start_postp_m"],
        "priors_sample_m": epoch_state["priors_sample_m"],
        "priors_mean_m": epoch_state["priors_mean_m"],
        "priors_percentage_max_sample_m": epoch_state["priors_percentage_max_sample_m"],
        "priors_percentage_max_mean_m": epoch_state["priors_percentage_max_mean_m"],
        "priors_adjustment_sample_m": epoch_state["priors_adjustment_sample_m"],
        "priors_adjustment_mean_m": epoch_state["priors_adjustment_mean_m"],
        "priors_for_Y_m": epoch_state["priors_for_Y_m"],
        "priors_percentage_max_for_Y_m": epoch_state["priors_percentage_max_for_Y_m"],
        "priors_adjustment_for_Y_m": epoch_state["priors_adjustment_for_Y_m"],
        "priors_missing_sample_m": epoch_state["priors_missing_sample_m"],
        "priors_missing_mean_m": epoch_state["priors_missing_mean_m"],
    }


def _initialize_gibbs_epoch_runtime_state(epoch_state, num_p_increases):
    # Mutable per-epoch running totals and restart-related flags.
    epoch_runtime = {key: epoch_state[key] for key in _GIBBS_EPOCH_RUNTIME_SUM_KEYS}
    epoch_runtime["gibbs_good"] = True
    epoch_runtime["num_p_increases"] = num_p_increases
    return epoch_runtime


def _maybe_write_gibbs_gene_stats_trace(
    gene_stats_trace_fh,
    iteration_num,
    trace_chain_offset,
    genes,
    priors_for_Y_m,
    Y_sample_m,
    log_bf_m,
    p_sample_m,
    priors_percentage_max_for_Y_m,
    priors_adjustment_for_Y_m,
):
    if gene_stats_trace_fh is None:
        return
    log("Writing gene stats trace", TRACE)
    _write_gene_stats_trace_rows(
        gene_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        genes,
        priors_for_Y_m,
        Y_sample_m,
        log_bf_m,
        p_sample_m,
        priors_percentage_max_for_Y_m,
        priors_adjustment_for_Y_m,
    )
    gene_stats_trace_fh.flush()


def _maybe_write_gibbs_gene_set_stats_trace(
    gene_set_stats_trace_fh,
    iteration_num,
    trace_chain_offset,
    gene_sets,
    scale_factors,
    full_beta_tildes_m,
    full_p_values_m,
    full_z_scores_m,
    full_ses_m,
    uncorrected_betas_mean_m,
    uncorrected_betas_sample_m,
    full_betas_mean_m,
    full_betas_sample_m,
    full_postp_mean_m,
    full_postp_sample_m,
    full_z_cur_beta_tildes_m,
    R_beta_v,
    betas_sem2_v,
    use_mean_betas,
):
    if gene_set_stats_trace_fh is None:
        return
    log("Writing gene set stats trace", TRACE)
    _write_gene_set_stats_trace_rows(
        gene_set_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        gene_sets,
        scale_factors,
        full_beta_tildes_m,
        full_p_values_m,
        full_z_scores_m,
        full_ses_m,
        uncorrected_betas_mean_m,
        uncorrected_betas_sample_m,
        full_betas_mean_m,
        full_betas_sample_m,
        full_postp_mean_m,
        full_postp_sample_m,
        full_z_cur_beta_tildes_m,
        R_beta_v,
        betas_sem2_v,
        use_mean_betas,
    )
    gene_set_stats_trace_fh.flush()


def _outlier_resistant_mean(sum_m, num_sum_m, num_mad, outlier_mask_m=None, record_param_fn=None):
    if outlier_mask_m is None:
        if record_param_fn is not None:
            record_param_fn("mad_threshold", num_mad)

        chain_means_m = sum_m / num_sum_m
        medians_v = np.median(chain_means_m, axis=0)
        mad_m = np.abs(chain_means_m - medians_v)
        mad_median_v = np.median(mad_m, axis=0)
        outlier_mask_m = chain_means_m > medians_v + num_mad * mad_median_v

    num_sum_v = np.sum(~outlier_mask_m, axis=0)
    num_sum_v[num_sum_v == 0] = 1

    copy_sum_m = copy.copy(sum_m)
    copy_sum_m[outlier_mask_m] = 0
    avg_v = np.sum(copy_sum_m / num_sum_m, axis=0) / num_sum_v
    return (outlier_mask_m, avg_v)


def _write_gene_stats_trace_rows(
    fh,
    iteration_num,
    trace_chain_offset,
    genes,
    priors_for_Y_m,
    Y_sample_m,
    log_bf_m,
    p_sample_m,
    priors_percentage_max_for_Y_m,
    priors_adjustment_for_Y_m,
):
    for chain_num in range(priors_for_Y_m.shape[0]):
        for i in range(len(genes)):
            fh.write(
                "%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n"
                % (
                    iteration_num + 1,
                    trace_chain_offset + chain_num + 1,
                    genes[i],
                    priors_for_Y_m[chain_num, i],
                    Y_sample_m[chain_num, i],
                    log_bf_m[chain_num, i],
                    p_sample_m[chain_num, i],
                    priors_percentage_max_for_Y_m[chain_num, i],
                    priors_adjustment_for_Y_m[chain_num, i],
                )
            )


def _write_gene_set_stats_trace_rows(
    fh,
    iteration_num,
    trace_chain_offset,
    gene_sets,
    scale_factors,
    full_beta_tildes_m,
    full_p_values_m,
    full_z_scores_m,
    full_ses_m,
    uncorrected_betas_mean_m,
    uncorrected_betas_sample_m,
    full_betas_mean_m,
    full_betas_sample_m,
    full_postp_mean_m,
    full_postp_sample_m,
    full_z_cur_beta_tildes_m,
    R_beta_v,
    betas_sem2_v,
    use_mean_betas,
):
    for chain_num in range(full_beta_tildes_m.shape[0]):
        for i in range(len(gene_sets)):
            fh.write(
                "%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n"
                % (
                    iteration_num + 1,
                    trace_chain_offset + chain_num + 1,
                    gene_sets[i],
                    full_beta_tildes_m[chain_num, i] / scale_factors[i],
                    full_p_values_m[chain_num, i],
                    full_z_scores_m[chain_num, i],
                    full_ses_m[chain_num, i] / scale_factors[i],
                    (uncorrected_betas_mean_m[chain_num, i] if use_mean_betas else uncorrected_betas_sample_m[chain_num, i]) / scale_factors[i],
                    (full_betas_mean_m[chain_num, i] if use_mean_betas else full_betas_sample_m[chain_num, i]) / scale_factors[i],
                    (full_postp_mean_m[chain_num, i] if use_mean_betas else full_postp_sample_m[chain_num, i]),
                    full_z_cur_beta_tildes_m[chain_num, i],
                    R_beta_v[i],
                    betas_sem2_v[i],
                )
            )


def _combine_optional_gene_bf_terms(Y_exomes, Y_positive_controls, Y_case_counts):
    combined = 0
    for term in (Y_exomes, Y_positive_controls, Y_case_counts):
        if term is not None:
            combined = combined + term
    return combined


def _add_optional_gene_bf_terms(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m, Y_exomes, Y_positive_controls, Y_case_counts):
    # Add external evidence terms back after HuGE distillation.
    for term in (Y_exomes, Y_positive_controls, Y_case_counts):
        if term is not None:
            log_bf_m += term
            log_bf_uncorrected_m += term
            log_bf_raw_m += term


def _get_gibbs_gene_set_mask(uncorrected_betas_mean_m, uncorrected_betas_sample_m, p_values_m, sparse_frac=0.01, sparse_max=0.001):
    # By design we sparsify using posterior means and apply the same mask to
    # both sample and mean matrices so the next Gibbs iteration warm-start is
    # consistent with the retained support.
    uncorrected_betas_m = uncorrected_betas_mean_m
    gene_set_mask_m = uncorrected_betas_m != 0

    if sparse_frac is not None:
        gene_set_mask_m = np.logical_and(
            gene_set_mask_m,
            (np.abs(uncorrected_betas_m).T >= sparse_frac * np.max(np.abs(uncorrected_betas_m), axis=1)).T,
        )
        gene_set_mask_m = np.logical_and(
            gene_set_mask_m,
            (np.abs(uncorrected_betas_m).T >= sparse_max).T,
        )
        uncorrected_betas_sample_m[~gene_set_mask_m] = 0
        uncorrected_betas_mean_m[~gene_set_mask_m] = 0

    if np.sum(gene_set_mask_m) == 0:
        gene_set_mask_m = p_values_m <= np.min(p_values_m)
    return gene_set_mask_m


def _compute_post_burn_beta_diagnostics(
    diag_sum_betas_m,
    diag_sum_betas2_m,
    diag_num_sum_beta_m,
    num_chains_effective_for_diag,
    active_beta_top_k,
    active_beta_min_abs,
    stop_mcse_quantile,
    beta_rel_mcse_denom_floor,
):
    active_beta_mask, beta_chain_means_m, beta_mean_v = _get_active_beta_mask(
        diag_sum_betas_m,
        diag_num_sum_beta_m,
        active_beta_top_k,
        active_beta_min_abs,
    )
    num_active_betas = int(np.sum(active_beta_mask))

    beta_mcse_v = np.sqrt(np.var(beta_chain_means_m, axis=0, ddof=1) / float(num_chains_effective_for_diag))
    num_post_burn_beta = int(np.min(diag_num_sum_beta_m))
    _, _, post_R_beta_v, _ = _calculate_rhat_from_sums(diag_sum_betas_m, diag_sum_betas2_m, num_post_burn_beta)

    if np.any(active_beta_mask):
        post_R_beta_active_v = post_R_beta_v[active_beta_mask]
        post_rhat_candidates = post_R_beta_active_v[np.logical_and(np.isfinite(post_R_beta_active_v), post_R_beta_active_v >= 1)]
        beta_rhat_q_post = _safe_quantile(post_rhat_candidates, stop_mcse_quantile, 1.0)

        beta_ratio_v = beta_mcse_v / np.maximum(np.abs(beta_mean_v), beta_rel_mcse_denom_floor)
        beta_ratio_q = _safe_quantile(beta_ratio_v[active_beta_mask], stop_mcse_quantile, np.inf)
    else:
        beta_rhat_q_post = 1.0
        beta_ratio_q = 0.0

    return {
        "active_beta_mask": active_beta_mask,
        "beta_mean_v": beta_mean_v,
        "num_active_betas": num_active_betas,
        "beta_mcse_v": beta_mcse_v,
        "num_post_burn_beta": num_post_burn_beta,
        "beta_rhat_q_post": beta_rhat_q_post,
        "beta_ratio_q": beta_ratio_q,
    }


def _compute_post_burn_gene_diagnostics(
    diag_sum_Ds_m,
    diag_num_sum_Y_m,
    num_chains_effective_for_diag,
    stop_top_gene_k,
    stop_min_gene_d,
    stop_mcse_quantile,
):
    D_chain_means_m = _means_from_sums(diag_sum_Ds_m, diag_num_sum_Y_m)
    D_mean_v = np.mean(D_chain_means_m, axis=0)
    D_mcse_v = np.sqrt(np.var(D_chain_means_m, axis=0, ddof=1) / float(num_chains_effective_for_diag))

    top_gene_k = min(stop_top_gene_k, len(D_mean_v))
    num_eligible_genes = len(D_mean_v)
    if stop_min_gene_d is not None:
        eligible_gene_indices = np.where(D_mean_v >= stop_min_gene_d)[0]
        num_eligible_genes = len(eligible_gene_indices)
        if num_eligible_genes > 0:
            if top_gene_k >= num_eligible_genes:
                top_gene_indices = eligible_gene_indices
            else:
                top_gene_indices = eligible_gene_indices[np.argpartition(-D_mean_v[eligible_gene_indices], top_gene_k - 1)[:top_gene_k]]
        else:
            if top_gene_k >= len(D_mean_v):
                top_gene_indices = np.arange(len(D_mean_v))
            else:
                top_gene_indices = np.argpartition(-D_mean_v, top_gene_k - 1)[:top_gene_k]
    else:
        if top_gene_k >= len(D_mean_v):
            top_gene_indices = np.arange(len(D_mean_v))
        else:
            top_gene_indices = np.argpartition(-D_mean_v, top_gene_k - 1)[:top_gene_k]

    num_monitored_genes = len(top_gene_indices)
    D_mcse_q = _safe_quantile(D_mcse_v[top_gene_indices], stop_mcse_quantile, np.inf)

    return {
        "D_mean_v": D_mean_v,
        "D_mcse_v": D_mcse_v,
        "top_gene_k": top_gene_k,
        "top_gene_indices": top_gene_indices,
        "num_monitored_genes": num_monitored_genes,
        "num_eligible_genes": num_eligible_genes,
        "D_mcse_q": D_mcse_q,
    }


def _summarize_gibbs_chain_aggregates(
    sum_Ys_m,
    sum_Y_raws_m,
    sum_log_pos_m,
    sum_log_po_raws_m,
    sum_log_po_raws2_m,
    sum_priors_m,
    sum_priors2_m,
    sum_Ds_m,
    sum_D_raws_m,
    sum_bf_orig_m,
    sum_bf_orig_raw_m,
    sum_bf_orig_raw2_m,
    sum_betas_m,
    sum_betas2_m,
    sum_betas_uncorrected_m,
    sum_betas_uncorrected2_m,
    sum_postp_m,
    sum_beta_tildes_m,
    sum_z_scores_m,
    num_sum_Y_m,
    num_sum_beta_m,
    num_chains_effective,
    num_mad,
    record_param_fn=None,
    sum_priors_missing_m=None,
    sum_Ds_missing_m=None,
    num_sum_priors_missing_m=None,
):
    Y_outlier_mask_m, _ = _outlier_resistant_mean(sum_Ys_m, num_sum_Y_m, num_mad, record_param_fn=record_param_fn)
    beta_outlier_mask_m, avg_betas_v = _outlier_resistant_mean(sum_betas_m, num_sum_beta_m, num_mad, record_param_fn=record_param_fn)

    _, _ = _outlier_resistant_mean(sum_Y_raws_m, num_sum_Y_m, num_mad, record_param_fn=record_param_fn)
    _, avg_log_pos_v = _outlier_resistant_mean(sum_log_pos_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_log_po_raws_v = _outlier_resistant_mean(sum_log_po_raws_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_Ds_v = _outlier_resistant_mean(sum_Ds_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_D_raws_v = _outlier_resistant_mean(sum_D_raws_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_priors_v = _outlier_resistant_mean(sum_priors_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_bf_orig_v = _outlier_resistant_mean(sum_bf_orig_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_bf_orig_raw_v = _outlier_resistant_mean(sum_bf_orig_raw_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)

    avg_priors_missing_v = np.array([])
    avg_Ds_missing_v = np.array([])
    if sum_priors_missing_m is not None and sum_Ds_missing_m is not None and num_sum_priors_missing_m is not None:
        priors_missing_outlier_mask_m, avg_priors_missing_v = _outlier_resistant_mean(sum_priors_missing_m, num_sum_priors_missing_m, num_mad, record_param_fn=record_param_fn)
        _, avg_Ds_missing_v = _outlier_resistant_mean(sum_Ds_missing_m, num_sum_priors_missing_m, num_mad, priors_missing_outlier_mask_m)

    _, avg_betas_uncorrected_v = _outlier_resistant_mean(sum_betas_uncorrected_m, num_sum_beta_m, num_mad, beta_outlier_mask_m)
    _, avg_postp_v = _outlier_resistant_mean(sum_postp_m, num_sum_beta_m, num_mad, beta_outlier_mask_m)
    _, avg_beta_tildes_v = _outlier_resistant_mean(sum_beta_tildes_m, num_sum_beta_m, num_mad, beta_outlier_mask_m)
    _, avg_z_scores_v = _outlier_resistant_mean(sum_z_scores_m, num_sum_beta_m, num_mad, beta_outlier_mask_m)

    num_post_burn_in_Y = int(np.min(num_sum_Y_m))
    num_post_burn_in_beta = int(np.min(num_sum_beta_m))

    _, _, prior_r_hat_v, _ = _calculate_rhat_from_sums(sum_priors_m, sum_priors2_m, num_post_burn_in_Y)
    _, _, combined_r_hat_v, _ = _calculate_rhat_from_sums(sum_log_po_raws_m, sum_log_po_raws2_m, num_post_burn_in_Y)
    _, _, log_bf_r_hat_v, _ = _calculate_rhat_from_sums(sum_bf_orig_raw_m, sum_bf_orig_raw2_m, num_post_burn_in_Y)
    _, _, beta_r_hat_v, _ = _calculate_rhat_from_sums(sum_betas_m, sum_betas2_m, num_post_burn_in_beta)
    _, _, beta_uncorrected_r_hat_v, _ = _calculate_rhat_from_sums(sum_betas_uncorrected_m, sum_betas_uncorrected2_m, num_post_burn_in_beta)

    prior_chain_means_m = _means_from_sums(sum_priors_m, num_sum_Y_m)
    combined_chain_means_m = _means_from_sums(sum_log_po_raws_m, num_sum_Y_m)
    log_bf_chain_means_m = _means_from_sums(sum_bf_orig_raw_m, num_sum_Y_m)
    beta_chain_means_m = _means_from_sums(sum_betas_m, num_sum_beta_m)
    beta_uncorrected_chain_means_m = _means_from_sums(sum_betas_uncorrected_m, num_sum_beta_m)

    prior_mcse_v = np.sqrt(np.var(prior_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))
    combined_mcse_v = np.sqrt(np.var(combined_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))
    log_bf_mcse_v = np.sqrt(np.var(log_bf_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))
    beta_mcse_v = np.sqrt(np.var(beta_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))
    beta_uncorrected_mcse_v = np.sqrt(np.var(beta_uncorrected_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))

    return {
        "avg_log_pos_v": avg_log_pos_v,
        "avg_log_po_raws_v": avg_log_po_raws_v,
        "avg_Ds_v": avg_Ds_v,
        "avg_D_raws_v": avg_D_raws_v,
        "avg_priors_v": avg_priors_v,
        "avg_bf_orig_v": avg_bf_orig_v,
        "avg_bf_orig_raw_v": avg_bf_orig_raw_v,
        "avg_priors_missing_v": avg_priors_missing_v,
        "avg_Ds_missing_v": avg_Ds_missing_v,
        "avg_betas_v": avg_betas_v,
        "avg_betas_uncorrected_v": avg_betas_uncorrected_v,
        "avg_postp_v": avg_postp_v,
        "avg_beta_tildes_v": avg_beta_tildes_v,
        "avg_z_scores_v": avg_z_scores_v,
        "prior_r_hat_v": prior_r_hat_v,
        "combined_r_hat_v": combined_r_hat_v,
        "log_bf_r_hat_v": log_bf_r_hat_v,
        "beta_r_hat_v": beta_r_hat_v,
        "beta_uncorrected_r_hat_v": beta_uncorrected_r_hat_v,
        "prior_mcse_v": prior_mcse_v,
        "combined_mcse_v": combined_mcse_v,
        "log_bf_mcse_v": log_bf_mcse_v,
        "beta_mcse_v": beta_mcse_v,
        "beta_uncorrected_mcse_v": beta_uncorrected_mcse_v,
    }


def _apply_gibbs_final_state(state, final_summary, adjust_priors):
    state.beta_tildes = final_summary["avg_beta_tildes_v"]
    state.z_scores = final_summary["avg_z_scores_v"]
    state.p_values = 2 * scipy.stats.norm.cdf(-np.abs(state.z_scores))
    state.ses = np.full(state.beta_tildes.shape, 100.0)
    state.ses[state.z_scores != 0] = np.abs(state.beta_tildes[state.z_scores != 0] / state.z_scores[state.z_scores != 0])

    state.betas = final_summary["avg_betas_v"]
    state.betas_uncorrected = final_summary["avg_betas_uncorrected_v"]
    state.betas_r_hat = final_summary["beta_r_hat_v"]
    state.betas_mcse = final_summary["beta_mcse_v"]
    state.betas_uncorrected_r_hat = final_summary["beta_uncorrected_r_hat_v"]
    state.betas_uncorrected_mcse = final_summary["beta_uncorrected_mcse_v"]
    state.non_inf_avg_cond_betas = None
    state.non_inf_avg_postps = final_summary["avg_postp_v"]

    state.priors = final_summary["avg_priors_v"]
    state.priors_r_hat = final_summary["prior_r_hat_v"]
    state.priors_mcse = final_summary["prior_mcse_v"]
    state.priors_missing = final_summary["avg_priors_missing_v"]
    state.combined_Ds_missing = final_summary["avg_Ds_missing_v"]

    state.Y_for_regression = final_summary["avg_bf_orig_v"]
    state.Y = final_summary["avg_bf_orig_raw_v"]
    state.Y_r_hat = final_summary["log_bf_r_hat_v"]
    state.Y_mcse = final_summary["log_bf_mcse_v"]

    state.combined_Ds_for_regression = final_summary["avg_Ds_v"]
    state.combined_Ds = final_summary["avg_D_raws_v"]

    state.combined_prior_Ys_for_regression = final_summary["avg_log_pos_v"] - state.background_log_bf
    state.combined_prior_Ys = final_summary["avg_log_po_raws_v"] - state.background_log_bf
    state.combined_prior_Ys_r_hat = final_summary["combined_r_hat_v"]
    state.combined_prior_Ys_mcse = final_summary["combined_mcse_v"]

    gene_N = state.get_gene_N()
    gene_N_missing = state.get_gene_N(get_missing=True)

    all_gene_N = gene_N
    if state.genes_missing is not None:
        assert(gene_N_missing is not None)
        all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

    total_priors = np.concatenate((state.priors, state.priors_missing))
    priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
    priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

    if adjust_priors:
        log("Adjusting priors with slope %.4g" % priors_slope)
        state.priors_adj = state.priors - priors_slope * gene_N - priors_intercept
        if state.genes_missing is not None:
            state.priors_adj_missing = state.priors_missing - priors_slope * gene_N_missing

        combined_slope = np.cov(state.combined_prior_Ys, gene_N)[0,1] / np.var(gene_N)
        combined_intercept = np.mean(state.combined_prior_Ys - gene_N * combined_slope)

        log("Adjusting combined with slope %.4g" % combined_slope)
        state.combined_prior_Ys_adj = state.combined_prior_Ys - combined_slope * gene_N - combined_intercept


def _stack_gibbs_epoch_aggregates(epoch_aggregates, include_missing):
    stacked = {}
    for key in _GIBBS_EPOCH_SUM_KEYS:
        stacked[key] = np.vstack(epoch_aggregates[key])
    if include_missing:
        for key in _GIBBS_EPOCH_MISSING_SUM_KEYS:
            stacked[key] = np.vstack(epoch_aggregates[key])
    return stacked


def _compute_gibbs_iteration_y_terms(
    priors_for_Y_m,
    log_bf_m,
    log_bf_raw_m,
    cur_background_log_bf_v,
    max_log=15.0,
    min_D=1e-5,
    max_D=1 - 1e-5,
):
    Y_sample_m = priors_for_Y_m + log_bf_m
    Y_raw_sample_m = priors_for_Y_m + log_bf_raw_m
    y_var = np.var(Y_sample_m, axis=1)

    cur_log_bf_m = Y_sample_m.T + cur_background_log_bf_v
    cur_log_bf_m[cur_log_bf_m > max_log] = max_log
    bf_sample_m = np.exp(cur_log_bf_m).T

    cur_log_bf_raw_m = Y_raw_sample_m.T + cur_background_log_bf_v
    cur_log_bf_raw_m[cur_log_bf_raw_m > max_log] = max_log
    bf_raw_sample_m = np.exp(cur_log_bf_raw_m).T

    D_sample_m = bf_sample_m / (1 + bf_sample_m)
    D_sample_m[D_sample_m > max_D] = max_D
    D_sample_m[D_sample_m < min_D] = min_D
    log_po_sample_m = np.log(D_sample_m / (1 - D_sample_m))

    D_raw_sample_m = bf_raw_sample_m / (1 + bf_raw_sample_m)
    D_raw_sample_m[D_raw_sample_m > max_D] = max_D
    D_raw_sample_m[D_raw_sample_m < min_D] = min_D
    log_po_raw_sample_m = np.log(D_raw_sample_m / (1 - D_raw_sample_m))

    return {
        "Y_sample_m": Y_sample_m,
        "Y_raw_sample_m": Y_raw_sample_m,
        "y_var": y_var,
        "D_sample_m": D_sample_m,
        "log_po_sample_m": log_po_sample_m,
        "D_raw_sample_m": D_raw_sample_m,
        "log_po_raw_sample_m": log_po_raw_sample_m,
    }


def _build_gibbs_post_burn_accumulation_payload(
    iter_state,
    full_betas_mean_m,
    full_postp_sample_m,
    priors_for_Y_m,
    priors_missing_mean_m,
    log_bf_m,
    log_bf_raw_m,
):
    return {
        "Y_sample_m": iter_state["Y_sample_m"],
        "Y_raw_sample_m": iter_state["Y_raw_sample_m"],
        "log_po_sample_m": iter_state["log_po_sample_m"],
        "log_po_raw_sample_m": iter_state["log_po_raw_sample_m"],
        "priors_for_Y_m": priors_for_Y_m,
        "D_sample_m": iter_state["D_sample_m"],
        "D_raw_sample_m": iter_state["D_raw_sample_m"],
        "log_bf_m": log_bf_m,
        "log_bf_raw_m": log_bf_raw_m,
        "full_betas_mean_m": full_betas_mean_m,
        "uncorrected_betas_mean_m": iter_state["uncorrected_betas_mean_m"],
        "full_postp_sample_m": full_postp_sample_m,
        "full_beta_tildes_m": iter_state["full_beta_tildes_m"],
        "full_z_scores_m": iter_state["full_z_scores_m"],
        "priors_missing_mean_m": priors_missing_mean_m,
    }


def _accumulate_gibbs_post_burn_iteration(
    state,
    accumulation_payload,
    epoch_sums,
):
    Y_sample_m = accumulation_payload["Y_sample_m"]
    Y_raw_sample_m = accumulation_payload["Y_raw_sample_m"]
    log_po_sample_m = accumulation_payload["log_po_sample_m"]
    log_po_raw_sample_m = accumulation_payload["log_po_raw_sample_m"]
    priors_for_Y_m = accumulation_payload["priors_for_Y_m"]
    D_sample_m = accumulation_payload["D_sample_m"]
    D_raw_sample_m = accumulation_payload["D_raw_sample_m"]
    log_bf_m = accumulation_payload["log_bf_m"]
    log_bf_raw_m = accumulation_payload["log_bf_raw_m"]
    full_betas_mean_m = accumulation_payload["full_betas_mean_m"]
    uncorrected_betas_mean_m = accumulation_payload["uncorrected_betas_mean_m"]
    full_postp_sample_m = accumulation_payload["full_postp_sample_m"]
    full_beta_tildes_m = accumulation_payload["full_beta_tildes_m"]
    full_z_scores_m = accumulation_payload["full_z_scores_m"]
    priors_missing_mean_m = accumulation_payload["priors_missing_mean_m"]

    # Collect one post-burn Gibbs draw into running sums used for MCSE/R-hat and
    # final epoch aggregation.
    epoch_sums["sum_Ys_m"] += Y_sample_m
    epoch_sums["sum_Y_raws_m"] += Y_raw_sample_m
    epoch_sums["sum_log_pos_m"] += log_po_sample_m
    epoch_sums["sum_log_po_raws_m"] += log_po_raw_sample_m
    epoch_sums["sum_log_po_raws2_m"] += np.power(log_po_raw_sample_m, 2)
    epoch_sums["sum_priors_m"] += priors_for_Y_m
    epoch_sums["sum_priors2_m"] += np.power(priors_for_Y_m, 2)
    epoch_sums["sum_Ds_m"] += D_sample_m
    epoch_sums["sum_D_raws_m"] += D_raw_sample_m
    epoch_sums["sum_bf_orig_m"] += log_bf_m
    epoch_sums["sum_bf_orig_raw_m"] += log_bf_raw_m
    epoch_sums["sum_bf_orig_raw2_m"] += np.power(log_bf_raw_m, 2)
    epoch_sums["num_sum_Y_m"] += 1

    epoch_sums["sum_betas_m"] += full_betas_mean_m
    epoch_sums["sum_betas2_m"] += np.power(full_betas_mean_m, 2)
    epoch_sums["sum_betas_uncorrected_m"] += uncorrected_betas_mean_m
    epoch_sums["sum_betas_uncorrected2_m"] += np.power(uncorrected_betas_mean_m, 2)
    epoch_sums["sum_postp_m"] += full_postp_sample_m
    epoch_sums["sum_beta_tildes_m"] += full_beta_tildes_m
    epoch_sums["sum_z_scores_m"] += full_z_scores_m
    epoch_sums["num_sum_beta_m"] += 1

    if state.genes_missing is not None:
        epoch_sums["sum_priors_missing_m"] += priors_missing_mean_m
        max_log = 15
        cur_log_priors_missing_m = priors_missing_mean_m + state.background_log_bf
        cur_log_priors_missing_m[cur_log_priors_missing_m > max_log] = max_log
        epoch_sums["sum_Ds_missing_m"] += np.exp(cur_log_priors_missing_m) / (1 + np.exp(cur_log_priors_missing_m))
        epoch_sums["num_sum_priors_missing_m"] += 1


def _sample_gibbs_iteration_y_state(
    state,
    priors_for_Y_m,
    log_bf_m,
    log_bf_raw_m,
    cur_background_log_bf_v,
    y_var_orig,
):
    log("Sampling new Ys")
    log("Setting logistic Ys", TRACE)
    y_terms = _compute_gibbs_iteration_y_terms(
        priors_for_Y_m,
        log_bf_m,
        log_bf_raw_m,
        cur_background_log_bf_v,
    )

    if state.y_corr_sparse is not None:
        log("Adjusting correlation matrix")
    y_corr_sparse = _compute_gibbs_y_corr_sparse(state.y_corr_sparse, priors_for_Y_m, y_var_orig)

    y_terms["y_corr_sparse"] = y_corr_sparse
    return y_terms


def _prepare_gibbs_iteration_inputs(
    state,
    iteration_num,
    log_bf_m,
    log_bf_raw_m,
    iteration_input_config,
):
    epoch_total_iter_offset = iteration_input_config["epoch_total_iter_offset"]
    trace_chain_offset = iteration_input_config["trace_chain_offset"]
    cur_background_log_bf_v = iteration_input_config["cur_background_log_bf_v"]
    y_var_orig = iteration_input_config["y_var_orig"]
    epoch_priors = iteration_input_config["epoch_priors"]
    gene_stats_trace_fh = iteration_input_config["gene_stats_trace_fh"]

    priors_for_Y_m = epoch_priors["priors_for_Y_m"]
    priors_percentage_max_for_Y_m = epoch_priors["priors_percentage_max_for_Y_m"]
    priors_adjustment_for_Y_m = epoch_priors["priors_adjustment_for_Y_m"]

    epoch_iter_num = iteration_num + 1
    total_iter_num = epoch_total_iter_offset + epoch_iter_num

    log("Beginning Gibbs iteration %d (global %d)" % (epoch_iter_num, total_iter_num))
    state._record_param("num_gibbs_iter", iteration_num, overwrite=True)
    state._record_param("num_gibbs_iter_total", total_iter_num, overwrite=True)

    y_terms = _sample_gibbs_iteration_y_state(
        state=state,
        priors_for_Y_m=priors_for_Y_m,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
        cur_background_log_bf_v=cur_background_log_bf_v,
        y_var_orig=y_var_orig,
    )
    (
        Y_sample_m,
        _Y_raw_sample_m,
        y_var,
        D_sample_m,
        _log_po_sample_m,
        _D_raw_sample_m,
        _log_po_raw_sample_m,
        y_corr_sparse,
    ) = (
        y_terms["Y_sample_m"],
        y_terms["Y_raw_sample_m"],
        y_terms["y_var"],
        y_terms["D_sample_m"],
        y_terms["log_po_sample_m"],
        y_terms["D_raw_sample_m"],
        y_terms["log_po_raw_sample_m"],
        y_terms["y_corr_sparse"],
    )

    logistic_config = {
        "full_betas_m_shape": iteration_input_config["full_betas_m_shape"],
        "num_stack_batches": iteration_input_config["num_stack_batches"],
        "stack_batch_size": iteration_input_config["stack_batch_size"],
        "X_hstacked": iteration_input_config["X_hstacked"],
        "inner_beta_kwargs": iteration_input_config["inner_beta_kwargs"],
        "num_chains": len(Y_sample_m),
        "gauss_seidel": iteration_input_config["gauss_seidel"],
        "initial_linear_filter": iteration_input_config["initial_linear_filter"],
        "sparse_frac_gibbs": iteration_input_config["sparse_frac_gibbs"],
        "sparse_max_gibbs": iteration_input_config["sparse_max_gibbs"],
        "correct_betas_mean": iteration_input_config["correct_betas_mean"],
        "correct_betas_var": iteration_input_config["correct_betas_var"],
    }
    logistic_setup = _compute_gibbs_logistic_beta_tildes(
        state=state,
        Y_sample_m=Y_sample_m,
        y_var=y_var,
        D_sample_m=D_sample_m,
        y_corr_sparse=y_corr_sparse,
        logistic_config=logistic_config,
    )

    _maybe_write_gibbs_gene_stats_trace(
        gene_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        state.genes,
        priors_for_Y_m,
        Y_sample_m,
        log_bf_m,
        logistic_setup["p_sample_m"],
        priors_percentage_max_for_Y_m,
        priors_adjustment_for_Y_m,
    )

    return {
        "epoch_iter_num": epoch_iter_num,
        "total_iter_num": total_iter_num,
        "Y_sample_m": y_terms["Y_sample_m"],
        "Y_raw_sample_m": y_terms["Y_raw_sample_m"],
        "y_var": y_terms["y_var"],
        "D_sample_m": y_terms["D_sample_m"],
        "log_po_sample_m": y_terms["log_po_sample_m"],
        "D_raw_sample_m": y_terms["D_raw_sample_m"],
        "log_po_raw_sample_m": y_terms["log_po_raw_sample_m"],
        "full_scale_factors_m": logistic_setup["full_scale_factors_m"],
        "full_mean_shifts_m": logistic_setup["full_mean_shifts_m"],
        "full_is_dense_gene_set_m": logistic_setup["full_is_dense_gene_set_m"],
        "full_ps_m": logistic_setup["full_ps_m"],
        "full_sigma2s_m": logistic_setup["full_sigma2s_m"],
        "p_sample_m": logistic_setup["p_sample_m"],
        "pre_gene_set_filter_mask": logistic_setup["pre_gene_set_filter_mask"],
        "full_z_cur_beta_tildes_m": logistic_setup["full_z_cur_beta_tildes_m"],
        "full_beta_tildes_m": logistic_setup["full_beta_tildes_m"],
        "full_ses_m": logistic_setup["full_ses_m"],
        "full_z_scores_m": logistic_setup["full_z_scores_m"],
        "full_p_values_m": logistic_setup["full_p_values_m"],
    }


def _augment_gibbs_iteration_state_with_uncorrected_and_mask(
    state,
    iter_state,
    prefilter_config,
    inner_beta_kwargs,
):
    uncorrected_setup = pigean_model.compute_gibbs_uncorrected_betas_and_defaults(
        state,
        full_beta_tildes_m=iter_state["full_beta_tildes_m"],
        full_ses_m=iter_state["full_ses_m"],
        full_scale_factors_m=iter_state["full_scale_factors_m"],
        full_mean_shifts_m=iter_state["full_mean_shifts_m"],
        full_is_dense_gene_set_m=iter_state["full_is_dense_gene_set_m"],
        full_ps_m=iter_state["full_ps_m"],
        full_sigma2s_m=iter_state["full_sigma2s_m"],
        **inner_beta_kwargs,
    )
    iter_state.update(uncorrected_setup)

    (
        gene_set_mask_m,
        iter_state["default_betas_sample_m"],
        iter_state["default_postp_sample_m"],
        iter_state["default_betas_mean_m"],
        iter_state["default_postp_mean_m"],
    ) = _prepare_gibbs_gene_set_mask_with_prefilter(
        state,
        iter_state=iter_state,
        prefilter_config=prefilter_config,
        inner_beta_kwargs=inner_beta_kwargs,
        default_betas_sample_m=iter_state["default_betas_sample_m"],
        default_postp_sample_m=iter_state["default_postp_sample_m"],
        default_betas_mean_m=iter_state["default_betas_mean_m"],
        default_postp_mean_m=iter_state["default_postp_mean_m"],
    )

    return (iter_state, gene_set_mask_m)


def _snapshot_pre_gibbs_state(state):
    # Preserve pre-Gibbs values so downstream reporting can compare original vs
    # Gibbs-adjusted statistics.
    state.beta_tildes_orig = copy.copy(state.beta_tildes)
    state.p_values_orig = copy.copy(state.p_values)
    state.ses_orig = copy.copy(state.ses)
    state.z_scores_orig = copy.copy(state.z_scores)
    state.beta_tildes_missing_orig = copy.copy(state.beta_tildes_missing)
    state.p_values_missing_orig = copy.copy(state.p_values_missing)
    state.ses_missing_orig = copy.copy(state.ses_missing)
    state.z_scores_missing_orig = copy.copy(state.z_scores_missing)

    state.betas_orig = copy.copy(state.betas)
    state.betas_uncorrected_orig = copy.copy(state.betas_uncorrected)
    state.non_inf_avg_cond_betas_orig = copy.copy(state.non_inf_avg_cond_betas)
    state.non_inf_avg_postps_orig = copy.copy(state.non_inf_avg_postps)
    state.betas_missing_orig = copy.copy(state.betas_missing)
    state.betas_uncorrected_missing_orig = copy.copy(state.betas_uncorrected_missing)
    state.non_inf_avg_cond_betas_missing_orig = copy.copy(state.non_inf_avg_cond_betas_missing)
    state.non_inf_avg_postps_missing_orig = copy.copy(state.non_inf_avg_postps_missing)

    state.Y_orig = copy.copy(state.Y)
    state.Y_for_regression_orig = copy.copy(state.Y_for_regression)
    state.priors_orig = copy.copy(state.priors)
    state.priors_adj_orig = copy.copy(state.priors_adj)
    state.priors_missing_orig = copy.copy(state.priors_missing)
    state.priors_adj_missing_orig = copy.copy(state.priors_adj_missing)


def _maybe_log_gibbs_conditional_variance(state, top_gene_prior):
    # Legacy logging of implied conditional Y variance from prior settings.
    priors_guess = np.array(
        state.X_orig.dot(state.betas / state.scale_factors)
        - np.sum(state.mean_shifts * state.betas / state.scale_factors)
    )
    y_resid = np.var(state.Y_for_regression_orig - priors_guess)
    y_cond_var = y_resid
    if top_gene_prior is not None:
        if top_gene_prior <= 0 or top_gene_prior >= 1:
            bail("--top-gene-prior needs to be in (0,1)")
        y_total_var = state.convert_prior_to_var(top_gene_prior, len(state.genes))
        y_cond_var = y_total_var - state.get_sigma2(convert_sigma_to_external_units=True) * np.mean(state.get_gene_N())
        if y_cond_var < 0:
            y_cond_var = 0.1
        log("Setting Y cond var=%.4g (total var = %.4g) given top gene prior of %.4g" % (y_cond_var, y_total_var, top_gene_prior))


# ========================= Outer Gibbs Input Snapshot + Prep =========================
_apply_gibbs_log_bf_update = pigean_gibbs._apply_gibbs_log_bf_update


def _run_gibbs_corrected_betas_step(
    state,
    iter_state,
    gene_set_mask_m,
    epoch_priors,
    iteration_update_config,
    inner_beta_kwargs,
):
    return _compute_gibbs_corrected_betas_for_gene_set_mask(
        state,
        gene_set_mask_m=gene_set_mask_m,
        default_betas_sample_m=iter_state["default_betas_sample_m"],
        default_postp_sample_m=iter_state["default_postp_sample_m"],
        default_betas_mean_m=iter_state["default_betas_mean_m"],
        default_postp_mean_m=iter_state["default_postp_mean_m"],
        full_beta_tildes_m=iter_state["full_beta_tildes_m"],
        full_ses_m=iter_state["full_ses_m"],
        full_scale_factors_m=iter_state["full_scale_factors_m"],
        full_mean_shifts_m=iter_state["full_mean_shifts_m"],
        full_is_dense_gene_set_m=iter_state["full_is_dense_gene_set_m"],
        full_ps_m=iter_state["full_ps_m"],
        full_sigma2s_m=iter_state["full_sigma2s_m"],
        uncorrected_betas_mean_m=iter_state["uncorrected_betas_mean_m"],
        use_mean_betas=iteration_update_config.use_mean_betas,
        warm_start=iteration_update_config.warm_start,
        prev_warm_start_betas_m=epoch_priors["prev_warm_start_betas_m"],
        prev_warm_start_postp_m=epoch_priors["prev_warm_start_postp_m"],
        debug_zero_sparse=iteration_update_config.debug_zero_sparse,
        num_chains=iteration_update_config.num_chains,
        num_batches_parallel=iteration_update_config.num_batches_parallel,
        **inner_beta_kwargs,
        betas_trace_out=iteration_update_config.betas_trace_out,
    )


def _apply_refresh_update_to_epoch_priors(epoch_priors, refresh_update):
    epoch_priors["prev_warm_start_betas_m"] = refresh_update["prev_warm_start_betas_m"]
    epoch_priors["prev_warm_start_postp_m"] = refresh_update["prev_warm_start_postp_m"]
    epoch_priors["priors_sample_m"] = refresh_update["priors_sample_m"]
    epoch_priors["priors_mean_m"] = refresh_update["priors_mean_m"]
    epoch_priors["priors_missing_sample_m"] = refresh_update["priors_missing_sample_m"]
    epoch_priors["priors_missing_mean_m"] = refresh_update["priors_missing_mean_m"]
    return _apply_gibbs_log_bf_update(refresh_update)


def _apply_prior_update_to_epoch_priors(epoch_priors, prior_update):
    epoch_priors["priors_sample_m"] = prior_update["priors_sample_m"]
    epoch_priors["priors_mean_m"] = prior_update["priors_mean_m"]
    epoch_priors["priors_missing_sample_m"] = prior_update["priors_missing_sample_m"]
    epoch_priors["priors_missing_mean_m"] = prior_update["priors_missing_mean_m"]
    epoch_priors["priors_for_Y_m"] = prior_update["priors_for_Y_m"]
    epoch_priors["priors_percentage_max_for_Y_m"] = prior_update["priors_percentage_max_for_Y_m"]
    epoch_priors["priors_adjustment_for_Y_m"] = prior_update["priors_adjustment_for_Y_m"]


def _build_refresh_gibbs_iteration_inputs(epoch_priors, iteration_update_config, log_bf_state):
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    return {
        "warm_start": iteration_update_config.warm_start,
        "use_mean_betas": iteration_update_config.use_mean_betas,
        "prev_warm_start_betas_m": epoch_priors["prev_warm_start_betas_m"],
        "prev_warm_start_postp_m": epoch_priors["prev_warm_start_postp_m"],
        "priors_missing_sample_m": epoch_priors["priors_missing_sample_m"],
        "priors_missing_mean_m": epoch_priors["priors_missing_mean_m"],
        "priors_for_Y_m": epoch_priors["priors_for_Y_m"],
        "update_huge_scores": iteration_update_config.update_huge_scores,
        "compute_Y_raw": iteration_update_config.compute_Y_raw,
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
    }


def _build_finalize_gibbs_prior_inputs(epoch_priors, iteration_update_config):
    return {
        "priors_sample_m": epoch_priors["priors_sample_m"],
        "priors_mean_m": epoch_priors["priors_mean_m"],
        "priors_missing_sample_m": epoch_priors["priors_missing_sample_m"],
        "priors_missing_mean_m": epoch_priors["priors_missing_mean_m"],
        "adjust_priors": iteration_update_config.adjust_priors,
        "use_mean_betas": iteration_update_config.use_mean_betas,
        "priors_percentage_max_sample_m": epoch_priors["priors_percentage_max_sample_m"],
        "priors_percentage_max_mean_m": epoch_priors["priors_percentage_max_mean_m"],
        "priors_adjustment_sample_m": epoch_priors["priors_adjustment_sample_m"],
        "priors_adjustment_mean_m": epoch_priors["priors_adjustment_mean_m"],
    }


def _compute_gibbs_iteration_betas_and_priors(
    state,
    iter_state,
    gene_set_mask_m,
    correction_config,
    epoch_priors,
    log_bf_state,
):
    inner_beta_kwargs = correction_config.inner_beta_kwargs
    iteration_update_config = correction_config.iteration_update_config

    (
        full_betas_sample_m,
        full_postp_sample_m,
        full_betas_mean_m,
        full_postp_mean_m,
    ) = _run_gibbs_corrected_betas_step(
        state=state,
        iter_state=iter_state,
        gene_set_mask_m=gene_set_mask_m,
        epoch_priors=epoch_priors,
        iteration_update_config=iteration_update_config,
        inner_beta_kwargs=inner_beta_kwargs,
    )

    refresh_inputs = _build_refresh_gibbs_iteration_inputs(
        epoch_priors,
        iteration_update_config,
        log_bf_state,
    )
    refresh_update = _refresh_gibbs_iteration_priors_and_huge(
        state,
        full_betas_sample_m=full_betas_sample_m,
        full_betas_mean_m=full_betas_mean_m,
        full_postp_sample_m=full_postp_sample_m,
        full_postp_mean_m=full_postp_mean_m,
        **refresh_inputs,
    )
    log_bf_state = _apply_refresh_update_to_epoch_priors(
        epoch_priors,
        refresh_update,
    )

    finalize_inputs = _build_finalize_gibbs_prior_inputs(
        epoch_priors,
        iteration_update_config,
    )
    prior_update = pigean_model.finalize_gibbs_priors_for_sampling(
        state,
        **finalize_inputs,
        log_fn=log,
        trace_level=TRACE,
    )
    _apply_prior_update_to_epoch_priors(epoch_priors, prior_update)

    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    return {
        "full_betas_sample_m": full_betas_sample_m,
        "full_postp_sample_m": full_postp_sample_m,
        "full_betas_mean_m": full_betas_mean_m,
        "full_postp_mean_m": full_postp_mean_m,
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
    }


@dataclass
class GibbsLowBetaRestartControls:
    num_mad: int
    num_attempts: int
    max_num_attempt_restarts: int
    increase_hyper_if_betas_below_for_epoch: float | None
    experimental_hyper_mutation: bool
    num_before_checking_p_increase: int
    p_scale_factor: float


@dataclass
class GibbsLowBetaRestartUpdate:
    gibbs_good: bool
    num_p_increases: int
    should_break: bool


@dataclass
class GibbsAllIterationUpdate:
    all_sum_betas_m: object
    all_sum_betas2_m: object
    all_num_sum_m: object
    R_beta_v: object
    gibbs_good: bool
    num_p_increases: int
    should_break: bool


def _build_gibbs_low_beta_restart_controls(correction_config):
    return GibbsLowBetaRestartControls(
        num_mad=correction_config.num_mad,
        num_attempts=correction_config.num_attempts,
        max_num_attempt_restarts=correction_config.max_num_attempt_restarts,
        increase_hyper_if_betas_below_for_epoch=correction_config.increase_hyper_if_betas_below_for_epoch,
        experimental_hyper_mutation=correction_config.experimental_hyper_mutation,
        num_before_checking_p_increase=correction_config.num_before_checking_p_increase,
        p_scale_factor=correction_config.p_scale_factor,
    )


def _apply_gibbs_all_iteration_update(epoch_runtime, epoch_control, all_iteration_update):
    for key in _GIBBS_EPOCH_RUNTIME_SUM_KEYS:
        epoch_runtime[key] = getattr(all_iteration_update, key)
    epoch_control["R_beta_v"] = all_iteration_update.R_beta_v
    epoch_runtime["gibbs_good"] = all_iteration_update.gibbs_good
    epoch_runtime["num_p_increases"] = all_iteration_update.num_p_increases
    return all_iteration_update.should_break


def _compute_gibbs_y_corr_sparse(y_corr_sparse_base, priors_for_Y_m, y_var_orig):
    if y_corr_sparse_base is None:
        return None

    y_corr_sparse = copy.copy(y_corr_sparse_base)
    y_corr_sparse = y_corr_sparse.multiply(y_var_orig)

    new_y_sd = np.sqrt(np.square(np.mean(priors_for_Y_m, axis=0)) + y_var_orig)[np.newaxis,:]
    new_y_sd[new_y_sd == 0] = 1e-10

    y_corr_sparse = y_corr_sparse.multiply(1 / new_y_sd.T)
    y_corr_sparse = y_corr_sparse.multiply(1 / new_y_sd)
    y_corr_sparse.setdiag(1)
    return y_corr_sparse.tocsc()


def _sample_gibbs_p_targets(Y_sample_m, D_sample_m, gauss_seidel):
    # Keep the legacy initialization shape/type behavior, then overwrite by mode.
    p_sample_m = copy.copy(Y_sample_m)
    if not gauss_seidel:
        p_sample_m = np.zeros(D_sample_m.shape)
        p_sample_m[np.random.random(D_sample_m.shape) < D_sample_m] = 1
    else:
        p_sample_m = D_sample_m
    return p_sample_m


# ========================= Outer Gibbs Logistic Updates =========================


def _compute_gibbs_pre_gene_set_filter_mask(
    state,
    initial_linear_filter,
    Y_sample_m,
    y_var,
    y_corr_sparse,
    inner_beta_kwargs_linear,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    sparse_frac_gibbs,
    sparse_max_gibbs,
    num_gene_sets,
):
    if not initial_linear_filter:
        return np.full(num_gene_sets, True)

    (
        linear_uncorrected_betas_sample_m,
        linear_uncorrected_betas_mean_m,
        linear_p_values_m,
    ) = _compute_gibbs_linear_prefilter_betas(
        state=state,
        Y_sample_m=Y_sample_m,
        y_var=y_var,
        y_corr_sparse=y_corr_sparse,
        inner_beta_kwargs_linear=inner_beta_kwargs_linear,
        full_scale_factors_m=full_scale_factors_m,
        full_mean_shifts_m=full_mean_shifts_m,
        full_is_dense_gene_set_m=full_is_dense_gene_set_m,
        full_ps_m=full_ps_m,
        full_sigma2s_m=full_sigma2s_m,
    )
    pre_gene_set_filter_mask_m = _get_gibbs_gene_set_mask(
        linear_uncorrected_betas_mean_m,
        linear_uncorrected_betas_sample_m,
        linear_p_values_m,
        sparse_frac=sparse_frac_gibbs,
        sparse_max=sparse_max_gibbs,
    )
    pre_gene_set_filter_mask = np.any(pre_gene_set_filter_mask_m, axis=0)
    log("Filtered down to %d gene sets using linear pre-filtering" % np.sum(pre_gene_set_filter_mask))
    return pre_gene_set_filter_mask


def _compute_gibbs_linear_prefilter_betas(
    state,
    Y_sample_m,
    y_var,
    y_corr_sparse,
    inner_beta_kwargs_linear,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
):
    (linear_beta_tildes_m, linear_ses_m, _, linear_p_values_m, _) = state._compute_beta_tildes(
        state.X_orig,
        Y_sample_m,
        y_var,
        state.scale_factors,
        state.mean_shifts,
        resid_correlation_matrix=y_corr_sparse,
    )
    (
        linear_uncorrected_betas_sample_m,
        _,
        linear_uncorrected_betas_mean_m,
        _,
    ) = state._calculate_non_inf_betas(
        assume_independent=True,
        initial_p=None,
        beta_tildes=linear_beta_tildes_m,
        ses=linear_ses_m,
        V=None,
        X_orig=None,
        scale_factors=full_scale_factors_m,
        mean_shifts=full_mean_shifts_m,
        is_dense_gene_set=full_is_dense_gene_set_m,
        ps=full_ps_m,
        sigma2s=full_sigma2s_m,
        return_sample=True,
        update_hyper_sigma=False,
        update_hyper_p=False,
        debug_gene_sets=state.gene_sets,
        **inner_beta_kwargs_linear
    )
    return (
        linear_uncorrected_betas_sample_m,
        linear_uncorrected_betas_mean_m,
        linear_p_values_m,
    )


def _get_gibbs_chain_batch_bounds(batch, stack_batch_size, num_chains):
    begin = batch * stack_batch_size
    end = (batch + 1) * stack_batch_size
    if end > num_chains:
        end = num_chains
    return (begin, end)


def _compute_gibbs_logistic_outputs_for_batches(
    state,
    pre_gene_set_filter_mask,
    p_sample_m,
    y_corr_sparse,
    stack_batch_size,
    num_stack_batches,
    X_hstacked,
    num_chains,
    full_betas_m_shape,
):
    (
        full_beta_tildes_m,
        full_ses_m,
        full_z_scores_m,
        full_p_values_m,
        se_inflation_factors_m,
        diverged_m,
    ) = (
        np.zeros(full_betas_m_shape),
        np.zeros(full_betas_m_shape),
        np.zeros(full_betas_m_shape),
        np.zeros(full_betas_m_shape),
        np.zeros(full_betas_m_shape),
        np.full(full_betas_m_shape, False),
    )

    for batch in range(num_stack_batches):
        (begin, end) = _get_gibbs_chain_batch_bounds(
            batch=batch,
            stack_batch_size=stack_batch_size,
            num_chains=num_chains,
        )

        log("Batch %d: chains %d-%d" % (batch, begin, end), TRACE)
        (
            batch_beta_tildes_m,
            batch_ses_m,
            batch_z_scores_m,
            batch_p_values_m,
            init_se_inflation_factors_m,
            _batch_alpha_tildes_m,
            batch_diverged_m,
        ) = _compute_gibbs_logistic_beta_tildes_batch(
            state=state,
            pre_gene_set_filter_mask=pre_gene_set_filter_mask,
            p_sample_m=p_sample_m,
            begin=begin,
            end=end,
            y_corr_sparse=y_corr_sparse,
            stack_batch_size=stack_batch_size,
            X_hstacked=X_hstacked,
        )
        se_inflation_factors_m = _apply_gibbs_logistic_batch_outputs(
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            full_z_scores_m=full_z_scores_m,
            full_p_values_m=full_p_values_m,
            se_inflation_factors_m=se_inflation_factors_m,
            diverged_m=diverged_m,
            begin=begin,
            end=end,
            pre_gene_set_filter_mask=pre_gene_set_filter_mask,
            batch_beta_tildes_m=batch_beta_tildes_m,
            batch_ses_m=batch_ses_m,
            batch_z_scores_m=batch_z_scores_m,
            batch_p_values_m=batch_p_values_m,
            batch_diverged_m=batch_diverged_m,
            init_se_inflation_factors_m=init_se_inflation_factors_m,
        )

    return {
        "full_beta_tildes_m": full_beta_tildes_m,
        "full_ses_m": full_ses_m,
        "full_z_scores_m": full_z_scores_m,
        "full_p_values_m": full_p_values_m,
        "se_inflation_factors_m": se_inflation_factors_m,
        "diverged_m": diverged_m,
    }


def _apply_gibbs_logistic_batch_outputs(
    full_beta_tildes_m,
    full_ses_m,
    full_z_scores_m,
    full_p_values_m,
    se_inflation_factors_m,
    diverged_m,
    begin,
    end,
    pre_gene_set_filter_mask,
    batch_beta_tildes_m,
    batch_ses_m,
    batch_z_scores_m,
    batch_p_values_m,
    batch_diverged_m,
    init_se_inflation_factors_m,
):
    full_beta_tildes_m[begin:end,pre_gene_set_filter_mask] = batch_beta_tildes_m
    full_ses_m[begin:end,pre_gene_set_filter_mask] = batch_ses_m
    full_z_scores_m[begin:end,pre_gene_set_filter_mask] = batch_z_scores_m
    full_p_values_m[begin:end,pre_gene_set_filter_mask] = batch_p_values_m
    diverged_m[begin:end,pre_gene_set_filter_mask] = batch_diverged_m
    full_ses_m[begin:end,~pre_gene_set_filter_mask] = 100
    full_p_values_m[begin:end,~pre_gene_set_filter_mask] = 1
    if init_se_inflation_factors_m is not None:
        se_inflation_factors_m[begin:end,pre_gene_set_filter_mask] = init_se_inflation_factors_m
    else:
        se_inflation_factors_m = None
    return se_inflation_factors_m


def _log_gibbs_logistic_divergence(diverged_m, gene_sets):
    if np.sum(diverged_m) == 0:
        return
    for c in range(diverged_m.shape[0]):
        if np.sum(diverged_m[c,:] > 0):
            for p in np.nditer(np.where(diverged_m[c,:])):
                log("Chain %d: gene set %s diverged" % (c + 1, gene_sets[p]), DEBUG)


def _compute_gibbs_logistic_beta_tildes_batch(
    state,
    pre_gene_set_filter_mask,
    p_sample_m,
    begin,
    end,
    y_corr_sparse,
    stack_batch_size,
    X_hstacked,
):
    num_cur_stack = (end - begin)
    if num_cur_stack == stack_batch_size:
        cur_X_hstacked = X_hstacked
    else:
        cur_X_hstacked = sparse.hstack([state.X_orig] * num_cur_stack)

    stack_mask = np.tile(pre_gene_set_filter_mask, num_cur_stack)
    return state._compute_logistic_beta_tildes(
        state.X_orig[:,pre_gene_set_filter_mask],
        p_sample_m[begin:end,:],
        state.scale_factors[pre_gene_set_filter_mask],
        state.mean_shifts[pre_gene_set_filter_mask],
        resid_correlation_matrix=y_corr_sparse,
        X_stacked=cur_X_hstacked[:,stack_mask],
    )


def _maybe_correct_gibbs_logistic_beta_tildes(
    state,
    full_beta_tildes_m,
    full_ses_m,
    full_z_scores_m,
    full_p_values_m,
    se_inflation_factors_m,
    correct_betas_mean,
    correct_betas_var,
):
    if not (correct_betas_mean or correct_betas_var):
        return (
            full_beta_tildes_m,
            full_ses_m,
            full_z_scores_m,
            full_p_values_m,
            se_inflation_factors_m,
        )
    return state._correct_beta_tildes(
        full_beta_tildes_m,
        full_ses_m,
        se_inflation_factors_m,
        state.total_qc_metrics,
        state.total_qc_metrics_directions,
        correct_mean=correct_betas_mean,
        correct_var=correct_betas_var,
        fit=False,
    )


def _compute_gibbs_logistic_beta_tildes(
    state,
    Y_sample_m,
    y_var,
    D_sample_m,
    y_corr_sparse,
    logistic_config,
):
    (
        full_betas_m_shape,
        num_stack_batches,
        stack_batch_size,
        X_hstacked,
        inner_beta_kwargs,
        num_chains,
        gauss_seidel,
        initial_linear_filter,
        sparse_frac_gibbs,
        sparse_max_gibbs,
        correct_betas_mean,
        correct_betas_var,
    ) = (
        logistic_config["full_betas_m_shape"],
        logistic_config["num_stack_batches"],
        logistic_config["stack_batch_size"],
        logistic_config["X_hstacked"],
        logistic_config["inner_beta_kwargs"],
        logistic_config["num_chains"],
        logistic_config["gauss_seidel"],
        logistic_config["initial_linear_filter"],
        logistic_config["sparse_frac_gibbs"],
        logistic_config["sparse_max_gibbs"],
        logistic_config["correct_betas_mean"],
        logistic_config["correct_betas_var"],
    )

    inner_beta_kwargs_linear = pigean_model.build_non_inf_beta_sampler_kwargs(inner_beta_kwargs)
    num_gene_sets = len(state.scale_factors)
    full_scale_factors_m = np.tile(state.scale_factors, num_chains).reshape((num_chains, num_gene_sets))
    full_mean_shifts_m = np.tile(state.mean_shifts, num_chains).reshape((num_chains, num_gene_sets))
    full_is_dense_gene_set_m = np.tile(state.is_dense_gene_set, num_chains).reshape((num_chains, num_gene_sets))

    full_ps_m = None
    if state.ps is not None:
        full_ps_m = np.tile(state.ps, num_chains).reshape((num_chains, num_gene_sets))

    full_sigma2s_m = None
    if state.sigma2s is not None:
        full_sigma2s_m = np.tile(state.sigma2s, num_chains).reshape((num_chains, num_gene_sets))

    if not gauss_seidel:
        log("Sampling Ds for logistic", TRACE)
    else:
        log("Setting Ds to mean probabilities", TRACE)
    p_sample_m = _sample_gibbs_p_targets(Y_sample_m, D_sample_m, gauss_seidel)

    pre_gene_set_filter_mask = _compute_gibbs_pre_gene_set_filter_mask(
        state=state,
        initial_linear_filter=initial_linear_filter,
        Y_sample_m=Y_sample_m,
        y_var=y_var,
        y_corr_sparse=y_corr_sparse,
        inner_beta_kwargs_linear=inner_beta_kwargs_linear,
        full_scale_factors_m=full_scale_factors_m,
        full_mean_shifts_m=full_mean_shifts_m,
        full_is_dense_gene_set_m=full_is_dense_gene_set_m,
        full_ps_m=full_ps_m,
        full_sigma2s_m=full_sigma2s_m,
        sparse_frac_gibbs=sparse_frac_gibbs,
        sparse_max_gibbs=sparse_max_gibbs,
        num_gene_sets=full_betas_m_shape[1],
    )
    logistic_outputs = _compute_gibbs_logistic_outputs_for_batches(
        state=state,
        pre_gene_set_filter_mask=pre_gene_set_filter_mask,
        p_sample_m=p_sample_m,
        y_corr_sparse=y_corr_sparse,
        stack_batch_size=stack_batch_size,
        num_stack_batches=num_stack_batches,
        X_hstacked=X_hstacked,
        num_chains=num_chains,
        full_betas_m_shape=full_betas_m_shape,
    )
    full_beta_tildes_m = logistic_outputs["full_beta_tildes_m"]
    full_ses_m = logistic_outputs["full_ses_m"]
    full_z_scores_m = logistic_outputs["full_z_scores_m"]
    full_p_values_m = logistic_outputs["full_p_values_m"]
    se_inflation_factors_m = logistic_outputs["se_inflation_factors_m"]
    diverged_m = logistic_outputs["diverged_m"]

    # Legacy outlier-z filtering was experimental and is disabled in this path.
    full_z_cur_beta_tildes_m = np.zeros(full_beta_tildes_m.shape)

    _log_gibbs_logistic_divergence(diverged_m, state.gene_sets)
    (
        full_beta_tildes_m,
        full_ses_m,
        full_z_scores_m,
        full_p_values_m,
        se_inflation_factors_m,
    ) = _maybe_correct_gibbs_logistic_beta_tildes(
        state=state,
        full_beta_tildes_m=full_beta_tildes_m,
        full_ses_m=full_ses_m,
        full_z_scores_m=full_z_scores_m,
        full_p_values_m=full_p_values_m,
        se_inflation_factors_m=se_inflation_factors_m,
        correct_betas_mean=correct_betas_mean,
        correct_betas_var=correct_betas_var,
    )

    return {
        "full_scale_factors_m": full_scale_factors_m,
        "full_mean_shifts_m": full_mean_shifts_m,
        "full_is_dense_gene_set_m": full_is_dense_gene_set_m,
        "full_ps_m": full_ps_m,
        "full_sigma2s_m": full_sigma2s_m,
        "p_sample_m": p_sample_m,
        "pre_gene_set_filter_mask": pre_gene_set_filter_mask,
        "full_z_cur_beta_tildes_m": full_z_cur_beta_tildes_m,
        "full_beta_tildes_m": full_beta_tildes_m,
        "full_ses_m": full_ses_m,
        "full_z_scores_m": full_z_scores_m,
        "full_p_values_m": full_p_values_m,
    }


def _build_post_stall_snapshot_arrays(
    sum_betas_m,
    sum_betas2_m,
    sum_Ds_m,
    post_stall_beta_indices,
    post_stall_gene_indices,
    num_chains,
):
    if post_stall_beta_indices.size > 0:
        post_stall_beta_sum_m = copy.copy(sum_betas_m[:,post_stall_beta_indices])
        post_stall_beta_sum2_m = copy.copy(sum_betas2_m[:,post_stall_beta_indices])
    else:
        post_stall_beta_sum_m = np.zeros((num_chains, 0))
        post_stall_beta_sum2_m = np.zeros((num_chains, 0))
    if post_stall_gene_indices.size > 0:
        post_stall_D_sum_m = copy.copy(sum_Ds_m[:,post_stall_gene_indices])
    else:
        post_stall_D_sum_m = np.zeros((num_chains, 0))
    return (post_stall_beta_sum_m, post_stall_beta_sum2_m, post_stall_D_sum_m)


def _update_post_stall_best_histories(
    post_stall_best_beta_rhat_history,
    post_stall_best_D_mcse_history,
    beta_rhat_q_post,
    D_mcse_q,
):
    post_best_beta_rhat_q = beta_rhat_q_post if len(post_stall_best_beta_rhat_history) == 0 else min(post_stall_best_beta_rhat_history[-1], beta_rhat_q_post)
    post_best_D_mcse_q = D_mcse_q if len(post_stall_best_D_mcse_history) == 0 else min(post_stall_best_D_mcse_history[-1], D_mcse_q)
    post_stall_best_beta_rhat_history.append(post_best_beta_rhat_q)
    post_stall_best_D_mcse_history.append(post_best_D_mcse_q)


def _unpack_post_burn_stall_tracking_config(stall_tracking_config):
    return (
        stall_tracking_config["active_beta_top_k"],
        stall_tracking_config["post_stall_best_beta_rhat_history"],
        stall_tracking_config["post_stall_best_D_mcse_history"],
        stall_tracking_config["post_stall_snapshots"],
        stall_tracking_config["post_stall_beta_indices"],
        stall_tracking_config["post_stall_gene_indices"],
        stall_tracking_config["stop_mcse_quantile"],
        stall_tracking_config["stall_window"],
        stall_tracking_config["stall_min_post_burn_in"],
        stall_tracking_config["min_num_post_burn_in_for_epoch"],
        stall_tracking_config["stall_delta_rhat"],
        stall_tracking_config["stall_delta_mcse"],
        stall_tracking_config["stall_recent_window"],
        stall_tracking_config["stall_recent_eps"],
    )


def _resolve_post_stall_indices(
    post_stall_beta_indices,
    post_stall_gene_indices,
    active_beta_mask,
    beta_mean_v,
    active_beta_top_k,
    top_gene_indices,
):
    if post_stall_beta_indices is None:
        post_stall_beta_indices = _prepare_stall_indices(active_beta_mask, beta_mean_v, active_beta_top_k)
    if post_stall_gene_indices is None:
        post_stall_gene_indices = copy.copy(top_gene_indices)
    return (post_stall_beta_indices, post_stall_gene_indices)


def _update_post_burn_stall_tracking(
    sum_betas_m,
    sum_betas2_m,
    sum_Ds_m,
    num_sum_Y_m,
    num_chains,
    beta_rhat_q_post,
    D_mcse_q,
    active_beta_mask,
    beta_mean_v,
    top_gene_indices,
    num_post_burn_beta,
    stall_tracking_config,
):
    (
        active_beta_top_k,
        post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history,
        post_stall_snapshots,
        post_stall_beta_indices,
        post_stall_gene_indices,
        stop_mcse_quantile,
        stall_window,
        stall_min_post_burn_in,
        min_num_post_burn_in_for_epoch,
        stall_delta_rhat,
        stall_delta_mcse,
        stall_recent_window,
        stall_recent_eps,
    ) = _unpack_post_burn_stall_tracking_config(stall_tracking_config)

    _update_post_stall_best_histories(
        post_stall_best_beta_rhat_history=post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history=post_stall_best_D_mcse_history,
        beta_rhat_q_post=beta_rhat_q_post,
        D_mcse_q=D_mcse_q,
    )
    (post_stall_beta_indices, post_stall_gene_indices) = _resolve_post_stall_indices(
        post_stall_beta_indices=post_stall_beta_indices,
        post_stall_gene_indices=post_stall_gene_indices,
        active_beta_mask=active_beta_mask,
        beta_mean_v=beta_mean_v,
        active_beta_top_k=active_beta_top_k,
        top_gene_indices=top_gene_indices,
    )

    (post_stall_beta_sum_m, post_stall_beta_sum2_m, post_stall_D_sum_m) = _build_post_stall_snapshot_arrays(
        sum_betas_m=sum_betas_m,
        sum_betas2_m=sum_betas2_m,
        sum_Ds_m=sum_Ds_m,
        post_stall_beta_indices=post_stall_beta_indices,
        post_stall_gene_indices=post_stall_gene_indices,
        num_chains=num_chains,
    )
    num_post_burn_D = int(np.min(num_sum_Y_m))
    post_stall_snapshots.append((num_post_burn_beta, post_stall_beta_sum_m, post_stall_beta_sum2_m, num_post_burn_D, post_stall_D_sum_m))

    _trim_post_stall_history_windows(
        post_stall_best_beta_rhat_history=post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history=post_stall_best_D_mcse_history,
        post_stall_snapshots=post_stall_snapshots,
        stall_window=stall_window,
        stall_recent_window=stall_recent_window,
    )

    (
        post_stall_plateau,
        post_stall_recent_worse,
        post_stall_recent_beta_rhat_q,
        post_stall_recent_D_mcse_q,
    ) = _evaluate_post_stall_status(
        num_post_burn_beta,
        num_post_burn_D,
        post_stall_beta_sum_m,
        post_stall_beta_sum2_m,
        post_stall_D_sum_m,
        post_stall_beta_indices,
        post_stall_gene_indices,
        beta_rhat_q_post,
        D_mcse_q,
        stop_mcse_quantile,
        stall_window,
        stall_min_post_burn_in,
        min_num_post_burn_in_for_epoch,
        stall_delta_rhat,
        stall_delta_mcse,
        stall_recent_window,
        stall_recent_eps,
        post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history,
        post_stall_snapshots,
        num_chains,
    )

    return {
        "post_stall_beta_indices": post_stall_beta_indices,
        "post_stall_gene_indices": post_stall_gene_indices,
        "num_post_burn_D": num_post_burn_D,
        "post_stall_plateau": post_stall_plateau,
        "post_stall_recent_worse": post_stall_recent_worse,
        "post_stall_recent_beta_rhat_q": post_stall_recent_beta_rhat_q,
        "post_stall_recent_D_mcse_q": post_stall_recent_D_mcse_q,
        "post_stall_detected": post_stall_plateau or post_stall_recent_worse,
    }


def _trim_post_stall_history_windows(
    post_stall_best_beta_rhat_history,
    post_stall_best_D_mcse_history,
    post_stall_snapshots,
    stall_window,
    stall_recent_window,
):
    max_stall_history_len = max(stall_window + 2, stall_recent_window + 2, 10)
    _trim_stall_histories(
        post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history,
        post_stall_snapshots,
        max_stall_history_len,
    )


def _decide_gibbs_post_burn_action(
    precision_achieved,
    post_stall_detected,
    post_burn_action_config,
):
    num_attempts = post_burn_action_config["num_attempts"]
    max_num_attempt_restarts = post_burn_action_config["max_num_attempt_restarts"]
    epoch_iter_num = post_burn_action_config["epoch_iter_num"]
    total_iter_num = post_burn_action_config["total_iter_num"]
    post_stall_plateau = post_burn_action_config["post_stall_plateau"]
    post_stall_recent_worse = post_burn_action_config["post_stall_recent_worse"]
    beta_rhat_q_post = post_burn_action_config["beta_rhat_q_post"]
    D_mcse_q = post_burn_action_config["D_mcse_q"]
    post_stall_recent_beta_rhat_q = post_burn_action_config["post_stall_recent_beta_rhat_q"]
    post_stall_recent_D_mcse_q = post_burn_action_config["post_stall_recent_D_mcse_q"]

    decision = {
        "done": False,
        "stop_due_to_precision": False,
        "restart_due_to_stall": False,
        "stop_due_to_stall": False,
    }

    if precision_achieved:
        decision["done"] = True
        decision["stop_due_to_precision"] = True
        log("Desired Gibbs precision achieved; stopping sampling", INFO)
        return decision

    if not post_stall_detected:
        return decision

    if num_attempts < max_num_attempt_restarts:
        decision["done"] = True
        decision["restart_due_to_stall"] = True
        # Keep and aggregate this epoch's post-burn samples, then continue
        # with a new epoch to add more effective chain means.
        log(
            "Restarting Gibbs epoch due to post-burn stall at iter %d (global %d) because precision is not yet met (plateau=%s, recent_worse=%s, beta_Rhat_q=%.4g, D_mcse_q=%.4g, recent_beta_Rhat_q=%s, recent_D_mcse_q=%s); aggregating current epoch samples before restart"
            % (
                epoch_iter_num,
                total_iter_num,
                str(post_stall_plateau),
                str(post_stall_recent_worse),
                beta_rhat_q_post,
                D_mcse_q,
                ("%.4g" % post_stall_recent_beta_rhat_q) if np.isfinite(post_stall_recent_beta_rhat_q) else "NA",
                ("%.4g" % post_stall_recent_D_mcse_q) if np.isfinite(post_stall_recent_D_mcse_q) else "NA",
            ),
            INFO,
        )
        return decision

    decision["done"] = True
    decision["stop_due_to_stall"] = True
    log(
        "Post-burn stall detected at iter %d (global %d) and precision is not yet met, but no restart attempts remain; stopping this epoch (beta_Rhat_q=%.4g, D_mcse_q=%.4g)"
        % (epoch_iter_num, total_iter_num, beta_rhat_q_post, D_mcse_q),
        INFO,
    )
    return decision


def _should_run_gibbs_post_burn_diagnostics(
    epoch_sums,
    diag_every,
    epoch_iter_num,
    epoch_max_num_iter,
):
    return (
        np.all(epoch_sums["num_sum_Y_m"] > 1)
        and np.all(epoch_sums["num_sum_beta_m"] > 1)
        and (epoch_iter_num % diag_every == 0 or epoch_iter_num == epoch_max_num_iter)
    )


GIBBS_POST_BURN_CONTROL_KEYS = (
    "stop_pass_streak",
    "post_stall_beta_indices",
    "post_stall_gene_indices",
    "betas_sem2_v",
    "sem2_v",
    "stop_due_to_precision",
    "restart_due_to_stall",
    "stop_due_to_stall",
)


GIBBS_BURN_IN_CONTROL_KEYS = (
    "in_burn_in",
    "burn_in_pass_streak",
    "stop_pass_streak",
    "prev_Ys_m",
    "burn_stall_beta_indices",
    "R_beta_v",
)


def _apply_gibbs_control_update(epoch_control, control_update, control_keys):
    for key in control_keys:
        epoch_control[key] = control_update[key]


def _build_gibbs_burn_in_control_update(
    in_burn_in,
    burn_in_pass_streak,
    stop_pass_streak,
    prev_Ys_m,
    burn_stall_beta_indices,
    R_beta_v,
):
    return {
        "in_burn_in": in_burn_in,
        "burn_in_pass_streak": burn_in_pass_streak,
        "stop_pass_streak": stop_pass_streak,
        "prev_Ys_m": prev_Ys_m,
        "burn_stall_beta_indices": burn_stall_beta_indices,
        "R_beta_v": R_beta_v,
    }


def _build_gibbs_burn_in_control_from_epoch(epoch_control):
    return _build_gibbs_burn_in_control_update(
        in_burn_in=epoch_control["in_burn_in"],
        burn_in_pass_streak=epoch_control["burn_in_pass_streak"],
        stop_pass_streak=epoch_control["stop_pass_streak"],
        prev_Ys_m=epoch_control["prev_Ys_m"],
        burn_stall_beta_indices=epoch_control["burn_stall_beta_indices"],
        R_beta_v=epoch_control["R_beta_v"],
    )


def _apply_gibbs_burn_in_control_update(epoch_control, burn_in_update):
    _apply_gibbs_control_update(
        epoch_control=epoch_control,
        control_update=burn_in_update,
        control_keys=GIBBS_BURN_IN_CONTROL_KEYS,
    )


def _apply_gibbs_post_burn_control_update(epoch_control, post_burn_update):
    _apply_gibbs_control_update(
        epoch_control=epoch_control,
        control_update=post_burn_update,
        control_keys=GIBBS_POST_BURN_CONTROL_KEYS,
    )


def _build_gibbs_post_burn_control_update(
    stop_pass_streak,
    post_stall_beta_indices,
    post_stall_gene_indices,
    betas_sem2_v,
    sem2_v,
    done,
    stop_due_to_precision,
    restart_due_to_stall,
    stop_due_to_stall,
):
    return {
        "stop_pass_streak": stop_pass_streak,
        "post_stall_beta_indices": post_stall_beta_indices,
        "post_stall_gene_indices": post_stall_gene_indices,
        "betas_sem2_v": betas_sem2_v,
        "sem2_v": sem2_v,
        "done": done,
        "stop_due_to_precision": stop_due_to_precision,
        "restart_due_to_stall": restart_due_to_stall,
        "stop_due_to_stall": stop_due_to_stall,
    }


def _build_gibbs_post_burn_control_from_epoch(epoch_control, done=False):
    return _build_gibbs_post_burn_control_update(
        stop_pass_streak=epoch_control["stop_pass_streak"],
        post_stall_beta_indices=epoch_control["post_stall_beta_indices"],
        post_stall_gene_indices=epoch_control["post_stall_gene_indices"],
        betas_sem2_v=epoch_control["betas_sem2_v"],
        sem2_v=epoch_control["sem2_v"],
        done=done,
        stop_due_to_precision=epoch_control["stop_due_to_precision"],
        restart_due_to_stall=epoch_control["restart_due_to_stall"],
        stop_due_to_stall=epoch_control["stop_due_to_stall"],
    )


def _merge_gibbs_post_burn_control_updates(base_update, diag_update):
    return _build_gibbs_post_burn_control_update(
        stop_pass_streak=diag_update["stop_pass_streak"],
        post_stall_beta_indices=diag_update["post_stall_beta_indices"],
        post_stall_gene_indices=diag_update["post_stall_gene_indices"],
        betas_sem2_v=diag_update["betas_sem2_v"],
        sem2_v=diag_update["sem2_v"],
        done=(base_update["done"] or diag_update["done"]),
        stop_due_to_precision=(base_update["stop_due_to_precision"] or diag_update["stop_due_to_precision"]),
        restart_due_to_stall=(base_update["restart_due_to_stall"] or diag_update["restart_due_to_stall"]),
        stop_due_to_stall=(base_update["stop_due_to_stall"] or diag_update["stop_due_to_stall"]),
    )


def _log_gibbs_post_burn_diagnostics(
    epoch_iter_num,
    total_iter_num,
    stop_mcse_quantile,
    beta_rhat_q_post,
    beta_ratio_q,
    max_rel_mcse_beta,
    beta_rel_mcse_denom_floor,
    top_gene_k,
    stop_min_gene_d,
    num_monitored_genes,
    num_eligible_genes,
    D_mcse_q,
    max_abs_mcse_d,
    num_active_betas,
    num_full_gene_sets,
    num_chains_effective_for_diag,
    burn_in_pass_streak,
    burn_in_patience,
    stop_pass_streak,
    stop_patience,
):
    if stop_min_gene_d is None:
        log(
            "Gibbs iteration %d (global %d): beta_Rhat_q(%.2f)=%.4g; beta_rel_mcse_q(%.2f)=%.4g (threshold=%.4g, denom_floor=%.4g); D_mcse_q(%.2f, topK=%d)=%.4g (threshold=%.4g); active_betas=%d/%d; eff_chains=%d; burn_streak=%d/%d; stop_streak=%d/%d"
            % (
                epoch_iter_num,
                total_iter_num,
                stop_mcse_quantile,
                beta_rhat_q_post,
                stop_mcse_quantile,
                beta_ratio_q,
                max_rel_mcse_beta,
                beta_rel_mcse_denom_floor,
                stop_mcse_quantile,
                top_gene_k,
                D_mcse_q,
                max_abs_mcse_d,
                num_active_betas,
                num_full_gene_sets,
                num_chains_effective_for_diag,
                burn_in_pass_streak,
                burn_in_patience,
                stop_pass_streak,
                stop_patience,
            ),
            INFO,
        )
        return

    log(
        "Gibbs iteration %d (global %d): beta_Rhat_q(%.2f)=%.4g; beta_rel_mcse_q(%.2f)=%.4g (threshold=%.4g, denom_floor=%.4g); D_mcse_q(%.2f, topK=%d, minD=%.4g, monitored=%d, eligible=%d)=%.4g (threshold=%.4g); active_betas=%d/%d; eff_chains=%d; burn_streak=%d/%d; stop_streak=%d/%d"
        % (
            epoch_iter_num,
            total_iter_num,
            stop_mcse_quantile,
            beta_rhat_q_post,
            stop_mcse_quantile,
            beta_ratio_q,
            max_rel_mcse_beta,
            beta_rel_mcse_denom_floor,
            stop_mcse_quantile,
            top_gene_k,
            stop_min_gene_d,
            num_monitored_genes,
            num_eligible_genes,
            D_mcse_q,
            max_abs_mcse_d,
            num_active_betas,
            num_full_gene_sets,
            num_chains_effective_for_diag,
            burn_in_pass_streak,
            burn_in_patience,
            stop_pass_streak,
            stop_patience,
        ),
        INFO,
    )


def _build_post_burn_stall_tracking_config(
    diag_config,
    epoch_control,
    min_num_post_burn_in_for_epoch,
):
    return {
        "active_beta_top_k": diag_config["active_beta_top_k"],
        "post_stall_best_beta_rhat_history": epoch_control["post_stall_best_beta_rhat_history"],
        "post_stall_best_D_mcse_history": epoch_control["post_stall_best_D_mcse_history"],
        "post_stall_snapshots": epoch_control["post_stall_snapshots"],
        "post_stall_beta_indices": epoch_control["post_stall_beta_indices"],
        "post_stall_gene_indices": epoch_control["post_stall_gene_indices"],
        "stop_mcse_quantile": diag_config["stop_mcse_quantile"],
        "stall_window": diag_config["stall_window"],
        "stall_min_post_burn_in": diag_config["stall_min_post_burn_in"],
        "min_num_post_burn_in_for_epoch": min_num_post_burn_in_for_epoch,
        "stall_delta_rhat": diag_config["stall_delta_rhat"],
        "stall_delta_mcse": diag_config["stall_delta_mcse"],
        "stall_recent_window": diag_config["stall_recent_window"],
        "stall_recent_eps": diag_config["stall_recent_eps"],
    }


def _build_post_burn_action_config(
    run_state,
    iter_state,
    post_stall_update,
    beta_rhat_q_post,
    D_mcse_q,
):
    return {
        "num_attempts": run_state.num_attempts,
        "max_num_attempt_restarts": run_state.max_num_attempt_restarts,
        "epoch_iter_num": iter_state["epoch_iter_num"],
        "total_iter_num": iter_state["total_iter_num"],
        "post_stall_plateau": post_stall_update["post_stall_plateau"],
        "post_stall_recent_worse": post_stall_update["post_stall_recent_worse"],
        "beta_rhat_q_post": beta_rhat_q_post,
        "D_mcse_q": D_mcse_q,
        "post_stall_recent_beta_rhat_q": post_stall_update["post_stall_recent_beta_rhat_q"],
        "post_stall_recent_D_mcse_q": post_stall_update["post_stall_recent_D_mcse_q"],
    }


# ========================= Outer Gibbs Post-burn Diagnostics =========================
def _compute_gibbs_post_burn_diag_metrics(
    min_num_post_burn_in_for_epoch,
    diag_config,
    epoch_sums,
    epoch_control,
):
    (
        epoch_aggregates,
        sum_betas_m,
        sum_betas2_m,
        num_sum_beta_m,
        sum_Ds_m,
        num_sum_Y_m,
    ) = (
        epoch_sums["epoch_aggregates"],
        epoch_sums["sum_betas_m"],
        epoch_sums["sum_betas2_m"],
        epoch_sums["num_sum_beta_m"],
        epoch_sums["sum_Ds_m"],
        epoch_sums["num_sum_Y_m"],
    )
    num_chains = diag_config["num_chains"]
    stop_mcse_quantile = diag_config["stop_mcse_quantile"]

    # For stopping diagnostics, aggregate previous completed epochs with the
    # current in-progress epoch so MCSE aligns with final reported MCSE.
    (
        diag_sum_betas_m,
        diag_sum_betas2_m,
        diag_num_sum_beta_m,
        diag_sum_Ds_m,
        diag_num_sum_Y_m,
    ) = _build_gibbs_diag_sums(
        epoch_aggregates,
        sum_betas_m,
        sum_betas2_m,
        num_sum_beta_m,
        sum_Ds_m,
        num_sum_Y_m,
    )

    num_chains_effective_for_diag = diag_sum_betas_m.shape[0]
    beta_diag = _compute_post_burn_beta_diagnostics(
        diag_sum_betas_m,
        diag_sum_betas2_m,
        diag_num_sum_beta_m,
        num_chains_effective_for_diag,
        diag_config["active_beta_top_k"],
        diag_config["active_beta_min_abs"],
        stop_mcse_quantile,
        diag_config["beta_rel_mcse_denom_floor"],
    )
    gene_diag = _compute_post_burn_gene_diagnostics(
        diag_sum_Ds_m,
        diag_num_sum_Y_m,
        num_chains_effective_for_diag,
        diag_config["stop_top_gene_k"],
        diag_config["stop_min_gene_d"],
        stop_mcse_quantile,
    )

    post_stall_update = _update_post_burn_stall_tracking(
        sum_betas_m,
        sum_betas2_m,
        sum_Ds_m,
        num_sum_Y_m,
        num_chains,
        beta_diag["beta_rhat_q_post"],
        gene_diag["D_mcse_q"],
        beta_diag["active_beta_mask"],
        beta_diag["beta_mean_v"],
        gene_diag["top_gene_indices"],
        beta_diag["num_post_burn_beta"],
        stall_tracking_config=_build_post_burn_stall_tracking_config(
            diag_config=diag_config,
            epoch_control=epoch_control,
            min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        ),
    )

    return {
        "num_chains_effective_for_diag": num_chains_effective_for_diag,
        "num_active_betas": beta_diag["num_active_betas"],
        "beta_mcse_v": beta_diag["beta_mcse_v"],
        "beta_rhat_q_post": beta_diag["beta_rhat_q_post"],
        "beta_ratio_q": beta_diag["beta_ratio_q"],
        "num_post_burn_beta": beta_diag["num_post_burn_beta"],
        "D_mcse_v": gene_diag["D_mcse_v"],
        "top_gene_k": gene_diag["top_gene_k"],
        "num_monitored_genes": gene_diag["num_monitored_genes"],
        "num_eligible_genes": gene_diag["num_eligible_genes"],
        "D_mcse_q": gene_diag["D_mcse_q"],
        "post_stall_update": post_stall_update,
    }


def _update_gibbs_post_burn_precision_streak(
    stop_pass_streak,
    beta_ratio_q,
    D_mcse_q,
    max_rel_mcse_beta,
    max_abs_mcse_d,
    num_post_burn_D,
    min_num_post_burn_in_for_epoch,
):
    min_post_burn_reached = num_post_burn_D >= min_num_post_burn_in_for_epoch
    precision_pass = beta_ratio_q <= max_rel_mcse_beta and D_mcse_q <= max_abs_mcse_d
    if precision_pass and min_post_burn_reached:
        stop_pass_streak += 1
    else:
        stop_pass_streak = 0
    return (stop_pass_streak, min_post_burn_reached)


def _evaluate_gibbs_post_burn_diagnostics_and_decision(
    min_num_post_burn_in_for_epoch,
    diag_config,
    iter_state,
    epoch_sums,
    epoch_control,
    run_state,
):
    epoch_iter_num = iter_state["epoch_iter_num"]
    total_iter_num = iter_state["total_iter_num"]

    stop_pass_streak = epoch_control["stop_pass_streak"]
    burn_in_pass_streak = epoch_control["burn_in_pass_streak"]

    diag_metrics = _compute_gibbs_post_burn_diag_metrics(
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        diag_config=diag_config,
        epoch_sums=epoch_sums,
        epoch_control=epoch_control,
    )
    post_stall_update = diag_metrics["post_stall_update"]

    num_post_burn_D = post_stall_update["num_post_burn_D"]
    post_stall_detected = post_stall_update["post_stall_detected"]

    stop_pass_streak, min_post_burn_reached = _update_gibbs_post_burn_precision_streak(
        stop_pass_streak=stop_pass_streak,
        beta_ratio_q=diag_metrics["beta_ratio_q"],
        D_mcse_q=diag_metrics["D_mcse_q"],
        max_rel_mcse_beta=diag_config["max_rel_mcse_beta"],
        max_abs_mcse_d=diag_config["max_abs_mcse_d"],
        num_post_burn_D=num_post_burn_D,
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
    )

    _log_gibbs_post_burn_diagnostics(
        epoch_iter_num=epoch_iter_num,
        total_iter_num=total_iter_num,
        stop_mcse_quantile=diag_config["stop_mcse_quantile"],
        beta_rhat_q_post=diag_metrics["beta_rhat_q_post"],
        beta_ratio_q=diag_metrics["beta_ratio_q"],
        max_rel_mcse_beta=diag_config["max_rel_mcse_beta"],
        beta_rel_mcse_denom_floor=diag_config["beta_rel_mcse_denom_floor"],
        top_gene_k=diag_metrics["top_gene_k"],
        stop_min_gene_d=diag_config["stop_min_gene_d"],
        num_monitored_genes=diag_metrics["num_monitored_genes"],
        num_eligible_genes=diag_metrics["num_eligible_genes"],
        D_mcse_q=diag_metrics["D_mcse_q"],
        max_abs_mcse_d=diag_config["max_abs_mcse_d"],
        num_active_betas=diag_metrics["num_active_betas"],
        num_full_gene_sets=diag_config["num_full_gene_sets"],
        num_chains_effective_for_diag=diag_metrics["num_chains_effective_for_diag"],
        burn_in_pass_streak=burn_in_pass_streak,
        burn_in_patience=diag_config["burn_in_patience"],
        stop_pass_streak=stop_pass_streak,
        stop_patience=diag_config["stop_patience"],
    )

    precision_achieved = min_post_burn_reached and stop_pass_streak >= diag_config["stop_patience"]
    decision = _decide_gibbs_post_burn_action(
        precision_achieved=precision_achieved,
        post_stall_detected=post_stall_detected,
        post_burn_action_config=_build_post_burn_action_config(
            run_state=run_state,
            iter_state=iter_state,
            post_stall_update=post_stall_update,
            beta_rhat_q_post=diag_metrics["beta_rhat_q_post"],
            D_mcse_q=diag_metrics["D_mcse_q"],
        ),
    )

    return _build_gibbs_post_burn_control_update(
        stop_pass_streak=stop_pass_streak,
        post_stall_beta_indices=post_stall_update["post_stall_beta_indices"],
        post_stall_gene_indices=post_stall_update["post_stall_gene_indices"],
        betas_sem2_v=np.square(diag_metrics["beta_mcse_v"]),
        sem2_v=np.square(diag_metrics["D_mcse_v"]),
        done=decision["done"],
        stop_due_to_precision=decision["stop_due_to_precision"],
        restart_due_to_stall=decision["restart_due_to_stall"],
        stop_due_to_stall=decision["stop_due_to_stall"],
    )


def _run_optional_gibbs_post_burn_diagnostics(
    min_num_post_burn_in_for_epoch,
    diag_every,
    epoch_iter_num,
    epoch_max_num_iter,
    post_burn_diag_config,
    iter_state,
    epoch_sums,
    epoch_control,
    run_state,
    post_burn_update,
):
    if not _should_run_gibbs_post_burn_diagnostics(
        epoch_sums=epoch_sums,
        diag_every=diag_every,
        epoch_iter_num=epoch_iter_num,
        epoch_max_num_iter=epoch_max_num_iter,
    ):
        return post_burn_update

    return _run_due_gibbs_post_burn_diagnostics(
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        post_burn_diag_config=post_burn_diag_config,
        iter_state=iter_state,
        epoch_sums=epoch_sums,
        epoch_control=epoch_control,
        run_state=run_state,
        post_burn_update=post_burn_update,
    )


def _run_due_gibbs_post_burn_diagnostics(
    min_num_post_burn_in_for_epoch,
    post_burn_diag_config,
    iter_state,
    epoch_sums,
    epoch_control,
    run_state,
    post_burn_update,
):
    post_burn_diag = _evaluate_gibbs_post_burn_diagnostics_and_decision(
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        diag_config=post_burn_diag_config,
        iter_state=iter_state,
        epoch_sums=epoch_sums,
        epoch_control=epoch_control,
        run_state=run_state,
    )

    return _merge_gibbs_post_burn_control_updates(
        base_update=post_burn_update,
        diag_update=post_burn_diag,
    )


def _maybe_end_gibbs_epoch_for_post_burn_cap(
    done,
    num_sum_Y_m,
    max_num_post_burn_in_for_epoch,
    epoch_iter_num,
    total_iter_num,
):
    num_post_burn_samples_now = int(np.min(num_sum_Y_m))
    if (not done) and num_post_burn_samples_now >= max_num_post_burn_in_for_epoch:
        done = True
        log(
            "Ending Gibbs epoch at iter %d (global %d) after %d post-burn samples (per-epoch max post-burn reached)"
            % (epoch_iter_num, total_iter_num, num_post_burn_samples_now),
            INFO,
        )
    return done


def _update_gibbs_post_burn_state(
    state,
    max_num_post_burn_in_for_epoch,
    min_num_post_burn_in_for_epoch,
    epoch_max_num_iter,
    diag_every,
    post_burn_diag_config,
    iter_state,
    epoch_sums,
    epoch_priors,
    epoch_control,
    run_state,
    log_bf_m,
    log_bf_raw_m,
    full_betas_mean_m,
    full_postp_sample_m,
):
    in_burn_in = epoch_control["in_burn_in"]
    post_burn_update = _build_gibbs_post_burn_control_from_epoch(
        epoch_control,
        done=False,
    )
    epoch_iter_num = iter_state["epoch_iter_num"]
    total_iter_num = iter_state["total_iter_num"]

    priors_for_Y_m = epoch_priors["priors_for_Y_m"]
    priors_missing_mean_m = epoch_priors["priors_missing_mean_m"]

    if in_burn_in:
        return post_burn_update

    return _advance_gibbs_post_burn_state(
        state=state,
        priors_for_Y_m=priors_for_Y_m,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
        full_betas_mean_m=full_betas_mean_m,
        full_postp_sample_m=full_postp_sample_m,
        priors_missing_mean_m=priors_missing_mean_m,
        epoch_sums=epoch_sums,
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        diag_every=diag_every,
        epoch_iter_num=epoch_iter_num,
        epoch_max_num_iter=epoch_max_num_iter,
        post_burn_diag_config=post_burn_diag_config,
        iter_state=iter_state,
        epoch_control=epoch_control,
        run_state=run_state,
        post_burn_update=post_burn_update,
        max_num_post_burn_in_for_epoch=max_num_post_burn_in_for_epoch,
        total_iter_num=total_iter_num,
    )


def _advance_gibbs_post_burn_state(
    state,
    priors_for_Y_m,
    log_bf_m,
    log_bf_raw_m,
    full_betas_mean_m,
    full_postp_sample_m,
    priors_missing_mean_m,
    epoch_sums,
    min_num_post_burn_in_for_epoch,
    diag_every,
    epoch_iter_num,
    epoch_max_num_iter,
    post_burn_diag_config,
    iter_state,
    epoch_control,
    run_state,
    post_burn_update,
    max_num_post_burn_in_for_epoch,
    total_iter_num,
):
    accumulation_payload = _build_gibbs_post_burn_accumulation_payload(
        iter_state=iter_state,
        full_betas_mean_m=full_betas_mean_m,
        full_postp_sample_m=full_postp_sample_m,
        priors_for_Y_m=priors_for_Y_m,
        priors_missing_mean_m=priors_missing_mean_m,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
    )
    _accumulate_gibbs_post_burn_iteration(
        state=state,
        accumulation_payload=accumulation_payload,
        epoch_sums=epoch_sums,
    )

    post_burn_diag_update = _run_optional_gibbs_post_burn_diagnostics(
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        diag_every=diag_every,
        epoch_iter_num=epoch_iter_num,
        epoch_max_num_iter=epoch_max_num_iter,
        post_burn_diag_config=post_burn_diag_config,
        iter_state=iter_state,
        epoch_sums=epoch_sums,
        epoch_control=epoch_control,
        run_state=run_state,
        post_burn_update=post_burn_update,
    )
    post_burn_diag_update["done"] = _maybe_end_gibbs_epoch_for_post_burn_cap(
        done=post_burn_diag_update["done"],
        num_sum_Y_m=epoch_sums["num_sum_Y_m"],
        max_num_post_burn_in_for_epoch=max_num_post_burn_in_for_epoch,
        epoch_iter_num=epoch_iter_num,
        total_iter_num=total_iter_num,
    )
    return post_burn_diag_update


def _build_gibbs_iteration_progress_context(progress_runtime_config, iteration_update):
    return {
        "trace_chain_offset": progress_runtime_config.trace_chain_offset,
        "epoch_total_iter_offset": progress_runtime_config.epoch_total_iter_offset,
        "epoch_max_num_iter": progress_runtime_config.epoch_max_num_iter,
        "max_num_burn_in_for_epoch": progress_runtime_config.max_num_burn_in_for_epoch,
        "min_num_iter_for_epoch": progress_runtime_config.min_num_iter_for_epoch,
        "min_num_burn_in_for_epoch": progress_runtime_config.min_num_burn_in_for_epoch,
        "max_num_post_burn_in_for_epoch": progress_runtime_config.max_num_post_burn_in_for_epoch,
        "min_num_post_burn_in_for_epoch": progress_runtime_config.min_num_post_burn_in_for_epoch,
        "post_burn_reset_arrays": progress_runtime_config.post_burn_reset_arrays,
        "post_burn_reset_missing_arrays": progress_runtime_config.post_burn_reset_missing_arrays,
        "iteration_progress_config": progress_runtime_config.iteration_progress_config,
        "full_betas_sample_m": iteration_update["full_betas_sample_m"],
        "full_postp_sample_m": iteration_update["full_postp_sample_m"],
        "full_betas_mean_m": iteration_update["full_betas_mean_m"],
        "full_postp_mean_m": iteration_update["full_postp_mean_m"],
    }


def _write_gibbs_iteration_gene_set_stats_trace(
    gene_set_stats_trace_fh,
    iteration_num,
    trace_chain_offset,
    state,
    iter_state,
    full_betas_mean_m,
    full_betas_sample_m,
    full_postp_mean_m,
    full_postp_sample_m,
    R_beta_v,
    betas_sem2_v,
    use_mean_betas,
):
    _maybe_write_gibbs_gene_set_stats_trace(
        gene_set_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        state.gene_sets,
        state.scale_factors,
        iter_state["full_beta_tildes_m"],
        iter_state["full_p_values_m"],
        iter_state["full_z_scores_m"],
        iter_state["full_ses_m"],
        iter_state["uncorrected_betas_mean_m"],
        iter_state["uncorrected_betas_sample_m"],
        full_betas_mean_m,
        full_betas_sample_m,
        full_postp_mean_m,
        full_postp_sample_m,
        iter_state["full_z_cur_beta_tildes_m"],
        R_beta_v,
        betas_sem2_v,
        use_mean_betas,
    )


def _finalize_gibbs_iteration_progress(
    state,
    gene_set_stats_trace_fh,
    iteration_num,
    trace_chain_offset,
    iter_state,
    full_betas_mean_m,
    full_betas_sample_m,
    full_postp_mean_m,
    full_postp_sample_m,
    R_beta_v,
    betas_sem2_v,
    use_mean_betas,
    epoch_control,
    post_burn_update,
):
    _write_gibbs_iteration_gene_set_stats_trace(
        gene_set_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        state,
        iter_state,
        full_betas_mean_m,
        full_betas_sample_m,
        full_postp_mean_m,
        full_postp_sample_m,
        R_beta_v,
        betas_sem2_v,
        use_mean_betas,
    )

    _apply_gibbs_post_burn_control_update(epoch_control=epoch_control, post_burn_update=post_burn_update)
    return {"done": post_burn_update["done"]}


def _prepare_gibbs_gene_set_mask_with_prefilter(
    state,
    iter_state,
    prefilter_config,
    inner_beta_kwargs,
    default_betas_sample_m,
    default_postp_sample_m,
    default_betas_mean_m,
    default_postp_mean_m,
):
    (
        uncorrected_betas_mean_m,
        uncorrected_betas_sample_m,
        full_p_values_m,
        full_beta_tildes_m,
        full_ses_m,
        full_scale_factors_m,
        full_mean_shifts_m,
        full_is_dense_gene_set_m,
        full_ps_m,
        full_sigma2s_m,
    ) = (
        iter_state["uncorrected_betas_mean_m"],
        iter_state["uncorrected_betas_sample_m"],
        iter_state["full_p_values_m"],
        iter_state["full_beta_tildes_m"],
        iter_state["full_ses_m"],
        iter_state["full_scale_factors_m"],
        iter_state["full_mean_shifts_m"],
        iter_state["full_is_dense_gene_set_m"],
        iter_state["full_ps_m"],
        iter_state["full_sigma2s_m"],
    )
    (
        sparse_frac_gibbs,
        sparse_max_gibbs,
        pre_filter_batch_size,
        pre_filter_small_batch_size,
    ) = (
        prefilter_config["sparse_frac_gibbs"],
        prefilter_config["sparse_max_gibbs"],
        prefilter_config["pre_filter_batch_size"],
        prefilter_config["pre_filter_small_batch_size"],
    )
    inner_beta_kwargs_linear = pigean_model.build_non_inf_beta_sampler_kwargs(inner_beta_kwargs)

    # Start from sparsity mask on uncorrected betas, then optionally run a
    # cheaper prefilter pass to mark additional near-zero sets before the
    # corrected beta batching stage.
    gene_set_mask_m = np.full(full_p_values_m.shape, True)
    gene_set_mask_m = _get_gibbs_gene_set_mask(
        uncorrected_betas_mean_m,
        uncorrected_betas_sample_m,
        full_p_values_m,
        sparse_frac=sparse_frac_gibbs,
        sparse_max=sparse_max_gibbs,
    )

    any_gene_set_mask = np.any(gene_set_mask_m, axis=0)
    if pre_filter_batch_size is not None and np.sum(any_gene_set_mask) > pre_filter_batch_size:
        (
            gene_set_mask_m,
            default_betas_sample_m,
            default_postp_sample_m,
            default_betas_mean_m,
            default_postp_mean_m,
        ) = _apply_gibbs_block_prefilter_pruning(
            state=state,
            any_gene_set_mask=any_gene_set_mask,
            pre_filter_small_batch_size=pre_filter_small_batch_size,
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            full_scale_factors_m=full_scale_factors_m,
            full_mean_shifts_m=full_mean_shifts_m,
            full_is_dense_gene_set_m=full_is_dense_gene_set_m,
            full_ps_m=full_ps_m,
            full_sigma2s_m=full_sigma2s_m,
            inner_beta_kwargs_linear=inner_beta_kwargs_linear,
            full_p_values_m=full_p_values_m,
            sparse_frac_gibbs=sparse_frac_gibbs,
            sparse_max_gibbs=sparse_max_gibbs,
            gene_set_mask_m=gene_set_mask_m,
            default_betas_sample_m=default_betas_sample_m,
            default_postp_sample_m=default_postp_sample_m,
            default_betas_mean_m=default_betas_mean_m,
            default_postp_mean_m=default_postp_mean_m,
        )

    gene_set_mask_m = _normalize_gibbs_gene_set_mask_across_chains(gene_set_mask_m)

    return (
        gene_set_mask_m,
        default_betas_sample_m,
        default_postp_sample_m,
        default_betas_mean_m,
        default_postp_mean_m,
    )


def _apply_gibbs_block_prefilter_pruning(
    state,
    any_gene_set_mask,
    pre_filter_small_batch_size,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    inner_beta_kwargs_linear,
    full_p_values_m,
    sparse_frac_gibbs,
    sparse_max_gibbs,
    gene_set_mask_m,
    default_betas_sample_m,
    default_postp_sample_m,
    default_betas_mean_m,
    default_postp_mean_m,
):
    num_batches = state._get_num_X_blocks(
        state.X_orig[:,any_gene_set_mask],
        batch_size=pre_filter_small_batch_size,
    )
    if num_batches <= 1:
        return (
            gene_set_mask_m,
            default_betas_sample_m,
            default_postp_sample_m,
            default_betas_mean_m,
            default_postp_mean_m,
        )

    gene_set_block_masks = state._compute_gene_set_batches(
        V=None,
        X_orig=state.X_orig[:,any_gene_set_mask],
        mean_shifts=state.mean_shifts[any_gene_set_mask],
        scale_factors=state.scale_factors[any_gene_set_mask],
        find_correlated_instead=pre_filter_small_batch_size,
    )

    if len(gene_set_block_masks) == 0:
        return (
            gene_set_mask_m,
            default_betas_sample_m,
            default_postp_sample_m,
            default_betas_mean_m,
            default_postp_mean_m,
        )

    if np.sum(gene_set_block_masks[-1]) == 1 and len(gene_set_block_masks) > 1:
        # Merge singleton tail block into previous block.
        gene_set_block_masks[-2][gene_set_block_masks[-1]] = True
        gene_set_block_masks = gene_set_block_masks[:-1]
    if len(gene_set_block_masks) <= 1 or np.sum(gene_set_block_masks[0]) <= 1:
        return (
            gene_set_mask_m,
            default_betas_sample_m,
            default_postp_sample_m,
            default_betas_mean_m,
            default_postp_mean_m,
        )

    V_sparse = _build_gibbs_prefilter_sparse_V(
        state=state,
        any_gene_set_mask=any_gene_set_mask,
        gene_set_block_masks=gene_set_block_masks,
    )
    log("Running %d blocks to check for zeros..." % len(gene_set_block_masks), DEBUG)
    (
        half_corrected_betas_sample_m,
        half_corrected_postp_sample_m,
        half_corrected_betas_mean_m,
        half_corrected_postp_mean_m,
    ) = state._calculate_non_inf_betas(
        initial_p=None,
        beta_tildes=full_beta_tildes_m[:,any_gene_set_mask],
        ses=full_ses_m[:,any_gene_set_mask],
        V=V_sparse,
        X_orig=None,
        scale_factors=full_scale_factors_m[:,any_gene_set_mask],
        mean_shifts=full_mean_shifts_m[:,any_gene_set_mask],
        is_dense_gene_set=full_is_dense_gene_set_m[:,any_gene_set_mask],
        ps=full_ps_m[:,any_gene_set_mask],
        sigma2s=full_sigma2s_m[:,any_gene_set_mask],
        return_sample=True,
        update_hyper_sigma=False,
        update_hyper_p=False,
        **inner_beta_kwargs_linear,
    )

    add_zero_mask_m = ~(
        _get_gibbs_gene_set_mask(
            half_corrected_betas_mean_m,
            half_corrected_betas_sample_m,
            full_p_values_m,
            sparse_frac=sparse_frac_gibbs,
            sparse_max=sparse_max_gibbs,
        )
    )

    if np.any(add_zero_mask_m):
        map_to_full = np.where(any_gene_set_mask)[0]
        set_to_zero_full = np.where(add_zero_mask_m)
        set_to_zero_full = (set_to_zero_full[0], map_to_full[set_to_zero_full[1]])
        orig_zero = np.sum(np.any(gene_set_mask_m, axis=0))
        gene_set_mask_m[set_to_zero_full] = False
        new_zero = np.sum(np.any(gene_set_mask_m, axis=0))
        log("Found %d additional zero gene sets" % (orig_zero - new_zero), DEBUG)

        default_betas_sample_m[set_to_zero_full] = half_corrected_betas_sample_m[add_zero_mask_m]
        default_postp_sample_m[set_to_zero_full] = half_corrected_postp_sample_m[add_zero_mask_m]
        default_betas_mean_m[set_to_zero_full] = half_corrected_betas_mean_m[add_zero_mask_m]
        default_postp_mean_m[set_to_zero_full] = half_corrected_postp_mean_m[add_zero_mask_m]

    return (
        gene_set_mask_m,
        default_betas_sample_m,
        default_postp_sample_m,
        default_betas_mean_m,
        default_postp_mean_m,
    )


def _build_gibbs_prefilter_sparse_V(state, any_gene_set_mask, gene_set_block_masks):
    V_data = []
    V_rows = []
    V_cols = []
    for gene_set_block_mask in gene_set_block_masks:
        V_block = state._calculate_V_internal(
            state.X_orig[:,any_gene_set_mask][:,gene_set_block_mask],
            state.y_corr_cholesky,
            state.mean_shifts[any_gene_set_mask][gene_set_block_mask],
            state.scale_factors[any_gene_set_mask][gene_set_block_mask],
        )
        orig_indices = np.where(gene_set_block_mask)[0]
        V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
        V_cols += list(np.tile(orig_indices, V_block.shape[0]))
        V_data += list(V_block.ravel())

    return sparse.csc_matrix(
        (V_data, (V_rows, V_cols)),
        shape=(np.sum(any_gene_set_mask), np.sum(any_gene_set_mask)),
    )


def _normalize_gibbs_gene_set_mask_across_chains(gene_set_mask_m):
    if sum(np.any(gene_set_mask_m, axis=0)) == 0:
        gene_set_mask_m[:,0] = 1

    num_non_missing_v = np.sum(gene_set_mask_m, axis=1)
    max_num_non_missing = np.max(num_non_missing_v)
    max_num_non_missing_idx = np.argmax(num_non_missing_v)
    log("Max number of gene sets to keep across all chains is %d" % (max_num_non_missing))
    log("Keeping %d gene sets that had non-zero uncorected betas" % (sum(np.any(gene_set_mask_m, axis=0))))
    for chain_num in range(gene_set_mask_m.shape[0]):
        if num_non_missing_v[chain_num] < max_num_non_missing:
            cur_num = 0
            for index in np.nonzero(
                gene_set_mask_m[max_num_non_missing_idx,:] & ~gene_set_mask_m[chain_num,:]
            )[0]:
                assert(gene_set_mask_m[chain_num,index] == False)
                gene_set_mask_m[chain_num,index] = True
                cur_num += 1
                if cur_num >= max_num_non_missing - num_non_missing_v[chain_num]:
                    break
    return gene_set_mask_m


def _compute_gibbs_corrected_betas_for_gene_set_mask(
    state,
    gene_set_mask_m,
    default_betas_sample_m,
    default_postp_sample_m,
    default_betas_mean_m,
    default_postp_mean_m,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    uncorrected_betas_mean_m,
    use_mean_betas,
    warm_start,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
    debug_zero_sparse,
    num_chains,
    num_batches_parallel,
    passed_in_max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    gauss_seidel_betas,
    sparse_solution,
    sparse_frac_betas,
    betas_trace_out,
):
    # Estimate corrected non-inf betas in chain batches to keep V construction
    # memory bounded while preserving legacy batching behavior.
    (
        full_betas_mean_m,
        full_betas_sample_m,
        full_postp_mean_m,
        full_postp_sample_m,
    ) = _initialize_gibbs_corrected_beta_output_matrices(
        default_betas_mean_m=default_betas_mean_m,
        default_betas_sample_m=default_betas_sample_m,
        default_postp_mean_m=default_postp_mean_m,
        default_postp_sample_m=default_postp_sample_m,
        debug_zero_sparse=debug_zero_sparse,
    )

    num_calculations = int(np.ceil(num_chains / num_batches_parallel))
    for calc in range(num_calculations):
        (begin, end) = _get_gibbs_chain_batch_bounds(
            batch=calc,
            stack_batch_size=num_batches_parallel,
            num_chains=num_chains,
        )

        (
            cur_gene_set_mask,
            num_missing,
            run_one_V,
            beta_tildes_m,
            ses_m,
            V_m,
            scale_factors_m,
            mean_shifts_m,
            is_dense_gene_set_m,
            ps_m,
            sigma2s_m,
            init_betas_m,
            init_postp_m,
        ) = _prepare_gibbs_corrected_batch_inputs(
            state=state,
            gene_set_mask_m=gene_set_mask_m,
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            full_scale_factors_m=full_scale_factors_m,
            full_mean_shifts_m=full_mean_shifts_m,
            full_is_dense_gene_set_m=full_is_dense_gene_set_m,
            full_ps_m=full_ps_m,
            full_sigma2s_m=full_sigma2s_m,
            warm_start=warm_start,
            prev_warm_start_betas_m=prev_warm_start_betas_m,
            prev_warm_start_postp_m=prev_warm_start_postp_m,
            begin=begin,
            end=end,
            num_chains=num_chains,
        )

        (
            cur_betas_sample_m,
            cur_postp_sample_m,
            cur_betas_mean_m,
            cur_postp_mean_m,
        ) = _run_gibbs_corrected_beta_sampler(
            state=state,
            beta_tildes_m=beta_tildes_m,
            ses_m=ses_m,
            V_m=V_m,
            scale_factors_m=scale_factors_m,
            mean_shifts_m=mean_shifts_m,
            is_dense_gene_set_m=is_dense_gene_set_m,
            ps_m=ps_m,
            sigma2s_m=sigma2s_m,
            passed_in_max_num_burn_in=passed_in_max_num_burn_in,
            max_num_iter_betas=max_num_iter_betas,
            min_num_iter_betas=min_num_iter_betas,
            num_chains_betas=num_chains_betas,
            r_threshold_burn_in_betas=r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=use_max_r_for_convergence_betas,
            max_frac_sem_betas=max_frac_sem_betas,
            max_allowed_batch_correlation=max_allowed_batch_correlation,
            gauss_seidel_betas=gauss_seidel_betas,
            num_missing=num_missing,
            sparse_solution=sparse_solution,
            sparse_frac_betas=sparse_frac_betas,
            betas_trace_out=betas_trace_out,
            gene_set_mask_m=gene_set_mask_m,
            init_betas_m=init_betas_m,
            init_postp_m=init_postp_m,
        )

        _store_gibbs_corrected_batch_results(
            run_one_V=run_one_V,
            full_betas_sample_m=full_betas_sample_m,
            full_postp_sample_m=full_postp_sample_m,
            full_betas_mean_m=full_betas_mean_m,
            full_postp_mean_m=full_postp_mean_m,
            cur_betas_sample_m=cur_betas_sample_m,
            cur_postp_sample_m=cur_postp_sample_m,
            cur_betas_mean_m=cur_betas_mean_m,
            cur_postp_mean_m=cur_postp_mean_m,
            begin=begin,
            end=end,
            cur_gene_set_mask=cur_gene_set_mask,
            gene_set_mask_m=gene_set_mask_m,
        )

        if run_one_V:
            print_overlapping = None
            _log_gibbs_overlapping_corrected_beta_details(
                print_overlapping=print_overlapping,
                state=state,
                cur_gene_set_mask=cur_gene_set_mask,
                V_m=V_m,
                cur_betas_mean_m=cur_betas_mean_m,
                cur_betas_sample_m=cur_betas_sample_m,
                use_mean_betas=use_mean_betas,
                uncorrected_betas_mean_m=uncorrected_betas_mean_m,
            )

    return (
        full_betas_sample_m,
        full_postp_sample_m,
        full_betas_mean_m,
        full_postp_mean_m,
    )


def _store_gibbs_corrected_batch_results(
    run_one_V,
    full_betas_sample_m,
    full_postp_sample_m,
    full_betas_mean_m,
    full_postp_mean_m,
    cur_betas_sample_m,
    cur_postp_sample_m,
    cur_betas_mean_m,
    cur_postp_mean_m,
    begin,
    end,
    cur_gene_set_mask,
    gene_set_mask_m,
):
    if run_one_V:
        full_betas_sample_m[begin:end,cur_gene_set_mask] = cur_betas_sample_m
        full_postp_sample_m[begin:end,cur_gene_set_mask] = cur_postp_sample_m
        full_betas_mean_m[begin:end,cur_gene_set_mask] = cur_betas_mean_m
        full_postp_mean_m[begin:end,cur_gene_set_mask] = cur_postp_mean_m
        return

    full_betas_sample_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_betas_sample_m.ravel()
    full_postp_sample_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_postp_sample_m.ravel()
    full_betas_mean_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_betas_mean_m.ravel()
    full_postp_mean_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_postp_mean_m.ravel()


def _log_gibbs_overlapping_corrected_beta_details(
    print_overlapping,
    state,
    cur_gene_set_mask,
    V_m,
    cur_betas_mean_m,
    cur_betas_sample_m,
    use_mean_betas,
    uncorrected_betas_mean_m,
):
    if print_overlapping is None:
        return
    gene_sets_run = [state.gene_sets[i] for i in range(len(state.gene_sets)) if cur_gene_set_mask[i]]
    gene_set_to_ind = pegs_construct_map_to_ind(gene_sets_run)
    for gene_set in print_overlapping:
        if gene_set in gene_set_to_ind:
            log("For gene set %s" % (gene_set))
            ind = gene_set_to_ind[gene_set]
            values = V_m[ind,:] * (cur_betas_mean_m if use_mean_betas else cur_betas_sample_m)
            indices = np.argsort(values, axis=1)
            for chain in range(values.shape[0]):
                log("Chain %d (uncorrected beta=%.4g, corrected beta=%.4g)" % (chain, uncorrected_betas_mean_m[chain,state.gene_set_to_ind[gene_set]], (cur_betas_mean_m[chain,ind] if use_mean_betas else cur_betas_sample_m[chain,ind])))
                for i in indices[chain,::-1]:
                    if values[chain,i] == 0:
                        break
                    log("%s, V=%.4g, beta=%.4g, prod=%.4g" % (gene_sets_run[i], V_m[ind,i], (cur_betas_mean_m[chain,i] if use_mean_betas else cur_betas_sample_m[chain,i]), values[chain,i]))


def _prepare_gibbs_corrected_batch_inputs(
    state,
    gene_set_mask_m,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    warm_start,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
    begin,
    end,
    num_chains,
):
    # Build the superset V matrix once for this chain batch.
    cur_gene_set_mask = np.any(gene_set_mask_m[begin:end,:], axis=0)
    num_gene_set_mask = np.sum(cur_gene_set_mask)
    max_num_gene_set_mask = np.max(np.sum(gene_set_mask_m, axis=1))
    V_superset = state._calculate_V_internal(
        state.X_orig[:,cur_gene_set_mask],
        state.y_corr_cholesky,
        state.mean_shifts[cur_gene_set_mask],
        state.scale_factors[cur_gene_set_mask],
    )

    # If the superset is not too much larger than max-per-chain, reuse one V;
    # otherwise subset V per chain.
    run_one_V = num_gene_set_mask < 5 * max_num_gene_set_mask
    if run_one_V:
        num_non_missing = np.sum(cur_gene_set_mask)
    else:
        num_non_missing = np.max(np.sum(gene_set_mask_m, axis=1))
    num_missing = gene_set_mask_m.shape[1] - num_non_missing

    if run_one_V:
        (
            beta_tildes_m,
            ses_m,
            V_m,
            scale_factors_m,
            mean_shifts_m,
            is_dense_gene_set_m,
            ps_m,
            sigma2s_m,
            init_betas_m,
            init_postp_m,
        ) = _prepare_gibbs_corrected_run_one_v_inputs(
            state=state,
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            cur_gene_set_mask=cur_gene_set_mask,
            begin=begin,
            end=end,
            V_superset=V_superset,
            warm_start=warm_start,
            prev_warm_start_betas_m=prev_warm_start_betas_m,
            prev_warm_start_postp_m=prev_warm_start_postp_m,
        )
    else:
        (
            beta_tildes_m,
            ses_m,
            V_m,
            scale_factors_m,
            mean_shifts_m,
            is_dense_gene_set_m,
            ps_m,
            sigma2s_m,
            init_betas_m,
            init_postp_m,
        ) = _prepare_gibbs_corrected_per_chain_v_inputs(
            gene_set_mask_m=gene_set_mask_m,
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            full_scale_factors_m=full_scale_factors_m,
            full_mean_shifts_m=full_mean_shifts_m,
            full_is_dense_gene_set_m=full_is_dense_gene_set_m,
            full_ps_m=full_ps_m,
            full_sigma2s_m=full_sigma2s_m,
            cur_gene_set_mask=cur_gene_set_mask,
            V_superset=V_superset,
            begin=begin,
            end=end,
            num_chains=num_chains,
            num_non_missing=num_non_missing,
            warm_start=warm_start,
            prev_warm_start_betas_m=prev_warm_start_betas_m,
            prev_warm_start_postp_m=prev_warm_start_postp_m,
        )

    return (
        cur_gene_set_mask,
        num_missing,
        run_one_V,
        beta_tildes_m,
        ses_m,
        V_m,
        scale_factors_m,
        mean_shifts_m,
        is_dense_gene_set_m,
        ps_m,
        sigma2s_m,
        init_betas_m,
        init_postp_m,
    )


def _prepare_gibbs_corrected_run_one_v_inputs(
    state,
    full_beta_tildes_m,
    full_ses_m,
    cur_gene_set_mask,
    begin,
    end,
    V_superset,
    warm_start,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
):
    beta_tildes_m = full_beta_tildes_m[begin:end,cur_gene_set_mask]
    ses_m = full_ses_m[begin:end,cur_gene_set_mask]
    V_m = V_superset
    scale_factors_m = state.scale_factors[cur_gene_set_mask]
    mean_shifts_m = state.mean_shifts[cur_gene_set_mask]
    is_dense_gene_set_m = state.is_dense_gene_set[cur_gene_set_mask]
    ps_m = None
    if state.ps is not None:
        ps_m = state.ps[cur_gene_set_mask]
    sigma2s_m = None
    if state.sigma2s is not None:
        sigma2s_m = state.sigma2s[cur_gene_set_mask]

    init_betas_m = None
    init_postp_m = None
    if warm_start and prev_warm_start_betas_m is not None:
        init_betas_m = prev_warm_start_betas_m[begin:end,cur_gene_set_mask]
        init_postp_m = prev_warm_start_postp_m[begin:end,cur_gene_set_mask]

    return (
        beta_tildes_m,
        ses_m,
        V_m,
        scale_factors_m,
        mean_shifts_m,
        is_dense_gene_set_m,
        ps_m,
        sigma2s_m,
        init_betas_m,
        init_postp_m,
    )


def _prepare_gibbs_corrected_per_chain_v_inputs(
    gene_set_mask_m,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    cur_gene_set_mask,
    V_superset,
    begin,
    end,
    num_chains,
    num_non_missing,
    warm_start,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
):
    non_missing_matrix_shape = (num_chains, num_non_missing)
    beta_tildes_m = full_beta_tildes_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    ses_m = full_ses_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    scale_factors_m = full_scale_factors_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    mean_shifts_m = full_mean_shifts_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    is_dense_gene_set_m = full_is_dense_gene_set_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    ps_m = None
    if full_ps_m is not None:
        ps_m = full_ps_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    sigma2s_m = None
    if full_sigma2s_m is not None:
        sigma2s_m = full_sigma2s_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]

    init_betas_m = None
    init_postp_m = None
    if warm_start and prev_warm_start_betas_m is not None:
        init_betas_m = prev_warm_start_betas_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
        init_postp_m = prev_warm_start_postp_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]

    V_m = np.zeros((end-begin, beta_tildes_m.shape[1], beta_tildes_m.shape[1]))
    for i,j in zip(range(begin, end),range(end-begin)):
        gene_set_mask_subset = gene_set_mask_m[i,cur_gene_set_mask]
        V_m[j,:,:] = V_superset[gene_set_mask_subset,:][:,gene_set_mask_subset]

    return (
        beta_tildes_m,
        ses_m,
        V_m,
        scale_factors_m,
        mean_shifts_m,
        is_dense_gene_set_m,
        ps_m,
        sigma2s_m,
        init_betas_m,
        init_postp_m,
    )


def _run_gibbs_corrected_beta_sampler(
    state,
    beta_tildes_m,
    ses_m,
    V_m,
    scale_factors_m,
    mean_shifts_m,
    is_dense_gene_set_m,
    ps_m,
    sigma2s_m,
    passed_in_max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    gauss_seidel_betas,
    num_missing,
    sparse_solution,
    sparse_frac_betas,
    betas_trace_out,
    gene_set_mask_m,
    init_betas_m,
    init_postp_m,
):
    return state._calculate_non_inf_betas(
        initial_p=None,
        beta_tildes=beta_tildes_m,
        ses=ses_m,
        V=V_m,
        scale_factors=scale_factors_m,
        mean_shifts=mean_shifts_m,
        is_dense_gene_set=is_dense_gene_set_m,
        ps=ps_m,
        sigma2s=sigma2s_m,
        return_sample=True,
        max_num_burn_in=passed_in_max_num_burn_in,
        max_num_iter=max_num_iter_betas,
        min_num_iter=min_num_iter_betas,
        num_chains=num_chains_betas,
        r_threshold_burn_in=r_threshold_burn_in_betas,
        use_max_r_for_convergence=use_max_r_for_convergence_betas,
        max_frac_sem=max_frac_sem_betas,
        max_allowed_batch_correlation=max_allowed_batch_correlation,
        gauss_seidel=gauss_seidel_betas,
        update_hyper_sigma=False,
        update_hyper_p=False,
        num_missing_gene_sets=num_missing,
        sparse_solution=sparse_solution,
        sparse_frac_betas=sparse_frac_betas,
        betas_trace_out=betas_trace_out,
        debug_gene_sets=[state.gene_sets[i] for i in range(len(state.gene_sets)) if gene_set_mask_m[0,i]],
        init_betas=init_betas_m,
        init_postp=init_postp_m,
    )


def _initialize_gibbs_corrected_beta_output_matrices(
    default_betas_mean_m,
    default_betas_sample_m,
    default_postp_mean_m,
    default_postp_sample_m,
    debug_zero_sparse,
):
    if debug_zero_sparse:
        return (
            copy.copy(default_betas_mean_m),
            copy.copy(default_betas_sample_m),
            copy.copy(default_postp_mean_m),
            copy.copy(default_postp_sample_m),
        )
    return (
        np.zeros(default_betas_mean_m.shape),
        np.zeros(default_betas_sample_m.shape),
        np.zeros(default_postp_mean_m.shape),
        np.zeros(default_postp_sample_m.shape),
    )


def _prepare_gibbs_next_warm_start(
    warm_start,
    use_mean_betas,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
    full_betas_sample_m,
    full_betas_mean_m,
    full_postp_sample_m,
    full_postp_mean_m,
):
    if warm_start:
        # Filtered-out gene sets remain zero because the full_* matrices are
        # initialized to zero each iteration.
        if use_mean_betas:
            prev_warm_start_betas_m = copy.copy(full_betas_mean_m)
            prev_warm_start_postp_m = copy.copy(full_postp_mean_m)
        else:
            prev_warm_start_betas_m = copy.copy(full_betas_sample_m)
            prev_warm_start_postp_m = copy.copy(full_postp_sample_m)
    return (prev_warm_start_betas_m, prev_warm_start_postp_m)


def _maybe_refresh_gibbs_huge_scores(
    state,
    update_huge_scores,
    compute_Y_raw,
    priors_for_Y_m,
    log_bf_m,
    log_bf_uncorrected_m,
    log_bf_raw_m,
):
    if not (state.huge_signal_bfs is not None and update_huge_scores):
        return (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m)

    log("Updating HuGE scores")
    combined_optional_bf_terms = _combine_optional_gene_bf_terms(state.Y_exomes, state.Y_positive_controls, state.Y_case_counts)
    rel_prior_log_bf = priors_for_Y_m + combined_optional_bf_terms

    (log_bf_m, log_bf_uncorrected_m, absent_genes, _) = state._distill_huge_signal_bfs(state.huge_signal_bfs_for_regression, state.huge_signal_posteriors_for_regression, state.huge_signal_sum_gene_cond_probabilities_for_regression, state.huge_signal_mean_gene_pos_for_regression, state.huge_signal_max_closest_gene_prob, state.huge_cap_region_posterior, state.huge_scale_region_posterior, state.huge_phantom_region_posterior, state.huge_allow_evidence_of_absence, state.gene_covariates, state.gene_covariates_mask, state.gene_covariates_mat_inv, state.gene_covariate_names, state.gene_covariate_intercept_index, state.genes, rel_prior_log_bf=rel_prior_log_bf)

    if compute_Y_raw:
        (log_bf_raw_m, _, _, _) = state._distill_huge_signal_bfs(state.huge_signal_bfs, state.huge_signal_posteriors, state.huge_signal_sum_gene_cond_probabilities, state.huge_signal_mean_gene_pos, state.huge_signal_max_closest_gene_prob, state.huge_cap_region_posterior, state.huge_scale_region_posterior, state.huge_phantom_region_posterior, state.huge_allow_evidence_of_absence, state.gene_covariates, state.gene_covariates_mask, state.gene_covariates_mat_inv, state.gene_covariate_names, state.gene_covariate_intercept_index, state.genes, rel_prior_log_bf=rel_prior_log_bf)
    else:
        log_bf_raw_m = copy.copy(log_bf_m)

    # Distillation uses optional terms internally for locus resolution;
    # add them back on the total gene-level BF scale.
    _add_optional_gene_bf_terms(
        log_bf_m,
        log_bf_uncorrected_m,
        log_bf_raw_m,
        state.Y_exomes,
        state.Y_positive_controls,
        state.Y_case_counts,
    )

    if len(absent_genes) > 0:
        bail("Error: huge_signal_bfs was incorrectly set and contains extra genes")
    return (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m)


def _refresh_gibbs_iteration_priors_and_huge(
    state,
    warm_start,
    use_mean_betas,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
    full_betas_sample_m,
    full_betas_mean_m,
    full_postp_sample_m,
    full_postp_mean_m,
    priors_missing_sample_m,
    priors_missing_mean_m,
    priors_for_Y_m,
    update_huge_scores,
    compute_Y_raw,
    log_bf_m,
    log_bf_uncorrected_m,
    log_bf_raw_m,
):
    (prev_warm_start_betas_m, prev_warm_start_postp_m) = _prepare_gibbs_next_warm_start(
        warm_start=warm_start,
        use_mean_betas=use_mean_betas,
        prev_warm_start_betas_m=prev_warm_start_betas_m,
        prev_warm_start_postp_m=prev_warm_start_postp_m,
        full_betas_sample_m=full_betas_sample_m,
        full_betas_mean_m=full_betas_mean_m,
        full_postp_sample_m=full_postp_sample_m,
        full_postp_mean_m=full_postp_mean_m,
    )
    (
        priors_sample_m,
        priors_mean_m,
        priors_missing_sample_m,
        priors_missing_mean_m,
    ) = pigean_model.compute_gibbs_iteration_priors_from_betas(
        state=state,
        full_betas_sample_m=full_betas_sample_m,
        full_betas_mean_m=full_betas_mean_m,
        priors_missing_sample_m=priors_missing_sample_m,
        priors_missing_mean_m=priors_missing_mean_m,
    )
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = _maybe_refresh_gibbs_huge_scores(
        state=state,
        update_huge_scores=update_huge_scores,
        compute_Y_raw=compute_Y_raw,
        priors_for_Y_m=priors_for_Y_m,
        log_bf_m=log_bf_m,
        log_bf_uncorrected_m=log_bf_uncorrected_m,
        log_bf_raw_m=log_bf_raw_m,
    )

    return {
        "prev_warm_start_betas_m": prev_warm_start_betas_m,
        "prev_warm_start_postp_m": prev_warm_start_postp_m,
        "priors_sample_m": priors_sample_m,
        "priors_mean_m": priors_mean_m,
        "priors_missing_sample_m": priors_missing_sample_m,
        "priors_missing_mean_m": priors_missing_mean_m,
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
    }


def _update_gibbs_all_sums_and_maybe_restart_low_betas(
    state,
    epoch_runtime,
    epoch_sums,
    restart_controls,
    iteration_num,
    full_betas_mean_m,
):
    all_sum_betas_m = epoch_runtime["all_sum_betas_m"]
    all_sum_betas2_m = epoch_runtime["all_sum_betas2_m"]
    all_num_sum_m = epoch_runtime["all_num_sum_m"]

    all_sum_betas_m = np.add(all_sum_betas_m, full_betas_mean_m)
    all_sum_betas2_m = np.add(all_sum_betas2_m, np.power(full_betas_mean_m, 2))
    all_num_sum_m += 1

    R_beta_v = np.zeros(all_sum_betas_m.shape[1])

    low_beta_restart_update = _maybe_restart_gibbs_for_low_betas(
        state=state,
        increase_hyper_if_betas_below_for_epoch=restart_controls.increase_hyper_if_betas_below_for_epoch,
        experimental_hyper_mutation=restart_controls.experimental_hyper_mutation,
        num_before_checking_p_increase=restart_controls.num_before_checking_p_increase,
        p_scale_factor=restart_controls.p_scale_factor,
        epoch_runtime=epoch_runtime,
        epoch_sums=epoch_sums,
        num_mad=restart_controls.num_mad,
        num_attempts=restart_controls.num_attempts,
        max_num_attempt_restarts=restart_controls.max_num_attempt_restarts,
        iteration_num=iteration_num,
    )

    return GibbsAllIterationUpdate(
        all_sum_betas_m=all_sum_betas_m,
        all_sum_betas2_m=all_sum_betas2_m,
        all_num_sum_m=all_num_sum_m,
        R_beta_v=R_beta_v,
        gibbs_good=low_beta_restart_update.gibbs_good,
        num_p_increases=low_beta_restart_update.num_p_increases,
        should_break=low_beta_restart_update.should_break,
    )


_build_inner_beta_sampler_common_kwargs = pigean_model.build_inner_beta_sampler_common_kwargs


_configure_hyperparameters_for_main = functools.partial(
    pigean_runtime.configure_hyperparameters_for_main,
    read_gene_map_fn=functools.partial(
        pigean_y_inputs_core.read_gene_map,
        bail_fn=bail,
    ),
    init_gene_locs_fn=functools.partial(
        pigean_y_inputs_core.init_gene_locs,
        warn_fn=warn,
        bail_fn=bail,
        log_fn=log,
    ),
    bail_fn=bail,
    log_fn=log,
)

YPrimaryInputsContract = pigean_y_inputs.YPrimaryInputsContract
YReadContract = pigean_y_inputs.YReadContract
_build_main_y_read_contract = pigean_y_inputs.build_main_y_read_contract
_read_Y = functools.partial(
    pigean_y_inputs_core.read_y_pipeline,
    warn_fn=warn,
    bail_fn=bail,
    log_fn=log,
    trace_level=TRACE,
    apply_gene_covariates_and_correct_huge_fn=functools.partial(
        pigean_y_inputs_core.apply_gene_covariates_and_correct_huge,
        log_fn=log,
        trace_level=TRACE,
        bail_fn=bail,
    ),
)
_read_Y_from_contract = functools.partial(
    pigean_y_inputs_core.read_y_from_contract,
    read_y_fn=_read_Y,
    bail_fn=bail,
)
_run_read_y_stage = _read_Y
_run_read_y_contract_stage = _read_Y_from_contract


def _run_main_adaptive_read_x(state, options, mode_state, sigma2_cond):
    return pigean_x_inputs.run_main_adaptive_read_x(
        state,
        options,
        mode_state,
        sigma2_cond,
        build_xin_to_p_noninf_index_map_fn=pegs_build_xin_to_p_noninf_index_map,
        run_read_x_stage_fn=_run_read_x_stage,
        warn_fn=warn,
        bail_fn=bail,
        log_fn=log,
    )


def _run_read_x_stage(runtime, X_in, **read_x_kwargs):
    return pigean_x_inputs.run_read_x_stage(
        runtime,
        X_in,
        read_x_kwargs=read_x_kwargs,
        build_read_x_pipeline_config_fn=pegs_build_read_x_pipeline_config,
        bail_fn=bail,
        read_x_pipeline_fn=_read_x_pipeline,
    )


def _read_x_pipeline(runtime, read_x_pipeline_config):
    return pigean_x_inputs.read_x_pipeline(
        runtime,
        read_x_pipeline_config,
        open_gz_fn=open_gz,
        open_dense_fn=open,
        log_fn=log,
        info_level=INFO,
        debug_level=DEBUG,
        remove_tag_from_input_fn=pegs_remove_tag_from_input,
        record_read_x_counts_fn=pegs_record_read_x_counts,
        ensure_gene_universe_fn=_ensure_gene_universe_for_x,
        process_x_input_file_fn=_process_x_input_file,
        normalize_dense_gene_rows_fn=_normalize_dense_gene_rows,
        build_sparse_x_from_dense_input_fn=_build_sparse_x_from_dense_input,
        reindex_x_rows_to_current_genes_fn=_reindex_x_rows_to_current_genes,
        normalize_gene_set_weights_fn=_normalize_gene_set_weights,
        partition_missing_gene_rows_fn=_partition_missing_gene_rows,
        maybe_permute_gene_set_rows_fn=_maybe_permute_gene_set_rows,
        maybe_prefilter_x_block_fn=_maybe_prefilter_x_block,
        merge_missing_gene_rows_fn=_merge_missing_gene_rows,
        finalize_added_x_block_fn=_finalize_added_x_block,
        standardize_qc_metrics_after_x_read_fn=_standardize_qc_metrics_after_x_read,
        maybe_correct_gene_set_betas_after_x_read_fn=_maybe_correct_gene_set_betas_after_x_read,
        maybe_limit_initial_gene_sets_by_p_fn=_maybe_limit_initial_gene_sets_by_p,
        maybe_prune_gene_sets_after_x_read_fn=_maybe_prune_gene_sets_after_x_read,
        initialize_hyper_defaults_after_x_read_fn=_initialize_hyper_defaults_after_x_read,
        maybe_learn_batch_hyper_after_x_read_fn=_maybe_learn_batch_hyper_after_x_read,
        maybe_adjust_overaggressive_p_filter_after_x_read_fn=_maybe_adjust_overaggressive_p_filter_after_x_read,
        apply_post_read_gene_set_size_and_qc_filters_fn=_apply_post_read_gene_set_size_and_qc_filters,
        maybe_filter_zero_uncorrected_betas_after_x_read_fn=_maybe_filter_zero_uncorrected_betas_after_x_read,
        maybe_reduce_gene_sets_to_max_after_x_read_fn=_maybe_reduce_gene_sets_to_max_after_x_read,
    )


_normalize_dense_gene_rows = pigean_x_inputs_core.normalize_dense_gene_rows
_build_sparse_x_from_dense_input = functools.partial(
    pigean_x_inputs_core.build_sparse_x_from_dense_input,
    log_fn=log,
    debug_level=DEBUG,
    warn_fn=warn,
    bail_fn=bail,
    ext_tag=EXT_TAG,
    bot_tag=BOT_TAG,
    top_tag=TOP_TAG,
)
_process_x_input_file = functools.partial(
    pigean_x_inputs_core.process_x_input_file,
    open_gz_fn=open_gz,
    warn_fn=warn,
    bail_fn=bail,
    log_fn=log,
    debug_level=DEBUG,
    ext_tag=EXT_TAG,
    top_tag=TOP_TAG,
    bot_tag=BOT_TAG,
)
_normalize_gene_set_weights = pigean_x_inputs_core.normalize_gene_set_weights
_maybe_permute_gene_set_rows = pigean_x_inputs_core.maybe_permute_gene_set_rows
_build_prefilter_keep_mask = functools.partial(
    pigean_x_inputs_core.build_prefilter_keep_mask,
    log_fn=log,
    debug_level=DEBUG,
)
_maybe_prefilter_x_block = functools.partial(
    pigean_x_inputs_core.maybe_prefilter_x_block,
    log_fn=log,
    debug_level=DEBUG,
)
_merge_missing_gene_rows = pigean_x_inputs_core.merge_missing_gene_rows
_finalize_added_x_block = pigean_x_inputs_core.finalize_added_x_block
_partition_missing_gene_rows = functools.partial(
    pigean_x_inputs_core.partition_missing_gene_rows,
    bail_fn=bail,
)
_reindex_x_rows_to_current_genes = pigean_x_inputs_core.reindex_x_rows_to_current_genes
_ensure_gene_universe_for_x = functools.partial(
    pigean_x_inputs_core.ensure_gene_universe_for_x,
    open_gz_fn=open_gz,
    remove_tag_from_input_fn=pegs_remove_tag_from_input,
    log_fn=log,
    debug_level=DEBUG,
    bail_fn=bail,
)


_mode_requires_gene_scores = pigean_y_inputs.mode_requires_gene_scores


def _read_gene_phewas_bfs(
    state,
    gene_phewas_bfs_in,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    anchor_genes=None,
    anchor_phenos=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
    phewas_gene_to_X_gene_in=None,
    min_value=None,
    max_num_entries_at_once=None,
    **kwargs
):
    cached = dict(locals())
    cached.pop("state", None)
    cached.pop("kwargs", None)
    state.cached_gene_phewas_call = cached

    if gene_phewas_bfs_in is None:
        bail("Require --gene-stats-in or --gene-phewas-bfs-in for this operation")

    log("Reading --gene-phewas-bfs-in file %s" % gene_phewas_bfs_in, INFO)
    if state.genes is None:
        bail("Need to initialixe --X before reading gene_phewas")

    phewas_gene_to_X_gene = None
    if phewas_gene_to_X_gene_in is not None:
        phewas_gene_to_X_gene = pegs_parse_gene_map_file(
            phewas_gene_to_X_gene_in,
            allow_multi=True,
            bail_fn=bail,
        )

    pegs_load_and_apply_gene_phewas_bfs_to_runtime(
        state,
        gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        anchor_genes=anchor_genes,
        anchor_phenos=anchor_phenos,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        phewas_gene_to_x_gene=phewas_gene_to_X_gene,
        min_value=min_value,
        max_num_entries_at_once=max_num_entries_at_once,
        open_text_fn=open_gz,
        get_col_fn=_get_col,
        construct_map_to_ind_fn=pegs_construct_map_to_ind,
        warn_fn=warn,
        bail_fn=bail,
        log_fn=lambda message: log(message, DEBUG),
    )
    state.phewas_state = pegs_sync_phewas_runtime_state(state)


def _reread_gene_phewas_bfs(state):
    if state.cached_gene_phewas_call is None:
        return
    log("Rereading gene phewas bfs...")
    _read_gene_phewas_bfs(state, **state.cached_gene_phewas_call)


def _load_advanced_set_b_y_inputs(state, options):
    return pigean_y_inputs.load_advanced_set_b_y_inputs(
        state,
        options,
        read_gene_phewas_bfs_fn=_read_gene_phewas_bfs,
        bail_fn=bail,
    )


def _load_main_Y_inputs(state, options, mode_state):
    return pigean_y_inputs.load_main_y_inputs(
        state,
        options,
        mode_state,
        run_read_y_stage_fn=_run_read_y_stage,
        run_read_y_contract_stage_fn=_run_read_y_contract_stage,
        read_gene_phewas_bfs_fn=_read_gene_phewas_bfs,
        bail_fn=bail,
    )
def _run_advanced_set_b_output_phewas_if_requested(state, options):
    try:
        from pigean import phewas as _pigean_phewas
    except ImportError:
        import pigean_phewas as _pigean_phewas

    return _pigean_phewas.run_advanced_set_b_output_phewas_if_requested(
        pigean_main_support.build_legacy_services(sys.modules[__name__]),
        state,
        options,
    )


def main(argv=None):
    try:
        should_continue = _bootstrap_cli(argv)
        if not should_continue:
            return 0
        try:
            from pigean import app as _pigean_app
        except ImportError:
            import pigean_app as _pigean_app
        _pigean_app.run_main_pipeline(
            options,
            mode,
            services=pigean_main_support.build_legacy_services(sys.modules[__name__]),
        )
        return 0
    except PegsCliError as exc:
        return pegs_handle_cli_exception(exc, argv=argv, debug_level=debug_level)
    except Exception as exc:
        return pegs_handle_unexpected_exception(exc, argv=argv, debug_level=debug_level)


if __name__ == '__main__':
    raise SystemExit(main())
