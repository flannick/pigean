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


def _state_bail(message):
    return bail(message)


pigean_state.bail = _state_bail

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


def _state_log(*args, **kwargs):
    return log(*args, **kwargs)


def _state_warn(*args, **kwargs):
    return warn(*args, **kwargs)


pigean_state.log = _state_log
pigean_state.warn = _state_warn

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


GibbsAllIterationUpdate = pigean_state.GibbsAllIterationUpdate
GibbsEpochIterationLoopConfig = pigean_state.GibbsEpochIterationLoopConfig
GibbsEpochIterationStaticConfig = pigean_state.GibbsEpochIterationStaticConfig
GibbsEpochPhaseConfig = pigean_state.GibbsEpochPhaseConfig
GibbsEpochRuntimeConfigs = pigean_state.GibbsEpochRuntimeConfigs
GibbsIterationCorrectionConfig = pigean_state.GibbsIterationCorrectionConfig
GibbsIterationProgressRuntimeConfig = pigean_state.GibbsIterationProgressRuntimeConfig
GibbsIterationRuntimeConfigs = pigean_state.GibbsIterationRuntimeConfigs
GibbsIterationUpdateConfig = pigean_state.GibbsIterationUpdateConfig
GibbsLowBetaRestartControls = pigean_state.GibbsLowBetaRestartControls
GibbsLowBetaRestartUpdate = pigean_state.GibbsLowBetaRestartUpdate
GibbsRunControls = pigean_state.GibbsRunControls
GibbsRunState = pigean_state.GibbsRunState
_IntervalTree = pigean_state._IntervalTree
_accumulate_gibbs_post_burn_iteration = pigean_state._accumulate_gibbs_post_burn_iteration
_add_optional_gene_bf_terms = pigean_state._add_optional_gene_bf_terms
_advance_gibbs_post_burn_state = pigean_state._advance_gibbs_post_burn_state
_apply_gibbs_all_iteration_update = pigean_state._apply_gibbs_all_iteration_update
_apply_gibbs_block_prefilter_pruning = pigean_state._apply_gibbs_block_prefilter_pruning
_apply_gibbs_burn_in_control_update = pigean_state._apply_gibbs_burn_in_control_update
_apply_gibbs_control_update = pigean_state._apply_gibbs_control_update
_apply_gibbs_final_state = pigean_state._apply_gibbs_final_state
_apply_gibbs_logistic_batch_outputs = pigean_state._apply_gibbs_logistic_batch_outputs
_apply_gibbs_post_burn_control_update = pigean_state._apply_gibbs_post_burn_control_update
_apply_inner_beta_sparsity_update = pigean_state._apply_inner_beta_sparsity_update
_apply_learned_batch_hyper_values = pigean_state._apply_learned_batch_hyper_values
_apply_post_read_gene_set_size_and_qc_filters = pigean_state._apply_post_read_gene_set_size_and_qc_filters
_apply_prior_update_to_epoch_priors = pigean_state._apply_prior_update_to_epoch_priors
_apply_refresh_update_to_epoch_priors = pigean_state._apply_refresh_update_to_epoch_priors
_augment_gibbs_iteration_state_with_uncorrected_and_mask = pigean_state._augment_gibbs_iteration_state_with_uncorrected_and_mask
_autodetect_gwas_columns = pigean_state._autodetect_gwas_columns
_build_finalize_gibbs_prior_inputs = pigean_state._build_finalize_gibbs_prior_inputs
_build_gibbs_burn_in_control_from_epoch = pigean_state._build_gibbs_burn_in_control_from_epoch
_build_gibbs_burn_in_control_update = pigean_state._build_gibbs_burn_in_control_update
_build_gibbs_diag_sums = pigean_state._build_gibbs_diag_sums
_build_gibbs_iteration_progress_context = pigean_state._build_gibbs_iteration_progress_context
_build_gibbs_low_beta_restart_controls = pigean_state._build_gibbs_low_beta_restart_controls
_build_gibbs_post_burn_accumulation_payload = pigean_state._build_gibbs_post_burn_accumulation_payload
_build_gibbs_post_burn_control_from_epoch = pigean_state._build_gibbs_post_burn_control_from_epoch
_build_gibbs_post_burn_control_update = pigean_state._build_gibbs_post_burn_control_update
_build_gibbs_prefilter_sparse_V = pigean_state._build_gibbs_prefilter_sparse_V
_build_gibbs_record_config = pigean_state._build_gibbs_record_config
_build_post_burn_action_config = pigean_state._build_post_burn_action_config
_build_post_burn_stall_tracking_config = pigean_state._build_post_burn_stall_tracking_config
_build_post_stall_snapshot_arrays = pigean_state._build_post_stall_snapshot_arrays
_build_refresh_gibbs_iteration_inputs = pigean_state._build_refresh_gibbs_iteration_inputs
_calculate_r_tensor_from_chain_sums = pigean_state._calculate_r_tensor_from_chain_sums
_calculate_rhat_from_sums = pigean_state._calculate_rhat_from_sums
_combine_optional_gene_bf_terms = pigean_state._combine_optional_gene_bf_terms
_compute_burn_in_active_beta_rhat_stats = pigean_state._compute_burn_in_active_beta_rhat_stats
_compute_burn_in_window_plateau_status = pigean_state._compute_burn_in_window_plateau_status
_compute_gibbs_corrected_betas_for_gene_set_mask = pigean_state._compute_gibbs_corrected_betas_for_gene_set_mask
_compute_gibbs_iteration_betas_and_priors = pigean_state._compute_gibbs_iteration_betas_and_priors
_compute_gibbs_iteration_y_terms = pigean_state._compute_gibbs_iteration_y_terms
_compute_gibbs_linear_prefilter_betas = pigean_state._compute_gibbs_linear_prefilter_betas
_compute_gibbs_logistic_beta_tildes = pigean_state._compute_gibbs_logistic_beta_tildes
_compute_gibbs_logistic_beta_tildes_batch = pigean_state._compute_gibbs_logistic_beta_tildes_batch
_compute_gibbs_logistic_outputs_for_batches = pigean_state._compute_gibbs_logistic_outputs_for_batches
_compute_gibbs_post_burn_diag_metrics = pigean_state._compute_gibbs_post_burn_diag_metrics
_compute_gibbs_pre_gene_set_filter_mask = pigean_state._compute_gibbs_pre_gene_set_filter_mask
_compute_gibbs_y_corr_sparse = pigean_state._compute_gibbs_y_corr_sparse
_compute_huge_variant_thresholds = pigean_state._compute_huge_variant_thresholds
_compute_inner_beta_hyper_update_targets = pigean_state._compute_inner_beta_hyper_update_targets
_compute_post_burn_beta_diagnostics = pigean_state._compute_post_burn_beta_diagnostics
_compute_post_burn_gene_diagnostics = pigean_state._compute_post_burn_gene_diagnostics
_compute_recent_burn_in_stall_beta_rhat = pigean_state._compute_recent_burn_in_stall_beta_rhat
_compute_recent_post_stall_metrics = pigean_state._compute_recent_post_stall_metrics
_decide_gibbs_post_burn_action = pigean_state._decide_gibbs_post_burn_action
_determine_columns_from_file = pigean_state._determine_columns_from_file
_end_gibbs_burn_in = pigean_state._end_gibbs_burn_in
_evaluate_burn_in_diagnostics = pigean_state._evaluate_burn_in_diagnostics
_evaluate_gibbs_low_beta_condition = pigean_state._evaluate_gibbs_low_beta_condition
_evaluate_gibbs_post_burn_diagnostics_and_decision = pigean_state._evaluate_gibbs_post_burn_diagnostics_and_decision
_evaluate_post_stall_status = pigean_state._evaluate_post_stall_status
_finalize_batch_hyper_vectors = pigean_state._finalize_batch_hyper_vectors
_finalize_gibbs_iteration_progress = pigean_state._finalize_gibbs_iteration_progress
_get_active_beta_mask = pigean_state._get_active_beta_mask
_get_col = pigean_state._get_col
_get_gibbs_chain_batch_bounds = pigean_state._get_gibbs_chain_batch_bounds
_get_gibbs_gene_set_mask = pigean_state._get_gibbs_gene_set_mask
_handle_gibbs_burn_in_diag_path = pigean_state._handle_gibbs_burn_in_diag_path
_handle_gibbs_burn_in_gauss_seidel_path = pigean_state._handle_gibbs_burn_in_gauss_seidel_path
_handle_gibbs_burn_in_max_iter = pigean_state._handle_gibbs_burn_in_max_iter
_has_gibbs_epoch_aggregates = pigean_state._has_gibbs_epoch_aggregates
_initialize_gibbs_corrected_beta_output_matrices = pigean_state._initialize_gibbs_corrected_beta_output_matrices
_initialize_gibbs_epoch_control_state = pigean_state._initialize_gibbs_epoch_control_state
_initialize_gibbs_epoch_priors_state = pigean_state._initialize_gibbs_epoch_priors_state
_initialize_gibbs_epoch_runtime_state = pigean_state._initialize_gibbs_epoch_runtime_state
_initialize_gibbs_epoch_state = pigean_state._initialize_gibbs_epoch_state
_initialize_gibbs_epoch_sums_state = pigean_state._initialize_gibbs_epoch_sums_state
_initialize_gibbs_run_state = pigean_state._initialize_gibbs_run_state
_initialize_hyper_defaults_after_x_read = pigean_state._initialize_hyper_defaults_after_x_read
_learn_hyper_for_gene_set_batch = pigean_state._learn_hyper_for_gene_set_batch
_load_huge_gene_and_exon_locations = pigean_state._load_huge_gene_and_exon_locations
_log_gibbs_configuration_summary = pigean_state._log_gibbs_configuration_summary
_log_gibbs_logistic_divergence = pigean_state._log_gibbs_logistic_divergence
_log_gibbs_overlapping_corrected_beta_details = pigean_state._log_gibbs_overlapping_corrected_beta_details
_log_gibbs_post_burn_diagnostics = pigean_state._log_gibbs_post_burn_diagnostics
_maybe_adjust_overaggressive_p_filter_after_x_read = pigean_state._maybe_adjust_overaggressive_p_filter_after_x_read
_maybe_correct_gene_set_betas_after_x_read = pigean_state._maybe_correct_gene_set_betas_after_x_read
_maybe_correct_gibbs_logistic_beta_tildes = pigean_state._maybe_correct_gibbs_logistic_beta_tildes
_maybe_end_gibbs_epoch_for_post_burn_cap = pigean_state._maybe_end_gibbs_epoch_for_post_burn_cap
_maybe_filter_zero_uncorrected_betas_after_x_read = pigean_state._maybe_filter_zero_uncorrected_betas_after_x_read
_maybe_increase_gibbs_hyper_and_restart = pigean_state._maybe_increase_gibbs_hyper_and_restart
_maybe_learn_batch_hyper_after_x_read = pigean_state._maybe_learn_batch_hyper_after_x_read
_maybe_limit_initial_gene_sets_by_p = pigean_state._maybe_limit_initial_gene_sets_by_p
_maybe_log_gibbs_conditional_variance = pigean_state._maybe_log_gibbs_conditional_variance
_maybe_prune_gene_sets_after_x_read = pigean_state._maybe_prune_gene_sets_after_x_read
_maybe_reduce_gene_sets_to_max_after_x_read = pigean_state._maybe_reduce_gene_sets_to_max_after_x_read
_maybe_refresh_gibbs_huge_scores = pigean_state._maybe_refresh_gibbs_huge_scores
_maybe_restart_gibbs_for_low_betas = pigean_state._maybe_restart_gibbs_for_low_betas
_maybe_write_gibbs_gene_set_stats_trace = pigean_state._maybe_write_gibbs_gene_set_stats_trace
_maybe_write_gibbs_gene_stats_trace = pigean_state._maybe_write_gibbs_gene_stats_trace
_means_from_sums = pigean_state._means_from_sums
_merge_gibbs_post_burn_control_updates = pigean_state._merge_gibbs_post_burn_control_updates
_needs_gwas_column_detection = pigean_state._needs_gwas_column_detection
_normalize_gibbs_epoch_iteration_controls = pigean_state._normalize_gibbs_epoch_iteration_controls
_normalize_gibbs_gene_set_mask_across_chains = pigean_state._normalize_gibbs_gene_set_mask_across_chains
_normalize_gibbs_run_controls = pigean_state._normalize_gibbs_run_controls
_outlier_resistant_mean = pigean_state._outlier_resistant_mean
_prepare_gibbs_corrected_batch_inputs = pigean_state._prepare_gibbs_corrected_batch_inputs
_prepare_gibbs_corrected_per_chain_v_inputs = pigean_state._prepare_gibbs_corrected_per_chain_v_inputs
_prepare_gibbs_corrected_run_one_v_inputs = pigean_state._prepare_gibbs_corrected_run_one_v_inputs
_prepare_gibbs_epoch_attempt = pigean_state._prepare_gibbs_epoch_attempt
_prepare_gibbs_gene_set_mask_with_prefilter = pigean_state._prepare_gibbs_gene_set_mask_with_prefilter
_prepare_gibbs_iteration_inputs = pigean_state._prepare_gibbs_iteration_inputs
_prepare_gibbs_next_warm_start = pigean_state._prepare_gibbs_next_warm_start
_prepare_stall_indices = pigean_state._prepare_stall_indices
_read_huge_statistics_bundle = pigean_state._read_huge_statistics_bundle
_record_gibbs_configuration_params = pigean_state._record_gibbs_configuration_params
_record_gibbs_hyper_mutation_event = pigean_state._record_gibbs_hyper_mutation_event
_refresh_gibbs_iteration_priors_and_huge = pigean_state._refresh_gibbs_iteration_priors_and_huge
_reset_gibbs_post_burn_accumulators = pigean_state._reset_gibbs_post_burn_accumulators
_resolve_epoch_iteration_budget = pigean_state._resolve_epoch_iteration_budget
_resolve_post_stall_indices = pigean_state._resolve_post_stall_indices
_run_due_gibbs_post_burn_diagnostics = pigean_state._run_due_gibbs_post_burn_diagnostics
_run_gibbs_corrected_beta_sampler = pigean_state._run_gibbs_corrected_beta_sampler
_run_gibbs_corrected_betas_step = pigean_state._run_gibbs_corrected_betas_step
_run_optional_gibbs_post_burn_diagnostics = pigean_state._run_optional_gibbs_post_burn_diagnostics
_safe_quantile = pigean_state._safe_quantile
_sample_gibbs_iteration_y_state = pigean_state._sample_gibbs_iteration_y_state
_sample_gibbs_p_targets = pigean_state._sample_gibbs_p_targets
_sanitize_gibbs_diagnostic_controls = pigean_state._sanitize_gibbs_diagnostic_controls
_should_run_gibbs_post_burn_diagnostics = pigean_state._should_run_gibbs_post_burn_diagnostics
_snapshot_pre_gibbs_state = pigean_state._snapshot_pre_gibbs_state
_stack_gibbs_epoch_aggregates = pigean_state._stack_gibbs_epoch_aggregates
_standardize_qc_metrics_after_x_read = pigean_state._standardize_qc_metrics_after_x_read
_store_gibbs_corrected_batch_results = pigean_state._store_gibbs_corrected_batch_results
_summarize_gibbs_chain_aggregates = pigean_state._summarize_gibbs_chain_aggregates
_trim_post_stall_history_windows = pigean_state._trim_post_stall_history_windows
_trim_stall_histories = pigean_state._trim_stall_histories
_unpack_post_burn_stall_tracking_config = pigean_state._unpack_post_burn_stall_tracking_config
_update_gibbs_all_sums_and_maybe_restart_low_betas = pigean_state._update_gibbs_all_sums_and_maybe_restart_low_betas
_update_gibbs_burn_in_state = pigean_state._update_gibbs_burn_in_state
_update_gibbs_post_burn_precision_streak = pigean_state._update_gibbs_post_burn_precision_streak
_update_gibbs_post_burn_state = pigean_state._update_gibbs_post_burn_state
_update_inner_beta_gene_set_batch = pigean_state._update_inner_beta_gene_set_batch
_update_inner_beta_rhat_and_outliers = pigean_state._update_inner_beta_rhat_and_outliers
_update_post_burn_stall_tracking = pigean_state._update_post_burn_stall_tracking
_update_post_stall_best_histories = pigean_state._update_post_stall_best_histories
_validate_and_normalize_huge_gwas_inputs = pigean_state._validate_and_normalize_huge_gwas_inputs
_write_gene_set_stats_trace_rows = pigean_state._write_gene_set_stats_trace_rows
_write_gene_stats_trace_rows = pigean_state._write_gene_stats_trace_rows
_write_gibbs_iteration_gene_set_stats_trace = pigean_state._write_gibbs_iteration_gene_set_stats_trace
_write_huge_statistics_bundle = pigean_state._write_huge_statistics_bundle
_write_inner_beta_trace_rows = pigean_state._write_inner_beta_trace_rows
_zero_arrays = pigean_state._zero_arrays

_build_inner_beta_sampler_common_kwargs = pigean_state._build_inner_beta_sampler_common_kwargs
_configure_hyperparameters_for_main = pigean_state._configure_hyperparameters_for_main

YPrimaryInputsContract = pigean_y_inputs.YPrimaryInputsContract
YReadContract = pigean_y_inputs.YReadContract
_build_main_y_read_contract = pigean_y_inputs.build_main_y_read_contract
_read_Y = pigean_state._read_Y
_read_Y_from_contract = pigean_state._read_Y_from_contract
_run_read_y_stage = pigean_state._run_read_y_stage
_run_read_y_contract_stage = pigean_state._run_read_y_contract_stage
_set_const_Y = pigean_y_inputs_core.set_const_Y

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
