"""Active transitional EAGGL inner core.

This file still owns deeper matrix/state behavior that has not yet been
drained into package modules. The current policy is:

1. keep workflow selection, factor/PheWAS orchestration, and new helpers in
   package modules under `src/eaggl/`
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

import urllib.error
import urllib.request

import optparse
import sys
import time
import os
import copy
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

import pegs_cli_errors as pegs_cli_errors
import pegs_shared.bundle as pegs_bundle
import pegs_shared.cli as pegs_cli
import pegs_shared.gene_io as pegs_gene_io
import pegs_shared.phewas as pegs_phewas
import pegs_shared.regression as pegs_regression
import pegs_shared.runtime_matrix as pegs_runtime_matrix
import pegs_shared.types as pegs_types
import pegs_shared.xdata as pegs_xdata
import pegs_shared.ydata as pegs_ydata
import pegs_utils as pegs_utils_mod

DataValidationError = pegs_cli_errors.DataValidationError
PegsCliError = pegs_cli_errors.PegsCliError
pegs_handle_cli_exception = pegs_cli_errors.handle_cli_exception
pegs_handle_unexpected_exception = pegs_cli_errors.handle_unexpected_exception

PegsXReadConfig = pegs_types.XReadConfig
PegsXReadCallbacks = pegs_types.XReadCallbacks
PegsXReadPostCallbacks = pegs_types.XReadPostCallbacks

pegs_apply_cli_config_overrides = pegs_cli.apply_cli_config_overrides
pegs_callback_set_comma_separated_args = pegs_cli.callback_set_comma_separated_args
pegs_callback_set_comma_separated_args_as_set = pegs_cli.callback_set_comma_separated_args_as_set
pegs_coerce_option_int_list = pegs_cli.coerce_option_int_list
pegs_configure_random_seed = pegs_cli.configure_random_seed
pegs_emit_stderr_warning = pegs_cli.emit_stderr_warning
pegs_fail_removed_cli_aliases = pegs_cli.fail_removed_cli_aliases
pegs_format_removed_option_message = pegs_cli.format_removed_option_message
pegs_harmonize_cli_mode_args = pegs_cli.harmonize_cli_mode_args
pegs_initialize_cli_logging = pegs_cli.initialize_cli_logging
pegs_is_path_like_dest = pegs_cli.is_path_like_dest
pegs_is_remote_path = pegs_cli.is_remote_path
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

pegs_accumulate_factor_phewas_outputs = pegs_phewas.accumulate_factor_phewas_outputs
pegs_accumulate_standard_phewas_outputs = pegs_phewas.accumulate_standard_phewas_outputs
pegs_append_phewas_metric_block = pegs_phewas.append_phewas_metric_block
pegs_prepare_phewas_phenos_from_file = pegs_phewas.prepare_phewas_phenos_from_file
pegs_read_phewas_file_batch = pegs_phewas.read_phewas_file_batch
pegs_derive_factor_anchor_masks = pegs_phewas.derive_factor_anchor_masks
pegs_resolve_gene_phewas_input_decision_for_stage = pegs_phewas.resolve_gene_phewas_input_decision_for_stage
pegs_build_phewas_stage_config = pegs_phewas.build_phewas_stage_config

pegs_finalize_regression_outputs = pegs_regression.finalize_regression_outputs
pegs_compute_beta_tildes = pegs_regression.compute_beta_tildes
pegs_compute_multivariate_beta_tildes = pegs_regression.compute_multivariate_beta_tildes

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
pegs_load_and_apply_gene_phewas_bfs_to_runtime = pegs_utils_mod.load_and_apply_gene_phewas_bfs_to_runtime
pegs_load_and_apply_gene_set_statistics_to_runtime = pegs_utils_mod.load_and_apply_gene_set_statistics_to_runtime
pegs_load_and_apply_gene_set_phewas_statistics_to_runtime = pegs_utils_mod.load_and_apply_gene_set_phewas_statistics_to_runtime
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
pegs_write_factor_phewas_statistics = pegs_utils_mod.write_factor_phewas_statistics
pegs_remove_tag_from_input = pegs_utils_mod.remove_tag_from_input
pegs_clean_chrom_name = pegs_utils_mod.clean_chrom_name
pegs_parse_gene_map_file = pegs_utils_mod.parse_gene_map_file
pegs_read_loc_file_with_gene_map = pegs_utils_mod.read_loc_file_with_gene_map
pegs_complete_p_beta_se = pegs_utils_mod.complete_p_beta_se
pegs_construct_map_to_ind = pegs_utils_mod.construct_map_to_ind
pegs_open_text_with_retry = pegs_utils_mod.open_text_with_retry
pegs_resolve_column_index = pegs_utils_mod.resolve_column_index

pegs_load_and_apply_bundle_defaults = pegs_bundle.load_and_apply_bundle_defaults
PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS = pegs_bundle.EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS
PEGS_EAGGL_BUNDLE_SCHEMA = pegs_bundle.EAGGL_BUNDLE_SCHEMA

# Canonical suffix tags used when expanding dense gene-set inputs into
# sparse derived sets (top/ext/bottom thresholds).
EXT_TAG = "ext"
BOT_TAG = "bot"
TOP_TAG = "top"

def bail(message):
    raise DataValidationError(message)

try:
    from . import cli as _eaggl_cli
except ImportError:
    import cli as _eaggl_cli
try:
    from . import domain as _eaggl_domain
except ImportError:
    import domain as _eaggl_domain
try:
    from . import io as _eaggl_io
except ImportError:
    import io as _eaggl_io
try:
    from . import y_inputs as _eaggl_y_inputs
except ImportError:
    import y_inputs as _eaggl_y_inputs
try:
    from . import factor_runtime as _eaggl_factor_runtime
except ImportError:
    import factor_runtime as _eaggl_factor_runtime
try:
    from . import phewas as _eaggl_phewas
except ImportError:
    import phewas as _eaggl_phewas
try:
    from . import regression as _eaggl_regression
except ImportError:
    import regression as _eaggl_regression

usage = _eaggl_cli.usage
parser = _eaggl_cli.parser
REMOVED_OPTION_REPLACEMENTS = _eaggl_cli.REMOVED_OPTION_REPLACEMENTS
query_lmm = _eaggl_cli.query_lmm
_classify_factor_workflow = _eaggl_cli._classify_factor_workflow
_FACTOR_WORKFLOW_STRATEGY_META = _eaggl_cli._FACTOR_WORKFLOW_STRATEGY_META
try:
    from . import workflows as _eaggl_workflows
except ImportError:
    import workflows as _eaggl_workflows
try:
    from . import factor as _eaggl_factor
except ImportError:
    import factor as _eaggl_factor
try:
    from . import outputs as _eaggl_outputs
except ImportError:
    import outputs as _eaggl_outputs
try:
    from . import labeling as _eaggl_labeling
except ImportError:
    import labeling as _eaggl_labeling

options = None
args = []
mode = None
config_mode = None
cli_specified_dests = set()
config_specified_dests = set()
eaggl_bundle_info = None
run_factor = False
run_phewas = False
run_naive_factor = False
use_phewas_for_factoring = False
factor_gene_set_x_pheno = False
expand_gene_sets = False
factor_workflow = None
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


def _bootstrap_cli(argv=None):
    global options, args, mode, config_mode, cli_specified_dests, config_specified_dests
    global eaggl_bundle_info, run_factor, run_phewas, run_naive_factor
    global use_phewas_for_factoring, factor_gene_set_x_pheno, expand_gene_sets, factor_workflow
    global NONE, INFO, DEBUG, TRACE, debug_level, log_fh, warnings_fh, log, warn

    should_continue = _eaggl_cli._bootstrap_cli(argv)
    options = _eaggl_cli.options
    args = _eaggl_cli.args
    mode = _eaggl_cli.mode
    config_mode = _eaggl_cli.config_mode
    cli_specified_dests = _eaggl_cli.cli_specified_dests
    config_specified_dests = _eaggl_cli.config_specified_dests
    eaggl_bundle_info = _eaggl_cli.eaggl_bundle_info
    run_factor = _eaggl_cli.run_factor
    run_phewas = _eaggl_cli.run_phewas
    run_naive_factor = _eaggl_cli.run_naive_factor
    use_phewas_for_factoring = _eaggl_cli.use_phewas_for_factoring
    factor_gene_set_x_pheno = _eaggl_cli.factor_gene_set_x_pheno
    expand_gene_sets = _eaggl_cli.expand_gene_sets
    factor_workflow = _eaggl_cli.factor_workflow
    NONE = _eaggl_cli.NONE
    INFO = _eaggl_cli.INFO
    DEBUG = _eaggl_cli.DEBUG
    TRACE = _eaggl_cli.TRACE
    debug_level = _eaggl_cli.debug_level
    log_fh = _eaggl_cli.log_fh
    warnings_fh = _eaggl_cli.warnings_fh
    log = _eaggl_cli.log
    warn = _eaggl_cli.warn
    _eaggl_state.bind_runtime_namespace(sys.modules[__name__])
    return should_continue


def open_gz(file, flag=None):
    return pegs_open_text_with_retry(
        file,
        flag=flag,
        log_fn=lambda message: log(message, INFO),
        bail_fn=bail,
    )


_HYPERPARAMETER_PROXY_FIELDS = (
    "p",
    "sigma2",
    "sigma_power",
    "sigma2_osc",
    "sigma2_se",
    "sigma2_p",
    "sigma2_total_var",
    "sigma2_total_var_lower",
    "sigma2_total_var_upper",
    "ps",
    "sigma2s",
    "sigma2s_missing",
)



try:
    from . import state as _eaggl_state
except ImportError:
    import state as _eaggl_state

_eaggl_state.bind_runtime_namespace(sys.modules[__name__])

_bind_hyperparameter_properties = _eaggl_state._bind_hyperparameter_properties
_append_with_any_user = _eaggl_state._append_with_any_user
EagglState = _eaggl_state.EagglState

def _build_main_domain():
    return _eaggl_domain.build_main_domain(sys.modules[__name__])


FactorOnlyStageResult = _eaggl_factor.FactorOnlyStageResult
PhewasStageResult = _eaggl_factor.PhewasStageResult
FactorStageResult = _eaggl_factor.FactorStageResult
FactorWorkflow = _eaggl_factor.FactorWorkflow
FactorInputs = _eaggl_factor.FactorInputs
FactorExecutionConfig = _eaggl_factor.FactorExecutionConfig
FactorOutputPlan = _eaggl_outputs.FactorOutputPlan
MainPipelineResult = _eaggl_factor.MainPipelineResult


_bind_hyperparameter_properties(EagglState)


_read_y_pipeline = _eaggl_y_inputs.read_y_pipeline
_run_read_y_stage = _eaggl_y_inputs.run_read_y_stage
_read_x_pipeline = _eaggl_io.read_x_pipeline
_run_read_x_stage = _eaggl_io.run_read_x_stage
_log_runtime_environment_if_requested = _eaggl_io.log_runtime_environment_if_requested
_read_gene_map = _eaggl_io.read_gene_map
_init_gene_locs = _eaggl_io.init_gene_locs
_initialize_main_mappings = _eaggl_io.initialize_main_mappings
_read_gene_set_statistics = _eaggl_io.read_gene_set_statistics
_read_gene_set_phewas_statistics = _eaggl_io.read_gene_set_phewas_statistics
_derive_factor_anchor_masks = _eaggl_io.derive_factor_anchor_masks
_read_gene_phewas_bfs = _eaggl_io.read_gene_phewas_bfs
_has_loaded_gene_phewas = _eaggl_io.has_loaded_gene_phewas


_extract_factor_workflow = _eaggl_factor.extract_factor_workflow
_extract_factor_inputs = _eaggl_factor.extract_factor_inputs
_resolve_factor_gene_or_pheno_filter_value = _eaggl_factor.resolve_factor_gene_or_pheno_filter_value
_build_factor_execution_config = _eaggl_factor.build_factor_execution_config
_run_factor_model = _eaggl_factor.run_factor_model
_build_factor_output_plan = _eaggl_outputs.build_factor_output_plan
_write_factor_outputs_for_plan = _eaggl_outputs.write_factor_outputs_for_plan


# Compatibility wrappers preserved for existing tests and direct imports.
def _run_main_factor_stage(g, options, mode_state, factor_input_state):
    return _eaggl_factor.run_main_factor_stage(_build_main_domain(), g, options, mode_state, factor_input_state)


def _write_main_factor_outputs(g, options):
    return _eaggl_outputs.write_main_factor_outputs(g, options)


def _reread_gene_phewas_bfs(state):
    return _eaggl_io.reread_gene_phewas_bfs(_build_main_domain(), state)


def _run_main_phewas_stage(g, options):
    return _eaggl_factor.run_main_phewas_stage(_build_main_domain(), g, options)


def _run_main_factor_phewas_stage(g, options):
    return _eaggl_factor.run_main_factor_phewas_stage(_build_main_domain(), g, options)



_normalize_dense_gene_rows = _eaggl_state._normalize_dense_gene_rows
_build_sparse_x_from_dense_input = _eaggl_state._build_sparse_x_from_dense_input
_estimate_dense_chunk_size = _eaggl_state._estimate_dense_chunk_size
_record_x_addition = _eaggl_state._record_x_addition
_process_dense_x_file = _eaggl_state._process_dense_x_file
_process_sparse_x_file = _eaggl_state._process_sparse_x_file
_process_x_input_file = _eaggl_state._process_x_input_file
_standardize_qc_metrics_after_x_read = _eaggl_state._standardize_qc_metrics_after_x_read
_maybe_correct_gene_set_betas_after_x_read = _eaggl_state._maybe_correct_gene_set_betas_after_x_read
_maybe_limit_initial_gene_sets_by_p = _eaggl_state._maybe_limit_initial_gene_sets_by_p
_maybe_prune_gene_sets_after_x_read = _eaggl_state._maybe_prune_gene_sets_after_x_read
_initialize_hyper_defaults_after_x_read = _eaggl_state._initialize_hyper_defaults_after_x_read
_learn_hyper_for_gene_set_batch = _eaggl_state._learn_hyper_for_gene_set_batch
_apply_learned_batch_hyper_values = _eaggl_state._apply_learned_batch_hyper_values
_finalize_batch_hyper_vectors = _eaggl_state._finalize_batch_hyper_vectors
_maybe_learn_batch_hyper_after_x_read = _eaggl_state._maybe_learn_batch_hyper_after_x_read
_maybe_adjust_overaggressive_p_filter_after_x_read = _eaggl_state._maybe_adjust_overaggressive_p_filter_after_x_read
_apply_post_read_gene_set_size_and_qc_filters = _eaggl_state._apply_post_read_gene_set_size_and_qc_filters
_maybe_filter_zero_uncorrected_betas_after_x_read = _eaggl_state._maybe_filter_zero_uncorrected_betas_after_x_read
_maybe_reduce_gene_sets_to_max_after_x_read = _eaggl_state._maybe_reduce_gene_sets_to_max_after_x_read
GeneSetData = _eaggl_state.GeneSetData


def run_main_pipeline(options):
    try:
        from . import dispatch as _eaggl_dispatch
    except ImportError:
        import dispatch as _eaggl_dispatch

    return _eaggl_dispatch.run_main_pipeline(_build_main_domain(), options)


def main(argv=None):
    try:
        should_continue = _bootstrap_cli(argv)
        if not should_continue:
            return 0
        run_main_pipeline(options)
        return 0
    except PegsCliError as exc:
        return pegs_handle_cli_exception(exc, argv=argv, debug_level=debug_level)
    except Exception as exc:
        return pegs_handle_unexpected_exception(exc, argv=argv, debug_level=debug_level)


if __name__ == '__main__':

    #profiler = cProfile.Profile()
    #profiler.enable()

    #cProfile.run('main()')
    raise SystemExit(main())

    #profiler.disable()
    #profiler.dump_stats('output.prof')
