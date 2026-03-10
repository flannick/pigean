from __future__ import annotations

import sys

from . import cli as eaggl_cli
from . import domain as eaggl_domain
from . import factor as eaggl_factor
from . import io as eaggl_io
from . import outputs as eaggl_outputs
from . import state as eaggl_state
from . import y_inputs as eaggl_y_inputs


usage = eaggl_cli.usage
parser = eaggl_cli.parser
REMOVED_OPTION_REPLACEMENTS = eaggl_cli.REMOVED_OPTION_REPLACEMENTS
query_lmm = eaggl_cli.query_lmm
_classify_factor_workflow = eaggl_cli._classify_factor_workflow
_FACTOR_WORKFLOW_STRATEGY_META = eaggl_cli._FACTOR_WORKFLOW_STRATEGY_META

options = eaggl_cli.options
args = eaggl_cli.args
mode = eaggl_cli.mode
config_mode = eaggl_cli.config_mode
cli_specified_dests = eaggl_cli.cli_specified_dests
config_specified_dests = eaggl_cli.config_specified_dests
eaggl_bundle_info = eaggl_cli.eaggl_bundle_info
run_factor = eaggl_cli.run_factor
run_phewas = eaggl_cli.run_phewas
run_naive_factor = eaggl_cli.run_naive_factor
use_phewas_for_factoring = eaggl_cli.use_phewas_for_factoring
factor_gene_set_x_pheno = eaggl_cli.factor_gene_set_x_pheno
expand_gene_sets = eaggl_cli.expand_gene_sets
factor_workflow = eaggl_cli.factor_workflow
NONE = eaggl_cli.NONE
INFO = eaggl_cli.INFO
DEBUG = eaggl_cli.DEBUG
TRACE = eaggl_cli.TRACE
debug_level = eaggl_cli.debug_level
log_fh = eaggl_cli.log_fh
warnings_fh = eaggl_cli.warnings_fh
log = eaggl_cli.log
warn = eaggl_cli.warn


def _sync_cli_exports():
    global options, args, mode, config_mode, cli_specified_dests, config_specified_dests
    global eaggl_bundle_info, run_factor, run_phewas, run_naive_factor
    global use_phewas_for_factoring, factor_gene_set_x_pheno, expand_gene_sets, factor_workflow
    global NONE, INFO, DEBUG, TRACE, debug_level, log_fh, warnings_fh, log, warn

    options = eaggl_cli.options
    args = eaggl_cli.args
    mode = eaggl_cli.mode
    config_mode = eaggl_cli.config_mode
    cli_specified_dests = eaggl_cli.cli_specified_dests
    config_specified_dests = eaggl_cli.config_specified_dests
    eaggl_bundle_info = eaggl_cli.eaggl_bundle_info
    run_factor = eaggl_cli.run_factor
    run_phewas = eaggl_cli.run_phewas
    run_naive_factor = eaggl_cli.run_naive_factor
    use_phewas_for_factoring = eaggl_cli.use_phewas_for_factoring
    factor_gene_set_x_pheno = eaggl_cli.factor_gene_set_x_pheno
    expand_gene_sets = eaggl_cli.expand_gene_sets
    factor_workflow = eaggl_cli.factor_workflow
    NONE = eaggl_cli.NONE
    INFO = eaggl_cli.INFO
    DEBUG = eaggl_cli.DEBUG
    TRACE = eaggl_cli.TRACE
    debug_level = eaggl_cli.debug_level
    log_fh = eaggl_cli.log_fh
    warnings_fh = eaggl_cli.warnings_fh
    log = eaggl_cli.log
    warn = eaggl_cli.warn


def _bootstrap_cli(argv=None):
    should_continue = eaggl_cli._bootstrap_cli(argv)
    _sync_cli_exports()
    eaggl_state.configure_runtime_context(cli_module=eaggl_cli)
    return should_continue


def build_main_domain():
    _sync_cli_exports()
    eaggl_state.configure_runtime_context(cli_module=eaggl_cli)
    return eaggl_domain.build_main_domain(sys.modules[__name__])


def open_gz(file, flag=None):
    return eaggl_state.pegs_open_text_with_retry(
        file,
        flag=flag,
        log_fn=lambda message: log(message, INFO),
        bail_fn=eaggl_state.bail,
    )


_bind_hyperparameter_properties = eaggl_state._bind_hyperparameter_properties
_append_with_any_user = eaggl_state._append_with_any_user
EagglState = eaggl_state.EagglState
GeneSetData = eaggl_state.GeneSetData

FactorOnlyStageResult = eaggl_factor.FactorOnlyStageResult
PhewasStageResult = eaggl_factor.PhewasStageResult
FactorStageResult = eaggl_factor.FactorStageResult
FactorWorkflow = eaggl_factor.FactorWorkflow
FactorInputs = eaggl_factor.FactorInputs
FactorExecutionConfig = eaggl_factor.FactorExecutionConfig
MainPipelineResult = eaggl_factor.MainPipelineResult
FactorOutputPlan = eaggl_outputs.FactorOutputPlan

_read_y_pipeline = eaggl_y_inputs.read_y_pipeline
_run_read_y_stage = eaggl_y_inputs.run_read_y_stage
_read_x_pipeline = eaggl_io.read_x_pipeline
_run_read_x_stage = eaggl_io.run_read_x_stage
_log_runtime_environment_if_requested = eaggl_io.log_runtime_environment_if_requested
_read_gene_map = eaggl_io.read_gene_map
_init_gene_locs = eaggl_io.init_gene_locs
_initialize_main_mappings = eaggl_io.initialize_main_mappings
_read_gene_set_statistics = eaggl_io.read_gene_set_statistics
_read_gene_set_phewas_statistics = eaggl_io.read_gene_set_phewas_statistics
_derive_factor_anchor_masks = eaggl_io.derive_factor_anchor_masks
_read_gene_phewas_bfs = eaggl_io.read_gene_phewas_bfs
_has_loaded_gene_phewas = eaggl_io.has_loaded_gene_phewas

_extract_factor_workflow = eaggl_factor.extract_factor_workflow
_extract_factor_inputs = eaggl_factor.extract_factor_inputs
_resolve_factor_gene_or_pheno_filter_value = eaggl_factor.resolve_factor_gene_or_pheno_filter_value
_build_factor_execution_config = eaggl_factor.build_factor_execution_config
_run_factor_model = eaggl_factor.run_factor_model
_build_factor_output_plan = eaggl_outputs.build_factor_output_plan
_write_factor_outputs_for_plan = eaggl_outputs.write_factor_outputs_for_plan

_normalize_dense_gene_rows = eaggl_state._normalize_dense_gene_rows
_build_sparse_x_from_dense_input = eaggl_state._build_sparse_x_from_dense_input
_estimate_dense_chunk_size = eaggl_state._estimate_dense_chunk_size
_record_x_addition = eaggl_state._record_x_addition
_process_dense_x_file = eaggl_state._process_dense_x_file
_process_sparse_x_file = eaggl_state._process_sparse_x_file
_process_x_input_file = eaggl_state._process_x_input_file
_standardize_qc_metrics_after_x_read = eaggl_state._standardize_qc_metrics_after_x_read
_maybe_correct_gene_set_betas_after_x_read = eaggl_state._maybe_correct_gene_set_betas_after_x_read
_maybe_limit_initial_gene_sets_by_p = eaggl_state._maybe_limit_initial_gene_sets_by_p
_maybe_prune_gene_sets_after_x_read = eaggl_state._maybe_prune_gene_sets_after_x_read
_initialize_hyper_defaults_after_x_read = eaggl_state._initialize_hyper_defaults_after_x_read
_learn_hyper_for_gene_set_batch = eaggl_state._learn_hyper_for_gene_set_batch
_apply_learned_batch_hyper_values = eaggl_state._apply_learned_batch_hyper_values
_finalize_batch_hyper_vectors = eaggl_state._finalize_batch_hyper_vectors
_maybe_learn_batch_hyper_after_x_read = eaggl_state._maybe_learn_batch_hyper_after_x_read
_maybe_adjust_overaggressive_p_filter_after_x_read = eaggl_state._maybe_adjust_overaggressive_p_filter_after_x_read
_apply_post_read_gene_set_size_and_qc_filters = eaggl_state._apply_post_read_gene_set_size_and_qc_filters
_maybe_filter_zero_uncorrected_betas_after_x_read = eaggl_state._maybe_filter_zero_uncorrected_betas_after_x_read
_maybe_reduce_gene_sets_to_max_after_x_read = eaggl_state._maybe_reduce_gene_sets_to_max_after_x_read

_bind_hyperparameter_properties(EagglState)


def _build_main_domain():
    eaggl_state.configure_runtime_context(cli_module=eaggl_cli)
    return eaggl_domain.build_main_domain(sys.modules[__name__])


def _run_main_factor_stage(g, options, mode_state, factor_input_state):
    return eaggl_factor.run_main_factor_stage(_build_main_domain(), g, options, mode_state, factor_input_state)


def _write_main_factor_outputs(g, options):
    return eaggl_outputs.write_main_factor_outputs(g, options)


def _reread_gene_phewas_bfs(state):
    return eaggl_io.reread_gene_phewas_bfs(_build_main_domain(), state)


def _run_main_phewas_stage(g, options):
    return eaggl_factor.run_main_phewas_stage(_build_main_domain(), g, options)


def _run_main_factor_phewas_stage(g, options):
    return eaggl_factor.run_main_factor_phewas_stage(_build_main_domain(), g, options)
