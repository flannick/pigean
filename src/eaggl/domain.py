from __future__ import annotations

from pegs_shared.phewas import (
    build_phewas_stage_config,
    derive_factor_anchor_masks as pegs_derive_factor_anchor_masks,
    resolve_gene_phewas_input_decision_for_stage,
)

from . import state as eaggl_state
from . import factor as eaggl_factor
from . import io as eaggl_io
from . import outputs as eaggl_outputs
from . import workflows as eaggl_workflows
from . import y_inputs as eaggl_y_inputs


class EagglMainDomain(object):
    def __init__(self, support_module):
        self._support = support_module
        self.EagglState = eaggl_state.EagglState
        self.FactorOnlyStageResult = eaggl_factor.FactorOnlyStageResult
        self.PhewasStageResult = eaggl_factor.PhewasStageResult
        self.FactorStageResult = eaggl_factor.FactorStageResult
        self.FactorWorkflow = eaggl_factor.FactorWorkflow
        self.FactorInputs = eaggl_factor.FactorInputs
        self.FactorExecutionConfig = eaggl_factor.FactorExecutionConfig
        self.MainPipelineResult = eaggl_factor.MainPipelineResult
        self.FactorOutputPlan = eaggl_outputs.FactorOutputPlan
        self.pegs_build_phewas_stage_config = build_phewas_stage_config
        self.pegs_resolve_gene_phewas_input_decision_for_stage = resolve_gene_phewas_input_decision_for_stage
        self.pegs_derive_factor_anchor_masks = pegs_derive_factor_anchor_masks

    def __getattr__(self, name):
        return getattr(self._support, name)

    def _build_main_mode_state(self):
        return eaggl_factor.build_main_mode_state(self)

    def _enforce_factor_only_input_boundary(self, options, mode_state):
        return eaggl_workflows.enforce_factor_only_input_boundary(options, mode_state, self.bail)

    def _run_main_factor_only_pipeline(self, runtime, options, mode_state):
        return eaggl_factor.run_main_factor_only_pipeline(self, runtime, options, mode_state)

    def _run_read_y_stage(self, runtime, **read_kwargs):
        return eaggl_y_inputs.run_read_y_stage(self, runtime, **read_kwargs)

    def _run_read_x_stage(self, runtime, X_in, **read_x_kwargs):
        return eaggl_io.run_read_x_stage(self, runtime, X_in, **read_x_kwargs)

    def _log_runtime_environment_if_requested(self, options):
        return eaggl_io.log_runtime_environment_if_requested(self, options)

    def _initialize_main_mappings(self, runtime, options):
        return eaggl_io.initialize_main_mappings(self, runtime, options)

    def _read_gene_set_statistics(self, runtime, stats_in, **kwargs):
        return eaggl_io.read_gene_set_statistics(self, runtime, stats_in, **kwargs)

    def _read_gene_set_phewas_statistics(self, runtime, stats_in, **kwargs):
        return eaggl_io.read_gene_set_phewas_statistics(self, runtime, stats_in, **kwargs)

    def _derive_factor_anchor_masks(self, runtime, options):
        return eaggl_io.derive_factor_anchor_masks(self, runtime, options)

    def _read_gene_phewas_bfs(self, runtime, **kwargs):
        return eaggl_io.read_gene_phewas_bfs(self, runtime, **kwargs)

    def _has_loaded_gene_phewas(self, runtime):
        return eaggl_io.has_loaded_gene_phewas(runtime)

    def _reread_gene_phewas_bfs(self, runtime):
        return eaggl_io.reread_gene_phewas_bfs(self, runtime)

    def _load_factor_phewas_inputs(self, runtime, options):
        return eaggl_factor.load_factor_phewas_inputs(self, runtime, options)

    def _write_main_primary_outputs(self, runtime, options):
        return eaggl_outputs.write_main_primary_outputs(runtime, options)

    def _run_main_phewas_stage(self, runtime, options):
        return eaggl_factor.run_main_phewas_stage(self, runtime, options)

    def _run_main_factor_stage(self, runtime, options, mode_state, factor_input_state):
        return eaggl_factor.run_main_factor_stage(self, runtime, options, mode_state, factor_input_state)

    def _write_main_factor_outputs(self, runtime, options):
        return eaggl_outputs.write_main_factor_outputs(runtime, options)

    def _run_main_factor_phewas_stage(self, runtime, options):
        return eaggl_factor.run_main_factor_phewas_stage(self, runtime, options)

    def _should_run_main_factor_phewas_stage(self, mode_state):
        return eaggl_factor.should_run_main_factor_phewas_stage(mode_state)


def build_main_domain(support_module):
    return EagglMainDomain(support_module)
