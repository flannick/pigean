from __future__ import annotations


def run_main_pipeline(domain, options):
    mode_state = domain._build_main_mode_state()
    domain._enforce_factor_only_input_boundary(options, mode_state)
    domain._log_runtime_environment_if_requested(options)

    state = domain.EagglState(background_prior=options.background_prior, batch_size=options.batch_size)
    domain._initialize_main_mappings(state, options)
    factor_input_state = domain._run_main_factor_only_pipeline(state, options, mode_state)
    factor_only_stage_result = domain.FactorOnlyStageResult(
        ran=True,
        num_gene_sets=len(state.gene_sets) if state.gene_sets is not None else 0,
        factor_input_state=factor_input_state,
    )

    domain._write_main_primary_outputs(state, options)

    phewas_stage_result = domain.PhewasStageResult(ran=False, output_path=options.phewas_stats_out)
    if mode_state["run_phewas"]:
        phewas_stage_result = domain._run_main_phewas_stage(state, options)

    factor_model_stage_result = domain.FactorStageResult(ran=False, workflow_id=None)
    if mode_state["run_factor"]:
        factor_model_stage_result = domain._run_main_factor_stage(state, options, mode_state, factor_input_state)

    domain._write_main_factor_outputs(state, options)

    factor_phewas_stage_result = domain.PhewasStageResult(ran=False, output_path=options.factor_phewas_stats_out)
    if domain._should_run_main_factor_phewas_stage(mode_state):
        factor_phewas_stage_result = domain._run_main_factor_phewas_stage(state, options)

    if options.params_out:
        state.write_params(options.params_out)

    return domain.MainPipelineResult(
        state=state,
        mode_state=mode_state,
        factor_only=factor_only_stage_result,
        phewas=phewas_stage_result,
        factor=factor_model_stage_result,
        factor_phewas=factor_phewas_stage_result,
    )
