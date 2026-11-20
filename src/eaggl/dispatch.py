from __future__ import annotations

import math


def run_main_pipeline(domain, options):
    mode_state = domain._build_main_mode_state()
    domain._enforce_factor_only_input_boundary(options, mode_state)
    clustering_provenance = domain._build_clustering_provenance(options, mode_state)
    for summary_line in domain._format_clustering_provenance_summary(clustering_provenance):
        domain.log(summary_line, domain.INFO)
    domain._log_runtime_environment_if_requested(options)

    state = domain.EagglState(background_prior=options.background_prior, batch_size=options.batch_size)
    from .labeling import configure_sources
    configure_sources(state, options, domain.bail)
    state.gene_set_clusters_format = getattr(options, "gene_set_clusters_format", "evidence")
    state.internal_projection_method = getattr(options, 'fixed_basis_projection_method', 'legacy')
    state.internal_projection_max_cycles = getattr(options, 'projection_max_cycles', 2000)
    state.internal_projection_tol = getattr(options, 'projection_stationarity_tol', 1e-8)
    if (state.internal_projection_max_cycles <= 0 or not math.isfinite(state.internal_projection_tol)
            or state.internal_projection_tol <= 0):
        domain.bail('Projection budget and tolerance must be positive')
    state._record_params(
        {
            "clustering_workflow_id": clustering_provenance.get("workflow_id"),
            "clustering_routing_family": clustering_provenance.get("routing_family"),
            "clustering_factorization_source": clustering_provenance.get("factorization_source"),
            "clustering_projection_only": clustering_provenance.get("projection_only"),
            "trait_linkage_enabled": clustering_provenance.get("trait_linkage", {}).get("enabled"),
            "trait_linkage_basis": clustering_provenance.get("trait_linkage", {}).get("basis"),
            "trait_linkage_source": clustering_provenance.get("trait_linkage", {}).get("source"),
            "trait_linkage_threshold": clustering_provenance.get("trait_linkage", {}).get("threshold"),
            "trait_linkage_computation_mode": clustering_provenance.get("trait_linkage", {}).get("computation_mode"),
            "clustering_anchor_mode": clustering_provenance.get("anchor_mode"),
        },
        overwrite=True,
    )
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
    requested = [state.primary_label_sources or (), *state.extra_label_sources]
    phenotype_labels = (any("phenotypes" in group for group in requested)
                        or getattr(options, "label_include_phenos", False)
                        or getattr(options, "label_individually", False))
    state.defer_source_labels = bool(mode_state["run_factor"] and phenotype_labels
        and not mode_state.get("use_phewas_for_factoring") and not getattr(options, "no_trait_linkage", False)
        and (getattr(options, "gene_phewas_bfs_in", None) or getattr(options, "gene_set_phewas_stats_in", None)))
    if mode_state["run_factor"]:
        factor_model_stage_result = domain._run_main_factor_stage(state, options, mode_state, factor_input_state)

    pheno_projection_stage_result = domain.PhewasStageResult(ran=False, output_path=getattr(options, "pheno_clusters_out", None))
    if getattr(options, 'projection_context_in', None) is not None:
        from .projection_context import restore_context
        if not mode_state.get('factor_projection_only'):
            domain.bail('--projection-context-in requires supplied factors, not a fresh discovery run')
        restore_context(state, options.projection_context_in, options)
    if domain._should_run_main_pheno_projection_stage(mode_state, options) or state.defer_source_labels:
        pheno_projection_stage_result = domain._run_main_pheno_projection_stage(state, options)

    if state.defer_source_labels and state.num_factors():
        from .label import _top_items
        from .labeling import label_selected
        # For additional phenotypes the linkage matrix becomes available only after discovery.
        if state.trait_linkage_nnls is not None:
            ids = [i for i, name in enumerate(state.phenos) if name != state.default_pheno]
            state.factor_top_phenos = _top_items([state.phenos[i] for i in ids], state.trait_linkage_nnls[ids],
                loading_type=options.factor_top_loading_type) if ids else None
        label_selected(state, lmm_auth_key=options.lmm_auth_key, lmm_model=options.lmm_model,
            lmm_provider=options.lmm_provider, label_gene_sets_only=options.label_gene_sets_only,
            label_include_phenos=options.label_include_phenos, label_individually=options.label_individually,
            bail_fn=domain.bail, log_fn=domain.log, warn_fn=domain.warn)

    domain._write_main_factor_outputs(state, options)
    if getattr(options, 'projection_context_out', None) is not None:
        from .projection_context import save_context
        save_context(state, options.projection_context_out, options)

    factor_phewas_stage_result = domain.PhewasStageResult(ran=False, output_path=options.factor_phewas_stats_out)
    if domain._should_run_main_factor_phewas_stage(mode_state):
        factor_phewas_stage_result = domain._run_main_factor_phewas_stage(state, options)

    outputs_written = {
        "primary_params_out": options.params_out,
        "clustering_params_out": options.clustering_params_out,
        "gene_set_stats_out": options.gene_set_stats_out,
        "phewas_gene_set_stats_out": options.phewas_gene_set_stats_out,
        "gene_stats_out": options.gene_stats_out,
        "gene_gene_set_stats_out": options.gene_gene_set_stats_out,
        "gene_set_overlap_stats_out": options.gene_set_overlap_stats_out,
        "gene_covs_out": options.gene_covs_out,
        "gene_effectors_out": options.gene_effectors_out,
        "phewas_stats_out": phewas_stage_result.output_path if phewas_stage_result.ran else None,
        "factors_out": options.factors_out,
        "factor_metrics_out": options.factor_metrics_out,
        "consensus_stats_out": options.consensus_stats_out,
        "gene_set_clusters_out": options.gene_set_clusters_out,
        "gene_set_clusters_marginal_out": getattr(options, "gene_set_clusters_marginal_out", None),
        "gene_clusters_out": options.gene_clusters_out,
        "gene_clusters_full_out": getattr(options, "gene_clusters_full_out", None),
        "gene_clusters_full_via_gene_sets_out": getattr(options, "gene_clusters_full_via_gene_sets_out", None),
        "trait_factor_links_out": options.trait_factor_links_out,
        "pheno_clusters_out": pheno_projection_stage_result.output_path if pheno_projection_stage_result.ran else None,
        "gene_pheno_stats_out": options.gene_pheno_stats_out,
        "factor_phewas_stats_out": factor_phewas_stage_result.output_path if factor_phewas_stage_result.ran else None,
    }
    final_clustering_provenance = domain._build_clustering_provenance(options, mode_state, outputs_written=outputs_written)
    state._record_params(
        {
            "clustering_executed": final_clustering_provenance.get("clustering_executed"),
            "precomputed_factors_loaded": final_clustering_provenance.get("precomputed_factors_loaded"),
        },
        overwrite=True,
    )
    clustering_params_paths = None
    if options.clustering_params_out:
        clustering_params_paths = state.write_clustering_params(options.clustering_params_out, final_clustering_provenance)
        if isinstance(clustering_params_paths, dict):
            state._record_params(
                {
                    "clustering_params_json_out": clustering_params_paths.get("json"),
                    "clustering_params_tsv_out": clustering_params_paths.get("tsv"),
                },
                overwrite=True,
            )
    if options.params_out:
        state.write_params(options.params_out)

    return domain.MainPipelineResult(
        state=state,
        mode_state=mode_state,
        factor_only=factor_only_stage_result,
        phewas=phewas_stage_result,
        factor=factor_model_stage_result,
        pheno_projection=pheno_projection_stage_result,
        factor_phewas=factor_phewas_stage_result,
    )
