"""Mode-aware scientific defaults for the recommended factor workflow."""
LEGACY_FACTOR_DEFAULTS = {
    "learn_phi": False,
    "phi_selection_objective": "composite",
    "phi": 0.05,
    "factor_runs": 1,
    "consensus_nmf": False,
    "consensus_matching": "reference",
    "factor_retention": "legacy",
    "fixed_basis_projection_method": "legacy",
    "factor_output_scope": "primary",
    "cluster_row_min_max_loading": 0.01,
    "factor_top_loading_type": "euclidean",
}
RECOMMENDED_FACTOR_DEFAULTS = {
    **LEGACY_FACTOR_DEFAULTS,
    "learn_phi": True,
    "phi_selection_objective": "requirements",
    "phi": 0.01,
    "factor_runs": 5,
    "consensus_nmf": True,
    "consensus_matching": "pooled",
    "factor_retention": "backend_scale",
    "fixed_basis_projection_method": "verified_quadratic",
    "factor_output_scope": "all",
    "cluster_row_min_max_loading": 0.0,
    "factor_top_loading_type": "raw",
}


def apply_factor_defaults(options, mode, explicit):
    """Resolve only omitted values; explicit CLI/config values always win.

    Supplied-factor, label and naive modes retain their previous defaults.
    Rectangular discovery remains available by explicitly selecting it.
    """
    supplied = any(getattr(options, name, None) is not None for name in (
        "factor_gene_clusters_in", "factor_gene_set_clusters_in", "factor_phewas_gene_clusters_in"))
    discovery = mode == "factor" and not supplied
    if "discovery_model" not in explicit:
        options.discovery_model = "gene_by_gene" if discovery else "gene_by_annotation"
    recommended = discovery and options.discovery_model == "gene_by_gene"
    defaults = RECOMMENDED_FACTOR_DEFAULTS if recommended else LEGACY_FACTOR_DEFAULTS
    for name, value in defaults.items():
        if name not in explicit:
            setattr(options, name, value)
    # A supplied phi is a fixed resolution unless learning is also explicit.
    # This preserves existing fixed-phi CLI and configuration recipes.
    if "phi" in explicit and "learn_phi" not in explicit:
        options.learn_phi = False
    # An explicit single-run request is a complete opt-out of restart aggregation.
    # Explicitly requesting consensus with one run still fails normal validation.
    if options.factor_runs == 1 and "consensus_nmf" not in explicit:
        options.consensus_nmf = False
    if discovery:
        if options.symnmf_update_method == "auto":
            options.symnmf_update_method = "projected_gradient" if recommended else "legacy"
        projected = options.symnmf_update_method == "projected_gradient"
        if options.max_num_iterations is None:
            options.max_num_iterations = 20000 if projected else 100
        if options.rel_tol is None:
            options.rel_tol = 1e-6 if projected else 1e-4
