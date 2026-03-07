from __future__ import annotations

def write_eaggl_bundle_if_requested(domain, state, options, mode):
    if options.eaggl_bundle_out is None:
        return

    out_path = options.eaggl_bundle_out
    domain.log("Writing EAGGL handoff bundle to %s" % out_path, domain.INFO)
    generated_file_specs = [
        (
            "X_in",
            "X.tsv.gz",
            lambda path: state.write_X(path),
            "X matrix",
            "run with --X-in/--X-list and ensure gene sets were loaded",
        ),
        (
            "gene_stats_in",
            "gene_stats.tsv.gz",
            lambda path: state.write_gene_statistics(path),
            "gene statistics",
            "run a mode that computes/loads gene scores",
        ),
        (
            "gene_set_stats_in",
            "gene_set_stats.tsv.gz",
            lambda path: state.write_gene_set_statistics(
                path,
                max_no_write_gene_set_beta=options.max_no_write_gene_set_beta,
                max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected,
            ),
            "gene-set statistics",
            "run a mode that computes/loads gene-set statistics",
        ),
    ]
    optional_existing_files = [
        ("gene_phewas_bfs_in", options.phewas_stats_out, "gene_phewas_stats.tsv.gz"),
        ("gene_set_phewas_stats_in", options.phewas_gene_set_stats_out, "gene_set_phewas_stats.tsv.gz"),
    ]
    domain.pegs_write_bundle_from_specs(
        out_path,
        schema=domain.PEGS_EAGGL_BUNDLE_SCHEMA,
        source_tool="pigean.py",
        source_mode=mode,
        source_argv=domain.sys.argv,
        generated_file_specs=generated_file_specs,
        optional_existing_files=optional_existing_files,
        option_name="--eaggl-bundle-out",
        temp_prefix="pigean_eaggl_bundle_",
        manifest_name="manifest.json",
        bail_fn=domain.bail,
    )

    domain.log("Finished writing EAGGL handoff bundle %s" % out_path, domain.INFO)


def write_main_outputs_and_optional_phewas(domain, state, options, mode_state, mode):
    if options.gene_set_stats_out:
        state.write_gene_set_statistics(
            options.gene_set_stats_out,
            max_no_write_gene_set_beta=options.max_no_write_gene_set_beta,
            max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected,
        )
    if options.phewas_gene_set_stats_out:
        state.write_phewas_gene_set_statistics(
            options.phewas_gene_set_stats_out,
            max_no_write_gene_set_beta=options.max_no_write_gene_set_beta,
            max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected,
        )
    if options.gene_set_overlap_stats_out:
        state.write_gene_set_overlap_statistics(options.gene_set_overlap_stats_out)

    if options.gene_stats_out:
        state.write_gene_statistics(options.gene_stats_out)
    if options.gene_gene_set_stats_out:
        state.write_gene_gene_set_statistics(
            options.gene_gene_set_stats_out,
            max_no_write_gene_gene_set_beta=options.max_no_write_gene_gene_set_beta,
            write_filter_beta_uncorrected=options.use_beta_uncorrected_for_gene_gene_set_write_filter,
        )
    if options.gene_covs_out:
        state.write_gene_covariates(options.gene_covs_out)
    if options.gene_effectors_out:
        state.write_gene_effectors(options.gene_effectors_out)

    if mode_state["run_phewas"]:
        domain._run_advanced_set_b_output_phewas_if_requested(state=state, options=options)

    if options.params_out:
        state.write_params(options.params_out)
    write_eaggl_bundle_if_requested(domain=domain, state=state, options=options, mode=mode)
