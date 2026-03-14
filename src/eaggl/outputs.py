from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FactorOutputPlan:
    factors_out: str | None = None
    factors_anchor_out: str | None = None
    consensus_stats_out: str | None = None
    gene_set_clusters_out: str | None = None
    gene_clusters_out: str | None = None
    pheno_clusters_out: str | None = None
    gene_set_anchor_clusters_out: str | None = None
    gene_anchor_clusters_out: str | None = None
    pheno_anchor_clusters_out: str | None = None
    gene_pheno_stats_out: str | None = None
    max_no_write_gene_pheno: object = None


def write_main_primary_outputs(runtime, options):
    if options.gene_set_stats_out:
        runtime.write_gene_set_statistics(
            options.gene_set_stats_out,
            max_no_write_gene_set_beta=options.max_no_write_gene_set_beta,
            max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected,
        )
    if options.phewas_gene_set_stats_out:
        runtime.write_phewas_gene_set_statistics(
            options.phewas_gene_set_stats_out,
            max_no_write_gene_set_beta=options.max_no_write_gene_set_beta,
            max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected,
        )
    if options.gene_stats_out:
        runtime.write_gene_statistics(options.gene_stats_out)
    if options.gene_gene_set_stats_out:
        runtime.write_gene_gene_set_statistics(
            options.gene_gene_set_stats_out,
            max_no_write_gene_gene_set_beta=options.max_no_write_gene_gene_set_beta,
            write_filter_beta_uncorrected=options.use_beta_uncorrected_for_gene_gene_set_write_filter,
        )
    if options.gene_set_overlap_stats_out:
        runtime.write_gene_set_overlap_statistics(options.gene_set_overlap_stats_out)
    if options.gene_covs_out:
        runtime.write_gene_covariates(options.gene_covs_out)
    if options.gene_effectors_out:
        runtime.write_gene_effectors(options.gene_effectors_out)


def build_factor_output_plan(options):
    return FactorOutputPlan(
        factors_out=options.factors_out,
        factors_anchor_out=options.factors_anchor_out,
        consensus_stats_out=options.consensus_stats_out,
        gene_set_clusters_out=options.gene_set_clusters_out,
        gene_clusters_out=options.gene_clusters_out,
        pheno_clusters_out=options.pheno_clusters_out,
        gene_set_anchor_clusters_out=options.gene_set_anchor_clusters_out,
        gene_anchor_clusters_out=options.gene_anchor_clusters_out,
        pheno_anchor_clusters_out=options.pheno_anchor_clusters_out,
        gene_pheno_stats_out=options.gene_pheno_stats_out,
        max_no_write_gene_pheno=options.max_no_write_gene_pheno,
    )


def write_factor_outputs_for_plan(runtime, output_plan):
    if output_plan.factors_out is not None:
        runtime.write_matrix_factors(output_plan.factors_out)
    if output_plan.factors_anchor_out is not None:
        runtime.write_matrix_factors(output_plan.factors_anchor_out, write_anchor_specific=True)
    if output_plan.consensus_stats_out is not None:
        runtime.write_consensus_factor_diagnostics(output_plan.consensus_stats_out)
    if (
        output_plan.gene_set_clusters_out is not None
        or output_plan.gene_clusters_out is not None
        or output_plan.pheno_clusters_out is not None
    ):
        runtime.write_clusters(
            output_plan.gene_set_clusters_out,
            output_plan.gene_clusters_out,
            output_plan.pheno_clusters_out,
        )
    if (
        output_plan.gene_set_anchor_clusters_out is not None
        or output_plan.gene_anchor_clusters_out is not None
        or output_plan.pheno_anchor_clusters_out is not None
    ):
        runtime.write_clusters(
            output_plan.gene_set_anchor_clusters_out,
            output_plan.gene_anchor_clusters_out,
            output_plan.pheno_anchor_clusters_out,
            write_anchor_specific=True,
        )
    if output_plan.gene_pheno_stats_out is not None:
        runtime.write_gene_pheno_statistics(
            output_plan.gene_pheno_stats_out,
            min_value_to_print=output_plan.max_no_write_gene_pheno,
        )


def write_main_factor_outputs(runtime, options):
    output_plan = build_factor_output_plan(options)
    write_factor_outputs_for_plan(runtime, output_plan)
