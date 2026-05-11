from __future__ import annotations

from dataclasses import dataclass

from . import annotation_diagnostics


@dataclass
class FactorOutputPlan:
    factors_out: str | None = None
    factor_metrics_out: str | None = None
    consensus_stats_out: str | None = None
    gene_set_clusters_out: str | None = None
    gene_clusters_out: str | None = None
    gene_clusters_full_out: str | None = None
    gene_clusters_full_via_gene_sets_out: str | None = None
    trait_factor_links_out: str | None = None
    trait_factor_links_output_detail: str = "main"
    gene_pheno_stats_out: str | None = None
    annotation_bridge_metrics_out: str | None = None
    annotation_bridge_suggested_exclude_out: str | None = None
    gene_factor_annotation_contribs_out: str | None = None
    gene_factor_annotation_contribs_top_n: int = 10
    max_no_write_gene_pheno: object = None
    cluster_row_min_max_loading: float = 0.01
    factor_output_scope: str = "primary"


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
        factor_metrics_out=options.factor_metrics_out,
        consensus_stats_out=options.consensus_stats_out,
        gene_set_clusters_out=options.gene_set_clusters_out,
        gene_clusters_out=options.gene_clusters_out,
        gene_clusters_full_out=getattr(options, "gene_clusters_full_out", None),
        gene_clusters_full_via_gene_sets_out=getattr(options, "gene_clusters_full_via_gene_sets_out", None),
        trait_factor_links_out=getattr(options, "trait_factor_links_out", None),
        trait_factor_links_output_detail=getattr(options, "trait_factor_links_output_detail", "main"),
        gene_pheno_stats_out=options.gene_pheno_stats_out,
        annotation_bridge_metrics_out=getattr(options, "annotation_bridge_metrics_out", None),
        annotation_bridge_suggested_exclude_out=getattr(options, "annotation_bridge_suggested_exclude_out", None),
        gene_factor_annotation_contribs_out=getattr(options, "gene_factor_annotation_contribs_out", None),
        gene_factor_annotation_contribs_top_n=getattr(options, "gene_factor_annotation_contribs_top_n", 10),
        max_no_write_gene_pheno=options.max_no_write_gene_pheno,
        cluster_row_min_max_loading=getattr(options, "cluster_row_min_max_loading", 0.01),
        factor_output_scope=getattr(options, "factor_output_scope", "primary"),
    )


def write_factor_outputs_for_plan(runtime, output_plan):
    if output_plan.factors_out is not None:
        runtime.write_matrix_factors(output_plan.factors_out, factor_output_scope=output_plan.factor_output_scope)
    if output_plan.factor_metrics_out is not None:
        runtime.write_factor_metrics(output_plan.factor_metrics_out)
    if output_plan.consensus_stats_out is not None:
        runtime.write_consensus_factor_diagnostics(output_plan.consensus_stats_out)
    if (
        output_plan.gene_set_clusters_out is not None
        or output_plan.gene_clusters_out is not None
    ):
        runtime.write_clusters(
            output_plan.gene_set_clusters_out,
            output_plan.gene_clusters_out,
            None,
            cluster_row_min_max_loading=output_plan.cluster_row_min_max_loading,
            factor_output_scope=output_plan.factor_output_scope,
        )
    if output_plan.gene_clusters_full_out is not None:
        runtime.write_full_gene_clusters(
            output_plan.gene_clusters_full_out,
            cluster_row_min_max_loading=output_plan.cluster_row_min_max_loading,
            factor_output_scope=output_plan.factor_output_scope,
            projection_method="auto",
        )
    if output_plan.gene_clusters_full_via_gene_sets_out is not None:
        runtime.write_full_gene_clusters(
            output_plan.gene_clusters_full_via_gene_sets_out,
            cluster_row_min_max_loading=output_plan.cluster_row_min_max_loading,
            factor_output_scope=output_plan.factor_output_scope,
            projection_method="gene_set_loadings",
        )
    trait_factor_link_paths = []
    if output_plan.trait_factor_links_out is not None:
        trait_factor_link_paths.append(output_plan.trait_factor_links_out)
    for output_path in dict.fromkeys(trait_factor_link_paths):
        runtime.write_trait_factor_links(output_path, output_detail=output_plan.trait_factor_links_output_detail)
    if output_plan.gene_pheno_stats_out is not None:
        runtime.write_gene_pheno_statistics(
            output_plan.gene_pheno_stats_out,
            min_value_to_print=output_plan.max_no_write_gene_pheno,
        )
    if output_plan.annotation_bridge_metrics_out is not None:
        annotation_diagnostics.write_annotation_bridge_metrics(
            runtime,
            output_plan.annotation_bridge_metrics_out,
        )
    if output_plan.annotation_bridge_suggested_exclude_out is not None:
        annotation_diagnostics.write_annotation_bridge_suggested_exclude(
            runtime,
            output_plan.annotation_bridge_suggested_exclude_out,
        )
    if output_plan.gene_factor_annotation_contribs_out is not None:
        annotation_diagnostics.write_gene_factor_annotation_contribs(
            runtime,
            output_plan.gene_factor_annotation_contribs_out,
            top_n=output_plan.gene_factor_annotation_contribs_top_n,
        )


def write_main_factor_outputs(runtime, options):
    output_plan = build_factor_output_plan(options)
    write_factor_outputs_for_plan(runtime, output_plan)
