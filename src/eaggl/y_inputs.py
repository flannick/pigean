from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sparse

from pegs_shared.io_common import clean_chrom_name, read_loc_file_with_gene_map

from . import covariates as eaggl_covariates


def run_read_y_stage(domain, runtime, **read_kwargs):
    return read_y_pipeline(domain, runtime, **read_kwargs)


def read_y_pipeline(
    domain,
    runtime,
    gwas_in=None,
    huge_statistics_in=None,
    huge_statistics_out=None,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    ctrl_counts_in=None,
    gene_bfs_in=None,
    gene_loc_file=None,
    gene_covs_in=None,
    hold_out_chrom=None,
    **kwargs
):
    unsupported_flags = []
    if gwas_in is not None:
        unsupported_flags.append("--gwas-in")
    if huge_statistics_in is not None:
        unsupported_flags.append("--huge-statistics-in")
    if huge_statistics_out is not None:
        unsupported_flags.append("--huge-statistics-out")
    if exomes_in is not None:
        unsupported_flags.append("--exomes-in")
    if case_counts_in is not None:
        unsupported_flags.append("--case-counts-in")
    if ctrl_counts_in is not None:
        unsupported_flags.append("--ctrl-counts-in")

    if len(unsupported_flags) > 0:
        domain.bail(
            "These inputs belong to pigean.py and are not supported in eaggl.py: %s. "
            "Run pigean.py first and pass outputs via --eaggl-bundle-in or --gene-stats-in/--gene-set-stats-in."
            % ", ".join(sorted(unsupported_flags))
        )

    if positive_controls_in is not None or positive_controls_list is not None:
        domain.warn("Ignoring positive-control inputs in eaggl.py read_Y; using --gene-stats-in values")

    if gene_bfs_in is None:
        domain.bail("Require --gene-stats-in for this operation")

    (Y1, extra_genes, extra_Y, gene_combined_map, gene_prior_map) = runtime._read_gene_bfs(gene_bfs_in, **kwargs)
    (Y1, extra_genes, extra_Y) = _apply_hold_out_chrom(
        domain,
        runtime,
        Y1,
        extra_genes,
        extra_Y,
        hold_out_chrom=hold_out_chrom,
        gene_loc_file=gene_loc_file,
    )
    Y1_for_regression = copy.copy(Y1)
    extra_Y_for_regression = copy.copy(extra_Y)

    if runtime.genes is None:
        genes_union = []
        seen = set()
        for gene in extra_genes:
            if gene not in seen:
                genes_union.append(gene)
                seen.add(gene)

        runtime._set_X(runtime.X_orig, genes_union, runtime.gene_sets, skip_N=False)
        Y = np.array(extra_Y, dtype=float)
        Y_for_regression = np.array(extra_Y_for_regression, dtype=float)
        extra_genes = []
        extra_Y = np.array([])
        extra_Y_for_regression = np.array([])
    else:
        missing_value = np.nanmean(Y1) if len(Y1) > 0 else 0.0
        Y = np.array(Y1, dtype=float)
        Y[np.isnan(Y)] = missing_value
        Y_for_regression = np.array(Y1_for_regression, dtype=float)
        Y_for_regression[np.isnan(Y_for_regression)] = missing_value

    if len(extra_Y) > 0:
        Y = np.concatenate((Y, extra_Y))
        Y_for_regression = np.concatenate((Y_for_regression, extra_Y_for_regression))

        expanded_X = runtime.X_orig
        if runtime.X_orig is not None:
            expanded_X = sparse.csc_matrix(
                (runtime.X_orig.data, runtime.X_orig.indices, runtime.X_orig.indptr),
                shape=(runtime.X_orig.shape[0] + len(extra_Y), runtime.X_orig.shape[1]),
            )

        if runtime.genes is not None:
            runtime._set_X(
                expanded_X,
                runtime.genes + extra_genes,
                runtime.gene_sets,
                skip_V=True,
                skip_scale_factors=True,
                skip_N=False,
            )

    runtime._set_Y(Y, Y_for_regression, skip_V=True, skip_scale_factors=True)
    _apply_gene_stat_maps(runtime, gene_combined_map, gene_prior_map)
    eaggl_covariates.apply_loaded_gene_covariates(domain, runtime, gene_covs_in, **kwargs)


def _apply_hold_out_chrom(domain, runtime, Y_values, extra_gene_names, extra_Y_values, *, hold_out_chrom, gene_loc_file):
    if hold_out_chrom is None:
        return (Y_values, extra_gene_names, extra_Y_values)

    if runtime.gene_to_chrom is None:
        if gene_loc_file is None:
            domain.bail("Option --hold-out-chrom requires --gene-loc-file")
        (
            runtime.gene_chrom_name_pos,
            runtime.gene_to_chrom,
            runtime.gene_to_pos,
        ) = read_loc_file_with_gene_map(
            gene_loc_file,
            gene_label_map=runtime.gene_label_map,
            clean_chrom_fn=clean_chrom_name,
            warn_fn=domain.warn,
            bail_fn=domain.bail,
        )

    Y_values = np.array(Y_values, dtype=float)
    extra_gene_names = list(extra_gene_names)
    extra_Y_values = np.array(extra_Y_values, dtype=float)

    if runtime.genes is not None:
        Y_nan_mask = np.full(len(Y_values), False)
        for i, gene in enumerate(runtime.genes):
            if gene in runtime.gene_to_chrom and runtime.gene_to_chrom[gene] == hold_out_chrom:
                Y_nan_mask[i] = True
        if np.sum(Y_nan_mask) > 0:
            Y_values[Y_nan_mask] = np.nan

    if len(extra_gene_names) > 0:
        keep_mask = np.full(len(extra_gene_names), True)
        for i, gene in enumerate(extra_gene_names):
            if gene in runtime.gene_to_chrom and runtime.gene_to_chrom[gene] == hold_out_chrom:
                keep_mask[i] = False
        if np.sum(~keep_mask) > 0:
            extra_gene_names = [extra_gene_names[i] for i in range(len(extra_gene_names)) if keep_mask[i]]
            extra_Y_values = extra_Y_values[keep_mask]

    return (Y_values, extra_gene_names, extra_Y_values)


def _apply_gene_stat_maps(runtime, gene_combined_map, gene_prior_map):
    if gene_combined_map is not None:
        runtime.combined_prior_Ys = copy.copy(runtime.Y)
        for i, gene in enumerate(runtime.genes):
            if gene in gene_combined_map:
                runtime.combined_prior_Ys[i] = gene_combined_map[gene]

    if gene_prior_map is not None:
        runtime.priors = np.zeros(len(runtime.genes))
        for i, gene in enumerate(runtime.genes):
            if gene in gene_prior_map:
                runtime.priors[i] = gene_prior_map[gene]
