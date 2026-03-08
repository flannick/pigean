from __future__ import annotations

import numpy as np
import scipy.stats

from pegs_shared.cli import _default_bail
from pegs_shared.io_common import resolve_column_index
from pegs_shared.types import (
    AlignedGeneBfs,
    AlignedGeneCovariates,
    ParsedGeneBfs,
    ParsedGeneCovariates,
    ParsedGeneSetStats,
)


def parse_gene_set_statistics_file(
    stats_in,
    *,
    stats_id_col,
    stats_exp_beta_tilde_col,
    stats_beta_tilde_col,
    stats_p_col,
    stats_se_col,
    stats_beta_col,
    stats_beta_uncorrected_col,
    ignore_negative_exp_beta,
    max_gene_set_p,
    min_gene_set_beta,
    min_gene_set_beta_uncorrected,
    open_text_fn,
    get_col_fn,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if stats_in is None:
        bail_fn("Require --stats-in for this operation")

    log_fn("Reading --stats-in file %s" % stats_in)
    need_to_take_log = False
    records = {}

    with open_text_fn(stats_in) as stats_fh:
        header_cols = stats_fh.readline().strip("\n").split()
        id_col = get_col_fn(stats_id_col, header_cols)
        beta_tilde_col = None

        if stats_beta_tilde_col is not None:
            beta_tilde_col = get_col_fn(stats_beta_tilde_col, header_cols, False)
        if beta_tilde_col is not None:
            log_fn("Using col %s for beta_tilde values" % stats_beta_tilde_col)
        elif stats_exp_beta_tilde_col is not None:
            beta_tilde_col = get_col_fn(stats_exp_beta_tilde_col, header_cols)
            need_to_take_log = True
            if beta_tilde_col is not None:
                log_fn("Using %s for exp(beta_tilde) values" % stats_exp_beta_tilde_col)
            else:
                bail_fn(
                    "Could not find beta_tilde column %s or %s in header: %s"
                    % (stats_beta_tilde_col, stats_exp_beta_tilde_col, "\t".join(header_cols))
                )

        p_col = None
        if stats_p_col is not None:
            p_col = get_col_fn(stats_p_col, header_cols, False)

        se_col = None
        if stats_se_col is not None:
            se_col = get_col_fn(stats_se_col, header_cols, False)

        beta_col = None
        if stats_beta_col is not None:
            beta_col = get_col_fn(stats_beta_col, header_cols, True)
        else:
            beta_col = get_col_fn("beta", header_cols, False)

        beta_uncorrected_col = None
        if stats_beta_uncorrected_col is not None:
            beta_uncorrected_col = get_col_fn(stats_beta_uncorrected_col, header_cols, True)
        else:
            beta_uncorrected_col = get_col_fn("beta_uncorrected", header_cols, False)

        if (
            se_col is None
            and p_col is None
            and beta_tilde_col is None
            and beta_col is None
            and beta_uncorrected_col is None
        ):
            bail_fn("Require at least something to read from --gene-set-stats-in")

        for line in stats_fh:
            beta_tilde = None
            p = None
            se = None
            z = None
            beta = None
            beta_uncorrected = None

            cols = line.strip("\n").split()
            if (
                id_col > len(cols)
                or (beta_tilde_col is not None and beta_tilde_col > len(cols))
                or (p_col is not None and p_col > len(cols))
                or (se_col is not None and se_col > len(cols))
            ):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            gene_set = cols[id_col]
            if gene_set in records:
                warn_fn("Already seen gene set %s; only considering first instance" % (gene_set))
                continue

            if beta_tilde_col is not None:
                try:
                    beta_tilde = float(cols[beta_tilde_col])
                except ValueError:
                    if cols[beta_tilde_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta_tilde value %s for gene_set %s"
                            % (cols[beta_tilde_col], gene_set)
                        )
                    continue

                if need_to_take_log:
                    if beta_tilde < 0:
                        if ignore_negative_exp_beta:
                            continue
                        bail_fn(
                            "Exp(beta) value %s for gene set %s is < 0; did you mean to specify --stats-beta-col? Otherwise, specify --ignore-negative-exp-beta to ignore these"
                            % (beta_tilde, gene_set)
                        )
                    beta_tilde = np.log(beta_tilde)

            if se_col is not None:
                try:
                    se = float(cols[se_col])
                except ValueError:
                    if cols[se_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible se value %s for gene_set %s"
                            % (cols[se_col], gene_set)
                        )
                    continue

                if beta_tilde_col is not None:
                    z = beta_tilde / se
                    p = 2 * scipy.stats.norm.cdf(-np.abs(z))
                    if max_gene_set_p is not None and p > max_gene_set_p:
                        continue
            elif p_col is not None:
                try:
                    p = float(cols[p_col])
                    if max_gene_set_p is not None and p > max_gene_set_p:
                        continue
                except ValueError:
                    if cols[p_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible p value %s for gene_set %s"
                            % (cols[p_col], gene_set)
                        )
                    continue

                z = np.abs(scipy.stats.norm.ppf(p / 2))
                if z == 0:
                    warn_fn("Skipping gene_set %s due to 0 z-score" % (gene_set))
                    continue

                if beta_tilde_col is not None:
                    se = np.abs(beta_tilde) / z

            if beta_col is not None:
                try:
                    beta = float(cols[beta_col])
                    if min_gene_set_beta is not None and beta < min_gene_set_beta:
                        continue
                except ValueError:
                    if cols[beta_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta value %s for gene_set %s"
                            % (cols[beta_col], gene_set)
                        )
                    continue

            if beta_uncorrected_col is not None:
                try:
                    beta_uncorrected = float(cols[beta_uncorrected_col])
                    if (
                        min_gene_set_beta_uncorrected is not None
                        and beta_uncorrected < min_gene_set_beta_uncorrected
                    ):
                        continue
                except ValueError:
                    if cols[beta_uncorrected_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta_uncorrected value %s for gene_set %s"
                            % (cols[beta_uncorrected_col], gene_set)
                        )
                    continue

            records[gene_set] = (
                beta_tilde,
                p,
                se,
                z,
                beta,
                beta_uncorrected,
            )

    return ParsedGeneSetStats(
        need_to_take_log=need_to_take_log,
        has_beta_tilde=beta_tilde_col is not None,
        has_p_or_se=(p_col is not None or se_col is not None),
        has_beta=beta_col is not None,
        has_beta_uncorrected=beta_uncorrected_col is not None,
        records=records,
    )


def parse_gene_bfs_file(
    gene_bfs_in,
    *,
    gene_bfs_id_col,
    gene_bfs_log_bf_col,
    gene_bfs_combined_col,
    gene_bfs_prob_col,
    gene_bfs_prior_col,
    background_log_bf,
    gene_label_map,
    open_text_fn,
    get_col_fn,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if gene_bfs_in is None:
        bail_fn("Require --gene-stats-in for this operation")

    log_fn("Reading --gene-stats-in file %s" % gene_bfs_in)
    gene_in_bfs = {}
    gene_in_combined = None
    gene_in_priors = None
    with open_text_fn(gene_bfs_in) as gene_bfs_fh:
        header_cols = gene_bfs_fh.readline().strip("\n").split()
        if gene_bfs_id_col is None:
            gene_bfs_id_col = "Gene"

        id_col = get_col_fn(gene_bfs_id_col, header_cols)

        prob_col = None
        if gene_bfs_prob_col is not None:
            prob_col = get_col_fn(gene_bfs_prob_col, header_cols, True)

        bf_col = None
        if gene_bfs_log_bf_col is not None:
            bf_col = get_col_fn(gene_bfs_log_bf_col, header_cols)
        else:
            if prob_col is None:
                bf_col = get_col_fn("log_bf", header_cols)

        if bf_col is None and prob_col is None:
            bail_fn("--gene-stats-log-bf-col or --gene-stats-prob-col required for this operation")

        combined_col = None
        if gene_bfs_combined_col is not None:
            combined_col = get_col_fn(gene_bfs_combined_col, header_cols, True)
        else:
            combined_col = get_col_fn("combined", header_cols, False)

        prior_col = None
        if gene_bfs_prior_col is not None:
            prior_col = get_col_fn(gene_bfs_prior_col, header_cols, True)
        else:
            prior_col = get_col_fn("prior", header_cols, False)

        if combined_col is not None or prob_col is not None:
            gene_in_combined = {}
        if prior_col is not None:
            gene_in_priors = {}

        for line in gene_bfs_fh:
            cols = line.strip("\n").split()
            if (
                id_col >= len(cols)
                or (bf_col is not None and bf_col >= len(cols))
                or (combined_col is not None and combined_col >= len(cols))
                or (prob_col is not None and prob_col >= len(cols))
                or (prior_col is not None and prior_col >= len(cols))
            ):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            gene = cols[id_col]
            if gene_label_map is not None and gene in gene_label_map:
                gene = gene_label_map[gene]

            if bf_col is not None:
                try:
                    bf = float(cols[bf_col])
                except ValueError:
                    if cols[bf_col] != "NA":
                        warn_fn("Skipping unconvertible bf value %s for gene %s" % (cols[bf_col], gene))
                    continue
            elif prob_col is not None:
                try:
                    prob = float(cols[prob_col])
                except ValueError:
                    if cols[prob_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible prob value %s for gene %s"
                            % (cols[prob_col], gene)
                        )
                    continue
                if prob <= 0 or prob >= 1:
                    warn_fn("Skipping probability %.3g outside of (0,1)" % (prob))
                    continue
                bf = np.log(prob / (1 - prob)) - background_log_bf

            gene_in_bfs[gene] = bf

            if combined_col is not None:
                try:
                    combined = float(cols[combined_col])
                except ValueError:
                    if cols[combined_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible combined value %s for gene %s"
                            % (cols[combined_col], gene)
                        )
                    continue
                gene_in_combined[gene] = combined

            if prior_col is not None:
                try:
                    prior = float(cols[prior_col])
                except ValueError:
                    if cols[prior_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible prior value %s for gene %s"
                            % (cols[prior_col], gene)
                        )
                    continue
                gene_in_priors[gene] = prior

    return ParsedGeneBfs(
        gene_in_bfs=gene_in_bfs,
        gene_in_combined=gene_in_combined,
        gene_in_priors=gene_in_priors,
    )


def parse_gene_covariates_file(
    gene_covs_in,
    *,
    gene_covs_id_col=None,
    open_text_fn=None,
    get_col_fn=None,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _m: None
    if warn_fn is None:
        warn_fn = lambda _m: None
    if open_text_fn is None:
        open_text_fn = lambda path: open(path)
    if get_col_fn is None:
        get_col_fn = resolve_column_index

    log_fn("Reading --gene-covs-in file %s" % gene_covs_in)

    gene_in_covs = {}
    cov_names = []
    with open_text_fn(gene_covs_in) as gene_covs_fh:
        header_cols = gene_covs_fh.readline().strip("\n").split()
        if gene_covs_id_col is None:
            gene_covs_id_col = "Gene"

        id_col = get_col_fn(gene_covs_id_col, header_cols)
        cov_names = [header_cols[i] for i in range(len(header_cols)) if i != id_col]

        if len(cov_names) > 0:
            log_fn("Read covariates %s" % (",".join(cov_names)))
            for line in gene_covs_fh:
                cols = line.strip("\n").split()
                if len(cols) != len(header_cols):
                    warn_fn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                try:
                    covs = np.array([float(cols[i]) for i in range(len(cols)) if i != id_col])
                except ValueError:
                    continue

                gene_in_covs[gene] = covs

    return ParsedGeneCovariates(cov_names=cov_names, gene_to_covs=gene_in_covs)


def align_gene_scalar_map(gene_to_value, genes=None, gene_to_ind=None, missing_value=np.nan):
    if genes is None:
        genes = []
    if gene_to_ind is None:
        gene_to_ind = {}

    aligned_values = np.array([missing_value] * len(genes))
    extra_values = []
    extra_genes = []
    for gene, value in gene_to_value.items():
        if gene in gene_to_ind:
            aligned_values[gene_to_ind[gene]] = value
        else:
            extra_values.append(value)
            extra_genes.append(gene)

    return (aligned_values, extra_genes, np.array(extra_values))


def align_gene_vector_map(gene_to_values, num_values, genes=None, gene_to_ind=None, missing_value=np.nan):
    if genes is None:
        genes = []
    if gene_to_ind is None:
        gene_to_ind = {}

    aligned_values = np.full((len(genes), num_values), missing_value)
    extra_values = []
    extra_genes = []
    for gene, values in gene_to_values.items():
        if gene in gene_to_ind:
            aligned_values[gene_to_ind[gene], :] = values
        else:
            extra_values.append(values)
            extra_genes.append(gene)

    return (aligned_values, extra_genes, np.array(extra_values))


def load_aligned_gene_bfs(
    gene_bfs_in,
    *,
    genes=None,
    gene_to_ind=None,
    gene_bfs_id_col=None,
    gene_bfs_log_bf_col=None,
    gene_bfs_combined_col=None,
    gene_bfs_prob_col=None,
    gene_bfs_prior_col=None,
    background_log_bf=0.0,
    gene_label_map=None,
    open_text_fn=None,
    get_col_fn=None,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    parsed_gene_bfs = parse_gene_bfs_file(
        gene_bfs_in,
        gene_bfs_id_col=gene_bfs_id_col,
        gene_bfs_log_bf_col=gene_bfs_log_bf_col,
        gene_bfs_combined_col=gene_bfs_combined_col,
        gene_bfs_prob_col=gene_bfs_prob_col,
        gene_bfs_prior_col=gene_bfs_prior_col,
        background_log_bf=background_log_bf,
        gene_label_map=gene_label_map,
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        log_fn=log_fn,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
    )
    gene_bfs, extra_genes, extra_gene_bfs = align_gene_scalar_map(
        parsed_gene_bfs.gene_in_bfs,
        genes=genes,
        gene_to_ind=gene_to_ind,
    )
    return AlignedGeneBfs(
        gene_bfs=gene_bfs,
        extra_genes=extra_genes,
        extra_gene_bfs=extra_gene_bfs,
        gene_in_combined=parsed_gene_bfs.gene_in_combined,
        gene_in_priors=parsed_gene_bfs.gene_in_priors,
    )


def load_aligned_gene_covariates(
    gene_covs_in,
    *,
    genes=None,
    gene_to_ind=None,
    gene_covs_id_col=None,
    open_text_fn=None,
    get_col_fn=None,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    parsed_gene_covs = parse_gene_covariates_file(
        gene_covs_in,
        gene_covs_id_col=gene_covs_id_col,
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        log_fn=log_fn,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
    )
    gene_covs, extra_genes, extra_gene_covs = align_gene_vector_map(
        parsed_gene_covs.gene_to_covs,
        num_values=len(parsed_gene_covs.cov_names),
        genes=genes,
        gene_to_ind=gene_to_ind,
    )
    return AlignedGeneCovariates(
        cov_names=parsed_gene_covs.cov_names,
        gene_covs=gene_covs,
        extra_genes=extra_genes,
        extra_gene_covs=extra_gene_covs,
    )
