from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse as sparse

from pegs_shared.bundle import is_huge_statistics_bundle_path
from pegs_shared.cli import _default_bail, json_safe


def coerce_runtime_state_dict(runtime_state, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if isinstance(runtime_state, dict):
        return runtime_state
    if hasattr(runtime_state, "__dict__"):
        return runtime_state.__dict__
    bail_fn("Internal error: unsupported runtime state container for HuGE cache IO")


def get_huge_statistics_paths_for_prefix(prefix):
    return {
        "meta": "%s.huge.meta.json.gz" % prefix,
        "cache_genes": "%s.huge.cache_genes.tsv.gz" % prefix,
        "extra_scores": "%s.huge.extra_scores.tsv.gz" % prefix,
        "matrix_row_genes": "%s.huge.matrix_row_genes.tsv.gz" % prefix,
        "gene_scores": "%s.huge.gene_scores.tsv.gz" % prefix,
        "gene_covariates": "%s.huge.gene_covariates.tsv.gz" % prefix,
        "bfs_data": "%s.huge_signal_bfs.data.tsv.gz" % prefix,
        "bfs_indices": "%s.huge_signal_bfs.indices.tsv.gz" % prefix,
        "bfs_indptr": "%s.huge_signal_bfs.indptr.tsv.gz" % prefix,
        "bfs_reg_data": "%s.huge_signal_bfs_for_regression.data.tsv.gz" % prefix,
        "bfs_reg_indices": "%s.huge_signal_bfs_for_regression.indices.tsv.gz" % prefix,
        "bfs_reg_indptr": "%s.huge_signal_bfs_for_regression.indptr.tsv.gz" % prefix,
        "signal_posteriors": "%s.huge_signal_posteriors.tsv.gz" % prefix,
        "signal_posteriors_for_regression": "%s.huge_signal_posteriors_for_regression.tsv.gz" % prefix,
        "signal_sum_gene_cond_probabilities": "%s.huge_signal_sum_gene_cond_probabilities.tsv.gz" % prefix,
        "signal_sum_gene_cond_probabilities_for_regression": "%s.huge_signal_sum_gene_cond_probabilities_for_regression.tsv.gz" % prefix,
        "signal_mean_gene_pos": "%s.huge_signal_mean_gene_pos.tsv.gz" % prefix,
        "signal_mean_gene_pos_for_regression": "%s.huge_signal_mean_gene_pos_for_regression.tsv.gz" % prefix,
    }


def write_numeric_vector_file(out_file, values, *, open_text_fn, value_type=float):
    with open_text_fn(out_file, "w") as out_fh:
        if values is None:
            return
        values = np.ravel(np.array(values))
        for value in values:
            if value_type == int:
                out_fh.write("%d\n" % int(value))
            else:
                out_fh.write("%.18g\n" % float(value))


def read_numeric_vector_file(in_file, *, open_text_fn, value_type=float):
    values = []
    with open_text_fn(in_file) as in_fh:
        for line in in_fh:
            line = line.strip()
            if line == "":
                continue
            if value_type == int:
                values.append(int(line))
            else:
                values.append(float(line))
    if value_type == int:
        return np.array(values, dtype=int)
    return np.array(values, dtype=float)


def build_huge_statistics_matrix_row_genes(cache_genes, extra_genes, num_matrix_rows, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if len(cache_genes) > 0:
        if num_matrix_rows < len(cache_genes):
            bail_fn("Error writing HuGE statistics cache: matrix rows %d < number of genes %d" % (num_matrix_rows, len(cache_genes)))
        num_extra_matrix_rows = num_matrix_rows - len(cache_genes)
        if num_extra_matrix_rows > len(extra_genes):
            bail_fn("Error writing HuGE statistics cache: matrix rows require %d extra genes but only %d were provided" % (num_extra_matrix_rows, len(extra_genes)))
        return cache_genes + extra_genes[:num_extra_matrix_rows]
    if num_matrix_rows > len(extra_genes):
        bail_fn("Error writing HuGE statistics cache: matrix rows %d > number of extra genes %d" % (num_matrix_rows, len(extra_genes)))
    return extra_genes[:num_matrix_rows]


def build_huge_statistics_score_maps(runtime_state, cache_genes, extra_genes, gene_bf, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression):
    gene_to_score = {}
    if runtime_state.get("gene_to_gwas_huge_score") is not None:
        gene_to_score = dict(runtime_state["gene_to_gwas_huge_score"])

    gene_to_score_uncorrected = {}
    if runtime_state.get("gene_to_gwas_huge_score_uncorrected") is not None:
        gene_to_score_uncorrected = dict(runtime_state["gene_to_gwas_huge_score_uncorrected"])

    gene_to_score_for_regression = {}

    for i, gene in enumerate(cache_genes):
        if gene_bf is not None and i < len(gene_bf):
            gene_to_score[gene] = float(gene_bf[i])
        if gene_bf_for_regression is not None and i < len(gene_bf_for_regression):
            gene_to_score_for_regression[gene] = float(gene_bf_for_regression[i])
        elif gene in gene_to_score:
            gene_to_score_for_regression[gene] = float(gene_to_score[gene])

    for i, gene in enumerate(extra_genes):
        if i < len(extra_gene_bf):
            gene_to_score[gene] = float(extra_gene_bf[i])
        if i < len(extra_gene_bf_for_regression):
            gene_to_score_for_regression[gene] = float(extra_gene_bf_for_regression[i])
        elif gene in gene_to_score:
            gene_to_score_for_regression[gene] = float(gene_to_score[gene])
        if gene not in gene_to_score_uncorrected:
            gene_to_score_uncorrected[gene] = float(extra_gene_bf[i])

    return (gene_to_score, gene_to_score_uncorrected, gene_to_score_for_regression)


def build_huge_statistics_meta(runtime_state, huge_signal_bfs, huge_signal_bfs_for_regression, *, json_safe_fn=None):
    if json_safe_fn is None:
        json_safe_fn = json_safe
    return {
        "version": 1,
        "huge_signal_bfs_shape": [int(huge_signal_bfs.shape[0]), int(huge_signal_bfs.shape[1])],
        "huge_signal_bfs_for_regression_shape": [int(huge_signal_bfs_for_regression.shape[0]), int(huge_signal_bfs_for_regression.shape[1])],
        "huge_signal_max_closest_gene_prob": (None if runtime_state.get("huge_signal_max_closest_gene_prob") is None else float(runtime_state.get("huge_signal_max_closest_gene_prob"))),
        "huge_cap_region_posterior": bool(runtime_state.get("huge_cap_region_posterior", True)),
        "huge_scale_region_posterior": bool(runtime_state.get("huge_scale_region_posterior", False)),
        "huge_phantom_region_posterior": bool(runtime_state.get("huge_phantom_region_posterior", False)),
        "huge_allow_evidence_of_absence": bool(runtime_state.get("huge_allow_evidence_of_absence", False)),
        "huge_sparse_mode": bool(runtime_state.get("huge_sparse_mode", False)),
        "huge_signals": [] if runtime_state.get("huge_signals") is None else [[str(x[0]), int(x[1]), float(x[2]), x[3]] for x in runtime_state.get("huge_signals")],
        "gene_covariate_names": (None if runtime_state.get("gene_covariate_names") is None else list(runtime_state.get("gene_covariate_names"))),
        "gene_covariate_directions": (None if runtime_state.get("gene_covariate_directions") is None else list(np.array(runtime_state.get("gene_covariate_directions"), dtype=float))),
        "gene_covariate_intercept_index": runtime_state.get("gene_covariate_intercept_index"),
        "gene_covariate_slope_defaults": (None if runtime_state.get("gene_covariate_slope_defaults") is None else list(np.array(runtime_state.get("gene_covariate_slope_defaults"), dtype=float))),
        "total_qc_metric_betas_defaults": (None if runtime_state.get("total_qc_metric_betas_defaults") is None else list(np.array(runtime_state.get("total_qc_metric_betas_defaults"), dtype=float))),
        "total_qc_metric_intercept_defaults": (None if runtime_state.get("total_qc_metric_intercept_defaults") is None else float(runtime_state.get("total_qc_metric_intercept_defaults"))),
        "total_qc_metric2_betas_defaults": (None if runtime_state.get("total_qc_metric2_betas_defaults") is None else list(np.array(runtime_state.get("total_qc_metric2_betas_defaults"), dtype=float))),
        "total_qc_metric2_intercept_defaults": (None if runtime_state.get("total_qc_metric2_intercept_defaults") is None else float(runtime_state.get("total_qc_metric2_intercept_defaults"))),
        "recorded_params": json_safe_fn(runtime_state.get("params")),
        "recorded_param_keys": json_safe_fn(runtime_state.get("param_keys")),
    }


def write_huge_statistics_text_tables(
    paths,
    runtime_state,
    cache_genes,
    extra_genes,
    extra_gene_bf,
    extra_gene_bf_for_regression,
    matrix_row_genes,
    gene_to_score,
    gene_to_score_uncorrected,
    gene_to_score_for_regression,
    *,
    open_text_fn,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    with open_text_fn(paths["cache_genes"], "w") as out_fh:
        for gene in cache_genes:
            out_fh.write("%s\n" % gene)

    with open_text_fn(paths["extra_scores"], "w") as out_fh:
        out_fh.write("Gene\tlog_bf\tlog_bf_for_regression\n")
        for i in range(len(extra_genes)):
            bf = np.nan
            if i < len(extra_gene_bf):
                bf = extra_gene_bf[i]
            bf_for_regression = np.nan
            if i < len(extra_gene_bf_for_regression):
                bf_for_regression = extra_gene_bf_for_regression[i]
            out_fh.write("%s\t%.18g\t%.18g\n" % (extra_genes[i], bf, bf_for_regression))

    with open_text_fn(paths["matrix_row_genes"], "w") as out_fh:
        for gene in matrix_row_genes:
            out_fh.write("%s\n" % gene)

    ordered_genes = []
    seen = set()
    for gene in cache_genes + extra_genes + list(gene_to_score.keys()) + list(gene_to_score_uncorrected.keys()):
        if gene not in seen:
            seen.add(gene)
            ordered_genes.append(gene)

    with open_text_fn(paths["gene_scores"], "w") as out_fh:
        out_fh.write("Gene\tlog_bf\tlog_bf_uncorrected\tlog_bf_for_regression\n")
        for gene in ordered_genes:
            score = gene_to_score.get(gene, np.nan)
            score_uncorrected = gene_to_score_uncorrected.get(gene, np.nan)
            score_for_regression = gene_to_score_for_regression.get(gene, np.nan)
            out_fh.write("%s\t%.18g\t%.18g\t%.18g\n" % (gene, score, score_uncorrected, score_for_regression))

    gene_covariates = runtime_state.get("gene_covariates")
    if gene_covariates is not None:
        if gene_covariates.shape[0] != len(matrix_row_genes):
            bail_fn("Error writing HuGE statistics cache: gene covariates have %d rows but matrix has %d rows" % (gene_covariates.shape[0], len(matrix_row_genes)))
        with open_text_fn(paths["gene_covariates"], "w") as out_fh:
            out_fh.write("Gene\t%s\n" % ("\t".join(runtime_state.get("gene_covariate_names"))))
            for i in range(len(matrix_row_genes)):
                out_fh.write("%s\t%s\n" % (matrix_row_genes[i], "\t".join(["%.18g" % x for x in gene_covariates[i, :]])))


def read_huge_statistics_text_tables(paths, *, open_text_fn):
    cache_genes = []
    with open_text_fn(paths["cache_genes"]) as in_fh:
        for line in in_fh:
            gene = line.strip()
            if gene != "":
                cache_genes.append(gene)

    extra_genes = []
    extra_gene_bf = []
    extra_gene_bf_for_regression = []
    with open_text_fn(paths["extra_scores"]) as in_fh:
        _header = in_fh.readline()
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) < 3:
                continue
            extra_genes.append(cols[0])
            extra_gene_bf.append(float(cols[1]))
            extra_gene_bf_for_regression.append(float(cols[2]))

    matrix_row_genes = []
    with open_text_fn(paths["matrix_row_genes"]) as in_fh:
        for line in in_fh:
            gene = line.strip()
            if gene != "":
                matrix_row_genes.append(gene)

    gene_to_score = {}
    gene_to_score_uncorrected = {}
    gene_to_score_for_regression = {}
    with open_text_fn(paths["gene_scores"]) as in_fh:
        _header = in_fh.readline()
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) < 4:
                continue
            gene = cols[0]
            gene_to_score[gene] = float(cols[1])
            gene_to_score_uncorrected[gene] = float(cols[2])
            gene_to_score_for_regression[gene] = float(cols[3])

    return (
        cache_genes,
        extra_genes,
        extra_gene_bf,
        extra_gene_bf_for_regression,
        matrix_row_genes,
        gene_to_score,
        gene_to_score_uncorrected,
        gene_to_score_for_regression,
    )


def resolve_huge_statistics_gene_vectors(
    runtime_state,
    cache_genes,
    extra_genes,
    matrix_row_genes,
    gene_to_score,
    gene_to_score_for_regression,
    *,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    genes = runtime_state.get("genes")
    if genes is None:
        if len(cache_genes) > 0:
            bail_fn("HuGE cache was generated with preloaded genes but this run has no preloaded genes")
        if len(matrix_row_genes) > len(extra_genes) or matrix_row_genes != extra_genes[:len(matrix_row_genes)]:
            bail_fn("HuGE cache is inconsistent: matrix rows do not match extra gene ordering")
        return (np.array([]), np.array([]))

    if cache_genes != genes:
        bail_fn("HuGE cache gene ordering does not match current run. Rebuild cache for this run setup.")
    if len(matrix_row_genes) < len(genes) or matrix_row_genes[:len(genes)] != genes:
        bail_fn("HuGE cache matrix row ordering does not match current run genes")

    gene_bf = np.array([gene_to_score.get(gene, np.nan) for gene in genes])
    gene_bf_for_regression = np.array([gene_to_score_for_regression.get(gene, np.nan) for gene in genes])
    return (gene_bf, gene_bf_for_regression)


def read_huge_statistics_covariates_if_present(runtime_state, paths, *, open_text_fn, exists_fn=None):
    if exists_fn is None:
        exists_fn = os.path.exists
    if not exists_fn(paths["gene_covariates"]):
        return
    covariate_rows = []
    with open_text_fn(paths["gene_covariates"]) as in_fh:
        header = in_fh.readline().strip("\n").split("\t")
        if len(header) > 1 and runtime_state.get("gene_covariate_names") is None:
            runtime_state["gene_covariate_names"] = header[1:]
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) <= 1:
                continue
            covariate_rows.append([float(x) for x in cols[1:]])
    runtime_state["gene_covariates"] = np.array(covariate_rows)


def load_huge_statistics_sparse_and_vectors(runtime_state, paths, meta, *, read_vector_fn):
    huge_signal_bfs_shape = tuple(meta["huge_signal_bfs_shape"])
    huge_signal_bfs_for_regression_shape = tuple(meta["huge_signal_bfs_for_regression_shape"])

    sparse_components = {
        "bfs_data": read_vector_fn(paths["bfs_data"], value_type=float),
        "bfs_indices": read_vector_fn(paths["bfs_indices"], value_type=int),
        "bfs_indptr": read_vector_fn(paths["bfs_indptr"], value_type=int),
        "bfs_reg_data": read_vector_fn(paths["bfs_reg_data"], value_type=float),
        "bfs_reg_indices": read_vector_fn(paths["bfs_reg_indices"], value_type=int),
        "bfs_reg_indptr": read_vector_fn(paths["bfs_reg_indptr"], value_type=int),
    }

    runtime_state["huge_signal_bfs"] = sparse.csc_matrix(
        (sparse_components["bfs_data"], sparse_components["bfs_indices"], sparse_components["bfs_indptr"]),
        shape=huge_signal_bfs_shape,
    )
    runtime_state["huge_signal_bfs_for_regression"] = sparse.csc_matrix(
        (sparse_components["bfs_reg_data"], sparse_components["bfs_reg_indices"], sparse_components["bfs_reg_indptr"]),
        shape=huge_signal_bfs_for_regression_shape,
    )

    runtime_vector_map = (
        ("huge_signal_posteriors", "signal_posteriors"),
        ("huge_signal_posteriors_for_regression", "signal_posteriors_for_regression"),
        ("huge_signal_sum_gene_cond_probabilities", "signal_sum_gene_cond_probabilities"),
        ("huge_signal_sum_gene_cond_probabilities_for_regression", "signal_sum_gene_cond_probabilities_for_regression"),
        ("huge_signal_mean_gene_pos", "signal_mean_gene_pos"),
        ("huge_signal_mean_gene_pos_for_regression", "signal_mean_gene_pos_for_regression"),
    )
    for state_key, path_key in runtime_vector_map:
        runtime_state[state_key] = read_vector_fn(paths[path_key], value_type=float)


def apply_huge_statistics_meta_to_runtime(runtime_state, meta):
    runtime_state["huge_signal_max_closest_gene_prob"] = meta["huge_signal_max_closest_gene_prob"]
    runtime_state["huge_cap_region_posterior"] = bool(meta["huge_cap_region_posterior"])
    runtime_state["huge_scale_region_posterior"] = bool(meta["huge_scale_region_posterior"])
    runtime_state["huge_phantom_region_posterior"] = bool(meta["huge_phantom_region_posterior"])
    runtime_state["huge_allow_evidence_of_absence"] = bool(meta["huge_allow_evidence_of_absence"])
    runtime_state["huge_sparse_mode"] = bool(meta["huge_sparse_mode"])
    runtime_state["huge_signals"] = [tuple(x) for x in meta["huge_signals"]]

    runtime_state["gene_covariates"] = None
    runtime_state["gene_covariates_mask"] = None
    runtime_state["gene_covariate_names"] = meta.get("gene_covariate_names")
    runtime_state["gene_covariate_directions"] = None
    if meta.get("gene_covariate_directions") is not None:
        runtime_state["gene_covariate_directions"] = np.array(meta["gene_covariate_directions"], dtype=float)
    runtime_state["gene_covariate_intercept_index"] = meta.get("gene_covariate_intercept_index")
    runtime_state["gene_covariates_mat_inv"] = None
    runtime_state["gene_covariate_zs"] = None
    runtime_state["gene_covariate_adjustments"] = None

    runtime_state["gene_covariate_slope_defaults"] = None if meta.get("gene_covariate_slope_defaults") is None else np.array(meta.get("gene_covariate_slope_defaults"), dtype=float)
    runtime_state["total_qc_metric_betas_defaults"] = None if meta.get("total_qc_metric_betas_defaults") is None else np.array(meta.get("total_qc_metric_betas_defaults"), dtype=float)
    runtime_state["total_qc_metric_intercept_defaults"] = meta.get("total_qc_metric_intercept_defaults")
    runtime_state["total_qc_metric2_betas_defaults"] = None if meta.get("total_qc_metric2_betas_defaults") is None else np.array(meta.get("total_qc_metric2_betas_defaults"), dtype=float)
    runtime_state["total_qc_metric2_intercept_defaults"] = meta.get("total_qc_metric2_intercept_defaults")


def combine_runtime_huge_scores(runtime_state):
    if runtime_state.get("gene_to_gwas_huge_score") is not None and runtime_state.get("gene_to_exomes_huge_score") is not None:
        runtime_state["gene_to_huge_score"] = {}
        genes = list(set().union(runtime_state["gene_to_gwas_huge_score"], runtime_state["gene_to_exomes_huge_score"]))
        for gene in genes:
            runtime_state["gene_to_huge_score"][gene] = 0
            if gene in runtime_state["gene_to_gwas_huge_score"]:
                runtime_state["gene_to_huge_score"][gene] += runtime_state["gene_to_gwas_huge_score"][gene]
            if gene in runtime_state["gene_to_exomes_huge_score"]:
                runtime_state["gene_to_huge_score"][gene] += runtime_state["gene_to_exomes_huge_score"][gene]


def validate_huge_statistics_loaded_shapes(runtime_state, matrix_row_genes, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if runtime_state["huge_signal_bfs"].shape[1] != len(runtime_state["huge_signals"]):
        bail_fn("HuGE cache is inconsistent: huge_signal_bfs has %d columns but found %d signals" % (runtime_state["huge_signal_bfs"].shape[1], len(runtime_state["huge_signals"])))
    if runtime_state["huge_signal_bfs"].shape[0] != len(matrix_row_genes):
        bail_fn("HuGE cache is inconsistent: huge_signal_bfs has %d rows but found %d matrix-row genes" % (runtime_state["huge_signal_bfs"].shape[0], len(matrix_row_genes)))


def write_huge_statistics_runtime_vectors(paths, runtime_state, *, write_vector_fn):
    runtime_vector_map = (
        ("signal_posteriors", "huge_signal_posteriors"),
        ("signal_posteriors_for_regression", "huge_signal_posteriors_for_regression"),
        ("signal_sum_gene_cond_probabilities", "huge_signal_sum_gene_cond_probabilities"),
        ("signal_sum_gene_cond_probabilities_for_regression", "huge_signal_sum_gene_cond_probabilities_for_regression"),
        ("signal_mean_gene_pos", "huge_signal_mean_gene_pos"),
        ("signal_mean_gene_pos_for_regression", "huge_signal_mean_gene_pos_for_regression"),
    )
    for path_key, state_key in runtime_vector_map:
        write_vector_fn(paths[path_key], runtime_state.get(state_key), value_type=float)


def write_huge_statistics_sparse_components(paths, huge_signal_bfs, huge_signal_bfs_for_regression, *, write_vector_fn):
    sparse_vector_map = (
        ("bfs_data", huge_signal_bfs.data, float),
        ("bfs_indices", huge_signal_bfs.indices, int),
        ("bfs_indptr", huge_signal_bfs.indptr, int),
        ("bfs_reg_data", huge_signal_bfs_for_regression.data, float),
        ("bfs_reg_indices", huge_signal_bfs_for_regression.indices, int),
        ("bfs_reg_indptr", huge_signal_bfs_for_regression.indptr, int),
    )
    for path_key, values, value_type in sparse_vector_map:
        write_vector_fn(paths[path_key], values, value_type=value_type)
