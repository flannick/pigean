from __future__ import annotations

import json
import os
from types import SimpleNamespace

import scipy.sparse as sparse

from pegs_shared.io_common import resolve_column_index
from pegs_utils import infer_columns_from_table_file


class IntervalTree(object):
    __slots__ = ('interval_starts', 'interval_stops', 'left', 'right', 'center')

    def __init__(self, intervals, depth=16, minbucket=96, _extent=None, maxbucket=4096):
        depth -= 1
        if (depth == 0 or len(intervals) < minbucket) and len(intervals) > maxbucket:
            self.interval_starts, self.interval_stops = zip(*intervals)
            self.left = self.right = None
            return

        left, right = _extent or (min(i[0] for i in intervals), max(i[1] for i in intervals))
        center = (left + right) / 2.0

        self.interval_starts = []
        self.interval_stops = []
        lefts, rights = [], []

        for interval in intervals:
            if interval[1] < center:
                lefts.append(interval)
            elif interval[0] > center:
                rights.append(interval)
            else:
                self.interval_starts.append(interval[0])
                self.interval_stops.append(interval[1])

        self.interval_starts = domain_np_array(self.interval_starts)
        self.interval_stops = domain_np_array(self.interval_stops)
        self.left = lefts and IntervalTree(lefts, depth, minbucket, (left, center)) or None
        self.right = rights and IntervalTree(rights, depth, minbucket, (center, right)) or None
        self.center = center

    def find(self, start, stop, index_map=None):
        less_mask = domain_np_less(self.interval_starts, stop[:, domain_np_newaxis()] + 1)
        greater_mask = domain_np_greater(self.interval_stops, start[:, domain_np_newaxis()] - 1)
        overlapping_mask = domain_np_logical_and(less_mask, greater_mask)
        overlapping_where = domain_np_where(overlapping_mask)

        overlapping_indices = (
            overlapping_where[0],
            self.interval_starts[overlapping_where[1]],
            self.interval_stops[overlapping_where[1]],
        )

        start_less_mask = start <= self.center
        if self.left and domain_np_any(start_less_mask):
            left_overlapping_indices = self.left.find(
                start[start_less_mask],
                stop[start_less_mask],
                index_map=domain_np_where(start_less_mask)[0],
            )
            overlapping_indices = (
                domain_np_append(overlapping_indices[0], left_overlapping_indices[0]),
                domain_np_append(overlapping_indices[1], left_overlapping_indices[1]),
                domain_np_append(overlapping_indices[2], left_overlapping_indices[2]),
            )

        stop_greater_mask = stop >= self.center
        if self.right and domain_np_any(stop_greater_mask):
            right_overlapping_indices = self.right.find(
                start[stop_greater_mask],
                stop[stop_greater_mask],
                index_map=domain_np_where(stop_greater_mask)[0],
            )
            overlapping_indices = (
                domain_np_append(overlapping_indices[0], right_overlapping_indices[0]),
                domain_np_append(overlapping_indices[1], right_overlapping_indices[1]),
                domain_np_append(overlapping_indices[2], right_overlapping_indices[2]),
            )

        if index_map is not None and len(overlapping_indices[0]) > 0:
            overlapping_indices = (
                index_map[overlapping_indices[0]],
                overlapping_indices[1],
                overlapping_indices[2],
            )

        return overlapping_indices


_NUMPY_FNS = {}


def configure_numpy(np_module):
    global _NUMPY_FNS
    _NUMPY_FNS = {
        "array": np_module.array,
        "less": np_module.less,
        "greater": np_module.greater,
        "logical_and": np_module.logical_and,
        "where": np_module.where,
        "any": np_module.any,
        "append": np_module.append,
        "newaxis": np_module.newaxis,
    }


def domain_np_array(values):
    return _NUMPY_FNS["array"](values)


def domain_np_less(a, b):
    return _NUMPY_FNS["less"](a, b)


def domain_np_greater(a, b):
    return _NUMPY_FNS["greater"](a, b)


def domain_np_logical_and(a, b):
    return _NUMPY_FNS["logical_and"](a, b)


def domain_np_where(mask):
    return _NUMPY_FNS["where"](mask)


def domain_np_any(mask):
    return _NUMPY_FNS["any"](mask)


def domain_np_append(a, b):
    return _NUMPY_FNS["append"](a, b)


def domain_np_newaxis():
    return _NUMPY_FNS["newaxis"]


def get_col(col_name_or_index, header_cols, require_match=True, *, bail_fn):
    return resolve_column_index(
        col_name_or_index,
        header_cols,
        require_match=require_match,
        bail_fn=bail_fn,
    )


def determine_columns_from_file(filename, *, open_gz_fn, log_fn, bail_fn):
    return infer_columns_from_table_file(
        filename,
        open_gz_fn,
        log_fn=log_fn,
        bail_fn=bail_fn,
    )


def _build_support_domain(**kwargs):
    return SimpleNamespace(**kwargs)


def needs_gwas_column_detection_explicit(
    *,
    pegs_needs_gwas_column_detection,
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_n_col,
    gwas_n,
):
    domain = _build_support_domain(
        pegs_needs_gwas_column_detection=pegs_needs_gwas_column_detection,
    )
    return needs_gwas_column_detection(
        domain,
        gwas_pos_col,
        gwas_chrom_col,
        gwas_locus_col,
        gwas_p_col,
        gwas_beta_col,
        gwas_se_col,
        gwas_n_col,
        gwas_n,
    )


def autodetect_gwas_columns_explicit(
    *,
    pegs_autodetect_gwas_columns,
    gwas_in,
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_freq_col,
    gwas_n_col,
    gwas_n,
    debug_just_check_header=False,
    infer_columns_fn,
    log_fn,
    bail_fn,
):
    domain = _build_support_domain(
        pegs_autodetect_gwas_columns=pegs_autodetect_gwas_columns,
        _determine_columns_from_file=infer_columns_fn,
        log=log_fn,
        bail=bail_fn,
    )
    return autodetect_gwas_columns(
        domain,
        gwas_in,
        gwas_pos_col,
        gwas_chrom_col,
        gwas_locus_col,
        gwas_p_col,
        gwas_beta_col,
        gwas_se_col,
        gwas_freq_col,
        gwas_n_col,
        gwas_n,
        debug_just_check_header=debug_just_check_header,
    )


def load_huge_gene_and_exon_locations_explicit(
    *,
    np_module,
    gene_loc_file,
    gene_label_map,
    hold_out_chrom=None,
    exons_loc_file=None,
    read_loc_file_with_gene_map_fn,
    clean_chrom_fn,
    log_fn,
    warn_fn,
    bail_fn,
):
    configure_numpy(np_module)
    domain = _build_support_domain(
        pegs_read_loc_file_with_gene_map=read_loc_file_with_gene_map_fn,
        pegs_clean_chrom_name=clean_chrom_fn,
        log=log_fn,
        warn=warn_fn,
        bail=bail_fn,
    )
    return load_huge_gene_and_exon_locations(
        domain,
        gene_loc_file,
        gene_label_map,
        hold_out_chrom=hold_out_chrom,
        exons_loc_file=exons_loc_file,
    )


def compute_huge_variant_thresholds_explicit(
    min_var_posterior,
    gwas_high_p_posterior,
    allelic_var_k,
    gwas_prior_odds,
    *,
    np_module,
    scipy_module,
    log_fn,
):
    domain = _build_support_domain(
        np=np_module,
        scipy=scipy_module,
        log=log_fn,
    )
    return compute_huge_variant_thresholds(
        domain,
        min_var_posterior,
        gwas_high_p_posterior,
        allelic_var_k,
        gwas_prior_odds,
    )


def validate_and_normalize_huge_gwas_inputs_explicit(
    gwas_in,
    gene_loc_file,
    credible_sets_in=None,
    credible_sets_chrom_col=None,
    credible_sets_pos_col=None,
    signal_window_size=250000,
    signal_min_sep=100000,
    signal_max_logp_ratio=None,
    *,
    warn_fn,
    bail_fn,
):
    domain = _build_support_domain(
        warn=warn_fn,
        bail=bail_fn,
    )
    return validate_and_normalize_huge_gwas_inputs(
        domain,
        gwas_in,
        gene_loc_file,
        credible_sets_in=credible_sets_in,
        credible_sets_chrom_col=credible_sets_chrom_col,
        credible_sets_pos_col=credible_sets_pos_col,
        signal_window_size=signal_window_size,
        signal_min_sep=signal_min_sep,
        signal_max_logp_ratio=signal_max_logp_ratio,
    )


def write_huge_statistics_bundle_explicit(
    runtime_state,
    prefix,
    gene_bf,
    extra_genes,
    extra_gene_bf,
    gene_bf_for_regression,
    extra_gene_bf_for_regression,
    *,
    pegs_coerce_runtime_state_dict,
    pegs_get_huge_statistics_paths_for_prefix,
    pegs_build_huge_statistics_matrix_row_genes,
    pegs_build_huge_statistics_score_maps,
    pegs_build_huge_statistics_meta,
    pegs_write_huge_statistics_text_tables,
    pegs_write_huge_statistics_runtime_vectors,
    pegs_write_huge_statistics_sparse_components,
    pegs_write_numeric_vector_file,
    open_gz_fn,
    json_safe_fn,
    bail_fn,
):
    domain = _build_support_domain(
        pegs_coerce_runtime_state_dict=pegs_coerce_runtime_state_dict,
        pegs_get_huge_statistics_paths_for_prefix=pegs_get_huge_statistics_paths_for_prefix,
        pegs_build_huge_statistics_matrix_row_genes=pegs_build_huge_statistics_matrix_row_genes,
        pegs_build_huge_statistics_score_maps=pegs_build_huge_statistics_score_maps,
        pegs_build_huge_statistics_meta=pegs_build_huge_statistics_meta,
        pegs_write_huge_statistics_text_tables=pegs_write_huge_statistics_text_tables,
        pegs_write_huge_statistics_runtime_vectors=pegs_write_huge_statistics_runtime_vectors,
        pegs_write_huge_statistics_sparse_components=pegs_write_huge_statistics_sparse_components,
        pegs_write_numeric_vector_file=pegs_write_numeric_vector_file,
        open_gz=open_gz_fn,
        _json_safe=json_safe_fn,
        bail=bail_fn,
        json=json,
        sparse=sparse,
    )
    return write_huge_statistics_bundle(
        domain,
        runtime_state,
        prefix,
        gene_bf,
        extra_genes,
        extra_gene_bf,
        gene_bf_for_regression,
        extra_gene_bf_for_regression,
    )


def read_huge_statistics_bundle_explicit(
    runtime_state,
    prefix,
    *,
    np_module,
    pegs_coerce_runtime_state_dict,
    pegs_get_huge_statistics_paths_for_prefix,
    pegs_read_huge_statistics_text_tables,
    pegs_resolve_huge_statistics_gene_vectors,
    pegs_load_huge_statistics_sparse_and_vectors,
    pegs_apply_huge_statistics_meta_to_runtime,
    pegs_read_huge_statistics_covariates_if_present,
    pegs_combine_runtime_huge_scores,
    pegs_validate_huge_statistics_loaded_shapes,
    pegs_read_numeric_vector_file,
    open_gz_fn,
    bail_fn,
):
    domain = _build_support_domain(
        np=np_module,
        pegs_coerce_runtime_state_dict=pegs_coerce_runtime_state_dict,
        pegs_get_huge_statistics_paths_for_prefix=pegs_get_huge_statistics_paths_for_prefix,
        pegs_read_huge_statistics_text_tables=pegs_read_huge_statistics_text_tables,
        pegs_resolve_huge_statistics_gene_vectors=pegs_resolve_huge_statistics_gene_vectors,
        pegs_load_huge_statistics_sparse_and_vectors=pegs_load_huge_statistics_sparse_and_vectors,
        pegs_apply_huge_statistics_meta_to_runtime=pegs_apply_huge_statistics_meta_to_runtime,
        pegs_read_huge_statistics_covariates_if_present=pegs_read_huge_statistics_covariates_if_present,
        pegs_combine_runtime_huge_scores=pegs_combine_runtime_huge_scores,
        pegs_validate_huge_statistics_loaded_shapes=pegs_validate_huge_statistics_loaded_shapes,
        pegs_read_numeric_vector_file=pegs_read_numeric_vector_file,
        open_gz=open_gz_fn,
        bail=bail_fn,
        json=json,
        os=os,
    )
    return read_huge_statistics_bundle(domain, runtime_state, prefix)


def needs_gwas_column_detection(domain, *args):
    return domain.pegs_needs_gwas_column_detection(*args)


def autodetect_gwas_columns(domain, *args, **kwargs):
    return domain.pegs_autodetect_gwas_columns(
        *args,
        infer_columns_fn=domain._determine_columns_from_file,
        log_fn=domain.log,
        bail_fn=domain.bail,
        **kwargs,
    )


def load_huge_gene_and_exon_locations(domain, gene_loc_file, gene_label_map, hold_out_chrom=None, exons_loc_file=None):
    domain.log("Reading gene locations")
    (gene_chrom_name_pos, gene_to_chrom, gene_to_pos) = domain.pegs_read_loc_file_with_gene_map(
        gene_loc_file,
        gene_label_map=gene_label_map,
        hold_out_chrom=hold_out_chrom,
        clean_chrom_fn=domain.pegs_clean_chrom_name,
        warn_fn=domain.warn,
        bail_fn=domain.bail,
    )

    for chrom in gene_chrom_name_pos:
        serialized_gene_info = []
        for gene in gene_chrom_name_pos[chrom]:
            for pos in gene_chrom_name_pos[chrom][gene]:
                serialized_gene_info.append((gene, pos))
        gene_chrom_name_pos[chrom] = serialized_gene_info

    chrom_to_interval_tree = None
    if exons_loc_file is not None:
        domain.log("Reading exon locations")
        chrom_interval_to_gene = domain.pegs_read_loc_file_with_gene_map(
            exons_loc_file,
            gene_label_map=gene_label_map,
            return_intervals=True,
            clean_chrom_fn=domain.pegs_clean_chrom_name,
            warn_fn=domain.warn,
            bail_fn=domain.bail,
        )
        chrom_to_interval_tree = {}
        for chrom in chrom_interval_to_gene:
            chrom_to_interval_tree[chrom] = IntervalTree(chrom_interval_to_gene[chrom].keys())

    return {
        "gene_chrom_name_pos": gene_chrom_name_pos,
        "gene_to_chrom": gene_to_chrom,
        "gene_to_pos": gene_to_pos,
        "chrom_to_interval_tree": chrom_to_interval_tree,
    }


def compute_huge_variant_thresholds(domain, min_var_posterior, gwas_high_p_posterior, allelic_var_k, gwas_prior_odds):
    var_z_threshold = None
    var_p_threshold = None

    if min_var_posterior is not None and min_var_posterior > gwas_high_p_posterior:
        log_bf_threshold = domain.np.log(min_var_posterior / (1 - min_var_posterior)) - domain.np.log(gwas_prior_odds) + domain.np.log(domain.np.sqrt(1 + allelic_var_k))
        if log_bf_threshold > 0:
            var_z_threshold = domain.np.sqrt(2 * (1 + allelic_var_k) * log_bf_threshold / allelic_var_k)
            var_p_threshold = 2 * domain.scipy.stats.norm.cdf(-domain.np.abs(var_z_threshold))
            domain.log("Keeping only variants with p < %.4g" % var_p_threshold)
    else:
        var_p_threshold = gwas_high_p_posterior
        var_z_threshold = domain.np.abs(domain.scipy.stats.norm.ppf(var_p_threshold / 2))

    return (var_z_threshold, var_p_threshold)


def validate_and_normalize_huge_gwas_inputs(domain, gwas_in, gene_loc_file, credible_sets_in=None, credible_sets_chrom_col=None, credible_sets_pos_col=None, signal_window_size=250000, signal_min_sep=100000, signal_max_logp_ratio=None):
    if gwas_in is None:
        domain.bail("Require --gwas-in for this operation")
    if gene_loc_file is None:
        domain.bail("Require --gene-loc-file for this operation")

    if credible_sets_in is not None:
        if credible_sets_chrom_col is None or credible_sets_pos_col is None:
            domain.bail("Need --credible-set-chrom-col and --credible-set-pos-col")

    if signal_window_size < 2 * signal_min_sep:
        signal_window_size = 2 * signal_min_sep

    if signal_max_logp_ratio is not None and signal_max_logp_ratio > 1:
        domain.warn("Thresholding --signal-max-logp-ratio at 1")
        signal_max_logp_ratio = 1

    return (signal_window_size, signal_max_logp_ratio)


def write_huge_statistics_bundle(domain, runtime_state, prefix, gene_bf, extra_genes, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression):
    runtime_state = domain.pegs_coerce_runtime_state_dict(runtime_state, bail_fn=domain.bail)
    paths = domain.pegs_get_huge_statistics_paths_for_prefix(prefix)

    cache_genes = list(runtime_state["genes"]) if runtime_state.get("genes") is not None else []
    extra_genes = list(extra_genes)

    huge_signal_bfs = runtime_state.get("huge_signal_bfs")
    if huge_signal_bfs is None:
        huge_signal_bfs = domain.sparse.csc_matrix((0, 0))
    huge_signal_bfs_for_regression = runtime_state.get("huge_signal_bfs_for_regression")
    if huge_signal_bfs_for_regression is None:
        huge_signal_bfs_for_regression = domain.sparse.csc_matrix((0, 0))

    matrix_row_genes = domain.pegs_build_huge_statistics_matrix_row_genes(
        cache_genes,
        extra_genes,
        huge_signal_bfs.shape[0],
        bail_fn=domain.bail,
    )
    (gene_to_score, gene_to_score_uncorrected, gene_to_score_for_regression) = domain.pegs_build_huge_statistics_score_maps(
        runtime_state,
        cache_genes,
        extra_genes,
        gene_bf,
        extra_gene_bf,
        gene_bf_for_regression,
        extra_gene_bf_for_regression,
    )
    meta = domain.pegs_build_huge_statistics_meta(
        runtime_state,
        huge_signal_bfs,
        huge_signal_bfs_for_regression,
        json_safe_fn=domain._json_safe,
    )
    with domain.open_gz(paths["meta"], 'w') as out_fh:
        domain.json.dump(meta, out_fh, sort_keys=True)
        out_fh.write("\n")

    domain.pegs_write_huge_statistics_text_tables(
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
        open_text_fn=domain.open_gz,
        bail_fn=domain.bail,
    )
    domain.pegs_write_huge_statistics_runtime_vectors(
        paths,
        runtime_state,
        write_vector_fn=lambda out_file, values, value_type=float: domain.pegs_write_numeric_vector_file(
            out_file,
            values,
            open_text_fn=domain.open_gz,
            value_type=value_type,
        ),
    )
    domain.pegs_write_huge_statistics_sparse_components(
        paths,
        huge_signal_bfs,
        huge_signal_bfs_for_regression,
        write_vector_fn=lambda out_file, values, value_type=float: domain.pegs_write_numeric_vector_file(
            out_file,
            values,
            open_text_fn=domain.open_gz,
            value_type=value_type,
        ),
    )


def read_huge_statistics_bundle(domain, runtime_state, prefix):
    runtime_state = domain.pegs_coerce_runtime_state_dict(runtime_state, bail_fn=domain.bail)
    paths = domain.pegs_get_huge_statistics_paths_for_prefix(prefix)
    if not domain.os.path.exists(paths["meta"]):
        domain.bail("Could not find HuGE statistics cache file %s" % paths["meta"])

    with domain.open_gz(paths["meta"]) as in_fh:
        meta = domain.json.load(in_fh)

    if meta.get("recorded_params") is not None:
        runtime_state["params"] = meta["recorded_params"]
    if meta.get("recorded_param_keys") is not None:
        runtime_state["param_keys"] = list(meta["recorded_param_keys"])

    (
        cache_genes,
        extra_genes,
        extra_gene_bf,
        extra_gene_bf_for_regression,
        matrix_row_genes,
        gene_to_score,
        gene_to_score_uncorrected,
        gene_to_score_for_regression,
    ) = domain.pegs_read_huge_statistics_text_tables(paths, open_text_fn=domain.open_gz)

    (gene_bf, gene_bf_for_regression) = domain.pegs_resolve_huge_statistics_gene_vectors(
        runtime_state,
        cache_genes,
        extra_genes,
        matrix_row_genes,
        gene_to_score,
        gene_to_score_for_regression,
        bail_fn=domain.bail,
    )

    domain.pegs_load_huge_statistics_sparse_and_vectors(
        runtime_state,
        paths,
        meta,
        read_vector_fn=lambda in_file, value_type=float: domain.pegs_read_numeric_vector_file(
            in_file,
            open_text_fn=domain.open_gz,
            value_type=value_type,
        ),
    )
    domain.pegs_apply_huge_statistics_meta_to_runtime(runtime_state, meta)
    domain.pegs_read_huge_statistics_covariates_if_present(
        runtime_state,
        paths,
        open_text_fn=domain.open_gz,
        exists_fn=domain.os.path.exists,
    )

    runtime_state["gene_to_gwas_huge_score"] = gene_to_score
    runtime_state["gene_to_gwas_huge_score_uncorrected"] = gene_to_score_uncorrected
    domain.pegs_combine_runtime_huge_scores(runtime_state)
    domain.pegs_validate_huge_statistics_loaded_shapes(runtime_state, matrix_row_genes, bail_fn=domain.bail)

    return (
        gene_bf,
        extra_genes,
        domain.np.array(extra_gene_bf),
        gene_bf_for_regression,
        domain.np.array(extra_gene_bf_for_regression),
    )
