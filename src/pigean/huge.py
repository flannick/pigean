from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
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


def read_huge_s2g_probabilities(
    state,
    s2g_in,
    seen_chrom_pos,
    hold_out_chrom=None,
    s2g_chrom_col=None,
    s2g_pos_col=None,
    s2g_gene_col=None,
    s2g_prob_col=None,
    s2g_normalize_values=None,
    *,
    determine_columns_fn,
    open_text_fn,
    get_col_fn,
    clean_chrom_fn,
    log_fn,
    warn_fn,
    bail_fn,
    info_level,
):
    if s2g_in is None:
        return None

    chrom_pos_to_gene_prob = {}
    log_fn("Reading --s2g-in file %s" % s2g_in, info_level)

    if s2g_pos_col is None or s2g_chrom_col is None or s2g_gene_col is None:
        (
            possible_s2g_gene_cols,
            _possible_s2g_var_id_cols,
            possible_s2g_chrom_cols,
            possible_s2g_pos_cols,
            _possible_s2g_locus_cols,
            _possible_s2g_p_cols,
            _possible_s2g_beta_cols,
            _possible_s2g_se_cols,
            _possible_s2g_freq_cols,
            _possible_s2g_n_cols,
        ) = determine_columns_fn(s2g_in)

        if s2g_pos_col is None:
            if len(possible_s2g_pos_cols) == 1:
                s2g_pos_col = possible_s2g_pos_cols[0]
                log_fn("Using %s for position column; change with --s2g-pos-col if incorrect" % s2g_pos_col)
            else:
                bail_fn("Could not determine position column; specify with --s2g-pos-col")
        if s2g_chrom_col is None:
            if len(possible_s2g_chrom_cols) == 1:
                s2g_chrom_col = possible_s2g_chrom_cols[0]
                log_fn("Using %s for chromition column; change with --s2g-chrom-col if incorrect" % s2g_chrom_col)
            else:
                bail_fn("Could not determine chrom column; specify with --s2g-chrom-col")
        if s2g_gene_col is None:
            if len(possible_s2g_gene_cols) == 1:
                s2g_gene_col = possible_s2g_gene_cols[0]
                log_fn("Using %s for geneition column; change with --s2g-gene-col if incorrect" % s2g_gene_col)
            else:
                bail_fn("Could not determine gene column; specify with --s2g-gene-col")

    with open_text_fn(s2g_in) as s2g_fh:
        header_cols = s2g_fh.readline().strip("\n").split()
        chrom_col = get_col_fn(s2g_chrom_col, header_cols)
        pos_col = get_col_fn(s2g_pos_col, header_cols)
        gene_col = get_col_fn(s2g_gene_col, header_cols)
        prob_col = None
        if s2g_prob_col is not None:
            prob_col = get_col_fn(s2g_prob_col, header_cols)

        for line in s2g_fh:
            cols = line.strip("\n").split()
            if chrom_col > len(cols) or pos_col > len(cols) or gene_col > len(cols) or (prob_col is not None and prob_col > len(cols)):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            chrom = clean_chrom_fn(cols[chrom_col])
            if hold_out_chrom is not None and chrom == hold_out_chrom:
                continue

            try:
                pos = int(cols[pos_col])
            except ValueError:
                warn_fn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                continue
            gene = cols[gene_col]

            if state.gene_label_map is not None and gene in state.gene_label_map:
                gene = state.gene_label_map[gene]

            max_s2g_prob = 0.95
            prob = max_s2g_prob
            if prob_col is not None:
                try:
                    prob = float(cols[prob_col])
                except ValueError:
                    warn_fn("Skipping unconvertible prob value %s" % (cols[prob_col]))
                    continue
            if prob > max_s2g_prob:
                prob = max_s2g_prob

            if chrom in seen_chrom_pos and pos in seen_chrom_pos[chrom]:
                if chrom not in chrom_pos_to_gene_prob:
                    chrom_pos_to_gene_prob[chrom] = {}
                if pos not in chrom_pos_to_gene_prob[chrom]:
                    chrom_pos_to_gene_prob[chrom][pos] = []
                chrom_pos_to_gene_prob[chrom][pos].append((gene, prob))

        if s2g_normalize_values is not None:
            for chrom in chrom_pos_to_gene_prob:
                for pos in chrom_pos_to_gene_prob[chrom]:
                    prob_sum = sum([x[1] for x in chrom_pos_to_gene_prob[chrom][pos]])
                    if prob_sum > 0:
                        norm_factor = s2g_normalize_values / prob_sum
                        chrom_pos_to_gene_prob[chrom][pos] = [(x[0], x[1] * norm_factor) for x in chrom_pos_to_gene_prob[chrom][pos]]

    return chrom_pos_to_gene_prob


def read_huge_input_credible_sets(
    state,
    credible_sets_in,
    seen_chrom_pos,
    chrom_pos_p_beta_se_freq,
    var_p_threshold,
    hold_out_chrom=None,
    credible_sets_id_col=None,
    credible_sets_chrom_col=None,
    credible_sets_pos_col=None,
    credible_sets_ppa_col=None,
    *,
    determine_columns_fn,
    open_text_fn,
    get_col_fn,
    clean_chrom_fn,
    log_fn,
    warn_fn,
    bail_fn,
    info_level,
):
    added_chrom_pos = {}
    input_credible_set_info = {}
    if credible_sets_in is None:
        return (added_chrom_pos, input_credible_set_info)

    log_fn("Reading --credible-sets-in file %s" % credible_sets_in, info_level)

    if credible_sets_pos_col is None or credible_sets_chrom_col is None:
        (_, _, possible_credible_sets_chrom_cols, possible_credible_sets_pos_cols, _, _, _, _, _, _, _header) = determine_columns_fn(credible_sets_in)

        if credible_sets_pos_col is None:
            if len(possible_credible_sets_pos_cols) == 1:
                credible_sets_pos_col = possible_credible_sets_pos_cols[0]
                log_fn("Using %s for position column; change with --credible-sets-pos-col if incorrect" % credible_sets_pos_col)
            else:
                bail_fn("Could not determine position column; specify with --credible-sets-pos-col")
        if credible_sets_chrom_col is None:
            if len(possible_credible_sets_chrom_cols) == 1:
                credible_sets_chrom_col = possible_credible_sets_chrom_cols[0]
                log_fn("Using %s for chromition column; change with --credible-sets-chrom-col if incorrect" % credible_sets_chrom_col)
            else:
                bail_fn("Could not determine chrom column; specify with --credible-sets-chrom-col")

    with open_text_fn(credible_sets_in) as credible_sets_fh:
        header_cols = credible_sets_fh.readline().strip("\n").split()
        chrom_col = get_col_fn(credible_sets_chrom_col, header_cols)
        pos_col = get_col_fn(credible_sets_pos_col, header_cols)
        id_col = None
        if credible_sets_id_col is not None:
            id_col = get_col_fn(credible_sets_id_col, header_cols)
        ppa_col = None
        if credible_sets_ppa_col is not None:
            ppa_col = get_col_fn(credible_sets_ppa_col, header_cols)

        for line in credible_sets_fh:
            cols = line.strip("\n").split()
            if (id_col is not None and id_col > len(cols)) or (chrom_col is not None and chrom_col > len(cols)) or (pos_col is not None and pos_col > len(cols)) or (ppa_col is not None and ppa_col > len(cols)):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            chrom = clean_chrom_fn(cols[chrom_col])
            if hold_out_chrom is not None and chrom == hold_out_chrom:
                continue

            try:
                pos = int(cols[pos_col])
            except ValueError:
                warn_fn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                continue

            if id_col is not None:
                cs_id = cols[id_col]
            else:
                cs_id = "%s:%s" % (chrom, pos)

            ppa = None
            if ppa_col is not None:
                try:
                    ppa = float(cols[ppa_col])
                    if ppa > 1:
                        ppa = 0.99
                    elif ppa < 0:
                        ppa = 0
                except ValueError:
                    warn_fn("Skipping unconvertible ppa value %s" % (cols[ppa_col]))
                    continue

            if chrom in seen_chrom_pos:
                if pos not in seen_chrom_pos[chrom]:
                    assert(var_p_threshold is not None)
                    (p, beta, se, freq) = (var_p_threshold, 1, None, None)
                    chrom_pos_p_beta_se_freq[chrom].append((pos, p, beta, se, freq))
                    seen_chrom_pos[chrom].add(pos)
                    if chrom not in added_chrom_pos:
                        added_chrom_pos[chrom] = set()
                    added_chrom_pos[chrom].add(pos)

                if chrom not in input_credible_set_info:
                    input_credible_set_info[chrom] = {}
                if cs_id not in input_credible_set_info[chrom]:
                    input_credible_set_info[chrom][cs_id] = []
                input_credible_set_info[chrom][cs_id].append((pos, ppa))

    return (added_chrom_pos, input_credible_set_info)


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


def initialize_huge_gwas_state(domain):
    domain.huge_signals = []
    domain.huge_signal_posteriors = []
    domain.huge_signal_posteriors_for_regression = []
    domain.huge_signal_sum_gene_cond_probabilities = []
    domain.huge_signal_sum_gene_cond_probabilities_for_regression = []
    domain.huge_signal_mean_gene_pos = []
    domain.huge_signal_mean_gene_pos_for_regression = []
    domain.gene_covariates = None
    domain.gene_covariates_mask = None
    domain.gene_covariate_names = None
    domain.gene_covariate_directions = None
    domain.gene_covariate_intercept_index = None
    domain.gene_covariate_adjustments = None

    return {
        "closest_dist_X": np.array([]),
        "closest_dist_Y": np.array([]),
        "var_all_p": np.array([]),
        "gene_bf_data": [],
        "gene_bf_data_detect": [],
        "gene_prob_rows": [],
        "gene_prob_rows_detect": [],
        "gene_prob_cols": [],
        "gene_prob_cols_detect": [],
        "gene_prob_genes": [],
        "gene_prob_col_num": 0,
        "gene_covariate_genes": [],
    }


def remap_huge_gene_probability_rows(domain, gene_to_chrom, gene_prob_genes, gene_prob_rows, gene_prob_rows_detect, *, construct_map_to_ind_fn):
    if domain.genes is not None:
        genes = domain.genes
        gene_to_ind = domain.gene_to_ind
    else:
        genes = list(gene_to_chrom.keys())
        gene_to_ind = construct_map_to_ind_fn(genes)

    extra_genes = []
    extra_gene_to_ind = {}
    for gene_prob_rows_to_process in [gene_prob_rows, gene_prob_rows_detect]:
        for i in range(len(gene_prob_rows_to_process)):
            cur_gene = gene_prob_genes[gene_prob_rows_to_process[i]]

            if cur_gene in gene_to_ind:
                new_ind = gene_to_ind[cur_gene]
            elif cur_gene in extra_gene_to_ind:
                new_ind = extra_gene_to_ind[cur_gene]
            else:
                new_ind = len(extra_genes) + len(genes)
                extra_genes.append(cur_gene)
                extra_gene_to_ind[cur_gene] = new_ind
            gene_prob_rows_to_process[i] = new_ind

    for cur_gene in list(gene_to_chrom.keys()) + gene_prob_genes:
        if cur_gene not in gene_to_ind and cur_gene not in extra_gene_to_ind:
            new_ind = len(extra_genes) + len(genes)
            extra_genes.append(cur_gene)
            extra_gene_to_ind[cur_gene] = new_ind

    gene_prob_gene_list = genes + extra_genes
    return (genes, gene_to_ind, extra_genes, extra_gene_to_ind, gene_prob_gene_list)


def align_huge_gene_covariates_to_gene_list(domain, gene_prob_gene_list, gene_covariate_genes, gene_to_ind, extra_gene_to_ind):
    if domain.gene_covariates is None:
        return

    sorted_gene_covariates = np.tile(
        np.nanmean(domain.gene_covariates, axis=0),
        len(gene_prob_gene_list),
    ).reshape((len(gene_prob_gene_list), domain.gene_covariates.shape[1]))

    for i in range(len(gene_covariate_genes)):
        cur_gene = gene_covariate_genes[i]
        assert cur_gene in gene_to_ind or cur_gene in extra_gene_to_ind

        if cur_gene in gene_to_ind:
            new_ind = gene_to_ind[cur_gene]
        else:
            new_ind = extra_gene_to_ind[cur_gene]
        noninf_mask = ~np.isnan(domain.gene_covariates[i, :])
        sorted_gene_covariates[new_ind, noninf_mask] = domain.gene_covariates[i, noninf_mask]

    domain.gene_covariates = sorted_gene_covariates


def compute_huge_variant_logbf_and_posteriors(
    domain,
    var_z,
    allelic_var_k,
    gwas_prior_odds,
    *,
    separate_detect=False,
    allelic_var_k_detect=None,
    gwas_prior_odds_detect=None,
):
    var_log_bf = -np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k)

    if separate_detect:
        var_log_bf_detect = -np.log(np.sqrt(1 + allelic_var_k_detect)) + 0.5 * np.square(var_z) * allelic_var_k_detect / (1 + allelic_var_k_detect)
    else:
        var_log_bf_detect = var_log_bf.copy()

    var_posterior = var_log_bf + np.log(gwas_prior_odds)
    if separate_detect:
        var_posterior_detect = var_log_bf_detect + np.log(gwas_prior_odds_detect)
        update_posterior = [var_posterior, var_posterior_detect]
    else:
        var_posterior_detect = var_posterior.copy()
        update_posterior = [var_posterior]

    max_log = 15
    for cur_var_posterior in update_posterior:
        max_mask = cur_var_posterior < max_log
        cur_var_posterior[~max_mask] = 1
        cur_var_posterior[max_mask] = np.exp(cur_var_posterior[max_mask])
        cur_var_posterior[max_mask] = cur_var_posterior[max_mask] / (1 + cur_var_posterior[max_mask])

    if not separate_detect:
        var_posterior_detect = var_posterior.copy()

    return (var_log_bf, var_log_bf_detect, var_posterior, var_posterior_detect)


def filter_huge_variants_for_signal_search(
    domain,
    var_pos,
    var_p,
    var_beta,
    var_se,
    var_se2,
    var_log_bf,
    var_log_bf_detect,
    var_posterior,
    var_posterior_detect,
    vars_zipped,
    *,
    freq_col,
    min_n_ratio,
    mean_n,
    learn_params,
    chrom,
    added_chrom_pos,
):
    variants_keep = np.full(len(var_pos), True)
    qc_fail = 1 / var_se2 < min_n_ratio * mean_n
    variants_keep[qc_fail] = False

    if not learn_params and chrom in added_chrom_pos:
        for cur_pos in added_chrom_pos[chrom]:
            variants_keep[var_pos == cur_pos] = True

    var_pos = var_pos[variants_keep]
    var_p = var_p[variants_keep]
    var_beta = var_beta[variants_keep]
    var_se = var_se[variants_keep]
    var_se2 = var_se2[variants_keep]
    var_log_bf = var_log_bf[variants_keep]
    var_log_bf_detect = var_log_bf_detect[variants_keep]
    var_posterior = var_posterior[variants_keep]
    var_posterior_detect = var_posterior_detect[variants_keep]

    var_logp = -np.log(var_p) / np.log(10)

    var_freq = None
    if freq_col is not None:
        var_freq = np.array(vars_zipped[4], dtype=float)[variants_keep]
        var_freq[var_freq > 0.5] = 1 - var_freq[var_freq > 0.5]

    return (
        var_pos,
        var_p,
        var_beta,
        var_se,
        var_se2,
        var_log_bf,
        var_log_bf_detect,
        var_posterior,
        var_posterior_detect,
        var_logp,
        var_freq,
    )
