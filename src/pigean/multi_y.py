from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass

import numpy as np

from pegs_shared.io_common import detect_table_delimiter, open_text_with_retry, resolve_column_index, split_table_line

from . import main_support as pigean_main_support
from . import phewas as pigean_phewas


_MULTI_Y_PHENO_CANDIDATES = ("Trait", "Pheno")
_MULTI_Y_LOG_BF_CANDIDATES = ("log_bf", "Direct")
_MULTI_Y_COMBINED_CANDIDATES = ("combined", "Combined")
_MULTI_Y_PRIOR_CANDIDATES = ("prior", "Prior")


@dataclass(frozen=True)
class MultiYColumnResolution:
    id_col_name: str
    pheno_col_name: str
    log_bf_col_name: str
    combined_col_name: str | None
    prior_col_name: str | None


@dataclass
class MultiYPipelineResult:
    state: object
    mode_state: dict
    sigma2_cond: object
    y_not_loaded: bool
    num_traits_total: int = 0
    num_traits_completed: int = 0
    phenos_per_batch: int = 0


class _AggregatedTraitTableWriter:
    def __init__(self, output_path: str, key_column: str):
        self.output_path = output_path
        self.key_column = key_column
        self._header = None
        self._fh = None
        self._insert_index = None

    def _resolve_insert_index(self, header_cols: list[str]) -> int:
        if self.key_column == "Gene_Set" and len(header_cols) > 1 and header_cols[1] == "label":
            return 2
        return 1

    def append_from(self, trait: str, source_path: str) -> int:
        rows_written = 0
        if source_path is None or not os.path.exists(source_path):
            return rows_written
        with open_text_with_retry(source_path) as source_fh:
            header_line = source_fh.readline().strip("\n")
            if not header_line:
                return rows_written
            header_cols = header_line.split("\t")
            if self._header is None:
                self._insert_index = self._resolve_insert_index(header_cols)
                self._header = list(header_cols)
                self._header.insert(self._insert_index, "trait")
                self._fh = open_text_with_retry(self.output_path, "w")
                self._fh.write("%s\n" % "\t".join(self._header))
            elif header_cols != [col for i, col in enumerate(self._header) if i != self._insert_index]:
                raise ValueError(
                    "Trait-level output schema mismatch while aggregating %s into %s"
                    % (source_path, self.output_path)
                )

            for line in source_fh:
                line = line.strip("\n")
                if not line:
                    continue
                cols = line.split("\t")
                cols.insert(self._insert_index, trait)
                self._fh.write("%s\n" % "\t".join(cols))
                rows_written += 1
        return rows_written

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _first_present_column(header_cols, explicit_name, fallback_names, *, required):
    if explicit_name is not None:
        return explicit_name
    for candidate in fallback_names:
        if candidate in header_cols:
            return candidate
    if required:
        raise ValueError(
            "Could not resolve a required column from header candidates %s"
            % (", ".join(fallback_names))
        )
    return None


def _resolve_multi_y_columns(options):
    delimiter = detect_table_delimiter(options.multi_y_in, open_text_fn=open_text_with_retry)
    with open_text_with_retry(options.multi_y_in) as fh:
        header_cols = split_table_line(fh.readline(), delimiter)

    id_col_name = options.multi_y_id_col if options.multi_y_id_col is not None else "Gene"
    resolve_column_index(id_col_name, header_cols)
    pheno_col_name = _first_present_column(
        header_cols,
        options.multi_y_pheno_col,
        _MULTI_Y_PHENO_CANDIDATES,
        required=True,
    )
    log_bf_col_name = _first_present_column(
        header_cols,
        options.multi_y_log_bf_col,
        _MULTI_Y_LOG_BF_CANDIDATES,
        required=True,
    )
    combined_col_name = _first_present_column(
        header_cols,
        options.multi_y_combined_col,
        _MULTI_Y_COMBINED_CANDIDATES,
        required=False,
    )
    prior_col_name = _first_present_column(
        header_cols,
        options.multi_y_prior_col,
        _MULTI_Y_PRIOR_CANDIDATES,
        required=False,
    )
    return MultiYColumnResolution(
        id_col_name=id_col_name,
        pheno_col_name=pheno_col_name,
        log_bf_col_name=log_bf_col_name,
        combined_col_name=combined_col_name,
        prior_col_name=prior_col_name,
    )


def _estimate_phenos_per_batch(num_genes, num_value_cols, max_gb):
    target_bytes = max(int(max_gb * (1024**3) * 0.25), 16 * 1024 * 1024)
    bytes_per_trait = max(1, num_genes * max(1, num_value_cols) * 32)
    return max(1, int(target_bytes / bytes_per_trait))


def _load_trait_blacklist(path, *, warn_fn=None):
    if path is None:
        return set()
    if warn_fn is None:
        warn_fn = lambda _msg: None
    traits = set()
    with open_text_with_retry(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Accept one-ID-per-line files and simple tabular files. The first
            # column is the trait identifier by default.
            trait = line.split("\t", 1)[0].strip()
            if not trait:
                warn_fn("Skipping empty trait blacklist entry on line %d of %s" % (line_no, path))
                continue
            traits.add(trait)
    return traits


def _filter_blacklisted_phenos(phenos, blacklist):
    if not blacklist:
        return list(phenos), 0, len(blacklist)
    filtered = [pheno for pheno in phenos if pheno not in blacklist]
    matched = len(phenos) - len(filtered)
    missing = len(blacklist.difference(phenos))
    return filtered, matched, missing


def _clear_primary_y_inputs(options):
    options.gwas_in = None
    options.huge_statistics_in = None
    options.huge_statistics_out = None
    options.exomes_in = None
    options.positive_controls_in = None
    options.positive_controls_list = None
    options.positive_controls_all_in = None
    options.case_counts_in = None
    options.ctrl_counts_in = None
    options.gene_stats_prob_col = None
    options.run_phewas = False
    options.run_phewas_input = None
    options.run_phewas_legacy_input = None
    options.phewas_stats_out = None
    options.phewas_gene_set_stats_out = None


def _write_trait_gene_stats_file(
    output_path,
    genes,
    log_bf_values,
    *,
    combined_values=None,
    prior_values=None,
):
    with open_text_with_retry(output_path, "w") as fh:
        header_cols = ["Gene", "log_bf"]
        if combined_values is not None:
            header_cols.append("combined")
        if prior_values is not None:
            header_cols.append("prior")
        fh.write("%s\n" % "\t".join(header_cols))
        for i, gene in enumerate(genes):
            cols = [gene, "%.12g" % float(log_bf_values[i])]
            if combined_values is not None:
                cols.append("%.12g" % float(combined_values[i]))
            if prior_values is not None:
                cols.append("%.12g" % float(prior_values[i]))
            fh.write("%s\n" % "\t".join(cols))


def _select_multi_y_response_matrix(batch_Y, batch_combined, options, services):
    response = getattr(options, "multi_y_response_col", "combined")
    if response == "combined":
        if batch_combined is None:
            services.bail(
                "Option --multi-y-response-col combined requires a resolved combined column; "
                "provide --multi-y-combined-col or pass --multi-y-response-col log_bf"
            )
        return batch_combined
    if response == "log_bf":
        if batch_Y is None:
            services.bail(
                "Option --multi-y-response-col log_bf requires a resolved log-BF column; "
                "provide --multi-y-log-bf-col or pass --multi-y-response-col combined"
            )
        return batch_Y
    services.bail("Unsupported --multi-y-response-col value: %s" % response)


def _record_multi_y_params(
    state,
    options,
    mode,
    *,
    columns,
    num_traits_total,
    phenos_per_batch,
    num_traits_before_blacklist=None,
    trait_blacklist_requested=0,
    trait_blacklist_matched=0,
    trait_blacklist_missing=0,
):
    gene_universe_in = getattr(options, "gene_universe_in", None)
    if gene_universe_in is not None:
        gene_universe_mode = "file"
    else:
        gene_universe_mode = "x"
    state._record_params(
        {
            "multi_y_enabled": True,
            "multi_y_mode": mode,
            "multi_y_input": options.multi_y_in,
            "multi_y_id_col": columns.id_col_name,
            "multi_y_pheno_col": columns.pheno_col_name,
            "multi_y_log_bf_col": columns.log_bf_col_name,
            "multi_y_combined_col": columns.combined_col_name,
            "multi_y_prior_col": columns.prior_col_name,
            "multi_y_response_col": getattr(options, "multi_y_response_col", "combined"),
            "multi_y_trait_blacklist_in": getattr(options, "multi_y_trait_blacklist_in", None),
            "multi_y_trait_blacklist_requested": trait_blacklist_requested,
            "multi_y_trait_blacklist_matched": trait_blacklist_matched,
            "multi_y_trait_blacklist_missing": trait_blacklist_missing,
            "multi_y_num_traits_before_blacklist": num_traits_before_blacklist
            if num_traits_before_blacklist is not None
            else num_traits_total,
            "multi_y_num_traits": num_traits_total,
            "multi_y_phenos_per_batch": phenos_per_batch,
            "multi_y_vectorize_betas": bool(getattr(options, "multi_y_vectorize_betas", False)),
            "multi_y_gene_universe_mode": gene_universe_mode,
            "multi_y_gene_universe_in": gene_universe_in,
        },
        overwrite=True,
    )


def _initialize_multi_y_gene_universe(seed_state, options, services):
    if getattr(options, "gene_universe_from_y", False):
        services.bail(
            "Option --gene-universe-from-y is not supported with --multi-y-in; "
            "use --gene-universe-in for an explicit shared universe or omit it to use --gene-universe-from-x semantics"
        )
    gene_universe_in = getattr(options, "gene_universe_in", None)
    if gene_universe_in is None:
        return

    universe_genes = pigean_main_support.pigean_y_inputs_core.load_gene_ids_from_file(
        gene_universe_in,
        gene_ids_id_col=getattr(options, "gene_universe_id_col", None),
        gene_ids_has_header=getattr(options, "gene_universe_has_header", True),
        gene_label_map=getattr(seed_state, "gene_label_map", None),
        open_text_fn=pigean_main_support.open_gz,
        get_col_fn=pigean_main_support.get_col,
        log_fn=services.log,
        warn_fn=services.warn,
        bail_fn=services.bail,
    )
    pigean_main_support.pigean_y_inputs_core.initialize_explicit_gene_universe_if_needed(
        seed_state,
        gene_universe_mode="file",
        gene_universe_genes=universe_genes,
        log_fn=services.log,
    )


def _as_trait_gene_set_matrix(values, num_traits):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.shape[0] != num_traits:
        raise ValueError(
            "Expected %d trait rows in vectorized multi-Y matrix; got %d"
            % (num_traits, values.shape[0])
        )
    return values


def _clear_gene_set_result_vectors(state):
    for attr in (
        "beta_tildes",
        "p_values",
        "z_scores",
        "ses",
        "se_inflation_factors",
        "betas",
        "betas_uncorrected",
        "non_inf_avg_postps",
        "non_inf_avg_cond_betas",
    ):
        setattr(state, attr, None)


def _apply_vectorized_gene_set_filter(state, options, p_values_m, services):
    if options.filter_gene_set_p is None or options.filter_gene_set_p >= 1:
        return np.full(p_values_m.shape[1], True, dtype=bool)

    best_p = np.min(p_values_m, axis=0)
    keep_mask = np.any(p_values_m <= options.filter_gene_set_p, axis=0)
    if np.sum(keep_mask) == 0 and len(best_p) > 0:
        keep_mask[np.argmin(best_p)] = True

    max_num_gene_sets = getattr(options, "max_num_gene_sets", None)
    if max_num_gene_sets is not None and max_num_gene_sets > 0 and np.sum(keep_mask) > max_num_gene_sets:
        kept_indices = np.where(keep_mask)[0]
        ranked_kept = kept_indices[np.argsort(best_p[kept_indices])]
        capped_keep = np.zeros_like(keep_mask)
        capped_keep[ranked_kept[:max_num_gene_sets]] = True
        keep_mask = capped_keep

    services.log(
        "Keeping %d gene sets that passed the vectorized multi-Y union p threshold of p<%.3g"
        % (int(np.sum(keep_mask)), options.filter_gene_set_p),
        services.INFO,
    )
    return keep_mask


def _set_trait_gene_set_results(
    state,
    trait_index,
    *,
    beta_tildes_m,
    ses_m,
    z_scores_m,
    p_values_m,
    se_inflation_factors_m,
    betas_m,
    betas_uncorrected_m,
    postp_m,
):
    state.beta_tildes = np.asarray(beta_tildes_m[trait_index, :]).copy()
    state.ses = np.asarray(ses_m[trait_index, :]).copy()
    state.z_scores = np.asarray(z_scores_m[trait_index, :]).copy()
    state.p_values = np.asarray(p_values_m[trait_index, :]).copy()
    if se_inflation_factors_m is None:
        state.se_inflation_factors = None
    else:
        state.se_inflation_factors = np.asarray(se_inflation_factors_m[trait_index, :]).copy()
    state.betas = np.asarray(betas_m[trait_index, :]).copy()
    state.betas_uncorrected = np.asarray(betas_uncorrected_m[trait_index, :]).copy()
    state.non_inf_avg_postps = np.asarray(postp_m[trait_index, :]).copy()
    state.non_inf_avg_cond_betas = state.betas.copy()
    positive_postp = state.non_inf_avg_postps > 0
    state.non_inf_avg_cond_betas[positive_postp] /= state.non_inf_avg_postps[positive_postp]


def _run_multi_y_vectorized_betas(
    *,
    services,
    options,
    seed_state,
    mode_state,
    sigma2_cond,
    columns,
    phenos,
    pheno_to_ind,
    col_info,
    phenos_per_batch,
):
    if getattr(options, "use_sampling_for_betas", None) not in (None, 0):
        services.bail("Option --multi-y-vectorize-betas does not yet support --use-sampling-for-betas")
    if getattr(options, "independent_betas_only", False):
        services.bail("Option --multi-y-vectorize-betas does not yet support --independent-betas-only")
    if getattr(options, "filter_negative", False):
        services.bail(
            "Option --multi-y-vectorize-betas requires --no-filter-negative because negative beta-tilde filtering is trait-specific and occurs during X read in the unvectorized workflow"
        )
    if (
        getattr(options, "prune_gene_sets", None) is not None
        and options.prune_gene_sets <= 1
    ) or (
        getattr(options, "weighted_prune_gene_sets", None) is not None
        and options.weighted_prune_gene_sets <= 1
    ):
        services.bail(
            "Option --multi-y-vectorize-betas requires disabled gene-set pruning (--prune-gene-sets > 1 and --weighted-prune-gene-sets > 1) because pruning is performed during X read in the unvectorized workflow"
        )

    update_hyper = bool(getattr(options, "update_hyper_p", False) or getattr(options, "update_hyper_sigma", False))
    if update_hyper:
        services.warn(
            "In vectorized multi-Y betas mode, hyperparameter updates will be shared across all traits; "
            "use the default unvectorized multi-Y workflow for per-trait hyperparameter updates."
        )
    seed_state._record_params(
        {
            "multi_y_vectorized_hyper_updates_shared": update_hyper,
            "multi_y_vectorized_beta_parallel_axis": "traits",
        },
        overwrite=True,
    )

    services.log(
        "Running vectorized multi-Y betas workflow for %d traits from %s with batch_size=%d"
        % (len(phenos), options.multi_y_in, phenos_per_batch),
        services.INFO,
    )

    gene_set_writer = _AggregatedTraitTableWriter(options.gene_set_stats_out, key_column="Gene_Set")
    num_traits_completed = 0
    common_sampler_kwargs = pigean_main_support.build_inner_beta_sampler_common_kwargs(options)
    common_sampler_kwargs.update(
        {
            "max_allowed_batch_correlation": options.max_allowed_batch_correlation,
        }
    )

    try:
        with tempfile.TemporaryDirectory(prefix="pigean_multi_y_vectorized_") as tmpdir:
            for begin in range(0, len(phenos), phenos_per_batch):
                end = min(begin + phenos_per_batch, len(phenos))
                batch_traits = phenos[begin:end]
                services.log(
                    "Processing vectorized multi-Y batch %d-%d of %d"
                    % (begin + 1, end, len(phenos)),
                    services.INFO,
                )
                batch_state = copy.deepcopy(seed_state)
                (batch_Y, batch_combined, _batch_priors) = pigean_phewas.read_phewas_file_batch(
                    batch_state,
                    options.multi_y_in,
                    begin=begin,
                    cur_batch_size=end - begin,
                    pheno_to_ind=pheno_to_ind,
                    id_col=col_info["id_col"],
                    pheno_col=col_info["pheno_col"],
                    bf_col=col_info["bf_col"],
                    combined_col=col_info["combined_col"],
                    prior_col=col_info["prior_col"],
                    open_text_fn=open_text_with_retry,
                    warn_fn=services.warn,
                )
                batch_response = _select_multi_y_response_matrix(
                    batch_Y,
                    batch_combined,
                    options,
                    services,
                )

                batch_state.calculate_gene_set_statistics(
                    Y=batch_response.T,
                    max_gene_set_p=None,
                    run_logistic=not options.linear,
                    max_for_linear=options.max_for_linear,
                    run_corrected_ols=not options.ols,
                    use_sampling_for_betas=options.use_sampling_for_betas,
                    correct_betas_mean=options.correct_betas_mean,
                    correct_betas_var=options.correct_betas_var,
                    gene_loc_file=options.gene_loc_file,
                    gene_cor_file=options.gene_cor_file,
                    gene_cor_file_gene_col=options.gene_cor_file_gene_col,
                    gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col,
                    skip_V=True,
                )

                num_batch_traits = len(batch_traits)
                beta_tildes_m = _as_trait_gene_set_matrix(batch_state.beta_tildes, num_batch_traits)
                ses_m = _as_trait_gene_set_matrix(batch_state.ses, num_batch_traits)
                z_scores_m = _as_trait_gene_set_matrix(batch_state.z_scores, num_batch_traits)
                p_values_m = _as_trait_gene_set_matrix(batch_state.p_values, num_batch_traits)
                se_inflation_factors_m = None
                if batch_state.se_inflation_factors is not None:
                    se_inflation_factors_m = _as_trait_gene_set_matrix(
                        batch_state.se_inflation_factors,
                        num_batch_traits,
                    )

                keep_mask = _apply_vectorized_gene_set_filter(batch_state, options, p_values_m, services)
                if np.sum(keep_mask) == 0:
                    services.log("Skipping vectorized batch because no gene sets survived filtering", services.INFO)
                    continue

                _clear_gene_set_result_vectors(batch_state)
                batch_state.subset_gene_sets(
                    keep_mask,
                    keep_missing=not getattr(batch_state, "track_filtered_beta_uncorrected", False),
                    ignore_missing=getattr(batch_state, "track_filtered_beta_uncorrected", False),
                    skip_V=True,
                    filter_reason="max_gene_set_p",
                )

                beta_tildes_m = beta_tildes_m[:, keep_mask]
                ses_m = ses_m[:, keep_mask]
                z_scores_m = z_scores_m[:, keep_mask]
                p_values_m = p_values_m[:, keep_mask]
                if se_inflation_factors_m is not None:
                    se_inflation_factors_m = se_inflation_factors_m[:, keep_mask]

                trait_output_keep_m = np.ones(beta_tildes_m.shape, dtype=bool)
                if options.filter_gene_set_p is not None and options.filter_gene_set_p < 1:
                    trait_output_keep_m = np.logical_and(
                        trait_output_keep_m,
                        p_values_m <= options.filter_gene_set_p,
                    )
                if getattr(options, "filter_negative", False):
                    trait_output_keep_m = np.logical_and(trait_output_keep_m, beta_tildes_m >= 0)

                avg_betas_uncorrected_m, avg_postp_uncorrected_m = batch_state._calculate_non_inf_betas(
                    batch_state.p,
                    beta_tildes=beta_tildes_m,
                    ses=ses_m,
                    assume_independent=True,
                    V=None,
                    update_hyper_sigma=False,
                    update_hyper_p=False,
                    **common_sampler_kwargs,
                )
                avg_betas_uncorrected_m = _as_trait_gene_set_matrix(
                    avg_betas_uncorrected_m,
                    num_batch_traits,
                )
                avg_postp_uncorrected_m = _as_trait_gene_set_matrix(
                    avg_postp_uncorrected_m,
                    num_batch_traits,
                )
                initial_run_mask_m = np.logical_and(avg_betas_uncorrected_m != 0, trait_output_keep_m)
                run_mask = np.any(initial_run_mask_m, axis=0)
                if np.sum(run_mask) == 0 and p_values_m.shape[1] > 0:
                    run_mask[np.argmin(np.min(p_values_m, axis=0))] = True

                avg_betas_m = np.zeros_like(avg_betas_uncorrected_m)
                avg_postp_m = np.zeros_like(avg_postp_uncorrected_m)
                if np.sum(run_mask) > 0:
                    corrected_betas_m, corrected_postp_m = batch_state._calculate_non_inf_betas(
                        batch_state.p,
                        beta_tildes=beta_tildes_m[:, run_mask],
                        ses=ses_m[:, run_mask],
                        X_orig=batch_state.X_orig[:, run_mask],
                        scale_factors=batch_state.scale_factors[run_mask],
                        mean_shifts=batch_state.mean_shifts[run_mask],
                        V=None,
                        ps=batch_state.ps[run_mask] if batch_state.ps is not None else None,
                        sigma2s=batch_state.sigma2s[run_mask] if batch_state.sigma2s is not None else None,
                        is_dense_gene_set=batch_state.is_dense_gene_set[run_mask],
                        update_hyper_sigma=getattr(options, "update_hyper_sigma", False),
                        update_hyper_p=getattr(options, "update_hyper_p", False),
                        **common_sampler_kwargs,
                    )
                    corrected_betas_m = _as_trait_gene_set_matrix(corrected_betas_m, num_batch_traits)
                    corrected_postp_m = _as_trait_gene_set_matrix(corrected_postp_m, num_batch_traits)
                    avg_betas_m[:, run_mask] = corrected_betas_m
                    avg_postp_m[:, run_mask] = corrected_postp_m
                    avg_betas_m[~initial_run_mask_m] = 0
                    avg_postp_m[~initial_run_mask_m] = 0

                for batch_offset, trait in enumerate(batch_traits):
                    trait_safe = trait.replace("/", "_").replace(" ", "_")
                    trait_gene_set_stats_out = os.path.join(
                        tmpdir,
                        "%06d_%s.gene_set_stats.out" % (begin + batch_offset, trait_safe),
                    )
                    trait_state = copy.deepcopy(batch_state)
                    _set_trait_gene_set_results(
                        trait_state,
                        batch_offset,
                        beta_tildes_m=beta_tildes_m,
                        ses_m=ses_m,
                        z_scores_m=z_scores_m,
                        p_values_m=p_values_m,
                        se_inflation_factors_m=se_inflation_factors_m,
                        betas_m=avg_betas_m,
                        betas_uncorrected_m=avg_betas_uncorrected_m,
                        postp_m=avg_postp_m,
                    )
                    trait_keep_mask = trait_output_keep_m[batch_offset, :]
                    if np.sum(~trait_keep_mask) > 0:
                        removed_negative = np.logical_and(~trait_keep_mask, beta_tildes_m[batch_offset, :] < 0)
                        filter_reason = "prefilter_negative_beta" if np.any(removed_negative) else "max_gene_set_p"
                        trait_state.subset_gene_sets(
                            trait_keep_mask,
                            keep_missing=True,
                            ignore_missing=False,
                            skip_V=True,
                            filter_reason=filter_reason,
                        )
                    trait_state.write_gene_set_statistics(
                        trait_gene_set_stats_out,
                        max_no_write_gene_set_beta=options.max_no_write_gene_set_beta,
                        max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected,
                        output_detail=options.output_detail,
                    )
                    rows_written = gene_set_writer.append_from(trait, trait_gene_set_stats_out)
                    if rows_written == 0:
                        services.log("Trait %s produced no gene-set rows after write filters" % trait, services.INFO)
                    num_traits_completed += 1
    finally:
        gene_set_writer.close()

    if options.params_out is not None:
        seed_state._record_params({"multi_y_num_traits_completed": num_traits_completed}, overwrite=True)
        seed_state.write_params(options.params_out)

    return MultiYPipelineResult(
        state=seed_state,
        mode_state=mode_state,
        sigma2_cond=sigma2_cond,
        y_not_loaded=False,
        num_traits_total=len(phenos),
        num_traits_completed=num_traits_completed,
        phenos_per_batch=phenos_per_batch,
    )


def run_multi_y_pipeline(services, options, mode):
    if mode not in {"betas", "gibbs"}:
        services.bail("Option --multi-y-in is only supported for modes betas and gibbs")
    if getattr(options, "multi_y_vectorize_betas", False) and mode != "betas":
        services.bail("Option --multi-y-vectorize-betas is only supported in betas mode")
    if options.gene_set_stats_out is None:
        services.bail("Option --multi-y-in requires --gene-set-stats-out")

    columns = _resolve_multi_y_columns(options)
    if getattr(options, "multi_y_response_col", "combined") == "combined" and columns.combined_col_name is None:
        services.bail(
            "Option --multi-y-response-col combined requires a combined column; "
            "provide --multi-y-combined-col or pass --multi-y-response-col log_bf"
        )
    seed_state = pigean_main_support.build_runtime_state(options)
    mode_state = pigean_main_support.build_mode_state(mode, False)
    sigma2_cond = pigean_main_support.configure_hyperparameters_for_main(seed_state, options)
    _initialize_multi_y_gene_universe(seed_state, options, services)
    pigean_main_support.run_main_adaptive_read_x(seed_state, options, mode_state, sigma2_cond)

    if not seed_state.has_gene_sets():
        services.log("No gene sets survived the input filters; stopping")
        services.sys.exit(0)

    phenos, pheno_to_ind, col_info = pigean_phewas.prepare_phewas_phenos_from_file(
        seed_state,
        options.multi_y_in,
        gene_phewas_bfs_id_col=columns.id_col_name,
        gene_phewas_bfs_pheno_col=columns.pheno_col_name,
        gene_phewas_bfs_log_bf_col=columns.log_bf_col_name,
        gene_phewas_bfs_combined_col=columns.combined_col_name,
        gene_phewas_bfs_prior_col=columns.prior_col_name,
        open_text_fn=open_text_with_retry,
        get_col_fn=resolve_column_index,
        construct_map_to_ind_fn=pigean_main_support.pegs_construct_map_to_ind,
        warn_fn=services.warn,
        log_fn=services.log,
        debug_level=services.DEBUG,
    )
    if len(phenos) == 0:
        services.bail("No phenotypes were found in --multi-y-in")

    num_traits_before_blacklist = len(phenos)
    trait_blacklist = _load_trait_blacklist(
        getattr(options, "multi_y_trait_blacklist_in", None),
        warn_fn=services.warn,
    )
    phenos, blacklist_matched, blacklist_missing = _filter_blacklisted_phenos(phenos, trait_blacklist)
    pheno_to_ind = pigean_main_support.pegs_construct_map_to_ind(phenos)
    seed_state.phenos = phenos
    seed_state.pheno_to_ind = pheno_to_ind
    if trait_blacklist:
        services.log(
            "Filtered %d of %d multi-Y traits using blacklist %s; %d requested blacklist traits were not present"
            % (
                blacklist_matched,
                num_traits_before_blacklist,
                options.multi_y_trait_blacklist_in,
                blacklist_missing,
            ),
            services.INFO,
        )
    if len(phenos) == 0:
        services.bail("All phenotypes from --multi-y-in were removed by --multi-y-trait-blacklist-in")

    num_value_cols = 1 + int(columns.combined_col_name is not None) + int(columns.prior_col_name is not None)
    phenos_per_batch = options.multi_y_max_phenos_per_batch
    if phenos_per_batch is None:
        phenos_per_batch = _estimate_phenos_per_batch(len(seed_state.genes), num_value_cols, options.max_gb)
    phenos_per_batch = max(1, min(int(phenos_per_batch), len(phenos)))

    _record_multi_y_params(
        seed_state,
        options,
        mode,
        columns=columns,
        num_traits_total=len(phenos),
        phenos_per_batch=phenos_per_batch,
        num_traits_before_blacklist=num_traits_before_blacklist,
        trait_blacklist_requested=len(trait_blacklist),
        trait_blacklist_matched=blacklist_matched,
        trait_blacklist_missing=blacklist_missing,
    )

    if getattr(options, "multi_y_vectorize_betas", False):
        return _run_multi_y_vectorized_betas(
            services=services,
            options=options,
            seed_state=seed_state,
            mode_state=mode_state,
            sigma2_cond=sigma2_cond,
            columns=columns,
            phenos=phenos,
            pheno_to_ind=pheno_to_ind,
            col_info=col_info,
            phenos_per_batch=phenos_per_batch,
        )

    services.log(
        "Running native multi-Y %s workflow for %d traits from %s with batch_size=%d"
        % (mode, len(phenos), options.multi_y_in, phenos_per_batch),
        services.INFO,
    )

    gene_set_writer = _AggregatedTraitTableWriter(options.gene_set_stats_out, key_column="Gene_Set")
    gene_writer = None
    if mode == "gibbs" and options.gene_stats_out is not None:
        gene_writer = _AggregatedTraitTableWriter(options.gene_stats_out, key_column="Gene")
    elif mode != "gibbs" and options.gene_stats_out is not None:
        services.log("Ignoring --gene-stats-out for multi-Y betas mode", services.INFO)

    num_traits_completed = 0
    try:
        with tempfile.TemporaryDirectory(prefix="pigean_multi_y_") as tmpdir:
            for begin in range(0, len(phenos), phenos_per_batch):
                end = min(begin + phenos_per_batch, len(phenos))
                services.log(
                    "Processing multi-Y batch %d-%d of %d"
                    % (begin + 1, end, len(phenos)),
                    services.INFO,
                )
                (batch_Y, batch_combined, batch_priors) = pigean_phewas.read_phewas_file_batch(
                    seed_state,
                    options.multi_y_in,
                    begin=begin,
                    cur_batch_size=end - begin,
                    pheno_to_ind=pheno_to_ind,
                    id_col=col_info["id_col"],
                    pheno_col=col_info["pheno_col"],
                    bf_col=col_info["bf_col"],
                    combined_col=col_info["combined_col"],
                    prior_col=col_info["prior_col"],
                    open_text_fn=open_text_with_retry,
                    warn_fn=services.warn,
                )
                batch_response = _select_multi_y_response_matrix(
                    batch_Y,
                    batch_combined,
                    options,
                    services,
                )
                for batch_offset, trait in enumerate(phenos[begin:end]):
                    trait_safe = trait.replace("/", "_").replace(" ", "_")
                    trait_gene_stats = os.path.join(tmpdir, "%06d_%s.gene_stats.tsv" % (begin + batch_offset, trait_safe))
                    _write_trait_gene_stats_file(
                        trait_gene_stats,
                        seed_state.genes,
                        batch_response[:, batch_offset],
                        combined_values=batch_combined[:, batch_offset] if batch_combined is not None else None,
                        prior_values=batch_priors[:, batch_offset] if batch_priors is not None else None,
                    )

                    trait_options = copy.copy(options)
                    trait_options.multi_y_in = None
                    trait_options.multi_y_id_col = None
                    trait_options.multi_y_pheno_col = None
                    trait_options.multi_y_log_bf_col = None
                    trait_options.multi_y_combined_col = None
                    trait_options.multi_y_prior_col = None
                    trait_options.multi_y_max_phenos_per_batch = None
                    _clear_primary_y_inputs(trait_options)
                    trait_options.gene_stats_in = trait_gene_stats
                    trait_options.gene_stats_id_col = "Gene"
                    trait_options.gene_stats_log_bf_col = "log_bf"
                    trait_options.gene_stats_combined_col = "combined" if batch_combined is not None else None
                    trait_options.gene_stats_prior_col = "prior" if batch_priors is not None else None
                    trait_options.gene_universe_from_x = getattr(options, "gene_universe_in", None) is None
                    trait_options.gene_universe_from_y = False
                    trait_options.params_out = None
                    trait_gene_set_stats_out = os.path.join(tmpdir, "%06d_%s.gene_set_stats.out" % (begin + batch_offset, trait_safe))
                    trait_options.gene_set_stats_out = trait_gene_set_stats_out
                    if gene_writer is not None:
                        trait_options.gene_stats_out = os.path.join(tmpdir, "%06d_%s.gene_stats.out" % (begin + batch_offset, trait_safe))
                    else:
                        trait_options.gene_stats_out = None

                    try:
                        from . import dispatch as pigean_dispatch

                        pigean_dispatch.run_main_pipeline(trait_options, mode, services=services)
                    except SystemExit as exc:
                        if exc.code not in (0, None):
                            raise
                        services.log("Skipping trait %s because no gene sets survived the input filters" % trait, services.INFO)
                        continue

                    rows_written = gene_set_writer.append_from(trait, trait_gene_set_stats_out)
                    if rows_written == 0:
                        services.log("Trait %s produced no gene-set rows after write filters" % trait, services.INFO)
                    if gene_writer is not None and trait_options.gene_stats_out is not None:
                        gene_writer.append_from(trait, trait_options.gene_stats_out)
                    num_traits_completed += 1
    finally:
        gene_set_writer.close()
        if gene_writer is not None:
            gene_writer.close()

    if options.params_out is not None:
        seed_state._record_params({"multi_y_num_traits_completed": num_traits_completed}, overwrite=True)
        seed_state.write_params(options.params_out)

    return MultiYPipelineResult(
        state=seed_state,
        mode_state=mode_state,
        sigma2_cond=sigma2_cond,
        y_not_loaded=False,
        num_traits_total=len(phenos),
        num_traits_completed=num_traits_completed,
        phenos_per_batch=phenos_per_batch,
    )
