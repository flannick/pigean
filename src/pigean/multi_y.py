from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass

from pegs_shared.io_common import open_text_with_retry, resolve_column_index

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
    with open_text_with_retry(options.multi_y_in) as fh:
        header_cols = fh.readline().strip("\n").split()

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
    options.betas_from_phewas = False
    options.betas_uncorrected_from_phewas = False
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


def _record_multi_y_params(state, options, mode, *, columns, num_traits_total, phenos_per_batch):
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
            "multi_y_num_traits": num_traits_total,
            "multi_y_phenos_per_batch": phenos_per_batch,
        },
        overwrite=True,
    )


def run_multi_y_pipeline(services, options, mode):
    if mode not in {"betas", "gibbs"}:
        services.bail("Option --multi-y-in is only supported for modes betas and gibbs")
    if options.gene_set_stats_out is None:
        services.bail("Option --multi-y-in requires --gene-set-stats-out")

    columns = _resolve_multi_y_columns(options)
    seed_state = pigean_main_support.build_runtime_state(options)
    mode_state = pigean_main_support.build_mode_state(mode, False)
    sigma2_cond = pigean_main_support.configure_hyperparameters_for_main(seed_state, options)
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
                for batch_offset, trait in enumerate(phenos[begin:end]):
                    trait_safe = trait.replace("/", "_").replace(" ", "_")
                    trait_gene_stats = os.path.join(tmpdir, "%06d_%s.gene_stats.tsv" % (begin + batch_offset, trait_safe))
                    _write_trait_gene_stats_file(
                        trait_gene_stats,
                        seed_state.genes,
                        batch_Y[:, batch_offset],
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
