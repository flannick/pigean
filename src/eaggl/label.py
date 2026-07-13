from __future__ import annotations

import csv
import re
from dataclasses import dataclass

import numpy as np

try:
    from pegs_cli_errors import CliUsageError
except ImportError:
    from .pegs_cli_errors import CliUsageError  # type: ignore

from . import cli as eaggl_cli
from . import factor as eaggl_factor
from . import labeling as eaggl_labeling
from . import state as eaggl_state


_FACTOR_COLUMN_RE = re.compile(r"^Factor([0-9]+)$")


@dataclass
class LoadingTable:
    kind: str
    path: str
    ids: list[str]
    factor_columns: list[str]
    loadings: np.ndarray
    rows: list[dict[str, str]]
    id_column: str


def _bail(message):
    raise CliUsageError(message)


def _warn(message):
    eaggl_cli.warn(message)


def _log(message, level=None):
    eaggl_cli.log(message, eaggl_cli.INFO if level is None else level)


def _factor_number(column_name):
    match = _FACTOR_COLUMN_RE.match(str(column_name))
    if match is None:
        return None
    return int(match.group(1))


def _factor_columns(fieldnames, path):
    found = []
    for column_name in fieldnames or []:
        factor_number = _factor_number(column_name)
        if factor_number is not None:
            found.append((factor_number, column_name))
    found.sort(key=lambda item: item[0])
    columns = [column_name for _, column_name in found]
    if not columns:
        _bail("Could not find raw Factor1..FactorK loading columns in %s" % path)
    expected = list(range(1, len(columns) + 1))
    observed = [number for number, _ in found]
    if observed != expected:
        _bail(
            "Factor columns in %s must be contiguous Factor1..FactorK; found %s"
            % (path, ",".join("Factor%d" % value for value in observed))
        )
    return columns


def _coerce_float(raw_value, *, column_name, row_id, path):
    if raw_value is None or raw_value == "" or raw_value == "NA":
        return 0.0
    try:
        value = float(raw_value)
    except ValueError:
        _bail("Could not parse numeric value for %s in %s row %s: %s" % (column_name, path, row_id, raw_value))
    if not np.isfinite(value):
        return 0.0
    return value


def _read_wide_loading_table(path, *, kind, id_column, id_column_candidates):
    with eaggl_state.open_gz(path, "r") as input_fh:
        reader = csv.DictReader(input_fh, delimiter="\t")
        if reader.fieldnames is None:
            _bail("Empty loading file: %s" % path)
        resolved_id_column = id_column
        if resolved_id_column is None:
            for candidate in id_column_candidates:
                if candidate in reader.fieldnames:
                    resolved_id_column = candidate
                    break
        if resolved_id_column is None or resolved_id_column not in reader.fieldnames:
            _bail(
                "Could not find an identifier column in %s; tried %s"
                % (path, ",".join(id_column_candidates))
            )
        factor_columns = _factor_columns(reader.fieldnames, path)
        ids = []
        rows = []
        loadings = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            row_id = row.get(resolved_id_column, "")
            if row_id is None or str(row_id).strip() == "":
                _bail("Missing %s value in %s at row %d" % (resolved_id_column, path, row_number))
            row_id = str(row_id)
            if row_id in seen:
                _bail("Duplicate %s value in %s: %s" % (resolved_id_column, path, row_id))
            seen.add(row_id)
            ids.append(row_id)
            rows.append(dict(row))
            loadings.append(
                [
                    _coerce_float(row.get(column_name), column_name=column_name, row_id=row_id, path=path)
                    for column_name in factor_columns
                ]
            )
    if not ids:
        _bail("No rows found in loading file: %s" % path)
    return LoadingTable(
        kind=kind,
        path=path,
        ids=ids,
        factor_columns=factor_columns,
        loadings=np.asarray(loadings, dtype=float),
        rows=rows,
        id_column=resolved_id_column,
    )


def _read_long_trait_factor_links(path, *, loading_column):
    with eaggl_state.open_gz(path, "r") as input_fh:
        reader = csv.DictReader(input_fh, delimiter="\t")
        if reader.fieldnames is None:
            _bail("Empty trait-factor links file: %s" % path)
        trait_column = None
        for candidate in ["trait", "Trait", "Pheno", "pheno"]:
            if candidate in reader.fieldnames:
                trait_column = candidate
                break
        if trait_column is None:
            _bail("Could not find trait/phenotype column in %s" % path)
        if "factor" not in reader.fieldnames and "Factor" not in reader.fieldnames:
            _bail("Could not find factor column in %s" % path)
        factor_column = "factor" if "factor" in reader.fieldnames else "Factor"
        if loading_column not in reader.fieldnames:
            _bail("--label-trait-factor-link-loading-col '%s' not found in %s" % (loading_column, path))

        trait_order = []
        trait_to_index = {}
        factor_numbers = set()
        observed = {}
        for row_number, row in enumerate(reader, start=2):
            trait = row.get(trait_column, "")
            if trait is None or str(trait).strip() == "":
                _bail("Missing trait value in %s at row %d" % (path, row_number))
            trait = str(trait)
            raw_factor = row.get(factor_column, "")
            factor_number = _factor_number(raw_factor)
            if factor_number is None:
                _bail("Could not parse factor value in %s at row %d: %s" % (path, row_number, raw_factor))
            if trait not in trait_to_index:
                trait_to_index[trait] = len(trait_order)
                trait_order.append(trait)
            factor_numbers.add(factor_number)
            observed[(trait, factor_number)] = _coerce_float(
                row.get(loading_column),
                column_name=loading_column,
                row_id="%s/%s" % (trait, raw_factor),
                path=path,
            )

    if not trait_order:
        _bail("No trait-factor rows found in %s" % path)
    max_factor = max(factor_numbers)
    expected = set(range(1, max_factor + 1))
    if factor_numbers != expected:
        _bail(
            "Trait-factor links in %s must contain contiguous Factor1..FactorK; found %s"
            % (path, ",".join("Factor%d" % value for value in sorted(factor_numbers)))
        )
    factor_columns = ["Factor%d" % value for value in range(1, max_factor + 1)]
    matrix = np.zeros((len(trait_order), max_factor), dtype=float)
    for (trait, factor_number), value in observed.items():
        matrix[trait_to_index[trait], factor_number - 1] = value
    return LoadingTable(
        kind="pheno",
        path=path,
        ids=trait_order,
        factor_columns=factor_columns,
        loadings=matrix,
        rows=[{"Pheno": trait} for trait in trait_order],
        id_column="Pheno",
    )


def _validate_factor_columns(tables):
    reference = None
    reference_path = None
    for table in tables:
        if table is None:
            continue
        if reference is None:
            reference = list(table.factor_columns)
            reference_path = table.path
            continue
        if list(table.factor_columns) != reference:
            _bail(
                "Label-only inputs disagree on factor columns: %s has %s, but %s has %s"
                % (
                    table.path,
                    ",".join(table.factor_columns),
                    reference_path,
                    ",".join(reference),
                )
            )
    if reference is None:
        _bail("label mode requires at least one loading input")
    return reference


def _top_items(ids, loadings, *, loading_type, num_top=5, allowed_ids=None, context="items"):
    if ids is None or loadings is None:
        return []
    matrix = np.asarray(loadings, dtype=float)
    if matrix.ndim != 2:
        _bail("Expected a 2D loading matrix for %s" % context)
    scores = eaggl_state.EagglState().get_factor_loadings(matrix, loading_type=loading_type)
    if loading_type == "euclidean":
        scores = eaggl_state.EagglState().get_factor_loadings(matrix, loading_type="euclidean_score")
    allowed = None
    if allowed_ids is not None:
        allowed = np.asarray([item in allowed_ids for item in ids], dtype=bool)
        if not np.any(allowed):
            _warn("--gene-sets-for-labeling has no overlap with supplied gene-set loadings; using all gene sets for labels")
            allowed = None
    top_by_factor = []
    for factor_index in range(scores.shape[1]):
        order = list(np.argsort(-scores[:, factor_index]))
        if allowed is not None:
            filtered = [idx for idx in order if bool(allowed[idx]) and scores[idx, factor_index] > 0]
            if filtered:
                order = filtered
            else:
                _warn("Factor%d has no positive-loading gene sets matching --gene-sets-for-labeling; using unfiltered gene sets for its label" % (factor_index + 1))
        chosen = [ids[idx] for idx in order[: min(num_top, len(order))] if scores[idx, factor_index] > 0]
        top_by_factor.append(chosen)
    return top_by_factor


def _set_labels_with_llm(labels, label_payloads, options):
    if getattr(options, "lmm_auth_key", None) is None or not label_payloads:
        return labels
    client = eaggl_labeling.build_labeling_client(
        auth_key=options.lmm_auth_key,
        lmm_model=options.lmm_model,
        lmm_provider=options.lmm_provider,
        bail_fn=_bail,
    )
    prompt = (
        "Print a label, five words maximum, for each group. "
        "Print only labels, one per line, label number followed by text: %s"
        % " ".join("%d. %s" % (index + 1, payload) for index, payload in enumerate(label_payloads))
    )
    _log("Querying LMM with prompt: %s" % prompt)
    response = client.query(prompt, warn_fn=_warn)
    if response is None:
        return labels
    responses = [line.strip() for line in response.strip("\n").split("\n") if line.strip()]
    if len(responses) != len(labels):
        _log("Couldn't decode LMM response %s; using simple label" % response)
        return labels
    out = []
    for raw in responses:
        tokens = raw.split()
        if len(tokens) > 1 and tokens[0].endswith("."):
            try:
                int(tokens[0][:-1])
                raw = " ".join(tokens[1:])
            except ValueError:
                pass
        out.append(raw)
    return out


def _populate_labels(runtime, *, gene_table, gene_set_table, pheno_table, options, gene_sets_for_labeling):
    num_factors = runtime.num_factors()
    gene_set_tops = _top_items(
        gene_set_table.ids if gene_set_table is not None else None,
        gene_set_table.loadings if gene_set_table is not None else None,
        loading_type=options.factor_top_loading_type,
        allowed_ids=gene_sets_for_labeling,
        context="gene sets",
    )
    gene_tops = _top_items(
        gene_table.ids if gene_table is not None else None,
        gene_table.loadings if gene_table is not None else None,
        loading_type=options.factor_top_loading_type,
        context="genes",
    )
    pheno_tops = _top_items(
        pheno_table.ids if pheno_table is not None else None,
        pheno_table.loadings if pheno_table is not None else None,
        loading_type=options.factor_top_loading_type,
        context="phenotypes",
    )

    runtime.factor_top_gene_sets = gene_set_tops if gene_set_tops else [[] for _ in range(num_factors)]
    runtime.factor_top_genes = gene_tops if gene_tops else ([] if gene_table is not None else None)
    runtime.factor_top_phenos = pheno_tops if pheno_tops else ([] if pheno_table is not None else None)
    runtime.factor_anchor_top_gene_sets = [[runtime.factor_top_gene_sets[i]] for i in range(num_factors)]
    runtime.factor_anchor_top_genes = [[runtime.factor_top_genes[i]] for i in range(num_factors)] if runtime.factor_top_genes is not None else None
    runtime.factor_anchor_top_phenos = [[runtime.factor_top_phenos[i]] for i in range(num_factors)] if runtime.factor_top_phenos is not None else None

    if options.label_gene_sets_only and gene_set_table is None:
        _bail("--label-gene-sets-only requires --label-gene-set-clusters-in")

    labels = []
    label_payloads = []
    for factor_index in range(num_factors):
        gene_sets = runtime.factor_top_gene_sets[factor_index] if runtime.factor_top_gene_sets is not None else []
        genes = runtime.factor_top_genes[factor_index] if runtime.factor_top_genes is not None else []
        phenos = runtime.factor_top_phenos[factor_index] if runtime.factor_top_phenos is not None else []
        if gene_sets:
            simple_label = gene_sets[0]
        elif not options.label_gene_sets_only and genes:
            simple_label = genes[0]
        elif phenos:
            simple_label = phenos[0]
        else:
            simple_label = "Factor%d" % (factor_index + 1)
        labels.append(simple_label)

        payload = list(gene_sets)
        if not options.label_gene_sets_only:
            payload.extend(genes)
        if options.label_include_phenos or (not gene_sets and not genes):
            payload.extend(phenos)
        label_payloads.append(",".join(payload))

    runtime.factor_labels = _set_labels_with_llm(labels, label_payloads, options)
    runtime.factor_labels_gene_sets = None
    runtime.factor_labels_genes = None
    runtime.factor_labels_phenos = None
    if options.label_individually and getattr(options, "lmm_auth_key", None) is not None:
        if gene_set_table is not None:
            simple_gene_set_labels = [
                (runtime.factor_top_gene_sets[i][0] if runtime.factor_top_gene_sets[i] else "Factor%d" % (i + 1))
                for i in range(num_factors)
            ]
            runtime.factor_labels_gene_sets = _set_labels_with_llm(
                simple_gene_set_labels,
                [",".join(runtime.factor_top_gene_sets[i]) for i in range(num_factors)],
                options,
            )
        if gene_table is not None and runtime.factor_top_genes is not None:
            simple_gene_labels = [
                (runtime.factor_top_genes[i][0] if runtime.factor_top_genes[i] else "Factor%d" % (i + 1))
                for i in range(num_factors)
            ]
            runtime.factor_labels_genes = _set_labels_with_llm(
                simple_gene_labels,
                [",".join(runtime.factor_top_genes[i]) for i in range(num_factors)],
                options,
            )
        if pheno_table is not None and runtime.factor_top_phenos is not None:
            simple_pheno_labels = [
                (runtime.factor_top_phenos[i][0] if runtime.factor_top_phenos[i] else "Factor%d" % (i + 1))
                for i in range(num_factors)
            ]
            runtime.factor_labels_phenos = _set_labels_with_llm(
                simple_pheno_labels,
                [",".join(runtime.factor_top_phenos[i]) for i in range(num_factors)],
                options,
            )


def _row_metric_values(loadings):
    matrix = np.asarray(loadings, dtype=float)
    cosine = eaggl_state.EagglState().get_factor_loadings(matrix, loading_type="cosine")
    euclidean = eaggl_state.EagglState().get_factor_loadings(matrix, loading_type="euclidean")
    return cosine, euclidean


def _write_wide_cluster_table(path, table, labels, *, cluster_row_min_max_loading):
    if path is None or table is None:
        return
    loadings = np.asarray(table.loadings, dtype=float)
    cosine, euclidean = _row_metric_values(loadings)
    metadata_columns = [
        column
        for column in (table.rows[0].keys() if table.rows else [table.id_column])
        if column not in set(table.factor_columns)
        and not column.startswith("Cosine_Factor")
        and not column.startswith("Euclidean_Factor")
        and column not in {"cluster", "label"}
    ]
    if table.id_column not in metadata_columns:
        metadata_columns = [table.id_column] + metadata_columns
    output_header = (
        metadata_columns
        + ["cluster", "label"]
        + table.factor_columns
        + ["Cosine_%s" % column for column in table.factor_columns]
        + ["Euclidean_%s" % column for column in table.factor_columns]
    )
    with eaggl_state.open_gz(path, "w") as output_fh:
        output_fh.write("%s\n" % "\t".join(output_header))
        for row_index, row in enumerate(table.rows):
            row_loadings = np.nan_to_num(loadings[row_index, :], nan=0.0, posinf=0.0, neginf=0.0)
            if cluster_row_min_max_loading is not None and float(cluster_row_min_max_loading) > 0:
                if row_loadings.size == 0 or float(np.max(row_loadings)) < float(cluster_row_min_max_loading):
                    continue
            if row_loadings.size == 0 or float(np.max(row_loadings)) <= 0:
                cluster_index = 0
            else:
                cluster_index = int(np.argmax(row_loadings))
            values = []
            for column in metadata_columns:
                if column == table.id_column:
                    values.append(str(table.ids[row_index]))
                else:
                    values.append(str(row.get(column, "")))
            values.extend(["Factor%d" % (cluster_index + 1), labels[cluster_index]])
            values.extend("%.4g" % value for value in row_loadings)
            values.extend("%.4g" % value for value in cosine[row_index, :])
            values.extend("%.4g" % value for value in euclidean[row_index, :])
            output_fh.write("%s\n" % "\t".join(values))


def _write_trait_factor_links(path, table):
    if path is None or table is None:
        return
    loadings = np.asarray(table.loadings, dtype=float)
    cosine, euclidean = _row_metric_values(loadings)
    with eaggl_state.open_gz(path, "w") as output_fh:
        output_fh.write("trait\tfactor\tnnls_loading\tcosine_loading\teuclidean_distance\n")
        for trait_index, trait in enumerate(table.ids):
            for factor_index, factor_name in enumerate(table.factor_columns):
                output_fh.write(
                    "%s\t%s\t%.6g\t%.6g\t%.6g\n"
                    % (
                        trait,
                        factor_name,
                        loadings[trait_index, factor_index],
                        cosine[trait_index, factor_index],
                        euclidean[trait_index, factor_index],
                    )
                )


def _build_runtime(options, *, gene_table, gene_set_table, pheno_table, factor_columns):
    runtime = eaggl_state.EagglState(background_prior=options.background_prior, batch_size=options.batch_size)
    num_factors = len(factor_columns)
    runtime.exp_lambdak = np.ones(num_factors, dtype=float)
    runtime.factor_relevance = np.ones(num_factors, dtype=float)
    runtime.factor_marginal_relevance = None
    runtime.factor_anchor_relevance = np.ones((num_factors, 1), dtype=float)
    runtime.factor_anchor_marginal_relevance = None

    if gene_table is not None:
        runtime.genes = gene_table.ids
        runtime.gene_to_ind = {gene: index for index, gene in enumerate(gene_table.ids)}
        runtime.exp_gene_factors = gene_table.loadings
        runtime.gene_in_discovery_mask = np.ones(len(gene_table.ids), dtype=bool)
        runtime.gene_factor_gene_mask = runtime.gene_in_discovery_mask
        runtime.gene_prob_factor_vector = np.ones((len(gene_table.ids), 1), dtype=float)
    if gene_set_table is not None:
        runtime.gene_sets = gene_set_table.ids
        runtime.gene_set_to_ind = {gene_set: index for index, gene_set in enumerate(gene_set_table.ids)}
        runtime.exp_gene_set_factors = gene_set_table.loadings
        runtime.gene_set_in_discovery_mask = np.ones(len(gene_set_table.ids), dtype=bool)
        runtime.gene_set_factor_gene_set_mask = runtime.gene_set_in_discovery_mask
        runtime.gene_set_prob_factor_vector = np.ones((len(gene_set_table.ids), 1), dtype=float)
    else:
        runtime.gene_sets = []
        runtime.gene_set_to_ind = {}
        runtime.exp_gene_set_factors = np.zeros((0, num_factors), dtype=float)
        runtime.gene_set_in_discovery_mask = np.zeros(0, dtype=bool)
        runtime.gene_set_factor_gene_set_mask = runtime.gene_set_in_discovery_mask
        runtime.gene_set_prob_factor_vector = np.zeros((0, 1), dtype=float)
    if pheno_table is not None:
        runtime.phenos = pheno_table.ids
        runtime.pheno_to_ind = {pheno: index for index, pheno in enumerate(pheno_table.ids)}
        runtime.exp_pheno_factors = pheno_table.loadings
        runtime.pheno_in_discovery_mask = np.ones(len(pheno_table.ids), dtype=bool)
        runtime.pheno_factor_pheno_mask = runtime.pheno_in_discovery_mask
        runtime.pheno_prob_factor_vector = np.ones((len(pheno_table.ids), 1), dtype=float)

    runtime._record_params(
        {
            "label_only_mode": True,
            "label_only_num_factors": num_factors,
            "label_only_gene_clusters_in": getattr(options, "label_gene_clusters_in", None),
            "label_only_gene_set_clusters_in": getattr(options, "label_gene_set_clusters_in", None),
            "label_only_pheno_clusters_in": getattr(options, "label_pheno_clusters_in", None),
            "label_only_trait_factor_links_in": getattr(options, "label_trait_factor_links_in", None),
            "factor_top_loading_type": options.factor_top_loading_type,
        },
        overwrite=True,
    )
    return runtime


def run_label_command(options, *, cli_specified_dests=None):
    eaggl_state.configure_runtime_context(cli_module=eaggl_cli)
    if options.factor_top_loading_type not in {"raw", "cosine", "euclidean"}:
        _bail("--factor-top-loading-type must be one of: raw, cosine, euclidean")

    gene_table = (
        _read_wide_loading_table(
            options.label_gene_clusters_in,
            kind="gene",
            id_column=options.label_gene_id_col,
            id_column_candidates=["Gene", "gene"],
        )
        if options.label_gene_clusters_in is not None
        else None
    )
    gene_set_table = (
        _read_wide_loading_table(
            options.label_gene_set_clusters_in,
            kind="gene_set",
            id_column=options.label_gene_set_id_col,
            id_column_candidates=["Gene_Set", "gene_set", "GeneSet", "geneset"],
        )
        if options.label_gene_set_clusters_in is not None
        else None
    )
    if options.label_pheno_clusters_in is not None:
        pheno_table = _read_wide_loading_table(
            options.label_pheno_clusters_in,
            kind="pheno",
            id_column=options.label_pheno_id_col,
            id_column_candidates=["Pheno", "Trait", "trait", "pheno"],
        )
    elif options.label_trait_factor_links_in is not None:
        pheno_table = _read_long_trait_factor_links(
            options.label_trait_factor_links_in,
            loading_column=options.label_trait_factor_link_loading_col,
        )
    else:
        pheno_table = None

    factor_columns = _validate_factor_columns([gene_table, gene_set_table, pheno_table])
    runtime = _build_runtime(
        options,
        gene_table=gene_table,
        gene_set_table=gene_set_table,
        pheno_table=pheno_table,
        factor_columns=factor_columns,
    )
    gene_sets_for_labeling = eaggl_factor._read_gene_sets_for_labeling(
        getattr(options, "gene_sets_for_labeling", None),
        id_col=getattr(options, "gene_sets_for_labeling_id_col", None),
        bail_fn=_bail,
    )
    if gene_sets_for_labeling is not None and gene_set_table is None:
        _warn("--gene-sets-for-labeling was supplied without --label-gene-set-clusters-in; ignoring")
        gene_sets_for_labeling = None

    _populate_labels(
        runtime,
        gene_table=gene_table,
        gene_set_table=gene_set_table,
        pheno_table=pheno_table,
        options=options,
        gene_sets_for_labeling=gene_sets_for_labeling,
    )

    output_scope = options.factor_output_scope
    if cli_specified_dests is None or "factor_output_scope" not in cli_specified_dests:
        output_scope = "all"
    runtime.write_matrix_factors(options.factors_out, factor_output_scope=output_scope)
    runtime.write_factor_metrics(options.factor_metrics_out)
    _write_wide_cluster_table(
        options.gene_clusters_out,
        gene_table,
        runtime.factor_labels,
        cluster_row_min_max_loading=options.cluster_row_min_max_loading,
    )
    _write_wide_cluster_table(
        options.gene_set_clusters_out,
        gene_set_table,
        runtime.factor_labels,
        cluster_row_min_max_loading=options.cluster_row_min_max_loading,
    )
    _write_wide_cluster_table(
        options.label_pheno_clusters_out,
        pheno_table,
        runtime.factor_labels,
        cluster_row_min_max_loading=options.cluster_row_min_max_loading,
    )
    _write_trait_factor_links(options.trait_factor_links_out, pheno_table)
    if options.params_out is not None:
        runtime.write_params(options.params_out)
    _log("Labeled %d factors from precomputed loadings" % runtime.num_factors(), eaggl_cli.INFO)
    return runtime
