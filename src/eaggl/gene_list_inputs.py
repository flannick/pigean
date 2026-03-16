from __future__ import annotations

import numpy as np
import scipy.stats


_HEADER_TOKENS = {"gene", "genes", "symbol", "gene_id", "geneid", "id"}


def has_standalone_gene_list_inputs(options):
    return bool(
        options.gene_list_in is not None
        or options.gene_list is not None
        or options.positive_controls_in is not None
        or options.positive_controls_list is not None
    )


def build_standalone_gene_list_inputs(domain, runtime, options):
    selected_genes = _resolve_requested_genes(domain, options)
    if len(selected_genes) == 0:
        domain.bail(
            "Standalone EAGGL gene-list mode requires genes from --gene-list/--gene-list-in "
            "or the compatibility aliases --positive-controls-list/--positive-controls-in"
        )

    if runtime.genes is None or runtime.X_orig is None or runtime.gene_sets is None:
        domain.bail("Standalone EAGGL gene-list mode requires X to be loaded before enrichment")

    gene_universe = set(runtime.genes)
    matched_genes = [gene for gene in selected_genes if gene in gene_universe]
    unmatched_genes = [gene for gene in selected_genes if gene not in gene_universe]
    if len(unmatched_genes) > 0:
        domain.warn(
            "Ignoring %d requested genes that were not found in the loaded X gene universe"
            % len(unmatched_genes)
        )
    if len(matched_genes) == 0:
        domain.bail("None of the requested genes were found in the loaded X gene universe")

    selected_gene_set = set(matched_genes)
    gene_mask = np.array([gene in selected_gene_set for gene in runtime.genes], dtype=bool)
    X_binary = runtime.X_orig.astype(bool)
    universe_size = int(X_binary.shape[0])
    selected_size = int(np.sum(gene_mask))
    gene_set_sizes = runtime.get_col_sums(X_binary, num_nonzero=True)
    overlaps = np.asarray(X_binary[gene_mask, :].sum(axis=0)).ravel().astype(int)

    p_values = np.ones(len(runtime.gene_sets), dtype=float)
    test_mask = np.logical_and(gene_set_sizes > 0, overlaps > 0)
    if np.any(test_mask):
        p_values[test_mask] = scipy.stats.hypergeom.sf(
            overlaps[test_mask] - 1,
            universe_size,
            selected_size,
            gene_set_sizes[test_mask],
        )
    q_values = _benjamini_hochberg_q_values(p_values)

    max_fdr_q = float(options.gene_list_max_fdr_q)
    keep_mask = q_values <= max_fdr_q
    if not np.any(keep_mask):
        domain.log(
            "No gene sets passed the standalone gene-list enrichment filter "
            "(q <= %.3g); stopping" % max_fdr_q
        )
        domain.sys.exit(0)

    retained_gene_set_sizes = gene_set_sizes[keep_mask].astype(float)
    retained_p_values = p_values[keep_mask]
    retained_q_values = q_values[keep_mask]
    retained_external_weights = -np.log(np.maximum(retained_p_values, np.finfo(float).tiny))
    retained_external_weights /= np.sqrt(np.maximum(retained_gene_set_sizes, 1.0))

    runtime.subset_gene_sets(
        keep_mask,
        keep_missing=False,
        ignore_missing=True,
        skip_V=True,
        skip_scale_factors=False,
    )

    retained_gene_mask = np.asarray(runtime.X_orig.astype(bool).sum(axis=1)).ravel() > 0
    runtime._subset_genes(
        retained_gene_mask,
        skip_V=True,
        skip_scale_factors=False,
    )
    if runtime.scale_factors is not None:
        runtime.scale_factors = np.array(runtime.scale_factors, copy=True)
        runtime.scale_factors[np.isclose(runtime.scale_factors, 0)] = 1.0
    runtime.p_values = retained_p_values
    runtime.q_values = retained_q_values
    runtime.betas_uncorrected = retained_external_weights * runtime.scale_factors
    runtime.betas = np.array(runtime.betas_uncorrected, copy=True)

    final_Y = np.array([1.0 if gene in selected_gene_set else 0.0 for gene in runtime.genes], dtype=float)
    runtime._set_Y(
        final_Y,
        final_Y.copy(),
        Y_positive_controls=final_Y.copy(),
        skip_V=True,
        skip_scale_factors=True,
    )
    runtime.gene_to_positive_controls = {gene: 1.0 for gene in runtime.genes if gene in selected_gene_set}
    runtime._record_params(
        {
            "gene_list_mode": "standalone_hypergeometric",
            "gene_list_input_genes_total": len(selected_genes),
            "gene_list_input_genes_matched": len(matched_genes),
            "gene_list_input_genes_unmatched": len(unmatched_genes),
            "gene_list_gene_universe_size": universe_size,
            "gene_list_max_fdr_q": max_fdr_q,
            "gene_list_num_gene_sets_retained": int(np.sum(keep_mask)),
            "gene_list_num_genes_retained": len(runtime.genes),
        },
        overwrite=False,
    )
    domain.log(
        "Standalone gene-list enrichment retained %d gene sets and %d genes from %d matched input genes"
        % (len(runtime.gene_sets), len(runtime.genes), len(matched_genes)),
        domain.INFO,
    )


def _resolve_requested_genes(domain, options):
    selected = []
    seen = set()

    def _add_gene(gene):
        if gene is None:
            return
        gene = str(gene).strip()
        if len(gene) == 0 or gene in seen:
            return
        selected.append(gene)
        seen.add(gene)

    for gene in options.gene_list or []:
        _add_gene(gene)
    for gene in options.positive_controls_list or []:
        _add_gene(gene)

    if options.gene_list_in is not None:
        for gene in _read_gene_list_file(
            domain,
            options.gene_list_in,
            id_selector=options.gene_list_id_col,
            no_header=bool(options.gene_list_no_header),
        ):
            _add_gene(gene)

    if options.positive_controls_in is not None:
        domain.warn(
            "Treating --positive-controls-in as a compatibility alias for standalone "
            "EAGGL gene-list mode; prefer --gene-list-in"
        )
        for gene in _read_gene_list_file(
            domain,
            options.positive_controls_in,
            id_selector=1,
            no_header=False,
        ):
            _add_gene(gene)

    if options.positive_controls_all_in is not None:
        domain.warn(
            "Ignoring --positive-controls-all-in in standalone EAGGL gene-list mode; "
            "the enrichment universe is the loaded X gene universe"
        )

    return selected


def _read_gene_list_file(domain, path, *, id_selector=1, no_header=False):
    genes = []
    selected_col = None
    seen_header = False
    with domain.open_gz(path) as input_fh:
        for raw_line in input_fh:
            line = raw_line.strip()
            if len(line) == 0:
                continue
            cols = line.split()
            if not seen_header:
                seen_header = True
                selected_col, skip_header = _resolve_gene_list_column(
                    cols,
                    id_selector=id_selector,
                    no_header=no_header,
                    bail_fn=domain.bail,
                )
                if skip_header:
                    continue
            if selected_col is None:
                selected_col = 0
            if selected_col >= len(cols):
                domain.warn("Skipping gene-list row with too few columns: %s" % raw_line.rstrip("\n"))
                continue
            genes.append(cols[selected_col])
    return genes


def _resolve_gene_list_column(cols, *, id_selector, no_header, bail_fn):
    if no_header:
        return (_normalize_gene_list_column_index(id_selector, len(cols), bail_fn), False)

    if isinstance(id_selector, str):
        if id_selector in cols:
            return (cols.index(id_selector), True)
        return (_normalize_gene_list_column_index(1, len(cols), bail_fn), False)

    normalized_index = _normalize_gene_list_column_index(id_selector, len(cols), bail_fn)
    first_token = cols[normalized_index].strip().lower()
    if first_token in _HEADER_TOKENS:
        return (normalized_index, True)
    return (normalized_index, False)


def _normalize_gene_list_column_index(id_selector, num_cols, bail_fn):
    if isinstance(id_selector, str):
        try:
            id_selector = int(id_selector)
        except Exception:
            bail_fn("Could not resolve gene-list ID column selector '%s'" % id_selector)
    index = int(id_selector) - 1
    if index < 0 or index >= num_cols:
        bail_fn("Gene-list ID column %s is out of range for a row with %d columns" % (id_selector, num_cols))
    return index


def _benjamini_hochberg_q_values(p_values):
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    if not np.any(finite_mask):
        return q_values

    finite_indices = np.where(finite_mask)[0]
    finite_p = p_values[finite_mask]
    order = np.argsort(finite_p, kind="mergesort")
    ordered_p = finite_p[order]
    m = float(len(ordered_p))
    adjusted = np.empty_like(ordered_p)
    running_min = 1.0
    for position in range(len(ordered_p) - 1, -1, -1):
        rank = float(position + 1)
        candidate = ordered_p[position] * (m / rank)
        running_min = min(running_min, candidate)
        adjusted[position] = running_min
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_values[finite_indices[order]] = adjusted
    return q_values
