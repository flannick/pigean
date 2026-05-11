from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
from scipy import sparse


BRIDGE_METRICS_HEADER = [
    "annotation_id",
    "annotation_source",
    "anchor_trait",
    "n_genes_total",
    "n_genes_active",
    "beta",
    "beta_uncorrected",
    "p_value",
    "annotation_degree",
    "factor_neff",
    "dominant_factor",
    "dominant_factor_share",
    "top_factor_1",
    "top_factor_1_overlap",
    "top_factor_2",
    "top_factor_2_overlap",
    "within_kernel_mass",
    "between_kernel_mass",
    "bridge_fraction",
    "separated_bridge_mass",
    "max_bridge_factor_a",
    "max_bridge_factor_b",
    "max_bridge_mass",
    "max_bridge_factor_similarity",
    "source_rank_bridge_fraction",
    "source_rank_separated_bridge_mass",
    "global_rank_separated_bridge_mass",
    "flag_review",
    "flag_suggest_exclude",
    "flag_reason",
]


GENE_FACTOR_CONTRIBS_HEADER = [
    "gene",
    "factor",
    "factor_label",
    "annotation_id",
    "annotation_source",
    "anchor_trait",
    "beta",
    "gene_in_annotation",
    "factor_annotation_overlap",
    "contribution_L_scale",
    "rank_within_gene_factor",
]


def _open_text(path, mode="rt"):
    path_obj = Path(path)
    if path_obj.suffix == ".gz":
        return gzip.open(path_obj, mode, encoding="utf-8")
    return path_obj.open(mode, encoding="utf-8")


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        f_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f_value):
        return "NA"
    return "%.6g" % f_value


def _as_dense_matrix(values):
    if values is None:
        return None
    if sparse.issparse(values):
        values = values.toarray()
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return values


def _effective_beta_matrix(state):
    anchor_mask = getattr(state, "anchor_pheno_mask", None)
    phewas_beta = getattr(state, "X_phewas_beta", None)
    if anchor_mask is not None and phewas_beta is not None:
        beta_matrix = _as_dense_matrix(phewas_beta).T
        anchor_mask = np.asarray(anchor_mask, dtype=bool)
        if anchor_mask.size == beta_matrix.shape[1]:
            beta_matrix = beta_matrix[:, anchor_mask]
            phenos = getattr(state, "phenos", None)
            if phenos is not None:
                anchor_names = [
                    str(phenos[i])
                    for i in np.where(anchor_mask)[0]
                    if i < len(phenos)
                ]
            else:
                anchor_names = []
            return beta_matrix, anchor_names

    betas = getattr(state, "betas", None)
    if betas is None:
        return None, []
    beta_matrix = _as_dense_matrix(betas)
    scale_factors = np.asarray(
        getattr(state, "scale_factors", np.ones(beta_matrix.shape[0], dtype=float)),
        dtype=float,
    )
    scale_factors = np.where(scale_factors == 0, 1.0, scale_factors)
    beta_matrix = beta_matrix / scale_factors[:, np.newaxis]
    return beta_matrix, ["default"] if beta_matrix.shape[1] == 1 else []


def _effective_beta_uncorrected(state):
    anchor_mask = getattr(state, "anchor_pheno_mask", None)
    phewas_beta_uncorrected = getattr(state, "X_phewas_beta_uncorrected", None)
    if anchor_mask is not None and phewas_beta_uncorrected is not None:
        beta_matrix = _as_dense_matrix(phewas_beta_uncorrected).T
        anchor_mask = np.asarray(anchor_mask, dtype=bool)
        if anchor_mask.size == beta_matrix.shape[1]:
            return beta_matrix[:, anchor_mask]

    betas = getattr(state, "betas_uncorrected", None)
    if betas is None:
        return None
    beta_matrix = _as_dense_matrix(betas)
    scale_factors = np.asarray(
        getattr(state, "scale_factors", np.ones(beta_matrix.shape[0], dtype=float)),
        dtype=float,
    )
    scale_factors = np.where(scale_factors == 0, 1.0, scale_factors)
    return beta_matrix / scale_factors[:, np.newaxis]


def _weighted_jaccard_similarity(columns):
    columns = np.asarray(columns, dtype=float)
    k = columns.shape[1]
    sim = np.eye(k, dtype=float)
    for a in range(k):
        left = np.maximum(columns[:, a], 0.0)
        for b in range(a + 1, k):
            right = np.maximum(columns[:, b], 0.0)
            denom = float(np.sum(np.maximum(left, right)))
            value = 0.0 if denom <= 0.0 else float(np.sum(np.minimum(left, right)) / denom)
            sim[a, b] = value
            sim[b, a] = value
    return sim


def _rank_desc(values):
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _column_nnz(matrix):
    if sparse.issparse(matrix):
        return np.asarray(matrix.getnnz(axis=0), dtype=int).ravel()
    return np.sum(np.asarray(matrix) != 0, axis=0).astype(int)


def _factor_name(index):
    return "Factor%d" % (int(index) + 1)


def compute_annotation_bridge_records(
    state,
    *,
    min_active_genes=10,
    review_factor_neff=2.0,
    review_bridge_fraction=0.5,
    exclude_factor_neff=2.5,
    exclude_bridge_fraction=0.75,
    exclude_source_top_frac=0.05,
    exclude_max_similarity=0.5,
):
    discovery_model = str(getattr(state, "discovery_model", None) or (getattr(state, "params", {}) or {}).get("discovery_model", "gene_by_annotation"))
    if discovery_model != "gene_by_gene":
        raise ValueError("annotation bridge diagnostics require --discovery-model gene_by_gene")
    if getattr(state, "X_orig", None) is None or getattr(state, "exp_gene_factors", None) is None:
        raise ValueError("annotation bridge diagnostics require fitted gene-by-gene factors and X annotations")

    gene_mask = np.asarray(getattr(state, "gene_in_discovery_mask", np.ones(len(state.genes), dtype=bool)), dtype=bool)
    gene_set_mask = np.asarray(getattr(state, "gene_set_in_discovery_mask", np.ones(len(state.gene_sets), dtype=bool)), dtype=bool)
    if not np.any(gene_mask) or not np.any(gene_set_mask):
        return []

    gene_indices = np.where(gene_mask)[0]
    gene_set_indices = np.where(gene_set_mask)[0]
    X_active = state.X_orig[gene_indices, :][:, gene_set_indices]
    if not sparse.issparse(X_active):
        X_active = sparse.csr_matrix(np.asarray(X_active, dtype=float))
    X_active = X_active.tocsr()

    W = np.asarray(state.exp_gene_factors, dtype=float)[gene_indices, :]
    W = np.nan_to_num(np.maximum(W, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    if W.ndim != 2 or W.shape[1] == 0:
        return []
    col_sums = np.sum(W, axis=0)
    W_norm = np.divide(W, col_sums[np.newaxis, :], out=np.zeros_like(W), where=col_sums[np.newaxis, :] > 0.0)
    factor_similarity = _weighted_jaccard_similarity(W_norm)
    sep_matrix = 1.0 - factor_similarity
    np.fill_diagonal(sep_matrix, 0.0)

    overlaps = X_active.T @ W_norm
    overlaps = overlaps.toarray() if sparse.issparse(overlaps) else np.asarray(overlaps, dtype=float)
    overlaps = np.nan_to_num(overlaps, nan=0.0, posinf=0.0, neginf=0.0)

    beta_matrix, anchor_names = _effective_beta_matrix(state)
    if beta_matrix is None:
        raise ValueError("annotation bridge diagnostics require corrected beta values")
    beta_matrix = np.nan_to_num(beta_matrix[gene_set_indices, :], nan=0.0, posinf=0.0, neginf=0.0)
    if not anchor_names:
        anchor_names = ["anchor_%d" % (i + 1) for i in range(beta_matrix.shape[1])]
    elif len(anchor_names) < beta_matrix.shape[1]:
        anchor_names = anchor_names + ["anchor_%d" % (i + 1) for i in range(len(anchor_names), beta_matrix.shape[1])]

    beta_uncorrected = _effective_beta_uncorrected(state)
    if beta_uncorrected is not None:
        beta_uncorrected = np.nan_to_num(beta_uncorrected[gene_set_indices, :], nan=0.0, posinf=0.0, neginf=0.0)
    p_values = getattr(state, "p_values", None)
    if p_values is not None:
        p_values = np.asarray(p_values, dtype=float)[gene_set_indices]

    labels = getattr(state, "gene_set_labels", None)
    sources = [str(labels[i]) if labels is not None and i < len(labels) else "unknown" for i in gene_set_indices]
    gene_sets = [str(state.gene_sets[i]) for i in gene_set_indices]
    n_active = _column_nnz(X_active)
    X_total = state.X_orig[:, gene_set_indices]
    n_total = _column_nnz(X_total)
    missing_gene_X = getattr(state, "X_orig_missing_genes", None)
    if missing_gene_X is not None:
        n_total = n_total + _column_nnz(missing_gene_X[:, gene_set_indices])

    records = []
    for anchor_index in range(beta_matrix.shape[1]):
        anchor_name = anchor_names[anchor_index]
        for local_index, annotation_id in enumerate(gene_sets):
            beta = float(beta_matrix[local_index, anchor_index])
            overlap = np.maximum(overlaps[local_index, :], 0.0)
            overlap_sum = float(np.sum(overlap))
            if overlap_sum > 0.0:
                p_factor = overlap / overlap_sum
                factor_neff = float(1.0 / np.sum(p_factor * p_factor))
                dominant_idx = int(np.argmax(p_factor))
                dominant_share = float(np.max(p_factor))
            else:
                p_factor = np.zeros_like(overlap)
                factor_neff = 0.0
                dominant_idx = 0
                dominant_share = 0.0

            top_order = np.argsort(-overlap, kind="mergesort")
            top_1 = int(top_order[0]) if top_order.size > 0 else 0
            top_2 = int(top_order[1]) if top_order.size > 1 else top_1
            within = float(beta * np.sum(overlap * overlap))
            total = float(beta * overlap_sum * overlap_sum)
            between = float(total - within)
            denom = abs(within) + abs(between)
            bridge_fraction = 0.0 if denom <= 0.0 else float(abs(between) / (denom + 1e-12))
            separated = float(beta * (overlap @ sep_matrix @ overlap))

            max_a = 0
            max_b = 0
            max_bridge_mass = 0.0
            max_bridge_similarity = 0.0
            if overlap.size >= 2:
                bridge_pair = 2.0 * beta * np.outer(overlap, overlap)
                np.fill_diagonal(bridge_pair, -np.inf)
                if np.any(np.isfinite(bridge_pair)):
                    flat = int(np.nanargmax(bridge_pair))
                    max_a, max_b = np.unravel_index(flat, bridge_pair.shape)
                    max_bridge_mass = float(bridge_pair[max_a, max_b])
                    max_bridge_similarity = float(factor_similarity[max_a, max_b])

            record = {
                "annotation_id": annotation_id,
                "annotation_source": sources[local_index],
                "anchor_trait": anchor_name,
                "n_genes_total": int(n_total[local_index]),
                "n_genes_active": int(n_active[local_index]),
                "beta": beta,
                "beta_uncorrected": (
                    float(beta_uncorrected[local_index, min(anchor_index, beta_uncorrected.shape[1] - 1)])
                    if beta_uncorrected is not None and beta_uncorrected.size > 0
                    else None
                ),
                "p_value": float(p_values[local_index]) if p_values is not None else None,
                "annotation_degree": int(n_active[local_index]),
                "factor_neff": factor_neff,
                "dominant_factor": _factor_name(dominant_idx),
                "dominant_factor_share": dominant_share,
                "top_factor_1": _factor_name(top_1),
                "top_factor_1_overlap": float(overlap[top_1]) if overlap.size > 0 else 0.0,
                "top_factor_2": _factor_name(top_2),
                "top_factor_2_overlap": float(overlap[top_2]) if overlap.size > 1 else 0.0,
                "within_kernel_mass": within,
                "between_kernel_mass": between,
                "bridge_fraction": bridge_fraction,
                "separated_bridge_mass": separated,
                "max_bridge_factor_a": _factor_name(max_a),
                "max_bridge_factor_b": _factor_name(max_b),
                "max_bridge_mass": max_bridge_mass,
                "max_bridge_factor_similarity": max_bridge_similarity,
                "source_rank_bridge_fraction": 0,
                "source_rank_separated_bridge_mass": 0,
                "global_rank_separated_bridge_mass": 0,
                "flag_review": False,
                "flag_suggest_exclude": False,
                "flag_reason": "",
            }
            records.append(record)

    global_ranks = _rank_desc([abs(record["separated_bridge_mass"]) for record in records])
    for record, rank in zip(records, global_ranks):
        record["global_rank_separated_bridge_mass"] = rank

    source_groups = {}
    for index, record in enumerate(records):
        source_groups.setdefault((record["annotation_source"], record["anchor_trait"]), []).append(index)
    for indices in source_groups.values():
        bf_ranks = _rank_desc([records[i]["bridge_fraction"] for i in indices])
        sep_ranks = _rank_desc([abs(records[i]["separated_bridge_mass"]) for i in indices])
        source_tail_count = max(1, int(math.ceil(len(indices) * float(exclude_source_top_frac))))
        for local, record_index in enumerate(indices):
            record = records[record_index]
            record["source_rank_bridge_fraction"] = bf_ranks[local]
            record["source_rank_separated_bridge_mass"] = sep_ranks[local]
            review_reasons = []
            if (
                record["beta"] > 0.0
                and record["n_genes_active"] >= int(min_active_genes)
                and record["factor_neff"] >= float(review_factor_neff)
                and record["bridge_fraction"] >= float(review_bridge_fraction)
            ):
                record["flag_review"] = True
                review_reasons.append("broad_positive_bridge")
            if (
                record["beta"] > 0.0
                and record["n_genes_active"] >= int(min_active_genes)
                and record["factor_neff"] >= float(exclude_factor_neff)
                and record["bridge_fraction"] >= float(exclude_bridge_fraction)
                and record["source_rank_separated_bridge_mass"] <= source_tail_count
                and record["max_bridge_factor_similarity"] < float(exclude_max_similarity)
            ):
                record["flag_suggest_exclude"] = True
                review_reasons.append("source_tail_separated_bridge")
            record["flag_reason"] = ";".join(review_reasons)
    return records


def compute_gene_factor_annotation_contrib_records(state, *, top_n=10):
    discovery_model = str(getattr(state, "discovery_model", None) or (getattr(state, "params", {}) or {}).get("discovery_model", "gene_by_annotation"))
    if discovery_model != "gene_by_gene":
        raise ValueError("gene-factor annotation contribution diagnostics require --discovery-model gene_by_gene")
    gene_mask = np.asarray(getattr(state, "gene_in_discovery_mask", np.ones(len(state.genes), dtype=bool)), dtype=bool)
    gene_set_mask = np.asarray(getattr(state, "gene_set_in_discovery_mask", np.ones(len(state.gene_sets), dtype=bool)), dtype=bool)
    gene_indices = np.where(gene_mask)[0]
    gene_set_indices = np.where(gene_set_mask)[0]
    if gene_indices.size == 0 or gene_set_indices.size == 0:
        return []
    X_active = state.X_orig[gene_indices, :][:, gene_set_indices]
    if not sparse.issparse(X_active):
        X_active = sparse.csr_matrix(np.asarray(X_active, dtype=float))
    X_active = X_active.tocsr()

    W = np.asarray(state.exp_gene_factors, dtype=float)[gene_indices, :]
    W = np.nan_to_num(np.maximum(W, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    col_sums = np.sum(W, axis=0)
    W_norm = np.divide(W, col_sums[np.newaxis, :], out=np.zeros_like(W), where=col_sums[np.newaxis, :] > 0.0)
    overlaps = X_active.T @ W_norm
    overlaps = overlaps.toarray() if sparse.issparse(overlaps) else np.asarray(overlaps, dtype=float)

    beta_matrix, anchor_names = _effective_beta_matrix(state)
    if beta_matrix is None:
        raise ValueError("gene-factor annotation contribution diagnostics require corrected beta values")
    beta_matrix = np.nan_to_num(beta_matrix[gene_set_indices, :], nan=0.0, posinf=0.0, neginf=0.0)
    if not anchor_names:
        anchor_names = ["anchor_%d" % (i + 1) for i in range(beta_matrix.shape[1])]
    elif len(anchor_names) < beta_matrix.shape[1]:
        anchor_names = anchor_names + ["anchor_%d" % (i + 1) for i in range(len(anchor_names), beta_matrix.shape[1])]
    labels = getattr(state, "gene_set_labels", None)
    sources = [str(labels[i]) if labels is not None and i < len(labels) else "unknown" for i in gene_set_indices]
    factor_labels = getattr(state, "factor_labels", None)
    factor_labels = factor_labels if factor_labels is not None else [_factor_name(i) for i in range(W.shape[1])]

    records = []
    top_n = max(0, int(top_n))
    if top_n == 0:
        return records
    X_csr = X_active.tocsr()
    for local_gene_index, gene_index in enumerate(gene_indices):
        row = X_csr.getrow(local_gene_index)
        annotation_local_indices = row.indices
        annotation_weights = row.data
        if annotation_local_indices.size == 0:
            continue
        for factor_index in range(W.shape[1]):
            for anchor_index in range(beta_matrix.shape[1]):
                contribs = []
                for ann_local, gene_weight in zip(annotation_local_indices, annotation_weights):
                    beta = float(beta_matrix[ann_local, anchor_index])
                    overlap = float(overlaps[ann_local, factor_index])
                    contribution = float(beta * float(gene_weight) * overlap)
                    if contribution <= 0.0:
                        continue
                    contribs.append((contribution, ann_local, gene_weight, overlap, beta))
                contribs.sort(key=lambda item: (-item[0], item[1]))
                for rank, (contribution, ann_local, gene_weight, overlap, beta) in enumerate(contribs[:top_n], start=1):
                    records.append(
                        {
                            "gene": str(state.genes[gene_index]),
                            "factor": _factor_name(factor_index),
                            "factor_label": str(factor_labels[factor_index]) if factor_index < len(factor_labels) else _factor_name(factor_index),
                            "annotation_id": str(state.gene_sets[gene_set_indices[ann_local]]),
                            "annotation_source": sources[ann_local],
                            "anchor_trait": anchor_names[anchor_index],
                            "beta": beta,
                            "gene_in_annotation": float(gene_weight),
                            "factor_annotation_overlap": overlap,
                            "contribution_L_scale": contribution,
                            "rank_within_gene_factor": rank,
                        }
                    )
    return records


def write_annotation_bridge_metrics(state, output_file):
    records = compute_annotation_bridge_records(state)
    with _open_text(output_file, "wt") as output_fh:
        output_fh.write("%s\n" % "\t".join(BRIDGE_METRICS_HEADER))
        for record in records:
            output_fh.write("%s\n" % "\t".join(_fmt(record.get(column, "")) for column in BRIDGE_METRICS_HEADER))


def write_annotation_bridge_suggested_exclude(state, output_file):
    records = compute_annotation_bridge_records(state)
    with _open_text(output_file, "wt") as output_fh:
        for record in records:
            if record.get("flag_suggest_exclude", False):
                output_fh.write("%s\n" % record.get("annotation_id", ""))


def write_gene_factor_annotation_contribs(state, output_file, *, top_n=10):
    records = compute_gene_factor_annotation_contrib_records(state, top_n=top_n)
    with _open_text(output_file, "wt") as output_fh:
        output_fh.write("%s\n" % "\t".join(GENE_FACTOR_CONTRIBS_HEADER))
        for record in records:
            output_fh.write("%s\n" % "\t".join(_fmt(record.get(column, "")) for column in GENE_FACTOR_CONTRIBS_HEADER))
