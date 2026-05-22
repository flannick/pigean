from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np


FACTOR_PREFIX = "Factor"


DEFAULT_DISTINCT_PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # rose
    "#E69F00",  # amber
    "#56B4E9",  # sky
    "#7F3C8D",  # plum
    "#8C6D31",  # umber
    "#00A6A6",  # teal
    "#B22222",  # brick
    "#4B6F44",  # moss
    "#6B4C9A",  # violet
    "#C49A00",  # ochre
    "#1B9E77",  # deep mint
    "#E7298A",  # magenta
    "#666666",  # graphite
)


@dataclass(frozen=True)
class FactorInfo:
    factor: str
    label: str
    relevance: float


@dataclass(frozen=True)
class EntityInfo:
    entity_id: str
    label: str
    kind: str
    combined: float
    direct: float
    loadings: dict[str, float]
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphConfig:
    gene_min_loading: float = 0.01
    trait_min_loading: float = 0.005
    trait_min_neff: float = 25.0
    gene_min_loading_frac: float = 0.5
    trait_min_loading_frac: float = 0.5
    max_num_factor_nodes: int = 50
    max_num_gene_nodes_per_factor: int = 3
    max_num_trait_nodes_per_factor: int = 3
    max_num_trait_provenance_per_factor: int = 20
    trait_factor_min_beta: float = 0.01
    trait_factor_min_beta_uncorrected: float = 0.05
    trait_factor_min_nnls: float = 0.5
    trait_factor_rank_field: str = "beta"
    factor_trait_enrichments_in: str | None = None
    max_anchor_support_rows_per_node: int = 20
    anchor_support_min_combined: float = 0.0
    coordinate_scale: float = 5.0
    trait_coordinate_scale: float = 0.2
    trait_layout_mode: str = "anchored_top_factor"
    trait_min_centroid_distance_frac: float = 0.35
    trait_edge_length_scale: float = 0.2
    node_size_scale: float = 2.0
    edge_max_width: float = 5.0
    label_max_chars: int = 20
    colors_red_blue: bool = False
    color_by: str = "auto"
    multi_anchor: bool = False
    anchor_trait_names: tuple[str, ...] = ()
    seed: int = 0


def _bail(message: str) -> None:
    raise SystemExit("Error: %s" % message)


def open_text(path: str | Path, mode: str = "rt"):
    path_obj = Path(path)
    path_str = str(path_obj)
    if "r" in mode and path_str.endswith(".gz"):
        try:
            with open(path_obj, "rb") as fh:
                is_gzip = fh.read(2) == b"\x1f\x8b"
        except OSError:
            is_gzip = True
        if is_gzip:
            return gzip.open(path_obj, mode, encoding="utf-8")
    elif "w" in mode and path_str.endswith(".gz"):
        return gzip.open(path_obj, mode, encoding="utf-8")
    return open(path_obj, mode, encoding="utf-8")


def _split_header(line: str) -> tuple[list[str], str | None]:
    delim = "\t" if "\t" in line else None
    return line.rstrip("\n").split(delim), delim


def _get_col(header: list[str], col: str | None, *, default: str | None = None, required: bool = True) -> int | None:
    if col is None:
        col = default
    if col is None:
        return None
    try:
        return int(col)
    except ValueError:
        pass
    matches = [i for i, value in enumerate(header) if value == col]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        _bail("Column %s appears more than once in header: %s" % (col, "\t".join(header)))
    if required:
        _bail("Could not find column %s in header: %s" % (col, "\t".join(header)))
    return None


def _safe_float(raw: str | None, default: float = 0.0) -> float:
    if raw is None or raw == "" or raw == "NA":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return value


def _factor_sort_key(name: str) -> tuple[int, str]:
    if name.startswith(FACTOR_PREFIX):
        suffix = name[len(FACTOR_PREFIX):]
        if suffix.isdigit():
            return int(suffix), name
    return sys.maxsize, name


def detect_factor_columns(header: list[str], factor_names: Iterable[str] | None = None) -> list[str]:
    if factor_names is not None:
        names = [str(value) for value in factor_names]
        missing = [name for name in names if name not in header]
        if missing:
            _bail("Missing factor columns: %s" % ", ".join(missing))
        return names
    factors = [col for col in header if col.startswith(FACTOR_PREFIX) and col[len(FACTOR_PREFIX):].isdigit()]
    return sorted(factors, key=_factor_sort_key)


def read_factors(path: str | Path, *, id_col: str | None = None, label_col: str | None = None, relevance_col: str | None = None) -> list[FactorInfo]:
    with open_text(path) as fh:
        header_line = fh.readline()
        if not header_line:
            _bail("Empty factors file: %s" % path)
        header, delim = _split_header(header_line)
        factor_i = _get_col(header, id_col, default="Factor")
        label_i = _get_col(header, label_col, default="label", required=False)
        relevance_i = _get_col(header, relevance_col, default="relevance", required=False)
        if relevance_i is None:
            relevance_i = _get_col(header, "any_relevance", required=False)
        factors: list[FactorInfo] = []
        for line in fh:
            cols = line.rstrip("\n").split(delim)
            if factor_i is None or factor_i >= len(cols):
                continue
            factor = cols[factor_i]
            label = cols[label_i] if label_i is not None and label_i < len(cols) and cols[label_i] else factor
            relevance = _safe_float(cols[relevance_i] if relevance_i is not None and relevance_i < len(cols) else None, 1.0)
            factors.append(FactorInfo(factor=factor, label=label, relevance=max(relevance, 0.0)))
    if not factors:
        _bail("No factors found in %s" % path)
    return factors


def _scale_entity_values(entities: list[EntityInfo]) -> list[EntityInfo]:
    if not entities:
        return entities
    max_combined = max([entity.combined for entity in entities] + [1.0])
    min_combined = 0.2 * max_combined
    scaled = []
    for entity in entities:
        combined = min(max(entity.combined, min_combined), max_combined) / max_combined
        direct = max(0.0, min(entity.direct, 1.0))
        scaled.append(
            EntityInfo(
                entity_id=entity.entity_id,
                label=entity.label,
                kind=entity.kind,
                combined=combined,
                direct=direct,
                loadings=entity.loadings,
                provenance=entity.provenance,
            )
        )
    return scaled


def _filter_entities_by_factor_rank(
    entities: list[EntityInfo],
    factors: list[str],
    *,
    min_loading: float,
    min_loading_frac: float,
    max_num_per_factor: int,
) -> list[EntityInfo]:
    filtered: list[EntityInfo] = []
    for entity in entities:
        max_loading = max([entity.loadings.get(factor, 0.0) for factor in factors] + [0.0])
        if max_loading <= min_loading:
            continue
        loadings = {
            factor: (value if value >= min_loading_frac * max_loading and value >= min_loading else 0.0)
            for factor, value in entity.loadings.items()
        }
        if max(loadings.values(), default=0.0) <= 0:
            continue
        filtered.append(
            EntityInfo(
                entity_id=entity.entity_id,
                label=entity.label,
                kind=entity.kind,
                combined=entity.combined,
                direct=entity.direct,
                loadings=loadings,
                provenance=entity.provenance,
            )
        )
    if max_num_per_factor is None or max_num_per_factor <= 0:
        return []
    keep: set[str] = set()
    for factor in factors:
        ranked = sorted(
            filtered,
            key=lambda entity: (-entity.loadings.get(factor, 0.0), entity.entity_id),
        )
        for entity in ranked[:max_num_per_factor]:
            if entity.loadings.get(factor, 0.0) > 0:
                keep.add(entity.entity_id)
    return [entity for entity in filtered if entity.entity_id in keep]


def _threshold_entity_loadings(entity: EntityInfo, *, min_loading: float, min_loading_frac: float) -> EntityInfo | None:
    max_loading = max(entity.loadings.values(), default=0.0)
    if max_loading <= min_loading:
        return None
    loadings = {
        factor: (value if value >= min_loading_frac * max_loading and value >= min_loading else 0.0)
        for factor, value in entity.loadings.items()
    }
    if max(loadings.values(), default=0.0) <= 0:
        return None
    return EntityInfo(
        entity_id=entity.entity_id,
        label=entity.label,
        kind=entity.kind,
        combined=entity.combined,
        direct=entity.direct,
        loadings=loadings,
        provenance=entity.provenance,
    )


def _read_wide_entities_raw(
    path: str | Path,
    *,
    kind: str,
    factors: list[str],
    id_col: str,
    label_col: str | None = None,
    combined_col: str | None = None,
    direct_col: str | None = None,
) -> list[EntityInfo]:
    with open_text(path) as fh:
        header_line = fh.readline()
        if not header_line:
            return []
        header, delim = _split_header(header_line)
        id_i = _get_col(header, id_col)
        label_i = _get_col(header, label_col, required=False) if label_col is not None else None
        combined_i = _get_col(header, combined_col, default="combined", required=False)
        direct_i = _get_col(header, direct_col, default="log_bf", required=False)
        factor_cols = {factor: _get_col(header, factor) for factor in factors}
        entities: list[EntityInfo] = []
        for line in fh:
            cols = line.rstrip("\n").split(delim)
            if id_i is None or id_i >= len(cols):
                continue
            entity_id = cols[id_i]
            label = cols[label_i] if label_i is not None and label_i < len(cols) and cols[label_i] else entity_id
            combined = _safe_float(cols[combined_i] if combined_i is not None and combined_i < len(cols) else None, 1.0)
            direct = _safe_float(cols[direct_i] if direct_i is not None and direct_i < len(cols) else None, combined)
            loadings = {
                factor: max(0.0, _safe_float(cols[col_i] if col_i is not None and col_i < len(cols) else None, 0.0))
                for factor, col_i in factor_cols.items()
            }
            provenance: dict[str, object] = {
                "source_table": f"{kind}_clusters",
                "source_id": entity_id,
                "source_fields": {
                    "id": id_col,
                    "combined": combined_col or "combined",
                    "direct": direct_col or "log_bf",
                    "loadings": "Factor* columns",
                },
                "support_summary": {
                    "combined": combined,
                    "direct": direct,
                },
            }
            entities.append(EntityInfo(entity_id=entity_id, label=label, kind=kind, combined=combined, direct=direct, loadings=loadings, provenance=provenance))
    return entities


def read_wide_entities(
    path: str | Path,
    *,
    kind: str,
    factors: list[str],
    id_col: str,
    label_col: str | None = None,
    combined_col: str | None = None,
    direct_col: str | None = None,
    min_loading: float,
    min_loading_frac: float,
    max_num_per_factor: int,
) -> list[EntityInfo]:
    entities = _read_wide_entities_raw(
        path,
        kind=kind,
        factors=factors,
        id_col=id_col,
        label_col=label_col,
        combined_col=combined_col,
        direct_col=direct_col,
    )
    entities = _filter_entities_by_factor_rank(
        entities,
        factors,
        min_loading=min_loading,
        min_loading_frac=min_loading_frac,
        max_num_per_factor=max_num_per_factor,
    )
    return _scale_entity_values(entities)


def read_wide_entity_candidates(
    path: str | Path,
    *,
    kind: str,
    factors: list[str],
    id_col: str,
    label_col: str | None = None,
    combined_col: str | None = None,
    direct_col: str | None = None,
    min_loading: float,
    min_loading_frac: float,
) -> list[EntityInfo]:
    raw_entities = _read_wide_entities_raw(
        path,
        kind=kind,
        factors=factors,
        id_col=id_col,
        label_col=label_col,
        combined_col=combined_col,
        direct_col=direct_col,
    )
    entities = [
        thresholded
        for entity in raw_entities
        for thresholded in [_threshold_entity_loadings(entity, min_loading=min_loading, min_loading_frac=min_loading_frac)]
        if thresholded is not None
    ]
    return _scale_entity_values(entities)


def read_trait_links(
    path: str | Path,
    *,
    factors: list[str],
    min_loading: float,
    min_neff: float | None,
    min_loading_frac: float,
    max_num_per_factor: int,
) -> list[EntityInfo]:
    by_trait: dict[str, dict[str, float]] = {}
    trait_strength: dict[str, float] = {}
    trait_neff: dict[str, float] = {}
    with open_text(path) as fh:
        header_line = fh.readline()
        if not header_line:
            return []
        header, delim = _split_header(header_line)
        trait_i = _get_col(header, "trait", required=False)
        if trait_i is None:
            trait_i = _get_col(header, "Pheno", required=False)
        factor_i = _get_col(header, "factor", required=False)
        if factor_i is None:
            factor_i = _get_col(header, "Factor", required=False)
        value_i = _get_col(header, "nnls_loading", required=False)
        if value_i is None:
            value_i = _get_col(header, "joint_fraction", required=False)
        if value_i is None:
            value_i = _get_col(header, "joint_coefficient", required=False)
        strength_i = _get_col(header, "trait_neff", required=False)
        if strength_i is None:
            strength_i = _get_col(header, "trait_n_eff", required=False)
        if trait_i is None or factor_i is None or value_i is None:
            return []
        for line in fh:
            cols = line.rstrip("\n").split(delim)
            if max(trait_i, factor_i, value_i) >= len(cols):
                continue
            trait = cols[trait_i]
            factor = cols[factor_i]
            if factor not in factors:
                continue
            value = max(0.0, _safe_float(cols[value_i], 0.0))
            by_trait.setdefault(trait, {})[factor] = value
            if strength_i is not None and strength_i < len(cols):
                neff = _safe_float(cols[strength_i], 1.0)
                trait_strength[trait] = max(trait_strength.get(trait, 0.0), neff)
                trait_neff[trait] = max(trait_neff.get(trait, 0.0), neff)
    if min_neff is not None and strength_i is not None:
        by_trait = {trait: loadings for trait, loadings in by_trait.items() if trait_neff.get(trait, 0.0) > min_neff}
    entities = [
        EntityInfo(
            entity_id=trait,
            label=trait,
            kind="trait",
            combined=trait_strength.get(trait, 1.0),
            direct=1.0,
            loadings={factor: loadings.get(factor, 0.0) for factor in factors},
            provenance={
                "source_table": "trait_factor_links",
                "source_id": trait,
                "source_fields": {
                    "anchor": "trait",
                    "loadings": "nnls_loading",
                    "cosine": "cosine_loading",
                    "euclidean": "euclidean_distance",
                    "effective_size": "trait_neff or trait_n_eff",
                },
                "support_summary": {
                    "trait_neff": trait_neff.get(trait),
                },
            },
        )
        for trait, loadings in by_trait.items()
    ]
    entities = _filter_entities_by_factor_rank(
        entities,
        factors,
        min_loading=min_loading,
        min_loading_frac=min_loading_frac,
        max_num_per_factor=max_num_per_factor,
    )
    return _scale_entity_values(entities)


def read_anchor_support_rows(
    paths: list[str] | None,
    *,
    id_col: str,
    anchor_col: str,
    combined_col: str | None,
    direct_col: str | None,
    indirect_col: str | None,
    source_label: str,
    max_rows_per_node: int = 20,
    min_combined: float = 0.0,
) -> dict[str, list[dict[str, object]]]:
    support: dict[str, list[dict[str, object]]] = {}
    for path in paths or []:
        with open_text(path) as fh:
            header_line = fh.readline()
            if not header_line:
                continue
            header, delim = _split_header(header_line)
            id_i = _get_col(header, id_col)
            anchor_i = _get_col(header, anchor_col)
            combined_i = _get_col(header, combined_col, required=False) if combined_col else None
            direct_i = _get_col(header, direct_col, required=False) if direct_col else None
            indirect_i = _get_col(header, indirect_col, required=False) if indirect_col else None
            for line in fh:
                cols = line.rstrip("\n").split(delim)
                if id_i is None or anchor_i is None or max(id_i, anchor_i) >= len(cols):
                    continue
                entity_id = cols[id_i]
                anchor = cols[anchor_i]
                combined = _safe_float(cols[combined_i], 0.0) if combined_i is not None and combined_i < len(cols) else None
                direct = _safe_float(cols[direct_i], 0.0) if direct_i is not None and direct_i < len(cols) else None
                indirect = _safe_float(cols[indirect_i], 0.0) if indirect_i is not None and indirect_i < len(cols) else None
                if min_combined is not None and float(min_combined) > 0.0 and (combined is None or combined < float(min_combined)):
                    continue
                row = {
                    "anchor": anchor,
                    "combined": combined,
                    "direct": direct,
                    "indirect": indirect,
                    "source": source_label,
                    "source_fields": {
                        "combined": combined_col,
                        "direct": direct_col,
                        "indirect": indirect_col,
                    },
                }
                support.setdefault(entity_id, []).append(row)
    for entity_id, rows in list(support.items()):
        rows.sort(
            key=lambda row: (
                -float(row.get("combined") or 0.0),
                -float(row.get("direct") or 0.0),
                str(row.get("anchor", "")),
            )
        )
        if max_rows_per_node is not None and int(max_rows_per_node) >= 0:
            support[entity_id] = rows[: int(max_rows_per_node)]
    return support


def attach_anchor_support(entities: list[EntityInfo], support_by_id: dict[str, list[dict[str, object]]]) -> list[EntityInfo]:
    if not support_by_id:
        return entities
    updated: list[EntityInfo] = []
    for entity in entities:
        provenance = dict(entity.provenance)
        rows = support_by_id.get(entity.entity_id, [])
        if rows:
            provenance["anchor_support"] = rows
        updated.append(replace(entity, provenance=provenance))
    return updated


def _passes_trait_factor_detail_filters(row: dict[str, object], *, min_beta: float, min_beta_uncorrected: float, min_nnls: float) -> bool:
    active = [
        ("beta", min_beta),
        ("beta_uncorrected", min_beta_uncorrected),
        ("nnls_loading", min_nnls),
    ]
    active = [(key, float(threshold)) for key, threshold in active if threshold is not None and float(threshold) > 0.0]
    if not active:
        return True
    for key, threshold in active:
        try:
            value = float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > threshold:
            return True
    return False


def _trait_factor_rank_value(row: dict[str, object], field: str) -> float:
    if field == "nnls":
        field = "nnls_loading"
    value = row.get(field)
    try:
        return float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0


def _read_factor_trait_enrichment_details(
    path: str | Path | None,
    factors: list[str],
) -> dict[tuple[str, str], dict[str, object]]:
    if path is None:
        return {}
    rows: dict[tuple[str, str], dict[str, object]] = {}
    with open_text(path) as fh:
        header_line = fh.readline()
        if not header_line:
            return rows
        header, delim = _split_header(header_line)
        trait_i = _get_col(header, "trait", required=False)
        if trait_i is None:
            trait_i = _get_col(header, "Trait", required=False)
        if trait_i is None:
            trait_i = _get_col(header, "Trait_Internal", required=False)
        if trait_i is None:
            trait_i = _get_col(header, "Pheno", required=False)
        factor_i = _get_col(header, "factor", required=False)
        if factor_i is None:
            factor_i = _get_col(header, "Factor", required=False)
        if factor_i is None:
            factor_i = _get_col(header, "Gene_Set", required=False)
        if factor_i is None:
            factor_i = _get_col(header, "gene_set", required=False)
        if trait_i is None or factor_i is None:
            return rows
        numeric_fields = ["beta", "beta_uncorrected", "beta_tilde", "beta_tilde_internal", "se", "se_internal", "SE", "z", "Z", "p_value", "p", "p_orig", "P"]
        numeric_indices = {field_name: _get_col(header, field_name, required=False) for field_name in numeric_fields}
        field_map = {
            "beta_tilde_internal": "beta_tilde",
            "se_internal": "se",
            "SE": "se",
            "Z": "z",
            "p": "p_value",
            "p_orig": "p_value",
            "P": "p_value",
        }
        for line in fh:
            cols = line.rstrip("\n").split(delim)
            if max(trait_i, factor_i) >= len(cols):
                continue
            trait = cols[trait_i]
            factor = cols[factor_i]
            if factor not in factors:
                continue
            row: dict[str, object] = {"anchor": trait, "source_table": "factor_trait_pigean_enrichments"}
            for field_name, idx in numeric_indices.items():
                if idx is not None and idx < len(cols):
                    row[field_map.get(field_name, field_name)] = _safe_float(cols[idx], 0.0)
            rows[(trait, factor)] = row
    return rows


def read_factor_trait_details(
    path: str | Path | None,
    factors: list[str],
    *,
    enrichment_path: str | Path | None = None,
    max_num_per_factor: int = 20,
    min_beta: float = 0.01,
    min_beta_uncorrected: float = 0.05,
    min_nnls: float = 0.5,
    rank_field: str = "beta",
) -> dict[str, list[dict[str, object]]]:
    allowed_rank_fields = {"beta", "beta_uncorrected", "nnls", "nnls_loading"}
    if rank_field not in allowed_rank_fields:
        raise ValueError("trait-factor rank field must be one of: beta, beta_uncorrected, nnls")
    enrichment_by_key = _read_factor_trait_enrichment_details(enrichment_path, factors)
    if path is None and not enrichment_by_key:
        return {}
    details: dict[str, list[dict[str, object]]] = {factor: [] for factor in factors}
    seen: set[tuple[str, str]] = set()
    if path is None:
        for (trait, factor), row in enrichment_by_key.items():
            if _passes_trait_factor_detail_filters(row, min_beta=min_beta, min_beta_uncorrected=min_beta_uncorrected, min_nnls=min_nnls):
                details[factor].append(row)
        path = None
    if path is None:
        for factor, rows in details.items():
            rows.sort(key=lambda row: (-_trait_factor_rank_value(row, rank_field), str(row.get("anchor", ""))))
            if max_num_per_factor is not None and int(max_num_per_factor) >= 0:
                details[factor] = rows[: int(max_num_per_factor)]
        return details
    with open_text(path) as fh:
        header_line = fh.readline()
        if not header_line:
            return details
        header, delim = _split_header(header_line)
        trait_i = _get_col(header, "trait", required=False)
        if trait_i is None:
            trait_i = _get_col(header, "Pheno", required=False)
        factor_i = _get_col(header, "factor", required=False)
        if factor_i is None:
            factor_i = _get_col(header, "Factor", required=False)
        if trait_i is None or factor_i is None:
            return details
        numeric_fields = [
            "nnls_loading",
            "cosine_loading",
            "euclidean_distance",
            "beta",
            "beta_uncorrected",
            "beta_tilde",
            "se",
            "z",
            "p_value",
            "joint_fraction",
            "joint_coefficient",
            "marginal_fraction",
            "marginal_coefficient",
            "marginal_overlap",
            "trait_neff",
            "trait_n_eff",
            "joint_coefficient_support_mass",
            "marginal_coefficient_support_mass",
            "retained_fraction",
            "joint_residual",
        ]
        numeric_indices = {field_name: _get_col(header, field_name, required=False) for field_name in numeric_fields}
        string_fields = ["is_anchor", "score_source", "basis", "trait_response_source", "factor_gene_basis"]
        string_indices = {field_name: _get_col(header, field_name, required=False) for field_name in string_fields}
        for line in fh:
            cols = line.rstrip("\n").split(delim)
            if max(trait_i, factor_i) >= len(cols):
                continue
            factor = cols[factor_i]
            if factor not in details:
                continue
            trait = cols[trait_i]
            seen.add((trait, factor))
            row: dict[str, object] = {"anchor": cols[trait_i], "source_table": "trait_factor_links"}
            for field_name, idx in numeric_indices.items():
                if idx is not None and idx < len(cols):
                    row[field_name] = _safe_float(cols[idx], 0.0)
            for field_name, idx in string_indices.items():
                if idx is not None and idx < len(cols):
                    row[field_name] = cols[idx]
            row.update(enrichment_by_key.get((trait, factor), {}))
            row["anchor"] = trait
            row["source_table"] = "trait_factor_links+factor_trait_pigean_enrichments" if (trait, factor) in enrichment_by_key else "trait_factor_links"
            if "nnls_loading" in row:
                row.setdefault("joint_fraction", row["nnls_loading"])
                row.setdefault("joint_coefficient", row["nnls_loading"])
            if not _passes_trait_factor_detail_filters(
                row,
                min_beta=min_beta,
                min_beta_uncorrected=min_beta_uncorrected,
                min_nnls=min_nnls,
            ):
                continue
            details[factor].append(row)
    for (trait, factor), row in enrichment_by_key.items():
        if (trait, factor) in seen:
            continue
        if not _passes_trait_factor_detail_filters(row, min_beta=min_beta, min_beta_uncorrected=min_beta_uncorrected, min_nnls=min_nnls):
            continue
        details[factor].append(row)
    for factor, rows in details.items():
        rows.sort(
            key=lambda row: (
                -_trait_factor_rank_value(row, rank_field),
                -float(row.get("nnls_loading", row.get("joint_fraction", row.get("joint_coefficient", 0.0))) or 0.0),
                -float(row.get("beta_uncorrected", 0.0) or 0.0),
                -float(row.get("beta", 0.0) or 0.0),
                str(row.get("anchor", "")),
            )
        )
        if max_num_per_factor is not None and int(max_num_per_factor) >= 0:
            details[factor] = rows[: int(max_num_per_factor)]
    return details


def _top_loadings_by_factor(entities: list[EntityInfo], factors: list[str], *, top_n: int = 5) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for factor in factors:
        ranked = sorted(
            (
                {
                    "id": entity.entity_id,
                    "label": entity.label,
                    "kind": entity.kind,
                    "loading": float(entity.loadings.get(factor, 0.0)),
                    "source_table": f"{entity.kind}_clusters",
                    "source_field": factor,
                }
                for entity in entities
                if entity.loadings.get(factor, 0.0) > 0
            ),
            key=lambda row: (-float(row["loading"]), str(row["id"])),
        )
        result[factor] = ranked[:top_n]
    return result


def attach_near_top_factor_loadings(
    entities: list[EntityInfo],
    factors: list[str],
    *,
    factor_labels: dict[str, str] | None = None,
    label_max_chars: int | None = None,
    within_top: float = 0.01,
) -> list[EntityInfo]:
    factor_labels = factor_labels or {}
    updated: list[EntityInfo] = []
    for entity in entities:
        max_loading = max([entity.loadings.get(factor, 0.0) for factor in factors] + [0.0])
        threshold = max(0.0, max_loading - within_top)
        near_top = [
            {
                "factor": factor,
                "factor_label": factor_labels.get(factor, factor),
                "factor_display_label": truncate_label(factor_labels.get(factor, factor), label_max_chars),
                "loading": float(entity.loadings.get(factor, 0.0)),
                "source_field": factor,
            }
            for factor in sorted(factors, key=_factor_sort_key)
            if entity.loadings.get(factor, 0.0) > 0 and entity.loadings.get(factor, 0.0) >= threshold
        ]
        provenance = dict(entity.provenance)
        provenance["near_top_factor_loadings"] = near_top
        provenance["near_top_factor_loading_rule"] = f"loading >= max_loading - {within_top:g}"
        updated.append(replace(entity, provenance=provenance))
    return updated


def generate_distinct_colors(n: int, *, start_with_red_blue: bool = False) -> list[tuple[float, float, float]]:
    if n <= 0:
        return []
    if start_with_red_blue and n >= 2:
        colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        import colorsys

        for _ in range(2, n):
            best_color = (0.0, 0.0, 0.0)
            best_distance = -1.0
            for h in range(0, 360, 10):
                for s in [0.7, 1.0]:
                    for v in [0.7, 1.0]:
                        candidate = colorsys.hsv_to_rgb(h / 360.0, s, v)
                        distance = min(sum((a - b) ** 2 for a, b in zip(candidate, color)) ** 0.5 for color in colors)
                        if distance > best_distance:
                            best_distance = distance
                            best_color = candidate
            colors.append(best_color)
        return colors
    colors = [_hex_to_rgb(value) for value in DEFAULT_DISTINCT_PALETTE[: min(n, len(DEFAULT_DISTINCT_PALETTE))]]
    if n <= len(colors):
        return colors
    import colorsys

    for i in range(len(colors), n):
        hue = (0.5 + i * 0.618033988749895) % 1.0
        saturation = 0.64 if i % 2 else 0.78
        value = 0.72 if i % 3 else 0.86
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colors


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def blend_colors(colors: list[tuple[float, float, float]], weights: Iterable[float], *, opacity: float = 1.0) -> tuple[float, float, float]:
    weights_array = np.asarray(list(weights), dtype=float)
    if weights_array.size == 0 or float(np.sum(weights_array)) <= 0:
        weights_array = np.ones(len(colors), dtype=float)
    weights_array = weights_array / np.sum(weights_array)
    opacity = min(max(float(opacity), 0.1), 1.0)
    color_array = np.asarray(colors, dtype=float)
    blended = 1 - opacity * np.average(1 - color_array, axis=0, weights=weights_array)
    blended = np.clip(blended, 0, 1)
    return tuple(float(value) for value in blended)


def rgb_to_hex(rgb: Iterable[float], alpha: float | None = None) -> str:
    values = [max(0, min(255, int(float(value) * 255))) for value in rgb]
    result = "#%02x%02x%02x" % tuple(values[:3])
    if alpha is not None and alpha < 1:
        result += "%02x" % max(0, min(255, int(alpha * 255)))
    return result


def truncate_label(label: str, max_chars: int | None) -> str:
    label = str(label)
    if max_chars is None or max_chars <= 0 or len(label) <= max_chars:
        return label
    if max_chars <= 3:
        return "." * max_chars
    return label[: max_chars - 3] + "..."


def _classical_mds(distance: np.ndarray, *, seed: int = 0) -> np.ndarray:
    n = distance.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if n == 1:
        return np.zeros((1, 2), dtype=float)
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h.dot(distance ** 2).dot(h)
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order[:2]], 0.0)
    vecs = vecs[:, order[:2]]
    coords = vecs.dot(np.diag(np.sqrt(vals)))
    if coords.shape[1] < 2 or np.max(np.abs(coords)) == 0:
        rng = random.Random(seed)
        angles = [2 * math.pi * (i / n) + rng.random() * 1e-6 for i in range(n)]
        coords = np.asarray([[math.cos(angle), math.sin(angle)] for angle in angles], dtype=float)
    return coords


def compute_layout(matrix: np.ndarray, *, coordinate_scale: float, seed: int = 0) -> np.ndarray:
    n = matrix.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] == 1:
        values = matrix[:, 0]
        distance = np.abs(values[:, None] - values[None, :])
    else:
        jitter = np.random.default_rng(seed).normal(0, 1e-8, size=matrix.shape)
        corr = np.corrcoef(matrix + jitter)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        distance = 1 - corr
    distance = np.clip(distance, 0.01, 2.0)
    np.fill_diagonal(distance, 0.0)
    coords = _classical_mds(distance, seed=seed)
    for axis in [0, 1]:
        max_abs = float(np.max(np.abs(coords[:, axis]))) if coords.shape[0] else 0.0
        if max_abs > 0:
            coords[:, axis] = coords[:, axis] / max_abs
    return coords * coordinate_scale


def _reorder_colors_by_factor_correlation(colors: list[tuple[float, float, float]], matrix: np.ndarray) -> list[tuple[float, float, float]]:
    if len(colors) <= 1 or matrix.shape[0] < 2:
        return colors
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    indices = [0]
    corr[:, 0] = -np.inf
    for _ in range(1, len(colors)):
        next_factor = int(np.argmax(corr[indices[-1], :]))
        corr[:, next_factor] = -np.inf
        indices.append(next_factor)
    reordered = [None] * len(colors)
    for color_index, factor_index in enumerate(indices):
        reordered[factor_index] = colors[color_index]
    return [color if color is not None else colors[i] for i, color in enumerate(reordered)]


def _place_traits_near_top_factors(
    coords: np.ndarray,
    entities: list[EntityInfo],
    factors: list[str],
    factor_coords: np.ndarray,
    *,
    trait_coordinate_scale: float,
    min_centroid_distance_frac: float,
) -> None:
    if not len(entities) or not len(factors) or factor_coords.size == 0:
        return
    factor_center = np.mean(factor_coords, axis=0)
    factor_distances = np.sqrt(np.sum(np.square(factor_coords - factor_center), axis=1))
    min_distance = max(0.0, float(min_centroid_distance_frac)) * (float(np.median(factor_distances)) if len(factor_distances) else 0.0)
    scale = max(0.0, min(float(trait_coordinate_scale), 1.0))
    for entity_index, entity in enumerate(entities):
        if entity.kind != "trait":
            continue
        weights = np.asarray([max(0.0, entity.loadings.get(factor, 0.0)) for factor in factors], dtype=float)
        if not np.any(weights > 0):
            continue
        ranked = np.argsort(weights)[::-1]
        support = ranked[: min(2, len(ranked))]
        support = support[weights[support] > 0]
        if len(support) == 0:
            continue
        local_weights = weights[support]
        local_weights = local_weights / max(float(np.sum(local_weights)), 1e-12)
        anchor = np.sum(factor_coords[support, :] * local_weights[:, np.newaxis], axis=0)
        # Keep some MDS signal, but make the top-factor mixture the stable anchor.
        coords[entity_index] = anchor + scale * (coords[entity_index] - anchor)
        offset = coords[entity_index] - factor_center
        distance = float(np.sqrt(np.sum(np.square(offset))))
        if min_distance > 0 and distance < min_distance:
            if distance <= 1e-12:
                angle = 2.0 * math.pi * (int(support[0]) / max(len(factors), 1))
                offset = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
                distance = 1.0
            coords[entity_index] = factor_center + offset * (min_distance / distance)


def build_graph(
    factors_info: list[FactorInfo],
    genes: list[EntityInfo],
    traits: list[EntityInfo],
    config: GraphConfig,
    *,
    candidate_genes: list[EntityInfo] | None = None,
    candidate_traits: list[EntityInfo] | None = None,
    factor_trait_details: dict[str, list[dict[str, object]]] | None = None,
    factor_trait_color_details: dict[str, list[dict[str, object]]] | None = None,
    top_gene_loadings_by_factor: dict[str, list[dict[str, object]]] | None = None,
    top_gene_set_loadings_by_factor: dict[str, list[dict[str, object]]] | None = None,
) -> dict:
    factors = [factor.factor for factor in factors_info]
    factor_to_label = {factor.factor: factor.label for factor in factors_info}
    max_relevance = max([factor.relevance for factor in factors_info] + [1.0])
    factor_trait_details = factor_trait_details or {}
    factor_trait_color_details = factor_trait_color_details or factor_trait_details
    top_gene_loadings_by_factor = top_gene_loadings_by_factor or {}
    top_gene_set_loadings_by_factor = top_gene_set_loadings_by_factor or {}
    entities = genes + traits
    entity_matrix = np.asarray([[entity.loadings.get(factor, 0.0) for factor in factors] for entity in entities], dtype=float)
    factor_identity = np.eye(len(factors), dtype=float)
    layout_matrix = np.vstack([entity_matrix, factor_identity]) if len(entities) else factor_identity
    color_by = str(config.color_by)
    if color_by not in {"auto", "factor", "trait"}:
        _bail("--color-by must be one of: auto, factor, trait")
    detail_anchor_names: list[str] = []
    for rows in factor_trait_color_details.values():
        for row in rows:
            is_anchor = str(row.get("is_anchor", "")).strip().lower() in {"1", "true", "yes"}
            anchor = str(row.get("anchor", ""))
            if is_anchor and anchor and anchor not in detail_anchor_names:
                detail_anchor_names.append(anchor)
    configured_anchor_names = list(config.anchor_trait_names)
    anchor_names_for_coloring = configured_anchor_names or detail_anchor_names or [trait.entity_id for trait in traits]
    use_trait_weight_colors = bool(anchor_names_for_coloring and (color_by == "trait" or (color_by == "auto" and config.multi_anchor)))
    colors = generate_distinct_colors(len(factors), start_with_red_blue=config.colors_red_blue)
    trait_colors_by_id: dict[str, tuple[float, float, float]] = {}
    if use_trait_weight_colors:
        trait_colors = generate_distinct_colors(len(anchor_names_for_coloring), start_with_red_blue=config.colors_red_blue)
        trait_colors_by_id = {anchor: trait_colors[i] for i, anchor in enumerate(anchor_names_for_coloring)}
        colors = []
        for factor in factors:
            raw_by_anchor = {str(row.get("anchor", "")): float(row.get("joint_fraction", row.get("joint_coefficient", 0.0)) or 0.0) for row in factor_trait_color_details.get(factor, [])}
            factor_trait_weights = [raw_by_anchor.get(anchor, 0.0) for anchor in anchor_names_for_coloring]
            if not any(weight > 0 for weight in factor_trait_weights) and not configured_anchor_names:
                factor_trait_weights = [trait.loadings.get(factor, 0.0) for trait in traits]
            colors.append(blend_colors(trait_colors, factor_trait_weights, opacity=1.0))
    elif entity_matrix.size:
        colors = _reorder_colors_by_factor_correlation(colors, entity_matrix)
    coords = compute_layout(layout_matrix, coordinate_scale=config.coordinate_scale, seed=config.seed)
    if traits and config.trait_layout_mode == "anchored_top_factor":
        _place_traits_near_top_factors(
            coords,
            entities,
            factors,
            coords[len(entities) :],
            trait_coordinate_scale=config.trait_coordinate_scale,
            min_centroid_distance_frac=config.trait_min_centroid_distance_frac,
        )
    elif traits and 0 <= config.trait_coordinate_scale < 1:
        factor_coords = coords[len(entities) :]
        factor_center = np.mean(factor_coords, axis=0) if factor_coords.size else np.zeros(2, dtype=float)
        for entity_index, entity in enumerate(entities):
            if entity.kind == "trait":
                coords[entity_index] = factor_center + config.trait_coordinate_scale * (coords[entity_index] - factor_center)
    nodes = []
    edges = []
    selected_ids = {entity.entity_id for entity in entities}
    factor_offset = len(entities)
    for factor_index, factor_info in enumerate(factors_info):
        color = colors[factor_index]
        x, y = coords[factor_offset + factor_index]
        relevance = min(max(factor_info.relevance / max_relevance, 0.3), 1.0) if max_relevance > 0 else 1.0
        nodes.append(
            {
                "id": factor_info.factor,
                "label": factor_info.label,
                "display_label": truncate_label(factor_info.label, config.label_max_chars),
                "kind": "factor",
                "shape": "square",
                "x": float(x),
                "y": float(y),
                "size": float(36 + 44 * relevance * config.node_size_scale),
                "color": rgb_to_hex(color),
                "border_color": rgb_to_hex(color),
                "relevance": float(factor_info.relevance),
                "provenance": {
                    "source_table": "factors",
                    "relevance": float(factor_info.relevance),
                    "relevance_by_anchor": factor_trait_details.get(factor_info.factor, []),
                    "top_gene_loadings": top_gene_loadings_by_factor.get(factor_info.factor, []),
                    "top_gene_set_loadings": top_gene_set_loadings_by_factor.get(factor_info.factor, []),
                },
            }
        )
    for entity_index, entity in enumerate(entities):
        weights = [entity.loadings.get(factor, 0.0) for factor in factors]
        if entity.kind == "trait" and entity.entity_id in trait_colors_by_id:
            color = trait_colors_by_id[entity.entity_id]
            border = color
        else:
            color = blend_colors(colors, weights, opacity=entity.direct)
            border = blend_colors(colors, weights, opacity=1.0)
        x, y = coords[entity_index]
        nodes.append(
            {
                "id": entity.entity_id,
                "label": entity.label,
                "display_label": truncate_label(entity.label, config.label_max_chars),
                "kind": entity.kind,
                "shape": "diamond" if entity.kind == "trait" else "circle",
                "x": float(x),
                "y": float(y),
                "size": float(18 + 42 * entity.combined * config.node_size_scale * (0.7 if entity.kind == "trait" else 1.0)),
                "color": rgb_to_hex(color),
                "border_color": rgb_to_hex(border),
                "combined_scaled": float(entity.combined),
                "direct_scaled": float(entity.direct),
                "provenance": entity.provenance,
            }
        )
        for factor, loading in sorted(entity.loadings.items(), key=lambda item: _factor_sort_key(item[0])):
            threshold = config.trait_min_loading if entity.kind == "trait" else config.gene_min_loading
            if loading < threshold:
                continue
            edges.append(
                {
                    "from": factor,
                    "to": entity.entity_id,
                    "kind": "factor_trait" if entity.kind == "trait" else "factor_gene",
                    "weight": float(loading),
                    "width": float(max(0.5, config.edge_max_width * loading)),
                    "color": rgb_to_hex(trait_colors_by_id[entity.entity_id])
                    if entity.kind == "trait" and entity.entity_id in trait_colors_by_id
                    else nodes[factors.index(factor)]["color"],
                    "provenance": {
                        "source_table": "trait_factor_links" if entity.kind == "trait" else "gene_clusters",
                        "weight": float(loading),
                        "weight_field": "joint_fraction/joint_coefficient" if entity.kind == "trait" else factor,
                        "threshold_fields": {
                            "min_loading": config.trait_min_loading if entity.kind == "trait" else config.gene_min_loading,
                            "min_loading_frac": config.trait_min_loading_frac if entity.kind == "trait" else config.gene_min_loading_frac,
                        },
                    },
                }
            )
    candidate_nodes = []
    candidate_edges = []
    factor_coord_by_id = {node["id"]: (float(node["x"]), float(node["y"])) for node in nodes if node["kind"] == "factor"}
    candidate_entities = []
    if candidate_genes is not None:
        candidate_entities.extend(candidate_genes)
    if candidate_traits is not None:
        candidate_entities.extend(candidate_traits)
    for entity in candidate_entities:
        if entity.entity_id in selected_ids:
            continue
        weights = [entity.loadings.get(factor, 0.0) for factor in factors]
        positive = [(factor, weight) for factor, weight in zip(factors, weights) if weight > 0]
        if not positive:
            continue
        weight_sum = sum(weight for _, weight in positive)
        if weight_sum <= 0:
            continue
        x = sum(factor_coord_by_id[factor][0] * weight for factor, weight in positive) / weight_sum
        y = sum(factor_coord_by_id[factor][1] * weight for factor, weight in positive) / weight_sum
        jitter_seed = sum(ord(ch) for ch in entity.entity_id) % 360
        jitter = 0.18 + 0.06 * (len(candidate_nodes) % 5)
        x += jitter * math.cos(math.radians(jitter_seed))
        y += jitter * math.sin(math.radians(jitter_seed))
        if entity.kind == "trait" and entity.entity_id in trait_colors_by_id:
            color = trait_colors_by_id[entity.entity_id]
            border = color
        else:
            color = blend_colors(colors, weights, opacity=entity.direct)
            border = blend_colors(colors, weights, opacity=1.0)
        candidate_nodes.append(
            {
                "id": entity.entity_id,
                "label": entity.label,
                "display_label": truncate_label(entity.label, config.label_max_chars),
                "kind": entity.kind,
                "shape": "diamond" if entity.kind == "trait" else "circle",
                "x": float(x),
                "y": float(y),
                "size": float(18 + 42 * entity.combined * config.node_size_scale * (0.7 if entity.kind == "trait" else 1.0)),
                "color": rgb_to_hex(color),
                "border_color": rgb_to_hex(border),
                "combined_scaled": float(entity.combined),
                "direct_scaled": float(entity.direct),
                "provenance": entity.provenance,
            }
        )
        for factor, loading in sorted(entity.loadings.items(), key=lambda item: _factor_sort_key(item[0])):
            threshold = config.trait_min_loading if entity.kind == "trait" else config.gene_min_loading
            if loading < threshold:
                continue
            candidate_edges.append(
                {
                    "from": factor,
                    "to": entity.entity_id,
                    "kind": "factor_trait" if entity.kind == "trait" else "factor_gene",
                    "weight": float(loading),
                    "width": float(max(0.5, config.edge_max_width * loading)),
                    "color": rgb_to_hex(trait_colors_by_id[entity.entity_id])
                    if entity.kind == "trait" and entity.entity_id in trait_colors_by_id
                    else nodes[factors.index(factor)]["color"],
                    "provenance": {
                        "source_table": "trait_factor_links" if entity.kind == "trait" else "gene_clusters",
                        "weight": float(loading),
                        "weight_field": "joint_fraction/joint_coefficient" if entity.kind == "trait" else factor,
                        "threshold_fields": {
                            "min_loading": config.trait_min_loading if entity.kind == "trait" else config.gene_min_loading,
                            "min_loading_frac": config.trait_min_loading_frac if entity.kind == "trait" else config.gene_min_loading_frac,
                        },
                    },
                }
            )
    return {
        "schema": "eaggl_factor_graph/v1",
        "layout": {
            "trait_coordinate_scale": float(config.trait_coordinate_scale),
            "trait_layout_mode": str(config.trait_layout_mode),
            "trait_min_centroid_distance_frac": float(config.trait_min_centroid_distance_frac),
            "trait_edge_length_scale": float(config.trait_edge_length_scale),
        },
        "coloring": {
            "color_by": color_by,
            "resolved_color_by": "trait" if use_trait_weight_colors else "factor",
            "multi_anchor": bool(config.multi_anchor),
            "trait_count_for_coloring": int(len(anchor_names_for_coloring) if use_trait_weight_colors else 0),
            "anchor_traits_for_coloring": list(anchor_names_for_coloring) if use_trait_weight_colors else [],
            "trait_color_weight_source": "params_anchor_trait_names" if use_trait_weight_colors and configured_anchor_names else ("trait_factor_links_unfiltered" if use_trait_weight_colors and factor_trait_color_details else "visible_trait_nodes"),
        },
        "factors": factors,
        "factor_labels": [factor_to_label[factor] for factor in factors],
        "nodes": nodes,
        "edges": edges,
        "candidate_nodes": candidate_nodes,
        "candidate_edges": candidate_edges,
    }


def write_json(graph: dict, path: str | Path) -> None:
    with open_text(path, "wt") as fh:
        json.dump(graph, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _scaled_graph_for_html(graph: dict, *, width: int = 1200, height: int = 900) -> dict:
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not nodes:
        _bail("Cannot write graph HTML with no nodes")
    min_x = min(node["x"] for node in nodes)
    max_x = max(node["x"] for node in nodes)
    min_y = min(node["y"] for node in nodes)
    max_y = max(node["y"] for node in nodes)
    pad = 80

    def sx(x: float) -> float:
        denom = max(max_x - min_x, 1e-9)
        return pad + (x - min_x) / denom * (width - 2 * pad)

    def sy(y: float) -> float:
        denom = max(max_y - min_y, 1e-9)
        return height - (pad + (y - min_y) / denom * (height - 2 * pad))

    def scale_node(node: dict) -> dict:
        scaled_node = dict(node)
        scaled_node["fixed_x"] = sx(node["x"])
        scaled_node["fixed_y"] = sy(node["y"])
        scaled_node["x"] = scaled_node["fixed_x"]
        scaled_node["y"] = scaled_node["fixed_y"]
        scaled_node["radius"] = max(6.0, float(node["size"]) / 5.0)
        return scaled_node

    scaled_nodes = []
    for node in nodes:
        scaled_node = scale_node(node)
        scaled_nodes.append(scaled_node)
    return {
        **graph,
        "nodes": scaled_nodes,
        "edges": [dict(edge) for edge in edges],
        "candidate_nodes": [scale_node(node) for node in graph.get("candidate_nodes", [])],
        "candidate_edges": [dict(edge) for edge in graph.get("candidate_edges", [])],
        "viewport": {"width": width, "height": height},
    }


def _shape_svg(node: dict) -> str:
    x = float(node["x"])
    y = float(node["y"])
    radius = float(node["radius"])
    title = html.escape("%s: %s" % (node["kind"], node["id"]))
    if node["shape"] == "square":
        size = radius * 2
        return '<rect x="%.3f" y="%.3f" width="%.3f" height="%.3f" rx="4" fill="%s" stroke="%s" stroke-width="3"><title>%s</title></rect>' % (
            x - radius,
            y - radius,
            size,
            size,
            node["color"],
            node["border_color"],
            title,
        )
    if node["shape"] == "diamond":
        points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
        return '<polygon points="%s" fill="%s" stroke="%s" stroke-width="3"><title>%s</title></polygon>' % (
            " ".join("%.3f,%.3f" % point for point in points),
            node["color"],
            node["border_color"],
            title,
        )
    return '<circle cx="%.3f" cy="%.3f" r="%.3f" fill="%s" stroke="%s" stroke-width="3"><title>%s</title></circle>' % (
        x,
        y,
        radius,
        node["color"],
        node["border_color"],
        title,
    )


def _static_svg_markup(graph: dict) -> str:
    nodes = graph["nodes"]
    edges = graph["edges"]
    node_by_id = {node["id"]: node for node in nodes}
    svg_parts: list[str] = []
    for edge in edges:
        source = node_by_id.get(edge["from"])
        target = node_by_id.get(edge["to"])
        if source is None or target is None:
            continue
        svg_parts.append(
            '<line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-opacity="0.45" stroke-width="%.3f" />'
            % (source["x"], source["y"], target["x"], target["y"], edge["color"], edge["width"])
        )
    for node in nodes:
        label = html.escape(str(node.get("display_label", node["label"])))
        svg_parts.append(_shape_svg(node))
        if node["kind"] == "factor" or node["radius"] > 9:
            svg_parts.append(
                '<text x="%.3f" y="%.3f" text-anchor="middle" font-size="12" font-family="sans-serif">%s</text>'
                % (node["x"], node["y"] + node["radius"] + 14, label)
            )
    return "\n    ".join(svg_parts)


def _interactive_html_script(*, physics_enabled: bool) -> str:
    return r"""
<script>
(function() {
  const graph = JSON.parse(document.getElementById("eaggl-factor-graph-data").textContent);
  const svg = document.getElementById("graph-svg");
  const edgesLayer = document.getElementById("edges-layer");
  const nodesLayer = document.getElementById("nodes-layer");
  const labelsLayer = document.getElementById("labels-layer");
  const tooltip = document.getElementById("tooltip");
  const nodeById = new Map(graph.nodes.map(node => [node.id, node]));
  const candidateNodeById = new Map((graph.candidate_nodes || []).map(node => [node.id, node]));
  const neighborsByNode = new Map();
  const incidentEdgesByNode = new Map();
  let hoveredNodeId = null;
  for (const node of graph.nodes) {
    neighborsByNode.set(node.id, new Set());
    incidentEdgesByNode.set(node.id, new Set());
  }
  graph.edges.forEach((edge, index) => {
    edge._index = index;
    if (!neighborsByNode.has(edge.from)) neighborsByNode.set(edge.from, new Set());
    if (!neighborsByNode.has(edge.to)) neighborsByNode.set(edge.to, new Set());
    if (!incidentEdgesByNode.has(edge.from)) incidentEdgesByNode.set(edge.from, new Set());
    if (!incidentEdgesByNode.has(edge.to)) incidentEdgesByNode.set(edge.to, new Set());
    neighborsByNode.get(edge.from).add(edge.to);
    neighborsByNode.get(edge.to).add(edge.from);
    incidentEdgesByNode.get(edge.from).add(index);
    incidentEdgesByNode.get(edge.to).add(index);
  });
  const candidateEdgesByNode = new Map();
  for (const edge of (graph.candidate_edges || [])) {
    if (!candidateEdgesByNode.has(edge.to)) candidateEdgesByNode.set(edge.to, []);
    candidateEdgesByNode.get(edge.to).push(edge);
  }
  const activeNodeTypes = new Set(["factor", "gene", "trait"]);
  const textFilters = [];
  let hideUnmatched = false;
  let highlightNeighbors = false;
  let physicsEnabled = __PHYSICS_ENABLED__;
  let running = physicsEnabled;
  let viewBox = {x: 0, y: 0, w: graph.viewport.width, h: graph.viewport.height};
  let draggingNode = null;
  let draggingCanvas = null;
  let lastPointer = null;

  function setViewBox() {
    svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
  }

  function makeSvg(tag, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      el.setAttribute(key, value);
    }
    return el;
  }

  function nodeShape(node) {
    if (node.shape === "square") {
      return makeSvg("rect", {rx: 4});
    }
    if (node.shape === "diamond") {
      return makeSvg("polygon", {});
    }
    return makeSvg("circle", {});
  }

  function placeNodeShape(el, node) {
    const r = node.radius;
    if (node.shape === "square") {
      el.setAttribute("x", node.x - r);
      el.setAttribute("y", node.y - r);
      el.setAttribute("width", 2 * r);
      el.setAttribute("height", 2 * r);
    } else if (node.shape === "diamond") {
      el.setAttribute("points", `${node.x},${node.y-r} ${node.x+r},${node.y} ${node.x},${node.y+r} ${node.x-r},${node.y}`);
    } else {
      el.setAttribute("cx", node.x);
      el.setAttribute("cy", node.y);
      el.setAttribute("r", r);
    }
  }

  function renderEdge(edge) {
    const el = makeSvg("line", {
        stroke: edge.color,
        "stroke-opacity": 0.45,
        "stroke-width": edge.width,
        "data-source": edge.from,
        "data-target": edge.to
      });
    el.addEventListener("pointerenter", event => showEdgeTooltip(event, edge));
    el.addEventListener("pointermove", event => showEdgeTooltip(event, edge));
    el.addEventListener("pointerleave", hideTooltip);
    edge._el = el;
    edgesLayer.appendChild(el);
  }

  function renderNode(node) {
    const group = makeSvg("g", {"class": "node", "data-node-id": node.id});
    const shape = nodeShape(node);
    shape.setAttribute("fill", node.color);
    shape.setAttribute("stroke", node.border_color);
    shape.setAttribute("stroke-width", 3);
    group.appendChild(shape);
    group.addEventListener("pointerdown", event => {
      draggingNode = node;
      lastPointer = pointerInSvg(event);
      group.setPointerCapture(event.pointerId);
      event.stopPropagation();
    });
    group.addEventListener("pointerenter", event => { hoveredNodeId = node._hoverEligible ? node.id : null; showTooltip(event, node); update(); });
    group.addEventListener("pointermove", event => showTooltip(event, node));
    group.addEventListener("pointerleave", () => { hoveredNodeId = null; hideTooltip(); update(); });
    group.addEventListener("click", event => {
      event.stopPropagation();
      showNodeDetails(node);
    });
    node._shape = shape;
    node._group = group;
    nodesLayer.appendChild(group);
    if (node.kind === "factor" || node.radius > 9) {
      const label = makeSvg("text", {
        "text-anchor": "middle",
        "font-size": 12,
        "font-family": "sans-serif",
        "pointer-events": "none"
      });
      label.textContent = node.display_label || node.label;
      node._label = label;
      labelsLayer.appendChild(label);
    }
  }

  function renderInitial() {
    edgesLayer.innerHTML = "";
    nodesLayer.innerHTML = "";
    labelsLayer.innerHTML = "";
    for (const edge of graph.edges) renderEdge(edge);
    for (const node of graph.nodes) renderNode(node);
    refreshAddNodeOptions();
    applyFilters();
    update();
  }

  function pointerInSvg(event) {
    const pt = svg.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  function showTooltip(event, node) {
    tooltip.style.display = "block";
    tooltip.style.left = `${event.clientX + 12}px`;
    tooltip.style.top = `${event.clientY + 12}px`;
    tooltip.textContent = `${node.kind}: ${node.label}`;
  }

  function showEdgeTooltip(event, edge) {
    const provenance = edge.provenance || {};
    const pieces = [
      `${edge.kind || "edge"}: ${edge.from} -> ${edge.to}`,
      `weight=${formatNumber(edge.weight)}`,
      `source=${provenance.source_table || "unknown"}`,
      `weight field=${provenance.weight_field || "weight"}`
    ];
    tooltip.style.display = "block";
    tooltip.style.left = `${event.clientX + 12}px`;
    tooltip.style.top = `${event.clientY + 12}px`;
    tooltip.textContent = pieces.join("\n");
  }

  function hideTooltip() {
    tooltip.style.display = "none";
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function formatNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    if (Math.abs(number) >= 100) return number.toFixed(2);
    if (Math.abs(number) >= 1) return number.toFixed(3);
    if (Math.abs(number) >= 0.001) return number.toFixed(4);
    return number.toExponential(3);
  }

  function valueIsPresent(value) {
    return value !== null && value !== undefined && value !== "";
  }

  function tableHtml(rows, columns) {
    if (!rows || rows.length === 0) return "<p class=\"empty-detail\">No rows available.</p>";
    const visibleColumns = columns.filter(col =>
      col.always || rows.some(row => valueIsPresent(row[col.key]))
    );
    if (visibleColumns.length === 0) return "<p class=\"empty-detail\">No non-empty fields available.</p>";
    const head = visibleColumns.map(col => `<th>${escapeHtml(col.label)}</th>`).join("");
    const body = rows.map(row => {
      return "<tr>" + visibleColumns.map(col => {
        const value = row[col.key];
        const rendered = typeof value === "number" ? formatNumber(value) : value;
        return `<td>${escapeHtml(rendered)}</td>`;
      }).join("") + "</tr>";
    }).join("");
    return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function showNodeDetails(node) {
    const panel = document.getElementById("detailsPanel");
    if (!panel) return;
    const provenance = node.provenance || {};
    let html = `<h3>${escapeHtml(node.kind)}: ${escapeHtml(node.label)}</h3>`;
    html += `<p><strong>ID:</strong> ${escapeHtml(node.id)}</p>`;
    if (node.kind === "factor") {
      html += `<p><strong>Overall relevance:</strong> ${formatNumber(provenance.relevance ?? node.relevance)}</p>`;
      html += "<h4>Relevance By Anchor Trait</h4>";
      html += tableHtml(provenance.relevance_by_anchor || [], [
        {key: "anchor", label: "anchor", always: true},
        {key: "joint_fraction", label: "joint fraction"},
        {key: "joint_coefficient", label: "joint coefficient"},
        {key: "marginal_coefficient", label: "marginal coefficient"},
        {key: "trait_n_eff", label: "trait n eff"},
        {key: "score_source", label: "source"},
        {key: "basis", label: "basis"}
      ]);
      html += "<h4>Top Gene Loadings</h4>";
      html += tableHtml(provenance.top_gene_loadings || [], [
        {key: "id", label: "gene", always: true},
        {key: "loading", label: "loading"},
        {key: "source_field", label: "field"}
      ]);
      html += "<h4>Top Gene Set Loadings</h4>";
      html += tableHtml(provenance.top_gene_set_loadings || [], [
        {key: "id", label: "gene set", always: true},
        {key: "loading", label: "loading"},
        {key: "source_field", label: "field"}
      ]);
    } else {
      html += "<h4>Anchor Trait Support</h4>";
      html += tableHtml(provenance.anchor_support || [], [
        {key: "anchor", label: "anchor", always: true},
        {key: "combined", label: "combined"},
        {key: "direct", label: "direct"},
        {key: "indirect", label: "indirect"},
        {key: "source", label: "source"}
      ]);
      if (node.kind === "gene") {
        html += `<h4>Near-Top Factor Loadings</h4>`;
        html += `<p>${escapeHtml(provenance.near_top_factor_loading_rule || "loading near maximum")}</p>`;
        html += tableHtml(provenance.near_top_factor_loadings || [], [
          {key: "factor_display_label", label: "factor", always: true},
          {key: "loading", label: "loading"},
          {key: "factor", label: "factor id"},
          {key: "source_field", label: "field"}
        ]);
      }
      html += "<h4>Cluster-Table Summary</h4>";
      html += tableHtml([provenance.support_summary || {}], [
        {key: "combined", label: "combined"},
        {key: "direct", label: "direct"},
        {key: "trait_neff", label: "trait neff"}
      ]);
    }
    panel.innerHTML = html;
  }

  function zoomAt(point, scale) {
    viewBox.x = point.x - (point.x - viewBox.x) * scale;
    viewBox.y = point.y - (point.y - viewBox.y) * scale;
    viewBox.w *= scale;
    viewBox.h *= scale;
    setViewBox();
  }

  function zoomCenter(scale) {
    zoomAt({x: viewBox.x + viewBox.w / 2, y: viewBox.y + viewBox.h / 2}, scale);
  }

  function update() {
    const hoveredNeighbors = hoveredNodeId ? (neighborsByNode.get(hoveredNodeId) || new Set()) : new Set();
    const hoveredEdges = hoveredNodeId ? (incidentEdgesByNode.get(hoveredNodeId) || new Set()) : new Set();
    for (const edge of graph.edges) {
      const source = nodeById.get(edge.from);
      const target = nodeById.get(edge.to);
      if (!source || !target) continue;
      edge._el.setAttribute("x1", source.x);
      edge._el.setAttribute("y1", source.y);
      edge._el.setAttribute("x2", target.x);
      edge._el.setAttribute("y2", target.y);
      const edgeVisible = source._visible && target._visible;
      edge._el.style.display = edgeVisible ? "" : "none";
      const edgeHighlighted = hoveredEdges.has(edge._index);
      const baseOpacity = source._dimmed || target._dimmed ? 0.18 : 1.0;
      edge._el.style.opacity = edgeVisible ? (hoveredNodeId ? (edgeHighlighted ? "1" : "0.08") : String(baseOpacity)) : "0";
      edge._el.setAttribute("stroke-width", edgeHighlighted ? Math.max(Number(edge.width || 1) * 2.2, Number(edge.width || 1) + 2.0) : edge.width);
      edge._el.setAttribute("stroke-opacity", edgeHighlighted ? "0.9" : "0.45");
    }
    for (const node of graph.nodes) {
      placeNodeShape(node._shape, node);
      const nodeHighlighted = hoveredNodeId && (node.id === hoveredNodeId || hoveredNeighbors.has(node.id));
      const filterOpacity = node._filterMatched ? 1 : (node._filterNeighbor ? 0.52 : (node._dimmed ? 0.16 : 1));
      node._group.style.display = node._visible ? "" : "none";
      node._group.style.opacity = hoveredNodeId ? (nodeHighlighted ? "1" : "0.16") : String(filterOpacity);
      node._shape.setAttribute("stroke-width", nodeHighlighted ? 5 : 3);
      if (node._label) {
        node._label.setAttribute("x", node.x);
        node._label.setAttribute("y", node.y + node.radius + 14);
        node._label.style.display = node._visible ? "" : "none";
        node._label.style.opacity = hoveredNodeId ? (nodeHighlighted ? "1" : "0.16") : String(filterOpacity);
        node._label.style.fontWeight = nodeHighlighted ? "800" : "400";
      }
    }
  }

  function nodeSearchText(node) {
    return `${node.id} ${node.label} ${node.kind}`.toLowerCase();
  }

  function nodeMatchesText(node) {
    if (textFilters.length === 0) return true;
    return textFilters.some(term => nodeSearchText(node).includes(term));
  }

  function filterStateForNode(node) {
    const hasTextFilter = textFilters.length > 0;
    const typeIsTargeted = activeNodeTypes.has(node.kind);
    if (hasTextFilter && !typeIsTargeted) {
      return {visible: true, dimmed: true};
    }
    const matches = typeIsTargeted && nodeMatchesText(node);
    if (matches) return {visible: true, dimmed: false};
    return {visible: !hideUnmatched, dimmed: true};
  }

  function filterStateForNodeWithNeighbors(node, matchedNodeIds, neighborNodeIds) {
    const base = filterStateForNode(node);
    const hasActiveFilter = textFilters.length > 0;
    if (!highlightNeighbors || !hasActiveFilter || matchedNodeIds.has(node.id)) return base;
    if (!neighborNodeIds.has(node.id)) return base;
    if (hideUnmatched) return {visible: true, dimmed: true, neighbor: true};
    return {visible: true, dimmed: false, neighbor: true};
  }

  function renderFilterChips() {
    const container = document.getElementById("filterChips");
    container.innerHTML = "";
    for (const term of textFilters) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "filter-chip";
      chip.textContent = term + " x";
      chip.addEventListener("click", () => {
        const index = textFilters.indexOf(term);
        if (index >= 0) textFilters.splice(index, 1);
        applyFilters();
      });
      container.appendChild(chip);
    }
  }

  function applyFilters() {
    let visibleCount = 0;
    const matchedNodeIds = new Set();
    const neighborNodeIds = new Set();
    if (textFilters.length > 0) {
      for (const node of graph.nodes) {
        if (activeNodeTypes.has(node.kind) && nodeMatchesText(node)) {
          matchedNodeIds.add(node.id);
          for (const neighbor of (neighborsByNode.get(node.id) || [])) {
            neighborNodeIds.add(neighbor);
          }
        }
      }
    }
    for (const node of graph.nodes) {
      const state = filterStateForNodeWithNeighbors(node, matchedNodeIds, neighborNodeIds);
      node._visible = state.visible;
      node._dimmed = state.dimmed;
      node._filterMatched = matchedNodeIds.has(node.id);
      node._filterNeighbor = Boolean(state.neighbor);
      node._hoverEligible = textFilters.length === 0 || node._filterMatched;
      if (node._visible) visibleCount += 1;
    }
    renderFilterChips();
    const mode = `${hideUnmatched ? "hide unmatched" : "dim unmatched"}${highlightNeighbors ? ", highlight neighbors" : ""}`;
    document.getElementById("filterStatus").textContent = `${visibleCount} / ${graph.nodes.length} nodes visible (${mode})`;
    update();
  }

  function refreshAddNodeOptions() {
    const dataList = document.getElementById("addNodeOptions");
    if (!dataList) return;
    dataList.innerHTML = "";
    const activeIds = new Set(graph.nodes.map(node => node.id));
    const candidates = Array.from(candidateNodeById.values())
      .filter(node => !activeIds.has(node.id))
      .sort((a, b) => `${a.kind}:${a.label}`.localeCompare(`${b.kind}:${b.label}`))
      .slice(0, 2000);
    for (const node of candidates) {
      const option = document.createElement("option");
      option.value = node.id;
      option.label = `${node.kind}: ${node.label}`;
      dataList.appendChild(option);
    }
  }

  function addCandidateNode(rawValue) {
    const query = String(rawValue || "").trim();
    if (!query) return;
    const lower = query.toLowerCase();
    let node = candidateNodeById.get(query);
    if (!node) {
      node = Array.from(candidateNodeById.values()).find(candidate =>
        !nodeById.has(candidate.id) &&
        (`${candidate.id} ${candidate.label} ${candidate.kind}`.toLowerCase().includes(lower))
      );
    }
    if (!node || nodeById.has(node.id)) {
      document.getElementById("addNodeStatus").textContent = node ? "Node is already shown." : "No matching hidden node.";
      return;
    }
    graph.nodes.push(node);
    nodeById.set(node.id, node);
    renderNode(node);
    for (const edge of candidateEdgesByNode.get(node.id) || []) {
      graph.edges.push(edge);
      renderEdge(edge);
    }
    document.getElementById("addNodeInput").value = "";
    document.getElementById("addNodeStatus").textContent = `Added ${node.kind}: ${node.label}`;
    refreshAddNodeOptions();
    applyFilters();
  }

  function addTextFilters(rawValue) {
    for (const rawTerm of rawValue.split(",")) {
      const term = rawTerm.trim().toLowerCase();
      if (term && !textFilters.includes(term)) textFilters.push(term);
    }
    document.getElementById("nodeFilterInput").value = "";
    applyFilters();
  }

  function tick() {
    if (!running) return;
    const centerX = graph.viewport.width / 2;
    const centerY = graph.viewport.height / 2;
    for (const edge of graph.edges) {
      const source = nodeById.get(edge.from);
      const target = nodeById.get(edge.to);
      if (!source || !target) continue;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      let desired = 90 + 15 / Math.max(edge.weight || 0.01, 0.01);
      if (edge.kind === "factor_trait") {
        const scale = Math.max(0.05, Math.min((graph.layout && graph.layout.trait_edge_length_scale) || 0.2, 1.0));
        desired = Math.max(35, desired * scale);
      }
      const force = 0.004 * (dist - desired);
      const fx = force * dx / dist;
      const fy = force * dy / dist;
      if (!source.pinned) { source.vx = (source.vx || 0) + fx; source.vy = (source.vy || 0) + fy; }
      if (!target.pinned) { target.vx = (target.vx || 0) - fx; target.vy = (target.vy || 0) - fy; }
    }
    for (let i = 0; i < graph.nodes.length; i++) {
      for (let j = i + 1; j < graph.nodes.length; j++) {
        const a = graph.nodes[i];
        const b = graph.nodes[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist2 = Math.max(dx * dx + dy * dy, 25);
        const dist = Math.sqrt(dist2);
        const force = 110 / dist2;
        const fx = force * dx / dist;
        const fy = force * dy / dist;
        if (!a.pinned) { a.vx = (a.vx || 0) - fx; a.vy = (a.vy || 0) - fy; }
        if (!b.pinned) { b.vx = (b.vx || 0) + fx; b.vy = (b.vy || 0) + fy; }
      }
    }
    for (const node of graph.nodes) {
      if (node.pinned) continue;
      const anchorPull = node.kind === "trait" ? 0.025 : 0.002;
      node.vx = ((node.vx || 0) + (node.fixed_x - node.x) * anchorPull + (centerX - node.x) * 0.0005) * 0.82;
      node.vy = ((node.vy || 0) + (node.fixed_y - node.y) * anchorPull + (centerY - node.y) * 0.0005) * 0.82;
      node.x += node.vx;
      node.y += node.vy;
    }
    update();
    requestAnimationFrame(tick);
  }

  svg.addEventListener("pointerdown", event => {
    draggingCanvas = true;
    lastPointer = pointerInSvg(event);
  });
  svg.addEventListener("pointermove", event => {
    if (!lastPointer) return;
    const current = pointerInSvg(event);
    if (draggingNode) {
      draggingNode.x += current.x - lastPointer.x;
      draggingNode.y += current.y - lastPointer.y;
      draggingNode.pinned = true;
      update();
    } else if (draggingCanvas) {
      viewBox.x -= current.x - lastPointer.x;
      viewBox.y -= current.y - lastPointer.y;
      setViewBox();
    }
    lastPointer = current;
  });
  svg.addEventListener("pointerup", () => { draggingNode = null; draggingCanvas = null; lastPointer = null; });
  svg.addEventListener("pointerleave", () => { draggingNode = null; draggingCanvas = null; lastPointer = null; hideTooltip(); });
  document.getElementById("zoomInButton").addEventListener("click", function() { zoomCenter(0.82); });
  document.getElementById("zoomOutButton").addEventListener("click", function() { zoomCenter(1.22); });
  document.querySelectorAll(".node-type-filter").forEach(input => {
    input.addEventListener("change", function() {
      if (this.checked) {
        activeNodeTypes.add(this.value);
      } else {
        activeNodeTypes.delete(this.value);
      }
      applyFilters();
    });
  });
  document.getElementById("hideUnmatchedCheckbox").addEventListener("change", function() {
    hideUnmatched = this.checked;
    applyFilters();
  });
  document.getElementById("highlightNeighborsCheckbox").addEventListener("change", function() {
    highlightNeighbors = this.checked;
    applyFilters();
  });
  document.getElementById("addNodeFilterButton").addEventListener("click", function() {
    addTextFilters(document.getElementById("nodeFilterInput").value);
  });
  document.getElementById("nodeFilterInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      addTextFilters(this.value);
    }
  });
  document.getElementById("clearNodeFiltersButton").addEventListener("click", function() {
    textFilters.splice(0, textFilters.length);
    document.querySelectorAll(".node-type-filter").forEach(input => {
      input.checked = true;
      activeNodeTypes.add(input.value);
    });
    applyFilters();
  });
  document.getElementById("addNodeButton").addEventListener("click", function() {
    addCandidateNode(document.getElementById("addNodeInput").value);
  });
  document.getElementById("addNodeInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      addCandidateNode(this.value);
    }
  });
  document.getElementById("togglePhysicsButton").addEventListener("click", function() {
    physicsEnabled = !physicsEnabled;
    running = physicsEnabled;
    this.textContent = physicsEnabled ? "Disable Physics" : "Enable Physics";
    if (running) requestAnimationFrame(tick);
  });
  document.getElementById("resetLayoutButton").addEventListener("click", function() {
    for (const node of graph.nodes) {
      node.x = node.fixed_x;
      node.y = node.fixed_y;
      node.vx = 0;
      node.vy = 0;
      node.pinned = false;
    }
    viewBox = {x: 0, y: 0, w: graph.viewport.width, h: graph.viewport.height};
    setViewBox();
    update();
  });

  setViewBox();
  renderInitial();
  document.getElementById("togglePhysicsButton").textContent = physicsEnabled ? "Disable Physics" : "Enable Physics";
  if (running) requestAnimationFrame(tick);
})();
</script>
""".replace("__PHYSICS_ENABLED__", "true" if physics_enabled else "false")


def write_html(graph: dict, path: str | Path, *, width: int = 1200, height: int = 900, interactive: bool = True, physics: bool = True) -> None:
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not nodes:
        _bail("Cannot write graph HTML with no nodes")
    scaled_graph = _scaled_graph_for_html(graph, width=width, height=height)
    graph_json = json.dumps(scaled_graph, sort_keys=True).replace("</", "<\\/")
    if interactive:
        svg_markup = '<g id="edges-layer"></g>\n    <g id="nodes-layer"></g>\n    <g id="labels-layer"></g>'
        controls = """
  <div class="controls">
    <button id="togglePhysicsButton" type="button">Enable Physics</button>
    <button id="resetLayoutButton" type="button">Reset Layout</button>
    <button id="zoomInButton" type="button">+</button>
    <button id="zoomOutButton" type="button">-</button>
    <span>Drag nodes; drag blank space to pan; use +/- to zoom.</span>
  </div>
  <div class="filters">
    <strong>Filter types:</strong>
    <label><input class="node-type-filter" type="checkbox" value="factor" checked> factors</label>
    <label><input class="node-type-filter" type="checkbox" value="gene" checked> genes</label>
    <label><input class="node-type-filter" type="checkbox" value="trait" checked> phenotypes</label>
    <label><input id="hideUnmatchedCheckbox" type="checkbox"> hide unmatched</label>
    <label><input id="highlightNeighborsCheckbox" type="checkbox"> highlight neighbors</label>
    <input id="nodeFilterInput" type="search" placeholder="Add text filter, comma-separated OR terms">
    <button id="addNodeFilterButton" type="button">Add Filter</button>
    <button id="clearNodeFiltersButton" type="button">Clear</button>
    <span id="filterStatus"></span>
    <span id="filterChips"></span>
  </div>
  <div class="filters">
    <strong>Add node:</strong>
    <input id="addNodeInput" list="addNodeOptions" type="search" placeholder="Type a hidden gene or phenotype">
    <datalist id="addNodeOptions"></datalist>
    <button id="addNodeButton" type="button">Add Node</button>
    <span id="addNodeStatus"></span>
  </div>
"""
        script = _interactive_html_script(physics_enabled=physics)
    else:
        svg_markup = _static_svg_markup(scaled_graph)
        controls = ""
        script = ""
    document = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EAGGL factor graph</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #f7f7f4; color: #222; }}
    .wrap {{ padding: 18px; }}
    svg {{ background: white; border: 1px solid #ddd; width: 100%; height: auto; }}
    .meta {{ color: #555; font-size: 13px; margin-bottom: 8px; }}
    .controls {{ display: flex; align-items: center; gap: 8px; margin: 8px 0 12px; color: #555; font-size: 13px; }}
    .filters {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; color: #555; font-size: 13px; }}
    .filters input[type="search"] {{ min-width: 280px; padding: 6px 8px; border: 1px solid #bbb; border-radius: 4px; }}
    button {{ border: 1px solid #bbb; border-radius: 4px; background: #fff; padding: 6px 10px; cursor: pointer; }}
    button:hover {{ background: #f0f0ea; }}
    .filter-chip {{ padding: 3px 7px; font-size: 12px; }}
    .node {{ cursor: grab; }}
    .node:active {{ cursor: grabbing; }}
    .graph-and-details {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 14px; align-items: start; }}
    #detailsPanel {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; font-size: 12px; max-height: {height}px; overflow: auto; }}
    #detailsPanel h3 {{ margin: 0 0 8px; font-size: 15px; }}
    #detailsPanel h4 {{ margin: 12px 0 6px; font-size: 13px; }}
    #detailsPanel table {{ border-collapse: collapse; width: 100%; font-size: 11px; }}
    #detailsPanel th, #detailsPanel td {{ border-bottom: 1px solid #eee; padding: 3px 4px; text-align: left; vertical-align: top; }}
    #detailsPanel th {{ color: #555; font-weight: 600; }}
    .empty-detail {{ color: #777; margin: 4px 0; }}
    #tooltip {{ position: fixed; display: none; pointer-events: none; background: rgba(30, 30, 30, 0.88); color: white; padding: 5px 8px; border-radius: 4px; font-size: 12px; z-index: 10; }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="meta">EAGGL factor graph: {num_nodes} nodes, {num_edges} edges</div>
  {controls}
  <div class="graph-and-details">
    <svg id="graph-svg" viewBox="0 0 {width} {height}" role="img" aria-label="EAGGL factor graph">
      {svg}
    </svg>
    <aside id="detailsPanel">Click a node to show provenance, anchor support, and top factor loadings.</aside>
  </div>
</div>
<div id="tooltip"></div>
<script type="application/json" id="eaggl-factor-graph-data">{graph_json}</script>
{script}
</body>
</html>
""".format(
        num_nodes=len(nodes),
        num_edges=len(edges),
        width=width,
        height=height,
        controls=controls,
        svg=svg_markup,
        graph_json=graph_json,
        script=script,
    )
    with open_text(path, "wt") as fh:
        fh.write(document)


def write_pdf(graph: dict, path: str | Path, *, width: float = 12.0, height: float = 9.0) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _bail("--pdf-out requires matplotlib")
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not nodes:
        _bail("Cannot write graph PDF with no nodes")
    node_by_id = {node["id"]: node for node in nodes}
    fig, ax = plt.subplots(figsize=(width, height))
    for edge in edges:
        source = node_by_id.get(edge["from"])
        target = node_by_id.get(edge["to"])
        if source is None or target is None:
            continue
        ax.plot(
            [source["x"], target["x"]],
            [source["y"], target["y"]],
            color=edge["color"],
            alpha=0.45,
            linewidth=edge["width"],
            zorder=1,
        )
    for node in nodes:
        marker = "s" if node["shape"] == "square" else ("D" if node["shape"] == "diamond" else "o")
        ax.scatter(
            [node["x"]],
            [node["y"]],
            s=max(20.0, node["size"] * 8),
            c=node["color"],
            edgecolors=node["border_color"],
            linewidths=1.5,
            marker=marker,
            zorder=2,
        )
        if node["kind"] == "factor" or node["size"] > 36:
            ax.text(node["x"], node["y"], str(node["label"]), fontsize=8, ha="center", va="center", zorder=3)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def discover_inputs(eaggl_dir: str | Path | None) -> dict[str, str | None]:
    result = {"factors": None, "genes": None, "gene_sets": None, "traits": None, "params": None}
    if eaggl_dir is None:
        return result
    root = Path(eaggl_dir)
    candidates = {
        "factors": ["factors.out.gz", "factors.out", "factors.tsv.gz", "factors.tsv"],
        "genes": ["gene_clusters_full.out.gz", "gene_clusters.out.gz", "gene_clusters_full.out", "gene_clusters.out"],
        "gene_sets": ["gene_set_clusters.out.gz", "gene_set_clusters.out", "gene_set_clusters.tsv.gz", "gene_set_clusters.tsv"],
        "traits": ["trait_factor_links.out.gz", "trait_factor_links.out", "pheno_clusters.out.gz", "pheno_clusters.out"],
        "params": ["params.out.gz", "params.out", "params.tsv.gz", "params.tsv"],
    }
    for key, names in candidates.items():
        for name in names:
            path = root / name
            if path.exists():
                result[key] = str(path)
                break
    return result


def read_params(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    params_path = Path(path)
    if not params_path.exists():
        return {}
    params: dict[str, str] = {}
    with open_text(params_path) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if not cols or cols[0] == "Parameter":
                continue
            if len(cols) >= 3:
                params[cols[0]] = cols[2]
            elif len(cols) >= 2:
                params[cols[0]] = cols[1]
    return params


def _params_indicate_multi_anchor(params: dict[str, str]) -> bool:
    raw_count = params.get("num_anchor_traits")
    if raw_count is not None:
        try:
            return int(float(raw_count)) > 1
        except ValueError:
            pass
    names = params.get("anchor_trait_names", "")
    if names:
        names = names.strip()
        if "," in names:
            return len([value for value in names.split(",") if value.strip()]) > 1
    return False


def _anchor_trait_names_from_params(params: dict[str, str]) -> tuple[str, ...]:
    names = params.get("anchor_trait_names", "")
    if not names:
        return ()
    return tuple(value.strip() for value in names.split(",") if value.strip())


def _limit_factor_nodes(factors_info: list[FactorInfo], max_num_factor_nodes: int | None) -> list[FactorInfo]:
    if max_num_factor_nodes is None or max_num_factor_nodes <= 0 or len(factors_info) <= max_num_factor_nodes:
        return factors_info
    ranked = sorted(
        enumerate(factors_info),
        key=lambda item: (-item[1].relevance, _factor_sort_key(item[1].factor), item[0]),
    )
    keep_indices = {index for index, _factor in ranked[:max_num_factor_nodes]}
    return [factor for index, factor in enumerate(factors_info) if index in keep_indices]


def build_graph_from_files(args: argparse.Namespace) -> dict:
    discovered = discover_inputs(args.eaggl_dir)
    factors_in = args.factors_in or discovered["factors"]
    gene_clusters_in = args.gene_clusters_in or discovered["genes"]
    gene_set_clusters_in = args.gene_set_clusters_in or discovered["gene_sets"]
    trait_links_in = args.trait_factor_links_in or discovered["traits"]
    params = read_params(discovered["params"])
    if factors_in is None:
        _bail("Need --factors-in or an --eaggl-dir containing factors.out(.gz)")
    if gene_clusters_in is None and trait_links_in is None:
        _bail("Need gene or trait cluster inputs; provide --gene-clusters-in/--trait-factor-links-in or an --eaggl-dir with standard outputs")
    factors_info = read_factors(factors_in, id_col=args.factors_id_col, label_col=args.factors_label_col, relevance_col=args.factors_relevance_col)
    factors_info = _limit_factor_nodes(factors_info, args.max_num_factor_nodes)
    factors = [factor.factor for factor in factors_info]
    factor_labels = {factor.factor: factor.label for factor in factors_info}
    gene_support = read_anchor_support_rows(
        args.gene_phewas_stats_in,
        id_col=args.gene_phewas_stats_id_col,
        anchor_col=args.gene_phewas_stats_pheno_col,
        combined_col=args.gene_phewas_stats_combined_col,
        direct_col=args.gene_phewas_stats_log_bf_col,
        indirect_col=args.gene_phewas_stats_prior_col,
        source_label="gene_phewas_stats",
        max_rows_per_node=args.max_anchor_support_rows_per_node,
        min_combined=args.anchor_support_min_combined,
    )
    gene_set_support = read_anchor_support_rows(
        args.gene_set_phewas_stats_in,
        id_col=args.gene_set_phewas_stats_id_col,
        anchor_col=args.gene_set_phewas_stats_pheno_col,
        combined_col=None,
        direct_col=args.gene_set_phewas_stats_beta_col,
        indirect_col=args.gene_set_phewas_stats_beta_uncorrected_col,
        source_label="gene_set_phewas_stats",
        max_rows_per_node=args.max_anchor_support_rows_per_node,
        min_combined=args.anchor_support_min_combined,
    )
    config = GraphConfig(
        gene_min_loading=args.gene_min_loading,
        trait_min_loading=args.trait_min_loading,
        trait_min_neff=args.trait_min_neff,
        gene_min_loading_frac=args.gene_min_loading_frac,
        trait_min_loading_frac=args.trait_min_loading_frac,
        max_num_factor_nodes=args.max_num_factor_nodes,
        max_num_gene_nodes_per_factor=args.max_num_gene_nodes_per_factor,
        max_num_trait_nodes_per_factor=args.max_num_trait_nodes_per_factor,
        max_num_trait_provenance_per_factor=args.max_num_trait_provenance_per_factor,
        trait_factor_min_beta=args.trait_factor_min_beta,
        trait_factor_min_beta_uncorrected=args.trait_factor_min_beta_uncorrected,
        trait_factor_min_nnls=args.trait_factor_min_nnls,
        trait_factor_rank_field=args.trait_factor_rank_field,
        factor_trait_enrichments_in=args.factor_trait_enrichments_in,
        max_anchor_support_rows_per_node=args.max_anchor_support_rows_per_node,
        anchor_support_min_combined=args.anchor_support_min_combined,
        coordinate_scale=args.coordinate_scale,
        trait_coordinate_scale=args.trait_coordinate_scale,
        trait_layout_mode=args.trait_layout_mode,
        trait_min_centroid_distance_frac=args.trait_min_centroid_distance_frac,
        trait_edge_length_scale=args.trait_edge_length_scale,
        node_size_scale=args.node_size_scale,
        edge_max_width=args.edge_max_width,
        label_max_chars=args.label_max_chars,
        colors_red_blue=args.colors_red_blue,
        color_by=args.color_by,
        multi_anchor=_params_indicate_multi_anchor(params),
        anchor_trait_names=_anchor_trait_names_from_params(params),
        seed=args.seed,
    )
    genes: list[EntityInfo] = []
    candidate_genes: list[EntityInfo] = []
    if gene_clusters_in is not None:
        genes = read_wide_entities(
            gene_clusters_in,
            kind="gene",
            factors=factors,
            id_col=args.gene_id_col,
            label_col=args.gene_label_col,
            combined_col=args.gene_combined_col,
            direct_col=args.gene_direct_col,
            min_loading=config.gene_min_loading,
            min_loading_frac=config.gene_min_loading_frac,
            max_num_per_factor=config.max_num_gene_nodes_per_factor,
        )
        genes = attach_anchor_support(genes, gene_support)
        genes = attach_near_top_factor_loadings(genes, factors, factor_labels=factor_labels, label_max_chars=args.label_max_chars)
        candidate_genes = read_wide_entity_candidates(
            gene_clusters_in,
            kind="gene",
            factors=factors,
            id_col=args.gene_id_col,
            label_col=args.gene_label_col,
            combined_col=args.gene_combined_col,
            direct_col=args.gene_direct_col,
            min_loading=config.gene_min_loading,
            min_loading_frac=config.gene_min_loading_frac,
        )
        candidate_genes = attach_anchor_support(candidate_genes, gene_support)
        candidate_genes = attach_near_top_factor_loadings(candidate_genes, factors, factor_labels=factor_labels, label_max_chars=args.label_max_chars)
    gene_set_candidates: list[EntityInfo] = []
    if gene_set_clusters_in is not None:
        gene_set_candidates = read_wide_entity_candidates(
            gene_set_clusters_in,
            kind="gene_set",
            factors=factors,
            id_col=args.gene_set_id_col,
            label_col=args.gene_set_label_col,
            combined_col=args.gene_set_combined_col,
            direct_col=args.gene_set_direct_col,
            min_loading=0.0,
            min_loading_frac=0.0,
        )
        gene_set_candidates = attach_anchor_support(gene_set_candidates, gene_set_support)
    traits: list[EntityInfo] = []
    if trait_links_in is not None:
        traits = read_trait_links(
            trait_links_in,
            factors=factors,
            min_loading=config.trait_min_loading,
            min_neff=config.trait_min_neff,
            min_loading_frac=config.trait_min_loading_frac,
            max_num_per_factor=config.max_num_trait_nodes_per_factor,
        )
    factor_trait_details = read_factor_trait_details(
        trait_links_in,
        factors,
        enrichment_path=config.factor_trait_enrichments_in,
        max_num_per_factor=config.max_num_trait_provenance_per_factor,
        min_beta=config.trait_factor_min_beta,
        min_beta_uncorrected=config.trait_factor_min_beta_uncorrected,
        min_nnls=config.trait_factor_min_nnls,
        rank_field=config.trait_factor_rank_field,
    )
    factor_trait_color_details = read_factor_trait_details(
        trait_links_in,
        factors,
        enrichment_path=config.factor_trait_enrichments_in,
        max_num_per_factor=-1,
        min_beta=0.0,
        min_beta_uncorrected=0.0,
        min_nnls=0.0,
        rank_field=config.trait_factor_rank_field,
    )
    top_gene_loadings = _top_loadings_by_factor(candidate_genes or genes, factors, top_n=5)
    top_gene_set_loadings = _top_loadings_by_factor(gene_set_candidates, factors, top_n=5)
    return build_graph(
        factors_info,
        genes,
        traits,
        config,
        candidate_genes=candidate_genes,
        factor_trait_details=factor_trait_details,
        factor_trait_color_details=factor_trait_color_details,
        top_gene_loadings_by_factor=top_gene_loadings,
        top_gene_set_loadings_by_factor=top_gene_set_loadings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a post-processing factor graph from EAGGL outputs.")
    parser.add_argument("--eaggl-dir", default=None, help="Directory containing standard EAGGL output files.")
    parser.add_argument("--factors-in", default=None)
    parser.add_argument("--gene-clusters-in", default=None)
    parser.add_argument("--gene-set-clusters-in", default=None)
    parser.add_argument("--trait-factor-links-in", default=None)
    parser.add_argument("--factor-trait-enrichments-in", default=None, help="Optional PIGEAN factor-trait enrichment table to merge into trait-factor graph provenance.")
    parser.add_argument("--gene-phewas-stats-in", action="append", default=None, help="Optional gene PheWAS stats for node provenance; repeat to read multiple files.")
    parser.add_argument("--gene-phewas-stats-id-col", default="Gene")
    parser.add_argument("--gene-phewas-stats-pheno-col", default="Trait")
    parser.add_argument("--gene-phewas-stats-combined-col", default="Combined")
    parser.add_argument("--gene-phewas-stats-log-bf-col", default="Direct")
    parser.add_argument("--gene-phewas-stats-prior-col", default="Indirect")
    parser.add_argument("--gene-set-phewas-stats-in", action="append", default=None, help="Optional gene-set PheWAS stats for factor top-loading provenance; repeat to read multiple files.")
    parser.add_argument("--gene-set-phewas-stats-id-col", default="Gene_Set")
    parser.add_argument("--gene-set-phewas-stats-pheno-col", default="Trait")
    parser.add_argument("--gene-set-phewas-stats-beta-col", default="beta")
    parser.add_argument("--gene-set-phewas-stats-beta-uncorrected-col", default="beta_uncorrected")
    parser.add_argument("--html-out", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--pdf-out", default=None)
    parser.add_argument("--html-physics", action="store_true", default=True, help="Start the interactive HTML with browser-side force physics enabled. This is the default.")
    parser.add_argument("--no-html-physics", action="store_false", dest="html_physics", help="Start the interactive HTML with browser-side force physics disabled.")
    parser.add_argument(
        "--no-html-interactive",
        action="store_false",
        dest="html_interactive",
        default=True,
        help="Write static SVG HTML without drag, pan, zoom, or physics controls.",
    )
    parser.add_argument("--factors-id-col", default=None)
    parser.add_argument("--factors-label-col", default=None)
    parser.add_argument("--factors-relevance-col", default=None)
    parser.add_argument("--gene-id-col", default="Gene")
    parser.add_argument("--gene-label-col", default=None, help="Optional gene label column. Defaults to the gene ID column.")
    parser.add_argument("--gene-combined-col", default="combined")
    parser.add_argument("--gene-direct-col", default="log_bf")
    parser.add_argument("--gene-set-id-col", default="Gene_Set")
    parser.add_argument("--gene-set-label-col", default=None)
    parser.add_argument("--gene-set-combined-col", default="combined")
    parser.add_argument("--gene-set-direct-col", default="log_bf")
    parser.add_argument("--gene-min-loading", type=float, default=0.01)
    parser.add_argument("--trait-min-loading", type=float, default=0.005)
    parser.add_argument("--trait-min-neff", type=float, default=25.0, help="Minimum trait effective size for phenotype nodes when trait_neff/trait_n_eff is available.")
    parser.add_argument("--gene-min-loading-frac", type=float, default=0.5)
    parser.add_argument("--trait-min-loading-frac", type=float, default=0.5)
    parser.add_argument("--max-num-factor-nodes", type=int, default=50, help="Maximum factor nodes to show, ranked by relevance; use 0 to show all factors.")
    parser.add_argument("--max-num-gene-nodes-per-factor", type=int, default=3)
    parser.add_argument("--max-num-trait-nodes-per-factor", type=int, default=3)
    parser.add_argument("--max-num-trait-provenance-per-factor", type=int, default=20, help="Maximum trait-link provenance rows embedded per factor node; use -1 to keep all rows.")
    parser.add_argument("--trait-factor-min-beta", type=float, default=0.01, help="Embed trait-factor provenance rows when beta exceeds this threshold; OR-combined with beta_uncorrected and NNLS filters.")
    parser.add_argument("--trait-factor-min-beta-uncorrected", type=float, default=0.05, help="Embed trait-factor provenance rows when beta_uncorrected exceeds this threshold; OR-combined with beta and NNLS filters.")
    parser.add_argument("--trait-factor-min-nnls", type=float, default=0.5, help="Embed trait-factor provenance rows when NNLS loading exceeds this threshold; OR-combined with beta filters.")
    parser.add_argument("--trait-factor-rank-field", choices=["beta", "beta_uncorrected", "nnls"], default="beta", help="Rank embedded trait-factor provenance rows by this field after filtering.")
    parser.add_argument("--max-anchor-support-rows-per-node", type=int, default=20, help="Maximum gene/gene-set phenotype support provenance rows embedded per graph node; use -1 to keep all rows.")
    parser.add_argument("--anchor-support-min-combined", type=float, default=0.0, help="Minimum combined support for gene/gene-set phenotype support provenance rows embedded in graph nodes.")
    parser.add_argument("--coordinate-scale", type=float, default=5.0)
    parser.add_argument("--trait-coordinate-scale", type=float, default=0.2, help="Scale trait-node displacement from the factor centroid after layout; 1.0 preserves the raw MDS distance.")
    parser.add_argument(
        "--trait-layout-mode",
        choices=["mds", "anchored_top_factor"],
        default="anchored_top_factor",
        help="Place phenotype nodes by raw MDS (`mds`) or anchor them near their strongest factor mixture to avoid centroid collapse (`anchored_top_factor`, default).",
    )
    parser.add_argument(
        "--trait-min-centroid-distance-frac",
        type=float,
        default=0.35,
        help="In anchored trait layout mode, enforce this fraction of the median factor-centroid radius as a minimum phenotype-node radius.",
    )
    parser.add_argument("--trait-edge-length-scale", type=float, default=0.2, help="Scale factor-trait spring length in interactive physics; lower values keep phenotype nodes closer to factors.")
    parser.add_argument("--node-size-scale", type=float, default=2.0)
    parser.add_argument("--edge-max-width", type=float, default=5.0)
    parser.add_argument("--label-max-chars", type=int, default=20, help="Maximum displayed node label length; full labels remain available on hover. Use 0 to disable truncation.")
    parser.add_argument(
        "--color-by",
        choices=["auto", "factor", "trait"],
        default="auto",
        help="Node coloring mode. auto uses trait-weight coloring for multi-anchor EAGGL outputs with trait links, otherwise factor coloring.",
    )
    parser.add_argument("--colors-red-blue", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.html_out is None and args.json_out is None and args.pdf_out is None:
        _bail("Need at least one of --html-out, --json-out, or --pdf-out")
    graph = build_graph_from_files(args)
    if args.json_out is not None:
        write_json(graph, args.json_out)
    if args.html_out is not None:
        write_html(graph, args.html_out, interactive=args.html_interactive, physics=args.html_physics)
    if args.pdf_out is not None:
        write_pdf(graph, args.pdf_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
