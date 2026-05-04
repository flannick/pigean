from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import os
import random
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GraphConfig:
    gene_min_loading: float = 0.01
    trait_min_loading: float = 0.005
    trait_min_neff: float = 25.0
    gene_min_loading_frac: float = 0.5
    trait_min_loading_frac: float = 0.5
    max_num_gene_nodes_per_factor: int = 5
    max_num_trait_nodes_per_factor: int = 5
    coordinate_scale: float = 5.0
    trait_coordinate_scale: float = 0.2
    trait_edge_length_scale: float = 0.2
    node_size_scale: float = 2.0
    edge_max_width: float = 5.0
    label_max_chars: int = 20
    colors_red_blue: bool = False
    seed: int = 0


def _bail(message: str) -> None:
    raise SystemExit("Error: %s" % message)


def open_text(path: str | Path, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


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
            entities.append(EntityInfo(entity_id=entity_id, label=label, kind=kind, combined=combined, direct=direct, loadings=loadings))
    entities = _filter_entities_by_factor_rank(
        entities,
        factors,
        min_loading=min_loading,
        min_loading_frac=min_loading_frac,
        max_num_per_factor=max_num_per_factor,
    )
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


def build_graph(factors_info: list[FactorInfo], genes: list[EntityInfo], traits: list[EntityInfo], config: GraphConfig) -> dict:
    factors = [factor.factor for factor in factors_info]
    factor_to_label = {factor.factor: factor.label for factor in factors_info}
    max_relevance = max([factor.relevance for factor in factors_info] + [1.0])
    entities = genes + traits
    entity_matrix = np.asarray([[entity.loadings.get(factor, 0.0) for factor in factors] for entity in entities], dtype=float)
    factor_identity = np.eye(len(factors), dtype=float)
    layout_matrix = np.vstack([entity_matrix, factor_identity]) if len(entities) else factor_identity
    colors = generate_distinct_colors(len(factors), start_with_red_blue=config.colors_red_blue)
    trait_colors_by_id: dict[str, tuple[float, float, float]] = {}
    if traits:
        trait_colors = generate_distinct_colors(len(traits), start_with_red_blue=config.colors_red_blue)
        trait_colors_by_id = {trait.entity_id: trait_colors[i] for i, trait in enumerate(traits)}
        colors = []
        for factor in factors:
            factor_trait_weights = [trait.loadings.get(factor, 0.0) for trait in traits]
            colors.append(blend_colors(trait_colors, factor_trait_weights, opacity=1.0))
    elif entity_matrix.size:
        colors = _reorder_colors_by_factor_correlation(colors, entity_matrix)
    coords = compute_layout(layout_matrix, coordinate_scale=config.coordinate_scale, seed=config.seed)
    if traits and 0 <= config.trait_coordinate_scale < 1:
        factor_coords = coords[len(entities) :]
        factor_center = np.mean(factor_coords, axis=0) if factor_coords.size else np.zeros(2, dtype=float)
        for entity_index, entity in enumerate(entities):
            if entity.kind == "trait":
                coords[entity_index] = factor_center + config.trait_coordinate_scale * (coords[entity_index] - factor_center)
    nodes = []
    edges = []
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
                }
            )
    return {
        "schema": "eaggl_factor_graph/v1",
        "layout": {
            "trait_coordinate_scale": float(config.trait_coordinate_scale),
            "trait_edge_length_scale": float(config.trait_edge_length_scale),
        },
        "factors": factors,
        "factor_labels": [factor_to_label[factor] for factor in factors],
        "nodes": nodes,
        "edges": edges,
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

    scaled_nodes = []
    for node in nodes:
        scaled_node = dict(node)
        scaled_node["fixed_x"] = sx(node["x"])
        scaled_node["fixed_y"] = sy(node["y"])
        scaled_node["x"] = scaled_node["fixed_x"]
        scaled_node["y"] = scaled_node["fixed_y"]
        scaled_node["radius"] = max(6.0, float(node["size"]) / 5.0)
        scaled_nodes.append(scaled_node)
    return {
        **graph,
        "nodes": scaled_nodes,
        "edges": [dict(edge) for edge in edges],
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
  const activeNodeTypes = new Set(["factor", "gene", "trait"]);
  const textFilters = [];
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

  function renderInitial() {
    edgesLayer.innerHTML = "";
    nodesLayer.innerHTML = "";
    labelsLayer.innerHTML = "";
    for (const edge of graph.edges) {
      const el = makeSvg("line", {
        stroke: edge.color,
        "stroke-opacity": 0.45,
        "stroke-width": edge.width,
        "data-source": edge.from,
        "data-target": edge.to
      });
      edge._el = el;
      edgesLayer.appendChild(el);
    }
    for (const node of graph.nodes) {
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
      group.addEventListener("pointerenter", event => showTooltip(event, node));
      group.addEventListener("pointermove", event => showTooltip(event, node));
      group.addEventListener("pointerleave", hideTooltip);
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

  function hideTooltip() {
    tooltip.style.display = "none";
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
    for (const edge of graph.edges) {
      const source = nodeById.get(edge.from);
      const target = nodeById.get(edge.to);
      if (!source || !target) continue;
      edge._el.setAttribute("x1", source.x);
      edge._el.setAttribute("y1", source.y);
      edge._el.setAttribute("x2", target.x);
      edge._el.setAttribute("y2", target.y);
      edge._el.style.display = source._visible && target._visible ? "" : "none";
    }
    for (const node of graph.nodes) {
      placeNodeShape(node._shape, node);
      node._group.style.display = node._visible ? "" : "none";
      if (node._label) {
        node._label.setAttribute("x", node.x);
        node._label.setAttribute("y", node.y + node.radius + 14);
        node._label.style.display = node._visible ? "" : "none";
      }
    }
  }

  function nodeSearchText(node) {
    return `${node.id} ${node.label} ${node.kind}`.toLowerCase();
  }

  function nodePassesFilters(node) {
    const typeMatch = activeNodeTypes.has(node.kind);
    const textMatch = textFilters.length === 0 || textFilters.some(term => nodeSearchText(node).includes(term));
    return typeMatch && textMatch;
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
    for (const node of graph.nodes) {
      node._visible = nodePassesFilters(node);
      if (node._visible) visibleCount += 1;
    }
    renderFilterChips();
    document.getElementById("filterStatus").textContent = `${visibleCount} / ${graph.nodes.length} nodes visible`;
    update();
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


def write_html(graph: dict, path: str | Path, *, width: int = 1200, height: int = 900, interactive: bool = True, physics: bool = False) -> None:
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
    <strong>Show:</strong>
    <label><input class="node-type-filter" type="checkbox" value="factor" checked> factors</label>
    <label><input class="node-type-filter" type="checkbox" value="gene" checked> genes</label>
    <label><input class="node-type-filter" type="checkbox" value="trait" checked> phenotypes</label>
    <input id="nodeFilterInput" type="search" placeholder="Add text filter, comma-separated OR terms">
    <button id="addNodeFilterButton" type="button">Add Filter</button>
    <button id="clearNodeFiltersButton" type="button">Clear</button>
    <span id="filterStatus"></span>
    <span id="filterChips"></span>
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
    #tooltip {{ position: fixed; display: none; pointer-events: none; background: rgba(30, 30, 30, 0.88); color: white; padding: 5px 8px; border-radius: 4px; font-size: 12px; z-index: 10; }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="meta">EAGGL factor graph: {num_nodes} nodes, {num_edges} edges</div>
  {controls}
  <svg id="graph-svg" viewBox="0 0 {width} {height}" role="img" aria-label="EAGGL factor graph">
    {svg}
  </svg>
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
    result = {"factors": None, "genes": None, "traits": None}
    if eaggl_dir is None:
        return result
    root = Path(eaggl_dir)
    candidates = {
        "factors": ["factors.out.gz", "factors.out", "factors.tsv.gz", "factors.tsv"],
        "genes": ["gene_clusters_full.out.gz", "gene_clusters.out.gz", "gene_clusters_full.out", "gene_clusters.out"],
        "traits": ["trait_factor_links.out.gz", "trait_factor_links.out", "pheno_clusters.out.gz", "pheno_clusters.out"],
    }
    for key, names in candidates.items():
        for name in names:
            path = root / name
            if path.exists():
                result[key] = str(path)
                break
    return result


def build_graph_from_files(args: argparse.Namespace) -> dict:
    discovered = discover_inputs(args.eaggl_dir)
    factors_in = args.factors_in or discovered["factors"]
    gene_clusters_in = args.gene_clusters_in or discovered["genes"]
    trait_links_in = args.trait_factor_links_in or discovered["traits"]
    if factors_in is None:
        _bail("Need --factors-in or an --eaggl-dir containing factors.out(.gz)")
    if gene_clusters_in is None and trait_links_in is None:
        _bail("Need gene or trait cluster inputs; provide --gene-clusters-in/--trait-factor-links-in or an --eaggl-dir with standard outputs")
    factors_info = read_factors(factors_in, id_col=args.factors_id_col, label_col=args.factors_label_col, relevance_col=args.factors_relevance_col)
    factors = [factor.factor for factor in factors_info]
    config = GraphConfig(
        gene_min_loading=args.gene_min_loading,
        trait_min_loading=args.trait_min_loading,
        trait_min_neff=args.trait_min_neff,
        gene_min_loading_frac=args.gene_min_loading_frac,
        trait_min_loading_frac=args.trait_min_loading_frac,
        max_num_gene_nodes_per_factor=args.max_num_gene_nodes_per_factor,
        max_num_trait_nodes_per_factor=args.max_num_trait_nodes_per_factor,
        coordinate_scale=args.coordinate_scale,
        trait_coordinate_scale=args.trait_coordinate_scale,
        trait_edge_length_scale=args.trait_edge_length_scale,
        node_size_scale=args.node_size_scale,
        edge_max_width=args.edge_max_width,
        label_max_chars=args.label_max_chars,
        colors_red_blue=args.colors_red_blue,
        seed=args.seed,
    )
    genes: list[EntityInfo] = []
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
    return build_graph(factors_info, genes, traits, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a post-processing factor graph from EAGGL outputs.")
    parser.add_argument("--eaggl-dir", default=None, help="Directory containing standard EAGGL output files.")
    parser.add_argument("--factors-in", default=None)
    parser.add_argument("--gene-clusters-in", default=None)
    parser.add_argument("--trait-factor-links-in", default=None)
    parser.add_argument("--html-out", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--pdf-out", default=None)
    parser.add_argument("--html-physics", action="store_true", help="Start the interactive HTML with browser-side force physics enabled.")
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
    parser.add_argument("--gene-min-loading", type=float, default=0.01)
    parser.add_argument("--trait-min-loading", type=float, default=0.005)
    parser.add_argument("--trait-min-neff", type=float, default=25.0, help="Minimum trait effective size for phenotype nodes when trait_neff/trait_n_eff is available.")
    parser.add_argument("--gene-min-loading-frac", type=float, default=0.5)
    parser.add_argument("--trait-min-loading-frac", type=float, default=0.5)
    parser.add_argument("--max-num-gene-nodes-per-factor", type=int, default=5)
    parser.add_argument("--max-num-trait-nodes-per-factor", type=int, default=5)
    parser.add_argument("--coordinate-scale", type=float, default=5.0)
    parser.add_argument("--trait-coordinate-scale", type=float, default=0.2, help="Scale trait-node displacement from the factor centroid after layout; 1.0 preserves the raw MDS distance.")
    parser.add_argument("--trait-edge-length-scale", type=float, default=0.2, help="Scale factor-trait spring length in interactive physics; lower values keep phenotype nodes closer to factors.")
    parser.add_argument("--node-size-scale", type=float, default=2.0)
    parser.add_argument("--edge-max-width", type=float, default=5.0)
    parser.add_argument("--label-max-chars", type=int, default=20, help="Maximum displayed node label length; full labels remain available on hover. Use 0 to disable truncation.")
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
