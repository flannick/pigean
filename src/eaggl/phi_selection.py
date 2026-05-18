from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import scipy.sparse as sparse
except Exception:  # pragma: no cover
    sparse = None


COMPONENT_ORDER = [
    "factor_size",
    "nonoverlap",
    "entity_concentration",
    "coverage",
    "reconstruction",
    "coherence",
    "factor_balance",
    "annotation_bridge_qc",
]

DEFAULT_COMPONENT_WEIGHTS = {
    "factor_size": 0.15,
    "nonoverlap": 0.15,
    "entity_concentration": 0.15,
    "coverage": 0.25,
    "reconstruction": 0.10,
    "coherence": 0.10,
    "factor_balance": 0.05,
    "annotation_bridge_qc": 0.05,
}


@dataclass(frozen=True)
class CompositePhiSelectionConfig:
    weights: dict[str, float]
    target_factor_gene_mass: float = 100.0
    size_log2_width: float = 1.0
    loading_cap: float = 1.0
    min_entity_total_loading: float = 0.01
    bridge_concentration_threshold: float = 0.60
    coverage_min_loading: float = 0.05
    gene_coverage_top_frac: float = 0.05
    gene_coverage_top_n: int | None = None
    annotation_coverage_top_frac: float = 0.05
    annotation_coverage_top_n: int | None = None
    tie_tolerance: float = 0.01


@dataclass(frozen=True)
class PhiSelectionInputs:
    discovery_model: str
    gene_loadings: Any | None = None
    annotation_loadings: Any | None = None
    target_matrix: Any | None = None
    target_weight_matrix: Any | None = None
    target_gene_indices: Any | None = None
    target_annotation_indices: Any | None = None
    gene_importance: Any | None = None
    annotation_importance: Any | None = None
    annotation_bridge_metrics: list[dict[str, Any]] | None = None


def _canonical_component_name(name: str) -> str:
    name = str(name).strip()
    if name.endswith("_score"):
        name = name[: -len("_score")]
    aliases = {
        "factor_gene_size": "factor_size",
        "gene_size": "factor_size",
        "factor_nonoverlap": "nonoverlap",
        "entity_concentration": "entity_concentration",
        "anti_bridging": "entity_concentration",
        "high_priority_coverage": "coverage",
        "utilization": "factor_balance",
        "balance": "factor_balance",
        "bridge_qc": "annotation_bridge_qc",
    }
    name = aliases.get(name, name)
    if name not in DEFAULT_COMPONENT_WEIGHTS:
        raise ValueError("unknown phi-selection composite weight component: %s" % name)
    return name


def parse_composite_weights(raw: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_COMPONENT_WEIGHTS)
    if raw is not None and str(raw).strip() != "":
        for item in str(raw).split(","):
            item = item.strip()
            if item == "":
                continue
            if "=" not in item:
                raise ValueError("--phi-selection-composite-weights entries must be name=value")
            name, value = item.split("=", 1)
            component = _canonical_component_name(name)
            weight = float(value)
            if weight < 0:
                raise ValueError("phi-selection composite weights must be nonnegative")
            weights[component] = weight
    if not any(float(value) > 0.0 for value in weights.values()):
        raise ValueError("at least one phi-selection composite weight must be positive")
    return weights


def _dense(matrix):
    if matrix is None:
        return None
    if sparse is not None and sparse.issparse(matrix):
        matrix = matrix.toarray()
    arr = np.asarray(matrix, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _bounded(matrix, loading_cap: float):
    arr = _dense(matrix)
    if arr is None:
        return None
    return np.clip(arr, 0.0, float(loading_cap))


def _safe_quantile(values, q, default=0.0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.quantile(arr, q))


def _weighted_mean(values, weights=None):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if weights is None:
        return 0.0 if not np.any(mask) else float(np.mean(values[mask]))
    weights = np.asarray(weights, dtype=float)
    mask &= np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return 0.0 if values.size == 0 else float(np.mean(values[np.isfinite(values)]))
    return float(np.sum(values[mask] * weights[mask]) / (np.sum(weights[mask]) + 1e-50))


def _soft_jaccard_q90(matrix):
    X = _dense(matrix)
    if X is None or X.ndim != 2 or X.shape[1] <= 1:
        return 0.0, 0.0, 0.0
    X = np.clip(X, 0.0, None)
    masses = np.sum(X, axis=0)
    values = []
    shared = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            overlap = float(np.sum(X[:, i] * X[:, j]))
            denom = float(masses[i] + masses[j] - overlap)
            value = 0.0 if denom <= 0 else overlap / denom
            values.append(value)
            shared.append(overlap)
    if not values:
        return 0.0, 0.0, 0.0
    return float(np.mean(shared)), float(np.mean(values)), float(np.quantile(values, 0.9))


def _factor_size_metrics(Wc, config):
    out = {}
    if Wc is None or Wc.ndim != 2 or Wc.shape[1] == 0:
        return None, out, []
    masses = np.sum(Wc, axis=0)
    target = float(config.target_factor_gene_mass)
    width = max(float(config.size_log2_width), 1e-12)
    scores = 1.0 / (1.0 + (np.log2((masses + 1e-50) / (target + 1e-50)) / width) ** 2)
    scores = np.clip(scores, 0.0, 1.0)
    out.update(
        {
            "factor_gene_mass_mean": float(np.mean(masses)) if masses.size else 0.0,
            "factor_gene_mass_median": float(np.median(masses)) if masses.size else 0.0,
            "factor_gene_mass_q10": _safe_quantile(masses, 0.1),
            "factor_gene_mass_q90": _safe_quantile(masses, 0.9),
            "factor_gene_mass_target": target,
            "factor_size_score": float(np.mean(scores)) if scores.size else 0.0,
        }
    )
    per_factor = [
        {"factor_gene_mass": float(masses[i]), "factor_size_score": float(scores[i])}
        for i in range(len(masses))
    ]
    return out["factor_size_score"], out, per_factor


def _nonoverlap_metrics(Wc, Hc):
    if Wc is None or Wc.ndim != 2 or Wc.shape[1] < 2:
        return None, {
            "gene_factor_pair_soft_shared_mean": None,
            "gene_factor_pair_soft_shared_q90": None,
            "gene_factor_pair_soft_jaccard_mean": None,
            "gene_factor_pair_soft_jaccard_q90": None,
            "gene_nonoverlap_score": None,
            "annotation_nonoverlap_score": None,
            "nonoverlap_score": None,
        }
    gene_shared_mean, gene_j_mean, gene_j_q90 = _soft_jaccard_q90(Wc)
    gene_score = _clip01(1.0 - gene_j_q90)
    out = {
        "gene_factor_pair_soft_shared_mean": gene_shared_mean,
        "gene_factor_pair_soft_shared_q90": 0.0,
        "gene_factor_pair_soft_jaccard_mean": gene_j_mean,
        "gene_factor_pair_soft_jaccard_q90": gene_j_q90,
        "gene_nonoverlap_score": gene_score,
    }
    if Hc is not None and Hc.ndim == 2 and Hc.shape[1] > 1:
        ann_shared_mean, ann_j_mean, ann_j_q90 = _soft_jaccard_q90(Hc)
        ann_score = _clip01(1.0 - ann_j_q90)
        score = _clip01(0.75 * gene_score + 0.25 * ann_score)
        out.update(
            {
                "annotation_factor_pair_soft_shared_mean": ann_shared_mean,
                "annotation_factor_pair_soft_jaccard_mean": ann_j_mean,
                "annotation_factor_pair_soft_jaccard_q90": ann_j_q90,
                "annotation_nonoverlap_score": ann_score,
            }
        )
    else:
        score = gene_score
        out["annotation_nonoverlap_score"] = None
    out["nonoverlap_score"] = score
    return score, out


def _entity_concentration_one(Xc, importance, config, prefix):
    if Xc is None or Xc.ndim != 2 or Xc.shape[1] == 0:
        return None, {prefix + "_concentration_score": None}
    totals = np.sum(Xc, axis=1)
    active = totals >= float(config.min_entity_total_loading)
    if not np.any(active):
        return 0.0, {
            prefix + "_concentration_score": 0.0,
            prefix + "_bridge_fraction": 0.0,
            prefix + "_neff_mean": 0.0,
            prefix + "_neff_q90": 0.0,
        }
    P = Xc[active, :] / (totals[active, np.newaxis] + 1e-50)
    concentration = np.max(P, axis=1)
    neff = 1.0 / (np.sum(P * P, axis=1) + 1e-50)
    weights = None
    if importance is not None:
        imp = np.asarray(importance, dtype=float)
        if imp.shape[0] == Xc.shape[0]:
            weights = np.maximum(np.nan_to_num(imp[active], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if weights is None:
        weights = totals[active]
    bridge = concentration < float(config.bridge_concentration_threshold)
    score = _clip01(_weighted_mean(concentration, weights))
    return score, {
        prefix + "_concentration_score": score,
        prefix + "_bridge_fraction": _clip01(_weighted_mean(bridge.astype(float), weights)),
        prefix + "_neff_mean": _weighted_mean(neff, weights),
        prefix + "_neff_q90": _safe_quantile(neff, 0.9),
    }


def _coverage_one(Xc, importance, top_frac, top_n, min_loading, prefix):
    if Xc is None or Xc.ndim != 2 or Xc.shape[1] == 0:
        return None, {prefix + "_coverage_score": None, prefix + "_coverage_num_important": 0}
    if importance is None:
        return None, {prefix + "_coverage_score": None, prefix + "_coverage_num_important": 0}
    imp = np.asarray(importance, dtype=float)
    if imp.shape[0] != Xc.shape[0]:
        return None, {prefix + "_coverage_score": None, prefix + "_coverage_num_important": 0}
    imp = np.maximum(np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    positive = np.where(imp > 0)[0]
    if positive.size == 0:
        return 0.0, {prefix + "_coverage_score": 0.0, prefix + "_coverage_num_important": 0, prefix + "_coverage_weighted_fraction": 0.0}
    if top_n is None:
        n = max(1, int(math.ceil(float(top_frac) * positive.size)))
    else:
        n = max(1, min(int(top_n), positive.size))
    chosen = positive[np.argsort(-imp[positive], kind="mergesort")[:n]]
    covered = np.minimum(1.0, np.max(Xc[chosen, :], axis=1) / max(float(min_loading), 1e-50))
    score = _clip01(float(np.sum(covered * imp[chosen]) / (np.sum(imp[chosen]) + 1e-50)))
    return score, {prefix + "_coverage_score": score, prefix + "_coverage_num_important": int(chosen.size), prefix + "_coverage_weighted_fraction": score}


def _sample_entries(shape, nonzero_mask=None, max_entries=1_000_000):
    rows, cols = shape
    total = int(rows) * int(cols)
    if total <= max_entries:
        return None
    rng = np.random.default_rng(0)
    count = int(max_entries)
    flat = rng.choice(total, size=count, replace=False)
    return np.unravel_index(flat, shape)


def _reconstruction_and_coherence(inputs, Wc, Hc):
    target = _dense(inputs.target_matrix)
    weights = _dense(inputs.target_weight_matrix)
    out = {}
    per_factor = []
    if target is None or Wc is None or Wc.ndim != 2 or Wc.shape[1] == 0:
        return None, None, out, per_factor
    if str(inputs.discovery_model) == "gene_by_gene":
        Wt = Wc
        if inputs.target_gene_indices is not None:
            W_full = Wc
            idx = np.asarray(inputs.target_gene_indices, dtype=int)
            if W_full.shape[0] > np.max(idx, initial=-1):
                Wt = W_full[idx, :]
        if target.shape[0] != Wt.shape[0]:
            return None, None, {"reconstruction_score": None, "coherence_score": None}, per_factor
        pred = Wt @ Wt.T
        tri = np.triu_indices(target.shape[0], k=1)
        y = target[tri]
        yhat = pred[tri]
        w = np.ones_like(y) if weights is None else weights[tri]
        valid = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
        if not np.any(valid):
            return None, None, {"reconstruction_score": None, "coherence_score": None}, per_factor
        y = y[valid]
        yhat = yhat[valid]
        w = w[valid]
        mean_y = float(np.sum(w * y) / (np.sum(w) + 1e-50))
        sse = float(np.sum(w * (y - yhat) ** 2))
        null = float(np.sum(w * (y - mean_y) ** 2))
        r2 = 1.0 - sse / (null + 1e-50)
        recon = _clip01(r2)
        bg = mean_y
        coh_scores = []
        masses = np.sum(Wt, axis=0)
        for k in range(Wt.shape[1]):
            wk = Wt[:, k]
            pair_w = wk[tri[0]] * wk[tri[1]]
            pair_w = pair_w[valid]
            denom = float(np.sum(pair_w))
            coh = 0.0 if denom <= 0 else float(np.sum(pair_w * y) / denom)
            score = _clip01(coh / (coh + bg + 1e-50))
            coh_scores.append(score)
            per_factor.append({"factor_coherence_score": score, "within_factor_target_mean": coh, "within_factor_target_enrichment": coh / (bg + 1e-50)})
        coherence = _clip01(_weighted_mean(coh_scores, masses))
        out.update({"reconstruction_r2": float(r2), "reconstruction_sse": sse, "reconstruction_null_sse": null, "reconstruction_score": recon, "reconstruction_num_sampled_entries": int(y.size), "coherence_score": coherence})
        return recon, coherence, out, per_factor

    # gene_by_annotation: target is annotation x gene; prediction is H x W^T.
    Wt = Wc
    Ht = Hc
    if inputs.target_gene_indices is not None:
        idx = np.asarray(inputs.target_gene_indices, dtype=int)
        if Wc.shape[0] > np.max(idx, initial=-1):
            Wt = Wc[idx, :]
    if Hc is not None and inputs.target_annotation_indices is not None:
        idx = np.asarray(inputs.target_annotation_indices, dtype=int)
        if Hc.shape[0] > np.max(idx, initial=-1):
            Ht = Hc[idx, :]
    if Ht is None or target.shape[0] != Ht.shape[0] or target.shape[1] != Wt.shape[0]:
        return None, None, {"reconstruction_score": None, "coherence_score": None}, per_factor
    pred = Ht @ Wt.T
    index = _sample_entries(target.shape)
    if index is None:
        y = target.ravel()
        yhat = pred.ravel()
        w = np.ones_like(y) if weights is None else weights.ravel()
    else:
        y = target[index]
        yhat = pred[index]
        w = np.ones_like(y) if weights is None else weights[index]
    valid = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return None, None, {"reconstruction_score": None, "coherence_score": None}, per_factor
    y = y[valid]
    yhat = yhat[valid]
    w = w[valid]
    mean_y = float(np.sum(w * y) / (np.sum(w) + 1e-50))
    sse = float(np.sum(w * (y - yhat) ** 2))
    null = float(np.sum(w * (y - mean_y) ** 2))
    r2 = 1.0 - sse / (null + 1e-50)
    recon = _clip01(r2)
    masses = np.sum(Wt, axis=0)
    coh_scores = []
    for k in range(Wt.shape[1]):
        outer = np.outer(Ht[:, k], Wt[:, k])
        if index is None:
            pw = outer.ravel()[valid]
        else:
            pw = outer[index][valid]
        denom = float(np.sum(pw))
        coh = 0.0 if denom <= 0 else float(np.sum(pw * y) / denom)
        score = _clip01(coh / (coh + mean_y + 1e-50))
        coh_scores.append(score)
        per_factor.append({"factor_coherence_score": score, "within_factor_target_mean": coh, "within_factor_target_enrichment": coh / (mean_y + 1e-50)})
    coherence = _clip01(_weighted_mean(coh_scores, masses))
    out.update({"reconstruction_r2": float(r2), "reconstruction_sse": sse, "reconstruction_null_sse": null, "reconstruction_score": recon, "reconstruction_num_sampled_entries": int(y.size), "coherence_score": coherence})
    return recon, coherence, out, per_factor


def _factor_balance(Wc):
    if Wc is None or Wc.ndim != 2 or Wc.shape[1] == 0:
        return None, {"factor_balance_score": None}
    masses = np.sum(Wc, axis=0)
    total = float(np.sum(masses))
    if total <= 0 or len(masses) <= 1:
        score = 1.0 if len(masses) == 1 else 0.0
        return score, {"factor_mass_entropy": score, "factor_balance_score": score, "factor_mass_top_share": 1.0 if len(masses) else 0.0, "factor_mass_q90_q10_ratio": 0.0}
    r = masses / total
    entropy = float(-np.sum(r * np.log(r + 1e-50)) / math.log(len(masses)))
    return _clip01(entropy), {
        "factor_mass_entropy": _clip01(entropy),
        "factor_balance_score": _clip01(entropy),
        "factor_mass_top_share": float(np.max(r)),
        "factor_mass_q90_q10_ratio": _safe_quantile(masses, 0.9) / (_safe_quantile(masses, 0.1) + 1e-50),
    }


def _annotation_bridge_qc(inputs, annotation_importance):
    records = inputs.annotation_bridge_metrics or []
    if not records:
        return None, {"annotation_bridge_qc_score": None, "annotation_bridge_suggested_exclude_count": 0}
    flags = np.asarray([1.0 if str(row.get("flag_suggest_exclude", "")).lower() in {"1", "true", "yes"} else 0.0 for row in records], dtype=float)
    if annotation_importance is not None and len(annotation_importance) == len(flags):
        q = np.maximum(np.nan_to_num(np.asarray(annotation_importance, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    else:
        q = np.ones_like(flags)
    frac = float(np.sum(q * flags) / (np.sum(q) + 1e-50)) if flags.size else 0.0
    return _clip01(1.0 - frac), {
        "annotation_bridge_suggested_exclude_count": int(np.sum(flags > 0)),
        "annotation_bridge_weighted_exclude_fraction": _clip01(frac),
        "annotation_bridge_qc_score": _clip01(1.0 - frac),
    }


def score_phi_candidate(phi: float, num_factors: int, inputs: PhiSelectionInputs, config: CompositePhiSelectionConfig):
    Wc = _bounded(inputs.gene_loadings, config.loading_cap)
    Hc = _bounded(inputs.annotation_loadings, config.loading_cap)
    gene_importance = None if inputs.gene_importance is None else np.asarray(inputs.gene_importance, dtype=float)
    annotation_importance = None if inputs.annotation_importance is None else np.asarray(inputs.annotation_importance, dtype=float)

    wide = {"phi": float(phi), "num_factors": int(num_factors)}
    component_scores: dict[str, float | None] = {}
    unavailable: dict[str, str] = {}
    per_factor: list[dict[str, Any]] = []

    factor_size_score, metrics, per_factor = _factor_size_metrics(Wc, config)
    component_scores["factor_size"] = factor_size_score
    wide.update(metrics)
    if factor_size_score is None:
        unavailable["factor_size"] = "gene loadings unavailable"

    score, metrics = _nonoverlap_metrics(Wc, Hc)
    component_scores["nonoverlap"] = score
    wide.update(metrics)
    if score is None:
        unavailable["nonoverlap"] = "at least two factors are required"

    gene_conc, metrics = _entity_concentration_one(Wc, gene_importance, config, "gene")
    wide.update(metrics)
    ann_conc, ann_metrics = _entity_concentration_one(Hc, annotation_importance, config, "annotation")
    wide.update(ann_metrics)
    if gene_conc is None and ann_conc is None:
        component_scores["entity_concentration"] = None
        unavailable["entity_concentration"] = "gene and annotation loadings unavailable"
    elif ann_conc is None:
        component_scores["entity_concentration"] = gene_conc
    elif gene_conc is None:
        component_scores["entity_concentration"] = ann_conc
    else:
        component_scores["entity_concentration"] = _clip01(0.70 * gene_conc + 0.30 * ann_conc)
    wide["entity_concentration_score"] = component_scores["entity_concentration"]

    gene_cov, metrics = _coverage_one(Wc, gene_importance, config.gene_coverage_top_frac, config.gene_coverage_top_n, config.coverage_min_loading, "gene")
    wide.update(metrics)
    ann_cov, metrics = _coverage_one(Hc, annotation_importance, config.annotation_coverage_top_frac, config.annotation_coverage_top_n, config.coverage_min_loading, "annotation")
    wide.update(metrics)
    if gene_cov is None and ann_cov is None:
        component_scores["coverage"] = None
        unavailable["coverage"] = "importance vectors unavailable"
    elif ann_cov is None:
        component_scores["coverage"] = gene_cov
    elif gene_cov is None:
        component_scores["coverage"] = ann_cov
    else:
        component_scores["coverage"] = _clip01(0.60 * gene_cov + 0.40 * ann_cov)
    wide["coverage_score"] = component_scores["coverage"]

    recon, coh, metrics, coherence_per_factor = _reconstruction_and_coherence(inputs, Wc, Hc)
    wide.update(metrics)
    component_scores["reconstruction"] = recon
    component_scores["coherence"] = coh
    if recon is None:
        unavailable["reconstruction"] = "target matrix unavailable or shape mismatch"
    if coh is None:
        unavailable["coherence"] = "target matrix unavailable or shape mismatch"
    for i, row in enumerate(coherence_per_factor):
        if i < len(per_factor):
            per_factor[i].update(row)

    score, metrics = _factor_balance(Wc)
    wide.update(metrics)
    component_scores["factor_balance"] = score
    if score is None:
        unavailable["factor_balance"] = "gene loadings unavailable"

    score, metrics = _annotation_bridge_qc(inputs, annotation_importance)
    wide.update(metrics)
    component_scores["annotation_bridge_qc"] = score
    if score is None:
        unavailable["annotation_bridge_qc"] = "annotation bridge metrics unavailable"

    active = []
    for component in COMPONENT_ORDER:
        score = component_scores.get(component)
        weight = float(config.weights.get(component, 0.0))
        if score is None or weight <= 0.0:
            continue
        active.append((component, float(score), weight))
    weight_sum = float(sum(weight for _, _, weight in active))
    composite = 0.0 if weight_sum <= 0 else float(sum(score * weight for _, score, weight in active) / weight_sum)
    wide["phi_composite_score"] = _clip01(composite)

    long_rows = []
    for component in COMPONENT_ORDER:
        score = component_scores.get(component)
        weight = float(config.weights.get(component, 0.0))
        available = score is not None
        normalized = 0.0 if not available or weight_sum <= 0 or weight <= 0 else weight / weight_sum
        contribution = 0.0 if not available else normalized * float(score)
        long_rows.append(
            {
                "phi": float(phi),
                "component": component,
                "score": None if score is None else _clip01(float(score)),
                "weight": weight,
                "normalized_weight": normalized,
                "weighted_contribution": contribution,
                "available": bool(available),
                "unavailable_reason": "" if available else unavailable.get(component, "unavailable"),
            }
        )
        wide[component + "_score"] = None if score is None else _clip01(float(score))
        wide[component + "_weight"] = weight
        wide[component + "_weighted_contribution"] = contribution
    return wide, long_rows, per_factor


def select_composite_candidate(candidates: list[dict[str, Any]], tie_tolerance: float):
    if not candidates:
        raise ValueError("no phi candidates to select")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row.get("phi_composite_score", 0.0)),
            int(row.get("modal_factor_count", row.get("num_factors", 0))),
            -float(row.get("coverage_score", 0.0) or 0.0),
            float(row.get("phi", 0.0)),
        ),
    )
    best_score = float(ranked[0].get("phi_composite_score", 0.0))
    frontier = [row for row in ranked if best_score - float(row.get("phi_composite_score", 0.0)) <= float(tie_tolerance) + 1e-12]
    selected = sorted(
        frontier,
        key=lambda row: (
            int(row.get("modal_factor_count", row.get("num_factors", 0))),
            -float(row.get("coverage_score", 0.0) or 0.0),
            float(row.get("phi", 0.0)),
        ),
    )[0]
    for rank, candidate in enumerate(ranked, start=1):
        candidate["selection_rank"] = int(rank)
        candidate["selected"] = candidate is selected
    selected["selection_pool"] = "composite_tie_frontier" if len(frontier) > 1 else "composite_best"
    selected["selection_frontier_size"] = int(len(frontier))
    selected["selection_marginal_gain"] = None
    return selected, "composite_score"
