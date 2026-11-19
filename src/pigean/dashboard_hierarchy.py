"""Read an exported EAGGL hierarchy and its matched leaf-selection results."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _top(names, values, limit=15):
    indices = np.argsort(-values, kind="stable")[:limit]
    return [[str(names[i]), float(values[i])] for i in indices if values[i] > 0]


def load_hierarchy(directory: Path, source_root: Path, *, anchors=None, reference=None) -> dict:
    manifest = json.loads((directory / "hierarchy.json").read_text())
    sources = {}
    for key, entry in manifest["source_files"].items():
        path = source_root / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Hierarchy source checksum mismatch for {path}")
        sources[key] = path

    decision = json.loads(sources["decision"].read_text())
    summaries = _rows(sources["decision"].with_name("phi_summary.tsv.gz"))
    candidates = []
    for row in summaries:
        candidates.append({
            "phi": float(row["phi"]), "factors": int(row["factors"]), "flags": int(row["flagged"]),
            "global": float(row["global_error"]), "balanced": float(row["balanced_error"]),
            "error": float(row["distortion"]), "flaggedEnergy": float(row["flagged_energy"]),
            "qualified": row["numerically_qualified"] == "True", "usable": row["leaf_usable"] == "True",
            "eligible": row["id"] in decision["eligible"], "within": row["id"] in decision["within_tolerance"],
            "selected": row["id"] == decision["selected"],
        })
    selected = [c for c in candidates if c["selected"]]
    if len(selected) != 1 or selected[0]["phi"] != manifest["phi"] or decision["selected_phi"] != manifest["phi"]:
        raise ValueError("Hierarchy and candidate selection disagree on phi")

    with np.load(sources["axes"]) as axes, np.load(sources["loadings"]) as fit:
        genes, annotations = axes["genes"], axes["gene_sets"]
        W, H = fit["W"], fit["H"]
    qc = _rows(sources["qc"])
    if W.shape != (len(genes), len(qc)) or H.shape != (len(annotations), len(qc)):
        raise ValueError("Selected fit does not match the gene, annotation or QC axes")
    if sorted(int(r["factor"]) for r in qc) != list(range(1, len(qc) + 1)) or len(qc) != selected[0]["factors"]:
        raise ValueError("Selected factor IDs disagree with the phi summary")
    leaves = []
    for row in sorted(qc, key=lambda r: int(r["factor"])):
        index = int(row["factor"]) - 1
        leaves.append({
            "id": index + 1, "flagged": row["flagged"] == "True", "reason": row["flag_reasons"],
            "effective": float(row["effective_genes"]), "support": row["support"],
            "genes": _top(genes, W[:, index]), "annotations": _top(annotations, H[:, index]),
        })
    flagged = {f["id"] for f in leaves if f["flagged"]}
    if flagged != set(manifest["skipped_flagged"]) or len(flagged) != selected[0]["flags"]:
        raise ValueError("QC flags disagree with the hierarchy or selected candidate")

    topology = _rows(directory / "factor_hierarchy.tsv.gz")
    nodes = {}
    with np.load(directory / "hierarchy_loadings.npz") as profiles:
        if not np.array_equal(profiles["genes"], genes):
            raise ValueError("Inconsistent hierarchy gene axis")
        node_columns = {int(n): i for i, n in enumerate(profiles["node"])}
        matrix = profiles["W"]
        for row in topology:
            index = int(row["node"])
            factor = int(row["leaf_id"]) if row["leaf_id"] else None
            weights = matrix[:, node_columns[index]]
            if factor and not np.allclose(weights, W[:, factor - 1]):
                raise ValueError(f"Hierarchy leaf differs from fitted factor {factor}")
            nodes[index] = {
                "id": index, "factor": factor, "count": int(row["n_leaves"]),
                "children": [int(row[k]) for k in ("left", "right") if row[k]],
                "genes": _top(genes, weights), "score": float(row["merge_score"]),
                "levels": {key: float(row[key]) for key in ("min_leaf_cosine", "min_shared_mass", "gene_equivalents", "merge_prefix_floor",
                    "min_shared_squared_mass", "min_both_given_either", "min_prob_containment", "min_profile_containment")},
            }
    roots = [int(r["node"]) for r in topology if not r["parent"]]
    passing = {f["id"] for f in leaves if not f["flagged"]}
    if len(roots) != 1 or {n["factor"] for n in nodes.values() if n["factor"]} != passing:
        raise ValueError("Tree does not cover the passing leaves")
    cuts = {}
    for row in _rows(directory / "hierarchy_cuts.tsv.gz"):
        key = (row["metric"], float(row["floor"]))
        cuts.setdefault(key, []).append(int(row["node"]))
    size_cuts = directory / "hierarchy_size_cuts.tsv.gz"
    if size_cuts.exists():
        for row in _rows(size_cuts):
            key = ("maximum_gene_equivalents", float(row["ceiling"]))
            cuts.setdefault(key, []).append(int(row["node"]))

    def descendants(index):
        node = nodes[index]
        if node["factor"]:
            return [node["factor"]]
        return [f for child in node["children"] for f in descendants(child)]

    frontiers = []
    for (metric, floor), ids in cuts.items():
        if sorted(f for i in ids for f in descendants(i)) != sorted(passing):
            raise ValueError(f"Incomplete hierarchy cut at {metric} {floor}")
        frontiers.append({"metric": metric, "floor": floor, "nodes": sorted(ids, key=lambda i: (-nodes[i]["count"], i))})
    hierarchy = {
        "trait": manifest["trait"], "phi": manifest["phi"], "method": manifest["method"],
        "mergeMetric": ({"cosine":"min_leaf_cosine", "shared-mass":"min_shared_mass",
                         "shared-squared-mass":"min_shared_squared_mass", "both-given-either":"min_both_given_either",
                         "prob-containment":"min_prob_containment", "profile-containment":"min_profile_containment"}
                        [manifest["overlap_rule"]] if manifest["merge_scope"] == "leaves" else "merge_prefix_floor"),
        "parent": manifest["parent"], "nodes": nodes, "root": roots[0], "cuts": frontiers,
        "leaves": leaves, "decision": decision, "candidates": candidates,
    }

    from .dashboard_node_views import make_views
    views, graph = make_views(hierarchy, W, H, matrix, node_columns, genes, annotations,
                              (anchors or {}).get(manifest["trait"]), reference)
    hierarchy["views"] = views
    hierarchy["graph"] = graph
    return hierarchy
