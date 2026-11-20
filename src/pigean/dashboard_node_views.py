"""Visual and phenotype readouts on a frozen EAGGL hierarchy."""
import base64
import csv
import gzip
import tempfile
from pathlib import Path

import numpy as np
from eaggl import factor_graph as fg
from eaggl.phewas import _fit_ols_with_selected_coefficients


def packed(values):
    return base64.b64encode(np.ascontiguousarray(values, dtype="<f4").tobytes()).decode("ascii")


def read_reference(path):
    """Retain the binary trait response used by EAGGL for Combined > 1."""
    hits = {}
    with gzip.open(path, "rt") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            genes = hits.setdefault(row["Trait_Internal"], set())
            if float(row["Combined"]) > 1:
                genes.add(row["Gene"])
            else:
                genes.discard(row["Gene"])
    return hits


def marginal_scan(w, genes, anchor, reference):
    traits = sorted(reference)
    indices = {g: i for i, g in enumerate(genes)}
    effects = np.full((len(traits), w.shape[1]), np.nan, dtype=np.float32)
    errors = np.full_like(effects, np.nan)
    counts = []
    designs = [np.column_stack((column, anchor, np.ones(len(genes)))) for column in w.T]
    valid = [np.linalg.matrix_rank(x) == x.shape[1] for x in designs]
    for start in range(0, len(traits), 192):
        subset = traits[start:start+192]
        y = np.zeros((len(genes), len(subset)))
        for j, trait in enumerate(subset):
            rows = [indices[g] for g in reference[trait] if g in indices]
            y[rows, j] = 1
        n = y.sum(axis=0)
        counts.extend(n.astype(int).tolist())
        ok = (n > 0) & (n < len(genes))
        if not np.any(ok):
            continue
        y = y[:, ok]
        dest = np.flatnonzero(ok) + start
        for col, design in enumerate(designs):
            if valid[col]:
                beta, se, *_ = _fit_ols_with_selected_coefficients(design, y, [0], se_type="robust")
                effects[dest, col] = beta[0]
                errors[dest, col] = se[0]
    return {"rows": traits, "hits": counts, "beta": packed(effects), "se": packed(errors)}


def leaf_profile_layout(w, anchor_columns):
    """Classical MDS for cosine chord distance, with supplementary profiles.

    Unit-length gene profiles give squared Euclidean distance 2(1-cosine).
    Center and fit on leaves only. Project every column using the same center
    and principal axes. A zero profile has no direction and sits at the origin.
    """
    lengths = np.linalg.norm(w, axis=0)
    profiles = np.divide(w.T, lengths[:, None], out=np.zeros_like(w.T, dtype=float),
                         where=lengths[:, None] > 0)
    anchors = profiles[anchor_columns]
    if not len(anchors) or np.any(lengths[anchor_columns] == 0):
        raise ValueError("A leaf MDS anchor needs a nonzero gene-loading profile")
    center = anchors.mean(axis=0)
    centered = anchors - center
    values, vectors = np.linalg.eigh(centered @ centered.T)
    coords = np.zeros((w.shape[1], 2))
    for axis, j in enumerate(range(len(values) - 1, max(-1, len(values) - 3), -1)):
        if values[j] <= np.finfo(float).eps * max(values[-1], 1) * len(values):
            break
        direction = centered.T @ vectors[:, j] / np.sqrt(values[j])
        # Fix the arbitrary axis sign by its strongest gene coefficient.
        direction *= np.sign(direction[np.argmax(np.abs(direction))])
        coords[:, axis] = (profiles - center) @ direction
    coords[lengths == 0] = 0
    return coords


def graph_html(keys, labels, names, w, direct, combined, anchor_columns, *, named=False):
    # Match the native graph's three representative genes per factor. Rank on
    # each node independently, since broad OR parents can dominate a gene row.
    edges = {}
    for k, column in enumerate(w.T):
        for i in np.argsort(-column, kind="stable")[:3]:
            if column[i] >= 0.01:
                edges.setdefault(i, {})[keys[k]] = float(column[i])
    for i, weights in edges.items():
        for k in np.flatnonzero(w[i] >= max(.01, .5*w[i].max())):
            weights[keys[k]] = float(w[i,k])
    genes = [fg.EntityInfo(str(names[i]), str(names[i]), "gene", combined[i], direct[i], weights)
             for i, weights in edges.items()]
    genes = fg._scale_entity_values(genes)
    factors = [fg.FactorInfo(key, label if named else f"{label} · {names[np.argmax(w[:,j])]}", 1.)
               for j, (key, label) in enumerate(zip(keys, labels))]
    graph = fg.build_graph(factors, genes, [], fg.GraphConfig(max_num_factor_nodes=len(keys)))
    xy = leaf_profile_layout(w, anchor_columns)
    factor_index = {key: j for j, key in enumerate(keys)}
    gene_index = {str(gene): i for i, gene in enumerate(names)}
    leaf_columns = [j for j, key in enumerate(keys) if key.startswith("f")]
    for node in graph["nodes"]:
        if node["kind"] == "factor":
            position = xy[factor_index[node["id"]]]
            if named:
                node["display_label"] = node["label"]
        else:
            # Use original fitted loadings even when the graph draws only a few edges.
            weights = w[gene_index[node["id"]], leaf_columns]
            position = np.average(xy[leaf_columns], weights=weights, axis=0)
        node["x"], node["y"] = map(float, position)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "factor_graph.html"
        fg.write_html(graph, path, physics=False, equal_aspect=True)
        return path.read_text()


def make_views(h, fit_w, fit_h, node_w, node_columns, genes, annotations, anchor_file, reference, *,
               node_labels=None, include_graph=True):
    leaves = h["leaves"]
    keys = ["f"+str(f["id"]) for f in leaves]
    labels = [f"Factor {f['id']}" + (" · flagged" if f["flagged"] else "") for f in leaves]
    columns = {}
    parents_w, parents_h = [], []
    for i, n in sorted(h["nodes"].items()):
        if n["factor"]:
            index = n["factor"]-1
            columns[i] = fit_h[:, index]
            continue
        a, b = (columns[c] for c in n["children"])
        annotation = a + (1-a)*b
        columns[i] = annotation
        parents_w.append(node_w[:, node_columns[i]])
        parents_h.append(annotation)
        if node_labels is not None:
            ranked = np.lexsort((annotations, -annotation))[:15]
            n["annotations"] = [[str(annotations[j]), float(annotation[j])] for j in ranked if annotation[j] > 0]
        keys.append(f"n{i}")
        labels.append(f"Family {i}")
    w = np.column_stack([fit_w, *parents_w])
    annotation = np.column_stack([fit_h, *parents_h])
    if np.any(fit_h < 0) or np.any(fit_h > 1+1e-8):
        raise ValueError("Cannot form OR summaries from unbounded annotation loadings")
    direct = combined = np.zeros(len(genes))
    if anchor_file:
        with gzip.open(anchor_file, "rt") as stream:
            rows = {r["Gene"]: r for r in csv.DictReader(stream, delimiter="\t")}
        direct = np.array([float(rows[g]["log_bf"]) for g in genes])
        combined = np.array([float(rows[g]["combined"]) for g in genes])
    out = {"keys": keys, "labels": labels,
           "genes": {"rows": genes.tolist(), "data": packed(w)},
           "gene_sets": {"rows": annotations.tolist(), "data": packed(annotation)}}
    if node_labels is not None:
        from eaggl.hierarchy_labels import fit_signature
        signature = fit_signature(h, out)
        if node_labels.keys() != set(keys) or any(digest != signature for digest, _ in node_labels.values()):
            raise ValueError(f"Hierarchy labels do not match the fit and node IDs for {h['trait']}")
        h["leafLabels"] = {key: name for key, (_, name) in node_labels.items() if key.startswith("f")}
        h["familyLabels"] = {key: name for key, (_, name) in node_labels.items() if key.startswith("n")}
        labels = [f"{node_labels[key][1]} · {label}" for key, label in zip(keys, labels)]
        out["labels"] = labels
    if reference is not None:
        if not anchor_file:
            raise ValueError(f"A matched --pigean-run is required for {h['trait']} anchor adjustment")
        out["phewas"] = marginal_scan(w, genes, direct, reference)
    anchor_columns = [i for i, leaf in enumerate(leaves) if not leaf["flagged"]]
    graph = (graph_html(keys, labels, genes, w, direct, combined, anchor_columns, named=node_labels is not None)
             if include_graph else None)
    return out, graph
