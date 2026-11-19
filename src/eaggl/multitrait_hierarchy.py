"""Build a hierarchy from labelled, independently fitted gene-loading tables."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .hierarchy import build_tree, cut_tree


def load_leaves(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t"), [])
    if len(set(header)) != len(header):
        raise ValueError(f"{path}: repeated column name")
    factors = sorted((col for col in header if re.fullmatch(r"Factor[1-9]\d*", col)),
                     key=lambda col: int(col[6:]))
    if "Gene" not in header or not factors:
        raise ValueError(f"{path}: expected a Gene column and raw Factor1, Factor2, ... columns")
    table = pd.read_csv(path, sep="\t", usecols=["Gene", *factors],
                        dtype={"Gene": str}, keep_default_na=False, float_precision="round_trip").set_index("Gene")
    if table.empty or table.index.has_duplicates or any(not g.strip() for g in table.index):
        raise ValueError(f"{path}: expected nonempty, unique gene identifiers")
    weights = table[factors].astype(float)
    values = weights.to_numpy()
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or (values.max(axis=0) == 0).any()):
        raise ValueError(f"{path}: each leaf needs finite [0,1] loadings and positive support")
    return weights


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def integrate(files, output, cut=.50, duplicate_cosine=.90, min_norm_ratio=.80):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not (0 <= cut <= 1 and 0 < duplicate_cosine <= 1 and 0 < min_norm_ratio <= 1):
        raise ValueError("Expected cut in [0,1] and redundancy thresholds in (0,1]")
    labels = [trait for trait, _ in files]
    if len(set(labels)) != len(files) or len(files) < 2 or any(not t.strip() for t in labels):
        raise ValueError("Supply at least two distinct, nonempty trait labels")
    frames, sources, panels = {}, {}, {}
    for trait, filename in sorted(files):
        path = Path(filename).resolve()
        frame = load_leaves(path)
        frames[trait] = frame
        panels[trait] = set(frame.index)
        sources[trait] = dict(path=str(path), sha256=sha256(path), genes=len(frame), leaves=len(frame.columns))
    union = pd.concat(frames, axis=1).sort_index()
    genes = union.index.to_numpy(dtype=str)
    # Missing rows mean no observed contribution. Measure each norm over its whole supplied panel.
    w = union.fillna(0.).to_numpy()
    norms = np.linalg.norm(w, axis=0)
    unit = w / norms
    ids = np.arange(1, w.shape[1]+1)
    leaves = pd.DataFrame(union.columns.tolist(), columns=["trait", "source_factor"])
    leaves.insert(0, "leaf_id", pd.array(ids, dtype="Int64"))
    ranges = leaves.groupby("trait").indices
    matches, overlaps = [], []
    counts = Counter()
    comparable_counts = Counter()
    for a, b in combinations(frames, 2):
        ia, ib = ranges[a], ranges[b]
        common = np.isin(genes, list(panels[a] & panels[b]))
        sim = np.clip(unit[:, ia].T @ unit[:, ib], 0, 1)
        coverage_a = (unit[common][:, ia] ** 2).sum(axis=0)
        coverage_b = (unit[common][:, ib] ** 2).sum(axis=0)
        overlaps.append(dict(trait_a=a, trait_b=b, shared_genes=int(common.sum()),
                             genes_a=len(panels[a]), genes_b=len(panels[b]), max_cosine=float(sim.max())))
        for i, j in zip(*np.nonzero(sim >= duplicate_cosine - 1e-12)):
            x, y = ia[i], ib[j]
            ratio = min(norms[x], norms[y]) / max(norms[x], norms[y])
            comparable = bool(ratio >= min_norm_ratio - 1e-12)
            counts.update((ids[x], ids[y]))
            if comparable:
                comparable_counts.update((ids[x], ids[y]))
            matches.append([ids[x], a, leaves.source_factor[x], ids[y], b, leaves.source_factor[y],
                            sim[i,j], ratio, coverage_a[i], coverage_b[j], True, comparable])
    pairs = pd.DataFrame(matches, columns=["leaf_a","trait_a","factor_a","leaf_b","trait_b","factor_b",
                                          "cosine","norm_ratio","shared_energy_a","shared_energy_b","redundant","comparable_strength"])
    leaves["redundant_partners"] = [counts[k] for k in ids]
    leaves["redundant"] = leaves.redundant_partners > 0
    leaves["comparable_strength_partners"] = [comparable_counts[k] for k in ids]
    nodes, values, members = build_tree(w, ids, full_diagnostics=False)
    tree = pd.DataFrame(nodes)
    for col in ("parent","left","right","leaf_id"):
        tree[col] = tree[col].astype("Int64")
    tree = tree.merge(leaves, how="left", on="leaf_id", validate="many_to_one", sort=False)
    overview = []
    for node in cut_tree(nodes, cut):
        for k in members[node]:
            overview.append((node, k))
    groups = pd.DataFrame(overview, columns=["node","leaf_id"]).merge(leaves, on="leaf_id", validate="one_to_one")
    manifest = dict(version=1, method="complete linkage on leaf cosine", parent="noisy OR",
                    gene_alignment="union of exact gene IDs; absent rows contribute zero",
                    min_leaf_cosine=cut, redundant_cosine=duplicate_cosine, min_norm_ratio=min_norm_ratio,
                    sources=sources, genes=len(genes), leaves=len(ids), nodes=len(nodes),
                    overview_groups=int(groups.node.nunique()), redundant_pairs=len(pairs), comparable_strength_pairs=sum(comparable_counts.values())//2)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp)/"tree"
        stage.mkdir()
        for name, table in [("factor_hierarchy",tree), ("leaf_sources",leaves), ("overview",groups),
                            ("redundancy_pairs",pairs), ("gene_panel_overlap",pd.DataFrame(overlaps))]:
            table.to_csv(stage/f"{name}.tsv.gz", sep="\t", index=False)
        np.savez_compressed(stage/"hierarchy_loadings.npz", genes=genes, node=tree.node.to_numpy(), W=values,
                            traits=np.array(list(frames)), gene_observed=np.column_stack(
                                [np.isin(genes, list(panel)) for panel in panels.values()]))
        (stage/"hierarchy.json").write_text(json.dumps(manifest, indent=2)+"\n")
        stage.rename(output)
    return manifest


def main(argv):
    parser = argparse.ArgumentParser(prog="eaggl hierarchy", description=__doc__)
    parser.add_argument("--leaf-file", nargs=2, action="append", metavar=("TRAIT","PATH"), required=True,
                        help="Repeat once per trait. TSV or TSV.gz with Gene and raw FactorN columns.")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--min-leaf-cosine", type=float, default=.50, help="Overview cut (default 0.50)")
    parser.add_argument("--redundant-cosine", type=float, default=.90, help="Profile similarity screen (default 0.90)")
    parser.add_argument("--min-norm-ratio", type=float, default=.80, help="Minimum smaller/larger loading norm for the comparable-strength flag (default 0.80)")
    args = parser.parse_args(argv)
    try:
        manifest = integrate(args.leaf_file, args.output, args.min_leaf_cosine, args.redundant_cosine, args.min_norm_ratio)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"{manifest['leaves']} leaves, {manifest['overview_groups']} overview groups, "
          f"{manifest['redundant_pairs']} redundant cross-trait pairs")
    return 0
