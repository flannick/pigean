"""Match the integrated leaf inventory to dashboard matrices and embed its cosine map."""
import base64
from pathlib import Path

import numpy as np
import pandas as pd


def build_factor_atlas(directory: Path, hierarchies: dict) -> dict:
    topology = pd.read_csv(directory / "factor_hierarchy.tsv.gz", sep="\t",
                           dtype={"trait": str}, keep_default_na=False)
    leaves = topology[topology.leaf_id != ""].copy()
    leaves["leaf_id"] = leaves.leaf_id.astype(int)
    leaves = leaves.sort_values("leaf_id")
    overview = pd.read_csv(directory / "overview.tsv.gz", sep="\t", keep_default_na=False,
                           dtype={"trait": str})
    leaves = leaves.merge(overview[["leaf_id", "node", "trait", "source_factor"]].rename(columns={"node": "family"}),
                          on=["leaf_id", "trait", "source_factor"], how="left", validate="one_to_one")
    if leaves.empty or leaves.family.isna().any() or len(leaves) != len(overview):
        raise ValueError("Incomplete integrated leaf-to-family mapping")
    with np.load(directory / "hierarchy_loadings.npz") as z:
        node_index = pd.Index(z["node"])
        gene_index = pd.Index(z["genes"])
        columns = node_index.get_indexer(leaves.node)
        if min(columns) < 0 or not gene_index.is_unique:
            raise ValueError("Integrated leaf axes do not match the tree")
        weights = z["W"][:, columns].T.copy()
        traits = z["traits"].tolist()
        panels = z["gene_observed"]
    points = []
    for trait, subset in leaves.groupby("trait", sort=False):
        if trait not in hierarchies or trait not in traits:
            raise ValueError(f"Atlas needs the corresponding --eaggl-hierarchy for {trait}")
        view = hierarchies[trait]["views"]
        block = view["genes"]
        axis = gene_index.get_indexer(block["rows"])
        panel = panels[:, traits.index(trait)]
        if (len(set(axis)) != len(axis) or len(axis) != panel.sum() or min(axis) < 0 or not panel[axis].all()
                or len(set(view["keys"])) != len(view["keys"])):
            raise ValueError(f"Discovery gene or factor axes differ for {trait}")
        original = np.frombuffer(base64.b64decode(block["data"]), dtype="<f4").reshape(len(axis),len(view["keys"]))
        for row in subset.itertuples():
            key = "f" + row.source_factor.removeprefix("Factor")
            col = view["keys"].index(key)
            if (not np.array_equal(weights[row.Index, axis].astype("<f4"), original[:, col])
                    or np.any(weights[row.Index, ~panel])):
                raise ValueError(f"Atlas loading mismatch for {trait} {row.source_factor}")
            points.append(dict(index=row.Index, trait=trait, key=key, column=col, factor=row.source_factor,
                               family=int(row.family), redundant=str(row.redundant).lower()=="true"))
    points.sort(key=lambda p: p["index"])
    if len(points) >= 3:
        try:
            from umap import UMAP
        except ImportError as error:
            raise ValueError("The factor atlas requires umap-learn in the dashboard build environment") from error
        xy = UMAP(metric="cosine", n_neighbors=min(15, len(points)-1), min_dist=0.15,
                  init="random", random_state=42, n_jobs=1).fit_transform(weights)
        method = "Cosine UMAP"
    else:
        xy = np.column_stack((np.arange(len(points)), np.zeros(len(points))))
        method = "Leaf positions"
    if not np.all(np.isfinite(xy)):
        raise ValueError("Non-finite factor atlas coordinates")
    # Rotate the arbitrary embedding axes to fit the landscape viewport without distortion.
    xy = xy - xy.mean(axis=0)
    if len(points) > 1:
        _, _, axes = np.linalg.svd(xy, full_matrices=False)
        xy = xy @ axes.T
    for point, (x, y) in zip(points, xy):
        point.update(x=float(x), y=float(y))
    return dict(points=points, method=method)
