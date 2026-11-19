import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from eaggl.app import main
from eaggl.multitrait_hierarchy import integrate, load_leaves


def test_integrated_cli(tmp_path):
    a = tmp_path/"a.tsv"
    b = tmp_path/"b.tsv.gz"
    a.write_text("Gene\tFactor2\tFactor4\tCosine_Factor2\n001\t1\t0\t.1\n002\t0\t.9\t.2\nNA\t0\t0\t.3\n")
    with gzip.open(b, "wt") as stream:
        stream.write("Gene\tFactor2\tFactor7\tFactor9\n002\t0\t0.3\t0\n001\t1\t0\t0\n003\t0\t0\t1\n")
    dest = tmp_path/"result"
    assert main(["hierarchy","--leaf-file","B",str(b),"--leaf-file","A",str(a),"--output",str(dest)]) == 0
    read = lambda name: pd.read_csv(dest/f"{name}.tsv.gz", sep="\t")
    pairs = read("redundancy_pairs")
    assert set(zip(pairs.factor_a, pairs.factor_b, pairs.comparable_strength)) == {
        ("Factor2","Factor2",True), ("Factor4","Factor7",False)}
    sources = read("leaf_sources")
    assert pairs.redundant.all()
    assert sources.redundant.sum() == 4
    assert sources.redundant_partners.sum() == 4
    assert sources.comparable_strength_partners.sum() == 2
    cut = read("overview")
    assert len(cut) == 5 and cut.node.nunique() == 3
    assert cut.leaf_id.is_unique
    tree = read("factor_hierarchy")
    with np.load(dest/"hierarchy_loadings.npz") as z:
        assert z["genes"].tolist() == ["001","002","003","NA"]
        assert z["gene_observed"].tolist() == [[True,True],[True,True],[False,True],[True,False]]
        for row in tree.dropna(subset=["left"]).itertuples():
            x, y = z["W"][:,int(row.left)], z["W"][:,int(row.right)]
            np.testing.assert_allclose(z["W"][:,row.node], 1-(1-x)*(1-y))
    second = tmp_path/"second"
    integrate([("A",a),("B",b)], second)
    pd.testing.assert_frame_equal(tree, pd.read_csv(second/"factor_hierarchy.tsv.gz",sep="\t"))
    with pytest.raises(ValueError, match="already exists"):
        integrate([("A",a),("B",b)], dest)


def test_unshared_energy_prevents_false_redundancy(tmp_path):
    files = []
    for trait in ["A","B"]:
        path = tmp_path/f"{trait}.tsv"
        path.write_text(f"Gene\tFactor1\nshared\t0.1\n{trait}\t1\n")
        files.append((trait,path))
    dest = tmp_path/"result"
    manifest = integrate(files, dest)
    assert manifest["redundant_pairs"] == 0
    overlap = pd.read_csv(dest/"gene_panel_overlap.tsv.gz", sep="\t").iloc[0]
    assert overlap.shared_genes == 1 and overlap.max_cosine < .01
    with pytest.raises(ValueError, match="distinct"):
        integrate(files[:1]*2, tmp_path/"invalid")


@pytest.mark.parametrize("content", [
    "Gene\tFactor1\tFactor1\ng\t0.2\t0.3\n", "Gene\tFactor1\ng\t1\ng\t.5\n",
    "Gene\tFactor1\nf\t0\n", "Gene\tFactor1\ng\t2\n", "Gene\tFactor1\ng\tnan\n",
    "Gene\tCosine_Factor1\ng\t.2\n",
])
def test_malformed_leaf_files(tmp_path, content):
    path = tmp_path/"bad.tsv"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_leaves(path)
