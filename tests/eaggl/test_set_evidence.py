from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.sparse import csr_matrix

from eaggl import labeling
from eaggl.set_evidence import from_state, metric_values


def test_metrics_use_only_discovery_genes_and_handle_empty_sets_and_factors():
    state = SimpleNamespace(
        genes=["A", "outside", "B"], gene_sets=["set", "empty"],
        X_orig=csr_matrix([[.1,0],[1,1],[0,0]]),
        exp_gene_factors=np.array([[.8,0],[1,1],[.2,0]]),
        gene_in_discovery_mask=[True,False,True])
    evidence = from_state(state)
    assert metric_values(evidence, 0, [0]) == ["2","1","0.8","1.6","0.8","A"]
    assert metric_values(evidence, 1, [0]) == ["2","0","0","NA","NA",""]
    assert metric_values(evidence, 0, [1]) == ["2","1","NA","NA","0",""]


def test_single_factor_failure_keeps_alignment_and_skips_empty_factors():
    client = Mock(model="local")
    client.query.side_effect = ["**Insulin secretion**", "1. X\n2. Y"]
    fallback = ["a", "b", "c"]
    labels = labeling.assign_labels(fallback, ["first", "", "third"], client, Mock(), Mock())
    assert labels == ["Insulin secretion", "b", "c"]
    assert client.query.call_count == 2
    assert "third" not in client.query.call_args_list[0].args[0]


def test_legacy_cluster_switch_and_evidence_roundtrip(tmp_path):
    import csv
    from tests.eaggl.test_label_mode_unittest import EagglLabelModeTest
    source = tmp_path/"sets.tsv"
    source.write_text("Gene_Set\tFactor2\tCoverage_Factor2\tEnrichment_Factor2\tSupportingGenes_Factor2\n"
                      "islets\t.7\t.4\t4\tGCK,PDX1\n")
    for mode in ("evidence","legacy"):
        out = tmp_path/f"{mode}.tsv"
        result = EagglLabelModeTest()._run("label","--label-gene-set-clusters-in",str(source),
            "--gene-set-clusters-out",str(out),"--gene-set-clusters-format",mode)
        assert result.returncode == 0, result.stderr
        with out.open() as stream:
            row = next(csv.DictReader(stream,delimiter="\t"))
        assert row["cluster"] == "Factor2"
        assert float(row["Factor2"]) == .7
        assert ("Cosine_Factor2" in row) == (mode == "legacy")
        assert ("Coverage_Factor2" in row) == (mode == "evidence")
        if mode == "evidence":
            assert row["SupportingGenes_Factor2"] == "GCK,PDX1"

def test_ranked_union_counts_shared_members_once_and_breaks_ties_by_set_id():
    from eaggl.set_evidence import SetEvidence
    evidence = SetEvidence(csr_matrix([[1,1,0],[0,1,0],[0,0,1]]), np.array([[.5],[.25],[.25]]), ["A","B","C"])
    actual = evidence.cumulative_coverage(np.array([[.9],[.9],[.3]]), ["z","a","c"])
    np.testing.assert_allclose(actual[:,0], [.75,.75,1.])
