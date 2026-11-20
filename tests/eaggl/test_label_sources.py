import csv
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eaggl import labeling


def test_seven_prompts_hold_out_other_sources_and_reuse_duplicate_combinations():
    sources = labeling.SOURCES
    groups = [combo for n in (1, 2, 3) for combo in combinations(sources, n)]
    state = SimpleNamespace(num_factors=lambda: 1, factor_top_genes=[[f"G{i}" for i in range(6)]],
        factor_top_gene_sets=[[f"S{i}" for i in range(6)]], factor_top_phenos=[[f"P{i}" for i in range(6)]])
    primary = ("genes", "gene-sets")
    options = SimpleNamespace(label_sources=None, additional_label_sources=[",".join(g) for g in groups]
                              + ["gene-sets,genes"])
    labeling.configure_sources(state, options, pytest.fail)
    entries = [[labeling.evidence_entry(s, .8, .3, 2, "G1,SECRET") for s in state.factor_top_gene_sets[0]]]
    client = Mock(model="fixture")
    client.query.side_effect = [f"Label {i}" for i in range(7)]
    labeling.apply_source_labels(state, entries, client=client, sets_only=False,
        include_phenos=False, individually=False, log=Mock(), warn=Mock())
    assert client.query.call_count == 7
    assert state.factor_labels == state.factor_labels_by_sources["label_genes_gene_sets"]
    for group, call in zip(dict.fromkeys([primary, *groups]), client.query.call_args_list):
        prompt = call.args[0]
        for source, prefix in zip(sources, "GSP"):
            for i in range(5):
                assert (f"{prefix}{i}" in prompt) == (source in group)
            assert f"{prefix}5" not in prompt
        assert "SECRET" not in prompt
        assert ("@1=G1" in prompt) == ("genes" in group and "gene-sets" in group)


def test_saved_factor_summary_and_missing_evidence(tmp_path):
    from tests.eaggl.test_label_mode_unittest import EagglLabelModeTest
    genes, sets, out = (tmp_path / name for name in ("genes", "sets", "factors"))
    genes.write_text("Gene\tFactor7\nGeneA\t1\n")
    sets.write_text("Gene_Set\tFactor7\nSetA\t1\n")
    result = EagglLabelModeTest()._run("label", "--label-gene-clusters-in", str(genes),
        "--label-gene-set-clusters-in", str(sets), "--label-sources", "genes",
        "--additional-label-sources", "gene-sets", "--additional-label-sources", "phenotypes,genes",
        "--factors-out", str(out))
    assert result.returncode == 0, result.stderr
    with out.open() as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert (row["Factor"], row["label"], row["label_sources"], row["label_gene_sets"],
            row["label_genes_phenos"]) == ("Factor7", "GeneA", "genes", "SetA", "NA")


def test_strict_handoff_writes_three_independent_labels(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    genes, sets, phenos = (tmp_path / name for name in ("genes", "sets", "phenos"))
    for path, field, item in ((genes, "Gene", "Gsecret"), (sets, "Gene_Set", "Ssecret"),
                              (phenos, "Pheno", "Psecret")):
        path.write_text(f"{field}\tFactor7\n{item}\t1\n")
    out = tmp_path / "labels"
    cmd = [sys.executable, str(root/"scripts/eaggl/label_final_factors.py"), "--mock", "--source", str(root),
           "--gene-loadings", str(genes), "--gene-set-loadings", str(sets), "--phenotype-loadings", str(phenos),
           "--additional-label-sources", "phenotypes", "--additional-label-sources", "genes,phenotypes",
           "--out", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with (out/"factor_labels.tsv").open() as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert [row[k] for k in ["label","label_phenos","label_genes_phenos"]] == [
        "Mock factor label 1", "Mock factor label 2", "Mock factor label 3"]
    prompts = (out/"label_prompt.txt").read_text().strip().split("\n\n")
    assert len(prompts) == 3
    assert "Ssecret" in prompts[0] and "Gsecret" in prompts[0] and "Psecret" not in prompts[0]
    assert "Psecret" in prompts[1] and "Ssecret" not in prompts[1] and "Gsecret" not in prompts[1]
    assert "Psecret" in prompts[2] and "Ssecret" not in prompts[2] and "Gsecret" in prompts[2]


def test_fresh_fit_uses_final_projected_phenotypes(tmp_path):
    from eaggl import app, cli, main_support
    genes, sets, x, phenos = (tmp_path / name for name in ("genes", "sets", "x", "phenos"))
    genes.write_text("Gene\tcombined\tlog_bf\tprior\nG1\t5\t3\t2\nG2\t4\t2\t2\nG3\t4\t2\t2\n")
    sets.write_text("Gene_Set\tbeta\tbeta_uncorrected\tp\nGS1\t.5\t.5\t.001\nGS2\t.4\t.4\t.001\n")
    x.write_text("GS1 G1 G2\nGS2 G2 G3\n")
    phenos.write_text("Gene\tPheno\tcombined\tlog_bf\tprior\nG1\tProjectedTrait\t5\t3\t2\n"
                      "G2\tProjectedTrait\t4\t2\t2\n")
    out = tmp_path / "factors"
    cmd = ["factor","--X-in",str(x),"--gene-stats-in",str(genes),"--gene-set-stats-in",str(sets),
           "--discovery-model","gene_by_gene","--seed","7","--max-num-factors","2","--phi",".01",
           "--gene-phewas-bfs-in",str(phenos),"--additional-label-sources","phenotypes",
           "--factor-top-loading-type","raw","--factors-out",str(out)]
    assert main_support._bootstrap_cli(cmd)
    state = app.run_main_pipeline(cli.options).state
    assert state.num_factors() > 0
    with out.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert any(r["label_phenos"] == "ProjectedTrait" for r in rows)
    assert all(r["label"] != "ProjectedTrait" for r in rows)
