from __future__ import annotations

import gzip
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl import annotation_diagnostics  # noqa: E402


def _tiny_state():
    return SimpleNamespace(
        discovery_model="gene_by_gene",
        params={"discovery_model": "gene_by_gene"},
        genes=["G1", "G2", "G3"],
        gene_sets=["SET_BRIDGE", "SET_LOCAL", "SET_NEG"],
        gene_set_labels=np.array(["curated", "curated", "text"]),
        X_orig=sparse.csr_matrix(
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
        ),
        exp_gene_factors=np.array(
            [
                [0.5, 0.5],
                [0.5, 0.0],
                [0.0, 0.5],
            ],
            dtype=float,
        ),
        exp_lambdak=np.array([1.0, 1.0], dtype=float),
        factor_labels=["F1 label", "F2 label"],
        gene_in_discovery_mask=np.array([True, True, True]),
        gene_set_in_discovery_mask=np.array([True, True, True]),
        betas=np.array([2.0, 1.0, -1.0], dtype=float),
        betas_uncorrected=np.array([3.0, 1.5, -2.0], dtype=float),
        scale_factors=np.array([1.0, 1.0, 1.0], dtype=float),
        p_values=np.array([0.01, 0.2, 0.5], dtype=float),
        anchor_pheno_mask=None,
        X_phewas_beta=None,
        X_orig_missing_genes=None,
    )


def _source_adaptive_state():
    genes = [f"G{i}" for i in range(10)]
    factor_count = 5
    W = np.zeros((10, factor_count), dtype=float)
    for factor in range(factor_count):
        W[2 * factor, factor] = 0.5
        W[2 * factor + 1, factor] = 0.5

    columns = []
    gene_sets = []
    labels = []
    betas = []

    def add_annotation(name, source, gene_indices, beta):
        column = np.zeros(10, dtype=float)
        column[list(gene_indices)] = 1.0
        columns.append(column)
        gene_sets.append(name)
        labels.append(source)
        betas.append(beta)

    bridge_genes = [0, 2, 4, 6, 8]
    for index in range(20):
        add_annotation(f"BROAD_BRIDGE_{index}", "broad", bridge_genes, 10.0 + index)
    for index in range(19):
        factor = index % factor_count
        add_annotation(f"SPECIFIC_LOCAL_{index}", "specific", [2 * factor, 2 * factor + 1], 1.0 + index * 0.01)
    add_annotation("SPECIFIC_BRIDGE", "specific", bridge_genes, 9.5)
    add_annotation("SMALL_BRIDGE", "small", bridge_genes, 9.8)
    add_annotation("SMALL_LOCAL", "small", [0, 1], 1.0)
    add_annotation("NEGATIVE_BRIDGE", "broad", bridge_genes, -50.0)

    X = np.column_stack(columns)
    return SimpleNamespace(
        discovery_model="gene_by_gene",
        params={"discovery_model": "gene_by_gene"},
        genes=genes,
        gene_sets=gene_sets,
        gene_set_labels=np.array(labels),
        X_orig=sparse.csr_matrix(X),
        exp_gene_factors=W,
        exp_lambdak=np.ones(factor_count, dtype=float),
        factor_labels=[f"F{i + 1}" for i in range(factor_count)],
        gene_in_discovery_mask=np.ones(len(genes), dtype=bool),
        gene_set_in_discovery_mask=np.ones(len(gene_sets), dtype=bool),
        betas=np.array(betas, dtype=float),
        betas_uncorrected=np.array(betas, dtype=float),
        scale_factors=np.ones(len(gene_sets), dtype=float),
        p_values=np.ones(len(gene_sets), dtype=float),
        anchor_pheno_mask=None,
        X_phewas_beta=None,
        X_orig_missing_genes=None,
    )


class AnnotationDiagnosticsTest(unittest.TestCase):
    def test_bridge_metrics_match_rank_one_kernel_decomposition(self) -> None:
        state = _tiny_state()
        records = annotation_diagnostics.compute_annotation_bridge_records(
            state,
            min_active_genes=1,
            base_factor_neff=1.5,
            base_bridge_fraction=0.4,
            base_max_similarity=1.0,
            source_min_annotations=2,
        )
        bridge = next(record for record in records if record["annotation_id"] == "SET_BRIDGE")

        self.assertEqual(bridge["annotation_source"], "curated")
        self.assertEqual(bridge["anchor_trait"], "default")
        self.assertEqual(bridge["n_genes_active"], 2)
        self.assertAlmostEqual(bridge["top_factor_1_overlap"], 1.0)
        self.assertAlmostEqual(bridge["top_factor_2_overlap"], 0.5)
        self.assertAlmostEqual(bridge["within_kernel_mass"], 2.0 * (1.0**2 + 0.5**2))
        self.assertAlmostEqual(bridge["between_kernel_mass"], 2.0 * ((1.0 + 0.5) ** 2) - bridge["within_kernel_mass"])
        self.assertAlmostEqual(
            bridge["bridge_fraction"],
            bridge["between_kernel_mass"] / (bridge["between_kernel_mass"] + bridge["within_kernel_mass"]),
        )
        self.assertEqual(bridge["source_rank_separated_bridge_mass"], 1)
        self.assertTrue(bridge["flag_review"])

    def test_negative_beta_never_suggests_exclusion(self) -> None:
        state = _tiny_state()
        records = annotation_diagnostics.compute_annotation_bridge_records(
            state,
            min_active_genes=1,
            base_factor_neff=1.5,
            base_bridge_fraction=0.4,
            base_max_similarity=1.0,
            source_min_annotations=2,
        )
        neg = next(record for record in records if record["annotation_id"] == "SET_NEG")
        self.assertLess(neg["beta"], 0)
        self.assertFalse(neg["flag_review"])
        self.assertFalse(neg["flag_suggest_exclude"])

    def test_source_adaptive_policy_is_stricter_for_specific_sources(self) -> None:
        records = annotation_diagnostics.compute_annotation_bridge_records(
            _source_adaptive_state(),
            min_active_genes=5,
        )
        broad = next(record for record in records if record["annotation_id"] == "BROAD_BRIDGE_19")
        specific = next(record for record in records if record["annotation_id"] == "SPECIFIC_BRIDGE")

        self.assertEqual(broad["source_n_annotations"], 21)
        self.assertGreater(broad["source_bridge_burden"], specific["source_bridge_burden"])
        self.assertLess(
            broad["source_required_global_bridge_percentile"],
            specific["source_required_global_bridge_percentile"],
        )
        self.assertTrue(broad["flag_review"])
        self.assertTrue(broad["flag_suggest_exclude"])
        self.assertTrue(specific["flag_review"])
        self.assertFalse(specific["flag_suggest_exclude"])
        self.assertIn("source_adaptive_review", specific["flag_reason"])

    def test_suggest_exclude_requires_top_global_bridge_rank(self) -> None:
        records = annotation_diagnostics.compute_annotation_bridge_records(
            _source_adaptive_state(),
            min_active_genes=5,
            suggest_exclude_global_rank_max=1,
        )
        broad = next(record for record in records if record["annotation_id"] == "BROAD_BRIDGE_19")

        self.assertTrue(broad["flag_review"])
        self.assertGreater(broad["global_rank_separated_bridge_mass"], 1)
        self.assertFalse(broad["flag_suggest_exclude"])

    def test_small_sources_require_extreme_global_bridge_percentile(self) -> None:
        records = annotation_diagnostics.compute_annotation_bridge_records(
            _source_adaptive_state(),
            min_active_genes=5,
        )
        small = next(record for record in records if record["annotation_id"] == "SMALL_BRIDGE")

        self.assertEqual(small["source_n_annotations"], 2)
        self.assertEqual(small["source_separated_bridge_percentile"], small["global_separated_bridge_percentile"])
        self.assertAlmostEqual(small["source_required_global_bridge_percentile"], 0.99)
        self.assertFalse(small["flag_suggest_exclude"])

    def test_source_percentiles_are_computed_per_anchor_trait(self) -> None:
        state = _source_adaptive_state()
        state.betas = np.column_stack([state.betas, state.betas * 0.01])
        state.betas_uncorrected = np.column_stack([state.betas_uncorrected, state.betas_uncorrected * 0.01])
        records = annotation_diagnostics.compute_annotation_bridge_records(state, min_active_genes=5)
        anchor_1 = next(
            record
            for record in records
            if record["annotation_id"] == "BROAD_BRIDGE_19" and record["anchor_trait"] == "anchor_1"
        )
        anchor_2 = next(
            record
            for record in records
            if record["annotation_id"] == "BROAD_BRIDGE_19" and record["anchor_trait"] == "anchor_2"
        )

        self.assertEqual(anchor_1["source_n_annotations"], anchor_2["source_n_annotations"])
        self.assertEqual(anchor_1["source_separated_bridge_percentile"], anchor_2["source_separated_bridge_percentile"])

    def test_gene_factor_annotation_contribs_are_top_n_ranked(self) -> None:
        state = _tiny_state()
        records = annotation_diagnostics.compute_gene_factor_annotation_contrib_records(state, top_n=1)
        g1_f1 = [
            record
            for record in records
            if record["gene"] == "G1" and record["factor"] == "Factor1"
        ]
        self.assertEqual(len(g1_f1), 1)
        self.assertEqual(g1_f1[0]["rank_within_gene_factor"], 1)
        self.assertEqual(g1_f1[0]["annotation_id"], "SET_BRIDGE")
        self.assertEqual(g1_f1[0]["factor_label"], "F1 label")

    def test_writers_emit_gz_tables_and_exclude_list(self) -> None:
        state = _tiny_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "annotation_bridge_metrics.tsv.gz"
            contribs_path = Path(tmpdir) / "gene_factor_annotation_contribs.tsv.gz"
            exclude_path = Path(tmpdir) / "annotation_bridge_suggested_exclude.txt"

            annotation_diagnostics.write_annotation_bridge_metrics(state, metrics_path)
            annotation_diagnostics.write_gene_factor_annotation_contribs(state, contribs_path, top_n=1)
            annotation_diagnostics.write_annotation_bridge_suggested_exclude(state, exclude_path)

            with gzip.open(metrics_path, "rt", encoding="utf-8") as fh:
                header = fh.readline().strip().split("\t")
                self.assertIn("annotation_id", header)
                self.assertIn("separated_bridge_mass", header)
                self.assertIn("source_quality_score", header)
                self.assertIn("source_required_global_bridge_percentile", header)
                self.assertIn("global_separated_bridge_percentile", header)
            with gzip.open(contribs_path, "rt", encoding="utf-8") as fh:
                header = fh.readline().strip().split("\t")
                self.assertIn("contribution_L_scale", header)
            self.assertTrue(exclude_path.exists())

    def test_gene_by_annotation_rejected(self) -> None:
        state = _tiny_state()
        state.discovery_model = "gene_by_annotation"
        with self.assertRaisesRegex(ValueError, "gene_by_gene"):
            annotation_diagnostics.compute_annotation_bridge_records(state)


class AnnotationDiagnosticsCliTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        src_root = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return subprocess.run(
            [sys.executable, "-m", "eaggl", *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_rejects_annotation_bridge_outputs_for_gene_by_annotation(self) -> None:
        proc = self._run(
            "factor",
            "--discovery-model",
            "gene_by_annotation",
            "--annotation-bridge-metrics-out",
            "bridge.tsv.gz",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("require --discovery-model gene_by_gene", proc.stderr)

    def test_help_expert_lists_annotation_bridge_outputs(self) -> None:
        proc = self._run("factor", "--help-expert")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--annotation-bridge-metrics-out", proc.stdout)
        self.assertIn("--gene-factor-annotation-contribs-out", proc.stdout)


if __name__ == "__main__":
    unittest.main()
