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


class AnnotationDiagnosticsTest(unittest.TestCase):
    def test_bridge_metrics_match_rank_one_kernel_decomposition(self) -> None:
        state = _tiny_state()
        records = annotation_diagnostics.compute_annotation_bridge_records(
            state,
            min_active_genes=1,
            review_factor_neff=1.5,
            review_bridge_fraction=0.4,
            exclude_source_top_frac=1.0,
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
            review_factor_neff=1.5,
            review_bridge_fraction=0.4,
            exclude_source_top_frac=1.0,
        )
        neg = next(record for record in records if record["annotation_id"] == "SET_NEG")
        self.assertLess(neg["beta"], 0)
        self.assertFalse(neg["flag_review"])
        self.assertFalse(neg["flag_suggest_exclude"])

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
