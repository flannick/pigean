from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl import factor as eaggl_factor  # noqa: E402
from eaggl import factor_runtime  # noqa: E402
from eaggl import labeling  # noqa: E402


class _RuntimeStub:
    def __init__(self) -> None:
        self.gene_sets = ["GS1", "GS2", "GS3"]
        self.genes = ["G1", "G2"]
        self.phenos = ["P1", "P2"]
        self.default_pheno = "DEFAULT"

    def num_factors(self):
        return 2


class FactorLabelingFilterTest(unittest.TestCase):
    def test_gene_sets_for_labeling_first_column_and_repeated_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gmt = Path(tmp) / "a.gmt"
            tsv = Path(tmp) / "b.tsv"
            gmt.write_text("GS_A\tdesc\tG1\nGS_B\tdesc\tG2\n")
            tsv.write_text("GS_C\tother\n")
            ids = eaggl_factor._read_gene_sets_for_labeling([str(gmt), str(tsv)], id_col=None, bail_fn=None)
        self.assertEqual(ids, {"GS_A", "GS_B", "GS_C"})

    def test_gene_sets_for_labeling_header_column_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "gene_sets.tsv"
            table.write_text("name\tidentifier\nignored\tGS_X\nignored2\tGS_Y\n")
            ids = eaggl_factor._read_gene_sets_for_labeling([str(table)], id_col="identifier", bail_fn=None)
        self.assertEqual(ids, {"GS_X", "GS_Y"})

    def test_label_ranking_filters_before_top_selection(self) -> None:
        scores = np.array(
            [
                [0.90, 0.10],
                [0.80, 0.70],
                [0.10, 0.60],
            ]
        )
        allowed = np.array([False, False, True])
        warnings = []
        top = factor_runtime._top_indices_by_factor(scores, 2, allowed_mask=allowed, warn_fn=warnings.append)
        self.assertEqual(top[:, 0].tolist(), [2, -1])
        self.assertEqual(top[:, 1].tolist(), [2, -1])
        self.assertEqual(warnings, [])

    def test_label_ranking_falls_back_when_filter_has_no_positive_factor_overlap(self) -> None:
        scores = np.array(
            [
                [0.90],
                [0.80],
                [0.00],
            ]
        )
        allowed = np.array([False, False, True])
        warnings = []
        top = factor_runtime._top_indices_by_factor(scores, 2, allowed_mask=allowed, warn_fn=warnings.append)
        self.assertEqual(top[:, 0].tolist(), [0, 1])
        self.assertEqual(len(warnings), 1)
        self.assertIn("no positive-loading gene sets", warnings[0])

    def test_populate_factor_labels_skips_padding_indices(self) -> None:
        runtime = _RuntimeStub()
        labeling.populate_factor_labels(
            runtime,
            factor_gene_set_x_pheno=False,
            top_gene_set_inds=np.array([[2, 1], [-1, -1]]),
            top_anchor_gene_set_inds=np.array([[[2], [1]], [[-1], [-1]]]),
            top_gene_or_pheno_inds=np.array([[0, 1], [1, 0]]),
            top_anchor_gene_or_pheno_inds=np.array([[[0], [1]], [[1], [0]]]),
            top_pheno_or_gene_inds=np.array([[0, 1], [1, 0]]),
            lmm_auth_key=None,
            lmm_model=None,
            lmm_provider="openai",
            label_gene_sets_only=False,
            label_include_phenos=False,
            label_individually=False,
            log_fn=lambda _message, *_args: None,
            bail_fn=lambda message: (_ for _ in ()).throw(ValueError(message)),
            warn_fn=lambda _message: None,
        )
        self.assertEqual(runtime.factor_labels, ["GS3", "GS2"])
        self.assertEqual(runtime.factor_top_gene_sets, [["GS3"], ["GS2"]])
        self.assertEqual(runtime.factor_anchor_top_gene_sets, [[["GS3"]], [["GS2"]]])


if __name__ == "__main__":
    unittest.main()
