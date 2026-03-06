from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl import labeling  # noqa: E402


class _RuntimeStub:
    def __init__(self) -> None:
        self.gene_sets = ["GS1", "GS2", "GS3"]
        self.genes = ["G1", "G2", "G3"]
        self.phenos = ["P1", "P2", "P3"]
        self.default_pheno = "DEFAULT"
        self.factor_labels = None
        self.factor_labels_gene_sets = None
        self.factor_labels_genes = None
        self.factor_labels_phenos = None

    def num_factors(self):
        return 2


class LabelingTest(unittest.TestCase):
    def test_query_lmm_rejects_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            labeling.query_lmm(
                "prompt",
                auth_key="x",
                lmm_provider="bogus",
                bail_fn=lambda message: (_ for _ in ()).throw(ValueError(message)),
                warn_fn=lambda _message: None,
            )

    def test_populate_factor_labels_sets_defaults_without_llm(self) -> None:
        runtime = _RuntimeStub()
        labeling.populate_factor_labels(
            runtime,
            factor_gene_set_x_pheno=False,
            top_gene_set_inds=np.array([[0, 1], [1, 2]]),
            top_anchor_gene_set_inds=np.array([[[0], [1]], [[1], [2]]]),
            top_gene_or_pheno_inds=np.array([[0, 1], [1, 2]]),
            top_anchor_gene_or_pheno_inds=np.array([[[0], [1]], [[1], [2]]]),
            top_pheno_or_gene_inds=np.array([[0, 1], [1, 2]]),
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
        self.assertEqual(runtime.factor_labels, ["GS1", "GS2"])
        self.assertEqual(runtime.factor_top_gene_sets[0], ["GS1", "GS2"])
        self.assertEqual(runtime.factor_top_genes[0], ["G1", "G2"])
        self.assertEqual(runtime.factor_top_phenos[0], ["P1", "P2"])


if __name__ == "__main__":
    unittest.main()
