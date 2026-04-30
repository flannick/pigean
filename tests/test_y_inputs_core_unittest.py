from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from pigean.y_inputs_core import align_huge_gene_level_state_to_active_genes


class YInputsCoreTest(unittest.TestCase):
    def test_align_huge_gene_level_state_drops_out_of_universe_rows(self) -> None:
        runtime = SimpleNamespace(
            genes=["G1", "G2", "G3"],
            gene_covariates=np.arange(20, dtype=float).reshape(5, 4),
            huge_signal_bfs=sparse.csc_matrix(np.arange(10, dtype=float).reshape(5, 2)),
            huge_signal_bfs_for_regression=sparse.csc_matrix(np.arange(15, dtype=float).reshape(5, 3)),
            gene_covariate_zs=np.ones((5, 4)),
            gene_covariates_mask=np.ones(5, dtype=bool),
            gene_covariates_mat_inv=np.eye(4),
            gene_covariate_adjustments=np.ones(5),
        )

        align_huge_gene_level_state_to_active_genes(runtime, bail_fn=self.fail)

        self.assertEqual(runtime.gene_covariates.shape, (3, 4))
        np.testing.assert_array_equal(runtime.gene_covariates[:, 0], np.array([0.0, 4.0, 8.0]))
        self.assertEqual(runtime.huge_signal_bfs.shape, (3, 2))
        self.assertEqual(runtime.huge_signal_bfs_for_regression.shape, (3, 3))
        self.assertIsNone(runtime.gene_covariate_zs)
        self.assertIsNone(runtime.gene_covariates_mask)
        self.assertIsNone(runtime.gene_covariates_mat_inv)
        self.assertIsNone(runtime.gene_covariate_adjustments)

    def test_align_huge_gene_level_state_rejects_short_covariates(self) -> None:
        runtime = SimpleNamespace(
            genes=["G1", "G2", "G3"],
            gene_covariates=np.ones((2, 4)),
            huge_signal_bfs=None,
            huge_signal_bfs_for_regression=None,
            gene_covariate_zs=None,
            gene_covariates_mask=None,
            gene_covariates_mat_inv=None,
            gene_covariate_adjustments=None,
        )

        with self.assertRaises(AssertionError):
            align_huge_gene_level_state_to_active_genes(runtime, bail_fn=self.fail)


if __name__ == "__main__":
    unittest.main()
