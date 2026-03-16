from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl import phenotype_annotation as eaggl_phenotype_annotation  # noqa: E402
from eaggl import phewas as eaggl_phewas  # noqa: E402


class PhenotypeAnnotationTest(unittest.TestCase):
    def test_compositional_projection_separates_strength_from_capture_shape(self) -> None:
        basis = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        feature_by_pheno = np.array(
            [
                [8.0, 16.0],
                [2.0, 4.0],
            ]
        )
        capture, strengths = eaggl_phenotype_annotation.project_phenotype_capture(
            lambda W, X_new, max_sum=None: np.asarray(X_new, dtype=float),
            basis,
            feature_by_pheno,
            max_sum=1.0,
        )
        np.testing.assert_allclose(capture[0], capture[1])
        np.testing.assert_allclose(capture[0], np.array([0.8, 0.2]))
        np.testing.assert_allclose(strengths, np.array([10.0, 20.0]))

    def test_rank_top_capture_indices_uses_strength_as_tiebreak_only(self) -> None:
        capture = np.array(
            [
                [0.5, 0.1],
                [0.5, 0.3],
                [0.2, 0.3],
            ]
        )
        strengths = np.array([5.0, 10.0, 1.0])
        ranked = eaggl_phenotype_annotation.rank_top_capture_indices(capture, strengths, num_top=2)
        np.testing.assert_array_equal(ranked[:, 0], np.array([1, 0]))
        np.testing.assert_array_equal(ranked[:, 1], np.array([1, 2]))


class FactorPhewasSurfaceTest(unittest.TestCase):
    def _state(self):
        return SimpleNamespace(
            Y=np.array([0.1, 0.2]),
            combined_prior_Ys=np.array([0.3, 0.4]),
            X_orig=np.array([[1.0], [2.0]]),
            X_phewas_beta=np.array([[0.5], [0.6]]),
        )

    def _block_result(self):
        beta_tilde = np.array([[1.0]])
        se = np.array([[0.1]])
        z = np.array([[10.0]])
        p = np.array([[1e-3]])
        p_one = np.array([[5e-4]])
        return (None, None, beta_tilde, se, z, p, p_one)

    def test_default_factor_phewas_surface_runs_direct_only(self) -> None:
        state = self._state()
        input_values = np.array([[1.0], [2.0]])
        factor_keep_mask = np.array([True, True])
        gene_pheno_Y = np.array([[0.2], [0.8]])
        gene_pheno_combined = np.array([[0.3], [0.9]])
        options = SimpleNamespace(
            factor_phewas_full_output=False,
            debug_skip_huber=False,
            debug_skip_correlation=False,
        )
        calls = []
        with mock.patch.object(eaggl_phewas, "calculate_phewas_block", side_effect=[self._block_result()]) as calc:
            with mock.patch.object(
                eaggl_phewas,
                "accumulate_factor_phewas_outputs",
                side_effect=lambda _state, prefix, *_args, huber=False: calls.append((prefix, huber)),
            ):
                eaggl_phewas.run_factor_phewas_batch(
                    state,
                    input_values,
                    factor_keep_mask,
                    gene_pheno_Y,
                    gene_pheno_combined,
                    0,
                    1,
                    {},
                    options=options,
                )
        self.assertEqual(calc.call_count, 1)
        self.assertEqual(calls, [("Y", False)])

    def test_full_factor_phewas_surface_runs_direct_and_expert_outputs(self) -> None:
        state = self._state()
        input_values = np.array([[1.0], [2.0]])
        factor_keep_mask = np.array([True, True])
        gene_pheno_Y = np.array([[0.2], [0.8]])
        gene_pheno_combined = np.array([[0.3], [0.9]])
        options = SimpleNamespace(
            factor_phewas_full_output=True,
            debug_skip_huber=False,
            debug_skip_correlation=False,
        )
        calls = []
        block_results = [self._block_result() for _ in range(4)]
        with mock.patch.object(eaggl_phewas, "calculate_phewas_block", side_effect=block_results) as calc:
            with mock.patch.object(
                eaggl_phewas,
                "accumulate_factor_phewas_outputs",
                side_effect=lambda _state, prefix, *_args, huber=False: calls.append((prefix, huber)),
            ):
                eaggl_phewas.run_factor_phewas_batch(
                    state,
                    input_values,
                    factor_keep_mask,
                    gene_pheno_Y,
                    gene_pheno_combined,
                    0,
                    1,
                    {},
                    options=options,
                )
        self.assertEqual(calc.call_count, 4)
        self.assertEqual(
            calls,
            [
                ("Y", False),
                ("Y", True),
                ("combined_prior_Ys", False),
                ("combined_prior_Ys", True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
