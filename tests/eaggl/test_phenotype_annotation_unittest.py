from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl import phenotype_annotation as eaggl_phenotype_annotation  # noqa: E402
from eaggl import phewas as eaggl_phewas  # noqa: E402
from eaggl import factor_runtime as eaggl_factor_runtime  # noqa: E402
from pegs_shared import output_tables as pegs_output_tables  # noqa: E402


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

    def test_align_projection_inputs_keeps_pre_filtered_basis(self) -> None:
        basis = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        feature_by_pheno = np.array(
            [
                [10.0, 1.0],
                [20.0, 2.0],
                [30.0, 3.0],
                [40.0, 4.0],
            ]
        )
        mask = np.array([True, False, True, False])
        aligned_basis, aligned_feature = eaggl_factor_runtime._align_projection_inputs_to_mask(
            basis,
            feature_by_pheno,
            mask,
        )
        np.testing.assert_array_equal(aligned_basis, basis)
        np.testing.assert_array_equal(aligned_feature, feature_by_pheno[mask, :])

    def test_align_projection_inputs_subsets_full_basis_when_needed(self) -> None:
        basis = np.array(
            [
                [1.0, 0.0],
                [5.0, 5.0],
                [0.0, 1.0],
                [6.0, 6.0],
            ]
        )
        feature_by_pheno = np.array(
            [
                [10.0, 1.0],
                [20.0, 2.0],
                [30.0, 3.0],
                [40.0, 4.0],
            ]
        )
        mask = np.array([True, False, True, False])
        aligned_basis, aligned_feature = eaggl_factor_runtime._align_projection_inputs_to_mask(
            basis,
            feature_by_pheno,
            mask,
        )
        np.testing.assert_array_equal(aligned_basis, basis[mask, :])
        np.testing.assert_array_equal(aligned_feature, feature_by_pheno[mask, :])

    def test_prepare_thresholded_profile_input_supports_weighted_and_binary_modes(self) -> None:
        feature_by_pheno = np.array(
            [
                [2.5, 0.0],
                [0.0, 1.1],
                [3.0, 0.0],
            ]
        )
        weighted = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
            feature_by_pheno,
            "weighted_thresholded",
        )
        binary = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
            feature_by_pheno,
            "binary_thresholded",
        )
        np.testing.assert_array_equal(weighted, feature_by_pheno)
        np.testing.assert_array_equal(binary, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))


class FactorPhewasSurfaceTest(unittest.TestCase):
    def _state(self):
        return SimpleNamespace(
            Y=np.array([0.1, 0.2, 0.3]),
            combined_prior_Ys=np.array([0.3, 0.4, 0.5]),
            X_orig=np.array([[1.0], [2.0], [3.0]]),
            X_phewas_beta=np.array([[0.5], [0.6]]),
            exp_gene_factors=np.array([[0.2, 0.8], [0.7, 0.1], [0.3, 0.4]]),
            phenos=["P1", "P2"],
            factor_phewas_result_blocks=None,
        )

    def _block_result(self):
        beta_tilde = np.array([[1.0]])
        se = np.array([[0.1]])
        z = np.array([[10.0]])
        p = np.array([[1e-3]])
        p_one = np.array([[5e-4]])
        return (None, None, beta_tilde, se, z, p, p_one)

    def test_default_factor_phewas_surface_records_binary_anchor_adjusted_results(self) -> None:
        state = self._state()
        gene_pheno_Y = np.array([[0.0], [0.0], [0.0]])
        gene_pheno_combined = np.array([[0.0], [1.5], [2.0]])
        options = SimpleNamespace(
            factor_phewas_mode="marginal_anchor_adjusted_binary",
            factor_phewas_modes=None,
            factor_phewas_anchor_covariate="direct",
            factor_phewas_thresholded_combined_cutoff=1.0,
            factor_phewas_se="robust",
            factor_phewas_min_gene_factor_weight=0.0,
            factor_phewas_full_output=False,
            debug_skip_huber=False,
            debug_skip_correlation=False,
        )
        eaggl_phewas.run_factor_phewas_batch(
            state,
            gene_pheno_Y,
            gene_pheno_combined,
            0,
            1,
            {"bail_fn": lambda msg: (_ for _ in ()).throw(AssertionError(msg))},
            options=options,
        )
        self.assertEqual(len(state.factor_phewas_result_blocks), 1)
        block = state.factor_phewas_result_blocks[0]
        self.assertEqual(block["mode"], "marginal_anchor_adjusted_binary")
        self.assertEqual(block["anchor_covariate"], "direct")
        self.assertEqual(block["factor_model_scope"], "marginal_one_factor")
        self.assertEqual(block["outcome_surface"], "binary_thresholded")
        self.assertEqual(block["coefficients"].shape, (2, 1))

    def test_joint_binary_factor_phewas_records_all_factors_together(self) -> None:
        state = self._state()
        gene_pheno_Y = np.array([[0.0], [0.0], [0.0]])
        gene_pheno_combined = np.array([[0.0], [1.5], [2.0]])
        options = SimpleNamespace(
            factor_phewas_mode="joint_anchor_adjusted_binary",
            factor_phewas_modes=None,
            factor_phewas_anchor_covariate="direct",
            factor_phewas_thresholded_combined_cutoff=1.0,
            factor_phewas_se="robust",
            factor_phewas_min_gene_factor_weight=0.0,
            factor_phewas_full_output=True,
            debug_skip_huber=False,
            debug_skip_correlation=False,
        )
        eaggl_phewas.run_factor_phewas_batch(
            state,
            gene_pheno_Y,
            gene_pheno_combined,
            0,
            1,
            {"bail_fn": lambda msg: (_ for _ in ()).throw(AssertionError(msg))},
            options=options,
        )
        block = state.factor_phewas_result_blocks[0]
        self.assertEqual(block["mode"], "joint_anchor_adjusted_binary")
        self.assertEqual(block["factor_model_scope"], "joint_all_factors")
        self.assertEqual(block["coefficients"].shape, (2, 1))

    def test_legacy_factor_phewas_mode_uses_existing_continuous_path(self) -> None:
        state = self._state()
        state.factor_phewas_result_blocks = []
        gene_pheno_Y = np.array([[0.2, 0.4], [0.8, 0.6], [0.1, 0.9]])
        gene_pheno_combined = np.array([[0.3, 0.2], [0.9, 0.7], [0.4, 0.8]])
        options = SimpleNamespace(
            factor_phewas_mode="legacy_continuous_direct",
            factor_phewas_modes=None,
            factor_phewas_anchor_covariate="direct",
            factor_phewas_thresholded_combined_cutoff=1.0,
            factor_phewas_se="robust",
            factor_phewas_min_gene_factor_weight=0.0,
            factor_phewas_full_output=False,
            debug_skip_huber=False,
            debug_skip_correlation=False,
        )
        legacy_block_result = (
            None,
            None,
            np.array([[1.0], [2.0]]),
            np.array([[0.1], [0.2]]),
            np.array([[10.0], [20.0]]),
            np.array([[1e-3], [2e-3]]),
            np.array([[5e-4], [1e-3]]),
        )
        with mock.patch.object(eaggl_phewas, "calculate_phewas_block", side_effect=[legacy_block_result]) as calc:
            eaggl_phewas.run_factor_phewas_batch(
                state,
                gene_pheno_Y,
                gene_pheno_combined,
                0,
                1,
                {},
                options=options,
            )
        self.assertEqual(calc.call_count, 1)
        self.assertEqual(len(state.factor_phewas_result_blocks), 1)
        block = state.factor_phewas_result_blocks[0]
        self.assertEqual(block["analysis"], "legacy_continuous_direct")
        self.assertEqual(block["outcome_surface"], "continuous_direct")
        self.assertEqual(block["coefficients"].shape, (2, 1))

    def test_multiple_factor_phewas_modes_append_multiple_result_blocks(self) -> None:
        state = self._state()
        gene_pheno_Y = np.array([[0.2], [0.8], [0.1]])
        gene_pheno_combined = np.array([[0.3], [1.7], [1.4]])
        options = SimpleNamespace(
            factor_phewas_mode="marginal_anchor_adjusted_binary",
            factor_phewas_modes=["marginal_anchor_adjusted_binary", "joint_anchor_adjusted_binary"],
            factor_phewas_anchor_covariate="direct",
            factor_phewas_thresholded_combined_cutoff=1.0,
            factor_phewas_se="robust",
            factor_phewas_min_gene_factor_weight=0.0,
            factor_phewas_full_output=False,
            debug_skip_huber=False,
            debug_skip_correlation=False,
        )

        eaggl_phewas.run_factor_phewas_batch(
            state,
            gene_pheno_Y,
            gene_pheno_combined,
            0,
            1,
            {"bail_fn": lambda msg: (_ for _ in ()).throw(AssertionError(msg))},
            options=options,
        )

        self.assertEqual(
            [block["mode"] for block in state.factor_phewas_result_blocks],
            ["marginal_anchor_adjusted_binary", "joint_anchor_adjusted_binary"],
        )

    def test_factor_phewas_writer_includes_explicit_model_identity_columns(self) -> None:
        runtime = SimpleNamespace(
            phenos=["P1"],
            factor_labels=["FactorLabel1"],
            factor_phewas_result_blocks=[
                {
                    "phenos": ["P1"],
                    "analysis": "marginal_anchor_adjusted_binary",
                    "mode": "marginal_anchor_adjusted_binary",
                    "model_name": "marginal_anchor_adjusted_binary",
                    "factor_model_scope": "marginal_one_factor",
                    "outcome_surface": "binary_thresholded",
                    "anchor_covariate": "direct",
                    "threshold_cutoff": 1.0,
                    "se_type": "robust",
                    "coefficients": np.array([[1.2]]),
                    "ses": np.array([[0.3]]),
                    "z_scores": np.array([[4.0]]),
                    "p_values": np.array([[1e-4]]),
                    "one_sided_p_values": np.array([[5e-5]]),
                },
                {
                    "phenos": ["P1"],
                    "analysis": "joint_anchor_adjusted_binary",
                    "mode": "joint_anchor_adjusted_binary",
                    "model_name": "joint_anchor_adjusted_binary",
                    "factor_model_scope": "joint_all_factors",
                    "outcome_surface": "binary_thresholded",
                    "anchor_covariate": "direct",
                    "threshold_cutoff": 1.0,
                    "se_type": "robust",
                    "coefficients": np.array([[0.8]]),
                    "ses": np.array([[0.2]]),
                    "z_scores": np.array([[4.0]]),
                    "p_values": np.array([[2e-4]]),
                    "one_sided_p_values": np.array([[1e-4]]),
                },
            ],
        )
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factor_phewas_stats.out"
            pegs_output_tables.write_factor_phewas_statistics(runtime, str(output_path))
            lines = output_path.read_text().strip().splitlines()

        self.assertIn("model_name", lines[0])
        self.assertIn("factor_model_scope", lines[0])
        self.assertIn("outcome_surface", lines[0])
        self.assertEqual(len(lines), 3)
        self.assertIn("marginal_anchor_adjusted_binary", lines[1])
        self.assertIn("joint_anchor_adjusted_binary", lines[2])


if __name__ == "__main__":
    unittest.main()
