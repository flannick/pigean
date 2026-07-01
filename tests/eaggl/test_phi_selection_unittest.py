from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl.phi_selection import (  # noqa: E402
    CompositePhiSelectionConfig,
    DEFAULT_COMPONENT_WEIGHTS,
    PhiSelectionInputs,
    parse_composite_weights,
    score_phi_candidate,
    select_composite_candidate,
)


class PhiSelectionScoringTest(unittest.TestCase):
    def _config(self, **overrides):
        kwargs = {"weights": dict(DEFAULT_COMPONENT_WEIGHTS)}
        kwargs.update(overrides)
        return CompositePhiSelectionConfig(**kwargs)

    def test_parse_weights_accepts_aliases_and_rejects_bad_values(self):
        weights = parse_composite_weights("factor_size_score=0.2,coverage=0.5,bridge_qc=0")
        self.assertEqual(weights["factor_size"], 0.2)
        self.assertEqual(weights["coverage"], 0.5)
        self.assertEqual(weights["annotation_bridge_qc"], 0.0)
        with self.assertRaises(ValueError):
            parse_composite_weights("bogus=1")
        with self.assertRaises(ValueError):
            parse_composite_weights("coverage=-1")
        with self.assertRaises(ValueError):
            parse_composite_weights(",".join(f"{key}=0" for key in DEFAULT_COMPONENT_WEIGHTS))

    def test_factor_size_score_peaks_at_target_and_is_symmetric(self):
        cfg = self._config(target_factor_gene_mass=10.0, size_log2_width=1.0, loading_cap=100.0)
        inputs = PhiSelectionInputs(
            discovery_model="gene_by_gene",
            gene_loadings=np.array([[5.0, 2.5, 10.0], [5.0, 2.5, 10.0]]),
        )
        wide, _long, _per_factor = score_phi_candidate(0.01, 3, inputs, cfg)
        scores = [row["factor_size_score"] for row in _per_factor]
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], scores[2])
        self.assertLess(scores[1], scores[0])
        self.assertTrue(0.0 <= wide["factor_size_score"] <= 1.0)

    def test_nonoverlap_disjoint_high_identical_low(self):
        cfg = self._config()
        disjoint = PhiSelectionInputs(
            discovery_model="gene_by_gene",
            gene_loadings=np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float),
        )
        identical = PhiSelectionInputs(
            discovery_model="gene_by_gene",
            gene_loadings=np.array([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=float),
        )
        wide_disjoint, _, _ = score_phi_candidate(0.01, 2, disjoint, cfg)
        wide_identical, _, _ = score_phi_candidate(0.02, 2, identical, cfg)
        self.assertGreater(wide_disjoint["nonoverlap_score"], 0.99)
        self.assertLess(wide_identical["nonoverlap_score"], 0.1)

    def test_entity_concentration_and_coverage(self):
        cfg = self._config(coverage_min_loading=0.5)
        concentrated = PhiSelectionInputs(
            discovery_model="gene_by_gene",
            gene_loadings=np.array([[1.0, 0.0], [0.8, 0.0], [0.0, 0.1]], dtype=float),
            gene_importance=np.array([10.0, 5.0, 1.0]),
        )
        split = PhiSelectionInputs(
            discovery_model="gene_by_gene",
            gene_loadings=np.array([[0.2, 0.2], [0.1, 0.1], [0.05, 0.05]], dtype=float),
            gene_importance=np.array([10.0, 5.0, 1.0]),
        )
        wide_conc, _, _ = score_phi_candidate(0.01, 2, concentrated, cfg)
        wide_split, _, _ = score_phi_candidate(0.02, 2, split, cfg)
        self.assertGreater(wide_conc["entity_concentration_score"], wide_split["entity_concentration_score"])
        self.assertGreater(wide_conc["coverage_score"], wide_split["coverage_score"])

    def test_gene_by_gene_reconstruction_exact_scores_high(self):
        W = np.array([[1.0], [0.5], [0.0]], dtype=float)
        M = W @ W.T
        cfg = self._config()
        inputs = PhiSelectionInputs(discovery_model="gene_by_gene", gene_loadings=W, target_matrix=M)
        wide, long_rows, per_factor = score_phi_candidate(0.01, 1, inputs, cfg)
        self.assertGreater(wide["reconstruction_score"], 0.99)
        self.assertTrue(0.0 <= wide["coherence_score"] <= 1.0)
        self.assertIn("factor_coherence_score", per_factor[0])
        for row in long_rows:
            if row["available"]:
                self.assertTrue(0.0 <= row["score"] <= 1.0)

    def test_missing_annotation_metrics_are_unavailable_not_zero(self):
        cfg = self._config()
        inputs = PhiSelectionInputs(discovery_model="gene_by_gene", gene_loadings=np.eye(2))
        wide, long_rows, _ = score_phi_candidate(0.01, 2, inputs, cfg)
        bridge = [row for row in long_rows if row["component"] == "annotation_bridge_qc"][0]
        self.assertFalse(bridge["available"])
        self.assertIsNone(wide["annotation_bridge_qc_score"])
        active_weight = sum(row["normalized_weight"] for row in long_rows if row["available"])
        self.assertAlmostEqual(active_weight, 1.0)

    def test_select_composite_candidate_tie_prefers_fewer_factors_then_coverage_then_lower_phi(self):
        candidates = [
            {"phi": 0.02, "phi_composite_score": 0.9, "modal_factor_count": 5, "coverage_score": 0.5},
            {"phi": 0.01, "phi_composite_score": 0.895, "modal_factor_count": 4, "coverage_score": 0.4},
            {"phi": 0.005, "phi_composite_score": 0.7, "modal_factor_count": 2, "coverage_score": 1.0},
        ]
        selected, reason = select_composite_candidate(candidates, tie_tolerance=0.01)
        self.assertEqual(reason, "composite_score")
        self.assertEqual(selected["phi"], 0.01)
        self.assertTrue(selected["selected"])
        self.assertEqual(selected["selection_frontier_size"], 2)


class PhiSelectionReducedTargetRegressionTest(unittest.TestCase):
    def test_reduced_annotation_target_indices_do_not_index_candidate_matrix(self):
        # The factor runtime stores target matrices in the reduced candidate
        # space, but target_annotation_indices may still be full-X indices.
        # Composite scoring must not apply a full index such as 7 to a two-row
        # reduced annotation loading matrix.
        inputs = PhiSelectionInputs(
            discovery_model="gene_by_annotation",
            gene_loadings=np.array(
                [
                    [0.8, 0.1],
                    [0.2, 0.7],
                ],
                dtype=float,
            ),
            annotation_loadings=np.array(
                [
                    [0.9, 0.0],
                    [0.0, 0.8],
                ],
                dtype=float,
            ),
            target_matrix=np.array(
                [
                    [1.0, 0.1],
                    [0.2, 1.0],
                ],
                dtype=float,
            ),
            target_weight_matrix=np.ones((2, 2), dtype=float),
            target_gene_indices=np.array([3, 9], dtype=int),
            target_annotation_indices=np.array([7, 11], dtype=int),
            gene_importance=np.array([1.0, 0.5], dtype=float),
            annotation_importance=np.array([1.0, 0.5], dtype=float),
        )
        config = CompositePhiSelectionConfig(weights=dict(DEFAULT_COMPONENT_WEIGHTS))

        wide, long_rows, per_factor_rows = score_phi_candidate(0.05, 2, inputs, config)

        self.assertIn("phi_composite_score", wide)
        self.assertTrue(0.0 <= float(wide["phi_composite_score"]) <= 1.0)
        self.assertTrue(long_rows)
        self.assertEqual(len(per_factor_rows), 2)

    def test_zero_factor_candidate_scores_without_crashing(self):
        inputs = PhiSelectionInputs(
            discovery_model="gene_by_annotation",
            gene_loadings=np.zeros((2, 0), dtype=float),
            annotation_loadings=np.zeros((1, 0), dtype=float),
            target_matrix=np.zeros((1, 2), dtype=float),
            target_annotation_indices=np.array([7], dtype=int),
            target_gene_indices=np.array([3, 9], dtype=int),
        )
        config = CompositePhiSelectionConfig(weights=dict(DEFAULT_COMPONENT_WEIGHTS))

        wide, long_rows, per_factor_rows = score_phi_candidate(0.05, 0, inputs, config)

        self.assertEqual(wide["num_factors"], 0)
        self.assertEqual(float(wide["phi_composite_score"]), 0.0)
        self.assertEqual(per_factor_rows, [])
        unavailable = {row["component"]: row["available"] for row in long_rows}
        self.assertFalse(unavailable["factor_size"])
        self.assertFalse(unavailable["reconstruction"])


if __name__ == "__main__":
    unittest.main()
