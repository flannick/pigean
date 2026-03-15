from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl import factor_runtime as eaggl_factor_runtime  # noqa: E402


class _TinyState:
    def __init__(self) -> None:
        self.params = {}
        self.param_history = {}
        self.exp_gene_set_factors = None
        self.exp_gene_factors = None
        self.exp_pheno_factors = None
        self.exp_lambdak = None
        self.factor_anchor_relevance = None
        self.factor_relevance = None
        self.consensus_mode = None
        self.consensus_reference_run = None
        self.consensus_run_diagnostics = None
        self.consensus_factor_support = None
        self.consensus_stats_out = None

    def _record_params(self, values, overwrite=False) -> None:
        if overwrite:
            self.params.update(values)
        else:
            for key, value in values.items():
                self.params[key] = value

    def _record_param(self, name, value) -> None:
        self.param_history.setdefault(name, []).append(value)

    def num_factors(self) -> int:
        if self.exp_lambdak is None:
            return 0
        return int(len(self.exp_lambdak))


class PhiAutoFactorRuntimeTest(unittest.TestCase):
    def test_weighted_jaccard_similarity_handles_simple_nonnegative_vectors(self) -> None:
        left = np.array([1.0, 1.0, 0.0])
        right = np.array([1.0, 0.0, 1.0])
        similarity = eaggl_factor_runtime._weighted_jaccard_similarity(left, right, weight_floor=0.0)
        self.assertAlmostEqual(similarity, 1.0 / 3.0)

    def test_select_phi_candidate_prefers_largest_factor_count_within_constraints(self) -> None:
        candidates = [
            {
                "phi": 0.02,
                "modal_factor_count": 4,
                "run_support": 1.0,
                "stability": 0.95,
                "redundancy": 0.91,
                "best_error": 1.0,
                "best_evidence": 8.0,
            },
            {
                "phi": 0.1,
                "modal_factor_count": 3,
                "run_support": 1.0,
                "stability": 0.95,
                "redundancy": 0.2,
                "best_error": 1.02,
                "best_evidence": 7.0,
            },
            {
                "phi": 1.0,
                "modal_factor_count": 2,
                "run_support": 1.0,
                "stability": 0.95,
                "redundancy": 0.1,
                "best_error": 1.01,
                "best_evidence": 6.0,
            },
        ]
        selected, reason = eaggl_factor_runtime._select_phi_candidate(
            candidates,
            max_redundancy=0.6,
            min_run_support=0.6,
            min_stability=0.85,
            max_fit_loss_frac=0.05,
        )
        self.assertEqual(reason, "max_factor_count_within_constraints")
        self.assertEqual(selected["phi"], 0.1)

    def test_run_factor_with_learn_phi_selects_less_redundant_candidate_before_final_run(self) -> None:
        state = _TinyState()
        final_runs = []

        def _stub_single(run_state, **kwargs):
            phi = float(kwargs["phi"])
            if phi < 0.05:
                matrix = np.array(
                    [
                        [0.9, 0.9, 0.1, 0.1],
                        [0.8, 0.8, 0.2, 0.2],
                        [0.0, 0.0, 1.0, 1.0],
                    ]
                )
                error = 1.0
                evidence = 10.0
            elif phi < 0.5:
                matrix = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                error = 1.02
                evidence = 8.0
            else:
                matrix = np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 0.0],
                    ]
                )
                error = 1.01
                evidence = 7.0
            run_state.exp_gene_set_factors = matrix
            run_state.exp_gene_factors = matrix
            run_state.exp_pheno_factors = None
            run_state.exp_lambdak = np.ones(matrix.shape[1], dtype=float)
            run_state.factor_anchor_relevance = np.ones((matrix.shape[1], 1), dtype=float)
            run_state.factor_relevance = np.ones(matrix.shape[1], dtype=float)
            final_runs.append(phi)
            return {
                "run_index": 0,
                "seed": None,
                "evidence": evidence,
                "likelihood": evidence,
                "reconstruction_error": error,
                "num_factors": matrix.shape[1],
                "factor_gene_set_x_pheno": False,
            }

        with mock.patch.object(eaggl_factor_runtime, "_run_factor_single", side_effect=_stub_single):
            eaggl_factor_runtime.run_factor(
                state,
                phi=0.1,
                learn_phi=True,
                learn_phi_runs_per_step=3,
                learn_phi_max_redundancy=0.6,
                learn_phi_min_run_support=0.6,
                learn_phi_min_stability=0.85,
                learn_phi_max_fit_loss_frac=0.05,
                learn_phi_max_steps=3,
                factor_runs=1,
                consensus_nmf=False,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda *args, **kwargs: None,
                log_fn=lambda *args, **kwargs: None,
                info_level=1,
                debug_level=2,
                trace_level=3,
                labeling_module=object(),
            )

        self.assertTrue(state.params["learn_phi"])
        self.assertGreaterEqual(state.params["learn_phi_selected_phi"], 0.05)
        self.assertLess(state.params["learn_phi_selected_phi"], 0.5)
        self.assertEqual(state.params["learn_phi_selection_reason"], "max_factor_count_within_constraints")
        self.assertEqual(state.num_factors(), 3)
        np.testing.assert_allclose(state.exp_gene_set_factors, np.eye(3), atol=1e-8)
        self.assertEqual(state.consensus_mode, "single")
        self.assertIn("learn_phi_candidate_phi", state.param_history)
        self.assertGreaterEqual(len(state.param_history["learn_phi_candidate_phi"]), 2)
        self.assertAlmostEqual(final_runs[-1], state.params["learn_phi_selected_phi"])


if __name__ == "__main__":
    unittest.main()
