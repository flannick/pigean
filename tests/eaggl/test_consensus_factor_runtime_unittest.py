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

    def _record_params(self, values) -> None:
        self.params.update(values)

    def num_factors(self) -> int:
        if self.exp_lambdak is None:
            return 0
        return int(len(self.exp_lambdak))


class ConsensusFactorRuntimeTest(unittest.TestCase):
    def test_best_of_n_restarts_keeps_lowest_evidence_run(self) -> None:
        state = _TinyState()
        evidences = [5.0, 1.0, 3.0]
        outputs = [10.0, 20.0, 30.0]

        def _stub_single(run_state, **kwargs):
            run_index = len([x for x in outputs if x is not None]) - len(evidences)
            value = outputs.pop(0)
            evidence = evidences.pop(0)
            run_state.exp_gene_set_factors = np.array([[value]])
            run_state.exp_gene_factors = np.array([[value]])
            run_state.exp_pheno_factors = None
            run_state.exp_lambdak = np.array([1.0])
            run_state.factor_anchor_relevance = np.array([[0.5]])
            run_state.factor_relevance = np.array([value])
            return {
                "run_index": 0,
                "seed": None,
                "evidence": evidence,
                "likelihood": evidence,
                "reconstruction_error": evidence,
                "num_factors": 1,
                "factor_gene_set_x_pheno": False,
            }

        with mock.patch.object(eaggl_factor_runtime, "_run_factor_single", side_effect=_stub_single):
            eaggl_factor_runtime.run_factor(
                state,
                factor_runs=3,
                consensus_nmf=False,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda *args, **kwargs: None,
                log_fn=lambda *args, **kwargs: None,
                info_level=1,
                debug_level=2,
                trace_level=3,
                labeling_module=object(),
            )

        np.testing.assert_allclose(state.exp_gene_set_factors, np.array([[20.0]]))
        self.assertEqual(state.consensus_mode, "best_of_n")
        self.assertEqual(state.consensus_reference_run, 1)
        self.assertEqual(len(state.consensus_run_diagnostics), 3)

    def test_consensus_nmf_aligns_swapped_factor_order(self) -> None:
        state = _TinyState()
        factor_runs = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
        ]
        evidences = [3.0, 1.0, 2.0]

        def _stub_single(run_state, **kwargs):
            matrix = factor_runs.pop(0)
            evidence = evidences.pop(0)
            run_state.exp_gene_set_factors = matrix
            run_state.exp_gene_factors = matrix
            run_state.exp_pheno_factors = None
            run_state.exp_lambdak = np.array([1.0, 0.8])
            run_state.factor_anchor_relevance = np.array([[0.6], [0.4]])
            run_state.factor_relevance = np.array([0.6, 0.4])
            return {
                "run_index": 0,
                "seed": None,
                "evidence": evidence,
                "likelihood": evidence,
                "reconstruction_error": evidence,
                "num_factors": 2,
                "factor_gene_set_x_pheno": False,
            }

        with mock.patch.object(eaggl_factor_runtime, "_run_factor_single", side_effect=_stub_single):
            with mock.patch.object(eaggl_factor_runtime, "_finalize_factor_outputs", side_effect=lambda *args, **kwargs: None):
                eaggl_factor_runtime.run_factor(
                    state,
                    factor_runs=3,
                    consensus_nmf=True,
                    consensus_min_factor_cosine=0.5,
                    consensus_min_run_support=0.5,
                    consensus_aggregation="median",
                    bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                    warn_fn=lambda *args, **kwargs: None,
                    log_fn=lambda *args, **kwargs: None,
                    info_level=1,
                    debug_level=2,
                    trace_level=3,
                    labeling_module=object(),
                )

        self.assertEqual(state.consensus_mode, "consensus")
        self.assertEqual(state.consensus_reference_run, 1)
        self.assertEqual(state.exp_gene_set_factors.shape, (2, 2))
        np.testing.assert_allclose(state.exp_gene_set_factors[:, 0], np.array([1.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(state.exp_gene_set_factors[:, 1], np.array([0.0, 1.0]), atol=1e-8)
        self.assertEqual(len(state.consensus_factor_support), 2)
        self.assertTrue(all(row["support_runs"] == 3 for row in state.consensus_factor_support))


if __name__ == "__main__":
    unittest.main()
