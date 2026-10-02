from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pegs_shared.huge_cache import apply_huge_statistics_meta_to_runtime, build_huge_statistics_meta
from pigean.state import PigeanState
from pigean.y_inputs_core import (
    _cached_signal_posteriors,
    _resolve_cached_operation,
    postprocess_cached_huge_statistics,
    apply_gene_covariates_and_correct_huge,
)


class CachedHugeCalibrationTest(unittest.TestCase):
    def test_auto_clears_partial_covariates_after_generation_failure(self) -> None:
        runtime = SimpleNamespace(huge_statistics_meta={}, gene_covariates=None,
            cached_huge_score_correction_effective=True, cached_huge_score_correction_mode="auto")
        def build(**kwargs):
            runtime.gene_covariates = np.ones((2, 3))
            raise ValueError("incomplete gene coverage")
        runtime.build_cached_huge_gene_covariates = build
        warnings = []
        def bail(message):
            raise AssertionError(message)
        apply_gene_covariates_and_correct_huge(runtime, gene_loc_file="locations",
            log_fn=lambda *args: None, warn_fn=warnings.append, trace_level=0, bail_fn=bail)
        self.assertIsNone(runtime.gene_covariates)
        self.assertFalse(runtime.cached_huge_score_correction_effective)
        self.assertIn("incomplete gene coverage", warnings[0])

    def test_cache_v2_records_optional_calibration_provenance(self) -> None:
        runtime = {
            "high_power_calibration_applied": True,
            "huge_score_correction_applied": False,
            "high_power_calibration_provenance": {"method": "test"},
        }
        matrix = sparse.csc_matrix((0, 0))
        meta = build_huge_statistics_meta(runtime, matrix, matrix)
        self.assertEqual(meta["version"], 2)
        self.assertIs(meta["high_power_calibration_applied"], True)
        self.assertIs(meta["huge_score_correction_applied"], False)
        self.assertEqual(meta["high_power_calibration_provenance"]["method"], "test")

    def test_v1_cache_without_flags_loads_as_unknown(self) -> None:
        runtime = {}
        meta = {
            "version": 1,
            "huge_signal_max_closest_gene_prob": 0.9,
            "huge_cap_region_posterior": True,
            "huge_scale_region_posterior": False,
            "huge_phantom_region_posterior": False,
            "huge_allow_evidence_of_absence": False,
            "huge_sparse_mode": False,
            "huge_signals": [],
        }
        apply_huge_statistics_meta_to_runtime(runtime, meta)
        self.assertEqual(runtime["huge_statistics_cache_version"], 1)
        self.assertIsNone(runtime["high_power_calibration_applied"])
        self.assertIsNone(runtime["huge_score_correction_applied"])

    def test_auto_skips_applied_and_warns_then_applies_unknown(self) -> None:
        warnings: list[str] = []
        logs: list[str] = []
        self.assertFalse(
            _resolve_cached_operation("auto", True, True, "test", warn_fn=warnings.append, log_fn=logs.append)
        )
        self.assertTrue(
            _resolve_cached_operation("auto", True, None, "test", warn_fn=warnings.append, log_fn=logs.append)
        )
        self.assertEqual(len(warnings), 1)

    def test_force_and_skip_override_provenance(self) -> None:
        sink: list[str] = []
        self.assertTrue(
            _resolve_cached_operation("force", False, True, "test", warn_fn=sink.append, log_fn=sink.append)
        )
        self.assertFalse(
            _resolve_cached_operation("skip", True, False, "test", warn_fn=sink.append, log_fn=sink.append)
        )

    def test_signal_posterior_is_monotone_with_stronger_p(self) -> None:
        posterior = _cached_signal_posteriors([1e-4, 1e-8, 1e-20], 0.5, 0.01)
        self.assertTrue(np.all(np.diff(posterior) > 0))

    def test_signal_posterior_reproduces_calibration_endpoints(self) -> None:
        class CalibrationRuntime:
            compute_allelic_var_and_prior = PigeanState.compute_allelic_var_and_prior

        runtime = CalibrationRuntime()
        high_p, high_posterior = 1e-2, 1e-3
        low_p, low_posterior = 5e-8, 0.98
        k, prior_odds = runtime.compute_allelic_var_and_prior(
            high_p, high_posterior, low_p, low_posterior
        )
        observed = _cached_signal_posteriors([high_p, low_p], k, prior_odds)
        np.testing.assert_allclose(observed, [high_posterior, low_posterior], rtol=1e-10)

    def test_correction_only_preserves_unknown_input_scores_until_correction(self) -> None:
        runtime = SimpleNamespace(
            high_power_calibration_applied=None,
            huge_score_correction_applied=None,
        )
        cached = (np.array([1.0]), [], np.array([]), np.array([2.0]), np.array([]))
        with mock.patch("pigean.y_inputs_core._redistill_cached_huge_scores") as redistill:
            observed = postprocess_cached_huge_statistics(
                runtime,
                cached,
                cached_high_power_calibration="skip",
                cached_huge_score_correction="force",
                warn_fn=lambda _message: None,
                log_fn=lambda _message: None,
                bail_fn=self.fail,
            )
        redistill.assert_not_called()
        np.testing.assert_array_equal(observed[0], cached[0])
        np.testing.assert_array_equal(observed[3], cached[3])


if __name__ == "__main__":
    unittest.main()
