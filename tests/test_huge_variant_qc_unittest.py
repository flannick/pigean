from __future__ import annotations

import unittest

import numpy as np

from pigean.huge import (
    compute_huge_variant_qc_mask,
    normalize_reported_standard_error,
    select_p_derived_z_mask,
    summarize_huge_variant_qc,
)


class HugeVariantQcTest(unittest.TestCase):
    def test_negative_standard_error_is_flagged_and_normalized(self) -> None:
        self.assertEqual(normalize_reported_standard_error(-0.25), (0.25, True))
        self.assertEqual(normalize_reported_standard_error(0.25), (0.25, False))
        self.assertEqual(normalize_reported_standard_error(None), (None, False))

    def test_observed_beta_and_se_take_precedence_for_z(self) -> None:
        mask = select_p_derived_z_mask(
            np.array([0.01, 0.01, 0.01, np.nan]),
            np.array([True, True, False, True]),
            np.array([False, True, False, False]),
        )
        np.testing.assert_array_equal(mask, np.array([False, True, True, False]))

    def test_reported_n_controls_sample_size_qc_even_when_se_is_present(self) -> None:
        mask = compute_huge_variant_qc_mask(
            np.array([0.01, 100.0, 0.01]),
            var_n=np.array([100.0, 100.0, 10.0]),
            reported_n_available=True,
            min_n_ratio=0.5,
        )
        np.testing.assert_array_equal(mask, np.array([True, True, False]))

    def test_missing_reported_n_fails_sample_size_qc(self) -> None:
        mask = compute_huge_variant_qc_mask(
            np.ones(3),
            var_n=np.array([100.0, np.nan, 100.0]),
            reported_n_available=True,
            min_n_ratio=0.5,
        )
        np.testing.assert_array_equal(mask, np.array([True, False, True]))

    def test_inverse_variance_filter_is_independent_and_opt_in(self) -> None:
        without_filter = compute_huge_variant_qc_mask(
            np.array([1.0, 1.0, 100.0]),
            var_n=np.full(3, 100.0),
            reported_n_available=True,
            min_n_ratio=0.5,
        )
        with_filter = compute_huge_variant_qc_mask(
            np.array([1.0, 1.0, 100.0]),
            var_n=np.full(3, 100.0),
            reported_n_available=True,
            min_n_ratio=0.5,
            min_inverse_variance_ratio=0.5,
        )
        np.testing.assert_array_equal(without_filter, np.array([True, True, True]))
        np.testing.assert_array_equal(with_filter, np.array([True, True, False]))

    def test_inverse_variance_remains_sample_size_fallback_without_n(self) -> None:
        mask = compute_huge_variant_qc_mask(
            np.array([1.0, 1.0, 100.0]),
            reported_n_available=False,
            min_n_ratio=0.5,
        )
        np.testing.assert_array_equal(mask, np.array([True, True, False]))

    def test_separate_inverse_variance_gate_uses_only_observed_se(self) -> None:
        mask = compute_huge_variant_qc_mask(
            np.array([1.0, 100.0, 100.0]),
            var_n=np.full(3, 100.0),
            reported_n_available=True,
            min_n_ratio=0.5,
            min_inverse_variance_ratio=0.5,
            inverse_variance_eligible=np.array([True, True, False]),
        )
        np.testing.assert_array_equal(mask, np.array([True, False, True]))

    def test_inverse_variance_filter_rejects_unknown_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "winsorized_mean.*mean"):
            compute_huge_variant_qc_mask(
                np.ones(2),
                min_n_ratio=0,
                min_inverse_variance_ratio=0.5,
                inverse_variance_reference="mode",
            )

    def test_winsorized_mean_resists_extreme_upper_tail(self) -> None:
        inverse_variance = np.array([1.0] * 99 + [1_000_000.0])
        mask = compute_huge_variant_qc_mask(
            1 / inverse_variance,
            min_n_ratio=0,
            min_inverse_variance_ratio=0.5,
            inverse_variance_reference="winsorized_mean",
            inverse_variance_reference_quantile=0.9,
        )
        np.testing.assert_array_equal(mask, np.ones(100, dtype=bool))

    def test_qc_summary_separates_inverse_variance_removals_and_forced_positions(self) -> None:
        summary = summarize_huge_variant_qc(
            np.array([True, True, False, True]),
            np.array([True, False, False, True]),
            np.array([True, True, False, True]),
            inverse_variance_eligible=np.array([True, True, True, False]),
            var_p=np.array([1e-9, 2e-8, 1e-10, 0.1]),
        )
        self.assertEqual(
            summary,
            {
                "input_variants": 4,
                "sample_size_kept": 3,
                "inverse_variance_eligible": 2,
                "inverse_variance_removed": 1,
                "final_kept": 3,
                "forced_retained": 1,
                "strong_signal_sample_size_kept": 2,
                "strong_signal_inverse_variance_eligible": 2,
                "strong_signal_inverse_variance_removed": 1,
                "strong_signal_qc_kept": 1,
                "strong_signal_final_kept": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
