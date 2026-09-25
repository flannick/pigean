from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.stats


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pegs_utils  # noqa: E402


class GwasZConcordanceTest(unittest.TestCase):
    @staticmethod
    def _p_from_z(z):
        return 2 * scipy.stats.norm.cdf(-np.abs(z))

    def test_concordant_observed_columns_do_not_warn(self) -> None:
        z = np.linspace(-5, 5, 201)
        z[z == 0] = 0.01
        se = np.full(z.shape, 0.2)
        se[::2] *= -1
        beta = z * np.abs(se)

        stats = pegs_utils.initialize_gwas_z_concordance_stats()
        pegs_utils.update_gwas_z_concordance_stats(
            stats,
            self._p_from_z(z),
            beta,
            se,
        )
        metrics = pegs_utils.finalize_gwas_z_concordance_stats(stats)

        self.assertEqual(metrics["n"], z.size)
        self.assertAlmostEqual(metrics["pearson"], 1.0, places=12)
        self.assertAlmostEqual(metrics["beta_se_on_p_slope"], 1.0, places=12)
        self.assertAlmostEqual(metrics["mean_abs_delta_z"], 0.0, places=12)
        self.assertFalse(pegs_utils.gwas_z_concordance_is_material(metrics))

    def test_misscaled_se_produces_material_inflation_warning(self) -> None:
        p_z = np.linspace(-5, -2, 200)
        reported_se = np.full(p_z.shape, 0.1)
        beta = p_z * reported_se * 1.5

        stats = pegs_utils.initialize_gwas_z_concordance_stats()
        pegs_utils.update_gwas_z_concordance_stats(
            stats,
            self._p_from_z(p_z),
            beta,
            reported_se,
        )
        metrics = pegs_utils.finalize_gwas_z_concordance_stats(stats)

        self.assertAlmostEqual(metrics["pearson"], 1.0, places=12)
        self.assertAlmostEqual(metrics["beta_se_on_p_slope"], 1.5, places=12)
        self.assertGreater(metrics["mean_abs_delta_z"], 0.1)
        self.assertGreater(metrics["frac_inflated_gt_0_5"], 0.5)
        self.assertTrue(pegs_utils.gwas_z_concordance_is_material(metrics))
        self.assertIn("beta/SE inflated>0.5", pegs_utils.format_gwas_z_concordance(metrics))

    def test_inferred_se_values_are_excluded(self) -> None:
        p_z = np.linspace(2, 4, 200)
        p = self._p_from_z(p_z)
        beta = p_z * 0.1
        se = np.full(p_z.shape, 0.1)
        inferred = np.zeros(p_z.shape, dtype=bool)
        inferred[:50] = True
        beta[:50] *= 20

        stats = pegs_utils.initialize_gwas_z_concordance_stats()
        pegs_utils.update_gwas_z_concordance_stats(
            stats,
            p,
            beta,
            se,
            se_was_inferred=inferred,
        )
        metrics = pegs_utils.finalize_gwas_z_concordance_stats(stats)

        self.assertEqual(metrics["n"], 150)
        self.assertAlmostEqual(metrics["mean_abs_delta_z"], 0.0, places=12)
        self.assertFalse(pegs_utils.gwas_z_concordance_is_material(metrics))

    def test_fewer_than_100_variants_are_reported_but_not_warned(self) -> None:
        p_z = np.full(99, 3.0)
        stats = pegs_utils.initialize_gwas_z_concordance_stats()
        pegs_utils.update_gwas_z_concordance_stats(
            stats,
            self._p_from_z(p_z),
            p_z * 0.2,
            np.full(p_z.shape, 0.1),
        )
        metrics = pegs_utils.finalize_gwas_z_concordance_stats(stats)

        self.assertEqual(metrics["n"], 99)
        self.assertFalse(pegs_utils.gwas_z_concordance_is_material(metrics))


if __name__ == "__main__":
    unittest.main()
