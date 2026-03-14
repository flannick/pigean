from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pigean import model as pigean_model  # noqa: E402


class _StubState:
    def __init__(self) -> None:
        self.gene_sets = ["GS1", "GS2", "GS3"]
        self.beta_tildes_phewas = np.array([[1.0, 2.0, 3.0]])
        self.ses_phewas = np.array([[0.1, 0.2, 0.3]])
        self.scale_factors = np.ones(3)
        self.mean_shifts = np.zeros(3)
        self.X_orig = np.eye(3)
        self.ps = None
        self.sigma2s = None
        self.is_dense_gene_set = np.zeros(3, dtype=bool)
        self.p_values = np.array([0.1, 0.2, 0.3])
        self.betas = np.array([9.0, 9.0, 9.0])
        self.betas_phewas = None
        self.betas_uncorrected_phewas = None

    def _calculate_non_inf_betas(self, p, **kwargs):
        beta_tildes = kwargs.get("beta_tildes")
        num_gene_sets = beta_tildes.shape[-1]
        return np.arange(1, num_gene_sets + 1, dtype=float), np.ones(num_gene_sets, dtype=float)


class PhewasBetaSamplingTest(unittest.TestCase):
    def test_expand_phewas_gene_set_result_restores_full_axis(self) -> None:
        values = np.array([[1.5, 2.5]])
        mask = np.array([True, False, True])
        expanded = pigean_model._expand_phewas_gene_set_result(values, mask, 3)
        np.testing.assert_allclose(expanded, np.array([[1.5, 0.0, 2.5]]))

    def test_calculate_non_inf_betas_phewas_single_row_keeps_result_not_state_betas(self) -> None:
        state = _StubState()
        pigean_model.calculate_non_inf_betas(
            state,
            p=0.1,
            num_chains=2,
            run_betas_using_phewas=True,
            run_uncorrected_using_phewas=True,
            bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
            warn_fn=lambda *args, **kwargs: None,
            log_fn=lambda *args, **kwargs: None,
            info_level=1,
            debug_level=2,
            trace_level=3,
        )
        np.testing.assert_allclose(state.betas_phewas, np.array([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(state.betas_uncorrected_phewas, np.array([[1.0, 2.0, 3.0]]))


if __name__ == "__main__":
    unittest.main()
