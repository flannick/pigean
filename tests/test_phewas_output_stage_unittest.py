from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pigean import phewas as pigean_phewas  # noqa: E402


class _StubState:
    def __init__(self) -> None:
        self.genes = ["GENE1", "GENE2"]
        self.gene_to_ind = {"GENE1": 0, "GENE2": 1}
        self.gene_label_map = None
        self.phenos = None
        self.pheno_to_ind = None
        self.num_gene_phewas_filtered = 0

        self.Y = np.array([1.0, 0.6])
        self.combined_prior_Ys = np.array([0.9, 0.4])
        self.priors = np.array([0.7, 0.2])
        self.background_bf = 0.0

        self.X_orig = np.eye(2)
        self.X_phewas_beta = sparse.csr_matrix(np.ones((2, 2)))
        self.debug_skip_correlation = False

        for spec in pigean_phewas.pegs_phewas.GENE_LEVEL_PHEWAS_COMPARISONS:
            for suffix in ("beta", "beta_tilde", "se", "Z", "p_value"):
                setattr(self, f"{spec['output_base']}_{suffix}", None)

    def read_gene_phewas(self):
        return False


def _fake_beta_outputs():
    beta = np.array(
        [
            [11.0, 12.0],
            [21.0, 22.0],
            [31.0, 32.0],
        ]
    )
    beta_tilde = beta + 0.1
    se = beta + 0.2
    z_score = beta + 0.3
    p_value = beta + 0.4
    return beta, None, beta_tilde, se, z_score, p_value, None


class PhewasOutputStageTest(unittest.TestCase):
    def test_run_phewas_stages_file_once_and_defaults_to_matched_comparisons(self) -> None:
        state = _StubState()
        staged_y = sparse.csc_matrix(np.array([[2.0, 1.5], [1.8, 1.2]]))
        staged_combined = sparse.csc_matrix(np.array([[1.6, 1.1], [1.3, 0.9]]))

        with mock.patch.object(
            pigean_phewas,
            "stage_gene_level_phewas_file_once",
            return_value=(["P1", "P2"], staged_y, staged_combined),
        ) as stage_mock, mock.patch.object(
            pigean_phewas,
            "read_phewas_file_batch",
            side_effect=AssertionError("per-batch reread should not be used"),
        ), mock.patch.object(
            pigean_phewas,
            "calculate_phewas_block",
            side_effect=lambda *args, **kwargs: _fake_beta_outputs(),
        ) as direct_mock, mock.patch.object(
            pigean_phewas,
            "calculate_combined_phewas_block_with_sparse_correlation",
            side_effect=lambda *args, **kwargs: _fake_beta_outputs(),
        ) as combined_mock:
            pigean_phewas.run_phewas(
                state,
                gene_phewas_bfs_in="phewas.tsv",
                gene_phewas_bfs_id_col="Gene",
                gene_phewas_bfs_pheno_col="Pheno",
                gene_phewas_bfs_log_bf_col="log_bf",
                gene_phewas_bfs_combined_col="combined",
                min_value=0.0,
                phewas_comparison_set="matched",
                batch_size=50,
                max_num_burn_in=5,
                max_num_iter=10,
                min_num_iter=2,
                num_chains=2,
                r_threshold_burn_in=1.01,
                use_max_r_for_convergence=True,
                max_frac_sem=0.5,
                gauss_seidel=False,
                sparse_solution=False,
                sparse_frac_betas=None,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda _msg: None,
                log_fn=lambda _msg, _lvl=0: None,
                info_level=1,
                debug_level=0,
                trace_level=0,
                open_text_fn=open,
                get_col_fn=lambda *args, **kwargs: 0,
                construct_map_to_ind_fn=lambda values: {value: i for i, value in enumerate(values)},
            )

        self.assertEqual(stage_mock.call_count, 1)
        self.assertEqual(direct_mock.call_count, 1)
        self.assertEqual(combined_mock.call_count, 1)
        self.assertIsNotNone(state.pheno_Y_vs_input_Y_beta)
        self.assertIsNotNone(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta)
        self.assertIsNone(state.pheno_Y_vs_input_combined_prior_Ys_beta)
        self.assertIsNone(state.pheno_Y_vs_input_priors_beta)
        self.assertIsNone(state.pheno_combined_prior_Ys_vs_input_Y_beta)
        self.assertIsNone(state.pheno_combined_prior_Ys_vs_input_priors_beta)

    def test_run_phewas_diagnostic_comparison_set_populates_cross_family_outputs(self) -> None:
        state = _StubState()
        staged_y = sparse.csc_matrix(np.array([[2.0, 1.5], [1.8, 1.2]]))
        staged_combined = sparse.csc_matrix(np.array([[1.6, 1.1], [1.3, 0.9]]))

        with mock.patch.object(
            pigean_phewas,
            "stage_gene_level_phewas_file_once",
            return_value=(["P1", "P2"], staged_y, staged_combined),
        ), mock.patch.object(
            pigean_phewas,
            "calculate_phewas_block",
            side_effect=lambda *args, **kwargs: _fake_beta_outputs(),
        ), mock.patch.object(
            pigean_phewas,
            "calculate_combined_phewas_block_with_sparse_correlation",
            side_effect=lambda *args, **kwargs: _fake_beta_outputs(),
        ):
            pigean_phewas.run_phewas(
                state,
                gene_phewas_bfs_in="phewas.tsv",
                gene_phewas_bfs_id_col="Gene",
                gene_phewas_bfs_pheno_col="Pheno",
                gene_phewas_bfs_log_bf_col="log_bf",
                gene_phewas_bfs_combined_col="combined",
                min_value=0.0,
                phewas_comparison_set="diagnostic",
                batch_size=50,
                max_num_burn_in=5,
                max_num_iter=10,
                min_num_iter=2,
                num_chains=2,
                r_threshold_burn_in=1.01,
                use_max_r_for_convergence=True,
                max_frac_sem=0.5,
                gauss_seidel=False,
                sparse_solution=False,
                sparse_frac_betas=None,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda _msg: None,
                log_fn=lambda _msg, _lvl=0: None,
                info_level=1,
                debug_level=0,
                trace_level=0,
                open_text_fn=open,
                get_col_fn=lambda *args, **kwargs: 0,
                construct_map_to_ind_fn=lambda values: {value: i for i, value in enumerate(values)},
            )

        self.assertIsNotNone(state.pheno_Y_vs_input_Y_beta)
        self.assertIsNotNone(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta)
        self.assertIsNotNone(state.pheno_Y_vs_input_combined_prior_Ys_beta)
        self.assertIsNotNone(state.pheno_Y_vs_input_priors_beta)
        self.assertIsNotNone(state.pheno_combined_prior_Ys_vs_input_Y_beta)
        self.assertIsNotNone(state.pheno_combined_prior_Ys_vs_input_priors_beta)


if __name__ == "__main__":
    unittest.main()
