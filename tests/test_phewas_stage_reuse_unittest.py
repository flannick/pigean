from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = ["pigean.py", "beta_tildes"]
import src.pigean as pigean  # noqa: E402


class _StubState:
    def __init__(self, *, loaded_gene_phewas: bool, num_filtered: int) -> None:
        self._loaded_gene_phewas = loaded_gene_phewas
        self.num_gene_phewas_filtered = num_filtered
        self.run_phewas_calls: list[dict[str, object]] = []
        self.write_calls: list[str] = []

    def read_gene_phewas(self):
        return self._loaded_gene_phewas

    def run_phewas(self, **kwargs):
        self.run_phewas_calls.append(kwargs)

    def write_phewas_statistics(self, path):
        self.write_calls.append(path)


def _make_options(*, run_input: str, loaded_input: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        run_phewas_from_gene_phewas_stats_in=run_input,
        gene_phewas_bfs_in=loaded_input,
        gene_phewas_bfs_id_col=None,
        gene_phewas_bfs_pheno_col=None,
        gene_phewas_bfs_log_bf_col=None,
        gene_phewas_bfs_combined_col=None,
        gene_phewas_bfs_prior_col=None,
        max_num_burn_in=5,
        max_num_iter_betas=10,
        min_num_iter_betas=2,
        num_chains_betas=2,
        r_threshold_burn_in_betas=1.01,
        use_max_r_for_convergence_betas=True,
        max_frac_sem_betas=0.5,
        gauss_seidel_betas=False,
        sparse_solution=False,
        sparse_frac_betas=None,
        phewas_stats_out=None,
    )


class PhewasStageReuseTest(unittest.TestCase):
    def test_output_phewas_stage_rereads_when_matrix_not_loaded(self) -> None:
        state = _StubState(loaded_gene_phewas=False, num_filtered=0)
        options = _make_options(run_input="gene_phewas.tsv", loaded_input=None)
        logs: list[str] = []
        orig_log = pigean.log
        try:
            pigean.log = lambda message, *args, **kwargs: logs.append(message)
            pigean._run_advanced_set_b_output_phewas_if_requested(state, options)
        finally:
            pigean.log = orig_log

        self.assertEqual(len(state.run_phewas_calls), 1)
        self.assertEqual(state.run_phewas_calls[0]["gene_phewas_bfs_in"], "gene_phewas.tsv")
        self.assertTrue(any("mode=re_read_file" in line for line in logs))
        self.assertTrue(any("reason=matrix_not_loaded" in line for line in logs))

    def test_output_phewas_stage_reuses_loaded_matrix_when_compatible(self) -> None:
        state = _StubState(loaded_gene_phewas=True, num_filtered=0)
        options = _make_options(run_input="gene_phewas.tsv", loaded_input="gene_phewas.tsv")
        logs: list[str] = []
        orig_log = pigean.log
        try:
            pigean.log = lambda message, *args, **kwargs: logs.append(message)
            pigean._run_advanced_set_b_output_phewas_if_requested(state, options)
        finally:
            pigean.log = orig_log

        self.assertEqual(len(state.run_phewas_calls), 1)
        self.assertIsNone(state.run_phewas_calls[0]["gene_phewas_bfs_in"])
        self.assertTrue(any("mode=reuse_loaded_matrix" in line for line in logs))
        self.assertTrue(any("reason=requested_input_matches_loaded_source" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
