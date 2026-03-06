from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = ["eaggl.py", "factor"]
import src.eaggl as eaggl  # noqa: E402


class _StubRuntime:
    def __init__(self, *, loaded_gene_phewas: bool, num_filtered: int) -> None:
        self.gene_pheno_Y = [[1.0]] if loaded_gene_phewas else None
        self.gene_pheno_combined_prior_Ys = None
        self.gene_pheno_priors = None
        self.num_gene_phewas_filtered = num_filtered
        self.run_phewas_calls: list[dict[str, object]] = []
        self.write_phewas_calls: list[str] = []
        self.write_factor_phewas_calls: list[str] = []
        self._num_factors = 1

    def run_phewas(self, **kwargs):
        self.run_phewas_calls.append(kwargs)

    def write_phewas_statistics(self, path):
        self.write_phewas_calls.append(path)

    def write_factor_phewas_statistics(self, path):
        self.write_factor_phewas_calls.append(path)

    def num_factors(self):
        return self._num_factors


def _make_options(*, main_input: str | None, factor_input: str | None, loaded_input: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        run_phewas_from_gene_phewas_stats_in=main_input,
        factor_phewas_from_gene_phewas_stats_in=factor_input,
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
        factor_phewas_stats_out=None,
        factor_phewas_min_gene_factor_weight=0.0,
    )


class EagglPhewasStageReuseTest(unittest.TestCase):
    def test_main_phewas_stage_rereads_when_matrix_not_loaded(self) -> None:
        runtime = _StubRuntime(loaded_gene_phewas=False, num_filtered=0)
        options = _make_options(main_input="gene_phewas.tsv", factor_input=None, loaded_input=None)
        logs: list[str] = []
        orig_log = eaggl.log
        try:
            eaggl.log = lambda message, *args, **kwargs: logs.append(message)
            eaggl._run_main_phewas_stage(runtime, options)
        finally:
            eaggl.log = orig_log

        self.assertEqual(len(runtime.run_phewas_calls), 1)
        self.assertEqual(runtime.run_phewas_calls[0]["gene_phewas_bfs_in"], "gene_phewas.tsv")
        self.assertTrue(any("stage 'phewas': mode=re_read_file" in line for line in logs))
        self.assertTrue(any("reason=matrix_not_loaded" in line for line in logs))

    def test_factor_phewas_stage_reuses_loaded_matrix(self) -> None:
        runtime = _StubRuntime(loaded_gene_phewas=True, num_filtered=0)
        options = _make_options(
            main_input="gene_phewas.tsv",
            factor_input="gene_phewas.tsv",
            loaded_input="gene_phewas.tsv",
        )
        logs: list[str] = []
        orig_log = eaggl.log
        try:
            eaggl.log = lambda message, *args, **kwargs: logs.append(message)
            eaggl._run_main_factor_phewas_stage(runtime, options)
        finally:
            eaggl.log = orig_log

        self.assertEqual(len(runtime.run_phewas_calls), 1)
        self.assertIsNone(runtime.run_phewas_calls[0]["gene_phewas_bfs_in"])
        self.assertTrue(any("stage 'factor_phewas': mode=reuse_loaded_matrix" in line for line in logs))
        self.assertTrue(any("reason=requested_input_matches_loaded_source" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
