from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pigean import x_inputs_core as pigean_x_inputs_core  # noqa: E402
from pigean.state import PigeanState  # noqa: E402


class BetaUncorrectedFullUniverseTest(unittest.TestCase):
    def _base_env(self) -> dict[str, str]:
        env = dict(os.environ)
        src_root = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return env

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "pigean", *args],
            cwd=REPO_ROOT,
            env=self._base_env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def _build_runtime(self) -> PigeanState:
        runtime = PigeanState(background_prior=0.05, batch_size=10)
        runtime._set_X(
            sparse.csc_matrix(np.eye(3)),
            ["G1", "G2", "G3"],
            ["GS1", "GS2", "GS3"],
            skip_V=True,
            skip_N=True,
        )
        runtime.beta_tildes = np.array([1.0, 0.8, 0.6])
        runtime.p_values = np.array([1e-4, 2e-4, 3e-4])
        runtime.ses = np.array([0.1, 0.1, 0.1])
        runtime.z_scores = np.array([10.0, 8.0, 6.0])

        def _fake_calc(*_args, **_kwargs):
            return np.array([3.0, 2.0, 1.0]), np.array([1.0, 1.0, 1.0])

        runtime._calculate_non_inf_betas = _fake_calc  # type: ignore[method-assign]
        return runtime

    def test_cap_preserves_missing_beta_uncorrected_values(self) -> None:
        runtime = self._build_runtime()
        sort_rank = pigean_x_inputs_core.maybe_filter_zero_uncorrected_betas_after_x_read_for_runtime(
            runtime,
            sort_rank=None,
            skip_betas=False,
            filter_gene_set_p=1.0,
            filter_using_phewas=False,
            retain_all_beta_uncorrected=True,
            independent_betas_only=False,
            max_num_burn_in=None,
            max_num_iter_betas=20,
            min_num_iter_betas=5,
            num_chains_betas=2,
            r_threshold_burn_in_betas=1.01,
            use_max_r_for_convergence_betas=True,
            max_frac_sem_betas=0.01,
            max_allowed_batch_correlation=None,
            sparse_solution=False,
            sparse_frac_betas=None,
            log_fn=lambda *_args, **_kwargs: None,
        )
        np.testing.assert_allclose(runtime.betas_uncorrected, np.array([3.0, 2.0, 1.0]))
        np.testing.assert_allclose(sort_rank, np.array([-3.0, -2.0, -1.0]))

        pigean_x_inputs_core.maybe_reduce_gene_sets_to_max_after_x_read_for_runtime(
            runtime,
            skip_betas=False,
            max_num_gene_sets=2,
            sort_rank=sort_rank,
            retain_all_beta_uncorrected=True,
            independent_betas_only=False,
            log_fn=lambda *_args, **_kwargs: None,
            debug_level=1,
            trace_level=2,
        )

        self.assertEqual(runtime.gene_sets, ["GS1", "GS2"])
        self.assertEqual(runtime.gene_sets_missing, ["GS3"])
        self.assertEqual(runtime.gene_set_filter_reason_missing, ["max_num_gene_sets_cap"])
        np.testing.assert_allclose(runtime.betas_uncorrected, np.array([3.0, 2.0]))
        np.testing.assert_allclose(runtime.betas_uncorrected_missing, np.array([1.0]))

    def test_retain_all_beta_uncorrected_writes_rows_beyond_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_prefix = tmp_path / "retain_all_uncorrected"
            proc = self._run(
                "betas",
                "--deterministic",
                "--hide-opts",
                "--X-in",
                "tests/data/model_small/gene_set_list_mouse_2024.txt",
                "--gene-map-in",
                "tests/data/model_small/portal_gencode.gene.map",
                "--gene-loc-file",
                "tests/data/model_small/NCBI37.3.plink.gene.loc",
                "--gene-list-in",
                "tests/data/t2d_smoke/mody.gene.list",
                "--gene-list-no-header",
                "--gene-list-all-in",
                "tests/data/model_small/NCBI37.3.plink.gene.loc",
                "--gene-list-all-id-col",
                "6",
                "--gene-list-all-no-header",
                "--gene-set-stats-out",
                str(out_prefix.with_suffix(".gene_set_stats.out")),
                "--gene-stats-out",
                str(out_prefix.with_suffix(".gene_stats.out")),
                "--params-out",
                str(out_prefix.with_suffix(".params.out")),
                "--num-chains-betas",
                "2",
                "--max-num-iter-betas",
                "20",
                "--min-num-iter-betas",
                "5",
                "--max-num-burn-in",
                "5",
                "--min-gene-set-size",
                "1",
                "--filter-gene-set-p",
                "1",
                "--max-gene-set-read-p",
                "1",
                "--no-filter-negative",
                "--max-num-gene-sets-initial",
                "200",
                "--max-num-gene-sets-hyper",
                "200",
                "--max-num-gene-sets",
                "5",
                "--retain-all-beta-uncorrected",
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))

            with out_prefix.with_suffix(".gene_set_stats.out").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))

        self.assertGreater(len(rows), 1)
        self.assertIn("filter_reason", rows[0])
        retained_uncorrected_rows = [
            row
            for row in rows
            if row.get("beta_uncorrected") not in (None, "", "NA")
            and abs(float(row["beta_uncorrected"])) > 0
            and row.get("beta") not in (None, "", "NA")
            and float(row["beta"]) == 0.0
            and row.get("filter_reason") == "max_num_gene_sets_cap"
        ]
        self.assertTrue(
            retained_uncorrected_rows,
            msg="Expected at least one capped-out row with preserved nonzero beta_uncorrected and zero beta",
        )

    def test_independent_betas_only_skips_corrected_beta_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_prefix = tmp_path / "independent_only"
            proc = self._run(
                "betas",
                "--deterministic",
                "--hide-opts",
                "--X-in",
                "tests/data/model_small/gene_set_list_mouse_2024.txt",
                "--gene-map-in",
                "tests/data/model_small/portal_gencode.gene.map",
                "--gene-loc-file",
                "tests/data/model_small/NCBI37.3.plink.gene.loc",
                "--gene-list-in",
                "tests/data/t2d_smoke/mody.gene.list",
                "--gene-list-no-header",
                "--gene-list-all-in",
                "tests/data/model_small/NCBI37.3.plink.gene.loc",
                "--gene-list-all-id-col",
                "6",
                "--gene-list-all-no-header",
                "--gene-set-stats-out",
                str(out_prefix.with_suffix(".gene_set_stats.out")),
                "--gene-stats-out",
                str(out_prefix.with_suffix(".gene_stats.out")),
                "--params-out",
                str(out_prefix.with_suffix(".params.out")),
                "--num-chains-betas",
                "2",
                "--max-num-iter-betas",
                "20",
                "--min-num-iter-betas",
                "5",
                "--max-num-burn-in",
                "5",
                "--min-gene-set-size",
                "1",
                "--filter-gene-set-p",
                "1",
                "--max-gene-set-read-p",
                "1",
                "--no-filter-negative",
                "--max-num-gene-sets-initial",
                "200",
                "--max-num-gene-sets-hyper",
                "200",
                "--independent-betas-only",
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            header = out_prefix.with_suffix(".gene_set_stats.out").read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("beta_uncorrected", header)
        self.assertNotIn("beta_internal", header)
        self.assertNotIn("\tbeta\t", "\t" + header + "\t")


if __name__ == "__main__":
    unittest.main()
