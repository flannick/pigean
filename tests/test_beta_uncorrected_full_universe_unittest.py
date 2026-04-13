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
from pigean import model as pigean_model  # noqa: E402
from pigean import state as pigean_state_module  # noqa: E402
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
            track_filtered_beta_uncorrected=False,
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
            track_filtered_beta_uncorrected=False,
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

    def test_track_filtered_beta_uncorrected_routes_later_filtered_rows_to_ignored_sidecar(self) -> None:
        runtime = self._build_runtime()
        runtime.track_filtered_beta_uncorrected = True

        runtime.subset_gene_sets(
            np.array([True, False, True]),
            keep_missing=False,
            ignore_missing=True,
            skip_V=True,
            filter_reason="correlation_pruning",
        )

        self.assertEqual(runtime.gene_sets, ["GS1", "GS3"])
        self.assertEqual(runtime.gene_sets_ignored, ["GS2"])
        np.testing.assert_array_equal(runtime.gene_set_track_beta_uncorrected_ignored, np.array([True]))
        self.assertIsNotNone(runtime.X_orig_ignored_gene_sets)
        self.assertEqual(runtime.X_orig_ignored_gene_sets.shape, (3, 1))

    def test_tracked_ignored_uncorrected_betas_are_computed_for_ignored_rows(self) -> None:
        runtime = self._build_runtime()
        runtime.track_filtered_beta_uncorrected = True
        runtime.subset_gene_sets(
            np.array([True, False, True]),
            keep_missing=False,
            ignore_missing=True,
            skip_V=True,
            filter_reason="correlation_pruning",
        )

        def _fake_calc(*_args, **_kwargs):
            return np.array([1.25]), np.array([0.5])

        runtime._calculate_non_inf_betas = _fake_calc  # type: ignore[method-assign]

        pigean_model.update_tracked_ignored_uncorrected_betas(
            runtime,
            beta_tildes=runtime.beta_tildes_ignored[np.array([True])],
            ses=runtime.ses_ignored[np.array([True])],
            scale_factors=runtime.scale_factors_ignored[np.array([True])],
            mean_shifts=runtime.mean_shifts_ignored[np.array([True])],
            max_num_burn_in=5,
            max_num_iter=20,
            min_num_iter=5,
            num_chains=2,
            r_threshold_burn_in=1.01,
            use_max_r_for_convergence=True,
            max_frac_sem=0.01,
            max_allowed_batch_correlation=None,
            gauss_seidel=False,
            sparse_solution=False,
            sparse_frac_betas=None,
        )

        np.testing.assert_allclose(runtime.betas_uncorrected_ignored, np.array([1.25]))
        np.testing.assert_allclose(runtime.non_inf_avg_postps_ignored, np.array([0.5]))

    def test_tracked_ignored_uncorrected_betas_accept_tracked_only_ignored_metadata(self) -> None:
        runtime = self._build_runtime()
        runtime.track_filtered_beta_uncorrected = True
        runtime.gene_sets_ignored = ["PREFILTER1", "TRACKED1", "PREFILTER2", "TRACKED2", "PREFILTER3"]
        runtime.gene_set_track_beta_uncorrected_ignored = np.array([False, True, False, True, False])
        runtime.X_orig_ignored_gene_sets = sparse.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        runtime.is_dense_gene_set_ignored = np.array([True, False], dtype=bool)
        runtime.ps_ignored = np.array([0.2, 0.4], dtype=float)
        runtime.sigma2s_ignored = np.array([1.5, 2.5], dtype=float)

        captured = {}

        def _fake_calc(*_args, **kwargs):
            captured["is_dense_gene_set"] = np.array(kwargs["is_dense_gene_set"], copy=True)
            captured["ps"] = np.array(kwargs["ps"], copy=True)
            captured["sigma2s"] = np.array(kwargs["sigma2s"], copy=True)
            return np.array([1.25, 0.75]), np.array([0.5, 0.25])

        runtime._calculate_non_inf_betas = _fake_calc  # type: ignore[method-assign]

        pigean_model.update_tracked_ignored_uncorrected_betas(
            runtime,
            beta_tildes=np.array([0.8, 0.6]),
            ses=np.array([0.1, 0.1]),
            scale_factors=np.array([1.0, 1.0]),
            mean_shifts=np.array([0.0, 0.0]),
            max_num_burn_in=5,
            max_num_iter=20,
            min_num_iter=5,
            num_chains=2,
            r_threshold_burn_in=1.01,
            use_max_r_for_convergence=True,
            max_frac_sem=0.01,
            max_allowed_batch_correlation=None,
            gauss_seidel=False,
            sparse_solution=False,
            sparse_frac_betas=None,
        )

        np.testing.assert_array_equal(captured["is_dense_gene_set"], np.array([True, False]))
        np.testing.assert_allclose(captured["ps"], np.array([0.2, 0.4]))
        np.testing.assert_allclose(captured["sigma2s"], np.array([1.5, 2.5]))
        np.testing.assert_allclose(runtime.betas_uncorrected_ignored, np.array([0.0, 1.25, 0.0, 0.75, 0.0]))
        np.testing.assert_allclose(runtime.non_inf_avg_postps_ignored, np.array([0.0, 0.5, 0.0, 0.25, 0.0]))

    def test_tracked_ignored_uncorrected_betas_collapse_two_dimensional_means(self) -> None:
        runtime = self._build_runtime()
        runtime.track_filtered_beta_uncorrected = True
        runtime.gene_sets_ignored = ["TRACKED1", "TRACKED2"]
        runtime.gene_set_track_beta_uncorrected_ignored = np.array([True, True])
        runtime.X_orig_ignored_gene_sets = sparse.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

        def _fake_calc(*_args, **_kwargs):
            return (
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([[10.0, 20.0], [30.0, 40.0]]),
                np.array([[0.5, 0.6], [0.7, 0.8]]),
            )

        runtime._calculate_non_inf_betas = _fake_calc  # type: ignore[method-assign]

        result = pigean_model.update_tracked_ignored_uncorrected_betas(
            runtime,
            beta_tildes=np.array([[0.8, 0.6], [0.7, 0.5]]),
            ses=np.array([[0.1, 0.1], [0.1, 0.1]]),
            scale_factors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            mean_shifts=np.array([[0.0, 0.0], [0.0, 0.0]]),
            return_sample=True,
            max_num_burn_in=5,
            max_num_iter=20,
            min_num_iter=5,
            num_chains=2,
            r_threshold_burn_in=1.01,
            use_max_r_for_convergence=True,
            max_frac_sem=0.01,
            max_allowed_batch_correlation=None,
            gauss_seidel=False,
            sparse_solution=False,
            sparse_frac_betas=None,
        )

        np.testing.assert_allclose(result["betas_uncorrected_mean_m"], np.array([20.0, 30.0]))
        np.testing.assert_allclose(result["postp_mean_m"], np.array([0.6, 0.7]))
        np.testing.assert_allclose(runtime.betas_uncorrected_ignored, np.array([20.0, 30.0]))
        np.testing.assert_allclose(runtime.non_inf_avg_postps_ignored, np.array([0.6, 0.7]))

    def test_tracked_ignored_uncorrected_betas_collapse_object_like_rows(self) -> None:
        runtime = self._build_runtime()
        runtime.track_filtered_beta_uncorrected = True
        runtime.gene_sets_ignored = ["TRACKED1", "TRACKED2"]
        runtime.gene_set_track_beta_uncorrected_ignored = np.array([True, True])
        runtime.X_orig_ignored_gene_sets = sparse.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

        def _fake_calc(*_args, **_kwargs):
            return (
                None,
                None,
                [np.array([10.0, 20.0]), np.array([30.0, 40.0])],
                [np.array([0.5, 0.6]), np.array([0.7, 0.8])],
            )

        runtime._calculate_non_inf_betas = _fake_calc  # type: ignore[method-assign]

        result = pigean_model.update_tracked_ignored_uncorrected_betas(
            runtime,
            beta_tildes=np.array([[0.8, 0.6], [0.7, 0.5]]),
            ses=np.array([[0.1, 0.1], [0.1, 0.1]]),
            scale_factors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            mean_shifts=np.array([[0.0, 0.0], [0.0, 0.0]]),
            return_sample=True,
            max_num_burn_in=5,
            max_num_iter=20,
            min_num_iter=5,
            num_chains=2,
            r_threshold_burn_in=1.01,
            use_max_r_for_convergence=True,
            max_frac_sem=0.01,
            max_allowed_batch_correlation=None,
            gauss_seidel=False,
            sparse_solution=False,
            sparse_frac_betas=None,
        )

        np.testing.assert_allclose(result["betas_uncorrected_mean_m"], np.array([20.0, 30.0]))
        np.testing.assert_allclose(result["postp_mean_m"], np.array([0.6, 0.7]))
        np.testing.assert_allclose(runtime.betas_uncorrected_ignored, np.array([20.0, 30.0]))
        np.testing.assert_allclose(runtime.non_inf_avg_postps_ignored, np.array([0.6, 0.7]))

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

    def test_track_filtered_beta_uncorrected_writes_ignored_rows_beyond_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_prefix = tmp_path / "track_filtered_uncorrected"
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
                "--track-filtered-beta-uncorrected",
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))

            with out_prefix.with_suffix(".gene_set_stats.out").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))

        tracked_rows = [
            row
            for row in rows
            if row.get("beta_uncorrected") not in (None, "", "NA")
            and abs(float(row["beta_uncorrected"])) > 0
            and row.get("beta") not in (None, "", "NA")
            and float(row["beta"]) == 0.0
            and row.get("filter_reason") == "max_num_gene_sets_cap"
        ]
        self.assertTrue(
            tracked_rows,
            msg="Expected at least one capped-out ignored row with preserved nonzero beta_uncorrected and zero beta",
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

    def test_apply_gibbs_ignored_final_state_restores_full_ignored_alignment(self) -> None:
        runtime = self._build_runtime()
        runtime.gene_sets_ignored = ["PREFILTER", "TRACKED"]
        runtime.gene_set_track_beta_uncorrected_ignored = np.array([False, True])
        runtime._gibbs_sum_betas_uncorrected_ignored_m = np.array([[1.0], [3.0]])
        runtime._gibbs_sum_betas_uncorrected2_ignored_m = np.array([[1.0], [9.0]])
        runtime._gibbs_sum_postps_ignored_m = np.array([[0.25], [0.75]])
        runtime._gibbs_sum_postps2_ignored_m = np.array([[0.0625], [0.5625]])
        runtime._gibbs_num_sum_beta_ignored_m = np.array([[1.0], [1.0]])

        pigean_state_module._apply_gibbs_ignored_final_state(runtime)

        np.testing.assert_allclose(runtime.betas_uncorrected_ignored, np.array([0.0, 2.0]))
        np.testing.assert_allclose(runtime.non_inf_avg_postps_ignored, np.array([0.0, 0.5]))
        np.testing.assert_allclose(runtime.non_inf_avg_cond_betas_ignored, np.array([0.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
