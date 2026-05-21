from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


class MultiYWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    def _run(self, mode: str, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        src_root = str(self.repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return subprocess.run(
            [sys.executable, "-m", "pigean", mode, *args],
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def _effective_config(self, mode: str, *args: str) -> dict:
        proc = self._run(mode, *args, "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        text = (proc.stdout or "") + (proc.stderr or "")
        start = text.find("{")
        self.assertGreaterEqual(start, 0, msg=text)
        config, _end = json.JSONDecoder().raw_decode(text[start:])
        return config

    def _write_x(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "GS_A\tGENE1\tGENE2\tGENE3",
                    "GS_B\tGENE2\tGENE4",
                    "GS_C\tGENE3\tGENE5",
                    "GS_D\tGENE1\tGENE5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_multi_y(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "Gene\tTrait\tDirect\tCombined\tPrior",
                    "GENE1\tTRAIT_A\t2.5\t2.9\t0.4",
                    "GENE2\tTRAIT_A\t1.8\t2.1\t0.3",
                    "GENE3\tTRAIT_A\t0.9\t1.2\t0.1",
                    "GENE2\tTRAIT_B\t2.3\t2.7\t0.4",
                    "GENE4\tTRAIT_B\t1.7\t2.0\t0.2",
                    "GENE5\tTRAIT_B\t1.1\t1.4\t0.1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_multi_y_reader_prefers_tab_when_free_text_columns_contain_spaces(self) -> None:
        from pegs_shared import phewas as pegs_phewas
        from pigean import multi_y as pigean_multi_y

        path = self.tmpdir / "multi_y_free_text_trait.tsv"
        path.write_text(
            "\n".join(
                [
                    "Trait\tTrait_Internal\tGene\tDirect\tCombined\tIndirect",
                    "Type 2 diabetes\tT2D\tGENE1\t1.5\t2.5\t0.4",
                    "Coronary artery disease\tCAD\tGENE2\t0.7\t1.2\t0.3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        options = SimpleNamespace(
            multi_y_in=str(path),
            multi_y_id_col="Gene",
            multi_y_pheno_col="Trait_Internal",
            multi_y_log_bf_col="Direct",
            multi_y_combined_col="Combined",
            multi_y_prior_col="Indirect",
        )
        columns = pigean_multi_y._resolve_multi_y_columns(options)
        self.assertEqual(columns.id_col_name, "Gene")
        self.assertEqual(columns.pheno_col_name, "Trait_Internal")

        runtime = SimpleNamespace(
            genes=["GENE1", "GENE2"],
            gene_to_ind={"GENE1": 0, "GENE2": 1},
            gene_label_map=None,
            phenos=None,
            pheno_to_ind=None,
            num_gene_phewas_filtered=0,
            X_phewas_beta=None,
            X_phewas_beta_uncorrected=None,
            gene_pheno_Y=None,
            gene_pheno_combined_prior_Ys=None,
            gene_pheno_priors=None,
        )
        phenos, pheno_to_ind, col_info = pegs_phewas.prepare_phewas_phenos_from_file(
            runtime,
            str(path),
            gene_phewas_bfs_id_col=columns.id_col_name,
            gene_phewas_bfs_pheno_col=columns.pheno_col_name,
            gene_phewas_bfs_log_bf_col=columns.log_bf_col_name,
            gene_phewas_bfs_combined_col=columns.combined_col_name,
            gene_phewas_bfs_prior_col=columns.prior_col_name,
            open_text_fn=lambda p: open(p, "rt", encoding="utf-8"),
            warn_fn=lambda _m: None,
        )
        self.assertEqual(phenos, ["T2D", "CAD"])

        y, combined, priors = pegs_phewas.read_phewas_file_batch(
            runtime,
            str(path),
            begin=0,
            cur_batch_size=2,
            pheno_to_ind=pheno_to_ind,
            col_info=col_info,
            open_text_fn=lambda p: open(p, "rt", encoding="utf-8"),
            warn_fn=lambda _m: None,
        )
        np.testing.assert_allclose(y, np.array([[1.5, 0.0], [0.0, 0.7]]))
        np.testing.assert_allclose(combined, np.array([[2.5, 0.0], [0.0, 1.2]]))
        np.testing.assert_allclose(priors, np.array([[0.4, 0.0], [0.0, 0.3]]))

    def test_multi_y_response_defaults_to_combined(self) -> None:
        from pigean import multi_y as pigean_multi_y

        options = SimpleNamespace(multi_y_response_col="combined")
        services = SimpleNamespace(bail=lambda message: (_ for _ in ()).throw(ValueError(message)))
        direct = np.array([[1.0, 2.0], [3.0, 4.0]])
        combined = np.array([[10.0, 20.0], [30.0, 40.0]])
        selected = pigean_multi_y._select_multi_y_response_matrix(direct, combined, options, services)
        np.testing.assert_allclose(selected, combined)

    def test_multi_y_response_can_use_log_bf(self) -> None:
        from pigean import multi_y as pigean_multi_y

        options = SimpleNamespace(multi_y_response_col="log_bf")
        services = SimpleNamespace(bail=lambda message: (_ for _ in ()).throw(ValueError(message)))
        direct = np.array([[1.0, 2.0], [3.0, 4.0]])
        combined = np.array([[10.0, 20.0], [30.0, 40.0]])
        selected = pigean_multi_y._select_multi_y_response_matrix(direct, combined, options, services)
        np.testing.assert_allclose(selected, direct)

    def test_multi_y_response_combined_fails_without_combined_column(self) -> None:
        from pigean import multi_y as pigean_multi_y

        options = SimpleNamespace(multi_y_response_col="combined")
        services = SimpleNamespace(bail=lambda message: (_ for _ in ()).throw(ValueError(message)))
        with self.assertRaisesRegex(ValueError, "--multi-y-response-col combined requires"):
            pigean_multi_y._select_multi_y_response_matrix(
                np.array([[1.0], [2.0]]),
                None,
                options,
                services,
            )

    def test_multi_y_defaults_update_hyper_to_none(self) -> None:
        x_path = self.tmpdir / "multi_y_default_update_hyper.gmt"
        multi_y_path = self.tmpdir / "multi_y_default_update_hyper.tsv"
        self._write_x(x_path)
        self._write_multi_y(multi_y_path)
        config = self._effective_config(
            "betas",
            "--X-in",
            str(x_path),
            "--multi-y-in",
            str(multi_y_path),
            "--gene-universe-from-x",
            "--gene-set-stats-out",
            str(self.tmpdir / "multi_y_default_update_hyper.out"),
        )
        self.assertEqual(config["options"]["update_hyper"], "none")
        self.assertEqual(config["options"]["update_hyper_min_gene_sets"], 1000)

    def test_multi_y_explicit_update_hyper_is_preserved(self) -> None:
        x_path = self.tmpdir / "multi_y_explicit_update_hyper.gmt"
        multi_y_path = self.tmpdir / "multi_y_explicit_update_hyper.tsv"
        self._write_x(x_path)
        self._write_multi_y(multi_y_path)
        config = self._effective_config(
            "betas",
            "--X-in",
            str(x_path),
            "--multi-y-in",
            str(multi_y_path),
            "--gene-universe-from-x",
            "--gene-set-stats-out",
            str(self.tmpdir / "multi_y_explicit_update_hyper.out"),
            "--update-hyper",
            "p",
            "--update-hyper-min-gene-sets",
            "17",
        )
        self.assertEqual(config["options"]["update_hyper"], "p")
        self.assertEqual(config["options"]["update_hyper_min_gene_sets"], 17)

    def test_hyper_update_min_gene_sets_guard_warns_and_keeps_defaults(self) -> None:
        from pigean import x_inputs_core as pigean_x_inputs_core

        runtime = SimpleNamespace(
            p_values=np.array([0.01, 0.02, 0.03, 0.04]),
            gene_set_batches=np.array(["B1", "B1", "B2", "B2"], dtype=object),
            p=0.001,
            sigma2=0.002,
            ps=None,
            sigma2s=None,
        )
        warnings: list[str] = []
        pigean_x_inputs_core.maybe_learn_batch_hyper_after_x_read_for_runtime(
            runtime,
            skip_betas=False,
            update_hyper_p=True,
            update_hyper_sigma=False,
            batches=["B1", "B1", "B2", "B2"],
            num_ignored_gene_sets=[0, 0, 0, 0],
            first_for_hyper=False,
            max_num_gene_sets_hyper=None,
            update_hyper_min_gene_sets=1000,
            first_for_sigma_cond=False,
            fixed_sigma_cond=False,
            first_max_p_for_hyper=False,
            max_num_burn_in=5,
            max_num_iter_betas=10,
            min_num_iter_betas=5,
            num_chains_betas=2,
            r_threshold_burn_in_betas=1.01,
            use_max_r_for_convergence_betas=True,
            max_frac_sem_betas=0.01,
            max_allowed_batch_correlation=None,
            sigma_num_devs_to_top=2.0,
            p_noninf_inflate=1.0,
            sparse_solution=False,
            sparse_frac_betas=None,
            betas_trace_out=None,
            log_fn=lambda *_args, **_kwargs: None,
            warn_fn=warnings.append,
            debug_level=1,
        )
        self.assertTrue(any("Skipping hyperparameter update for batch B1" in message for message in warnings))
        self.assertTrue(any("Skipping hyperparameter update for batch B2" in message for message in warnings))
        self.assertTrue(any("No batches met --update-hyper-min-gene-sets=1000" in message for message in warnings))
        np.testing.assert_allclose(runtime.ps, np.full(4, 0.001))
        np.testing.assert_allclose(runtime.sigma2s, np.full(4, 0.002))

    def _common_args(self, x_path: Path, multi_y_path: Path) -> list[str]:
        return [
            "--X-in",
            str(x_path),
            "--multi-y-in",
            str(multi_y_path),
            "--gene-universe-from-x",
            "--hide-opts",
            "--deterministic",
            "--min-gene-set-size",
            "1",
            "--filter-gene-set-p",
            "1",
            "--max-gene-set-read-p",
            "1",
            "--max-num-gene-sets-initial",
            "10",
            "--max-num-gene-sets-hyper",
            "10",
            "--max-num-gene-sets",
            "10",
            "--max-num-burn-in",
            "5",
            "--max-num-iter-betas",
            "15",
            "--min-num-iter-betas",
            "5",
            "--num-chains-betas",
            "2",
        ]

    def test_multi_y_betas_appends_trait_column(self) -> None:
        x_path = self.tmpdir / "multi_y_betas.gmt"
        multi_y_path = self.tmpdir / "multi_y_betas.tsv"
        out_path = self.tmpdir / "multi_y_betas.gene_set_stats.out"
        self._write_x(x_path)
        self._write_multi_y(multi_y_path)

        proc = self._run(
            "betas",
            *self._common_args(x_path, multi_y_path),
            "--gene-set-stats-out",
            str(out_path),
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        self.assertTrue(out_path.exists())

        with out_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            self.assertIn("trait", reader.fieldnames)
            rows = list(reader)
        self.assertGreater(len(rows), 0)
        self.assertEqual({row["trait"] for row in rows}, {"TRAIT_A", "TRAIT_B"})

    def test_multi_y_trait_blacklist_filters_before_batching(self) -> None:
        x_path = self.tmpdir / "multi_y_blacklist.gmt"
        multi_y_path = self.tmpdir / "multi_y_blacklist.tsv"
        blacklist_path = self.tmpdir / "multi_y_blacklist.traits.txt"
        out_path = self.tmpdir / "multi_y_blacklist.gene_set_stats.out"
        params_path = self.tmpdir / "multi_y_blacklist.params.out"
        self._write_x(x_path)
        self._write_multi_y(multi_y_path)
        blacklist_path.write_text("TRAIT_B\nMISSING_TRAIT\n", encoding="utf-8")

        proc = self._run(
            "betas",
            *self._common_args(x_path, multi_y_path),
            "--multi-y-trait-blacklist-in",
            str(blacklist_path),
            "--multi-y-max-phenos-per-batch",
            "2",
            "--gene-set-stats-out",
            str(out_path),
            "--params-out",
            str(params_path),
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))

        with out_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        self.assertGreater(len(rows), 0)
        self.assertEqual({row["trait"] for row in rows}, {"TRAIT_A"})

        params_text = params_path.read_text(encoding="utf-8")
        self.assertIn("multi_y_num_traits_before_blacklist\t1\t2", params_text)
        self.assertIn("multi_y_num_traits\t1\t1", params_text)
        self.assertIn("multi_y_trait_blacklist_requested\t1\t2", params_text)
        self.assertIn("multi_y_trait_blacklist_matched\t1\t1", params_text)
        self.assertIn("multi_y_trait_blacklist_missing\t1\t1", params_text)

    def test_multi_y_vectorized_betas_appends_trait_column_and_records_params(self) -> None:
        x_path = self.tmpdir / "multi_y_vectorized_betas.gmt"
        multi_y_path = self.tmpdir / "multi_y_vectorized_betas.tsv"
        unvectorized_out_path = self.tmpdir / "multi_y_vectorized_betas.unvectorized.gene_set_stats.out"
        out_path = self.tmpdir / "multi_y_vectorized_betas.gene_set_stats.out"
        params_path = self.tmpdir / "multi_y_vectorized_betas.params.out"
        self._write_x(x_path)
        self._write_multi_y(multi_y_path)

        unvectorized_proc = self._run(
            "betas",
            *self._common_args(x_path, multi_y_path),
            "--multi-y-max-phenos-per-batch",
            "2",
            "--no-filter-negative",
            "--prune-gene-sets",
            "1.1",
            "--weighted-prune-gene-sets",
            "1.1",
            "--output-detail",
            "full",
            "--gene-set-stats-out",
            str(unvectorized_out_path),
        )
        self.assertEqual(
            unvectorized_proc.returncode,
            0,
            msg=(unvectorized_proc.stderr or "") + (unvectorized_proc.stdout or ""),
        )

        proc = self._run(
            "betas",
            *self._common_args(x_path, multi_y_path),
            "--multi-y-vectorize-betas",
            "--multi-y-max-phenos-per-batch",
            "2",
            "--no-filter-negative",
            "--prune-gene-sets",
            "1.1",
            "--weighted-prune-gene-sets",
            "1.1",
            "--output-detail",
            "full",
            "--gene-set-stats-out",
            str(out_path),
            "--params-out",
            str(params_path),
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        self.assertTrue(out_path.exists())

        with out_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            self.assertIn("trait", reader.fieldnames)
            rows = list(reader)
        self.assertGreater(len(rows), 0)
        self.assertEqual({row["trait"] for row in rows}, {"TRAIT_A", "TRAIT_B"})

        with unvectorized_out_path.open(encoding="utf-8") as fh:
            unvectorized_rows = list(csv.DictReader(fh, delimiter="\t"))
        unvectorized_by_key = {
            (row["trait"], row["Gene_Set"]): row
            for row in unvectorized_rows
        }
        for row in rows:
            key = (row["trait"], row["Gene_Set"])
            self.assertIn(key, unvectorized_by_key)
            self.assertAlmostEqual(
                float(row["beta_tilde_internal"]),
                float(unvectorized_by_key[key]["beta_tilde_internal"]),
                places=10,
            )

        params_text = params_path.read_text(encoding="utf-8")
        self.assertIn("multi_y_vectorize_betas\t1\tTrue", params_text)
        self.assertIn("multi_y_vectorized_beta_parallel_axis\t1\ttraits", params_text)
        self.assertIn("multi_y_num_traits_completed\t1\t2", params_text)

    def test_multi_y_gibbs_aggregates_gene_and_gene_set_outputs(self) -> None:
        from pigean import multi_y as pigean_multi_y  # imported lazily after PYTHONPATH setup
        from pigean import dispatch as pigean_dispatch

        class _StubState:
            def __init__(self) -> None:
                self.genes = ["GENE1", "GENE2", "GENE3"]
                self.gene_to_ind = {gene: i for i, gene in enumerate(self.genes)}
                self.params = {}
                self.param_keys = []

            def has_gene_sets(self) -> bool:
                return True

            def _record_params(self, params, overwrite=False, record_only_first_time=False):
                del record_only_first_time
                for key, value in params.items():
                    if value is None:
                        continue
                    if overwrite or key not in self.params:
                        self.params[key] = value
                        if key not in self.param_keys:
                            self.param_keys.append(key)

            def write_params(self, output_file):
                with open(output_file, "w", encoding="utf-8") as fh:
                    fh.write("Parameter\tVersion\tValue\n")
                    for key in self.param_keys:
                        fh.write(f"{key}\t1\t{self.params[key]}\n")

        def _fake_inner_run(trait_options, mode, services=None):
            del mode, services
            trait_name = Path(trait_options.gene_stats_in).stem.split("_", 1)[1].split(".")[0]
            self.assertTrue(trait_options.gene_universe_from_x)
            self.assertFalse(trait_options.gene_universe_from_y)
            with open(trait_options.gene_set_stats_out, "w", encoding="utf-8") as fh:
                fh.write("Gene_Set\tbeta_tilde\tP\n")
                fh.write(f"GS_{trait_name}\t1.5\t0.01\n")
            with open(trait_options.gene_stats_out, "w", encoding="utf-8") as fh:
                fh.write("Gene\tprior\tcombined\tlog_bf\n")
                fh.write(f"GENE1\t0.2\t0.3\t1.0\n")
            return None

        options = SimpleNamespace(
            multi_y_in=str(self.tmpdir / "stub_multi_y.tsv"),
            multi_y_id_col=None,
            multi_y_pheno_col="Trait",
            multi_y_log_bf_col="Direct",
            multi_y_combined_col="Combined",
            multi_y_prior_col="Prior",
            multi_y_max_phenos_per_batch=1,
            gene_set_stats_out=str(self.tmpdir / "stub_multi_y.gene_set_stats.out"),
            gene_stats_out=str(self.tmpdir / "stub_multi_y.gene_stats.out"),
            params_out=str(self.tmpdir / "stub_multi_y.params.out"),
            max_gb=2.0,
            gwas_in=None,
            huge_statistics_in=None,
            huge_statistics_out=None,
            exomes_in=None,
            case_counts_in=None,
            ctrl_counts_in=None,
            gene_stats_in=None,
            gene_set_stats_in=None,
            gene_set_betas_in=None,
            const_gene_set_beta=None,
            const_gene_Y=None,
            positive_controls_in=None,
            positive_controls_list=None,
            positive_controls_all_in=None,
            gene_phewas_bfs_in=None,
            gene_universe_in=None,
            gene_universe_id_col=None,
            gene_universe_has_header=True,
            gene_universe_from_y=False,
            gene_universe_from_x=False,
            run_phewas_from_gene_phewas_stats_in=None,
            phewas_stats_out=None,
            phewas_gene_set_stats_out=None,
        )
        Path(options.multi_y_in).write_text(
            "\n".join(
                [
                    "Gene\tTrait\tDirect\tCombined\tPrior",
                    "GENE1\tTRAIT_A\t1.0\t1.2\t0.1",
                    "GENE2\tTRAIT_B\t1.5\t1.7\t0.2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        services = SimpleNamespace(
            INFO=1,
            DEBUG=2,
            sys=SimpleNamespace(exit=lambda code=0: (_ for _ in ()).throw(SystemExit(code))),
            log=lambda *args, **kwargs: None,
            warn=lambda *args, **kwargs: None,
            bail=lambda message: (_ for _ in ()).throw(AssertionError(message)),
        )

        with mock.patch.object(pigean_multi_y.pigean_main_support, "build_runtime_state", return_value=_StubState()), \
            mock.patch.object(pigean_multi_y.pigean_main_support, "configure_hyperparameters_for_main", return_value=None), \
            mock.patch.object(pigean_multi_y.pigean_main_support, "run_main_adaptive_read_x", return_value=None), \
            mock.patch.object(
                pigean_multi_y.pigean_phewas,
                "prepare_phewas_phenos_from_file",
                return_value=(["TRAIT_A", "TRAIT_B"], {"TRAIT_A": 0, "TRAIT_B": 1}, {"id_col": 0, "pheno_col": 1, "bf_col": 2, "combined_col": 3, "prior_col": 4}),
            ), \
            mock.patch.object(
                pigean_multi_y.pigean_phewas,
                "read_phewas_file_batch",
                return_value=(
                    np.array([[1.0], [0.0], [0.0]]),
                    np.array([[1.2], [0.0], [0.0]]),
                    np.array([[0.1], [0.0], [0.0]]),
                ),
            ), \
            mock.patch.object(pigean_dispatch, "run_main_pipeline", side_effect=_fake_inner_run):
            result = pigean_multi_y.run_multi_y_pipeline(services=services, options=options, mode="gibbs")

        self.assertEqual(result.num_traits_total, 2)
        self.assertEqual(result.num_traits_completed, 2)
        with open(options.gene_set_stats_out, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            self.assertIn("trait", reader.fieldnames)
            rows = list(reader)
        self.assertEqual({row["trait"] for row in rows}, {"TRAIT_A", "TRAIT_B"})
        with open(options.gene_stats_out, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            self.assertIn("trait", reader.fieldnames)
            rows = list(reader)
        self.assertEqual({row["trait"] for row in rows}, {"TRAIT_A", "TRAIT_B"})
        params_text = Path(options.params_out).read_text(encoding="utf-8")
        self.assertIn("multi_y_enabled", params_text)
        self.assertIn("multi_y_num_traits\t1\t2", params_text)
        self.assertIn("multi_y_phenos_per_batch\t1\t1", params_text)
        self.assertIn("multi_y_gene_universe_mode\t1\tx", params_text)

    def test_multi_y_explicit_gene_universe_is_preserved_for_trait_runs(self) -> None:
        from pigean import multi_y as pigean_multi_y  # imported lazily after PYTHONPATH setup
        from pigean import dispatch as pigean_dispatch

        class _StubState:
            def __init__(self) -> None:
                self.genes = ["GENE1", "GENE2", "GENE3"]
                self.gene_to_ind = {gene: i for i, gene in enumerate(self.genes)}
                self.gene_label_map = None
                self.params = {}
                self.param_keys = []

            def has_gene_sets(self) -> bool:
                return True

            def _record_params(self, params, overwrite=False, record_only_first_time=False):
                del record_only_first_time
                for key, value in params.items():
                    if value is None:
                        continue
                    if overwrite or key not in self.params:
                        self.params[key] = value
                        if key not in self.param_keys:
                            self.param_keys.append(key)

            def write_params(self, output_file):
                with open(output_file, "w", encoding="utf-8") as fh:
                    fh.write("Parameter\tVersion\tValue\n")
                    for key in self.param_keys:
                        fh.write(f"{key}\t1\t{self.params[key]}\n")

        trait_gene_universe_modes = []

        def _fake_inner_run(trait_options, mode, services=None):
            del mode, services
            trait_gene_universe_modes.append(
                (
                    trait_options.gene_universe_in,
                    trait_options.gene_universe_from_x,
                    trait_options.gene_universe_from_y,
                )
            )
            with open(trait_options.gene_set_stats_out, "w", encoding="utf-8") as fh:
                fh.write("Gene_Set\tbeta_tilde\tP\n")
                fh.write("GS_A\t1.5\t0.01\n")
            return None

        options = SimpleNamespace(
            multi_y_in=str(self.tmpdir / "stub_multi_y_explicit_universe.tsv"),
            multi_y_id_col=None,
            multi_y_pheno_col="Trait",
            multi_y_log_bf_col="Direct",
            multi_y_combined_col="Combined",
            multi_y_prior_col="Prior",
            multi_y_max_phenos_per_batch=2,
            gene_set_stats_out=str(self.tmpdir / "stub_multi_y_explicit_universe.gene_set_stats.out"),
            gene_stats_out=None,
            params_out=str(self.tmpdir / "stub_multi_y_explicit_universe.params.out"),
            max_gb=2.0,
            gwas_in=None,
            huge_statistics_in=None,
            huge_statistics_out=None,
            exomes_in=None,
            case_counts_in=None,
            ctrl_counts_in=None,
            gene_stats_in=None,
            gene_set_stats_in=None,
            gene_set_betas_in=None,
            const_gene_set_beta=None,
            const_gene_Y=None,
            positive_controls_in=None,
            positive_controls_list=None,
            positive_controls_all_in=None,
            gene_phewas_bfs_in=None,
            gene_universe_in=str(self.tmpdir / "explicit_universe.tsv"),
            gene_universe_id_col="Gene",
            gene_universe_has_header=True,
            gene_universe_from_y=False,
            gene_universe_from_x=False,
            run_phewas_from_gene_phewas_stats_in=None,
            phewas_stats_out=None,
            phewas_gene_set_stats_out=None,
        )
        Path(options.multi_y_in).write_text(
            "\n".join(
                [
                    "Gene\tTrait\tDirect\tCombined\tPrior",
                    "GENE1\tTRAIT_A\t1.0\t1.2\t0.1",
                    "GENE2\tTRAIT_B\t1.5\t1.7\t0.2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        Path(options.gene_universe_in).write_text("Gene\nGENE1\nGENE2\n", encoding="utf-8")
        services = SimpleNamespace(
            INFO=1,
            DEBUG=2,
            sys=SimpleNamespace(exit=lambda code=0: (_ for _ in ()).throw(SystemExit(code))),
            log=lambda *args, **kwargs: None,
            warn=lambda *args, **kwargs: None,
            bail=lambda message: (_ for _ in ()).throw(AssertionError(message)),
        )

        with mock.patch.object(pigean_multi_y.pigean_main_support, "build_runtime_state", return_value=_StubState()), \
            mock.patch.object(pigean_multi_y.pigean_main_support, "configure_hyperparameters_for_main", return_value=None), \
            mock.patch.object(pigean_multi_y.pigean_main_support, "run_main_adaptive_read_x", return_value=None), \
            mock.patch.object(
                pigean_multi_y.pigean_main_support.pigean_y_inputs_core,
                "initialize_explicit_gene_universe_if_needed",
                return_value=None,
            ) as init_universe, \
            mock.patch.object(
                pigean_multi_y.pigean_phewas,
                "prepare_phewas_phenos_from_file",
                return_value=(["TRAIT_A", "TRAIT_B"], {"TRAIT_A": 0, "TRAIT_B": 1}, {"id_col": 0, "pheno_col": 1, "bf_col": 2, "combined_col": 3, "prior_col": 4}),
            ), \
            mock.patch.object(
                pigean_multi_y.pigean_phewas,
                "read_phewas_file_batch",
                return_value=(
                    np.array([[1.0, 0.0], [0.0, 1.5], [0.0, 0.0]]),
                    np.array([[1.2, 0.0], [0.0, 1.7], [0.0, 0.0]]),
                    np.array([[0.1, 0.0], [0.0, 0.2], [0.0, 0.0]]),
                ),
            ), \
            mock.patch.object(pigean_dispatch, "run_main_pipeline", side_effect=_fake_inner_run):
            result = pigean_multi_y.run_multi_y_pipeline(services=services, options=options, mode="betas")

        self.assertEqual(result.num_traits_completed, 2)
        init_universe.assert_called_once()
        self.assertEqual(
            trait_gene_universe_modes,
            [
                (options.gene_universe_in, False, False),
                (options.gene_universe_in, False, False),
            ],
        )
        params_text = Path(options.params_out).read_text(encoding="utf-8")
        self.assertIn("multi_y_gene_universe_mode\t1\tfile", params_text)


if __name__ == "__main__":
    unittest.main()
