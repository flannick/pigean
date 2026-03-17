from __future__ import annotations

import csv
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

    def _common_args(self, x_path: Path, multi_y_path: Path) -> list[str]:
        return [
            "--X-in",
            str(x_path),
            "--multi-y-in",
            str(multi_y_path),
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
            run_phewas_from_gene_phewas_stats_in=None,
            betas_from_phewas=False,
            betas_uncorrected_from_phewas=False,
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


if __name__ == "__main__":
    unittest.main()
