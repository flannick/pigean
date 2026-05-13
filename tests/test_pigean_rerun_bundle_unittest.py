from __future__ import annotations

import csv
import gzip
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np


class PigeanRerunBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.model_data = cls.repo_root / "tests/data/model_small"
        cls.gene_stats = cls.repo_root / "tests/data/mody_priors_gene_stats.tsv"
        required = [
            cls.model_data / "gene_set_list_mouse_2024.txt",
            cls.model_data / "portal_gencode.gene.map",
            cls.model_data / "NCBI37.3.plink.gene.loc",
            cls.gene_stats,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest("Missing rerun bundle fixtures: " + ", ".join(missing))
        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    def _base_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        src_root = str(self.repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return env

    def _run(self, mode: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "pigean", mode, *args],
            cwd=self.repo_root,
            env=self._base_env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def _common_args(self) -> list[str]:
        return [
            "--X-in",
            str(self.model_data / "gene_set_list_mouse_2024.txt"),
            "--gene-map-in",
            str(self.model_data / "portal_gencode.gene.map"),
            "--gene-stats-in",
            str(self.gene_stats),
            "--gene-stats-id-col",
            "GENE",
            "--gene-stats-log-bf-col",
            "log_bf",
            "--gene-stats-combined-col",
            "combined",
            "--gene-stats-prior-col",
            "prior",
            "--gene-universe-in",
            str(self.model_data / "NCBI37.3.plink.gene.loc"),
            "--gene-universe-id-col",
            "6",
            "--gene-universe-no-header",
            "--hide-opts",
            "--hide-progress",
            "--deterministic",
            "--min-gene-set-size",
            "1",
            "--filter-gene-set-p",
            "1",
            "--max-gene-set-read-p",
            "1",
            "--no-filter-negative",
            "--max-num-gene-sets-initial",
            "50",
            "--max-num-gene-sets-hyper",
            "50",
            "--max-num-gene-sets",
            "50",
            "--max-num-iter-betas",
            "10",
            "--min-num-iter-betas",
            "3",
            "--num-chains-betas",
            "2",
        ]

    def _read_gene_set_stats(self, path: Path) -> dict[str, dict[str, str]]:
        with gzip.open(path, "rt") as fh:
            return {row["Gene_Set"]: row for row in csv.DictReader(fh, delimiter="\t")}

    def test_betas_rerun_bundle_roundtrip_exclude_and_effective_config(self) -> None:
        original_gene_set_stats = self.tmpdir / "original.gene_set_stats.tsv.gz"
        original_gene_stats = self.tmpdir / "original.gene_stats.tsv.gz"
        original_params = self.tmpdir / "original.params.tsv"
        bundle = self.tmpdir / "original.rerun_bundle.tar.gz"
        proc = self._run(
            "betas",
            *self._common_args(),
            "--gene-set-stats-out",
            str(original_gene_set_stats),
            "--gene-stats-out",
            str(original_gene_stats),
            "--params-out",
            str(original_params),
            "--pigean-rerun-bundle-out",
            str(bundle),
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        self.assertTrue(bundle.exists())

        with tarfile.open(bundle, "r:*") as tar_fh:
            members = set(tar_fh.getnames())
            self.assertIn("manifest.json", members)
            self.assertIn("X.tsv.gz", members)
            self.assertIn("gene_stats.tsv.gz", members)
            self.assertIn("gene_universe.tsv.gz", members)
            self.assertIn("params.tsv.gz", members)
            manifest = json.load(tar_fh.extractfile("manifest.json"))
        self.assertEqual(manifest["schema"], "pigean_rerun_bundle/v1")
        self.assertEqual(manifest["column_mapping"]["gene_stats_combined_col"], "combined")
        self.assertEqual(manifest["rerun_defaults"]["update_hyper"], "none")

        cfg = self._run(
            "betas",
            "--pigean-rerun-bundle-in",
            str(bundle),
            "--sigma2",
            "0.123",
            "--print-effective-config",
        )
        self.assertEqual(cfg.returncode, 0, msg=(cfg.stderr or "") + (cfg.stdout or ""))
        payload = json.loads(cfg.stdout)
        options = payload["options"]
        self.assertEqual(payload["mode"], "betas")
        self.assertTrue(options["X_in"][0].endswith("X.tsv.gz"))
        self.assertTrue(options["gene_stats_in"].endswith("gene_stats.tsv.gz"))
        self.assertTrue(options["gene_universe_in"].endswith("gene_universe.tsv.gz"))
        self.assertEqual(options["gene_stats_id_col"], "Gene")
        self.assertEqual(options["gene_stats_combined_col"], "combined")
        self.assertEqual(options["update_hyper"], "none")
        self.assertEqual(options["sigma2"], 0.123)

        rerun_gene_set_stats = self.tmpdir / "rerun.gene_set_stats.tsv.gz"
        rerun_eaggl_bundle = self.tmpdir / "rerun.eaggl_bundle.tar.gz"
        rerun = self._run(
            "betas",
            "--pigean-rerun-bundle-in",
            str(bundle),
            "--hide-opts",
            "--hide-progress",
            "--gene-set-stats-out",
            str(rerun_gene_set_stats),
            "--eaggl-bundle-out",
            str(rerun_eaggl_bundle),
        )
        self.assertEqual(rerun.returncode, 0, msg=(rerun.stderr or "") + (rerun.stdout or ""))
        self.assertTrue(rerun_eaggl_bundle.exists())

        original = self._read_gene_set_stats(original_gene_set_stats)
        repeated = self._read_gene_set_stats(rerun_gene_set_stats)
        common = sorted(set(original).intersection(repeated))
        self.assertGreaterEqual(len(common), 10)
        for column, min_corr in (("beta_uncorrected", 0.999), ("beta", 0.98)):
            xs = []
            ys = []
            for gene_set in common:
                x = float(original[gene_set][column])
                y = float(repeated[gene_set][column])
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x)
                    ys.append(y)
            self.assertGreaterEqual(float(np.corrcoef(xs, ys)[0, 1]), min_corr)

        exclude_id = common[0]
        exclude_file = self.tmpdir / "exclude.txt"
        exclude_file.write_text(f"{exclude_id}\nmissing_annotation_id\n", encoding="utf-8")
        excluded_gene_set_stats = self.tmpdir / "excluded.gene_set_stats.tsv.gz"
        excluded_params = self.tmpdir / "excluded.params.tsv"
        excluded = self._run(
            "betas",
            "--pigean-rerun-bundle-in",
            str(bundle),
            "--gene-set-exclude-in",
            str(exclude_file),
            "--hide-opts",
            "--hide-progress",
            "--gene-set-stats-out",
            str(excluded_gene_set_stats),
            "--params-out",
            str(excluded_params),
        )
        self.assertEqual(excluded.returncode, 0, msg=(excluded.stderr or "") + (excluded.stdout or ""))
        excluded_rows = self._read_gene_set_stats(excluded_gene_set_stats)
        self.assertNotIn(exclude_id, excluded_rows)
        params_text = excluded_params.read_text(encoding="utf-8")
        self.assertIn("gene_set_exclude_requested_count\t1\t2", params_text)
        self.assertIn("gene_set_exclude_found_count\t1\t1", params_text)
        self.assertIn("gene_set_exclude_not_found_count\t1\t1", params_text)

    def test_rerun_bundle_in_rejects_gibbs_mode(self) -> None:
        bundle = self.tmpdir / "reject.rerun_bundle.tar.gz"
        proc = self._run(
            "betas",
            *self._common_args(),
            "--pigean-rerun-bundle-out",
            str(bundle),
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        bad = self._run("gibbs", "--pigean-rerun-bundle-in", str(bundle), "--print-effective-config")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("requires mode 'betas'", (bad.stderr or "") + (bad.stdout or ""))

    def test_params_in_replays_fixed_beta_hyperparameters(self) -> None:
        params_in = self.tmpdir / "replay.params.tsv"
        params_in.write_text(
            "\n".join(
                [
                    "Parameter\tVersion\tValue",
                    "p\t1\t0.02",
                    "sigma2\t1\t0.004",
                    "sigma_power\t1\t-2",
                    "option_filter_gene_set_p\t1\t1",
                    "option_max_num_gene_sets\t1\t50",
                    "option_max_num_gene_sets_hyper\t1\t50",
                    "option_min_gene_set_size\t1\t1",
                    "option_update_hyper\t1\tp",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        cfg = self._run(
            "betas",
            "--pigean-params-in",
            str(params_in),
            "--print-effective-config",
        )
        self.assertEqual(cfg.returncode, 0, msg=(cfg.stderr or "") + (cfg.stdout or ""))
        payload = json.loads(cfg.stdout)
        options = payload["options"]
        self.assertEqual(options["p_noninf"], [0.02])
        self.assertEqual(options["sigma2"], 0.004)
        self.assertEqual(options["sigma_power"], -2.0)
        self.assertEqual(options["filter_gene_set_p"], 1)
        self.assertEqual(options["max_num_gene_sets"], 50)
        self.assertEqual(options["update_hyper"], "none")

        gene_set_stats = self.tmpdir / "params_replay.gene_set_stats.tsv.gz"
        params_out = self.tmpdir / "params_replay.params.tsv"
        run = self._run(
            "betas",
            *self._common_args(),
            "--pigean-params-in",
            str(params_in),
            "--gene-set-stats-out",
            str(gene_set_stats),
            "--params-out",
            str(params_out),
        )
        self.assertEqual(run.returncode, 0, msg=(run.stderr or "") + (run.stdout or ""))
        params_text = params_out.read_text(encoding="utf-8")
        self.assertIn("pigean_params_replay_p_values_applied", params_text)
        self.assertIn("pigean_params_replay_sigma2_values_applied", params_text)


if __name__ == "__main__":
    unittest.main()
