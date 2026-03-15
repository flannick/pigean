from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class T2DToyBundleInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.fixture_root = cls.repo_root / "tests" / "data" / "t2d_smoke"
        cls.model_data = cls.repo_root / "tests" / "data" / "model_small"
        required = [
            cls.fixture_root / "T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz",
            cls.fixture_root / "T2D.exomes.p_lt_1e-4_or_mody.tsv",
            cls.fixture_root / "mody.gene.list",
            cls.fixture_root / "mody_case_counts.tsv",
            cls.fixture_root / "mody_ctrl_counts.tsv",
            cls.fixture_root / "gene_set_list_mouse_t2d_toy.txt",
            cls.model_data / "portal_gencode.gene.map",
            cls.model_data / "NCBI37.3.plink.gene.loc",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest("Missing bundled T2D toy fixtures: " + ", ".join(missing))

        cls.positive_controls_csv = ",".join(
            gene.strip()
            for gene in (cls.fixture_root / "mody.gene.list").read_text(encoding="utf-8").splitlines()
            if gene.strip()
        )

        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)
        cls.output_prefix = cls.tmpdir / "t2d_toy"
        cls.proc = cls._run_beta_tildes()
        cls.combined_output = (cls.proc.stdout or "") + (cls.proc.stderr or "")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    @classmethod
    def _run_beta_tildes(cls) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            "-m",
            "pigean",
            "beta_tildes",
            "--deterministic",
            "--hide-opts",
            "--X-in",
            str(cls.fixture_root / "gene_set_list_mouse_t2d_toy.txt"),
            "--gene-map-in",
            str(cls.model_data / "portal_gencode.gene.map"),
            "--gene-loc-file",
            str(cls.model_data / "NCBI37.3.plink.gene.loc"),
            "--gene-loc-file-huge",
            str(cls.model_data / "NCBI37.3.plink.gene.loc"),
            "--gwas-in",
            str(cls.fixture_root / "T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz"),
            "--gwas-chrom-col",
            "CHROM",
            "--gwas-pos-col",
            "POS",
            "--gwas-p-col",
            "P",
            "--gwas-n-col",
            "N",
            "--exomes-in",
            str(cls.fixture_root / "T2D.exomes.p_lt_1e-4_or_mody.tsv"),
            "--exomes-gene-col",
            "GeneSymbol",
            "--exomes-p-col",
            "P-value",
            "--exomes-beta-col",
            "beta",
            "--exomes-se-col",
            "se",
            "--positive-controls-list",
            cls.positive_controls_csv,
            "--positive-controls-all-in",
            str(cls.model_data / "NCBI37.3.plink.gene.loc"),
            "--positive-controls-all-id-col",
            "6",
            "--positive-controls-all-no-header",
            "--case-counts-in",
            str(cls.fixture_root / "mody_case_counts.tsv"),
            "--ctrl-counts-in",
            str(cls.fixture_root / "mody_ctrl_counts.tsv"),
            "--case-counts-max-freq-col",
            "max_freq",
            "--ctrl-counts-max-freq-col",
            "max_freq",
            "--min-gene-set-size",
            "1",
            "--filter-gene-set-metric-z",
            "0",
            "--gene-stats-out",
            str(cls.output_prefix.with_suffix(".gene_stats.out")),
            "--gene-set-stats-out",
            str(cls.output_prefix.with_suffix(".gene_set_stats.out")),
            "--params-out",
            str(cls.output_prefix.with_suffix(".params.out")),
        ]
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        src_root = str(cls.repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        proc = subprocess.run(cmd, cwd=cls.repo_root, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return proc

    @staticmethod
    def _load_rows(path: Path, key_col: str) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        with path.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                out[row[key_col]] = row
        return out

    @staticmethod
    def _load_params(path: Path) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        with path.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                out.setdefault(row["Parameter"], []).append(row["Value"])
        return out

    def test_logs_show_all_toy_fixture_inputs_are_read(self) -> None:
        text = self.combined_output
        self.assertIn("Reading --exomes-in file", text)
        self.assertIn("Reading case counts from", text)
        self.assertIn("Reading ctrl counts from", text)
        self.assertIn("Reading --gwas-in file", text)

    def test_toy_outputs_are_nonempty_and_include_expected_t2d_genes(self) -> None:
        gene_stats = self._load_rows(self.output_prefix.with_suffix(".gene_stats.out"), "Gene")
        gene_set_stats = self._load_rows(self.output_prefix.with_suffix(".gene_set_stats.out"), "Gene_Set")
        self.assertGreaterEqual(len(gene_stats), 1000)
        self.assertGreaterEqual(len(gene_set_stats), 5)
        self.assertIn("HNF1A", gene_stats)
        self.assertIn("GCK", gene_stats)
        self.assertIn("INS", gene_stats)
        self.assertGreater(float(gene_stats["HNF1A"]["positive_control"]), 0.0)
        self.assertGreater(float(gene_stats["GCK"]["huge_score_exomes"]), 0.0)
        self.assertGreater(float(gene_stats["HNF1A"]["huge_score_gwas_uncorrected"]), 0.0)

    def test_params_out_records_resolved_runtime_options(self) -> None:
        params = self._load_params(self.output_prefix.with_suffix(".params.out"))
        self.assertEqual(params["option_mode"][-1], "beta_tildes")
        self.assertEqual(params["runtime_y_not_loaded"][-1], "False")
        self.assertEqual(params["runtime_run_beta_tilde"][-1], "True")
        self.assertEqual(params["option_gwas_n_col"][-1], "N")
        self.assertTrue(params["option_gwas_in"][-1].endswith("T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz"))
        self.assertIn("option_max_gb", params)
        self.assertIn("option_batch_size", params)
        self.assertIn("option_max_read_entries_at_once", params)
        self.assertIn("option_update_hyper_p", params)
        self.assertIn("option_update_hyper_sigma", params)
        positive_controls = json.loads(params["option_positive_controls_list"][-1])
        self.assertIn("HNF1A", positive_controls)
        mean_rrs = json.loads(params["option_counts_mean_rrs"][-1])
        self.assertEqual(mean_rrs, [1.3, 1.6, 2.5, 3.8])
