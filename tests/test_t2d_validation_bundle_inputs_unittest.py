from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class T2DValidationBundleInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.fixture_root = cls.repo_root / "tests" / "data" / "t2d_smoke"
        cls.model_data = cls.repo_root / "tests" / "data" / "model_small"
        required = [
            cls.fixture_root / "T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz",
            cls.model_data / "gene_set_list_mouse_2024.txt",
            cls.model_data / "portal_gencode.gene.map",
            cls.model_data / "NCBI37.3.plink.gene.loc",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest("Missing bundled T2D validation fixtures: " + ", ".join(missing))

        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)
        cls.output_prefix = cls.tmpdir / "t2d_validation"
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
            str(cls.model_data / "gene_set_list_mouse_2024.txt"),
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

    def test_validation_outputs_survive_without_qc_override(self) -> None:
        gene_stats = self._load_rows(self.output_prefix.with_suffix(".gene_stats.out"), "Gene")
        gene_set_stats = self._load_rows(self.output_prefix.with_suffix(".gene_set_stats.out"), "Gene_Set")
        self.assertGreaterEqual(len(gene_stats), 10000)
        self.assertGreaterEqual(len(gene_set_stats), 300)
        self.assertIn("HNF1A", gene_stats)
        self.assertGreater(float(gene_stats["HNF1A"]["huge_score_gwas_uncorrected"]), 0.0)

    def test_validation_run_kept_gene_sets_without_qc_override(self) -> None:
        text = self.combined_output
        self.assertIn("Reading X 1 of 1", text)
        self.assertIn("Read 419 gene sets", text)
        self.assertNotIn("No gene sets survived the input filters", text)
