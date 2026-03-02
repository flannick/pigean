from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SetBSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.model_data = cls.repo_root / "tests/data/model_small"
        cls.reference_root = cls.repo_root / "tests/data/reference"
        cls.gene_stats = cls.repo_root / "tests/data/mody_priors_gene_stats.tsv"

        required = [
            cls.model_data / "gene_set_list_mouse_2024.txt",
            cls.model_data / "portal_gencode.gene.map",
            cls.gene_stats,
            cls.reference_root / "mody_beta_tildes_gene_set_beta_tilde.tsv",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise unittest.SkipTest("Missing Set B smoke fixtures: " + ", ".join(missing))

        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    def _run(self, mode: str, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, "src/pigean.py", mode, *args]
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        return subprocess.run(
            cmd,
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    @classmethod
    def _common_x_args(cls) -> list[str]:
        return [
            "--X-in",
            str(cls.model_data / "gene_set_list_mouse_2024.txt"),
            "--gene-map-in",
            str(cls.model_data / "portal_gencode.gene.map"),
            "--hide-opts",
            "--deterministic",
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
            "200",
            "--max-num-burn-in",
            "5",
            "--max-num-iter-betas",
            "20",
            "--min-num-iter-betas",
            "5",
            "--num-chains-betas",
            "2",
        ]

    @classmethod
    def _common_gene_stats_args(cls) -> list[str]:
        return [
            "--gene-stats-in",
            str(cls.gene_stats),
            "--gene-stats-id-col",
            "GENE",
            "--gene-stats-log-bf-col",
            "log_bf",
            "--gene-stats-combined-col",
            "combined",
            "--gene-stats-prior-col",
            "prior",
        ]

    def test_gene_set_stats_in_smoke_beta_tildes(self) -> None:
        proc = self._run(
            "beta_tildes",
            *self._common_x_args(),
            "--gene-set-stats-in",
            str(self.reference_root / "mody_beta_tildes_gene_set_beta_tilde.tsv"),
            "--gene-set-stats-id-col",
            "Gene_Set",
            "--gene-set-stats-beta-tilde-col",
            "beta_tilde",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        output = (proc.stdout or "") + (proc.stderr or "")
        self.assertIn("Reading --stats-in file", output)
        self.assertIn("Using col beta_tilde for beta_tilde values", output)

    def test_run_phewas_from_gene_phewas_stats_in_smoke(self) -> None:
        phewas_in = self.tmpdir / "set_b_gene_phewas_in.tsv"
        phewas_out = self.tmpdir / "set_b_phewas_stats.out"

        genes: list[str] = []
        with self.gene_stats.open(encoding="utf-8") as fh:
            next(fh)  # header
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                genes.append(line.split("\t", 1)[0])
                if len(genes) >= 12:
                    break
        self.assertGreaterEqual(len(genes), 4, msg="Expected enough genes to build phewas smoke input")

        with phewas_in.open("w", encoding="utf-8") as fh:
            fh.write("Gene\tPheno\tlog_bf\n")
            for gene in genes:
                fh.write(f"{gene}\tTEST_PHENO_A\t2.0\n")
                fh.write(f"{gene}\tTEST_PHENO_B\t1.5\n")

        proc = self._run(
            "beta_tildes",
            *self._common_x_args(),
            *self._common_gene_stats_args(),
            "--max-num-gene-sets-initial",
            "3",
            "--max-num-gene-sets-hyper",
            "3",
            "--max-num-gene-sets",
            "3",
            "--run-phewas-from-gene-phewas-stats-in",
            str(phewas_in),
            "--gene-phewas-bfs-id-col",
            "Gene",
            "--gene-phewas-bfs-pheno-col",
            "Pheno",
            "--gene-phewas-bfs-log-bf-col",
            "log_bf",
            "--min-gene-phewas-read-value",
            "0",
            "--phewas-stats-out",
            str(phewas_out),
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        self.assertTrue(phewas_out.exists(), msg="Expected phewas-stats-out file to be created")

        lines = [line for line in phewas_out.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreater(len(lines), 1, msg="Expected non-empty phewas output")
        self.assertTrue(lines[0].startswith("Pheno\tanalysis\tbeta_tilde"))

    def test_sim_mode_with_x_only_and_max_gene_set_caps_smoke(self) -> None:
        proc = self._run(
            "sim",
            *self._common_x_args(),
            "--max-num-gene-sets-initial",
            "20",
            "--max-num-gene-sets-hyper",
            "20",
            "--max-num-gene-sets",
            "20",
            "--p-noninf",
            "0.2",
            "--sigma-power",
            "0",
            "--sigma2",
            "0.001",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))


if __name__ == "__main__":
    unittest.main()
