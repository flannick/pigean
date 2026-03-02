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


if __name__ == "__main__":
    unittest.main()
