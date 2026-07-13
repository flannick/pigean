from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class EagglLabelModeTest(unittest.TestCase):
    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(SRC_ROOT) if not existing else str(SRC_ROOT) + os.pathsep + existing
        return env

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "eaggl", *args],
            cwd=REPO_ROOT,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def _read_tsv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8") as fh:
            return list(csv.DictReader(fh, delimiter="\t"))

    def test_label_mode_gene_set_only_writes_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gene_sets = root / "gene_set_clusters.out"
            factors = root / "factors.out"
            gene_sets.write_text(
                "Gene_Set\tbeta\tcluster\tlabel\tFactor1\tFactor2\n"
                "GS_beta\t0.9\tFactor1\told\t0.8\t0.1\n"
                "GS_adipose\t0.7\tFactor2\told\t0.2\t0.9\n",
                encoding="utf-8",
            )
            proc = self._run(
                "label",
                "--label-gene-set-clusters-in",
                str(gene_sets),
                "--factors-out",
                str(factors),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            rows = self._read_tsv(factors)
            self.assertEqual([row["label"] for row in rows], ["GS_beta", "GS_adipose"])
            self.assertEqual(rows[0]["top_gene_sets"].split(",")[0], "GS_beta")

    def test_label_mode_gene_only_falls_back_to_top_genes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            genes = root / "gene_clusters.out"
            factors = root / "factors.out"
            genes.write_text(
                "Gene\tcluster\tlabel\tFactor1\tFactor2\n"
                "PDX1\tFactor1\told\t1.0\t0.0\n"
                "ADIPOQ\tFactor2\told\t0.0\t1.0\n",
                encoding="utf-8",
            )
            proc = self._run(
                "label",
                "--label-gene-clusters-in",
                str(genes),
                "--factors-out",
                str(factors),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            rows = self._read_tsv(factors)
            self.assertEqual([row["label"] for row in rows], ["PDX1", "ADIPOQ"])
            self.assertEqual(rows[0]["top_genes"].split(",")[0], "PDX1")

    def test_label_mode_long_trait_links_can_write_pheno_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            links = root / "trait_factor_links.out"
            phenos = root / "pheno_clusters.out"
            links.write_text(
                "trait\tfactor\tnnls_loading\n"
                "T2D\tFactor1\t0.8\n"
                "T2D\tFactor2\t0.1\n"
                "CAD\tFactor1\t0.0\n"
                "CAD\tFactor2\t0.9\n",
                encoding="utf-8",
            )
            proc = self._run(
                "label",
                "--label-trait-factor-links-in",
                str(links),
                "--label-pheno-clusters-out",
                str(phenos),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            rows = self._read_tsv(phenos)
            self.assertEqual(rows[0]["Pheno"], "T2D")
            self.assertEqual(rows[0]["cluster"], "Factor1")
            self.assertIn("Cosine_Factor1", rows[0])
            self.assertIn("Euclidean_Factor1", rows[0])

    def test_label_mode_mismatched_factor_columns_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            genes = root / "gene_clusters.out"
            gene_sets = root / "gene_set_clusters.out"
            factors = root / "factors.out"
            genes.write_text("Gene\tFactor1\nG1\t1\n", encoding="utf-8")
            gene_sets.write_text("Gene_Set\tFactor1\tFactor2\nGS1\t1\t0\n", encoding="utf-8")
            proc = self._run(
                "label",
                "--label-gene-clusters-in",
                str(genes),
                "--label-gene-set-clusters-in",
                str(gene_sets),
                "--factors-out",
                str(factors),
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("disagree on factor columns", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
