from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class HugeRealGwasRegressionTest(unittest.TestCase):
    GENE_VALUE_COLS = [
        "huge_score_gwas",
        "huge_score_gwas_uncorrected",
        "log_bf",
    ]
    PARAM_VALUE_KEYS = [
        "gwas_allelic_var_k",
        "gwas_prior_odds",
        "window_fun_slope",
        "window_fun_intercept",
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.gwas_file = cls.repo_root.parent / "data" / "t2d.chrom_pos.sumstats.gz"
        cls.gene_loc_file = cls.repo_root / "tests" / "data" / "model_small" / "NCBI37.3.plink.gene.loc"
        if not cls.gwas_file.exists():
            raise unittest.SkipTest(f"Missing GWAS fixture: {cls.gwas_file}")
        if not cls.gene_loc_file.exists():
            raise unittest.SkipTest(f"Missing HuGE gene-loc fixture: {cls.gene_loc_file}")

        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)
        cls.runtime_ratio_limit = float(os.environ.get("PIGEAN_RUNTIME_RATIO_LIMIT", "1.1"))

        cls.new_prefix = cls.tmpdir / "new_real_gwas"
        cls.legacy_prefix = cls.tmpdir / "legacy_real_gwas"
        cls.legacy_runtime_sec = cls._run_huge(entrypoint="legacy/priors.py", out_prefix=cls.legacy_prefix)
        cls.new_runtime_sec = cls._run_huge(entrypoint="src/pigean.py", out_prefix=cls.new_prefix)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    @classmethod
    def _common_gwas_args(cls) -> list[str]:
        return [
            "--gwas-in",
            str(cls.gwas_file),
            "--gwas-chrom-col",
            "CHROM",
            "--gwas-pos-col",
            "POS",
            "--gwas-p-col",
            "P",
            "--gwas-n-col",
            "N",
            "--gene-loc-file-huge",
            str(cls.gene_loc_file),
            "--no-correct-huge",
        ]

    @classmethod
    def _run_huge(cls, entrypoint: str, out_prefix: Path) -> float:
        cmd = [
            sys.executable,
            entrypoint,
            "huge",
            "--deterministic",
            "--hide-opts",
            *cls._common_gwas_args(),
            "--gene-stats-out",
            str(out_prefix.with_suffix(".gene_stats.out")),
            "--params-out",
            str(out_prefix.with_suffix(".params.out")),
        ]
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=cls.repo_root, env=env, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({entrypoint}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return elapsed

    @staticmethod
    def _load_gene_stats(path: Path, value_cols: list[str]) -> dict[str, tuple[float, ...]]:
        out: dict[str, tuple[float, ...]] = {}
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = row["Gene"]
                if key in out:
                    raise AssertionError(f"Duplicate key {key} in {path}")
                out[key] = tuple(float(row[col]) for col in value_cols)
        return out

    @staticmethod
    def _load_params(path: Path) -> dict[str, float]:
        out: dict[str, float] = {}
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    out[row["Parameter"]] = float(row["Value"])
                except ValueError:
                    continue
        return out

    def _assert_metric_maps_equal(
        self,
        got: dict[str, tuple[float, ...]],
        expected: dict[str, tuple[float, ...]],
        atol: float = 0.0,
    ) -> None:
        self.assertEqual(set(got.keys()), set(expected.keys()))
        max_abs_diff = 0.0
        for key in got:
            a = got[key]
            b = expected[key]
            self.assertEqual(len(a), len(b))
            for i in range(len(a)):
                d = abs(a[i] - b[i])
                if d > max_abs_diff:
                    max_abs_diff = d
        self.assertLessEqual(max_abs_diff, atol, msg=f"max_abs_diff={max_abs_diff}")

    def _assert_param_subset_equal(self, a: dict[str, float], b: dict[str, float], keys: list[str], atol: float = 0.0) -> None:
        for key in keys:
            self.assertIn(key, a, msg=f"Missing parameter in first file: {key}")
            self.assertIn(key, b, msg=f"Missing parameter in second file: {key}")
            self.assertLessEqual(abs(a[key] - b[key]), atol, msg=f"{key} differs: {a[key]} vs {b[key]}")

    def test_real_gwas_huge_matches_legacy(self) -> None:
        new_gene = self._load_gene_stats(self.new_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        legacy_gene = self._load_gene_stats(self.legacy_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        self.assertGreaterEqual(len(new_gene), 10000, "Unexpectedly small real-GWAS output; check fixture wiring")
        self._assert_metric_maps_equal(new_gene, legacy_gene, atol=0.0)

        new_params = self._load_params(self.new_prefix.with_suffix(".params.out"))
        legacy_params = self._load_params(self.legacy_prefix.with_suffix(".params.out"))
        self._assert_param_subset_equal(new_params, legacy_params, self.PARAM_VALUE_KEYS, atol=0.0)

    def test_runtime_not_slower_than_legacy(self) -> None:
        max_allowed = self.legacy_runtime_sec * self.runtime_ratio_limit
        self.assertLessEqual(
            self.new_runtime_sec,
            max_allowed,
            msg=(
                f"Real-GWAS HuGE runtime slower than legacy: new={self.new_runtime_sec:.4f}s "
                f"legacy={self.legacy_runtime_sec:.4f}s limit={self.runtime_ratio_limit:.3f}x"
            ),
        )


if __name__ == "__main__":
    unittest.main()
