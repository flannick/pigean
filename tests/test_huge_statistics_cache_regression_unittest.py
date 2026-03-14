from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class HugeStatisticsCacheRegressionTest(unittest.TestCase):
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
        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)
        cls.runtime_ratio_limit = float(os.environ.get("PIGEAN_RUNTIME_RATIO_LIMIT", "1.1"))

        cls.gene_loc_file = cls.tmpdir / "tiny.gene.loc"
        cls.gwas_file = cls.tmpdir / "tiny.gwas.tsv"
        cls._write_tiny_huge_fixtures()

        # Baseline parity check against legacy implementation.
        cls.new_direct_prefix = cls.tmpdir / "new_direct"
        cls.legacy_direct_prefix = cls.tmpdir / "legacy_direct"
        cls.legacy_direct_runtime_sec = cls._run_huge(
            entrypoint="legacy/priors.py",
            extra_args=cls._common_gwas_args(),
            out_prefix=cls.legacy_direct_prefix,
        )
        cls.new_direct_runtime_sec = cls._run_huge(
            entrypoint="module:pigean",
            extra_args=cls._common_gwas_args(),
            out_prefix=cls.new_direct_prefix,
        )

        # Cache roundtrip through single-file tar.gz bundle.
        cls.cache_tar = cls.tmpdir / "huge_cache.tar.gz"
        cls.tar_direct_prefix = cls.tmpdir / "tar_direct"
        cls.tar_cached_prefix = cls.tmpdir / "tar_cached"
        cls._run_huge(
            entrypoint="module:pigean",
            extra_args=cls._common_gwas_args() + ["--huge-statistics-out", str(cls.cache_tar)],
            out_prefix=cls.tar_direct_prefix,
        )
        cls._run_huge(
            entrypoint="module:pigean",
            extra_args=["--huge-statistics-in", str(cls.cache_tar)],
            out_prefix=cls.tar_cached_prefix,
        )

        # Cache roundtrip through prefix-based multi-file cache.
        cls.cache_prefix = cls.tmpdir / "huge_cache_prefix"
        cls.prefix_direct_prefix = cls.tmpdir / "prefix_direct"
        cls.prefix_cached_prefix = cls.tmpdir / "prefix_cached"
        cls._run_huge(
            entrypoint="module:pigean",
            extra_args=cls._common_gwas_args() + ["--huge-statistics-out", str(cls.cache_prefix)],
            out_prefix=cls.prefix_direct_prefix,
        )
        cls._run_huge(
            entrypoint="module:pigean",
            extra_args=["--huge-statistics-in", str(cls.cache_prefix)],
            out_prefix=cls.prefix_cached_prefix,
        )

        # Deterministic repeatability check for the direct path.
        cls.new_direct_repeat_prefix = cls.tmpdir / "new_direct_repeat"
        cls._run_huge(
            entrypoint="module:pigean",
            extra_args=cls._common_gwas_args(),
            out_prefix=cls.new_direct_repeat_prefix,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    @classmethod
    def _write_tiny_huge_fixtures(cls) -> None:
        cls.gene_loc_file.write_text(
            "\n".join(
                [
                    "ENSG000001 1 100000 110000 + GENE1",
                    "ENSG000002 1 200000 210000 + GENE2",
                    "ENSG000003 1 300000 310000 + GENE3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        cls.gwas_file.write_text(
            "\n".join(
                [
                    "CHR\tPOS\tP\tBETA\tSE\tN",
                    "1\t100100\t1e-7\t0.20\t0.037546530337571084\t10000",
                    "1\t100300\t1e-6\t-0.10\t0.020443047967832226\t10000",
                    "1\t200100\t5e-8\t0.30\t0.05503263910953657\t10000",
                    "1\t250000\t5e-5\t0.05\t0.012328549995532977\t10000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def _common_gwas_args(cls) -> list[str]:
        return [
            "--gwas-in",
            str(cls.gwas_file),
            "--gwas-chrom-col",
            "CHR",
            "--gwas-pos-col",
            "POS",
            "--gwas-p-col",
            "P",
            "--gwas-beta-col",
            "BETA",
            "--gwas-se-col",
            "SE",
            "--gwas-n-col",
            "N",
            "--gene-loc-file-huge",
            str(cls.gene_loc_file),
            "--no-correct-huge",
        ]

    @classmethod
    def _run_huge(cls, entrypoint: str, extra_args: list[str], out_prefix: Path) -> float:
        cmd = cls._build_entrypoint_cmd(entrypoint) + [
            "huge",
            "--deterministic",
            "--hide-opts",
            *extra_args,
            "--gene-stats-out",
            str(out_prefix.with_suffix(".gene_stats.out")),
            "--params-out",
            str(out_prefix.with_suffix(".params.out")),
        ]
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        src_root = str(cls.repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=cls.repo_root, env=env, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({entrypoint}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return elapsed

    @staticmethod
    def _build_entrypoint_cmd(entrypoint: str) -> list[str]:
        if entrypoint.startswith("module:"):
            return [sys.executable, "-m", entrypoint.split(":", 1)[1]]
        return [sys.executable, entrypoint]

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
                    # params.out mixes numeric and boolean/string values; tests only compare numeric keys.
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

    def test_huge_gwas_matches_legacy(self) -> None:
        new_gene = self._load_gene_stats(self.new_direct_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        legacy_gene = self._load_gene_stats(self.legacy_direct_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        self._assert_metric_maps_equal(new_gene, legacy_gene, atol=0.0)

        new_params = self._load_params(self.new_direct_prefix.with_suffix(".params.out"))
        legacy_params = self._load_params(self.legacy_direct_prefix.with_suffix(".params.out"))
        self._assert_param_subset_equal(new_params, legacy_params, self.PARAM_VALUE_KEYS, atol=0.0)

    def test_huge_statistics_tar_cache_roundtrip_matches_direct(self) -> None:
        direct_gene = self._load_gene_stats(self.tar_direct_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        cached_gene = self._load_gene_stats(self.tar_cached_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        self._assert_metric_maps_equal(direct_gene, cached_gene, atol=0.0)

        direct_params = self._load_params(self.tar_direct_prefix.with_suffix(".params.out"))
        cached_params = self._load_params(self.tar_cached_prefix.with_suffix(".params.out"))
        self._assert_param_subset_equal(direct_params, cached_params, self.PARAM_VALUE_KEYS, atol=0.0)

    def test_huge_statistics_prefix_cache_roundtrip_matches_direct(self) -> None:
        direct_gene = self._load_gene_stats(self.prefix_direct_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        cached_gene = self._load_gene_stats(self.prefix_cached_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        self._assert_metric_maps_equal(direct_gene, cached_gene, atol=0.0)

        direct_params = self._load_params(self.prefix_direct_prefix.with_suffix(".params.out"))
        cached_params = self._load_params(self.prefix_cached_prefix.with_suffix(".params.out"))
        self._assert_param_subset_equal(direct_params, cached_params, self.PARAM_VALUE_KEYS, atol=0.0)

    def test_huge_direct_is_deterministic(self) -> None:
        first_gene = self._load_gene_stats(self.new_direct_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        second_gene = self._load_gene_stats(self.new_direct_repeat_prefix.with_suffix(".gene_stats.out"), self.GENE_VALUE_COLS)
        self._assert_metric_maps_equal(first_gene, second_gene, atol=0.0)

        first_params = self._load_params(self.new_direct_prefix.with_suffix(".params.out"))
        second_params = self._load_params(self.new_direct_repeat_prefix.with_suffix(".params.out"))
        self._assert_param_subset_equal(first_params, second_params, self.PARAM_VALUE_KEYS, atol=0.0)

    def test_runtime_not_slower_than_legacy_direct(self) -> None:
        max_allowed = self.legacy_direct_runtime_sec * self.runtime_ratio_limit
        self.assertLessEqual(
            self.new_direct_runtime_sec,
            max_allowed,
            msg=(
                f"HuGE direct runtime slower than legacy: new={self.new_direct_runtime_sec:.4f}s "
                f"legacy={self.legacy_direct_runtime_sec:.4f}s limit={self.runtime_ratio_limit:.3f}x"
            ),
        )


if __name__ == "__main__":
    unittest.main()
