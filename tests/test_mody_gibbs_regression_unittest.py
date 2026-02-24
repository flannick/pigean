from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class ModyGibbsRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.mody_gene_list = cls.repo_root / "tests/data/mody.gene.list"
        cls.ref_gene_prior = cls.repo_root / "tests/data/reference/mody_gibbs_gene_prior.tsv"
        cls.ref_gene_set_beta = cls.repo_root / "tests/data/reference/mody_gibbs_gene_set_beta_uncorrected.tsv"
        if not cls.mody_gene_list.exists() or not cls.ref_gene_prior.exists() or not cls.ref_gene_set_beta.exists():
            raise unittest.SkipTest("MODY regression fixtures are missing")

        cls.bundle_data = cls._resolve_bundle_data()
        if cls.bundle_data is None:
            raise unittest.SkipTest(
                "Model bundle data not found; set PIGEAN_TEST_MODEL_DATA or provide bundles/current/model_small/data"
            )

        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)
        cls.runtime_ratio_limit = float(os.environ.get("PIGEAN_RUNTIME_RATIO_LIMIT", "1.1"))

        genes = [line.strip() for line in cls.mody_gene_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        cls.positive_controls_csv = ",".join(genes)

        cls.new_prefix = cls.tmpdir / "mody_new"
        cls.legacy_prefix = cls.tmpdir / "mody_legacy"

        cls.legacy_runtime_sec = cls._run_gibbs("legacy/priors.py", cls.legacy_prefix)
        cls.new_runtime_sec = cls._run_gibbs("src/pigean.py", cls.new_prefix)

        cls.new_gene_prior = cls._load_metric(
            cls.new_prefix.with_suffix(".gene_stats.out"), key_col="Gene", value_col="prior"
        )
        cls.legacy_gene_prior = cls._load_metric(
            cls.legacy_prefix.with_suffix(".gene_stats.out"), key_col="Gene", value_col="prior"
        )
        cls.new_gene_set_beta = cls._load_metric(
            cls.new_prefix.with_suffix(".gene_set_stats.out"),
            key_col="Gene_Set",
            value_col="beta_uncorrected",
        )
        cls.legacy_gene_set_beta = cls._load_metric(
            cls.legacy_prefix.with_suffix(".gene_set_stats.out"),
            key_col="Gene_Set",
            value_col="beta_uncorrected",
        )
        cls.ref_gene_prior_map = cls._load_metric(cls.ref_gene_prior, key_col="Gene", value_col="prior")
        cls.ref_gene_set_beta_map = cls._load_metric(
            cls.ref_gene_set_beta, key_col="Gene_Set", value_col="beta_uncorrected"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    @classmethod
    def _resolve_bundle_data(cls) -> Path | None:
        candidates: list[Path] = []
        override = os.environ.get("PIGEAN_TEST_MODEL_DATA")
        if override:
            candidates.append(Path(override))
        candidates.append(cls.repo_root / "tests/data/model_small")
        candidates.append(cls.repo_root / "bundles/current/model_small/data")
        required = [
            "gene_set_list_mouse_2024.txt",
            "portal_gencode.gene.map",
            "NCBI37.3.plink.gene.loc",
        ]
        for path in candidates:
            if path.exists() and all((path / rel).exists() for rel in required):
                return path
        return None

    @classmethod
    def _run_gibbs(cls, entrypoint: str, out_prefix: Path) -> float:
        cmd = [
            sys.executable,
            entrypoint,
            "gibbs",
            "--X-in",
            str(cls.bundle_data / "gene_set_list_mouse_2024.txt"),
            "--gene-map-in",
            str(cls.bundle_data / "portal_gencode.gene.map"),
            "--positive-controls-list",
            cls.positive_controls_csv,
            "--positive-controls-all-in",
            str(cls.bundle_data / "NCBI37.3.plink.gene.loc"),
            "--positive-controls-all-id-col",
            "6",
            "--positive-controls-all-no-header",
            "--gene-stats-out",
            str(out_prefix.with_suffix(".gene_stats.out")),
            "--gene-set-stats-out",
            str(out_prefix.with_suffix(".gene_set_stats.out")),
            "--params-out",
            str(out_prefix.with_suffix(".params.out")),
            "--hide-opts",
            "--deterministic",
            "--num-chains",
            "4",
            "--num-chains-betas",
            "2",
            "--max-num-iter",
            "30",
            "--total-num-iter-gibbs",
            "30",
            "--max-num-restarts",
            "0",
            "--min-num-burn-in",
            "5",
            "--min-num-post-burn-in",
            "5",
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
    def _load_metric(path: Path, key_col: str, value_col: str) -> dict[str, float]:
        out: dict[str, float] = {}
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = row[key_col]
                if key in out:
                    raise AssertionError(f"Duplicate key {key} in {path}")
                out[key] = float(row[value_col])
        return out

    def _assert_metric_maps_equal(self, got: dict[str, float], expected: dict[str, float], atol: float = 0.0) -> None:
        self.assertEqual(set(got.keys()), set(expected.keys()))
        max_abs_diff = 0.0
        for key in got:
            d = abs(got[key] - expected[key])
            if d > max_abs_diff:
                max_abs_diff = d
        self.assertLessEqual(max_abs_diff, atol, msg=f"max_abs_diff={max_abs_diff}")

    def test_gene_prior_matches_legacy(self) -> None:
        self._assert_metric_maps_equal(self.new_gene_prior, self.legacy_gene_prior, atol=0.0)

    def test_gene_set_beta_uncorrected_matches_legacy(self) -> None:
        self._assert_metric_maps_equal(self.new_gene_set_beta, self.legacy_gene_set_beta, atol=0.0)

    def test_gene_prior_matches_reference(self) -> None:
        self._assert_metric_maps_equal(self.new_gene_prior, self.ref_gene_prior_map, atol=0.0)

    def test_gene_set_beta_uncorrected_matches_reference(self) -> None:
        self._assert_metric_maps_equal(self.new_gene_set_beta, self.ref_gene_set_beta_map, atol=0.0)

    def test_runtime_not_slower_than_legacy(self) -> None:
        max_allowed = self.legacy_runtime_sec * self.runtime_ratio_limit
        self.assertLessEqual(
            self.new_runtime_sec,
            max_allowed,
            msg=(
                f"New runtime slower than legacy: new={self.new_runtime_sec:.4f}s "
                f"legacy={self.legacy_runtime_sec:.4f}s limit={self.runtime_ratio_limit:.3f}x"
            ),
        )


if __name__ == "__main__":
    unittest.main()
