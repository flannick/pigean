from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ModyCoreModesRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.model_data = cls.repo_root / "tests/data/model_small"
        cls.gene_list = cls.repo_root / "tests/data/mody.gene.list"
        cls.gene_stats = cls.repo_root / "tests/data/mody_priors_gene_stats.tsv"
        cls.reference_root = cls.repo_root / "tests/data/reference"

        required = [
            cls.model_data / "gene_set_list_mouse_2024.txt",
            cls.model_data / "portal_gencode.gene.map",
            cls.model_data / "NCBI37.3.plink.gene.loc",
            cls.gene_list,
            cls.gene_stats,
            cls.reference_root / "mody_beta_tildes_gene_set_beta_tilde.tsv",
            cls.reference_root / "mody_betas_gene_set_betas.tsv",
            cls.reference_root / "mody_priors_fast_gene_prior.tsv",
            cls.reference_root / "mody_priors_fast_gene_set_beta_uncorrected.tsv",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise unittest.SkipTest("Missing MODY core-mode fixtures: " + ", ".join(missing))

        genes = [x.strip() for x in cls.gene_list.read_text(encoding="utf-8").splitlines() if x.strip()]
        cls.gene_csv = ",".join(genes)

        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)

        cls._run_mode_pair("beta_tildes", cls._beta_tildes_args())
        cls._run_mode_pair("betas", cls._betas_args())
        cls._run_mode_pair("priors_fast", cls._priors_fast_args())

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    @classmethod
    def _common_base_args(cls) -> list[str]:
        return [
            "--X-in",
            str(cls.model_data / "gene_set_list_mouse_2024.txt"),
            "--gene-map-in",
            str(cls.model_data / "portal_gencode.gene.map"),
            "--hide-opts",
            "--deterministic",
            "--num-chains-betas",
            "2",
            "--max-num-iter-betas",
            "20",
            "--min-num-iter-betas",
            "5",
            "--max-num-burn-in",
            "5",
        ]

    @classmethod
    def _beta_tildes_args(cls) -> list[str]:
        return cls._common_base_args() + [
            "--positive-controls-list",
            cls.gene_csv,
            "--positive-controls-all-in",
            str(cls.model_data / "NCBI37.3.plink.gene.loc"),
            "--positive-controls-all-id-col",
            "6",
            "--positive-controls-all-no-header",
        ]

    @classmethod
    def _betas_args(cls) -> list[str]:
        return cls._beta_tildes_args()

    @classmethod
    def _priors_fast_args(cls) -> list[str]:
        return cls._common_base_args() + [
            "--gene-loc-file",
            str(cls.model_data / "NCBI37.3.plink.gene.loc"),
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
            "--filter-gene-set-p",
            "1",
            "--max-gene-set-read-p",
            "1",
            "--min-gene-set-size",
            "1",
            "--no-filter-negative",
            "--max-num-gene-sets-initial",
            "200",
            "--max-num-gene-sets-hyper",
            "200",
            "--max-num-gene-sets",
            "200",
            "--priors-num-gene-batches",
            "4",
        ]

    @classmethod
    def _run_mode_pair(cls, mode: str, mode_args: list[str]) -> None:
        cls._run_single(
            "src/pigean.py",
            mode,
            mode_args,
            cls.tmpdir / f"{mode}_new",
        )
        cls._run_single(
            "legacy/priors.py",
            mode if mode != "priors_fast" else "priors",
            mode_args,
            cls.tmpdir / f"{mode}_legacy",
        )

    @classmethod
    def _run_single(cls, entrypoint: str, mode: str, mode_args: list[str], out_prefix: Path) -> None:
        effective_mode = mode if mode != "priors_fast" else "priors"
        cmd = [
            sys.executable,
            entrypoint,
            effective_mode,
            *mode_args,
            "--gene-stats-out",
            str(out_prefix.with_suffix(".gene_stats.out")),
            "--gene-set-stats-out",
            str(out_prefix.with_suffix(".gene_set_stats.out")),
            "--params-out",
            str(out_prefix.with_suffix(".params.out")),
        ]
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        proc = subprocess.run(cmd, cwd=cls.repo_root, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({entrypoint} {effective_mode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

    @staticmethod
    def _load_metric(path: Path, key_col: str, value_cols: list[str]) -> dict[str, tuple[float, ...]]:
        out: dict[str, tuple[float, ...]] = {}
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = row[key_col]
                if key in out:
                    raise AssertionError(f"Duplicate key {key} in {path}")
                out[key] = tuple(float(row[col]) for col in value_cols)
        return out

    def _assert_maps_equal(self, got: dict[str, tuple[float, ...]], expected: dict[str, tuple[float, ...]]) -> None:
        self.assertEqual(set(got.keys()), set(expected.keys()))
        max_diff = 0.0
        for key in got:
            a = got[key]
            b = expected[key]
            self.assertEqual(len(a), len(b))
            for i in range(len(a)):
                d = abs(a[i] - b[i])
                if d > max_diff:
                    max_diff = d
        self.assertLessEqual(max_diff, 0.0, msg=f"max_abs_diff={max_diff}")

    def test_beta_tildes_match_legacy_and_reference(self) -> None:
        new = self._load_metric(self.tmpdir / "beta_tildes_new.gene_set_stats.out", "Gene_Set", ["beta_tilde"])
        legacy = self._load_metric(self.tmpdir / "beta_tildes_legacy.gene_set_stats.out", "Gene_Set", ["beta_tilde"])
        ref = self._load_metric(
            self.reference_root / "mody_beta_tildes_gene_set_beta_tilde.tsv",
            "Gene_Set",
            ["beta_tilde"],
        )
        self._assert_maps_equal(new, legacy)
        self._assert_maps_equal(new, ref)

    def test_betas_match_legacy_and_reference(self) -> None:
        new = self._load_metric(
            self.tmpdir / "betas_new.gene_set_stats.out",
            "Gene_Set",
            ["beta", "beta_uncorrected"],
        )
        legacy = self._load_metric(
            self.tmpdir / "betas_legacy.gene_set_stats.out",
            "Gene_Set",
            ["beta", "beta_uncorrected"],
        )
        ref = self._load_metric(
            self.reference_root / "mody_betas_gene_set_betas.tsv",
            "Gene_Set",
            ["beta", "beta_uncorrected"],
        )
        self._assert_maps_equal(new, legacy)
        self._assert_maps_equal(new, ref)

    def test_priors_fast_match_legacy_and_reference(self) -> None:
        new_gene = self._load_metric(self.tmpdir / "priors_fast_new.gene_stats.out", "Gene", ["prior"])
        legacy_gene = self._load_metric(self.tmpdir / "priors_fast_legacy.gene_stats.out", "Gene", ["prior"])
        ref_gene = self._load_metric(
            self.reference_root / "mody_priors_fast_gene_prior.tsv",
            "Gene",
            ["prior"],
        )
        self._assert_maps_equal(new_gene, legacy_gene)
        self._assert_maps_equal(new_gene, ref_gene)

        new_set = self._load_metric(
            self.tmpdir / "priors_fast_new.gene_set_stats.out",
            "Gene_Set",
            ["beta_uncorrected"],
        )
        legacy_set = self._load_metric(
            self.tmpdir / "priors_fast_legacy.gene_set_stats.out",
            "Gene_Set",
            ["beta_uncorrected"],
        )
        ref_set = self._load_metric(
            self.reference_root / "mody_priors_fast_gene_set_beta_uncorrected.tsv",
            "Gene_Set",
            ["beta_uncorrected"],
        )
        self._assert_maps_equal(new_set, legacy_set)
        self._assert_maps_equal(new_set, ref_set)


if __name__ == "__main__":
    unittest.main()
