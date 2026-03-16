from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class T2DInputSubsetMatrixTest(unittest.TestCase):
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
            cls.fixture_root / "toy_positive_controls_all.tsv",
            cls.model_data / "portal_gencode.gene.map",
            cls.model_data / "NCBI37.3.plink.gene.loc",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest("Missing bundled T2D input-subset fixtures: " + ", ".join(missing))

        cls.positive_controls_csv = ",".join(
            gene.strip()
            for gene in (cls.fixture_root / "mody.gene.list").read_text(encoding="utf-8").splitlines()
            if gene.strip()
        )
        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    def _base_cmd(self, out_prefix: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pigean",
            "beta_tildes",
            "--deterministic",
            "--hide-opts",
            "--linear",
            "--max-for-linear",
            "100",
            "--no-correct-betas-mean",
            "--X-in",
            str(self.fixture_root / "gene_set_list_mouse_t2d_toy.txt"),
            "--gene-map-in",
            str(self.model_data / "portal_gencode.gene.map"),
            "--gene-loc-file",
            str(self.model_data / "NCBI37.3.plink.gene.loc"),
            "--gene-loc-file-huge",
            str(self.model_data / "NCBI37.3.plink.gene.loc"),
            "--min-gene-set-size",
            "1",
            "--filter-gene-set-metric-z",
            "0",
            "--gene-stats-out",
            str(out_prefix.with_suffix(".gene_stats.out")),
            "--gene-set-stats-out",
            str(out_prefix.with_suffix(".gene_set_stats.out")),
            "--params-out",
            str(out_prefix.with_suffix(".params.out")),
        ]

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        src_root = str(self.repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return env

    def _run_subset(self, name: str, *, gwas: bool, exomes: bool, pc_mode: str | None, counts: bool):
        out_prefix = self.tmpdir / name
        cmd = self._base_cmd(out_prefix)
        if gwas:
            cmd.extend(
                [
                    "--gwas-in",
                    str(self.fixture_root / "T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz"),
                    "--gwas-chrom-col",
                    "CHROM",
                    "--gwas-pos-col",
                    "POS",
                    "--gwas-p-col",
                    "P",
                    "--gwas-n-col",
                    "N",
                ]
            )
        if exomes:
            cmd.extend(
                [
                    "--exomes-in",
                    str(self.fixture_root / "T2D.exomes.p_lt_1e-4_or_mody.tsv"),
                    "--exomes-gene-col",
                    "GeneSymbol",
                    "--exomes-p-col",
                    "P-value",
                    "--exomes-beta-col",
                    "beta",
                    "--exomes-se-col",
                    "se",
                ]
            )
        if pc_mode is not None:
            if pc_mode in {"list", "both"}:
                cmd.extend(["--positive-controls-list", self.positive_controls_csv])
            if pc_mode in {"file", "both"}:
                cmd.extend(
                    [
                        "--positive-controls-in",
                        str(self.fixture_root / "mody.gene.list"),
                        "--positive-controls-no-header",
                    ]
                )
            cmd.extend(
                [
                    "--positive-controls-all-in",
                    str(self.fixture_root / "toy_positive_controls_all.tsv"),
                    "--positive-controls-all-id-col",
                    "gene",
                ]
            )
        if counts:
            cmd.extend(
                [
                    "--case-counts-in",
                    str(self.fixture_root / "mody_case_counts.tsv"),
                    "--ctrl-counts-in",
                    str(self.fixture_root / "mody_ctrl_counts.tsv"),
                    "--case-counts-max-freq-col",
                    "max_freq",
                    "--ctrl-counts-max-freq-col",
                    "max_freq",
                ]
            )
        proc = subprocess.run(cmd, cwd=self.repo_root, env=self._env(), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise AssertionError(
                f"Command failed for {name}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return out_prefix, proc

    @staticmethod
    def _load_rows(path: Path, key_col: str) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        with path.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                out[row[key_col]] = row
        return out

    def test_all_nonempty_input_subsets_produce_gene_set_betatildes(self) -> None:
        cases = []
        for gwas in [False, True]:
            for exomes in [False, True]:
                for pc in [False, True]:
                    for counts in [False, True]:
                        if not any([gwas, exomes, pc, counts]):
                            continue
                        name = "subset_" + "_".join(
                            tag
                            for tag, enabled in [
                                ("gwas", gwas),
                                ("exomes", exomes),
                                ("pc", pc),
                                ("counts", counts),
                            ]
                            if enabled
                        )
                        cases.append((name, gwas, exomes, "list" if pc else None, counts))

        for name, gwas, exomes, pc_mode, counts in cases:
            with self.subTest(name=name):
                out_prefix, proc = self._run_subset(name, gwas=gwas, exomes=exomes, pc_mode=pc_mode, counts=counts)
                combined = (proc.stdout or "") + (proc.stderr or "")
                gene_set_rows = self._load_rows(out_prefix.with_suffix(".gene_set_stats.out"), "Gene_Set")
                gene_rows = self._load_rows(out_prefix.with_suffix(".gene_stats.out"), "Gene")
                self.assertGreaterEqual(len(gene_set_rows), 1)
                self.assertGreaterEqual(len(gene_rows), 10)
                if gwas:
                    self.assertIn("Reading --gwas-in file", combined)
                if exomes:
                    self.assertIn("Reading --exomes-in file", combined)
                if pc_mode is not None:
                    self.assertIn("Reading --gene-list-in file", combined)
                if counts:
                    self.assertIn("Reading case counts from", combined)
                    self.assertIn("Reading ctrl counts from", combined)

    def test_positive_controls_in_only_produces_expected_signal(self) -> None:
        out_prefix, _proc = self._run_subset("pc_file_only", gwas=False, exomes=False, pc_mode="file", counts=False)
        gene_rows = self._load_rows(out_prefix.with_suffix(".gene_stats.out"), "Gene")
        self.assertGreater(float(gene_rows["HNF1A"]["positive_control"]), 0.0)
        self.assertGreater(float(gene_rows["GCK"]["positive_control"]), 0.0)

    def test_positive_controls_list_and_file_together_keep_list_priority_and_add_file_only_genes(self) -> None:
        custom_file = self.tmpdir / "custom_positive_controls.tsv"
        custom_file.write_text("gene\tprob\nHNF1A\t0.2\nTCF7L2\t0.8\n", encoding="utf-8")

        list_only_prefix = self.tmpdir / "pc_list_only"
        list_only_cmd = self._base_cmd(list_only_prefix) + [
            "--positive-controls-list",
            self.positive_controls_csv,
            "--positive-controls-all-in",
            str(self.fixture_root / "toy_positive_controls_all.tsv"),
            "--positive-controls-all-id-col",
            "gene",
        ]
        list_only = subprocess.run(
            list_only_cmd,
            cwd=self.repo_root,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if list_only.returncode != 0:
            raise AssertionError(list_only.stdout + list_only.stderr)

        both_prefix = self.tmpdir / "pc_both"
        both_cmd = self._base_cmd(both_prefix) + [
            "--positive-controls-list",
            self.positive_controls_csv,
            "--positive-controls-in",
            str(custom_file),
            "--positive-controls-id-col",
            "gene",
            "--positive-controls-prob-col",
            "prob",
            "--positive-controls-all-in",
            str(self.fixture_root / "toy_positive_controls_all.tsv"),
            "--positive-controls-all-id-col",
            "gene",
        ]
        both = subprocess.run(
            both_cmd,
            cwd=self.repo_root,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if both.returncode != 0:
            raise AssertionError(both.stdout + both.stderr)

        list_rows = self._load_rows(list_only_prefix.with_suffix(".gene_stats.out"), "Gene")
        both_rows = self._load_rows(both_prefix.with_suffix(".gene_stats.out"), "Gene")
        self.assertAlmostEqual(
            float(list_rows["HNF1A"]["positive_control"]),
            float(both_rows["HNF1A"]["positive_control"]),
            places=9,
        )
        self.assertEqual(float(list_rows["TCF7L2"]["positive_control"]), 0.0)
        self.assertGreater(float(both_rows["TCF7L2"]["positive_control"]), 0.0)


if __name__ == "__main__":
    unittest.main()
