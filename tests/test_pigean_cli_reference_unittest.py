from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PigeanCliReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.doc_path = cls.repo_root / "docs" / "PIGEAN_CLI_REFERENCE.md"
        cls.fixture_root = cls.repo_root / "tests" / "data" / "t2d_smoke"
        cls.model_data = cls.repo_root / "tests" / "data" / "model_small"
        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        src_root = str(self.repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return subprocess.run(
            [sys.executable, "-m", "pigean", *args],
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def _run_ok(self, *args: str) -> subprocess.CompletedProcess[str]:
        proc = self._run(*args)
        if proc.returncode != 0:
            raise AssertionError(
                f"Command failed: {' '.join(args)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return proc

    def test_reference_runtime_flags_round_trip(self) -> None:
        cfg = self.tmpdir / "reference_cfg.json"
        cfg.write_text(json.dumps({"debug_level": 2, "max_gb": 3}), encoding="utf-8")
        proc = self._run_ok(
            "gibbs",
            "--config",
            str(cfg),
            "--deterministic",
            "--seed",
            "17",
            "--hide-opts",
            "--debug-level",
            "4",
            "--max-gb",
            "5",
            "--print-effective-config",
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 17)
        self.assertTrue(payload["options"]["hide_opts"])
        self.assertEqual(payload["options"]["debug_level"], 4)
        self.assertEqual(payload["options"]["max_gb"], 5)

    def test_reference_matrix_and_schema_flags_round_trip(self) -> None:
        x_list = self.tmpdir / "x_inputs.list"
        x_list.write_text("alpha.tsv\n", encoding="utf-8")
        xd_list = self.tmpdir / "xd_inputs.list"
        xd_list.write_text("dense.tsv\n", encoding="utf-8")
        proc = self._run_ok(
            "gibbs",
            "--X-in",
            "alpha.tsv",
            "--X-list",
            str(x_list),
            "--Xd-in",
            "dense.tsv",
            "--Xd-list",
            str(xd_list),
            "--gene-map-in",
            "map.tsv",
            "--gene-loc-file",
            "gene.loc",
            "--gene-loc-file-huge",
            "gene.exons.loc",
            "--exons-loc-file-huge",
            "exons.loc",
            "--print-effective-config",
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["options"]["X_in"], ["alpha.tsv"])
        self.assertEqual(payload["options"]["X_list"], [str(x_list)])
        self.assertEqual(payload["options"]["Xd_in"], ["dense.tsv"])
        self.assertEqual(payload["options"]["Xd_list"], [str(xd_list)])
        self.assertEqual(payload["options"]["gene_map_in"], "map.tsv")
        self.assertEqual(payload["options"]["gene_loc_file"], "gene.loc")
        self.assertEqual(payload["options"]["gene_loc_file_huge"], "gene.exons.loc")
        self.assertEqual(payload["options"]["exons_loc_file_huge"], "exons.loc")

    def test_reference_gwas_and_exome_schema_flags_round_trip(self) -> None:
        proc = self._run_ok(
            "gibbs",
            "--gwas-in",
            "trait.tsv.gz",
            "--gwas-chrom-col",
            "CHR",
            "--gwas-pos-col",
            "BP",
            "--gwas-p-col",
            "PVAL",
            "--gwas-beta-col",
            "BETA",
            "--gwas-se-col",
            "SE",
            "--gwas-n-col",
            "N",
            "--exomes-in",
            "exomes.tsv",
            "--exomes-gene-col",
            "GENE",
            "--exomes-p-col",
            "PVALUE",
            "--exomes-beta-col",
            "BETA",
            "--exomes-se-col",
            "SE",
            "--exomes-n-col",
            "N",
            "--print-effective-config",
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["options"]["gwas_in"], "trait.tsv.gz")
        self.assertEqual(payload["options"]["gwas_chrom_col"], "CHR")
        self.assertEqual(payload["options"]["gwas_pos_col"], "BP")
        self.assertEqual(payload["options"]["gwas_p_col"], "PVAL")
        self.assertEqual(payload["options"]["gwas_beta_col"], "BETA")
        self.assertEqual(payload["options"]["gwas_se_col"], "SE")
        self.assertEqual(payload["options"]["gwas_n_col"], "N")
        self.assertEqual(payload["options"]["exomes_in"], "exomes.tsv")
        self.assertEqual(payload["options"]["exomes_gene_col"], "GENE")
        self.assertEqual(payload["options"]["exomes_p_col"], "PVALUE")
        self.assertEqual(payload["options"]["exomes_beta_col"], "BETA")
        self.assertEqual(payload["options"]["exomes_se_col"], "SE")
        self.assertEqual(payload["options"]["exomes_n_col"], "N")

    def test_reference_positive_controls_file_mode_runs(self) -> None:
        out_prefix = self.tmpdir / "positive_controls_file_mode"
        proc = self._run_ok(
            "beta_tildes",
            "--deterministic",
            "--hide-opts",
            "--X-in",
            str(self.fixture_root / "gene_set_list_mouse_t2d_toy.txt"),
            "--gene-map-in",
            str(self.model_data / "portal_gencode.gene.map"),
            "--gene-loc-file",
            str(self.model_data / "NCBI37.3.plink.gene.loc"),
            "--positive-controls-in",
            str(self.fixture_root / "mody.gene.list"),
            "--positive-controls-no-header",
            "--positive-controls-all-in",
            str(self.model_data / "NCBI37.3.plink.gene.loc"),
            "--positive-controls-all-id-col",
            "6",
            "--positive-controls-all-no-header",
            "--gene-stats-out",
            str(out_prefix.with_suffix(".gene_stats.out")),
            "--gene-set-stats-out",
            str(out_prefix.with_suffix(".gene_set_stats.out")),
            "--params-out",
            str(out_prefix.with_suffix(".params.out")),
        )
        self.assertIn("Reading --positive-controls-in file", (proc.stdout or "") + (proc.stderr or ""))
        self.assertTrue(out_prefix.with_suffix(".gene_stats.out").exists())

    def test_reference_huge_cache_round_trip_runs(self) -> None:
        cache_path = self.tmpdir / "reference_huge.tar.gz"
        huge_prefix = self.tmpdir / "reference_huge"
        huge_proc = self._run_ok(
            "huge",
            "--deterministic",
            "--hide-opts",
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
            "--gene-loc-file-huge",
            str(self.model_data / "NCBI37.3.plink.gene.loc"),
            "--huge-statistics-out",
            str(cache_path),
            "--gene-stats-out",
            str(huge_prefix.with_suffix(".gene_stats.out")),
            "--params-out",
            str(huge_prefix.with_suffix(".params.out")),
        )
        self.assertTrue(cache_path.exists())
        betas_prefix = self.tmpdir / "reference_betas_from_cache"
        betas_proc = self._run_ok(
            "betas",
            "--deterministic",
            "--hide-opts",
            "--X-in",
            str(self.model_data / "gene_set_list_mouse_2024.txt"),
            "--gene-map-in",
            str(self.model_data / "portal_gencode.gene.map"),
            "--huge-statistics-in",
            str(cache_path),
            "--gene-stats-out",
            str(betas_prefix.with_suffix(".gene_stats.out")),
            "--gene-set-stats-out",
            str(betas_prefix.with_suffix(".gene_set_stats.out")),
            "--params-out",
            str(betas_prefix.with_suffix(".params.out")),
        )
        combined = (huge_proc.stdout or "") + (huge_proc.stderr or "") + (betas_proc.stdout or "") + (betas_proc.stderr or "")
        self.assertIn("Reading --gwas-in file", combined)
        self.assertIn("Writing gene stats", combined)
        self.assertTrue(betas_prefix.with_suffix(".gene_set_stats.out").exists())

    def test_reference_precomputed_and_filter_flags_round_trip(self) -> None:
        proc = self._run_ok(
            "priors",
            "--gene-stats-in",
            str(self.repo_root / "tests/data/mody_priors_gene_stats.tsv"),
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
            "--no-filter-negative",
            "--min-gene-set-size",
            "1",
            "--print-effective-config",
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["options"]["gene_stats_in"], str(self.repo_root / "tests/data/mody_priors_gene_stats.tsv"))
        self.assertEqual(payload["options"]["gene_stats_id_col"], "GENE")
        self.assertEqual(payload["options"]["gene_stats_log_bf_col"], "log_bf")
        self.assertEqual(payload["options"]["gene_stats_combined_col"], "combined")
        self.assertEqual(payload["options"]["gene_stats_prior_col"], "prior")
        self.assertEqual(payload["options"]["filter_gene_set_p"], 1.0)
        self.assertEqual(payload["options"]["max_gene_set_read_p"], 1.0)
        self.assertFalse(payload["options"]["filter_negative"])
        self.assertEqual(payload["options"]["min_gene_set_size"], 1)

    def test_reference_documented_flags_are_mapped_to_real_tests(self) -> None:
        documented_flags = sorted(set(re.findall(r"`(--[A-Za-z0-9-]+)`", self.doc_path.read_text(encoding="utf-8"))))
        flag_to_tests = {
            "--config": ["test_reference_runtime_flags_round_trip", "test_missing_config_returns_config_error_without_traceback"],
            "--deterministic": ["test_reference_runtime_flags_round_trip", "test_toy_outputs_are_nonempty_and_include_expected_t2d_genes"],
            "--hide-opts": ["test_reference_runtime_flags_round_trip", "test_toy_outputs_are_nonempty_and_include_expected_t2d_genes"],
            "--seed": ["test_reference_runtime_flags_round_trip", "test_deterministic_keeps_explicit_seed"],
            "--debug-level": ["test_reference_runtime_flags_round_trip"],
            "--max-gb": ["test_reference_runtime_flags_round_trip"],
            "--print-effective-config": ["test_reference_runtime_flags_round_trip", "test_reference_matrix_and_schema_flags_round_trip", "test_reference_gwas_and_exome_schema_flags_round_trip", "test_reference_precomputed_and_filter_flags_round_trip"],
            "--X-in": ["test_toy_outputs_are_nonempty_and_include_expected_t2d_genes", "test_validation_outputs_survive_without_qc_override", "test_reference_matrix_and_schema_flags_round_trip"],
            "--X-list": ["test_reference_matrix_and_schema_flags_round_trip"],
            "--Xd-in": ["test_reference_matrix_and_schema_flags_round_trip"],
            "--Xd-list": ["test_reference_matrix_and_schema_flags_round_trip"],
            "--gene-map-in": ["test_toy_outputs_are_nonempty_and_include_expected_t2d_genes", "test_validation_outputs_survive_without_qc_override", "test_reference_matrix_and_schema_flags_round_trip"],
            "--gene-loc-file": ["test_reference_matrix_and_schema_flags_round_trip", "test_reference_positive_controls_file_mode_runs"],
            "--gene-loc-file-huge": ["test_reference_matrix_and_schema_flags_round_trip", "test_validation_outputs_survive_without_qc_override"],
            "--exons-loc-file-huge": ["test_reference_matrix_and_schema_flags_round_trip"],
            "--gwas-in": ["test_validation_outputs_survive_without_qc_override", "test_real_gwas_huge_matches_legacy", "test_reference_gwas_and_exome_schema_flags_round_trip", "test_reference_huge_cache_round_trip_runs"],
            "--gwas-chrom-col": ["test_reference_gwas_and_exome_schema_flags_round_trip", "test_reference_huge_cache_round_trip_runs"],
            "--gwas-pos-col": ["test_reference_gwas_and_exome_schema_flags_round_trip", "test_reference_huge_cache_round_trip_runs"],
            "--gwas-p-col": ["test_reference_gwas_and_exome_schema_flags_round_trip", "test_reference_huge_cache_round_trip_runs"],
            "--gwas-beta-col": ["test_reference_gwas_and_exome_schema_flags_round_trip"],
            "--gwas-se-col": ["test_reference_gwas_and_exome_schema_flags_round_trip"],
            "--gwas-n-col": ["test_reference_gwas_and_exome_schema_flags_round_trip", "test_reference_huge_cache_round_trip_runs"],
            "--exomes-in": ["test_logs_show_all_toy_fixture_inputs_are_read", "test_reference_gwas_and_exome_schema_flags_round_trip"],
            "--exomes-gene-col": ["test_reference_gwas_and_exome_schema_flags_round_trip", "test_logs_show_all_toy_fixture_inputs_are_read"],
            "--exomes-p-col": ["test_reference_gwas_and_exome_schema_flags_round_trip", "test_logs_show_all_toy_fixture_inputs_are_read"],
            "--exomes-beta-col": ["test_reference_gwas_and_exome_schema_flags_round_trip", "test_logs_show_all_toy_fixture_inputs_are_read"],
            "--exomes-se-col": ["test_reference_gwas_and_exome_schema_flags_round_trip", "test_logs_show_all_toy_fixture_inputs_are_read"],
            "--exomes-n-col": ["test_reference_gwas_and_exome_schema_flags_round_trip"],
            "--positive-controls-in": ["test_reference_positive_controls_file_mode_runs"],
            "--positive-controls-no-header": ["test_reference_positive_controls_file_mode_runs"],
            "--positive-controls-list": ["test_positive_controls_list_rejects_file_paths", "test_positive_controls_only_requires_positive_controls_all_in", "test_toy_outputs_are_nonempty_and_include_expected_t2d_genes"],
            "--positive-controls-all-in": ["test_reference_positive_controls_file_mode_runs", "test_positive_controls_only_requires_positive_controls_all_in", "test_toy_outputs_are_nonempty_and_include_expected_t2d_genes"],
            "--positive-controls-all-id-col": ["test_reference_positive_controls_file_mode_runs", "test_toy_outputs_are_nonempty_and_include_expected_t2d_genes"],
            "--positive-controls-all-no-header": ["test_reference_positive_controls_file_mode_runs", "test_toy_outputs_are_nonempty_and_include_expected_t2d_genes"],
            "--case-counts-in": ["test_logs_show_all_toy_fixture_inputs_are_read"],
            "--ctrl-counts-in": ["test_logs_show_all_toy_fixture_inputs_are_read"],
            "--case-counts-max-freq-col": ["test_logs_show_all_toy_fixture_inputs_are_read"],
            "--ctrl-counts-max-freq-col": ["test_logs_show_all_toy_fixture_inputs_are_read"],
            "--gene-stats-in": ["test_gene_stats_in_option_round_trips_in_effective_config", "test_reference_precomputed_and_filter_flags_round_trip"],
            "--gene-stats-id-col": ["test_reference_precomputed_and_filter_flags_round_trip"],
            "--gene-stats-log-bf-col": ["test_reference_precomputed_and_filter_flags_round_trip"],
            "--gene-stats-combined-col": ["test_reference_precomputed_and_filter_flags_round_trip"],
            "--gene-stats-prior-col": ["test_reference_precomputed_and_filter_flags_round_trip"],
            "--huge-statistics-out": ["test_huge_statistics_out_requires_gwas_in", "test_reference_huge_cache_round_trip_runs"],
            "--huge-statistics-in": ["test_huge_statistics_in_and_out_conflict", "test_reference_huge_cache_round_trip_runs"],
            "--min-gene-set-size": ["test_reference_precomputed_and_filter_flags_round_trip", "test_toy_outputs_are_nonempty_and_include_expected_t2d_genes", "test_validation_outputs_survive_without_qc_override"],
            "--filter-gene-set-metric-z": ["test_toy_outputs_are_nonempty_and_include_expected_t2d_genes", "test_reference_precomputed_and_filter_flags_round_trip"],
            "--filter-gene-set-p": ["test_reference_precomputed_and_filter_flags_round_trip"],
            "--max-gene-set-read-p": ["test_reference_precomputed_and_filter_flags_round_trip"],
            "--no-filter-negative": ["test_reference_precomputed_and_filter_flags_round_trip"],
            "--gene-stats-out": ["test_toy_outputs_are_nonempty_and_include_expected_t2d_genes", "test_validation_outputs_survive_without_qc_override", "test_reference_positive_controls_file_mode_runs", "test_reference_huge_cache_round_trip_runs"],
            "--gene-set-stats-out": ["test_toy_outputs_are_nonempty_and_include_expected_t2d_genes", "test_validation_outputs_survive_without_qc_override", "test_reference_positive_controls_file_mode_runs", "test_reference_huge_cache_round_trip_runs"],
            "--params-out": ["test_toy_outputs_are_nonempty_and_include_expected_t2d_genes", "test_validation_outputs_survive_without_qc_override", "test_reference_positive_controls_file_mode_runs", "test_reference_huge_cache_round_trip_runs"],
        }
        test_files = [
            self.repo_root / "tests/test_pigean_cli_reference_unittest.py",
            self.repo_root / "tests/test_pigean_cli_unittest.py",
            self.repo_root / "tests/test_t2d_toy_bundle_inputs_unittest.py",
            self.repo_root / "tests/test_t2d_validation_bundle_inputs_unittest.py",
            self.repo_root / "tests/test_huge_real_gwas_regression_unittest.py",
            self.repo_root / "tests/test_mody_core_modes_regression_unittest.py",
            self.repo_root / "tests/test_mody_gibbs_regression_unittest.py",
        ]
        known_tests = set()
        for path in test_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    known_tests.add(node.name)
        self.assertEqual(sorted(documented_flags), sorted(flag_to_tests.keys()))
        missing = {flag: [name for name in names if name not in known_tests] for flag, names in flag_to_tests.items()}
        missing = {flag: names for flag, names in missing.items() if names}
        self.assertEqual(missing, {})


if __name__ == "__main__":
    unittest.main()
