from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from tests.eaggl.test_eaggl_cli_unittest import EagglCliTest


class EagglCliReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.doc_path = cls.repo_root / "docs" / "eaggl" / "CLI_REFERENCE.md"
        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        src_root = str(self.repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return env

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "eaggl", *args],
            cwd=self.repo_root,
            env=self._env(),
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

    def _write_minimal_eaggl_bundle(self, root: Path) -> Path:
        manifest = {
            "schema": "pigean_eaggl_bundle/v1",
            "default_inputs": {
                "X_in": "X.tsv.gz",
                "gene_stats_in": "gene_stats.tsv.gz",
                "gene_set_stats_in": "gene_set_stats.tsv.gz",
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "X.tsv.gz").write_text("SET_A\tGENE1\n", encoding="utf-8")
        (root / "gene_stats.tsv.gz").write_text("Gene\tprior\nGENE1\t0.1\n", encoding="utf-8")
        (root / "gene_set_stats.tsv.gz").write_text("Gene_Set\tbeta_uncorrected\nSET_A\t0.1\n", encoding="utf-8")
        bundle_path = root / "handoff.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as tar_fh:
            tar_fh.add(root / "manifest.json", arcname="manifest.json")
            tar_fh.add(root / "X.tsv.gz", arcname="X.tsv.gz")
            tar_fh.add(root / "gene_stats.tsv.gz", arcname="gene_stats.tsv.gz")
            tar_fh.add(root / "gene_set_stats.tsv.gz", arcname="gene_set_stats.tsv.gz")
        return bundle_path

    def test_reference_runtime_flags_round_trip(self) -> None:
        cfg = self.tmpdir / "eaggl_reference_cfg.json"
        cfg.write_text(json.dumps({"debug_level": 2, "max_gb": 3}), encoding="utf-8")
        proc = self._run_ok(
            "factor",
            "--config",
            str(cfg),
            "--deterministic",
            "--seed",
            "17",
            "--debug-level",
            "4",
            "--max-gb",
            "5",
            "--print-effective-config",
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "factor")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 17)
        self.assertEqual(payload["options"]["debug_level"], 4)
        self.assertEqual(payload["options"]["max_gb"], 5)

    def test_reference_matrix_and_bundle_flags_round_trip(self) -> None:
        x_list = self.tmpdir / "x_inputs.list"
        x_list.write_text("alpha.tsv\n", encoding="utf-8")
        xd_list = self.tmpdir / "xd_inputs.list"
        xd_list.write_text("dense.tsv\n", encoding="utf-8")
        proc = self._run_ok(
            "factor",
            "--X-in",
            "alpha.tsv",
            "--X-list",
            str(x_list),
            "--Xd-in",
            "dense.tsv",
            "--Xd-list",
            str(xd_list),
            "--gene-stats-in",
            "gene_stats.tsv",
            "--gene-set-stats-in",
            "gene_set_stats.tsv",
            "--gene-map-in",
            "map.tsv",
            "--gene-loc-file",
            "gene.loc",
            "--print-effective-config",
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["options"]["X_in"], ["alpha.tsv"])
        self.assertEqual(payload["options"]["X_list"], [str(x_list)])
        self.assertEqual(payload["options"]["Xd_in"], ["dense.tsv"])
        self.assertEqual(payload["options"]["Xd_list"], [str(xd_list)])
        self.assertEqual(payload["options"]["gene_stats_in"], "gene_stats.tsv")
        self.assertEqual(payload["options"]["gene_set_stats_in"], "gene_set_stats.tsv")
        self.assertEqual(payload["options"]["gene_map_in"], "map.tsv")
        self.assertEqual(payload["options"]["gene_loc_file"], "gene.loc")

    def test_reference_bundle_defaults_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle_path = self._write_minimal_eaggl_bundle(Path(td))
            proc = self._run_ok("factor", "--eaggl-bundle-in", str(bundle_path), "--print-effective-config")
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["options"]["X_in"][0].endswith("X.tsv.gz"))
        self.assertTrue(payload["options"]["gene_stats_in"].endswith("gene_stats.tsv.gz"))
        self.assertTrue(payload["options"]["gene_set_stats_in"].endswith("gene_set_stats.tsv.gz"))
        self.assertEqual(payload["eaggl_bundle"]["schema"], "pigean_eaggl_bundle/v1")

    def test_reference_workflow_selector_flags_round_trip(self) -> None:
        positive_controls = self._run_ok(
            "factor",
            "--positive-controls-in",
            "controls.tsv",
            "--positive-controls-list",
            "INS,GCK",
            "--positive-controls-all-in",
            "all.tsv",
            "--print-effective-config",
        )
        positive_payload = json.loads(positive_controls.stdout)
        self.assertEqual(positive_payload["options"]["positive_controls_in"], "controls.tsv")
        self.assertEqual(positive_payload["options"]["positive_controls_list"], ["INS", "GCK"])
        self.assertEqual(positive_payload["options"]["positive_controls_all_in"], "all.tsv")

        pheno_anchor = self._run_ok(
            "factor",
            "--anchor-phenos",
            "T2D,T2D_ALT",
            "--anchor-any-pheno",
            "--gene-phewas-stats-in",
            "gene_phewas.tsv",
            "--gene-set-phewas-stats-in",
            "gene_set_phewas.tsv",
            "--print-effective-config",
        )
        pheno_payload = json.loads(pheno_anchor.stdout)
        self.assertEqual(pheno_payload["options"]["anchor_phenos"], ["T2D", "T2D_ALT"])
        self.assertTrue(pheno_payload["options"]["anchor_any_pheno"])

        gene_anchor = self._run_ok(
            "factor",
            "--anchor-genes",
            "INS,GCK",
            "--anchor-any-gene",
            "--gene-phewas-stats-in",
            "gene_phewas.tsv",
            "--gene-set-phewas-stats-in",
            "gene_set_phewas.tsv",
            "--print-effective-config",
        )
        gene_payload = json.loads(gene_anchor.stdout)
        self.assertEqual(sorted(gene_payload["options"]["anchor_genes"]), ["GCK", "INS"])
        self.assertTrue(gene_payload["options"]["anchor_any_gene"])

        gene_set_anchor = self._run_ok(
            "factor",
            "--anchor-gene-set",
            "--run-phewas-from-gene-phewas-stats-in",
            "gene_phewas.tsv",
            "--print-effective-config",
        )
        gene_set_payload = json.loads(gene_set_anchor.stdout)
        self.assertTrue(gene_set_payload["options"]["anchor_gene_set"])

    def test_reference_phewas_and_schema_flags_round_trip(self) -> None:
        proc = self._run_ok(
            "factor",
            "--gene-phewas-stats-in",
            "gene_phewas.tsv",
            "--gene-set-phewas-stats-in",
            "gene_set_phewas.tsv",
            "--run-phewas-from-gene-phewas-stats-in",
            "gene_phewas.tsv",
            "--factor-phewas-from-gene-phewas-stats-in",
            "gene_phewas.tsv",
            "--project-phenos-from-gene-sets",
            "--gene-stats-id-col",
            "GENE",
            "--gene-stats-prior-col",
            "prior_col",
            "--gene-set-stats-id-col",
            "SET",
            "--gene-set-stats-beta-uncorrected-col",
            "beta_u",
            "--gene-phewas-stats-id-col",
            "GENE",
            "--gene-phewas-stats-pheno-col",
            "PHENO",
            "--gene-set-phewas-stats-id-col",
            "SET",
            "--gene-set-phewas-stats-pheno-col",
            "PHENO",
            "--gene-phewas-id-to-X-id",
            "map.tsv",
            "--print-effective-config",
        )
        payload = json.loads(proc.stdout)
        opts = payload["options"]
        self.assertEqual(opts["gene_phewas_bfs_in"], "gene_phewas.tsv")
        self.assertEqual(opts["gene_set_phewas_stats_in"], "gene_set_phewas.tsv")
        self.assertEqual(opts["run_phewas_from_gene_phewas_stats_in"], "gene_phewas.tsv")
        self.assertEqual(opts["factor_phewas_from_gene_phewas_stats_in"], "gene_phewas.tsv")
        self.assertTrue(opts["project_phenos_from_gene_sets"])
        self.assertEqual(opts["gene_stats_id_col"], "GENE")
        self.assertEqual(opts["gene_stats_prior_col"], "prior_col")
        self.assertEqual(opts["gene_set_stats_id_col"], "SET")
        self.assertEqual(opts["gene_set_stats_beta_uncorrected_col"], "beta_u")
        self.assertEqual(opts["gene_phewas_bfs_id_col"], "GENE")
        self.assertEqual(opts["gene_phewas_bfs_pheno_col"], "PHENO")
        self.assertEqual(opts["gene_set_phewas_stats_id_col"], "SET")
        self.assertEqual(opts["gene_set_phewas_stats_pheno_col"], "PHENO")
        self.assertEqual(opts["gene_phewas_id_to_X_id"], "map.tsv")

    def test_reference_factor_and_labeling_flags_round_trip(self) -> None:
        proc = self._run_ok(
            "factor",
            "--max-num-factors",
            "12",
            "--phi",
            "0.1",
            "--alpha0",
            "7",
            "--beta0",
            "2",
            "--factor-runs",
            "4",
            "--consensus-nmf",
            "--consensus-min-factor-cosine",
            "0.8",
            "--consensus-min-run-support",
            "0.75",
            "--consensus-aggregation",
            "mean",
            "--min-lambda-threshold",
            "0.005",
            "--no-transpose",
            "--factor-prune-gene-sets-num",
            "5",
            "--factor-prune-genes-val",
            "0.25",
            "--factor-prune-phenos-num",
            "4",
            "--factor-phewas-min-gene-factor-weight",
            "0.02",
            "--threshold-weights",
            "0.4",
            "--lmm-provider",
            "openai",
            "--lmm-model",
            "gpt-4o-mini",
            "--lmm-auth-key",
            "ENV:OPENAI_API_KEY",
            "--label-gene-sets-only",
            "--label-include-phenos",
            "--label-individually",
            "--factors-out",
            "factors.tsv",
            "--factors-anchor-out",
            "factors_anchor.tsv",
            "--consensus-stats-out",
            "consensus.tsv",
            "--gene-set-clusters-out",
            "gene_set_clusters.tsv",
            "--gene-clusters-out",
            "gene_clusters.tsv",
            "--pheno-clusters-out",
            "pheno_clusters.tsv",
            "--gene-set-anchor-clusters-out",
            "gene_set_anchor.tsv",
            "--gene-anchor-clusters-out",
            "gene_anchor.tsv",
            "--pheno-anchor-clusters-out",
            "pheno_anchor.tsv",
            "--factor-phewas-stats-out",
            "factor_phewas.tsv",
            "--gene-pheno-stats-out",
            "gene_pheno.tsv",
            "--params-out",
            "params.tsv",
            "--print-effective-config",
        )
        opts = json.loads(proc.stdout)["options"]
        self.assertEqual(opts["max_num_factors"], 12)
        self.assertEqual(opts["phi"], 0.1)
        self.assertEqual(opts["alpha0"], 7.0)
        self.assertEqual(opts["beta0"], 2.0)
        self.assertEqual(opts["factor_runs"], 4)
        self.assertTrue(opts["consensus_nmf"])
        self.assertEqual(opts["consensus_min_factor_cosine"], 0.8)
        self.assertEqual(opts["consensus_min_run_support"], 0.75)
        self.assertEqual(opts["consensus_aggregation"], "mean")
        self.assertEqual(opts["min_lambda_threshold"], 0.005)
        self.assertTrue(opts["no_transpose"])
        self.assertEqual(opts["factor_prune_gene_sets_num"], 5)
        self.assertEqual(opts["factor_prune_genes_val"], 0.25)
        self.assertEqual(opts["factor_prune_phenos_num"], 4)
        self.assertEqual(opts["factor_phewas_min_gene_factor_weight"], 0.02)
        self.assertEqual(opts["threshold_weights"], 0.4)
        self.assertEqual(opts["lmm_provider"], "openai")
        self.assertEqual(opts["lmm_model"], "gpt-4o-mini")
        self.assertEqual(opts["lmm_auth_key"], "ENV:OPENAI_API_KEY")
        self.assertTrue(opts["label_gene_sets_only"])
        self.assertTrue(opts["label_include_phenos"])
        self.assertTrue(opts["label_individually"])
        self.assertEqual(opts["factors_out"], "factors.tsv")
        self.assertEqual(opts["consensus_stats_out"], "consensus.tsv")
        self.assertEqual(opts["params_out"], "params.tsv")

    def test_reference_documented_flags_are_mapped_to_real_tests(self) -> None:
        documented_flags = sorted(set(re.findall(r"`(--[A-Za-z0-9-]+)`", self.doc_path.read_text(encoding="utf-8"))))
        flag_to_tests = {
            "--config": ["test_reference_runtime_flags_round_trip", "test_missing_config_returns_config_error_without_traceback"],
            "--deterministic": ["test_reference_runtime_flags_round_trip", "test_deterministic_sets_seed_zero"],
            "--seed": ["test_reference_runtime_flags_round_trip", "test_deterministic_keeps_explicit_seed"],
            "--debug-level": ["test_reference_runtime_flags_round_trip"],
            "--max-gb": ["test_reference_runtime_flags_round_trip"],
            "--print-effective-config": ["test_reference_runtime_flags_round_trip", "test_reference_matrix_and_bundle_flags_round_trip", "test_reference_workflow_selector_flags_round_trip"],
            "--X-in": ["test_reference_matrix_and_bundle_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--X-list": ["test_reference_matrix_and_bundle_flags_round_trip"],
            "--Xd-in": ["test_reference_matrix_and_bundle_flags_round_trip"],
            "--Xd-list": ["test_reference_matrix_and_bundle_flags_round_trip"],
            "--gene-stats-in": ["test_reference_matrix_and_bundle_flags_round_trip", "test_reference_bundle_defaults_round_trip"],
            "--gene-set-stats-in": ["test_reference_matrix_and_bundle_flags_round_trip", "test_reference_bundle_defaults_round_trip"],
            "--eaggl-bundle-in": ["test_reference_bundle_defaults_round_trip", "test_eaggl_bundle_in_populates_default_inputs"],
            "--gene-map-in": ["test_reference_matrix_and_bundle_flags_round_trip"],
            "--gene-loc-file": ["test_reference_matrix_and_bundle_flags_round_trip"],
            "--positive-controls-in": ["test_reference_workflow_selector_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--positive-controls-list": ["test_reference_workflow_selector_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--positive-controls-all-in": ["test_reference_workflow_selector_flags_round_trip"],
            "--anchor-phenos": ["test_reference_workflow_selector_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--anchor-any-pheno": ["test_reference_workflow_selector_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--anchor-genes": ["test_reference_workflow_selector_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--anchor-any-gene": ["test_reference_workflow_selector_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--anchor-gene-set": ["test_reference_workflow_selector_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--gene-phewas-stats-in": ["test_reference_phewas_and_schema_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--gene-set-phewas-stats-in": ["test_reference_phewas_and_schema_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--run-phewas-from-gene-phewas-stats-in": ["test_reference_phewas_and_schema_flags_round_trip", "test_factor_workflow_ids_in_effective_config"],
            "--factor-phewas-from-gene-phewas-stats-in": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--project-phenos-from-gene-sets": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-stats-id-col": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-stats-prior-col": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-set-stats-id-col": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-set-stats-beta-uncorrected-col": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-phewas-stats-id-col": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-phewas-stats-pheno-col": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-set-phewas-stats-id-col": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-set-phewas-stats-pheno-col": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--gene-phewas-id-to-X-id": ["test_reference_phewas_and_schema_flags_round_trip"],
            "--max-num-factors": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--phi": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--alpha0": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--beta0": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-runs": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--consensus-nmf": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--consensus-min-factor-cosine": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--consensus-min-run-support": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--consensus-aggregation": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--min-lambda-threshold": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--no-transpose": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-prune-gene-sets-num": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-prune-gene-sets-val": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-prune-genes-num": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-prune-genes-val": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-prune-phenos-num": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-prune-phenos-val": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-phewas-min-gene-factor-weight": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--threshold-weights": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--lmm-provider": ["test_reference_factor_and_labeling_flags_round_trip", "test_help_expert_includes_projection_and_labeling_flags"],
            "--lmm-model": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--lmm-auth-key": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--label-gene-sets-only": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--label-include-phenos": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--label-individually": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factors-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factors-anchor-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--consensus-stats-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--gene-set-clusters-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--gene-clusters-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--pheno-clusters-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--gene-set-anchor-clusters-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--gene-anchor-clusters-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--pheno-anchor-clusters-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--factor-phewas-stats-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--gene-pheno-stats-out": ["test_reference_factor_and_labeling_flags_round_trip"],
            "--params-out": ["test_reference_factor_and_labeling_flags_round_trip"],
        }
        missing = [flag for flag in documented_flags if flag not in flag_to_tests]
        self.assertEqual([], missing, msg=f"Documented flags missing coverage mapping: {missing}")
        for flag, tests in flag_to_tests.items():
            for test_name in tests:
                self.assertTrue(
                    hasattr(self, test_name) or hasattr(EagglCliTest, test_name),
                    msg=f"Mapped test '{test_name}' for {flag} does not exist",
                )


if __name__ == "__main__":
    unittest.main()
