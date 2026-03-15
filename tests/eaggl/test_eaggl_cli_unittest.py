from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


class EagglCliTest(unittest.TestCase):
    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _env(self, repo_root: Path) -> dict[str, str]:
        env = os.environ.copy()
        src_root = str(repo_root / "src")
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src_root if not existing else src_root + os.pathsep + existing
        return env

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        repo_root = self._repo_root()
        cmd = [sys.executable, "-m", "eaggl", *args]
        return subprocess.run(
            cmd,
            cwd=repo_root,
            env=self._env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )

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

    def test_help_usage_uses_eaggl_name(self) -> None:
        proc = self._run("factor", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Usage: python -m eaggl", proc.stdout)
        self.assertNotIn("[factor|naive_factor|label]", proc.stdout)

    def test_help_states_labeling_has_no_separate_mode(self) -> None:
        proc = self._run("factor", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("there is no separate label mode", proc.stdout)

    def test_default_help_uses_canonical_anchor_flags(self) -> None:
        proc = self._run("factor", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--anchor-genes", proc.stdout)
        self.assertIn("--anchor-phenos", proc.stdout)
        self.assertNotIn("--anchor-gene ", proc.stdout)
        self.assertNotIn("--anchor-pheno ", proc.stdout)

    def test_help_expert_includes_projection_and_labeling_flags(self) -> None:
        proc = self._run("factor", "--help-expert")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Projection quickstart:", proc.stdout)
        self.assertIn("--gene-set-phewas-stats-in", proc.stdout)
        self.assertIn("--lmm-provider", proc.stdout)
        self.assertIn("--run-phewas-from-gene-phewas-stats-in", proc.stdout)

    def test_non_factor_modes_fail_with_routing_message(self) -> None:
        proc = self._run("gibbs")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("belongs to pigean.py", err)

    def test_removed_gene_zs_flag_is_rejected(self) -> None:
        proc = self._run("factor", "--gene-zs-in", "dummy.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --gene-zs-in has been removed and is no longer supported", err)

    def test_removed_run_gls_flag_is_rejected(self) -> None:
        proc = self._run("factor", "--run-gls")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --run-gls has been removed and is no longer supported", err)
        self.assertNotIn("Traceback", err)

    def test_invalid_option_returns_usage_error_without_traceback(self) -> None:
        proc = self._run("factor", "--definitely-invalid-option")
        self.assertEqual(proc.returncode, 2)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("no such option", err)
        self.assertNotIn("Traceback", err)

    def test_missing_config_returns_config_error_without_traceback(self) -> None:
        proc = self._run("factor", "--config", "definitely_missing_config.json")
        self.assertEqual(proc.returncode, 2)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Could not read config file", err)
        self.assertNotIn("Traceback", err)

    def test_removed_anchor_gene_alias_has_replacement_message(self) -> None:
        proc = self._run("factor", "--anchor-gene", "INS")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --anchor-gene has been removed; use --anchor-genes instead", err)

    def test_deterministic_sets_seed_zero(self) -> None:
        proc = self._run("factor", "--deterministic", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "factor")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 0)

    def test_deterministic_keeps_explicit_seed(self) -> None:
        proc = self._run("factor", "--deterministic", "--seed", "123", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "factor")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 123)

    def test_import_does_not_reset_python_random_seed(self) -> None:
        repo_root = self._repo_root()
        snippet = r'''
import random
import sys
random.seed(12345)
expected = random.Random(12345).random()
sys.argv = ["eaggl.py", "factor"]
import eaggl  # noqa: F401
actual = random.random()
print(f"{actual:.17f}\t{expected:.17f}")
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        last_line = (proc.stdout or "").strip().splitlines()[-1]
        actual, expected = last_line.split("\t")
        self.assertEqual(actual, expected)

    def test_import_does_not_parse_invalid_argv(self) -> None:
        repo_root = self._repo_root()
        snippet = r'''
import sys
sys.argv = ["eaggl.py", "--definitely-invalid-option"]
import eaggl  # noqa: F401
print("ok")
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        self.assertEqual((proc.stdout or "").strip().splitlines()[-1], "ok")

    def test_main_accepts_argv_directly(self) -> None:
        repo_root = self._repo_root()
        snippet = r'''
import contextlib
import io
import json
import eaggl
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = eaggl.main(["factor", "--deterministic", "--print-effective-config"])
payload = json.loads(buf.getvalue())
print(json.dumps({"rc": rc, "mode": payload["mode"], "seed": payload["options"]["seed"]}, sort_keys=True))
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
        self.assertEqual(payload["rc"], 0)
        self.assertEqual(payload["mode"], "factor")
        self.assertEqual(payload["seed"], 0)

    def test_factor_workflow_ids_in_effective_config(self) -> None:
        cases = [
            ("F1", []),
            ("F2", ["--positive-controls-list", "INS"]),
            ("F3", ["--gene-phewas-stats-in", "dummy_gene_phewas.tsv"]),
            (
                "F4",
                [
                    "--anchor-phenos",
                    "T2D,T2D_ALT",
                    "--gene-phewas-stats-in",
                    "dummy_gene_phewas.tsv",
                    "--gene-set-phewas-stats-in",
                    "dummy_gene_set_phewas.tsv",
                ],
            ),
            (
                "F5",
                [
                    "--anchor-any-pheno",
                    "--gene-phewas-stats-in",
                    "dummy_gene_phewas.tsv",
                    "--gene-set-phewas-stats-in",
                    "dummy_gene_set_phewas.tsv",
                ],
            ),
            (
                "F6",
                [
                    "--anchor-genes",
                    "INS",
                    "--gene-phewas-stats-in",
                    "dummy_gene_phewas.tsv",
                    "--gene-set-phewas-stats-in",
                    "dummy_gene_set_phewas.tsv",
                ],
            ),
            (
                "F7",
                [
                    "--anchor-genes",
                    "INS,GCK",
                    "--gene-phewas-stats-in",
                    "dummy_gene_phewas.tsv",
                    "--gene-set-phewas-stats-in",
                    "dummy_gene_set_phewas.tsv",
                ],
            ),
            (
                "F8",
                [
                    "--anchor-any-gene",
                    "--gene-phewas-stats-in",
                    "dummy_gene_phewas.tsv",
                    "--gene-set-phewas-stats-in",
                    "dummy_gene_set_phewas.tsv",
                ],
            ),
            (
                "F9",
                [
                    "--anchor-gene-set",
                    "--run-phewas-from-gene-phewas-stats-in",
                    "dummy_gene_phewas.tsv",
                ],
            ),
        ]

        for expected_id, args in cases:
            with self.subTest(workflow=expected_id):
                proc = self._run("factor", "--deterministic", "--print-effective-config", *args)
                self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
                payload = json.loads(proc.stdout)
                self.assertIn("factor_workflow", payload)
                self.assertEqual(payload["factor_workflow"]["id"], expected_id)
                self.assertIn("required_inputs", payload["factor_workflow"])
                self.assertIn("missing_required_inputs", payload["factor_workflow"])
                if expected_id == "F4":
                    self.assertEqual(
                        payload["factor_workflow"]["label"],
                        "multiple phenotype anchoring (to {'T2D', 'T2D_ALT'})",
                    )
                if expected_id == "F7":
                    self.assertEqual(
                        payload["factor_workflow"]["label"],
                        "multiple gene anchoring (to {'GCK', 'INS'})",
                    )

    def test_factor_workflow_missing_inputs_fails_fast(self) -> None:
        proc = self._run("factor", "--anchor-genes", "INS")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Require --gene-set-phewas-stats-in and --gene-phewas-stats-in", err)

    def test_eaggl_bundle_in_populates_default_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle_path = self._write_minimal_eaggl_bundle(Path(td))
            proc = self._run("factor", "--eaggl-bundle-in", str(bundle_path), "--print-effective-config")
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            payload = json.loads(proc.stdout)
            options = payload["options"]
            self.assertIsInstance(options["X_in"], list)
            self.assertEqual(len(options["X_in"]), 1)
            self.assertTrue(options["X_in"][0].endswith("X.tsv.gz"))
            self.assertTrue(options["gene_stats_in"].endswith("gene_stats.tsv.gz"))
            self.assertTrue(options["gene_set_stats_in"].endswith("gene_set_stats.tsv.gz"))
            self.assertIn("eaggl_bundle", payload)
            self.assertEqual(payload["eaggl_bundle"]["schema"], "pigean_eaggl_bundle/v1")

    def test_eaggl_bundle_in_respects_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle_path = self._write_minimal_eaggl_bundle(Path(td))
            override_gene_stats = Path(td) / "override_gene_stats.tsv"
            override_gene_stats.write_text("Gene\tprior\nGENE2\t0.2\n", encoding="utf-8")
            proc = self._run(
                "factor",
                "--eaggl-bundle-in",
                str(bundle_path),
                "--gene-stats-in",
                str(override_gene_stats),
                "--print-effective-config",
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            payload = json.loads(proc.stdout)
            options = payload["options"]
            self.assertEqual(options["gene_stats_in"], str(override_gene_stats))
            self.assertTrue(options["gene_set_stats_in"].endswith("gene_set_stats.tsv.gz"))

    def test_warns_when_direct_gmt_is_passed_to_x_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gmt_path = Path(td) / "toy.gmt"
            gmt_path.write_text("SET_A\tna\tGENE1\tGENE2\n", encoding="utf-8")
            proc = self._run("factor", "--X-list", str(gmt_path), "--print-effective-config")
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            self.assertIn("Direct GMT path passed to --X-list", proc.stderr or "")
            self.assertIn("Use --X-in for direct .gmt files", proc.stderr or "")
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["options"]["X_list"], [str(gmt_path)])

    def test_read_correlations_fails_fast_when_gls_cholesky_is_initialized(self) -> None:
        repo_root = self._repo_root()
        snippet = r"""
import sys
import numpy as np
sys.argv = ["eaggl.py", "factor", "--ols"]
from eaggl import state as eaggl_state
g = eaggl_state.GeneSetData()
g.y_corr_cholesky = np.array([[1.0]])
g.genes = ["GENE1"]
g.gene_to_ind = {"GENE1": 0}
try:
    g._read_correlations(gene_loc_file="definitely_missing.tsv")
except Exception as ex:
    msg = str(ex)
    if "full GLS correlation state" not in msg:
        raise SystemExit("unexpected error: " + msg)
    raise SystemExit(0)
raise SystemExit("expected _read_correlations to fail fast before file IO")
"""
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))

    def test_factor_rejects_raw_pigean_inputs(self) -> None:
        proc = self._run("factor", "--gwas-in", "dummy.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("moved to pigean.py", err)


if __name__ == "__main__":
    unittest.main()
