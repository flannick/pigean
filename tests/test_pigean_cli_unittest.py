from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PigeanCliTest(unittest.TestCase):
    def _base_env(self, repo_root: Path) -> dict[str, str]:
        env = dict(os.environ)
        src_root = str(repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return env

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, "-m", "pigean", *args]
        return subprocess.run(
            cmd,
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_removed_gene_bfs_flag_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--gene-bfs-in", "dummy.txt")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --gene-bfs-in has been removed; use --gene-stats-in instead", err)

    def test_removed_gene_zs_flag_has_removed_message(self) -> None:
        proc = self._run("gibbs", "--gene-zs-in", "dummy.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --gene-zs-in has been removed and is no longer supported", err)

    def test_removed_gene_percentiles_flag_has_removed_message(self) -> None:
        proc = self._run("gibbs", "--gene-percentiles-in", "dummy.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --gene-percentiles-in has been removed and is no longer supported", err)

    def test_removed_sigma_flag_has_removed_message(self) -> None:
        proc = self._run("gibbs", "--chisq-threshold", "5")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --chisq-threshold has been removed and is no longer supported", err)

    def test_removed_run_gls_flag_has_removed_message(self) -> None:
        proc = self._run("gibbs", "--run-gls")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --run-gls has been removed and is no longer supported", err)
        self.assertNotIn("Traceback", err)

    def test_invalid_option_returns_usage_error_without_traceback(self) -> None:
        proc = self._run("gibbs", "--definitely-invalid-option")
        self.assertEqual(proc.returncode, 2)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("no such option", err)
        self.assertNotIn("Traceback", err)

    def test_missing_config_returns_config_error_without_traceback(self) -> None:
        proc = self._run("gibbs", "--config", "definitely_missing_config.json")
        self.assertEqual(proc.returncode, 2)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Could not read config file", err)
        self.assertNotIn("Traceback", err)

    def test_positive_controls_list_rejects_file_paths(self) -> None:
        proc = self._run("gibbs", "--positive-controls-list", "tests/data/mody.gene.list")
        self.assertEqual(proc.returncode, 2)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("expects a comma-separated list of gene symbols", err)
        self.assertIn("--positive-controls-in", err)
        self.assertNotIn("Traceback", err)

    def test_removed_min_post_burn_alias_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--min-post-burn-in", "50")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --min-post-burn-in has been removed; use --min-num-post-burn-in instead", err)

    def test_removed_burn_in_post_reserve_alias_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--burn-in-post-reserve", "50")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --burn-in-post-reserve has been removed; use --min-num-post-burn-in instead", err)

    def test_removed_stall_min_post_burn_alias_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--stall-min-post-burn-in", "50")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --stall-min-post-burn-in has been removed; use --stall-min-post-burn-samples instead", err)

    def test_removed_min_num_iter_alias_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--min-num-iter", "50")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --min-num-iter has been removed; use --min-num-post-burn-in instead", err)

    def test_deterministic_sets_seed_zero(self) -> None:
        proc = self._run("gibbs", "--deterministic", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 0)

    def test_deterministic_keeps_explicit_seed(self) -> None:
        proc = self._run("gibbs", "--deterministic", "--seed", "123", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 123)

    def test_experimental_hyper_threshold_requires_experimental_mode(self) -> None:
        proc = self._run(
            "gibbs",
            "--experimental-increase-hyper-if-betas-below",
            "0.01",
            "--print-effective-config",
        )
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("requires --experimental-hyper-mutation", err)

    def test_legacy_hyper_threshold_alias_warns_and_maps(self) -> None:
        proc = self._run(
            "gibbs",
            "--experimental-hyper-mutation",
            "--increase-hyper-if-betas-below",
            "0.02",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["options"]["experimental_hyper_mutation"])
        self.assertEqual(payload["options"]["experimental_increase_hyper_if_betas_below"], 0.02)
        err = proc.stderr or ""
        self.assertIn("legacy alias", err)

    def test_import_does_not_reset_python_random_seed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        snippet = r'''
import random
import sys
random.seed(12345)
expected = random.Random(12345).random()
sys.argv = ["pigean.py", "gibbs"]
import pigean  # noqa: F401
actual = random.random()
print(f"{actual:.17f}\t{expected:.17f}")
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        last_line = (proc.stdout or "").strip().splitlines()[-1]
        actual, expected = last_line.split("\t")
        self.assertEqual(actual, expected)

    def test_import_does_not_parse_invalid_argv(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        snippet = r'''
import sys
sys.argv = ["pigean.py", "--definitely-invalid-option"]
import pigean  # noqa: F401
print("ok")
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        self.assertEqual((proc.stdout or "").strip().splitlines()[-1], "ok")

    def test_main_accepts_argv_directly(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        snippet = r'''
import contextlib
import io
import json
import sys
import pigean as pigean
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = pigean.main(["gibbs", "--deterministic", "--print-effective-config"])
payload = json.loads(buf.getvalue())
print(json.dumps({"rc": rc, "mode": payload["mode"], "seed": payload["options"]["seed"]}, sort_keys=True))
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
        self.assertEqual(payload["rc"], 0)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertEqual(payload["seed"], 0)

    def test_prefilter_keep_mask_phewas_relax_uses_gene_set_axis(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        snippet = r'''
import json
import sys
import numpy as np
sys.argv = ["pigean.py", "gibbs"]
from pigean import app as pigean_app
mask = pigean_app._build_prefilter_keep_mask(
    p_values=np.array([0.8, 0.9, 1.0], dtype=float),
    beta_tildes=np.array([1.0, 1.0, 1.0], dtype=float),
    filter_gene_set_p=0.01,
    filter_using_phewas=True,
    p_values_phewas=np.array([[0.95, 0.95, 0.95], [0.95, 0.1, 0.95]], dtype=float),
    beta_tildes_phewas=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float),
    increase_filter_gene_set_p=0.34,
    filter_negative=False,
)
print(json.dumps(mask.tolist()))
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
        self.assertEqual(payload, [True, True, False])

    def test_help_usage_uses_pigean_name(self) -> None:
        proc = self._run("gibbs", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Usage: python -m pigean", proc.stdout)

    def test_help_includes_core_and_expert_sections(self) -> None:
        proc = self._run("gibbs", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Core quickstart:", proc.stdout)
        self.assertIn("Alternative quickstart:", proc.stdout)
        self.assertIn("Use --help-expert", proc.stdout)
        self.assertIn("Core options:", proc.stdout)
        self.assertIn("Expert options:", proc.stdout)

    def test_default_help_hides_expert_flags(self) -> None:
        proc = self._run("gibbs", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn("--run-phewas-from-gene-phewas-stats-in", proc.stdout)
        self.assertNotIn("--huge-statistics-in", proc.stdout)

    def test_help_expert_includes_set_b_flags(self) -> None:
        proc = self._run("gibbs", "--help-expert")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--run-phewas-from-gene-phewas-stats-in", proc.stdout)
        self.assertIn("run gene-level phewas output stage", proc.stdout)
        self.assertIn("--gene-stats-in", proc.stdout)
        self.assertIn("use precomputed gene-level statistics", proc.stdout)
        self.assertIn("--huge-statistics-in", proc.stdout)
        self.assertIn("read precomputed HuGE statistics cache", proc.stdout)

    def test_huge_statistics_out_requires_gwas_in(self) -> None:
        proc = self._run("gibbs", "--huge-statistics-out", "cache_prefix")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --huge-statistics-out requires --gwas-in", err)

    def test_huge_statistics_in_and_out_conflict(self) -> None:
        proc = self._run(
            "gibbs",
            "--huge-statistics-in",
            "cache_prefix",
            "--huge-statistics-out",
            "cache_prefix_out",
        )
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Do not pass both --huge-statistics-in and --huge-statistics-out", err)

    def test_eaggl_bundle_out_requires_tar_extension(self) -> None:
        proc = self._run("gibbs", "--eaggl-bundle-out", "handoff_bundle.txt")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --eaggl-bundle-out must end with .tar, .tar.gz, or .tgz", err)

    def test_run_phewas_from_gene_phewas_stats_requires_output_path(self) -> None:
        proc = self._run("beta_tildes", "--run-phewas-from-gene-phewas-stats-in", "x.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("requires --phewas-stats-out", err)

    def test_gene_set_stats_column_option_requires_gene_set_stats_in(self) -> None:
        proc = self._run("gibbs", "--gene-set-stats-beta-col", "beta")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --gene-set-stats-beta-col requires --gene-set-stats-in", err)

    def test_gene_stats_column_option_requires_gene_stats_in(self) -> None:
        proc = self._run("gibbs", "--gene-stats-log-bf-col", "log_bf")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --gene-stats-log-bf-col requires --gene-stats-in", err)

    def test_gene_phewas_input_requires_consumer_flag(self) -> None:
        proc = self._run("gibbs", "--gene-phewas-bfs-in", "phewas.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --gene-phewas-bfs-in requires either --betas-uncorrected-from-phewas", err)

    def test_pops_mode_defaults_are_exposed_in_effective_config(self) -> None:
        proc = self._run("pops", "--deterministic", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "pops")
        self.assertTrue(payload["options"]["linear"])
        self.assertEqual(payload["options"]["update_hyper"], "none")
        self.assertTrue(payload["options"]["cross_val"])

    def test_sim_mode_is_supported_in_effective_config(self) -> None:
        proc = self._run("sim", "--deterministic", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "sim")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 0)

    def test_gene_stats_in_option_round_trips_in_effective_config(self) -> None:
        proc = self._run(
            "gibbs",
            "--gene-stats-in",
            "tests/data/mody.gene.list",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertEqual(payload["options"]["gene_stats_in"], "tests/data/mody.gene.list")

    def test_positive_controls_only_requires_positive_controls_all_in(self) -> None:
        proc = self._run("gibbs", "--positive-controls-list", "INS")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Specified positive controls without --positive-controls-all-in", err)

    def test_config_removed_gene_bfs_key_has_replacement_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"gene_bfs_in": "dummy.txt"}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'gene_bfs_in' has been removed", err)

    def test_config_gene_set_stats_column_requires_gene_set_stats_in(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"gene_set_stats_beta_col": "beta"}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Option --gene-set-stats-beta-col requires --gene-set-stats-in", err)

    def test_config_removed_gene_zs_key_has_removed_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"gene_zs_in": "dummy.txt"}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'gene_zs_in' has been removed", err)
            self.assertIn("is no longer supported", err)

    def test_config_removed_sigma_key_has_removed_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"chisq_threshold": 5}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'chisq_threshold' has been removed", err)
            self.assertIn("is no longer supported", err)

    def test_config_removed_run_gls_key_has_removed_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"run_gls": True}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'run_gls' has been removed", err)
            self.assertIn("is no longer supported", err)

    def test_config_removed_min_post_burn_alias_has_replacement_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"min_post_burn_in": 50}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'min_post_burn_in' has been removed", err)
            self.assertIn("use 'min_num_post_burn_in' (CLI: --min-num-post-burn-in) instead", err)

    def test_config_removed_min_num_iter_alias_has_replacement_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"min_num_iter": 50}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'min_num_iter' has been removed", err)
            self.assertIn("use 'min_num_post_burn_in' (CLI: --min-num-post-burn-in) instead", err)

    def test_factor_mode_disabled_in_pigean(self) -> None:
        proc = self._run("factor")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Mode 'factor' is not available in pigean.py after repository split", err)

    def test_factor_output_flag_disabled_in_pigean(self) -> None:
        proc = self._run("gibbs", "--factors-out", "factors.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --factors-out moved to eaggl.py after repository split", err)

    def test_config_factor_option_disabled_in_pigean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"anchor_genes": ["INS"]}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'anchor_genes' moved to eaggl.py after repository split", err)


if __name__ == "__main__":
    unittest.main()
