from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PigeanCliTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, "src/pigean.py", *args]
        return subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)

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

    def test_help_usage_uses_pigean_name(self) -> None:
        proc = self._run("gibbs", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Usage: pigean.py", proc.stdout)

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
