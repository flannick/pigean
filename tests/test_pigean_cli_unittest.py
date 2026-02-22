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

    def test_deterministic_sets_seed_zero(self) -> None:
        proc = self._run("gibbs", "--deterministic", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 0)

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

    def test_factor_mode_disabled_in_pigean(self) -> None:
        proc = self._run("factor")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Mode 'factor' is not available in pigean.py after repository split", err)


if __name__ == "__main__":
    unittest.main()
