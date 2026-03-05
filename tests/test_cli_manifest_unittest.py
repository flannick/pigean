from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class CliManifestTest(unittest.TestCase):
    def test_cli_manifest_and_docs_are_current(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, "scripts/generate_cli_manifest.py", "--check"]
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=(proc.stderr or "") + (proc.stdout or ""),
        )


if __name__ == "__main__":
    unittest.main()
