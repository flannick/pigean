from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class FetchBundlesTest(unittest.TestCase):
    def test_list_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            catalog = {
                "profiles": {"minimal": ["core_small"]},
                "modes": {"gene_list": ["core_small", "x_panels"]},
                "bundles": {
                    "core_small": {"version": "1", "url": "file:///tmp/core_small.tar.gz", "sha256": "x"},
                    "x_panels": {"version": "1", "url": "file:///tmp/x_panels.tar.gz", "sha256": "x"},
                },
            }
            cat = tdir / "catalog.json"
            cat.write_text(json.dumps(catalog), encoding="utf-8")

            repo_root = Path(__file__).resolve().parents[1]
            cmd = [
                "python3",
                "scripts/fetch_bundles.py",
                "--catalog",
                str(cat),
                "--profile",
                "minimal",
                "--mode",
                "gene_list",
                "--list",
            ]
            proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
            names = [line.split("\t")[0] for line in proc.stdout.strip().splitlines() if line.strip()]
            self.assertEqual(names, ["core_small", "x_panels"])


if __name__ == "__main__":
    unittest.main()
