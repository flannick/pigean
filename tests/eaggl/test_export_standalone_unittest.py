from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class ExportStandaloneEagglTest(unittest.TestCase):
    def test_export_script_copies_canonical_eaggl_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "eaggl"
            (target / "src").mkdir(parents=True, exist_ok=True)
            (target / "docs").mkdir(parents=True, exist_ok=True)
            (target / "tests").mkdir(parents=True, exist_ok=True)
            (target / "scripts").mkdir(parents=True, exist_ok=True)

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/eaggl/export_standalone_eaggl.py",
                    str(target),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))

            self.assertTrue((target / "src" / "eaggl" / "__main__.py").exists())
            self.assertTrue((target / "src" / "pegs_utils.py").exists())
            self.assertTrue((target / "docs" / "WORKFLOWS.md").exists())
            self.assertTrue((target / "tests" / "eaggl" / "test_labeling_unittest.py").exists())
            self.assertTrue((target / "scripts" / "generate_cli_manifest.py").exists())


if __name__ == "__main__":
    unittest.main()
