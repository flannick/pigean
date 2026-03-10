from __future__ import annotations

import unittest
from pathlib import Path


class CanonicalLayoutTest(unittest.TestCase):
    def test_canonical_eaggl_snapshot_exists_in_pigean_repo(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected = [
            repo_root / "docs" / "CANONICAL_SOURCE.md",
            repo_root / "docs" / "eaggl" / "WORKFLOWS.md",
            repo_root / "docs" / "eaggl" / "TRANSITION.md",
            repo_root / "src" / "eaggl" / "app.py",
            repo_root / "src" / "eaggl" / "main_support.py",
            repo_root / "src" / "eaggl" / "state.py",
            repo_root / "scripts" / "eaggl" / "generate_cli_manifest.py",
            repo_root / "tests" / "eaggl" / "test_eaggl_cli_unittest.py",
            repo_root / "tests" / "data" / "reference" / "eaggl_factor_workflow_effective_config" / "README.md",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), msg=f"missing canonical artifact: {path}")


if __name__ == "__main__":
    unittest.main()
