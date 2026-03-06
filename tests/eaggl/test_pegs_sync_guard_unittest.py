from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pegs_sync_guard  # noqa: E402


class PegsSyncGuardTest(unittest.TestCase):
    def test_compare_shared_files_detects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            left = Path(td) / "left"
            right = Path(td) / "right"
            (left / "src").mkdir(parents=True, exist_ok=True)
            (right / "src").mkdir(parents=True, exist_ok=True)
            rel = "src/pegs_utils_phewas.py"
            (left / rel).write_text("a\n", encoding="utf-8")
            (right / rel).write_text("b\n", encoding="utf-8")
            result = pegs_sync_guard.compare_shared_files(left, right, files=[rel])
            self.assertFalse(result.ok)
            self.assertEqual(result.mismatched, [rel])

    def test_compare_shared_files_detects_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            left = Path(td) / "left"
            right = Path(td) / "right"
            (left / "src").mkdir(parents=True, exist_ok=True)
            (right / "src").mkdir(parents=True, exist_ok=True)
            rel = "src/pegs_utils_bundle.py"
            (left / rel).write_text("x\n", encoding="utf-8")
            result = pegs_sync_guard.compare_shared_files(left, right, files=[rel])
            self.assertFalse(result.ok)
            self.assertEqual(result.missing_in_right, [rel])

    def test_compare_shared_files_with_sibling_repo_if_present(self) -> None:
        sibling = REPO_ROOT.parent / "pigean"
        if not sibling.exists():
            self.skipTest("sibling pigean repo not present")
        result = pegs_sync_guard.compare_shared_files(
            REPO_ROOT,
            sibling,
            files=pegs_sync_guard.DEFAULT_SHARED_FILES,
        )
        self.assertTrue(result.ok, msg=result.summary())


if __name__ == "__main__":
    unittest.main()
