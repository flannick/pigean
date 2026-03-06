from __future__ import annotations

import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class SharedModuleBoundaryTest(unittest.TestCase):
    def test_pigean_cli_uses_narrow_cli_helper_module(self) -> None:
        cli_source = (REPO_ROOT / "src" / "pigean_cli.py").read_text(encoding="utf-8")
        self.assertIn("from .pegs_cli_utils import", cli_source)
        self.assertNotIn("from .pegs_utils import", cli_source)
        self.assertNotIn("from pegs_utils import", cli_source)

    def test_eaggl_cli_uses_narrow_cli_helper_module(self) -> None:
        cli_source = (REPO_ROOT / "src" / "eaggl" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("from pegs_cli_utils import", cli_source)
        self.assertNotIn("from .pegs_utils import", cli_source)
        self.assertNotIn("from pegs_utils import", cli_source)


if __name__ == "__main__":
    unittest.main()
