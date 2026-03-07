from __future__ import annotations

import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class SharedModuleBoundaryTest(unittest.TestCase):
    def test_pigean_cli_uses_narrow_cli_helper_module(self) -> None:
        cli_source = (REPO_ROOT / "src" / "pigean_cli.py").read_text(encoding="utf-8")
        self.assertIn("from pegs_shared.cli import", cli_source)
        self.assertNotIn("from .pegs_utils import", cli_source)
        self.assertNotIn("from pegs_utils import", cli_source)

    def test_eaggl_cli_uses_narrow_cli_helper_module(self) -> None:
        cli_source = (REPO_ROOT / "src" / "eaggl" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("from pegs_shared.cli import", cli_source)
        self.assertNotIn("from .pegs_utils import", cli_source)
        self.assertNotIn("from pegs_utils import", cli_source)

    def test_core_legacy_launchers_use_pegs_shared_modules(self) -> None:
        pigean_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        eaggl_source = (REPO_ROOT / "src" / "eaggl" / "legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("from pegs_shared.types import", pigean_source)
        self.assertIn("from pegs_shared.cli import", pigean_source)
        self.assertIn("from pegs_shared.xdata import", pigean_source)
        self.assertIn("from pegs_shared.ydata import", pigean_source)
        self.assertIn("from pegs_shared.bundle import", pigean_source)
        self.assertIn("from pegs_shared.phewas import", pigean_source)
        self.assertIn("from pegs_shared.types import", eaggl_source)
        self.assertIn("from pegs_shared.cli import", eaggl_source)
        self.assertIn("from pegs_shared.xdata import", eaggl_source)
        self.assertIn("from pegs_shared.ydata import", eaggl_source)
        self.assertIn("from pegs_shared.bundle import", eaggl_source)
        self.assertIn("from pegs_shared.phewas import", eaggl_source)

    def test_eaggl_legacy_main_uses_package_domain_and_io_layers(self) -> None:
        eaggl_source = (REPO_ROOT / "src" / "eaggl" / "legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("from . import domain as _eaggl_domain", eaggl_source)
        self.assertIn("from . import io as _eaggl_io", eaggl_source)
        self.assertIn("return _eaggl_dispatch.run_main_pipeline(_build_main_domain(), options)", eaggl_source)


if __name__ == "__main__":
    unittest.main()
