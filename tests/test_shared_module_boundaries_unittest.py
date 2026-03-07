from __future__ import annotations

import importlib
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class SharedModuleBoundaryTest(unittest.TestCase):
    def test_flat_pigean_modules_are_compatibility_shims_for_package_modules(self) -> None:
        seam_expectations = (
            ("pigean.pipeline", "pigean_pipeline", "run_main_non_huge_pipeline"),
            ("pigean.dispatch", "pigean_dispatch", "run_main_pipeline"),
            ("pigean.outputs", "pigean_outputs", "write_main_outputs_and_optional_phewas"),
            ("pigean.huge", "pigean_huge", "read_huge_statistics_bundle"),
            ("pigean.phewas", "pigean_phewas", "run_advanced_set_b_output_phewas_if_requested"),
        )
        for package_module_name, flat_module_name, symbol_name in seam_expectations:
            with self.subTest(module=package_module_name, symbol=symbol_name):
                package_module = importlib.import_module(package_module_name)
                flat_module = importlib.import_module(flat_module_name)
                self.assertIs(getattr(package_module, symbol_name), getattr(flat_module, symbol_name))
                self.assertEqual(getattr(package_module, symbol_name).__module__, package_module_name)

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
