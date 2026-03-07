from __future__ import annotations

import importlib
import pathlib
import sys
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
        src_root = str(REPO_ROOT / "src")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        for package_module_name, flat_module_name, symbol_name in seam_expectations:
            with self.subTest(module=package_module_name, symbol=symbol_name):
                package_module = importlib.import_module(package_module_name)
                flat_module = importlib.import_module(flat_module_name)
                self.assertIs(getattr(package_module, symbol_name), getattr(flat_module, symbol_name))
                self.assertEqual(getattr(package_module, symbol_name).__module__, package_module_name)

    def test_pigean_package_legacy_entrypoint_owns_main_dispatch(self) -> None:
        legacy_source = (REPO_ROOT / "src" / "pigean" / "legacy_main.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("from . import dispatch as pigean_dispatch", legacy_source)
        self.assertIn("def run_main_pipeline(options, mode):", legacy_source)
        self.assertIn("return pigean_dispatch.run_main_pipeline(_legacy_main, options, mode)", legacy_source)
        self.assertNotIn("def _run_main_non_huge_pipeline", flat_source)
        self.assertNotIn("def _write_main_outputs_and_optional_phewas", flat_source)

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

    def test_eaggl_io_uses_pegs_shared_for_extracted_read_helpers(self) -> None:
        io_source = (REPO_ROOT / "src" / "eaggl" / "io.py").read_text(encoding="utf-8")
        self.assertIn("from pegs_shared.io_common import", io_source)
        self.assertIn("from pegs_shared.xdata import", io_source)
        pegs_utils_import_block = io_source.split("from pegs_utils import", 1)[1].split(")\n", 1)[0]
        self.assertNotIn("build_read_x_pipeline_config", pegs_utils_import_block)
        self.assertNotIn("clean_chrom_name", pegs_utils_import_block)
        self.assertNotIn("construct_map_to_ind", pegs_utils_import_block)
        self.assertNotIn("parse_gene_map_file", pegs_utils_import_block)
        self.assertNotIn("read_loc_file_with_gene_map", pegs_utils_import_block)


if __name__ == "__main__":
    unittest.main()
