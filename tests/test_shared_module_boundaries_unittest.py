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
            ("pigean.gibbs", "pigean_gibbs", "run_outer_gibbs"),
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
        self.assertIn("from pigean import dispatch as _pigean_dispatch", flat_source)
        self.assertIn("from pigean import huge as _pigean_huge", flat_source)
        self.assertIn("from pigean import phewas as _pigean_phewas", flat_source)
        self.assertNotIn("def _run_main_non_huge_pipeline", flat_source)
        self.assertNotIn("def _write_main_outputs_and_optional_phewas", flat_source)

    def test_pigean_runtime_helpers_live_in_package_runtime_module(self) -> None:
        runtime_source = (REPO_ROOT / "src" / "pigean" / "runtime.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("def build_runtime_state(state_cls, options):", runtime_source)
        self.assertIn("def temporary_state_fields(state, overrides, restore_fields):", runtime_source)
        self.assertIn("from pigean import runtime as pigean_runtime", flat_source)
        self.assertIn("return pigean_runtime.build_runtime_state(PigeanState, _options)", flat_source)
        self.assertIn("_temporary_state_fields = pigean_runtime.temporary_state_fields", flat_source)
        self.assertNotIn("def _snapshot_state_fields(", flat_source)
        self.assertNotIn("def _restore_state_fields(", flat_source)
        self.assertNotIn("def _temporary_state_fields(", flat_source)

    def test_pigean_y_input_contract_and_dispatch_live_in_package_module(self) -> None:
        y_source = (REPO_ROOT / "src" / "pigean" / "y_inputs.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("class YPrimaryInputsContract:", y_source)
        self.assertIn("class YReadContract:", y_source)
        self.assertIn("def build_main_y_read_contract(options):", y_source)
        self.assertIn("def load_main_y_inputs(", y_source)
        self.assertIn("from pigean import y_inputs as pigean_y_inputs", flat_source)
        self.assertIn("YPrimaryInputsContract = pigean_y_inputs.YPrimaryInputsContract", flat_source)
        self.assertIn("return pigean_y_inputs.load_main_y_inputs(", flat_source)
        self.assertNotIn("class YPrimaryInputsContract:", flat_source)
        self.assertNotIn("class YReadContract:", flat_source)

    def test_pigean_x_input_orchestration_lives_in_package_module(self) -> None:
        x_source = (REPO_ROOT / "src" / "pigean" / "x_inputs.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("def run_main_adaptive_read_x(", x_source)
        self.assertIn("def run_read_x_stage(", x_source)
        self.assertIn("def read_x_pipeline(", x_source)
        self.assertIn("from pigean import x_inputs as pigean_x_inputs", flat_source)
        self.assertIn("return pigean_x_inputs.run_main_adaptive_read_x(", flat_source)
        self.assertIn("return pigean_x_inputs.run_read_x_stage(", flat_source)
        self.assertIn("return pigean_x_inputs.read_x_pipeline(", flat_source)

    def test_pigean_gibbs_orchestration_lives_in_package_module(self) -> None:
        gibbs_source = (REPO_ROOT / "src" / "pigean" / "gibbs.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        shim_source = (REPO_ROOT / "src" / "pigean_gibbs.py").read_text(encoding="utf-8")
        self.assertIn("class GibbsOrchestrationCallbacks:", gibbs_source)
        self.assertIn("def run_outer_gibbs(", gibbs_source)
        self.assertIn("from pigean import gibbs as pigean_gibbs", flat_source)
        self.assertIn("return pigean_gibbs.run_outer_gibbs(", flat_source)
        self.assertIn("from pigean.gibbs import *", shim_source)

    def test_pigean_cli_uses_narrow_cli_helper_module(self) -> None:
        cli_source = (REPO_ROOT / "src" / "pigean" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("from pegs_shared.cli import", cli_source)
        self.assertNotIn("from .pegs_utils import", cli_source)
        self.assertNotIn("from pegs_utils import", cli_source)

    def test_flat_pigean_cli_module_is_a_package_shim(self) -> None:
        cli_source = (REPO_ROOT / "src" / "pigean_cli.py").read_text(encoding="utf-8")
        self.assertIn("from pigean.cli import *", cli_source)

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
        self.assertIn("from . import y_inputs as _eaggl_y_inputs", eaggl_source)
        self.assertIn("return _eaggl_dispatch.run_main_pipeline(_build_main_domain(), options)", eaggl_source)

    def test_eaggl_io_uses_pegs_shared_for_extracted_read_helpers(self) -> None:
        io_source = (REPO_ROOT / "src" / "eaggl" / "io.py").read_text(encoding="utf-8")
        self.assertIn("from pegs_shared.io_common import", io_source)
        self.assertIn("from pegs_shared.xdata import", io_source)
        self.assertIn("from . import y_inputs as eaggl_y_inputs", io_source)
        pegs_utils_import_block = io_source.split("from pegs_utils import", 1)[1].split(")\n", 1)[0]
        self.assertNotIn("build_read_x_pipeline_config", pegs_utils_import_block)
        self.assertNotIn("clean_chrom_name", pegs_utils_import_block)
        self.assertNotIn("construct_map_to_ind", pegs_utils_import_block)
        self.assertNotIn("parse_gene_map_file", pegs_utils_import_block)
        self.assertNotIn("read_loc_file_with_gene_map", pegs_utils_import_block)

    def test_eaggl_y_inputs_and_covariates_own_dense_read_y_logic(self) -> None:
        y_source = (REPO_ROOT / "src" / "eaggl" / "y_inputs.py").read_text(encoding="utf-8")
        cov_source = (REPO_ROOT / "src" / "eaggl" / "covariates.py").read_text(encoding="utf-8")
        io_source = (REPO_ROOT / "src" / "eaggl" / "io.py").read_text(encoding="utf-8")
        domain_source = (REPO_ROOT / "src" / "eaggl" / "domain.py").read_text(encoding="utf-8")
        self.assertIn("def read_y_pipeline(", y_source)
        self.assertIn("def _apply_hold_out_chrom(", y_source)
        self.assertIn("def apply_loaded_gene_covariates(", cov_source)
        self.assertIn("return eaggl_y_inputs.read_y_pipeline(", io_source)
        self.assertIn("return eaggl_y_inputs.run_read_y_stage(self, runtime, **read_kwargs)", domain_source)

    def test_pigean_methods_to_code_doc_points_at_package_modules(self) -> None:
        doc_source = (REPO_ROOT / "docs" / "pigean" / "METHODS_TO_CODE.md").read_text(encoding="utf-8")
        self.assertIn("docs/methods.tex", doc_source)
        self.assertIn("src/pigean/y_inputs.py", doc_source)
        self.assertIn("src/pigean/x_inputs.py", doc_source)
        self.assertIn("src/pigean/pipeline.py", doc_source)
        self.assertIn("src/pigean/gibbs.py", doc_source)
        self.assertIn("src/pigean/huge.py", doc_source)
        self.assertIn("src/pigean/outputs.py", doc_source)


if __name__ == "__main__":
    unittest.main()
