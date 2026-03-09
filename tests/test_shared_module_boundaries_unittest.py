from __future__ import annotations

import importlib
import json
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
        self.assertIn("def configure_hyperparameters_for_main(", runtime_source)
        self.assertIn("def temporary_state_fields(state, overrides, restore_fields):", runtime_source)
        self.assertIn("from pigean import runtime as pigean_runtime", flat_source)
        self.assertIn("return pigean_runtime.build_runtime_state(PigeanState, _options)", flat_source)
        self.assertIn("_configure_hyperparameters_for_main = functools.partial(", flat_source)
        self.assertIn("pigean_runtime.configure_hyperparameters_for_main", flat_source)
        self.assertIn("_temporary_state_fields = pigean_runtime.temporary_state_fields", flat_source)
        self.assertNotIn("def _snapshot_state_fields(", flat_source)
        self.assertNotIn("def _restore_state_fields(", flat_source)
        self.assertNotIn("def _temporary_state_fields(", flat_source)
        self.assertNotIn("def _configure_hyperparameters_for_main(", flat_source)

    def test_pigean_y_input_contract_and_dispatch_live_in_package_module(self) -> None:
        y_source = (REPO_ROOT / "src" / "pigean" / "y_inputs.py").read_text(encoding="utf-8")
        y_core_source = (REPO_ROOT / "src" / "pigean" / "y_inputs_core.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("class YPrimaryInputsContract:", y_source)
        self.assertIn("class YReadContract:", y_source)
        self.assertIn("def build_main_y_read_contract(options):", y_source)
        self.assertIn("def load_main_y_inputs(", y_source)
        self.assertIn("def read_y_pipeline(", y_core_source)
        self.assertIn("def set_const_Y(", y_core_source)
        self.assertIn("def read_primary_y_source(", y_core_source)
        self.assertIn("def materialize_y_on_gene_universe(", y_core_source)
        self.assertIn("from pigean import y_inputs as pigean_y_inputs", flat_source)
        self.assertIn("from pigean import y_inputs_core as pigean_y_inputs_core", flat_source)
        self.assertIn("YPrimaryInputsContract = pigean_y_inputs.YPrimaryInputsContract", flat_source)
        self.assertIn("_read_Y = functools.partial(", flat_source)
        self.assertIn("_set_const_Y = pigean_y_inputs_core.set_const_Y", flat_source)
        self.assertIn("pigean_y_inputs_core.read_y_pipeline", flat_source)
        self.assertIn("return pigean_y_inputs.load_main_y_inputs(", flat_source)
        self.assertNotIn("class YPrimaryInputsContract:", flat_source)
        self.assertNotIn("class YReadContract:", flat_source)
        self.assertNotIn("def _read_Y(", flat_source)
        self.assertNotIn("def _set_const_Y(", flat_source)
        self.assertNotIn("def _read_primary_y_source(", flat_source)
        self.assertNotIn("def _materialize_y_on_gene_universe(", flat_source)
        self.assertNotIn("def _initialize_y_from_new_gene_universe(", flat_source)
        self.assertNotIn("def _merge_y_into_existing_gene_universe(", flat_source)

    def test_pigean_x_input_orchestration_lives_in_package_module(self) -> None:
        x_source = (REPO_ROOT / "src" / "pigean" / "x_inputs.py").read_text(encoding="utf-8")
        x_core_source = (REPO_ROOT / "src" / "pigean" / "x_inputs_core.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("def run_main_adaptive_read_x(", x_source)
        self.assertIn("def run_read_x_stage(", x_source)
        self.assertIn("def read_x_pipeline(", x_source)
        self.assertIn("def process_x_input_file(", x_core_source)
        self.assertIn("def maybe_prefilter_x_block(", x_core_source)
        self.assertIn("def ensure_gene_universe_for_x(", x_core_source)
        self.assertIn("from pigean import x_inputs as pigean_x_inputs", flat_source)
        self.assertIn("from pigean import x_inputs_core as pigean_x_inputs_core", flat_source)
        self.assertIn("return pigean_x_inputs.run_main_adaptive_read_x(", flat_source)
        self.assertIn("return pigean_x_inputs.run_read_x_stage(", flat_source)
        self.assertIn("return pigean_x_inputs.read_x_pipeline(", flat_source)
        self.assertIn("_process_x_input_file = functools.partial(", flat_source)
        self.assertIn("pigean_x_inputs_core.process_x_input_file", flat_source)
        self.assertNotIn("def _process_x_input_file(", flat_source)
        self.assertNotIn("def _normalize_dense_gene_rows(", flat_source)
        self.assertNotIn("def _build_sparse_x_from_dense_input(", flat_source)
        self.assertNotIn("def _normalize_gene_set_weights(", flat_source)
        self.assertNotIn("def _maybe_prefilter_x_block(", flat_source)
        self.assertNotIn("def _ensure_gene_universe_for_x(", flat_source)

    def test_pigean_gibbs_orchestration_lives_in_package_module(self) -> None:
        gibbs_source = (REPO_ROOT / "src" / "pigean" / "gibbs.py").read_text(encoding="utf-8")
        callback_source = (REPO_ROOT / "src" / "pigean" / "gibbs_callbacks.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        shim_source = (REPO_ROOT / "src" / "pigean_gibbs.py").read_text(encoding="utf-8")
        self.assertIn("class GibbsOrchestrationCallbacks:", gibbs_source)
        self.assertIn("def run_outer_gibbs(", gibbs_source)
        self.assertIn("def build_gibbs_callbacks(", callback_source)
        self.assertIn("from pigean import gibbs as pigean_gibbs", flat_source)
        self.assertIn("from pigean import gibbs_callbacks as pigean_gibbs_callbacks", flat_source)
        self.assertIn("callbacks = pigean_gibbs_callbacks.build_gibbs_callbacks(", flat_source)
        self.assertIn("return pigean_gibbs.run_outer_gibbs(", flat_source)
        self.assertIn("from pigean.gibbs import *", shim_source)
        self.assertNotIn("def _prepare_gibbs_run_inputs(", flat_source)
        self.assertNotIn("def _new_gibbs_epoch_aggregates(", flat_source)
        self.assertNotIn("def _reset_gibbs_diagnostics(", flat_source)
        self.assertNotIn("def _start_gibbs_epoch(", flat_source)
        self.assertNotIn("def _build_gibbs_epoch_finalize_context(", flat_source)
        self.assertNotIn("def _finalize_gibbs_epoch_attempt(", flat_source)
        self.assertNotIn("def _prepare_gibbs_iteration_state(", flat_source)
        self.assertNotIn("def _run_gibbs_iteration_correction_and_updates(", flat_source)
        self.assertNotIn("def _advance_gibbs_iteration_progress(", flat_source)
        self.assertNotIn("def _build_gibbs_epoch_runtime_configs(", flat_source)
        self.assertNotIn("def _build_gibbs_epoch_runtime_config_inputs(", flat_source)
        self.assertNotIn("def _build_gibbs_dynamic_runtime_inputs(", flat_source)
        self.assertNotIn("def _build_gibbs_epoch_iteration_loop_config(", flat_source)
        self.assertNotIn("def _build_gibbs_iteration_runtime_configs(", flat_source)
        self.assertNotIn("def _build_gibbs_log_bf_payload(", flat_source)
        self.assertNotIn("def _build_gibbs_epoch_attempt_result(", flat_source)
        self.assertNotIn("def _run_gibbs_epoch_phase(", flat_source)
        self.assertNotIn("def _run_gibbs_epochs_with_optional_traces(", flat_source)
        self.assertNotIn("def _run_gibbs_epoch_iterations(", flat_source)
        self.assertNotIn("def _build_gibbs_iteration_correction_context(", flat_source)
        self.assertNotIn("def _build_gibbs_iteration_finalize_context(", flat_source)
        self.assertNotIn("def _build_gibbs_iteration_progress_update_context(", flat_source)
        self.assertNotIn("def _run_single_gibbs_iteration(", flat_source)
        self.assertNotIn("def _finalize_gibbs_iteration_after_correction(", flat_source)

    def test_pigean_beta_and_prior_math_live_in_package_model_module(self) -> None:
        model_source = (REPO_ROOT / "src" / "pigean" / "model.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "pigean_legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("def build_inner_beta_sampler_common_kwargs(", model_source)
        self.assertIn("def calc_priors_from_betas(", model_source)
        self.assertIn("def finalize_gibbs_priors_for_sampling(", model_source)
        self.assertIn("def compute_gibbs_uncorrected_betas_and_defaults(", model_source)
        self.assertIn("def build_non_inf_beta_sampler_kwargs(", model_source)
        self.assertIn("def compute_gibbs_iteration_priors_from_betas(", model_source)
        self.assertIn("def calculate_gene_set_statistics(", model_source)
        self.assertIn("def calculate_non_inf_betas(", model_source)
        self.assertIn("def calculate_priors(", model_source)
        self.assertIn("from pigean import model as pigean_model", flat_source)
        self.assertIn("_build_inner_beta_sampler_common_kwargs = pigean_model.build_inner_beta_sampler_common_kwargs", flat_source)
        self.assertIn("pigean_model.finalize_gibbs_priors_for_sampling(", flat_source)
        self.assertIn("pigean_model.compute_gibbs_uncorrected_betas_and_defaults(", flat_source)
        self.assertIn("pigean_model.build_non_inf_beta_sampler_kwargs(", flat_source)
        self.assertIn("pigean_model.compute_gibbs_iteration_priors_from_betas(", flat_source)
        self.assertIn("return pigean_model.calculate_gene_set_statistics(", flat_source)
        self.assertIn("return pigean_model.calculate_non_inf_betas(", flat_source)
        self.assertIn("return pigean_model.calculate_priors(", flat_source)
        self.assertNotIn("def _build_inner_beta_sampler_common_kwargs(", flat_source)
        self.assertNotIn("def _calc_priors_from_betas(", flat_source)
        self.assertNotIn("def _finalize_gibbs_priors_for_sampling(", flat_source)
        self.assertNotIn("def _compute_gibbs_uncorrected_betas_and_defaults(", flat_source)
        self.assertNotIn("def _build_non_inf_beta_sampler_kwargs(", flat_source)
        self.assertNotIn("def _compute_gibbs_iteration_priors_from_betas(", flat_source)
        self.assertNotIn("def calculate_gene_set_statistics(self", flat_source.split("return pigean_model.calculate_gene_set_statistics(", 1)[1])
        self.assertNotIn("def calculate_non_inf_betas(self", flat_source.split("return pigean_model.calculate_non_inf_betas(", 1)[1])
        self.assertNotIn("def calculate_priors(self", flat_source.split("return pigean_model.calculate_priors(", 1)[1])

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
        self.assertIn("from pegs_shared.gene_io import", pigean_source)
        self.assertIn("from pegs_shared.huge_cache import", pigean_source)
        self.assertIn("from pegs_shared.xdata import", pigean_source)
        self.assertIn("from pegs_shared.ydata import", pigean_source)
        self.assertIn("from pegs_shared.bundle import", pigean_source)
        self.assertIn("from pegs_shared.phewas import", pigean_source)
        self.assertIn("from pegs_shared.types import", eaggl_source)
        self.assertIn("from pegs_shared.cli import", eaggl_source)
        self.assertIn("from pegs_shared.gene_io import", eaggl_source)
        self.assertIn("from pegs_shared.xdata import", eaggl_source)
        self.assertIn("from pegs_shared.ydata import", eaggl_source)
        self.assertIn("from pegs_shared.bundle import", eaggl_source)
        self.assertIn("from pegs_shared.phewas import", eaggl_source)

    def test_pegs_shared_gene_io_and_huge_cache_own_extracted_helpers(self) -> None:
        gene_io_source = (REPO_ROOT / "src" / "pegs_shared" / "gene_io.py").read_text(encoding="utf-8")
        huge_cache_source = (REPO_ROOT / "src" / "pegs_shared" / "huge_cache.py").read_text(encoding="utf-8")
        phewas_source = (REPO_ROOT / "src" / "pegs_shared" / "phewas.py").read_text(encoding="utf-8")
        pegs_utils_source = (REPO_ROOT / "src" / "pegs_utils.py").read_text(encoding="utf-8")
        self.assertIn("def parse_gene_set_statistics_file(", gene_io_source)
        self.assertIn("def parse_gene_bfs_file(", gene_io_source)
        self.assertIn("def parse_gene_covariates_file(", gene_io_source)
        self.assertIn("def load_aligned_gene_bfs(", gene_io_source)
        self.assertIn("def load_aligned_gene_covariates(", gene_io_source)
        self.assertIn("def coerce_runtime_state_dict(", huge_cache_source)
        self.assertIn("def get_huge_statistics_paths_for_prefix(", huge_cache_source)
        self.assertIn("def build_huge_statistics_meta(", huge_cache_source)
        self.assertIn("def write_huge_statistics_text_tables(", huge_cache_source)
        self.assertIn("def load_huge_statistics_sparse_and_vectors(", huge_cache_source)
        self.assertIn("def resolve_phewas_file_columns(", phewas_source)
        self.assertIn("def prepare_phewas_phenos_from_file(", phewas_source)
        self.assertIn("def read_phewas_file_batch(", phewas_source)
        self.assertIn("def parse_gene_phewas_bfs_file(", phewas_source)
        self.assertIn("from pegs_shared.gene_io import", pegs_utils_source)
        self.assertIn("from pegs_shared.huge_cache import", pegs_utils_source)
        self.assertIn("from pegs_shared.phewas import", pegs_utils_source)
        self.assertNotIn("def parse_gene_set_statistics_file(", pegs_utils_source)
        self.assertNotIn("def parse_gene_bfs_file(", pegs_utils_source)
        self.assertNotIn("def parse_gene_covariates_file(", pegs_utils_source)
        self.assertNotIn("def load_aligned_gene_bfs(", pegs_utils_source)
        self.assertNotIn("def load_aligned_gene_covariates(", pegs_utils_source)
        self.assertNotIn("def coerce_runtime_state_dict(", pegs_utils_source)
        self.assertNotIn("def get_huge_statistics_paths_for_prefix(", pegs_utils_source)
        self.assertNotIn("def build_huge_statistics_meta(", pegs_utils_source)
        self.assertNotIn("def write_huge_statistics_text_tables(", pegs_utils_source)
        self.assertNotIn("def resolve_phewas_file_columns(", pegs_utils_source)
        self.assertNotIn("def prepare_phewas_phenos_from_file(", pegs_utils_source)
        self.assertNotIn("def read_phewas_file_batch(", pegs_utils_source)
        self.assertNotIn("def parse_gene_phewas_bfs_file(", pegs_utils_source)

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

    def test_eaggl_factor_and_phewas_runtime_methods_live_in_package_modules(self) -> None:
        factor_runtime_source = (REPO_ROOT / "src" / "eaggl" / "factor_runtime.py").read_text(encoding="utf-8")
        phewas_source = (REPO_ROOT / "src" / "eaggl" / "phewas.py").read_text(encoding="utf-8")
        flat_source = (REPO_ROOT / "src" / "eaggl" / "legacy_main.py").read_text(encoding="utf-8")
        self.assertIn("def run_factor(", factor_runtime_source)
        self.assertIn("def build_phewas_input_values(", phewas_source)
        self.assertIn("def calculate_phewas_block(", phewas_source)
        self.assertIn("def run_phewas(", phewas_source)
        self.assertIn("from . import factor_runtime as _eaggl_factor_runtime", flat_source)
        self.assertIn("from . import phewas as _eaggl_phewas", flat_source)
        self.assertIn("return _eaggl_factor_runtime.run_factor(", flat_source)
        self.assertIn("return _eaggl_phewas.build_phewas_input_values(", flat_source)
        self.assertIn("return _eaggl_phewas.calculate_phewas_block(", flat_source)
        self.assertIn("return _eaggl_phewas.prepare_phewas_phenos_from_file(", flat_source)
        self.assertIn("return _eaggl_phewas.run_phewas(", flat_source)
        self.assertIn("return _eaggl_phewas.run_factor_phewas_batch(", flat_source)
        self.assertIn("return _eaggl_phewas.run_standard_phewas_batch(", flat_source)

    def test_pigean_methods_to_code_doc_points_at_package_modules(self) -> None:
        doc_source = (REPO_ROOT / "docs" / "pigean" / "METHODS_TO_CODE.md").read_text(encoding="utf-8")
        self.assertIn("docs/methods.tex", doc_source)
        self.assertIn("src/pigean/y_inputs.py", doc_source)
        self.assertIn("src/pigean/x_inputs.py", doc_source)
        self.assertIn("src/pigean/pipeline.py", doc_source)
        self.assertIn("src/pigean/gibbs.py", doc_source)
        self.assertIn("src/pigean/huge.py", doc_source)
        self.assertIn("src/pigean/outputs.py", doc_source)

    def test_package_roots_export_only_bounded_surface(self) -> None:
        pigean_init = (REPO_ROOT / "src" / "pigean" / "__init__.py").read_text(encoding="utf-8")
        eaggl_init = (REPO_ROOT / "src" / "eaggl" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("_PUBLIC_SUBMODULES = frozenset(", pigean_init)
        self.assertIn("_COMPAT_EXPORTS = {", pigean_init)
        self.assertNotIn("return getattr(_legacy_module(), name)", pigean_init)
        self.assertNotIn("sorted(set(globals().keys()) | set(dir(_legacy_module())))", pigean_init)
        self.assertIn("_PUBLIC_SUBMODULES = frozenset(", eaggl_init)
        self.assertIn("_COMPAT_EXPORTS = {", eaggl_init)
        self.assertNotIn("from .legacy_main import *", eaggl_init)

    def test_manifest_normal_visibility_options_have_summaries(self) -> None:
        for rel_path in ("docs/cli_option_manifest.json", "docs/eaggl/cli_option_manifest.json"):
            manifest = json.loads((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
            missing = [
                row["primary_flag"]
                for row in manifest["options"]
                if row.get("public_visibility") == "normal" and not (row.get("summary") or "").strip()
            ]
            self.assertEqual(missing, [], msg="%s missing summaries for: %s" % (rel_path, ", ".join(missing)))


if __name__ == "__main__":
    unittest.main()
