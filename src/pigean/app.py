from __future__ import annotations

import importlib
import sys

import numpy as np
import scipy

from pegs_cli_errors import PegsCliError, handle_cli_exception, handle_unexpected_exception
import pegs_shared.bundle as pegs_bundle
import pegs_shared.gene_io as pegs_gene_io
import pegs_shared.phewas as pegs_phewas
import pegs_utils as pegs_utils_mod

from . import cli as pigean_cli
from . import dispatch as pigean_dispatch


_LEGACY_STATE_FIELDS = (
    "options",
    "args",
    "mode",
    "config_mode",
    "cli_specified_dests",
    "config_specified_dests",
    "NONE",
    "INFO",
    "DEBUG",
    "TRACE",
    "debug_level",
    "log_fh",
    "warnings_fh",
    "log",
    "warn",
)


class PigeanAppDomain:
    sys = sys
    np = np
    scipy = scipy

    pegs_load_and_apply_gene_set_statistics_to_runtime = staticmethod(
        pegs_utils_mod.load_and_apply_gene_set_statistics_to_runtime
    )
    pegs_resolve_gene_phewas_input_decision_for_stage = staticmethod(
        pegs_phewas.resolve_gene_phewas_input_decision_for_stage
    )
    pegs_build_phewas_stage_config = staticmethod(pegs_phewas.build_phewas_stage_config)
    pegs_write_bundle_from_specs = staticmethod(pegs_bundle.write_bundle_from_specs)
    PEGS_EAGGL_BUNDLE_SCHEMA = pegs_bundle.EAGGL_BUNDLE_SCHEMA

    @property
    def NONE(self):
        return pigean_cli.NONE

    @property
    def INFO(self):
        return pigean_cli.INFO

    @property
    def DEBUG(self):
        return pigean_cli.DEBUG

    @property
    def TRACE(self):
        return pigean_cli.TRACE

    def log(self, *args, **kwargs):
        return pigean_cli.log(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return pigean_cli.warn(*args, **kwargs)

    def bail(self, message):
        raise _load_legacy_core().DataValidationError(message)

    def _build_mode_state(self, mode, run_phewas_from_gene_phewas_stats_in):
        return pigean_cli._build_mode_state(mode, run_phewas_from_gene_phewas_stats_in)

    def _build_runtime_state(self, options):
        return _load_legacy_core()._build_runtime_state(options)

    def _configure_hyperparameters_for_main(self, state, options):
        return _load_legacy_core()._configure_hyperparameters_for_main(state, options)

    def _load_main_Y_inputs(self, state, options, mode_state):
        return _load_legacy_core()._load_main_Y_inputs(state, options, mode_state)

    def _build_inner_beta_sampler_common_kwargs(self, options):
        return _load_legacy_core()._build_inner_beta_sampler_common_kwargs(options)

    def _run_main_adaptive_read_x(self, state, options, mode_state, sigma2_cond):
        return _load_legacy_core()._run_main_adaptive_read_x(state, options, mode_state, sigma2_cond)

    def _set_const_Y(self, state, const_gene_Y):
        return _load_legacy_core()._set_const_Y(state, const_gene_Y)

    def open_gz(self, *args, **kwargs):
        return _load_legacy_core().open_gz(*args, **kwargs)

    def _get_col(self, *args, **kwargs):
        return _load_legacy_core()._get_col(*args, **kwargs)

    def _read_gene_phewas_bfs(self, *args, **kwargs):
        return _load_legacy_core()._read_gene_phewas_bfs(*args, **kwargs)


_APP_DOMAIN = PigeanAppDomain()
_LEGACY_CORE = None


def _sync_legacy_cli_state(legacy_module) -> None:
    for field_name in _LEGACY_STATE_FIELDS:
        setattr(legacy_module, field_name, getattr(pigean_cli, field_name))


def _load_legacy_core():
    global _LEGACY_CORE
    if _LEGACY_CORE is None:
        _LEGACY_CORE = importlib.import_module("pigean_legacy_main")
        _sync_legacy_cli_state(_LEGACY_CORE)
    return _LEGACY_CORE


def run_main_pipeline(options, mode):
    _load_legacy_core()
    return pigean_dispatch.run_main_pipeline(_APP_DOMAIN, options, mode)


def main(argv=None):
    try:
        should_continue = pigean_cli._bootstrap_cli(argv)
        if not should_continue:
            return 0
        _load_legacy_core()
        run_main_pipeline(pigean_cli.options, pigean_cli.mode)
        return 0
    except PegsCliError as exc:
        return handle_cli_exception(exc, argv=argv, debug_level=pigean_cli.debug_level)
    except Exception as exc:
        return handle_unexpected_exception(exc, argv=argv, debug_level=pigean_cli.debug_level)


def _build_prefilter_keep_mask(*args, **kwargs):
    return _load_legacy_core()._build_prefilter_keep_mask(*args, **kwargs)


__all__ = [
    "main",
    "run_main_pipeline",
    "_build_prefilter_keep_mask",
]
