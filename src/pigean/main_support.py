from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

import numpy as np
import scipy

import pegs_shared.bundle as pegs_bundle
import pegs_shared.gene_io as pegs_gene_io
import pegs_shared.phewas as pegs_phewas
import pegs_utils as pegs_utils_mod

from . import cli as pigean_cli


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


@dataclass(frozen=True)
class PigeanMainServices:
    sys: object
    np: object
    scipy: object
    NONE: int
    INFO: int
    DEBUG: int
    TRACE: int
    log_fn: object
    warn_fn: object
    bail_fn: object

    def log(self, *args, **kwargs):
        return self.log_fn(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return self.warn_fn(*args, **kwargs)

    def bail(self, message):
        return self.bail_fn(message)


_LEGACY_CORE = None


def _sync_legacy_cli_state(legacy_module) -> None:
    for field_name in _LEGACY_STATE_FIELDS:
        setattr(legacy_module, field_name, getattr(pigean_cli, field_name))


def load_legacy_core():
    global _LEGACY_CORE
    if _LEGACY_CORE is None:
        _LEGACY_CORE = importlib.import_module("pigean_legacy_main")
        _sync_legacy_cli_state(_LEGACY_CORE)
    return _LEGACY_CORE


def build_cli_services() -> PigeanMainServices:
    legacy_core = load_legacy_core()
    return PigeanMainServices(
        sys=sys,
        np=np,
        scipy=scipy,
        NONE=pigean_cli.NONE,
        INFO=pigean_cli.INFO,
        DEBUG=pigean_cli.DEBUG,
        TRACE=pigean_cli.TRACE,
        log_fn=pigean_cli.log,
        warn_fn=pigean_cli.warn,
        bail_fn=lambda message: (_ for _ in ()).throw(legacy_core.DataValidationError(message)),
    )


def build_legacy_services(legacy_module) -> PigeanMainServices:
    return PigeanMainServices(
        sys=legacy_module.sys,
        np=legacy_module.np,
        scipy=legacy_module.scipy,
        NONE=legacy_module.NONE,
        INFO=legacy_module.INFO,
        DEBUG=legacy_module.DEBUG,
        TRACE=legacy_module.TRACE,
        log_fn=legacy_module.log,
        warn_fn=legacy_module.warn,
        bail_fn=legacy_module.bail,
    )


def build_mode_state(mode, run_phewas_from_gene_phewas_stats_in):
    return pigean_cli._build_mode_state(mode, run_phewas_from_gene_phewas_stats_in)


def build_runtime_state(options):
    return load_legacy_core()._build_runtime_state(options)


def configure_hyperparameters_for_main(state, options):
    return load_legacy_core()._configure_hyperparameters_for_main(state, options)


def load_main_y_inputs(state, options, mode_state):
    return load_legacy_core()._load_main_Y_inputs(state, options, mode_state)


def build_inner_beta_sampler_common_kwargs(options):
    return load_legacy_core()._build_inner_beta_sampler_common_kwargs(options)


def run_main_adaptive_read_x(state, options, mode_state, sigma2_cond):
    return load_legacy_core()._run_main_adaptive_read_x(state, options, mode_state, sigma2_cond)


def set_const_Y(state, const_gene_Y):
    return load_legacy_core()._set_const_Y(state, const_gene_Y)


def open_gz(*args, **kwargs):
    return load_legacy_core().open_gz(*args, **kwargs)


def get_col(*args, **kwargs):
    return load_legacy_core()._get_col(*args, **kwargs)


def read_gene_phewas_bfs(*args, **kwargs):
    return load_legacy_core()._read_gene_phewas_bfs(*args, **kwargs)


def load_and_apply_gene_set_statistics_to_runtime(*args, **kwargs):
    return pegs_utils_mod.load_and_apply_gene_set_statistics_to_runtime(*args, **kwargs)


resolve_gene_phewas_input_decision_for_stage = pegs_phewas.resolve_gene_phewas_input_decision_for_stage
build_phewas_stage_config = pegs_phewas.build_phewas_stage_config
write_bundle_from_specs = pegs_bundle.write_bundle_from_specs
EAGGL_BUNDLE_SCHEMA = pegs_bundle.EAGGL_BUNDLE_SCHEMA
load_aligned_gene_bfs = pegs_gene_io.load_aligned_gene_bfs
