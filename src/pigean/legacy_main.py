from __future__ import annotations

import importlib

from . import dispatch as pigean_dispatch
from . import pipeline as pigean_pipeline


_legacy_main = importlib.import_module("pigean_legacy_main")

BetaStageResult = pigean_pipeline.BetaStageResult
PriorsStageResult = pigean_pipeline.PriorsStageResult
GibbsStageResult = pigean_pipeline.GibbsStageResult
GibbsStageConfig = pigean_pipeline.GibbsStageConfig
NonHugePipelineResult = pigean_pipeline.NonHugePipelineResult
MainPipelineResult = pigean_pipeline.MainPipelineResult

_COMPAT_EXPORTS = {
    "_build_prefilter_keep_mask": "_build_prefilter_keep_mask",
}


def run_main_pipeline(options, mode):
    return pigean_dispatch.run_main_pipeline(_legacy_main, options, mode)


def main(argv=None):
    try:
        should_continue = _legacy_main._bootstrap_cli(argv)
        if not should_continue:
            return 0
        run_main_pipeline(_legacy_main.options, _legacy_main.mode)
        return 0
    except _legacy_main.PegsCliError as exc:
        return _legacy_main.pegs_handle_cli_exception(exc, argv=argv, debug_level=_legacy_main.debug_level)
    except Exception as exc:
        return _legacy_main.pegs_handle_unexpected_exception(exc, argv=argv, debug_level=_legacy_main.debug_level)


def _build_prefilter_keep_mask(*args, **kwargs):
    return _legacy_main._build_prefilter_keep_mask(*args, **kwargs)


def __getattr__(name):
    if name not in _COMPAT_EXPORTS:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    return getattr(_legacy_main, _COMPAT_EXPORTS[name])


def __dir__():
    return sorted(set(globals().keys()) | set(_COMPAT_EXPORTS.keys()))


__all__ = [
    "BetaStageResult",
    "PriorsStageResult",
    "GibbsStageResult",
    "GibbsStageConfig",
    "NonHugePipelineResult",
    "MainPipelineResult",
    "run_main_pipeline",
    "main",
    "_build_prefilter_keep_mask",
]
