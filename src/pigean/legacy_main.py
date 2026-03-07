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


def __getattr__(name):
    return getattr(_legacy_main, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(_legacy_main)))


__all__ = [
    "BetaStageResult",
    "PriorsStageResult",
    "GibbsStageResult",
    "GibbsStageConfig",
    "NonHugePipelineResult",
    "MainPipelineResult",
    "run_main_pipeline",
    "main",
] + [name for name in vars(_legacy_main) if not name.startswith("__")]
