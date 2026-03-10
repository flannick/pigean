from __future__ import annotations

from pegs_cli_errors import PegsCliError, handle_cli_exception, handle_unexpected_exception

from . import cli as pigean_cli
from . import dispatch as pigean_dispatch
from . import main_support as pigean_main_support
from . import x_inputs_core as pigean_x_inputs_core


def run_main_pipeline(options, mode, services=None):
    if services is None:
        services = pigean_main_support.build_cli_services()
    return pigean_dispatch.run_main_pipeline(options, mode, services=services)


def main(argv=None):
    try:
        should_continue = pigean_cli._bootstrap_cli(argv)
        if not should_continue:
            return 0
        run_main_pipeline(pigean_cli.options, pigean_cli.mode)
        return 0
    except PegsCliError as exc:
        return handle_cli_exception(exc, argv=argv, debug_level=pigean_cli.debug_level)
    except Exception as exc:
        return handle_unexpected_exception(exc, argv=argv, debug_level=pigean_cli.debug_level)


def _build_prefilter_keep_mask(*args, **kwargs):
    return pigean_x_inputs_core.build_prefilter_keep_mask(
        *args,
        log_fn=pigean_cli.log,
        debug_level=pigean_cli.DEBUG,
        **kwargs,
    )


__all__ = [
    "main",
    "run_main_pipeline",
    "_build_prefilter_keep_mask",
]
