from __future__ import annotations

from pegs_cli_errors import PegsCliError, handle_cli_exception, handle_unexpected_exception

from . import cli as eaggl_cli
from . import dispatch as eaggl_dispatch
from . import main_support as eaggl_main_support


def run_main_pipeline(options):
    return eaggl_dispatch.run_main_pipeline(eaggl_main_support.build_main_domain(), options)


def main(argv=None):
    try:
        should_continue = eaggl_cli._bootstrap_cli(argv)
        if not should_continue:
            return 0
        run_main_pipeline(eaggl_cli.options)
        return 0
    except PegsCliError as exc:
        return handle_cli_exception(exc, argv=argv, debug_level=eaggl_cli.debug_level)
    except Exception as exc:
        return handle_unexpected_exception(exc, argv=argv, debug_level=eaggl_cli.debug_level)


__all__ = [
    "main",
    "run_main_pipeline",
]
