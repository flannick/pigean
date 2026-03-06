from __future__ import annotations

import optparse
import sys
import traceback


class PegsCliError(Exception):
    exit_code = 1

    def __init__(self, message, *, exit_code=None):
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class CliUsageError(PegsCliError):
    exit_code = 2


class CliConfigError(PegsCliError):
    exit_code = 2


class DataValidationError(PegsCliError):
    exit_code = 1


class CliOptionParser(optparse.OptionParser):
    def error(self, msg):
        raise CliUsageError(msg)


def _coerce_argv(argv):
    if argv is None:
        return list(sys.argv[1:])
    return list(argv)


def _argv_debug_level(argv):
    argv_list = _coerce_argv(argv)
    for index, arg in enumerate(argv_list):
        if not isinstance(arg, str):
            continue
        if arg.startswith("--debug-level="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
        if arg == "--debug-level" and index + 1 < len(argv_list):
            try:
                return int(argv_list[index + 1])
            except ValueError:
                return None
    return None


def should_show_traceback(*, argv=None, debug_level=None):
    if debug_level is not None:
        return debug_level >= 2
    argv_level = _argv_debug_level(argv)
    return argv_level is not None and argv_level >= 2


def format_cli_error_message(message):
    text = str(message)
    if text.startswith("Error:"):
        return text
    return "Error: %s" % text


def handle_cli_exception(exc, *, argv=None, debug_level=None, stderr=None):
    if stderr is None:
        stderr = sys.stderr
    if should_show_traceback(argv=argv, debug_level=debug_level):
        traceback.print_exc(file=stderr)
    else:
        stderr.write("%s\n" % format_cli_error_message(exc))
        stderr.flush()
    return getattr(exc, "exit_code", 1)


def handle_unexpected_exception(exc, *, argv=None, debug_level=None, stderr=None):
    if stderr is None:
        stderr = sys.stderr
    if should_show_traceback(argv=argv, debug_level=debug_level):
        traceback.print_exc(file=stderr)
    else:
        stderr.write("Error: unexpected internal error: %s\n" % exc)
        stderr.flush()
    return 1
