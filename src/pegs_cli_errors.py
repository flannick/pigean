from __future__ import annotations

import argparse
import sys
import traceback
from types import SimpleNamespace


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


SUPPRESS_HELP = argparse.SUPPRESS


def _normalize_usage(usage):
    if usage is None:
        return None
    if usage.lower().startswith("usage: "):
        return usage.split(": ", 1)[1]
    return usage


def _normalize_option_type(option_type):
    if option_type in (None, "string", str):
        return "string"
    if option_type in ("int", int):
        return "int"
    if option_type in ("float", float):
        return "float"
    return option_type


def _argparse_type_for_option(option_type):
    if option_type == "int":
        return int
    if option_type == "float":
        return float
    return str


class CliOption:
    def __init__(self, *flags, **kwargs):
        clean_flags = [flag for flag in flags if isinstance(flag, str) and len(flag) > 0]
        if len(clean_flags) == 0:
            raise ValueError("CliOption requires at least one flag")
        self._short_opts = [flag for flag in clean_flags if flag.startswith("-") and not flag.startswith("--")]
        self._long_opts = [flag for flag in clean_flags if flag.startswith("--")]
        self.dest = kwargs.get("dest")
        if self.dest is None:
            fallback = self._long_opts[0] if len(self._long_opts) > 0 else clean_flags[0]
            self.dest = fallback.lstrip("-").replace("-", "_")
        self.action = kwargs.get("action")
        self.default = kwargs.get("default")
        self.help = kwargs.get("help")
        self.type = _normalize_option_type(kwargs.get("type"))
        self.callback = kwargs.get("callback")

    def takes_value(self):
        return self.action not in ("store_true", "store_false")

    def _build_callback_action(self, parser_wrapper):
        option = self

        class _CallbackAction(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                parser_wrapper.values = namespace
                parser_proxy = SimpleNamespace(values=namespace)
                option.callback(option, option_string, values, parser_proxy)
                parser_wrapper.values = namespace

        return _CallbackAction

    def add_to_argparse(self, parser_wrapper, argparse_target):
        flags = list(self._short_opts) + list(self._long_opts)
        kwargs = {
            "dest": self.dest,
            "default": self.default,
        }
        if self.help is not None:
            kwargs["help"] = self.help
        if self.action == "store_true":
            kwargs["action"] = "store_true"
        elif self.action == "store_false":
            kwargs["action"] = "store_false"
        elif self.action == "append":
            kwargs["action"] = "append"
            kwargs["type"] = _argparse_type_for_option(self.type)
        elif self.action == "callback":
            kwargs["action"] = self._build_callback_action(parser_wrapper)
            if self.takes_value():
                kwargs["type"] = _argparse_type_for_option(self.type)
        else:
            kwargs["action"] = "store"
            if self.takes_value():
                kwargs["type"] = _argparse_type_for_option(self.type)
        argparse_target.add_argument(*flags, **kwargs)


class CliOptionGroup:
    def __init__(self, parser, title, description=None):
        self.parser = parser
        self.title = title
        self.description = description
        self.option_list = []

    def add_option(self, *flags, **kwargs):
        option = CliOption(*flags, **kwargs)
        if option not in self.option_list:
            self.option_list.append(option)
        return option


class _CliArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise CliUsageError(message)


class CliOptionParser:
    def __init__(self, usage=None):
        self.usage = usage
        self.description = None
        self.epilog = None
        self.option_list = []
        self.option_groups = []
        self.values = None

    def add_option(self, *flags, **kwargs):
        option = CliOption(*flags, **kwargs)
        self.option_list.append(option)
        return option

    def add_option_group(self, group):
        if group not in self.option_groups:
            self.option_groups.append(group)
        return group

    def _build_argument_parser(self):
        parser = _CliArgumentParser(
            usage=_normalize_usage(self.usage),
            description=self.description,
            epilog=self.epilog,
            add_help=True,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        for option in self.option_list:
            option.add_to_argparse(self, parser)
        for group in self.option_groups:
            argparse_group = parser.add_argument_group(group.title, group.description)
            for option in group.option_list:
                option.add_to_argparse(self, argparse_group)
        return parser

    def format_help(self):
        text = self._build_argument_parser().format_help()
        if text.startswith("usage:"):
            text = "Usage:" + text[len("usage:"):]
        return text

    def print_help(self, file=None):
        if file is None:
            file = sys.stdout
        file.write(self.format_help())

    def parse_args(self, args=None):
        parser = self._build_argument_parser()
        namespace, extras = parser.parse_known_args(args)
        for arg in extras:
            if isinstance(arg, str) and arg.startswith("-"):
                raise CliUsageError("no such option: %s" % arg)
        self.values = namespace
        return namespace, extras


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
