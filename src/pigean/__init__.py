from __future__ import annotations

import importlib


def _legacy_module():
    return importlib.import_module(__name__ + ".legacy_main")


def main(argv=None):
    return _legacy_module().main(argv)


def __getattr__(name):
    return getattr(_legacy_module(), name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(_legacy_module())))
