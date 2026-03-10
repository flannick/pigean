from __future__ import annotations

import importlib


_PUBLIC_SUBMODULES = frozenset(
    [
        "app",
        "cli",
        "covariates",
        "dispatch",
        "domain",
        "factor",
        "factor_runtime",
        "io",
        "labeling",
        "labeling_providers",
        "main_support",
        "outputs",
        "phewas",
        "regression",
        "workflows",
        "y_inputs",
    ]
)

_COMPAT_EXPORTS = {
    "main": ("app", "main"),
    "GeneSetData": ("legacy_main", "GeneSetData"),
}


def _load_submodule(name):
    if name not in _PUBLIC_SUBMODULES:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    return importlib.import_module("." + name, __name__)


def _load_compat_attr(name):
    if name not in _COMPAT_EXPORTS:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    module_name, attr_name = _COMPAT_EXPORTS[name]
    module = importlib.import_module("." + module_name, __name__)
    return getattr(module, attr_name)


def main(argv=None):
    return _load_compat_attr("main")(argv)


def __getattr__(name):
    if name in _PUBLIC_SUBMODULES:
        return _load_submodule(name)
    return _load_compat_attr(name)


def __dir__():
    return sorted(set(globals().keys()) | set(_PUBLIC_SUBMODULES) | set(_COMPAT_EXPORTS.keys()))


__all__ = ["main", "GeneSetData"] + sorted(_PUBLIC_SUBMODULES)
