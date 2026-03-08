from __future__ import annotations

import importlib


_PUBLIC_SUBMODULES = frozenset(
    [
        "cli",
        "dispatch",
        "gibbs",
        "gibbs_callbacks",
        "huge",
        "legacy_main",
        "model",
        "outputs",
        "phewas",
        "pipeline",
        "runtime",
        "x_inputs",
        "x_inputs_core",
        "y_inputs",
        "y_inputs_core",
    ]
)

_COMPAT_EXPORTS = {
    "main": ("legacy_main", "main"),
    "_build_prefilter_keep_mask": ("legacy_main", "_build_prefilter_keep_mask"),
}


def _load_submodule(name):
    if name not in _PUBLIC_SUBMODULES:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    return importlib.import_module("." + name, __name__)


def _load_compat_attr(name):
    if name not in _COMPAT_EXPORTS:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    module_name, attr_name = _COMPAT_EXPORTS[name]
    module = _load_submodule(module_name)
    return getattr(module, attr_name)


def main(argv=None):
    return _load_compat_attr("main")(argv)


def __getattr__(name):
    if name in _PUBLIC_SUBMODULES:
        return _load_submodule(name)
    return _load_compat_attr(name)


def __dir__():
    return sorted(set(globals().keys()) | set(_PUBLIC_SUBMODULES) | set(_COMPAT_EXPORTS.keys()))


__all__ = ["main", "_build_prefilter_keep_mask"] + sorted(_PUBLIC_SUBMODULES)
