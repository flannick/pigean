from __future__ import annotations

import importlib

from . import domain as eaggl_domain


_LEGACY_CORE = None


def load_legacy_core():
    global _LEGACY_CORE
    if _LEGACY_CORE is None:
        _LEGACY_CORE = importlib.import_module("eaggl.legacy_main")
    return _LEGACY_CORE


def build_main_domain():
    return eaggl_domain.build_main_domain(load_legacy_core())
