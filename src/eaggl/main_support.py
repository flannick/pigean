from __future__ import annotations

import importlib

from . import domain as eaggl_domain
from . import state as eaggl_state


_LEGACY_CORE = None


def load_legacy_core():
    global _LEGACY_CORE
    if _LEGACY_CORE is None:
        _LEGACY_CORE = importlib.import_module("eaggl.legacy_main")
    return _LEGACY_CORE


def build_main_domain():
    legacy_core = load_legacy_core()
    eaggl_state.bind_runtime_namespace(legacy_core)
    return eaggl_domain.build_main_domain(legacy_core)
