from __future__ import annotations

from . import legacy_main as _legacy_main


for _name, _value in vars(_legacy_main).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

main = _legacy_main.main

__all__ = [name for name in globals() if not name.startswith("__")]
