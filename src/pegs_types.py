from __future__ import annotations

import pegs_shared.types as _types


for _name, _value in vars(_types).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

__all__ = [name for name in globals() if not name.startswith("__")]

