from __future__ import annotations

import pegs_shared.phewas as _phewas


for _name, _value in vars(_phewas).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

__all__ = [name for name in globals() if not name.startswith("__")]
