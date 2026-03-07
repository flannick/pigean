from __future__ import annotations

import pegs_shared.bundle as _bundle


for _name, _value in vars(_bundle).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

__all__ = [name for name in globals() if not name.startswith("__")]

