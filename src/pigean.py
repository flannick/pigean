#!/usr/bin/env python3
"""New PIGEAN entrypoint (placeholder).

The long-term goal is to move cleaned functionality here and keep
legacy behavior in `legacy/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="PIGEAN (new implementation scaffold)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--print-config", action="store_true", default=False)
    args = ap.parse_args()

    if args.print_config:
        payload = {
            "config": str(args.config) if args.config else None,
            "status": "scaffold",
            "next_step": "implement simplified pigean.py pipeline",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("pigean.py scaffold ready. Use scripts/run_legacy.py for current production runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
