#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy a legacy priors script snapshot into legacy/")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--dest", type=Path, default=Path("legacy/priors.warm_stopping.py"))
    args = ap.parse_args()

    src = args.source.resolve()
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, args.dest)

    h = sha256_file(args.dest)
    digest_file = args.dest.with_suffix(args.dest.suffix + ".sha256")
    digest_file.write_text(f"{h}  {args.dest.name}\n", encoding="utf-8")

    print(f"Staged legacy script to {args.dest}")
    print(f"SHA256: {h}")
    print(f"Wrote {digest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
