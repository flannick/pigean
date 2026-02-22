#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
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


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a versioned bundle tar.gz and emit catalog snippet")
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    source = args.source_dir.resolve()
    if not source.is_dir():
        raise SystemExit(f"source-dir not found: {source}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar_name = f"{args.name}-{args.version}.tar.gz"
    tar_path = args.out_dir / tar_name
    manifest_path = args.out_dir / f"{args.name}-{args.version}.manifest.json"

    archive_root = f"{args.name}-{args.version}"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(source, arcname=archive_root)

    files = []
    for p in iter_files(source):
        files.append({
            "path": str(p.relative_to(source)).replace("\\", "/"),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        })

    manifest = {
        "name": args.name,
        "version": args.version,
        "archive_root": archive_root,
        "files": files,
        "archive_sha256": sha256_file(tar_path),
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    snippet = {
        args.name: {
            "version": args.version,
            "url": f"https://REPLACE_WITH_STORAGE_URL/{tar_name}",
            "sha256": manifest["archive_sha256"],
        }
    }

    print(f"Wrote {tar_path}")
    print(f"Wrote {manifest_path}")
    print("Catalog snippet:")
    print(json.dumps(snippet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
