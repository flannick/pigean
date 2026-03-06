#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Check shared pigean/eaggl utility files are in sync."
    )
    parser.add_argument(
        "--other-repo",
        default=None,
        help="Path to sibling repo (defaults to ../eaggl from current repo root).",
    )
    parser.add_argument(
        "--require-other",
        action="store_true",
        help="Fail if the sibling repo path is absent.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    import pegs_sync_guard

    other_repo = Path(args.other_repo) if args.other_repo is not None else repo_root.parent / "eaggl"
    if not other_repo.exists():
        if args.require_other:
            print("Missing sibling repo: %s" % other_repo, file=sys.stderr)
            return 1
        print("SKIP: sibling repo not found at %s" % other_repo)
        return 0

    result = pegs_sync_guard.compare_shared_files(
        repo_root,
        other_repo,
        files=pegs_sync_guard.DEFAULT_SHARED_FILES,
    )
    if result.ok:
        print("OK: %s" % result.summary())
        return 0
    print("ERROR: %s" % result.summary(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
