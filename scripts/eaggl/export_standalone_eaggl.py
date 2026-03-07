#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


COPY_PATHS = (
    ("src/eaggl", "src/eaggl"),
    ("src/pegs_cli_errors.py", "src/pegs_cli_errors.py"),
    ("src/pegs_cli_utils.py", "src/pegs_cli_utils.py"),
    ("src/pegs_sync_guard.py", "src/pegs_sync_guard.py"),
    ("src/pegs_types.py", "src/pegs_types.py"),
    ("src/pegs_utils.py", "src/pegs_utils.py"),
    ("src/pegs_utils_bundle.py", "src/pegs_utils_bundle.py"),
    ("src/pegs_utils_phewas.py", "src/pegs_utils_phewas.py"),
    ("docs/eaggl", "docs"),
    ("tests/eaggl", "tests/eaggl"),
    ("scripts/eaggl/check_shared_utils_sync.py", "scripts/check_shared_utils_sync.py"),
    ("scripts/eaggl/finalize_regression_checks.sh", "scripts/finalize_regression_checks.sh"),
    ("scripts/eaggl/freeze_factor_workflow_effective_configs.sh", "scripts/freeze_factor_workflow_effective_configs.sh"),
    ("scripts/eaggl/generate_cli_manifest.py", "scripts/generate_cli_manifest.py"),
    ("scripts/eaggl/release_readiness_check.sh", "scripts/release_readiness_check.sh"),
    ("scripts/eaggl/run_with_metrics.py", "scripts/run_with_metrics.py"),
)


def _copy_path(src_root: Path, dst_root: Path, src_rel: str, dst_rel: str) -> None:
    src_path = src_root / src_rel
    dst_path = dst_root / dst_rel
    if not src_path.exists():
        raise FileNotFoundError("Canonical path not found: %s" % src_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_dir():
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(
            src_path,
            dst_path,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        shutil.copy2(src_path, dst_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export canonical EAGGL sources from the pigean repo into a downstream standalone checkout.",
    )
    parser.add_argument(
        "target_repo",
        nargs="?",
        default=None,
        help="Path to the downstream standalone eaggl repo. Defaults to ../eaggl relative to the canonical pigean repo.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    target_repo = Path(args.target_repo).resolve() if args.target_repo is not None else repo_root.parent / "eaggl"

    if not target_repo.exists():
        raise SystemExit("Target downstream repo does not exist: %s" % target_repo)
    if not target_repo.is_dir():
        raise SystemExit("Target downstream path is not a directory: %s" % target_repo)

    for src_rel, dst_rel in COPY_PATHS:
        _copy_path(repo_root, target_repo, src_rel, dst_rel)

    print("Exported canonical EAGGL snapshot from %s to %s" % (repo_root, target_repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
