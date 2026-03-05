from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SHARED_FILES = (
    "src/pegs_utils.py",
    "src/pegs_utils_bundle.py",
    "src/pegs_utils_phewas.py",
    "src/pegs_sync_guard.py",
)


@dataclass
class SyncComparisonResult:
    checked_files: list[str] = field(default_factory=list)
    missing_in_left: list[str] = field(default_factory=list)
    missing_in_right: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)

    @property
    def ok(self):
        return (
            len(self.missing_in_left) == 0
            and len(self.missing_in_right) == 0
            and len(self.mismatched) == 0
        )

    def summary(self):
        if self.ok:
            return "Shared files are in sync"
        parts = []
        if len(self.missing_in_left) > 0:
            parts.append("missing in left: %s" % ", ".join(self.missing_in_left))
        if len(self.missing_in_right) > 0:
            parts.append("missing in right: %s" % ", ".join(self.missing_in_right))
        if len(self.mismatched) > 0:
            parts.append("hash mismatch: %s" % ", ".join(self.mismatched))
        return "; ".join(parts)


def _hash_file(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as in_fh:
        while True:
            chunk = in_fh.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def compare_shared_files(left_repo_root, right_repo_root, files=None):
    if files is None:
        files = DEFAULT_SHARED_FILES
    left_root = Path(left_repo_root)
    right_root = Path(right_repo_root)
    result = SyncComparisonResult()
    for rel in files:
        rel_path = Path(rel)
        left_path = left_root / rel_path
        right_path = right_root / rel_path
        result.checked_files.append(str(rel_path))
        if not left_path.exists():
            result.missing_in_left.append(str(rel_path))
            continue
        if not right_path.exists():
            result.missing_in_right.append(str(rel_path))
            continue
        if _hash_file(left_path) != _hash_file(right_path):
            result.mismatched.append(str(rel_path))
    return result
