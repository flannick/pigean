from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SHARED_FILES = (
    "src/pegs_utils.py",
    "src/pegs_shared/bundle.py",
    "src/pegs_shared/phewas.py",
    "src/pegs_shared/types.py",
    "src/pegs_shared/cli.py",
    "src/pegs_shared/io_common.py",
    "src/pegs_shared/xdata.py",
    "src/pegs_shared/ydata.py",
    "src/pegs_sync_guard.py",
)


TRANSITION_CANONICAL_SOURCE_DOC = "docs/CANONICAL_SOURCE.md"
DOWNSTREAM_EXPORT_ONLY_PHRASE = "standalone local `eaggl/` checkout is a downstream export target only"


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


def should_skip_sibling_sync_check(left_repo_root, right_repo_root):
    left_root = Path(left_repo_root)
    right_root = Path(right_repo_root)
    canonical_doc = left_root / TRANSITION_CANONICAL_SOURCE_DOC
    if right_root.name != "eaggl" or not canonical_doc.exists():
        return False
    try:
        doc_text = canonical_doc.read_text(encoding="utf-8")
    except OSError:
        return False
    return DOWNSTREAM_EXPORT_ONLY_PHRASE in doc_text


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
