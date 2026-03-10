#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

PIGEAN_LEGACY_PATH = REPO_ROOT / "src" / "pigean_legacy_main.py"
EAGGL_LEGACY_PATH = REPO_ROOT / "src" / "eaggl" / "legacy_main.py"

SOURCE_GLOBS = ("src/**/*.py", "tests/**/*.py")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_python_files() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in SOURCE_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def _iter_source_python_files() -> list[Path]:
    return sorted((REPO_ROOT / "src").rglob("*.py"))


def _collect_top_level_defs(path: Path) -> list[str]:
    if not path.exists():
        return []
    tree = ast.parse(_read_text(path), filename=str(path))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names


def _find_symbol_references(symbols: list[str], owner_path: Path) -> dict[str, list[str]]:
    if not owner_path.exists():
        return {}
    refs: dict[str, list[str]] = defaultdict(list)
    for path in _iter_python_files():
        if path == owner_path:
            continue
        text = _read_text(path)
        rel = str(path.relative_to(REPO_ROOT))
        for symbol in symbols:
            if symbol in text:
                refs[symbol].append(rel)
    return dict(refs)


def _count_text_occurrences(needle: str) -> int:
    count = 0
    for path in _iter_python_files():
        count += _read_text(path).count(needle)
    return count


def _collect_import_sites(module_name: str) -> list[str]:
    hits: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(_read_text(path), filename=str(path))
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == module_name:
                        found = True
                        break
            elif isinstance(node, ast.ImportFrom) and node.module == module_name:
                found = True
            if found:
                break
        if found:
            hits.append(str(path.relative_to(REPO_ROOT)))
    return hits


def _collect_dynamic_import_sites(module_name: str) -> list[str]:
    needle = 'import_module("%s")' % module_name
    hits: list[str] = []
    for path in _iter_source_python_files():
        if needle in _read_text(path):
            hits.append(str(path.relative_to(REPO_ROOT)))
    return hits


def _count_source_text_occurrences(needle: str) -> int:
    count = 0
    for path in _iter_source_python_files():
        count += _read_text(path).count(needle)
    return count


def build_report() -> dict[str, object]:
    pigean_symbols = _collect_top_level_defs(PIGEAN_LEGACY_PATH)
    eaggl_symbols = _collect_top_level_defs(EAGGL_LEGACY_PATH)
    return {
        "repo_root": str(REPO_ROOT),
        "line_counts": {
            "src/pigean_legacy_main.py": len(_read_text(PIGEAN_LEGACY_PATH).splitlines()) if PIGEAN_LEGACY_PATH.exists() else None,
            "src/eaggl/legacy_main.py": len(_read_text(EAGGL_LEGACY_PATH).splitlines()) if EAGGL_LEGACY_PATH.exists() else None,
        },
        "legacy_import_sites": {
            "pigean_legacy_main": _collect_import_sites("pigean_legacy_main"),
            "eaggl.legacy_main": _collect_import_sites("eaggl.legacy_main"),
        },
        "legacy_dynamic_import_sites": {
            "pigean_legacy_main": _collect_dynamic_import_sites("pigean_legacy_main"),
            "eaggl.legacy_main": _collect_dynamic_import_sites("eaggl.legacy_main"),
        },
        "legacy_import_counts": {
            "pigean_legacy_main": len(_collect_import_sites("pigean_legacy_main")),
            "eaggl.legacy_main": len(_collect_import_sites("eaggl.legacy_main")),
        },
        "module_object_dispatch_counts": {
            "dispatch.run_main_pipeline(_legacy_main": _count_source_text_occurrences("dispatch.run_main_pipeline(_legacy_main"),
            "dispatch.run_main_pipeline(sys.modules[__name__]": _count_source_text_occurrences("dispatch.run_main_pipeline(sys.modules[__name__]"),
            "_eaggl_dispatch.run_main_pipeline(_build_main_domain()": _count_source_text_occurrences("_eaggl_dispatch.run_main_pipeline(_build_main_domain()"),
        },
        "pigean_state_reference_count": _count_text_occurrences("PigeanState"),
        "pigean_legacy_top_level_defs": pigean_symbols,
        "eaggl_legacy_top_level_defs": eaggl_symbols,
        "pigean_legacy_external_symbol_refs": _find_symbol_references(pigean_symbols, PIGEAN_LEGACY_PATH),
        "eaggl_legacy_external_symbol_refs": _find_symbol_references(eaggl_symbols, EAGGL_LEGACY_PATH),
    }


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
