#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENTRYPOINT = "src/eaggl/cli.py"
DEFAULT_MODULE = "eaggl.cli"
DEFAULT_MANIFEST = "docs/eaggl/cli_option_manifest.json"
DEFAULT_DOC = "docs/eaggl/CLI_OPTIONS.md"
CATEGORY_ORDER = [
    "method_required",
    "method_optional",
    "engineering",
    "experimental",
    "compat_alias",
    "deprecated",
    "moved_to_other_repo",
    "debug_only",
]


@dataclass
class OptionSpec:
    primary_flag: str
    flags: list[str]
    dest: str
    option_type: str | None
    action: str | None
    default_repr: str | None
    help_text: str | None
    summary: str | None
    category: str
    public_visibility: str
    scientific_semantic_impact: str
    documentation_target: str
    source_line: int
    usage_references: int
    replacement: str | None = None
    deprecation_timeline: str | None = None

    def as_dict(self):
        return {
            "primary_flag": self.primary_flag,
            "flags": self.flags,
            "dest": self.dest,
            "type": self.option_type,
            "action": self.action,
            "default": self.default_repr,
            "help": self.help_text,
            "summary": self.summary,
            "category": self.category,
            "public_visibility": self.public_visibility,
            "scientific_semantic_impact": self.scientific_semantic_impact,
            "documentation_target": self.documentation_target,
            "source_line": self.source_line,
            "usage_references": self.usage_references,
            "replacement": self.replacement,
            "deprecation_timeline": self.deprecation_timeline,
        }


def _source_text_for_node(src_text: str, node):
    seg = ast.get_source_segment(src_text, node)
    if seg is not None:
        return seg
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _literal_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_option_calls(src_text: str):
    tree = ast.parse(src_text)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_option"):
            continue
        flags = []
        for arg in node.args:
            s = _literal_string(arg)
            if s is not None:
                flags.append(s)
        if len(flags) == 0:
            continue
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            kwargs[kw.arg] = kw.value
        calls.append((node.lineno, flags, kwargs))
    return sorted(calls, key=lambda x: x[0])


def _infer_dest(flags: list[str], kwargs, src_text: str):
    if "dest" in kwargs:
        maybe = _literal_string(kwargs["dest"])
        if maybe is not None:
            return maybe
        raw = _source_text_for_node(src_text, kwargs["dest"])
        if raw is not None:
            return raw.strip("\"'")
    long_flags = [f for f in flags if f.startswith("--")]
    if len(long_flags) == 0:
        fallback = flags[0].lstrip("-")
    else:
        fallback = max(long_flags, key=len).lstrip("-")
    return fallback.replace("-", "_")


def _extract_kw_repr(kwargs, key, src_text: str):
    if key not in kwargs:
        return None
    node = kwargs[key]
    if isinstance(node, ast.Constant):
        return str(node.value)
    return _source_text_for_node(src_text, node)


def _extract_help(kwargs):
    node = kwargs.get("help")
    if node is None:
        return None
    s = _literal_string(node)
    if s is not None:
        return s
    return None


def _iter_text_files(repo_root: Path):
    skip_dirs = {".git", ".pytest_cache", "__pycache__", "dist", ".tmp"}
    skip_files = {"CLI_OPTIONS.md", "cli_option_manifest.json"}
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name in skip_files:
            continue
        if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".sh", ".toml", ".tsv"}:
            yield path


def _count_usage_references(repo_root: Path, primary_flags: list[str]):
    refs = Counter()
    for path in _iter_text_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for flag in primary_flags:
            refs[flag] += text.count(flag)
    return refs


def _load_overrides(path: Path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {}
    return data


def _load_cli_metadata(repo_root: Path, module_name: str):
    sys.path.insert(0, str(repo_root / "src"))
    module = importlib.import_module(module_name)
    metadata = module.get_cli_manifest_metadata()
    if not isinstance(metadata, dict):
        raise ValueError("CLI metadata must be a dict: %s" % module_name)
    return metadata


def build_manifest(repo_root: Path, entrypoint: Path, module_name: str, overrides: dict):
    metadata_map = _load_cli_metadata(repo_root, module_name)
    src_text = entrypoint.read_text(encoding="utf-8")
    calls = _collect_option_calls(src_text)
    option_rows = []
    for lineno, flags, kwargs in calls:
        long_flags = [f for f in flags if f.startswith("--")]
        primary_flag = long_flags[0] if len(long_flags) > 0 else flags[0]
        row = {
            "source_line": lineno,
            "flags": flags,
            "primary_flag": primary_flag,
            "dest": _infer_dest(flags, kwargs, src_text),
            "type": _extract_kw_repr(kwargs, "type", src_text),
            "action": _extract_kw_repr(kwargs, "action", src_text),
            "default": _extract_kw_repr(kwargs, "default", src_text),
            "help": _extract_help(kwargs),
        }
        option_rows.append(row)

    usage_refs = _count_usage_references(repo_root, [r["primary_flag"] for r in option_rows])
    specs = []
    missing_metadata = []
    for row in option_rows:
        primary = row["primary_flag"]
        meta = metadata_map.get(primary)
        if meta is None:
            missing_metadata.append(primary)
            continue
        override = overrides.get(primary, {})
        spec = OptionSpec(
            primary_flag=primary,
            flags=row["flags"],
            dest=row["dest"],
            option_type=row["type"],
            action=row["action"],
            default_repr=row["default"],
            help_text=row["help"],
            summary=meta.get("summary"),
            category=override.get("category", meta["category"]),
            public_visibility=override.get("public_visibility", meta["public_visibility"]),
            scientific_semantic_impact=override.get(
                "scientific_semantic_impact",
                meta["scientific_semantic_impact"],
            ),
            documentation_target=override.get("documentation_target", meta["documentation_target"]),
            source_line=row["source_line"],
            usage_references=int(usage_refs[primary]),
            replacement=override.get("replacement"),
            deprecation_timeline=override.get("deprecation_timeline"),
        )
        specs.append(spec)

    if len(missing_metadata) > 0:
        raise ValueError("Missing CLI metadata for: %s" % ", ".join(sorted(missing_metadata)))

    specs.sort(key=lambda s: s.primary_flag)
    categories = Counter(s.category for s in specs)
    visibilities = Counter(s.public_visibility for s in specs)
    manifest = {
        "entrypoint": str(entrypoint.relative_to(repo_root)),
        "module": module_name,
        "num_options": len(specs),
        "categories": {k: categories[k] for k in CATEGORY_ORDER if k in categories},
        "visibilities": dict(sorted(visibilities.items())),
        "options": [s.as_dict() for s in specs],
    }
    return manifest


def render_doc(manifest: dict):
    lines = []
    lines.append("# CLI Option Inventory")
    lines.append("")
    lines.append(
        "This document is generated from parser definitions in `%s` and CLI metadata in `%s`."
        % (manifest["entrypoint"], manifest["module"])
    )
    lines.append("Do not edit manually; run `scripts/eaggl/generate_cli_manifest.py`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Total options: `%d`" % manifest["num_options"])
    for category, count in manifest["categories"].items():
        lines.append("- `%s`: `%d`" % (category, count))
    for visibility, count in manifest.get("visibilities", {}).items():
        lines.append("- visibility `%s`: `%d`" % (visibility, count))
    lines.append("")

    grouped = defaultdict(list)
    for row in manifest["options"]:
        grouped[row["category"]].append(row)

    for category in CATEGORY_ORDER:
        rows = grouped.get(category, [])
        if len(rows) == 0:
            continue
        lines.append("## %s" % category.replace("_", " ").title())
        lines.append("")
        lines.append("| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in rows:
            note = row.get("summary") or row.get("help") or "-"
            note = note.replace("\n", " ").replace("|", "\\|")
            lines.append(
                "| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | %s |"
                % (
                    row["primary_flag"],
                    row["public_visibility"],
                    row["scientific_semantic_impact"],
                    row["documentation_target"],
                    row["dest"],
                    row.get("default") or "-",
                    note,
                )
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_manifest_metadata(manifest: dict):
    for row in manifest.get("options", []):
        for key in (
            "category",
            "public_visibility",
            "scientific_semantic_impact",
            "documentation_target",
        ):
            if not row.get(key):
                raise ValueError("Option missing %s metadata: %s" % (key, row["primary_flag"]))


def main():
    parser = argparse.ArgumentParser(description="Generate/check CLI option manifest + docs.")
    parser.add_argument("--entrypoint", default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--manifest-out", default=DEFAULT_MANIFEST)
    parser.add_argument("--doc-out", default=DEFAULT_DOC)
    parser.add_argument("--overrides", default="config/cli_manifest_overrides.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    entrypoint = repo_root / args.entrypoint
    manifest_path = repo_root / args.manifest_out
    doc_path = repo_root / args.doc_out
    overrides_path = repo_root / args.overrides

    overrides = _load_overrides(overrides_path)
    manifest = build_manifest(repo_root, entrypoint, args.module, overrides)
    _check_manifest_metadata(manifest)
    doc_text = render_doc(manifest)

    if args.check:
        if not manifest_path.exists():
            raise SystemExit("Missing manifest: %s" % manifest_path)
        if not doc_path.exists():
            raise SystemExit("Missing doc: %s" % doc_path)
        current_manifest = _load_json(manifest_path)
        if current_manifest != manifest:
            raise SystemExit("CLI manifest is out of date: %s" % manifest_path)
        current_doc = doc_path.read_text(encoding="utf-8")
        if current_doc != doc_text:
            raise SystemExit("CLI docs are out of date: %s" % doc_path)
        print("OK: CLI manifest and docs are up to date")
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(doc_text, encoding="utf-8")
    print("Wrote %s and %s" % (manifest_path, doc_path))


if __name__ == "__main__":
    main()
