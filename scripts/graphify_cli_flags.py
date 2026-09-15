#!/usr/bin/env python3
"""Enrich a graphify graph with CLI flag declarations and code-use edges."""

from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import defaultdict
from pathlib import Path


LINE_RE = re.compile(r"L(\d+)")


def _line_number(node: dict) -> int:
    match = LINE_RE.search(str(node.get("source_location", "")))
    return int(match.group(1)) if match else 0


def _nearest_symbol(nodes_by_file: dict, repo: str, relpath: str, lineno: int):
    lines, candidates = nodes_by_file.get((repo, relpath), ([], []))
    if candidates:
        index = bisect.bisect_right(lines, lineno) - 1
        if index >= 0:
            return candidates[index]
    return candidates[0] if candidates else None


def _iter_code_files(repo_root: Path):
    for base, repo in ((repo_root / "src", "src"), (repo_root / "scripts", "scripts")):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".sh", ".r", ".js", ".ts"}:
                if "graphify-out" not in path.parts:
                    yield repo, path, path.relative_to(base).as_posix()


def enrich(repo_root: Path, graph_path: Path) -> tuple[int, int]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph["nodes"]
    links = graph.get("links", graph.get("edges", []))
    nodes_by_file = defaultdict(list)
    for node in nodes:
        if node.get("source_file"):
            nodes_by_file[(node.get("repo", "src"), node["source_file"])].append(node)
    for key, candidates in list(nodes_by_file.items()):
        candidates.sort(key=_line_number)
        nodes_by_file[key] = ([_line_number(node) for node in candidates], candidates)

    manifests = (
        ("pigean", repo_root / "docs/cli_option_manifest.json", "src", "pigean/cli.py"),
        ("eaggl", repo_root / "docs/eaggl/cli_option_manifest.json", "src", "eaggl/cli.py"),
    )
    option_rows = []
    for cli, path, repo, entrypoint in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        for option in data["options"]:
            option_rows.append((cli, repo, entrypoint, option))

    existing_ids = {node["id"] for node in nodes}
    existing_edges = {(edge["source"], edge["target"], edge.get("relation")) for edge in links}
    added_nodes = 0
    added_edges = 0
    report_rows = []

    source_cache = []
    for repo, path, relpath in _iter_code_files(repo_root):
        source_cache.append((repo, relpath, path.read_text(encoding="utf-8", errors="replace").splitlines()))
    flag_occurrences = defaultdict(list)
    dest_occurrences = defaultdict(list)
    flag_token_re = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")
    dotted_dest_re = re.compile(r"\b(?:args|opts|options|namespace)\.([A-Za-z_]\w*)\b")
    keyed_dest_re = re.compile(r"(?:\[['\"]|\.get\(['\"])([A-Za-z_]\w*)['\"]")
    for repo, relpath, lines in source_cache:
        for lineno, line in enumerate(lines, 1):
            location = (repo, relpath, lineno, line)
            for flag in set(flag_token_re.findall(line)):
                flag_occurrences[flag].append(location)
            for dest in set(dotted_dest_re.findall(line) + keyed_dest_re.findall(line)):
                dest_occurrences[dest].append(location)

    for cli, decl_repo, entrypoint, option in option_rows:
        primary = option["primary_flag"]
        node_id = f"cli_flag::{cli}::{primary}"
        if node_id not in existing_ids:
            nodes.append({
                "id": node_id,
                "label": primary,
                "norm_label": primary.lower(),
                "file_type": "cli_flag",
                "source_file": entrypoint,
                "source_location": f"L{option['source_line']}",
                "repo": decl_repo,
                "cli": cli,
                "dest": option.get("dest"),
                "category": option.get("category"),
                "help": option.get("help"),
                "_origin": "cli_manifest",
                "community_name": "CLI Flags",
            })
            existing_ids.add(node_id)
            added_nodes += 1

        declaration = _nearest_symbol(nodes_by_file, decl_repo, entrypoint, option["source_line"])
        if declaration and (node_id, declaration["id"], "declared_by") not in existing_edges:
            links.append({
                "source": node_id, "target": declaration["id"], "relation": "declared_by",
                "context": "CLI manifest declaration", "confidence": "EXTRACTED",
                "confidence_score": 1.0, "weight": 1.0, "_origin": "cli_manifest",
                "source_file": entrypoint, "source_location": f"L{option['source_line']}",
            })
            existing_edges.add((node_id, declaration["id"], "declared_by"))
            added_edges += 1

        flags = set(option.get("flags", []))
        dest = option.get("dest")
        uses = []
        occurrences = []
        for flag in flags:
            occurrences.extend((item, True) for item in flag_occurrences.get(flag, []))
        if dest and dest.isidentifier():
            occurrences.extend((item, False) for item in dest_occurrences.get(dest, []))
        seen_occurrences = set()
        for (repo, relpath, lineno, line), literal_hit in occurrences:
                occurrence_key = (repo, relpath, lineno, literal_hit)
                if occurrence_key in seen_occurrences:
                    continue
                seen_occurrences.add(occurrence_key)
                if repo == decl_repo and relpath == entrypoint and lineno == option["source_line"]:
                    continue
                symbol = _nearest_symbol(nodes_by_file, repo, relpath, lineno)
                if not symbol:
                    continue
                relation = "referenced_as_flag" if literal_hit else "consumed_as_dest"
                edge_key = (node_id, symbol["id"], relation)
                if edge_key not in existing_edges:
                    links.append({
                        "source": node_id, "target": symbol["id"], "relation": relation,
                        "context": line.strip()[:300], "confidence": "EXTRACTED",
                        "confidence_score": 1.0, "weight": 1.0, "_origin": "cli_usage",
                        "source_file": relpath, "source_location": f"L{lineno}",
                    })
                    existing_edges.add(edge_key)
                    added_edges += 1
                uses.append(f"{repo}/{relpath}:{lineno}")
        report_rows.append((cli, primary, option.get("dest", ""), option.get("category", ""), sorted(set(uses))))

    graph["nodes"] = nodes
    if "links" in graph:
        graph["links"] = links
    else:
        graph["edges"] = links
    graph_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

    report = ["# CLI Flag Usage Index", "", "Generated from the authoritative CLI manifests and source references.", ""]
    for cli, flag, dest, category, uses in sorted(report_rows):
        report.append(f"## `{flag}` ({cli})")
        report.append("")
        report.append(f"- Destination: `{dest}`")
        report.append(f"- Category: `{category}`")
        report.append(f"- Code-use locations: {len(uses)}")
        for use in uses:
            report.append(f"  - `{use}`")
        report.append("")
    (graph_path.parent / "CLI_FLAG_USAGE.md").write_text("\n".join(report), encoding="utf-8")
    return added_nodes, added_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--graph", type=Path, default=Path("graphify-out/graph.json"))
    args = parser.parse_args()
    added_nodes, added_edges = enrich(args.repo_root.resolve(), args.graph.resolve())
    print(f"Added {added_nodes} CLI flag nodes and {added_edges} usage edges")


if __name__ == "__main__":
    main()
