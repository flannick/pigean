#!/usr/bin/env python3
"""Generate an audited modern-to-legacy Gibbs CLI flag crosswalk."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


CATEGORIES = {
    "Sampler mathematics and evolving state": [
        "--use-sampled-betas-in-gibbs", "--warm-start", "--no-warm-start",
        "--gauss-seidel", "--gauss-seidel-betas", "--sparse-solution",
        "--no-sparse-solution", "--sparse-frac-betas", "--sparse-frac-gibbs",
        "--sparse-max-gibbs", "--adjust-priors", "--no-adjust-priors",
        "--correct-betas-mean", "--no-correct-betas-mean", "--correct-betas-var",
        "--max-allowed-batch-correlation", "--no-initial-linear-filter",
        "--no-update-huge-scores", "--top-gene-prior",
        "--experimental-hyper-mutation", "--experimental-increase-hyper-if-betas-below",
        "--increase-hyper-if-betas-below", "--debug-zero-sparse",
    ],
    "Sampling budget and chain count": [
        "--max-num-iter", "--total-num-iter-gibbs", "--min-num-burn-in",
        "--max-num-burn-in", "--min-num-post-burn-in", "--max-num-post-burn-in",
        "--num-chains", "--max-num-restarts", "--min-num-iter-betas",
        "--max-num-iter-betas", "--num-chains-betas", "--r-threshold-burn-in",
        "--r-threshold-burn-in-betas", "--use-max-r-for-convergence",
        "--use-max-r-for-convergence-betas", "--max-frac-sem-betas",
    ],
    "Diagnostics and adaptive stopping": [
        "--strict-stopping", "--disable-stall-detection", "--diag-every",
        "--burn-in-rhat-quantile", "--burn-in-patience", "--burn-in-stall-window",
        "--burn-in-stall-delta", "--max-post-beta-rhat", "--max-abs-mcse-d",
        "--max-rel-mcse-beta", "--max-rel-prior-beta-inconsistency",
        "--stop-mcse-quantile", "--stop-patience", "--stop-top-gene-k",
        "--stop-min-gene-d", "--active-beta-top-k", "--active-beta-min-abs",
        "--beta-rel-mcse-denom-floor", "--stall-min-post-burn-samples",
        "--stall-window", "--stall-min-burn-in", "--stall-delta-rhat",
        "--stall-delta-mcse", "--stall-recent-window", "--stall-recent-eps",
    ],
    "Performance and memory": [
        "--gibbs-num-batches-parallel", "--gibbs-max-mb-X-h",
        "--pre-filter-batch-size", "--pre-filter-small-batch-size",
    ],
    "Summaries and trace output": [
        "--gibbs-summary-mode", "--write-gibbs-global-filtered-summaries",
        "--gene-set-p-active-threshold", "--num-mad", "--betas-trace-out",
        "--gene-set-stats-trace-out", "--gene-stats-trace-out",
    ],
}

LEGACY_ALIASES = {
    "--experimental-increase-hyper-if-betas-below": "--increase-hyper-if-betas-below",
}

MODERN_EFFECTIVE_NOTES = {
    "--adjust-priors": "parser None; effective True for the standard non-linear workflow",
    "--no-adjust-priors": "paired toggle; effective True unless explicitly disabled",
    "--correct-betas-mean": "parser None; effective True for the standard non-linear workflow",
    "--no-correct-betas-mean": "paired toggle; effective True unless explicitly disabled",
    "--sparse-solution": "parser None; effective True for the standard non-linear workflow",
    "--no-sparse-solution": "paired toggle; effective True unless explicitly disabled",
    "--sparse-frac-betas": "parser None; effective 0.001 for the standard non-linear workflow",
    "--max-num-burn-in": "None; derived from the per-epoch iteration budget",
    "--total-num-iter-gibbs": "None; falls back to the normalized per-epoch budget",
    "--strict-stopping": "False; when enabled, applies the strict stopping preset",
    "--disable-stall-detection": "False; when enabled, rewrites stall controls",
    "--experimental-increase-hyper-if-betas-below": "None; resolved with its compatibility alias",
    "--increase-hyper-if-betas-below": "None; compatibility alias for the experimental name",
    "--no-initial-linear-filter": "absent by default; `initial_linear_filter=True`",
    "--no-update-huge-scores": "absent by default; `update_huge_scores=True`",
    "--no-warm-start": "absent by default; `warm_start=True`",
}

DESCRIPTION_OVERRIDES = {
    "--correct-betas-mean": "Enable correction of gene-set beta means for confounding variables.",
    "--no-correct-betas-mean": "Disable correction of gene-set beta means for confounding variables.",
    "--correct-betas-var": "Enable correction of gene-set beta variances for confounding variables.",
    "--no-initial-linear-filter": "Disable the initial linear-regression filter before full Gibbs logistic updates.",
    "--no-update-huge-scores": "Prevent sampled priors from updating huge-score competition among nearby genes.",
    "--debug-zero-sparse": "Debug-only: force sparse components to zero during Gibbs updates.",
    "--gibbs-num-batches-parallel": "Maximum number of Gibbs matrix batches processed in parallel.",
    "--gibbs-max-mb-X-h": "Memory ceiling, in MB, for the Gibbs X_h working matrix before batching.",
    "--betas-trace-out": "Write inner beta-sampler traces to the requested path.",
    "--gene-set-stats-trace-out": "Write per-iteration gene-set statistic traces.",
    "--gene-stats-trace-out": "Write per-iteration gene statistic traces.",
}


def literal(node):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def parser_options(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    options = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"add_option", "add_argument"}:
            continue
        flags = [literal(arg) for arg in node.args]
        flags = [flag for flag in flags if isinstance(flag, str) and flag.startswith("--")]
        if not flags:
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        comment = ""
        source_line = lines[node.lineno - 1]
        if "#" in source_line:
            comment = source_line.split("#", 1)[1].strip()
        for flag in flags:
            dest = literal(kwargs.get("dest")) or flag[2:].replace("-", "_")
            action = literal(kwargs.get("action"))
            default = literal(kwargs.get("default"))
            if "default" not in kwargs and action == "store_true":
                default = False
            options[flag] = {
                "line": node.lineno, "dest": dest, "default": default,
                "action": action, "description": comment,
            }
    return options


def fmt(value):
    if value is None:
        return "`None`"
    if isinstance(value, str):
        return f"`{value}`"
    return f"`{value!r}`"


def main():
    root = Path(__file__).resolve().parents[1]
    modern = parser_options(root / "src/pigean/cli.py")
    legacy = parser_options(root / "legacy/priors.py")
    manifest = json.loads((root / "docs/cli_option_manifest.json").read_text(encoding="utf-8"))
    manifest_by_flag = {row["primary_flag"]: row for row in manifest["options"]}
    output = root / "graphify-out/GIBBS_CLI_FLAG_LEGACY_REPORT.md"

    rows = []
    missing_modern = []
    for category, flags in CATEGORIES.items():
        for flag in flags:
            if flag not in modern:
                missing_modern.append(flag)
                continue
            current = modern[flag]
            legacy_flag = flag if flag in legacy else LEGACY_ALIASES.get(flag)
            old = legacy.get(legacy_flag) if legacy_flag else None
            description = DESCRIPTION_OVERRIDES.get(flag) or current["description"] or (manifest_by_flag.get(flag) or {}).get("help") or "No parser description; behavior inferred from downstream use."
            if old:
                match = "Exact" if legacy_flag == flag else f"Alias to `{legacy_flag}`"
                legacy_cell = (
                    f"{match}; `{old['dest']}`; default {fmt(old['default'])}; "
                    f"[`legacy/priors.py:{old['line']}`](../legacy/priors.py#L{old['line']})"
                )
            else:
                legacy_cell = "No direct legacy CLI parameter"
            default = MODERN_EFFECTIVE_NOTES.get(flag, fmt(current["default"]))
            rows.append((category, flag, current, default, description, legacy_cell))

    direct = sum("No direct" not in row[5] and "Alias" not in row[5] for row in rows)
    aliases = sum("Alias" in row[5] for row in rows)
    absent = sum("No direct" in row[5] for row in rows)
    out = [
        "# Gibbs CLI Flags: Modern-to-Legacy Crosswalk",
        "",
        "Generated from `src/pigean/cli.py`, `docs/cli_option_manifest.json`, the graphify call graph, and `legacy/priors.py`.",
        "",
        "## Executive summary",
        "",
        f"- {len(rows)} modern Gibbs-related flags audited.",
        f"- {direct} have an exact same-name legacy declaration; {aliases} map through a renamed compatibility alias; {absent} have no direct legacy CLI parameter.",
        "- A parser default of `None` is not always the effective runtime default. Conditional defaults are called out explicitly.",
        "- “No direct legacy parameter” means no equivalent parser destination was found; it does not prove the legacy implementation lacked all related behavior.",
        "",
        "## Reading the crosswalk",
        "",
        "Modern source links identify the parser declaration. Legacy links identify the closest parser parameter. Exact matches share the same flag name; aliases share the same intended destination/behavior under a renamed flag.",
        "",
    ]
    for category in CATEGORIES:
        out += [f"## {category}", "", "| Modern flag | Default / effective default | What it does | Legacy correspondence |", "|---|---|---|---|"]
        for row_category, flag, current, default, description, legacy_cell in rows:
            if row_category != category:
                continue
            description = str(description).replace("|", "\\|")
            out.append(
                f"| [`{flag}`](../src/pigean/cli.py#L{current['line']}) | {default} | {description} | {legacy_cell} |"
            )
        out.append("")

    out += [
        "## Material compatibility differences",
        "",
        "1. Modern PIGEAN adds explicit summary selection (`--gibbs-summary-mode`), optional global-filtered sensitivity summaries, and an active-probability threshold. These have no direct legacy CLI declarations.",
        "2. Modern PIGEAN adds posterior beta R-hat and prior/beta consistency stopping gates without direct legacy flags.",
        "3. The modern experimental hyper-increase name resolves to the legacy `--increase-hyper-if-betas-below` behavior; the modern CLI retains the old spelling as a compatibility alias.",
        "4. Several important defaults remain conditional in both versions (`adjust_priors`, `correct_betas_mean`, `sparse_solution`, and `sparse_frac_betas`). Compare effective workflow defaults, not only parser literals.",
        "5. The modern and legacy snapshots both derive some memory controls from `--max-gb`, so explicit parser defaults can be capped or replaced during CLI bootstrap.",
        "",
        "## Audit limitations",
        "",
        "This report maps CLI parameters and their closest implementation concepts. It does not assert numerical equivalence between modern refactored functions and the monolithic legacy implementation. Exact equivalence requires paired regression runs with identical inputs and seeds.",
    ]
    if missing_modern:
        out += ["", "Unresolved requested flags: " + ", ".join(f"`{x}`" for x in missing_modern)]
    output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(rows)} flags; {direct} exact, {aliases} alias, {absent} no direct legacy parameter)")


if __name__ == "__main__":
    main()
