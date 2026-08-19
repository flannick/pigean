"""Put several sweeps side by side, to read one ablation arm against another.

Each sweep already writes its own `aggregate/stability_summary.json`. What a
knob study needs on top of that is the same metric from every arm on one row,
plus the thing that says whether the knob did what it was supposed to: the
per-seed Gibbs sampling budget.

That budget check is not decoration. PIGEAN's restart/early-stopping machinery
stops each seed at a different epoch, so seeds differ in how much sampling they
got, not only in which draws they got. An arm that claims to pin the budget has
to be shown to have pinned it -- `budget_spread` reports min..max across seeds
for epochs, iterations and effective chains, and reads `pinned` only when all
three are identical in every seed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# (table, metric) rows worth showing side by side, in reading order.
HEADLINE_ROWS = [
    ("gene_stats", "huge_score_gwas"),
    ("gene_stats", "log_bf"),
    ("gene_stats", "prior"),
    ("gene_stats", "combined"),
    ("gene_set_stats", "beta_tilde_orig"),
    ("gene_set_stats", "beta_uncorrected"),
    ("gene_set_stats", "beta"),
    ("gene_set_stats", "avg_postp"),
    ("gene_gene_set_stats", "beta"),
]

HEADLINE_STATS = [
    ("spearman_mean", "spearman"),
    ("top_100", "top100_J"),
    ("cv_median", "cv_med"),
    ("rank_sd_median", "rank_sd_med"),
]

# params.tsv keys that describe how much sampling a run actually did.
BUDGET_KEYS = {
    "epochs": "num_gibbs_epochs_completed",
    "iters": "num_gibbs_iter_total",
    "chains": "gibbs_global_summary_chain_keep_count",
}


def _log(message: str) -> None:
    print(message, flush=True)


def read_threshold_counts(sweep_dir: Path) -> dict:
    """Per-arm threshold-count spread, e.g. how many genes clear prior > 1.

    Reported as min..max across seeds because the spread is the arm-level
    quality number: two arms can agree on ordering and still disagree on how
    long the answer is.
    """
    summary = read_summary(sweep_dir)
    out = {}
    for entry in summary or []:
        for label, counts in (entry.get("threshold_counts") or {}).items():
            out[label] = (
                str(counts["min"])
                if counts["min"] == counts["max"]
                else "%d..%d" % (counts["min"], counts["max"])
            )
    return out


def read_summary(sweep_dir: Path):
    path = sweep_dir / "aggregate" / "stability_summary.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def read_budgets(sweep_dir: Path) -> dict:
    """Per-seed sampling budget, straight out of each run's params.tsv."""
    budgets = {}
    for run_dir in sorted((sweep_dir / "runs").glob("seed_*")):
        params_path = run_dir / "params.tsv"
        if not params_path.exists():
            continue
        wanted = {}
        with open(params_path) as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) >= 3 and row[0] in BUDGET_KEYS.values():
                    wanted[row[0]] = row[2]
        budgets[run_dir.name] = {
            label: wanted.get(key) for label, key in BUDGET_KEYS.items()
        }
    return budgets


def budget_spread(budgets: dict) -> dict:
    """Collapse per-seed budgets into a min..max per quantity, plus a verdict."""
    spread = {}
    pinned = bool(budgets)
    for label in BUDGET_KEYS:
        values = []
        for seed_values in budgets.values():
            raw = seed_values.get(label)
            if raw is None:
                continue
            try:
                values.append(int(float(raw)))
            except ValueError:
                continue
        if not values:
            spread[label] = "?"
            pinned = False
            continue
        low, high = min(values), max(values)
        spread[label] = str(low) if low == high else "%d..%d" % (low, high)
        if low != high:
            pinned = False
    spread["verdict"] = "pinned" if pinned else "varies"
    return spread


def _lookup(summary, table, metric, stat):
    if summary is None:
        return None
    for entry in summary:
        if entry.get("table") != table:
            continue
        metrics = entry.get("metrics", {})
        if metric not in metrics:
            return None
        found = metrics[metric]
        if stat == "top_100":
            return found.get("top_k_jaccard", {}).get("top_100", {}).get("mean")
        return found.get(stat)
    return None


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "%.4g" % value
    return str(value)


def compare(sweeps: list[tuple[str, Path]], out_path: Path | None = None, log=_log) -> None:
    summaries = {name: read_summary(path) for name, path in sweeps}
    budgets = {name: budget_spread(read_budgets(path)) for name, path in sweeps}
    names = [name for name, _ in sweeps]

    width = max(12, max(len(n) for n in names) + 2)

    log("")
    log("Gibbs sampling budget across seeds (min..max; 'pinned' = identical in every seed)")
    log("-" * (22 + width * len(names)))
    log("%-22s" % "" + "".join("%-*s" % (width, n) for n in names))
    for label in list(BUDGET_KEYS) + ["verdict"]:
        log("%-22s" % label + "".join("%-*s" % (width, budgets[n][label]) for n in names))

    counts = {name: read_threshold_counts(path) for name, path in sweeps}
    labels = sorted({label for c in counts.values() for label in c})
    if labels:
        log("")
        log("Threshold counts across seeds (min..max)")
        log("-" * (22 + width * len(names)))
        log("%-22s" % "" + "".join("%-*s" % (width, n) for n in names))
        for label in labels:
            log("%-22s" % label + "".join("%-*s" % (width, counts[n].get(label, "-")) for n in names))

    for stat_key, stat_label in HEADLINE_STATS:
        log("")
        log("%s" % stat_label)
        log("-" * (40 + width * len(names)))
        log("%-40s" % "table / metric" + "".join("%-*s" % (width, n) for n in names))
        for table, metric in HEADLINE_ROWS:
            values = [_lookup(summaries[n], table, metric, stat_key) for n in names]
            if all(v is None for v in values):
                continue
            log(
                "%-40s" % ("%s.%s" % (table, metric))
                + "".join("%-*s" % (width, _fmt(v)) for v in values)
            )
    log("")

    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["arm", "table", "metric", "statistic", "value"])
        for name, path in sweeps:
            for label in list(BUDGET_KEYS) + ["verdict"]:
                writer.writerow([name, "", "", "budget_" + label, budgets[name][label]])
            for label, value in read_threshold_counts(path).items():
                writer.writerow([name, "", "", "count_" + label, value])
            for table, metric in HEADLINE_ROWS:
                for stat_key, stat_label in HEADLINE_STATS:
                    value = _lookup(summaries[name], table, metric, stat_key)
                    if value is not None:
                        writer.writerow([name, table, metric, stat_label, _fmt(value)])
    log("wrote %s" % out_path)
