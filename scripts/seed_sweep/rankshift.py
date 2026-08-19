"""How far does an id move between seeds, as a function of where it ranks?

A single median rank sd over every gene is close to meaningless: most of the
18k genes sit in a long tail where the metric is flat and neighbours swap on
noise, so the tail dominates the median and reports instability that nobody
reads. What matters is whether the ids at the *top* stay at the top.

So everything here is stratified by consensus rank band, and the headline
quantity is not spread but **worst rank**: for an id whose consensus rank is 1,
the number that matters is the worst rank any single seed gave it. A gene set
that is rank 1 in one seed and rank 2000 in another is a different kind of
problem from one that wobbles between 40 and 60, and an sd computed over the
whole table hides both inside the same number.

Definitions used throughout:

``consensus_rank``   mean of the id's per-seed ranks on the metric
``best/worst_rank``  min / max of those per-seed ranks
``rank_range``       worst - best
``absent``           the id was not written by that run at all. Treated as a
                     rank worse than every present id, because disappearing
                     from the output is the most complete way of falling out
                     of the top -- scoring it as "missing" would quietly
                     exclude the worst cases from the statistic.
``retention@K``      of the ids whose consensus rank is in the top K, the
                     fraction that are in the top K in *every* seed
``blowout``          worst_rank > 10x consensus_rank, i.e. the id left its
                     neighbourhood entirely in at least one seed
``disagreement``     mean over ids in the band, and over run pairs, of
                     ``|value_a - value_b|`` -- how far apart the seeds put the
                     *number*, not the rank. Reported per band because it is
                     several times larger at the top than over the whole table
                     (on real T2D, 0.129 in the top 50 against 0.033 overall),
                     so a single figure would be dominated by a flat tail no
                     one reads.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from aggregate import TABLE_SPECS, _parse_float, rank_desc, read_table

# Cumulative top-K views, plus an "all" row.
TOP_K_BANDS = (50, 100, 500, 1000)

# Per-table metrics worth a rank-shift breakdown.
DEFAULT_METRICS = {
    "gene_stats": ["combined", "prior", "log_bf"],
    # beta_uncorrected first: see the note in aggregate.py's TABLE_SPECS.
    "gene_set_stats": ["beta_uncorrected", "beta", "avg_postp"],
    "gene_gene_set_stats": ["beta"],
}

BLOWOUT_FACTOR = 10.0


def _log(message: str) -> None:
    print(message, flush=True)


def collect_ranks(run_dirs, spec, metric, keep_values=False):
    """Per-run rank of every id on ``metric``. Absent ids get a sentinel rank.

    The sentinel is ``n_ranked + 1`` for that run -- one worse than the worst
    id the run actually ranked. Using a real number rather than a gap keeps
    every downstream statistic (worst rank, range, retention) defined for ids
    that some seeds drop, which are exactly the cases worth seeing.
    """
    per_run = []
    per_run_values = []
    labels = []
    for run_dir in run_dirs:
        path = Path(run_dir) / spec.filename
        if not path.exists():
            continue
        fieldnames, rows = read_table(path)
        if metric not in fieldnames:
            continue
        values = [_parse_float(row.get(metric)) for row in rows]
        ranks = rank_desc(values)
        keys = [tuple(row.get(col, "") for col in spec.key_cols) for row in rows]
        ranked = {key: rank for key, rank in zip(keys, ranks) if rank is not None}
        if not ranked:
            continue
        per_run.append(ranked)
        if keep_values:
            per_run_values.append(
                {key: value for key, value in zip(keys, values) if value is not None}
            )
        labels.append(Path(run_dir).name)
    return (labels, per_run, per_run_values) if keep_values else (labels, per_run)


def build_rows(labels, per_run):
    """One record per id seen anywhere, with its per-seed ranks filled in."""
    if len(per_run) < 2:
        return []
    sentinels = [len(ranked) + 1 for ranked in per_run]
    all_keys = set()
    for ranked in per_run:
        all_keys.update(ranked)

    records = []
    for key in all_keys:
        ranks, absent = [], 0
        for ranked, sentinel in zip(per_run, sentinels):
            rank = ranked.get(key)
            if rank is None:
                rank = sentinel
                absent += 1
            ranks.append(rank)
        records.append(
            {
                "key": key,
                "ranks": ranks,
                "consensus_rank": sum(ranks) / len(ranks),
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "rank_range": max(ranks) - min(ranks),
                "absent_runs": absent,
            }
        )
    records.sort(key=lambda r: r["consensus_rank"])
    for position, record in enumerate(records, start=1):
        record["consensus_position"] = position
    return records


def _pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def band_disagreement(records, low, high, per_run_values):
    """Mean unsigned pairwise value gap for the ids in [low, high].

    Also reported relative to the band's mean absolute value, so the number
    reads across metrics whose scales differ by orders of magnitude.
    """
    band = [r for r in records if low <= r["consensus_position"] <= high]
    gaps, magnitudes = [], []
    for record in band:
        values = [v.get(record["key"]) for v in per_run_values]
        values = [v for v in values if v is not None and math.isfinite(v)]
        if len(values) < 2:
            continue
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                gaps.append(abs(values[i] - values[j]))
        magnitudes.append(abs(sum(values) / len(values)))
    if not gaps:
        return None
    mean_gap = float(np.mean(gaps))
    mean_magnitude = float(np.mean(magnitudes)) if magnitudes else 0.0
    return {
        "n": len(magnitudes),
        "mean_abs_diff": mean_gap,
        "p90_abs_diff": _pct(gaps, 90),
        "max_abs_diff": max(gaps),
        "mean_abs_value": mean_magnitude,
        "relative": mean_gap / mean_magnitude if mean_magnitude else None,
    }


def band_stats(records, low, high):
    """Summarise the ids whose consensus position falls in [low, high]."""
    band = [r for r in records if low <= r["consensus_position"] <= high]
    if not band:
        return None
    ranges = [r["rank_range"] for r in band]
    worst = [r["worst_rank"] for r in band]
    blowouts = [
        r for r in band if r["worst_rank"] > BLOWOUT_FACTOR * max(r["consensus_rank"], 1.0)
    ]
    dropped = [r for r in band if r["absent_runs"]]
    return {
        "n": len(band),
        "range_median": float(np.median(ranges)),
        "range_p90": _pct(ranges, 90),
        "range_max": max(ranges),
        "worst_rank_median": float(np.median(worst)),
        "worst_rank_max": max(worst),
        "blowouts": len(blowouts),
        "dropped_by_some_seed": len(dropped),
    }


def retention_at_k(records, k):
    """Of the consensus top-K ids, how many are top-K in every seed."""
    top = [r for r in records if r["consensus_position"] <= k]
    if not top:
        return None
    kept = sum(1 for r in top if r["worst_rank"] <= k)
    return {"k": k, "n": len(top), "retained": kept, "fraction": kept / len(top)}


def analyse(run_dirs, spec, metric, log=_log):
    labels, per_run, per_run_values = collect_ranks(run_dirs, spec, metric, keep_values=True)
    if len(per_run) < 2:
        return None
    records = build_rows(labels, per_run)
    if not records:
        return None

    total = len(records)
    bands = []
    disagreement = []

    def _record_band(low, high, name):
        stats = band_stats(records, low, high)
        if stats:
            stats["band"] = name
            bands.append(stats)
        gap = band_disagreement(records, low, high, per_run_values)
        if gap:
            gap["band"] = name
            disagreement.append(gap)

    # Cumulative top-K views, so "top 100" means ranks 1-100 and not 51-100 --
    # that is how a reader reads a top-K list.
    for k in TOP_K_BANDS:
        if k <= total:
            gap = band_disagreement(records, 1, k, per_run_values)
            if gap:
                gap["band"] = "top_%d" % k
                disagreement.append(gap)

    previous = 0
    for k in TOP_K_BANDS:
        if previous >= total:
            break
        stats = band_stats(records, previous + 1, min(k, total))
        if stats:
            stats["band"] = "%d-%d" % (previous + 1, min(k, total))
            bands.append(stats)
        previous = k
    if previous < total:
        stats = band_stats(records, previous + 1, total)
        if stats:
            stats["band"] = "%d+" % (previous + 1)
            bands.append(stats)
    overall = band_stats(records, 1, total)
    if overall:
        overall["band"] = "all"
        bands.append(overall)
    overall_gap = band_disagreement(records, 1, total, per_run_values)
    if overall_gap:
        overall_gap["band"] = "all"
        disagreement.append(overall_gap)

    retention = [retention_at_k(records, k) for k in TOP_K_BANDS]
    retention = [r for r in retention if r]

    return {
        "table": spec.name,
        "metric": metric,
        "runs": labels,
        "n_ids": total,
        "bands": bands,
        "disagreement": disagreement,
        "retention": retention,
        "records": records,
    }


def print_report(result, log=_log):
    log("")
    log("=== %s / %s  (%d ids, %d runs) ===" % (result["table"], result["metric"], result["n_ids"], len(result["runs"])))
    log("")
    log("Retention: of the consensus top-K, how many are top-K in EVERY seed")
    log("%-10s %-10s %-12s %-10s" % ("K", "n", "retained", "fraction"))
    for entry in result["retention"]:
        log(
            "%-10d %-10d %-12d %-10.3f"
            % (entry["k"], entry["n"], entry["retained"], entry["fraction"])
        )
    log("")
    log("Movement by consensus rank band")
    log(
        "%-12s %-8s %-12s %-10s %-12s %-12s %-10s %-10s"
        % ("band", "n", "range_med", "range_p90", "range_max", "worst_med", "worst_max", "blowouts")
    )
    for stats in result["bands"]:
        log(
            "%-12s %-8d %-12.1f %-10.1f %-12d %-12.1f %-10d %-10d"
            % (
                stats["band"],
                stats["n"],
                stats["range_median"],
                stats["range_p90"],
                stats["range_max"],
                stats["worst_rank_median"],
                stats["worst_rank_max"],
                stats["blowouts"],
            )
        )


def print_disagreement(result, log=_log):
    rows = result.get("disagreement") or []
    if not rows:
        return
    log("")
    log("Mean UNSIGNED value disagreement between seeds")
    log("%-12s %-8s %-14s %-12s %-12s %-12s" % ("band", "n", "mean_abs_diff", "p90", "max", "relative"))
    for gap in rows:
        log(
            "%-12s %-8d %-14.5g %-12.5g %-12.5g %-12s"
            % (
                gap["band"],
                gap["n"],
                gap["mean_abs_diff"],
                gap["p90_abs_diff"],
                gap["max_abs_diff"],
                "-" if gap["relative"] is None else "%.4g" % gap["relative"],
            )
        )


def print_offenders(result, top_k=100, limit=15, log=_log):
    """The cases the user actually fears: high consensus rank, terrible worst rank."""
    candidates = [
        r for r in result["records"] if r["consensus_position"] <= top_k
    ]
    candidates.sort(key=lambda r: -r["worst_rank"])
    shown = [r for r in candidates if r["rank_range"] > 0][:limit]
    if not shown:
        log("")
        log("no movement among the consensus top %d" % top_k)
        return
    log("")
    log("Worst falls among the consensus top %d (%s / %s)" % (top_k, result["table"], result["metric"]))
    log(
        "%-52s %-10s %-8s %-8s %-8s %s"
        % ("id", "consensus", "best", "worst", "range", "per-seed ranks")
    )
    for record in shown:
        name = "|".join(record["key"])
        log(
            "%-52.52s %-10.1f %-8d %-8d %-8d %s%s"
            % (
                name,
                record["consensus_rank"],
                record["best_rank"],
                record["worst_rank"],
                record["rank_range"],
                ", ".join("%d" % r for r in record["ranks"]),
                "  (absent from %d run(s))" % record["absent_runs"] if record["absent_runs"] else "",
            )
        )


def _merge_rows(path: Path, header, fresh_rows, key_cols=(0, 1)):
    """Rewrite one TSV, replacing only the (table, metric) pairs just computed.

    Running rankshift for a subset of tables must not delete the other tables'
    rows, the way a plain overwrite would -- that silently turns a narrow rerun
    into data loss in a file someone is comparing arms from.
    """
    fresh_keys = {tuple(row[i] for i in key_cols) for row in fresh_rows}
    carried = []
    if path.exists():
        with open(path, newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            existing_header = next(reader, None)
            if existing_header == list(header):
                carried = [
                    row for row in reader
                    if len(row) > max(key_cols)
                    and tuple(row[i] for i in key_cols) not in fresh_keys
                ]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(carried + fresh_rows)


def write_tables(results, out_dir: Path, top_k=100, log=_log):
    out_dir.mkdir(parents=True, exist_ok=True)

    bands_header = ["table", "metric", "band", "n", "range_median", "range_p90", "range_max",
                    "worst_rank_median", "worst_rank_max", "blowouts", "dropped_by_some_seed"]
    bands_rows = [
        [result["table"], result["metric"], stats["band"], stats["n"],
         "%.4g" % stats["range_median"], "%.4g" % stats["range_p90"], stats["range_max"],
         "%.4g" % stats["worst_rank_median"], stats["worst_rank_max"],
         stats["blowouts"], stats["dropped_by_some_seed"]]
        for result in results for stats in result["bands"]
    ]
    bands_path = out_dir / "rank_shift_bands.tsv"
    _merge_rows(bands_path, bands_header, bands_rows)

    retention_header = ["table", "metric", "k", "n", "retained", "fraction"]
    retention_rows = [
        [result["table"], result["metric"], entry["k"], entry["n"],
         entry["retained"], "%.4g" % entry["fraction"]]
        for result in results for entry in result["retention"]
    ]
    retention_path = out_dir / "rank_shift_retention.tsv"
    _merge_rows(retention_path, retention_header, retention_rows)

    offenders_header = ["table", "metric", "id", "consensus_rank", "best_rank", "worst_rank",
                        "rank_range", "absent_runs", "per_seed_ranks"]
    offenders_rows = []
    for result in results:
        rows = [r for r in result["records"] if r["consensus_position"] <= top_k]
        rows.sort(key=lambda r: -r["worst_rank"])
        offenders_rows += [
            [result["table"], result["metric"], "|".join(record["key"]),
             "%.4g" % record["consensus_rank"], record["best_rank"], record["worst_rank"],
             record["rank_range"], record["absent_runs"],
             ",".join("%d" % r for r in record["ranks"])]
            for record in rows
        ]
    offenders_path = out_dir / "rank_shift_offenders.tsv"
    _merge_rows(offenders_path, offenders_header, offenders_rows)

    dis_header = ["table", "metric", "band", "n", "mean_abs_diff", "p90_abs_diff",
                  "max_abs_diff", "mean_abs_value", "relative"]
    dis_rows = [
        [result["table"], result["metric"], gap["band"], gap["n"],
         "%.6g" % gap["mean_abs_diff"], "%.6g" % gap["p90_abs_diff"],
         "%.6g" % gap["max_abs_diff"], "%.6g" % gap["mean_abs_value"],
         "" if gap["relative"] is None else "%.6g" % gap["relative"]]
        for result in results for gap in (result.get("disagreement") or [])
    ]
    dis_path = out_dir / "rank_shift_disagreement.tsv"
    _merge_rows(dis_path, dis_header, dis_rows)

    log("wrote %s, %s, %s, %s" % (bands_path.name, retention_path.name,
                                   offenders_path.name, dis_path.name))


def run(sweep_dir: Path, tables=None, metrics=None, top_k=100, limit=15, log=_log):
    import json

    manifest_path = sweep_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        run_dirs = [r["run_dir"] for r in manifest["runs"] if r.get("returncode") == 0]
    else:
        run_dirs = sorted(str(p) for p in (sweep_dir / "runs").glob("seed_*"))
    if len(run_dirs) < 2:
        raise RuntimeError("need at least two successful runs in %s" % sweep_dir)

    results = []
    for name in tables or list(TABLE_SPECS):
        spec = TABLE_SPECS[name]
        for metric in metrics or DEFAULT_METRICS.get(name, []):
            result = analyse(run_dirs, spec, metric, log=log)
            if result is None:
                continue
            results.append(result)
            print_report(result, log=log)
            print_disagreement(result, log=log)
            print_offenders(result, top_k=top_k, limit=limit, log=log)

    if results:
        write_tables(results, sweep_dir / "aggregate", top_k=top_k, log=log)
    return results
