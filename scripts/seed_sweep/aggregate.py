"""Combine per-seed PIGEAN stats tables into cross-seed mean/sd/rank tables.

The three PIGEAN stats tables all have the same shape for this purpose: an id
(one or two columns), some numeric score columns whose names depend on which
inputs the run had, and a couple of string columns. So the aggregation is
written schema-agnostically — columns are classified as numeric or categorical
by looking at the values, not by a hardcoded list — and the same code serves
gene stats, gene-set stats and gene/gene-set stats.

For every numeric column ``c`` the output carries, per id:

===================  =========================================================
``c_mean``           mean of ``c`` over the runs where the id appears
``c_sd``             sample standard deviation (ddof=1); NA when n < 2
``c_cv``             ``c_sd / |c_mean|`` — scale-free spread
``c_min`` ``c_max``  range across runs, for spotting a single outlier seed
``c_n``              how many runs contributed
``c_rank_mean``      mean of the id's within-run rank on ``c`` (1 = highest)
``c_rank_sd``        sd of that rank — how much the id moves in the ranking
===================  =========================================================

Rank stats are the point of the exercise: a gene whose ``combined`` mean is
7.99 ± 0.02 but whose rank swings between 3 and 340 is unstable in the way that
actually matters downstream, and only the rank columns show it.

``n_runs``/``run_frequency`` capture the other failure mode: an id that some
seeds emit and others filter out entirely. Score spread says nothing about
those; presence frequency does.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

MISSING_TOKENS = {"", "NA", "N/A", "nan", "NaN", "None", "null", "-"}

# Stats emitted per numeric column, in output order.
ALL_STATS = ("mean", "sd", "cv", "min", "max", "n", "rank_mean", "rank_sd")
DEFAULT_STATS = ALL_STATS

TOP_K_LEVELS = (10, 50, 100, 500)


class TableSpec:
    def __init__(self, name, filename, key_cols, primary_metrics, summary_metrics):
        self.name = name
        self.filename = filename
        self.key_cols = key_cols
        # First one present in the file wins; used for sort order and for the
        # convenience consensus_rank columns.
        self.primary_metrics = primary_metrics
        self.summary_metrics = summary_metrics


TABLE_SPECS = {
    "gene_stats": TableSpec(
        "gene_stats",
        "gene_stats.tsv.gz",
        ["Gene"],
        ["combined", "prior", "log_bf"],
        ["combined", "prior", "log_bf", "huge_score", "huge_score_gwas", "huge_score_exomes"],
    ),
    "gene_set_stats": TableSpec(
        "gene_set_stats",
        "gene_set_stats.tsv.gz",
        ["Gene_Set"],
        ["beta", "beta_uncorrected", "beta_tilde_orig"],
        ["beta", "beta_uncorrected", "beta_tilde_orig", "avg_postp", "p_active_beta_gt_eps"],
    ),
    "gene_gene_set_stats": TableSpec(
        "gene_gene_set_stats",
        "gene_gene_set_stats.tsv.gz",
        ["Gene", "gene_set"],
        ["beta", "combined", "prior"],
        # Only the pair-specific columns are summarised. `combined`/`prior` are
        # gene-level values repeated across every gene-set row for that gene,
        # so ranking them here just measures how many rows each gene got; the
        # gene_stats table already covers them properly.
        ["beta", "weight"],
    ),
}


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "rt", newline="")


def _parse_float(text):
    if text is None or text in MISSING_TOKENS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_table(path: Path):
    """Return (fieldnames, rows) for a PIGEAN stats TSV, gz or plain."""
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def classify_columns(fieldnames, rows, key_cols):
    """Split value columns into numeric and categorical by inspecting values.

    A column is numeric when at least one value parses as a float and no
    non-missing value fails to. That keeps ``label``/``filter_reason`` and the
    True/False discovery flags out of the arithmetic without naming them.
    """
    numeric, categorical = [], []
    for col in fieldnames:
        if col in key_cols or col is None:
            continue
        saw_number = False
        saw_non_number = False
        for row in rows:
            raw = row.get(col)
            if raw is None or raw in MISSING_TOKENS:
                continue
            if _parse_float(raw) is None:
                saw_non_number = True
                break
            saw_number = True
        if saw_number and not saw_non_number:
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def rank_desc(values):
    """1-based average-tie ranks, largest value = rank 1. None stays None.

    Average ties matter here because PIGEAN writes several columns at 3
    significant figures, which manufactures ties that would otherwise bias the
    rank sd in whichever direction the row order happened to fall.

    A column with a single distinct value (``weight`` is 1 everywhere in a
    binary X) gets no ranks at all: every row would take the same average rank
    (n+1)/2, which then differs between runs purely because the runs wrote a
    different number of rows. That is a row-count artifact, not instability.
    """
    present = [i for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if len({values[i] for i in present}) < 2:
        return [None] * len(values)
    order = sorted(present, key=lambda i: -values[i])
    ranks = [None] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


class _ColumnAccumulator:
    __slots__ = ("n", "total", "total_sq", "minimum", "maximum", "rank_total", "rank_total_sq")

    def __init__(self, size):
        self.n = [0] * size
        self.total = [0.0] * size
        self.total_sq = [0.0] * size
        self.minimum = [math.inf] * size
        self.maximum = [-math.inf] * size
        self.rank_total = [0.0] * size
        self.rank_total_sq = [0.0] * size

    def grow(self, size):
        extra = size - len(self.n)
        if extra <= 0:
            return
        self.n.extend([0] * extra)
        self.total.extend([0.0] * extra)
        self.total_sq.extend([0.0] * extra)
        self.minimum.extend([math.inf] * extra)
        self.maximum.extend([-math.inf] * extra)
        self.rank_total.extend([0.0] * extra)
        self.rank_total_sq.extend([0.0] * extra)

    def add(self, idx, value, rank):
        self.n[idx] += 1
        self.total[idx] += value
        self.total_sq[idx] += value * value
        if value < self.minimum[idx]:
            self.minimum[idx] = value
        if value > self.maximum[idx]:
            self.maximum[idx] = value
        if rank is not None:
            self.rank_total[idx] += rank
            self.rank_total_sq[idx] += rank * rank


def _mean_sd(n, total, total_sq):
    if n == 0:
        return None, None
    mean = total / n
    if n < 2:
        return mean, None
    # Two-pass would be more accurate, but these are O(1)-magnitude scores and
    # the sums stay small; guard the negative-from-rounding case instead.
    var = max(0.0, (total_sq - n * mean * mean) / (n - 1))
    return mean, math.sqrt(var)


def _fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return "%.6g" % value
    return str(value)


def aggregate_table(run_dirs, spec, out_path, *, stats=DEFAULT_STATS, min_runs=1, log=print):
    """Fold every run's copy of one stats table into a single mean/sd/rank table.

    Returns a summary dict describing cross-run stability for this table.
    """
    key_index = {}
    keys = []
    present_runs = []            # per key: how many runs contained it
    numeric_cols = []            # union across runs, first-seen order
    categorical_cols = []
    accumulators = {}
    categorical_values = {}      # col -> per-key Counter
    # Per-run values of the summary metrics, kept for the pairwise diagnostics.
    per_run_metric_values = {}   # metric -> list (per run) of {key: value}
    run_labels = []

    used_runs = 0
    for run_dir in run_dirs:
        path = Path(run_dir) / spec.filename
        if not path.exists():
            log("  %s: missing %s, skipping" % (Path(run_dir).name, spec.filename))
            continue
        fieldnames, rows = read_table(path)
        if not rows:
            log("  %s: %s is empty, skipping" % (Path(run_dir).name, spec.filename))
            continue
        run_labels.append(Path(run_dir).name)
        used_runs += 1

        run_numeric, run_categorical = classify_columns(fieldnames, rows, spec.key_cols)
        for col in run_numeric:
            if col not in accumulators:
                numeric_cols.append(col)
                accumulators[col] = _ColumnAccumulator(len(keys))
        for col in run_categorical:
            if col not in categorical_values:
                categorical_cols.append(col)
                categorical_values[col] = {}

        # Parse once, then rank once per column, then fold.
        parsed = {col: [_parse_float(row.get(col)) for row in rows] for col in run_numeric}
        ranks = {col: rank_desc(values) for col, values in parsed.items()}

        row_keys = []
        for row in rows:
            key = tuple(row.get(col, "") for col in spec.key_cols)
            idx = key_index.get(key)
            if idx is None:
                idx = len(keys)
                key_index[key] = idx
                keys.append(key)
                present_runs.append(0)
                for acc in accumulators.values():
                    acc.grow(len(keys))
            row_keys.append(idx)
            present_runs[idx] += 1

        for col in run_numeric:
            acc = accumulators[col]
            acc.grow(len(keys))
            values = parsed[col]
            col_ranks = ranks[col]
            for row_i, idx in enumerate(row_keys):
                value = values[row_i]
                if value is None or not math.isfinite(value):
                    continue
                acc.add(idx, value, col_ranks[row_i])

        for col in run_categorical:
            store = categorical_values[col]
            for row_i, idx in enumerate(row_keys):
                raw = rows[row_i].get(col)
                if raw is None or raw in MISSING_TOKENS:
                    continue
                store.setdefault(idx, Counter())[raw] += 1

        for metric in spec.summary_metrics:
            if metric in parsed:
                per_run_metric_values.setdefault(metric, {})[Path(run_dir).name] = {
                    keys[row_keys[row_i]]: parsed[metric][row_i]
                    for row_i in range(len(rows))
                    if parsed[metric][row_i] is not None
                }

    if used_runs == 0:
        log("  no runs contributed %s; nothing written" % spec.filename)
        return None

    primary = next((m for m in spec.primary_metrics if m in accumulators), None)

    # ---- write the aggregated table -------------------------------------
    header = list(spec.key_cols) + ["n_runs", "run_frequency"]
    if primary is not None:
        header += ["consensus_rank", "consensus_rank_sd", "primary_metric"]
    for col in numeric_cols:
        header += ["%s_%s" % (col, stat) for stat in stats]
    for col in categorical_cols:
        header += [col, "%s_n_values" % col]

    order = range(len(keys))
    if primary is not None:
        acc = accumulators[primary]
        primary_mean = [
            _mean_sd(acc.n[i], acc.total[i], acc.total_sq[i])[0] for i in range(len(keys))
        ]
        order = sorted(order, key=lambda i: (-(primary_mean[i] if primary_mean[i] is not None else -math.inf), keys[i]))

    kept = 0
    with gzip.open(out_path, "wt", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for i in order:
            if present_runs[i] < min_runs:
                continue
            kept += 1
            row = list(keys[i]) + [present_runs[i], _fmt(present_runs[i] / used_runs)]
            if primary is not None:
                acc = accumulators[primary]
                n = acc.n[i]
                rank_mean, rank_sd = _mean_sd(n, acc.rank_total[i], acc.rank_total_sq[i])
                row += [_fmt(rank_mean), _fmt(rank_sd), primary]
            for col in numeric_cols:
                acc = accumulators[col]
                n = acc.n[i]
                mean, sd = _mean_sd(n, acc.total[i], acc.total_sq[i])
                rank_mean, rank_sd = _mean_sd(n, acc.rank_total[i], acc.rank_total_sq[i])
                cv = None
                if mean is not None and sd is not None and mean != 0:
                    cv = sd / abs(mean)
                values = {
                    "mean": mean,
                    "sd": sd,
                    "cv": cv,
                    "min": acc.minimum[i] if n else None,
                    "max": acc.maximum[i] if n else None,
                    "n": n,
                    "rank_mean": rank_mean,
                    "rank_sd": rank_sd,
                }
                row += [_fmt(values[stat]) for stat in stats]
            for col in categorical_cols:
                counter = categorical_values[col].get(i)
                if not counter:
                    row += ["NA", 0]
                else:
                    row += [counter.most_common(1)[0][0], len(counter)]
            writer.writerow(row)

    log("  wrote %s (%d ids from %d run(s), %d numeric column(s))" % (out_path, kept, used_runs, len(numeric_cols)))

    return _stability_summary(
        spec, run_labels, keys, present_runs, used_runs, accumulators, per_run_metric_values, primary
    )


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra = _avg_ranks(np.asarray(a, dtype=float))
    rb = _avg_ranks(np.asarray(b, dtype=float))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _avg_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(sorted_values):
        j = i + 1
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _stability_summary(spec, run_labels, keys, present_runs, used_runs, accumulators, per_run_metric_values, primary):
    """Pairwise agreement between runs, which is what 'can it replicate?' means.

    Per-id sd answers 'how noisy is this number'. It does not answer 'would a
    reader of the top-100 list get the same list'. Top-K Jaccard and pairwise
    Spearman do, so both are reported here alongside the id-level spread.
    """
    summary = {
        "table": spec.name,
        "runs": run_labels,
        "n_runs": used_runs,
        "n_ids_union": len(keys),
        "n_ids_in_all_runs": sum(1 for c in present_runs if c == used_runs),
        "primary_metric": primary,
        "metrics": {},
    }
    summary["id_presence_stability"] = (
        summary["n_ids_in_all_runs"] / len(keys) if keys else None
    )

    for metric, by_run in per_run_metric_values.items():
        if len(by_run) < 2:
            continue
        labels = [label for label in run_labels if label in by_run]
        common = set(by_run[labels[0]])
        for label in labels[1:]:
            common &= set(by_run[label])
        common_keys = sorted(common)

        entry = {"n_common_ids": len(common_keys), "pairwise_spearman": {}, "top_k_jaccard": {}}

        if len(common_keys) >= 3:
            vectors = {
                label: [by_run[label][key] for key in common_keys] for label in labels
            }
            rhos = []
            for a in range(len(labels)):
                for b in range(a + 1, len(labels)):
                    rho = _spearman(vectors[labels[a]], vectors[labels[b]])
                    if rho is not None:
                        entry["pairwise_spearman"]["%s|%s" % (labels[a], labels[b])] = round(rho, 6)
                        rhos.append(rho)
            if rhos:
                entry["spearman_mean"] = round(sum(rhos) / len(rhos), 6)
                entry["spearman_min"] = round(min(rhos), 6)

        for k in TOP_K_LEVELS:
            tops = {}
            boundary_tied = False
            for label in labels:
                # Tie-break by id, not by file order: several of these columns
                # are written at 3 significant figures, so the K-th and K+1-th
                # values are frequently equal and whichever row the writer
                # happened to emit first would otherwise decide the top-K set.
                # Breaking ties on the id makes the choice identical in every
                # run, so a reported disagreement is a real one.
                items = sorted(by_run[label].items(), key=lambda kv: (-kv[1], kv[0]))
                if len(items) < k + 1:
                    tops = {}
                    break
                if items[k - 1][1] == items[k][1]:
                    boundary_tied = True
                tops[label] = {key for key, _ in items[:k]}
            if not tops:
                continue
            jaccards = []
            for a in range(len(labels)):
                for b in range(a + 1, len(labels)):
                    sa, sb = tops[labels[a]], tops[labels[b]]
                    jaccards.append(len(sa & sb) / len(sa | sb))
            if jaccards:
                entry["top_k_jaccard"]["top_%d" % k] = {
                    "mean": round(sum(jaccards) / len(jaccards), 6),
                    "min": round(min(jaccards), 6),
                    # True means the cut falls inside a block of equal values,
                    # so the number is about the tie-break as much as the model.
                    "boundary_tied": boundary_tied,
                }

        if metric in accumulators:
            acc = accumulators[metric]
            cvs, rank_sds = [], []
            for i in range(len(keys)):
                if acc.n[i] < 2:
                    continue
                mean, sd = _mean_sd(acc.n[i], acc.total[i], acc.total_sq[i])
                if mean and sd is not None and mean != 0:
                    cvs.append(sd / abs(mean))
                _, rank_sd = _mean_sd(acc.n[i], acc.rank_total[i], acc.rank_total_sq[i])
                if rank_sd is not None:
                    rank_sds.append(rank_sd)
            if cvs:
                entry["cv_median"] = round(float(np.median(cvs)), 6)
                entry["cv_p90"] = round(float(np.percentile(cvs, 90)), 6)
            if rank_sds:
                entry["rank_sd_median"] = round(float(np.median(rank_sds)), 6)
                entry["rank_sd_p90"] = round(float(np.percentile(rank_sds, 90)), 6)

        summary["metrics"][metric] = entry

    return summary


def write_stability_report(summaries, out_dir: Path, log=print):
    """Emit the stability diagnostics as both JSON and a long-format TSV."""
    json_path = out_dir / "stability_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summaries, fh, indent=2)

    tsv_path = out_dir / "stability_summary.tsv"
    with open(tsv_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["table", "metric", "statistic", "value"])
        for summary in summaries:
            if summary is None:
                continue
            table = summary["table"]
            for name in ("n_runs", "n_ids_union", "n_ids_in_all_runs", "id_presence_stability"):
                writer.writerow([table, "", name, _fmt(summary.get(name))])
            for metric, entry in summary["metrics"].items():
                for name in (
                    "n_common_ids",
                    "spearman_mean",
                    "spearman_min",
                    "cv_median",
                    "cv_p90",
                    "rank_sd_median",
                    "rank_sd_p90",
                ):
                    if name in entry:
                        writer.writerow([table, metric, name, _fmt(entry[name])])
                for pair, rho in entry.get("pairwise_spearman", {}).items():
                    writer.writerow([table, metric, "spearman[%s]" % pair, _fmt(rho)])
                for level, values in entry.get("top_k_jaccard", {}).items():
                    writer.writerow([table, metric, "jaccard_%s_mean" % level, _fmt(values["mean"])])
                    writer.writerow([table, metric, "jaccard_%s_min" % level, _fmt(values["min"])])
                    if values.get("boundary_tied"):
                        writer.writerow([table, metric, "jaccard_%s_boundary_tied" % level, "1"])
    log("  wrote %s and %s" % (json_path, tsv_path))


def print_headline(summaries, log=print):
    """One screenful: does this trait reproduce across seeds, and where not."""
    log("")
    log("%-20s %-22s %8s %10s %10s %10s" % ("table", "metric", "spearman", "top100_J", "cv_med", "rank_sd_med"))
    log("-" * 86)
    for summary in summaries:
        if summary is None:
            continue
        for metric, entry in summary["metrics"].items():
            top100 = entry.get("top_k_jaccard", {}).get("top_100", {})
            log(
                "%-20s %-22s %8s %10s %10s %10s"
                % (
                    summary["table"],
                    metric,
                    _fmt(entry.get("spearman_mean")),
                    _fmt(top100.get("mean")),
                    _fmt(entry.get("cv_median")),
                    _fmt(entry.get("rank_sd_median")),
                )
            )
        log(
            "%-20s %-22s ids in every run: %d/%d (%s)"
            % (
                summary["table"],
                "",
                summary["n_ids_in_all_runs"],
                summary["n_ids_union"],
                _fmt(summary["id_presence_stability"]),
            )
        )
    log("")


def aggregate_sweep(out_dir: Path, *, tables=None, stats=DEFAULT_STATS, min_runs=1, log=print):
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        run_dirs = [
            run["run_dir"] for run in manifest["runs"] if run.get("returncode") == 0
        ]
        skipped = [run["seed"] for run in manifest["runs"] if run.get("returncode") != 0]
        if skipped:
            log("excluding seeds with non-zero exit: %s" % skipped)
    else:
        run_dirs = sorted(str(p) for p in (out_dir / "runs").glob("seed_*"))

    if not run_dirs:
        raise RuntimeError("no successful runs found under %s" % out_dir)

    agg_dir = out_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    table_names = tables or list(TABLE_SPECS)
    summaries = []
    for name in table_names:
        spec = TABLE_SPECS[name]
        log("aggregating %s across %d run(s)" % (name, len(run_dirs)))
        out_path = agg_dir / ("%s.seed_agg.tsv.gz" % name)
        summaries.append(
            aggregate_table(run_dirs, spec, out_path, stats=stats, min_runs=min_runs, log=log)
        )

    kept = [s for s in summaries if s]

    # Re-aggregating a subset of tables must not delete the other tables'
    # entries from the shared report, so merge onto whatever is already there
    # and let the fresh summaries win by table name.
    merged = _merge_with_existing(kept, agg_dir / "stability_summary.json")
    write_stability_report(merged, agg_dir, log=log)
    print_headline(kept, log=log)
    return summaries


def _merge_with_existing(fresh, json_path: Path):
    if not json_path.exists():
        return fresh
    try:
        with open(json_path) as fh:
            prior = json.load(fh)
    except (OSError, ValueError):
        return fresh
    if not isinstance(prior, list):
        return fresh
    fresh_tables = {s["table"] for s in fresh}
    carried = [s for s in prior if isinstance(s, dict) and s.get("table") not in fresh_tables]
    by_name = {s["table"]: s for s in fresh + carried}
    return [by_name[name] for name in TABLE_SPECS if name in by_name]
