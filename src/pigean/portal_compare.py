"""Two-run comparison queries for the PIGEAN Comparer portal.

Pure functions over an open `portal_db` connection. Both runs' thresholded rows are pulled into
memory (a run keeps at most tens of thousands of genes and a few thousand gene sets), merged in
Python as a full outer join on the gene / gene-set id, and summarised. Ranks come from the
`rank_*` columns written at build time over the full PIGEAN output (schema v5), so a rank is
meaningful even though only rows passing the build thresholds are stored.

Standard library only.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Iterable, Optional

GENE_METRICS = ("combined", "prior", "log_bf")
GENE_SET_METRICS = ("beta", "beta_uncorrected")
GENE_EXTRA = ("huge_score", "n")
GENE_SET_EXTRA = ("label", "n")
TOP_N_CHOICES = (50, 100, 250, 500, 1000, 0)  # 0 = all common


def _run_exists(conn: sqlite3.Connection, run_id: str) -> bool:
    return conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone() is not None


def _fetch(conn: sqlite3.Connection, table: str, key: str, run_id: str, columns: Iterable[str]) -> dict[str, dict]:
    cols = ", ".join([key, *columns])
    return {row[key]: dict(row) for row in conn.execute(f"SELECT {cols} FROM {table} WHERE run_id=?", (run_id,))}


def _merge(a: dict[str, dict], b: dict[str, dict], key: str, metrics: tuple[str, ...], extra: tuple[str, ...]) -> list[dict]:
    """Full outer join of two {id: row} maps -> rows with a_*/b_*/delta_* columns and a status."""
    rows = []
    for ident in a.keys() | b.keys():
        ra, rb = a.get(ident), b.get(ident)
        row: dict = {key: ident, "status": "both" if ra and rb else ("a_only" if ra else "b_only")}
        for m in metrics:
            va = ra.get(m) if ra else None
            vb = rb.get(m) if rb else None
            row[f"a_{m}"], row[f"b_{m}"] = va, vb
            row[f"delta_{m}"] = (vb - va) if (va is not None and vb is not None) else None
            rka = ra.get(f"rank_{m}") if ra else None
            rkb = rb.get(f"rank_{m}") if rb else None
            row[f"a_rank_{m}"], row[f"b_rank_{m}"] = rka, rkb
            row[f"delta_rank_{m}"] = (rkb - rka) if (rka is not None and rkb is not None) else None
        for e in extra:
            row[f"a_{e}"] = ra.get(e) if ra else None
            row[f"b_{e}"] = rb.get(e) if rb else None
        rows.append(row)
    return rows


def _sort_rows(rows: list[dict], metric: str, sort: str) -> list[dict]:
    """sort: abs_delta_rank (default), abs_delta, a_rank, b_rank, delta, delta_rank, id."""
    def missing_last(value):
        return (value is None, value)

    if sort == "a_rank":
        rows.sort(key=lambda r: missing_last(r.get(f"a_rank_{metric}")))
    elif sort == "b_rank":
        rows.sort(key=lambda r: missing_last(r.get(f"b_rank_{metric}")))
    elif sort == "delta":
        rows.sort(key=lambda r: (r.get(f"delta_{metric}") is None, -(r.get(f"delta_{metric}") or 0)))
    elif sort == "delta_rank":
        rows.sort(key=lambda r: (r.get(f"delta_rank_{metric}") is None, r.get(f"delta_rank_{metric}") or 0))
    elif sort == "abs_delta":
        rows.sort(key=lambda r: (r.get(f"delta_{metric}") is None, -abs(r.get(f"delta_{metric}") or 0)))
    elif sort == "id":
        rows.sort(key=lambda r: next(iter(r.values())))
    else:  # abs_delta_rank: biggest rank moves first; one-sided entries after, ordered by their own rank
        rows.sort(key=lambda r: (r.get(f"delta_rank_{metric}") is None, -abs(r.get(f"delta_rank_{metric}") or 0),
                                 min(x for x in (r.get(f"a_rank_{metric}"), r.get(f"b_rank_{metric}"), 10**9) if x is not None)))
    return rows


def _filter_search(rows: list[dict], key: str, search: str, extra_keys: tuple[str, ...] = ()) -> list[dict]:
    if not search:
        return rows
    q = search.lower()
    return [r for r in rows if q in str(r[key]).lower() or any(q in str(r.get(k) or "").lower() for k in extra_keys)]


def compare_genes(conn: sqlite3.Connection, a: str, b: str, *, metric: str = "combined", search: str = "",
                  sort: str = "abs_delta_rank", limit: int = 5000, status: str = "") -> dict:
    metric = metric if metric in GENE_METRICS else "combined"
    cols = [*GENE_METRICS, *(f"rank_{m}" for m in GENE_METRICS), *GENE_EXTRA]
    rows = _merge(_fetch(conn, "genes", "gene", a, cols), _fetch(conn, "genes", "gene", b, cols), "gene", GENE_METRICS, GENE_EXTRA)
    if status in ("both", "a_only", "b_only"):
        rows = [r for r in rows if r["status"] == status]
    rows = _filter_search(rows, "gene", search)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("both", "a_only", "b_only")}
    rows = _sort_rows(rows, metric, sort)
    return {"metric": metric, "counts": counts, "total": len(rows), "rows": rows[: max(1, int(limit))]}


def compare_gene_sets(conn: sqlite3.Connection, a: str, b: str, *, metric: str = "beta", search: str = "",
                      sort: str = "abs_delta_rank", limit: int = 5000, status: str = "") -> dict:
    metric = metric if metric in GENE_SET_METRICS else "beta"
    cols = [*GENE_SET_METRICS, *(f"rank_{m}" for m in GENE_SET_METRICS), *GENE_SET_EXTRA]
    rows = _merge(_fetch(conn, "gene_sets", "gene_set", a, cols), _fetch(conn, "gene_sets", "gene_set", b, cols),
                  "gene_set", GENE_SET_METRICS, GENE_SET_EXTRA)
    for r in rows:
        r["label"] = r.get("a_label") or r.get("b_label") or ""
    if status in ("both", "a_only", "b_only"):
        rows = [r for r in rows if r["status"] == status]
    rows = _filter_search(rows, "gene_set", search, ("label",))
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("both", "a_only", "b_only")}
    rows = _sort_rows(rows, metric, sort)
    return {"metric": metric, "counts": counts, "total": len(rows), "rows": rows[: max(1, int(limit))]}


# ---------------------------------------------------------------------------------------------
# correlations (stdlib)

def pearson(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _fractional_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    return pearson(_fractional_ranks(x), _fractional_ranks(y))


def _metric_summary(rows: list[dict], metric: str, top_n: int) -> dict:
    """Correlations on the union of A's and B's top-N (by that metric's rank), plus overlap."""
    both = [r for r in rows if r["status"] == "both" and r[f"a_{metric}"] is not None and r[f"b_{metric}"] is not None]
    if top_n and top_n > 0:
        top_a = {i for i, r in enumerate(rows) if (r.get(f"a_rank_{metric}") or 10**9) <= top_n}
        top_b = {i for i, r in enumerate(rows) if (r.get(f"b_rank_{metric}") or 10**9) <= top_n}
        union = top_a | top_b
        selected = [r for i, r in enumerate(rows) if i in union and r["status"] == "both"
                    and r[f"a_{metric}"] is not None and r[f"b_{metric}"] is not None]
        overlap = len(top_a & top_b)
        jaccard = overlap / len(union) if union else None
        n_top_a, n_top_b = len(top_a), len(top_b)
    else:
        selected, overlap, jaccard, n_top_a, n_top_b = both, None, None, None, None
    xs = [r[f"a_{metric}"] for r in selected]
    ys = [r[f"b_{metric}"] for r in selected]
    return {
        "metric": metric, "top_n": top_n, "n": len(selected), "n_common": len(both),
        "pearson": pearson(xs, ys), "spearman": spearman(xs, ys),
        "n_top_a": n_top_a, "n_top_b": n_top_b, "overlap": overlap, "jaccard": jaccard,
        "rank_pearson_all": pearson([r[f"a_rank_{metric}"] for r in both if r[f"a_rank_{metric}"] and r[f"b_rank_{metric}"]],
                                    [r[f"b_rank_{metric}"] for r in both if r[f"a_rank_{metric}"] and r[f"b_rank_{metric}"]]),
    }


def compare_summary(conn: sqlite3.Connection, a: str, b: str, *, top_n: int = 100) -> dict:
    genes = compare_genes(conn, a, b, limit=10**7)
    gene_sets = compare_gene_sets(conn, a, b, limit=10**7)
    run_a = dict(conn.execute("SELECT * FROM runs WHERE run_id=?", (a,)).fetchone())
    run_b = dict(conn.execute("SELECT * FROM runs WHERE run_id=?", (b,)).fetchone())
    ranks_available = all(r.get("n_ranked_genes") for r in (run_a, run_b))
    return {
        "a": a, "b": b, "top_n": top_n, "ranks_available": ranks_available,
        "genes": {"counts": genes["counts"], "metrics": [_metric_summary(genes["rows"], m, top_n) for m in GENE_METRICS]},
        "gene_sets": {"counts": gene_sets["counts"], "metrics": [_metric_summary(gene_sets["rows"], m, top_n) for m in GENE_SET_METRICS]},
        "ranked": {"a_genes": run_a.get("n_ranked_genes"), "b_genes": run_b.get("n_ranked_genes"),
                   "a_gene_sets": run_a.get("n_ranked_gene_sets"), "b_gene_sets": run_b.get("n_ranked_gene_sets")},
    }


def _norm_param(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        return repr(number) if not number.is_integer() else str(int(number))
    except ValueError:
        return text


def compare_params(conn: sqlite3.Connection, a: str, b: str) -> dict:
    pa = {(r["parameter"], r["version"]): r["value"] for r in conn.execute("SELECT parameter, version, value FROM run_params WHERE run_id=?", (a,))}
    pb = {(r["parameter"], r["version"]): r["value"] for r in conn.execute("SELECT parameter, version, value FROM run_params WHERE run_id=?", (b,))}
    rows = []
    for key in sorted(pa.keys() | pb.keys()):
        va, vb = pa.get(key), pb.get(key)
        rows.append({"parameter": key[0], "version": key[1], "a_value": va, "b_value": vb,
                     "differs": _norm_param(va) != _norm_param(vb)})
    return {"a": a, "b": b, "n_params": len(rows), "n_differ": sum(1 for r in rows if r["differs"]),
            "a_has_params": bool(pa), "b_has_params": bool(pb), "rows": rows}


def lookup(conn: sqlite3.Connection, a: str, b: str, *, kind: str, ident: str) -> Optional[dict]:
    if kind == "gene":
        rows = [r for r in compare_genes(conn, a, b, limit=10**7)["rows"] if r["gene"] == ident]
    else:
        rows = [r for r in compare_gene_sets(conn, a, b, limit=10**7)["rows"] if r["gene_set"] == ident]
    return rows[0] if rows else None


def validate_pair(conn: sqlite3.Connection, a: str, b: str) -> Optional[tuple[int, str]]:
    """(status, message) if the pair is unusable, else None."""
    if not a or not b:
        return 400, "both 'a' and 'b' run ids are required"
    if a == b:
        return 400, "'a' and 'b' must be different runs"
    for run_id in (a, b):
        if not _run_exists(conn, run_id):
            return 404, f"unknown run '{run_id}'"
    return None
