"""SQLite storage for the PIGEAN results portal.

`build_database` reads one or more PIGEAN runs (gene stats, gene-set stats and the
gene x gene-set loading table), applies user thresholds, and writes a compact SQLite
file. The query helpers at the bottom are what `pigean.portal_server` serves as JSON.

Only the standard library is used, matching the rest of the package.
"""

from __future__ import annotations

import csv
import json
import operator
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .dashboard import _first, open_text, parse_float

SCHEMA_VERSION = 5

GENE_COLUMN_ALIASES = {
    "gene": ["Gene", "gene", "id", "ID"],
    "prior": ["prior", "Indirect", "indirect", "indirect_log_bf"],
    "combined": ["combined", "Combined", "combined_log_bf"],
    "log_bf": ["log_bf", "Direct", "direct", "direct_log_bf"],
    "huge_score": ["huge_score_gwas", "huge_score", "positive_control"],
    "n": ["N", "n"],
    "chrom": ["Chrom", "chrom", "chr"],
    "start": ["Start", "start"],
    "end": ["End", "end"],
}
GENE_SET_COLUMN_ALIASES = {
    "gene_set": ["Gene_Set", "gene_set", "id", "ID"],
    "label": ["label", "Label"],
    "n": ["N", "n"],
    "beta": ["beta", "Beta"],
    "beta_uncorrected": ["beta_uncorrected", "Beta_uncorrected"],
    "p_orig": ["P_orig", "p_orig", "P"],
    "z_orig": ["Z_orig", "z_orig", "Z"],
}
LOADING_COLUMN_ALIASES = {
    "gene": ["Gene", "gene"],
    "gene_set": ["gene_set", "Gene_Set"],
    "beta": ["beta", "Beta"],
    "weight": ["weight", "Weight"],
    "prior": ["prior"],
    "combined": ["combined"],
    "log_bf": ["log_bf"],
    "huge_score": ["huge_score_gwas", "huge_score"],
}
NUMERIC_GENE_COLUMNS = ["prior", "combined", "log_bf", "huge_score", "n", "start", "end"]
NUMERIC_GENE_SET_COLUMNS = ["n", "beta", "beta_uncorrected", "p_orig", "z_orig"]
NUMERIC_LOADING_COLUMNS = ["beta", "weight", "prior", "combined", "log_bf", "huge_score"]

GENE_STATS_CANDIDATES = ["pigean.gene_stats.out.gz", "pigean.gene_stats.out", "gene_stats.tsv", "gene_stats.out.gz"]
GENE_SET_STATS_CANDIDATES = ["pigean.gene_set_stats.out.gz", "pigean.gene_set_stats.out", "gene_set_stats.tsv", "gene_set_stats.out.gz"]
LOADING_CANDIDATES = ["pigean.gene_gene_set_stats.out.gz", "pigean.gene_gene_set_stats.out", "gene_gene_set_stats.tsv", "gene_gene_set_stats.out.gz"]

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "!=": operator.ne,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
}
_FILTER_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|==|>|<|=)\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*$")


@dataclass(frozen=True)
class Filter:
    """One numeric threshold, e.g. `prior>1` or `beta>=0.01`."""

    column: str
    op: str
    value: float

    @property
    def expr(self) -> str:
        return f"{self.column}{self.op}{self.value:g}"

    def passes(self, row: dict) -> bool:
        value = row.get(self.column)
        if value is None:
            return False
        return _OPS[self.op](value, self.value)


def parse_filter(expr: str) -> Filter:
    """Parse `column op number` (ops: > >= < <= == !=)."""
    match = _FILTER_RE.match(expr)
    if not match:
        raise ValueError(f"invalid filter '{expr}'; expected e.g. prior>1 or beta>=0.01")
    column, op, value = match.groups()
    return Filter(column, op, float(value))


def passes_filters(row: dict, filters: list[Filter], mode: str) -> bool:
    """`any`: keep a row if at least one filter passes; `all`: every filter must pass."""
    if not filters:
        return True
    results = (f.passes(row) for f in filters)
    return any(results) if mode == "any" else all(results)


_RUN_ID_RE = re.compile(r"^(?P<model>.+?)__(?P<trait>[^_].*?)(?:__s(?P<seed>\d+))?$")


@dataclass
class RunFiles:
    """Input files plus metadata for one run. `gene_gene_set_stats` is optional.

    `model` / `trait` / `seed` drive the portal's model -> trait -> run selectors. When not
    given explicitly they are inferred from a `<model>__<trait>[__s<seed>]` run id.
    """

    run_id: str
    gene_stats: Path
    gene_set_stats: Path
    gene_gene_set_stats: Optional[Path] = None
    title: str = ""
    model: str = ""
    model_title: str = ""
    trait: str = ""
    seed: str = ""          # run label within (model, trait): "main", "s1", "run2", ...
    params: Optional[Path] = None   # PIGEAN params.out / params.tsv (Parameter/Version/Value) for provenance
    warnings: list[str] = field(default_factory=list)

    def infer_metadata(self) -> None:
        """Fill blank model/trait/seed from the run id if it follows the LAP naming pattern."""
        match = _RUN_ID_RE.match(self.run_id)
        if not match:
            return
        self.model = self.model or match["model"]
        self.trait = self.trait or match["trait"]
        self.seed = self.seed or (match["seed"] or "")


def _first_existing(directory: Path, names: Iterable[str], glob_pattern: str, exclude: str = "") -> Optional[Path]:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    for candidate in sorted(directory.glob(glob_pattern)):
        # LAP writes hidden .<name>.cmd/.finished sidecars next to outputs; pathlib's glob matches dotfiles
        if candidate.name.startswith(".") or (exclude and exclude in candidate.name):
            continue
        return candidate
    return None


def resolve_run_dir(run_id: str, directory: Path) -> RunFiles:
    """Locate the three stats tables in a run directory (dashboard-style or LAP-style names)."""
    gene = _first_existing(directory, GENE_STATS_CANDIDATES, "*.gene_stats.*", exclude="gene_gene_set")
    gene_set = _first_existing(directory, GENE_SET_STATS_CANDIDATES, "*.gene_set_stats.*", exclude="gene_gene_set")
    loading = _first_existing(directory, LOADING_CANDIDATES, "*.gene_gene_set_stats.*")
    if gene is None or gene_set is None:
        raise FileNotFoundError(f"run '{run_id}': could not find gene_stats and gene_set_stats tables in {directory}")
    files = RunFiles(run_id=run_id, gene_stats=gene, gene_set_stats=gene_set, gene_gene_set_stats=loading)
    if loading is None:
        files.warnings.append(f"run '{run_id}': no gene_gene_set_stats table found in {directory}; loadings will be empty")
    return files


def _iter_rows(path: Path) -> Iterator[dict[str, str]]:
    with open_text(path) as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _normalize(row: dict[str, str], aliases: dict[str, list[str]], numeric: list[str]) -> dict:
    out: dict = {}
    for key, names in aliases.items():
        raw = _first(row, names, None)
        out[key] = parse_float(raw) if key in numeric else raw
    return out


def _extra_json(row: dict[str, str], keep_numeric: bool = True) -> str:
    """All original columns, numbers parsed where possible, for the detail panels."""
    payload = {}
    for key, value in row.items():
        number = parse_float(value)
        payload[key] = number if (number is not None and keep_numeric) else value
    return json.dumps(payload, separators=(",", ":"))


GENE_RANK_METRICS = ("combined", "prior", "log_bf")
GENE_SET_RANK_METRICS = ("beta", "beta_uncorrected")


def compute_ranks(path: Path, id_aliases: list[str], metric_aliases: dict[str, list[str]]) -> tuple[dict[str, dict[str, int]], int]:
    """
    Competition ranks (1 = highest; ties share the lowest rank; missing values unranked) of every
    row in a PIGEAN table, per metric, computed over the FULL file before any thresholding.

    Returns:
        ({metric: {id: rank}}, n_rows_with_id)
    """
    values: dict[str, list[tuple[str, float]]] = {m: [] for m in metric_aliases}
    n = 0
    for raw in _iter_rows(path):
        ident = _first(raw, id_aliases, "")
        if not ident:
            continue
        n += 1
        for metric, names in metric_aliases.items():
            value = parse_float(_first(raw, names, None))
            if value is not None:
                values[metric].append((ident, value))
    ranks: dict[str, dict[str, int]] = {}
    for metric, pairs in values.items():
        pairs.sort(key=lambda item: -item[1])
        out: dict[str, int] = {}
        current_rank, previous = 0, None
        for position, (ident, value) in enumerate(pairs, start=1):
            if value != previous:
                current_rank, previous = position, value
            out.setdefault(ident, current_rank)
        ranks[metric] = out
    return ranks, n


DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, title TEXT, model TEXT, trait TEXT, seed TEXT,
    gene_stats_path TEXT, gene_set_stats_path TEXT,
    gene_gene_set_stats_path TEXT, n_genes INTEGER, n_gene_sets INTEGER, n_loadings INTEGER,
    n_genes_input INTEGER, n_gene_sets_input INTEGER, n_loadings_input INTEGER,
    filters_json TEXT, warnings_json TEXT, built_at TEXT, model_title TEXT DEFAULT '', params_path TEXT DEFAULT '',
    n_ranked_genes INTEGER, n_ranked_gene_sets INTEGER
);
CREATE TABLE IF NOT EXISTS run_params (
    run_id TEXT NOT NULL, parameter TEXT NOT NULL, version TEXT NOT NULL, value TEXT, PRIMARY KEY (run_id, parameter, version)
);
CREATE TABLE IF NOT EXISTS genes (
    run_id TEXT NOT NULL, gene TEXT NOT NULL, prior REAL, combined REAL, log_bf REAL, huge_score REAL,
    n REAL, chrom TEXT, start REAL, end REAL, extra_json TEXT,
    rank_combined INTEGER, rank_prior INTEGER, rank_log_bf INTEGER, PRIMARY KEY (run_id, gene)
);
CREATE TABLE IF NOT EXISTS gene_sets (
    run_id TEXT NOT NULL, gene_set TEXT NOT NULL, label TEXT, n REAL, beta REAL, beta_uncorrected REAL,
    p_orig REAL, z_orig REAL, extra_json TEXT,
    rank_beta INTEGER, rank_beta_uncorrected INTEGER, PRIMARY KEY (run_id, gene_set)
);
CREATE TABLE IF NOT EXISTS gene_gene_sets (
    run_id TEXT NOT NULL, gene_set TEXT NOT NULL, gene TEXT NOT NULL, beta REAL, weight REAL,
    prior REAL, combined REAL, log_bf REAL, huge_score REAL, PRIMARY KEY (run_id, gene_set, gene)
);
CREATE TABLE IF NOT EXISTS phenotypes (
    legacy_id TEXT PRIMARY KEY, portal_id TEXT, name TEXT, description TEXT, trait_group TEXT,
    legacy_trait_group TEXT, trait_type TEXT, is_dichotomous TEXT, mapping_count INTEGER
);
CREATE TABLE IF NOT EXISTS phenotype_mappings (
    legacy_id TEXT NOT NULL, target_id TEXT NOT NULL, target_label TEXT, target_ontology TEXT,
    predicate TEXT, confidence REAL, justification TEXT, source TEXT, PRIMARY KEY (legacy_id, target_id)
);
CREATE INDEX IF NOT EXISTS idx_genes_prior ON genes (run_id, prior);
CREATE INDEX IF NOT EXISTS idx_genes_combined ON genes (run_id, combined);
CREATE INDEX IF NOT EXISTS idx_gene_sets_beta ON gene_sets (run_id, beta);
CREATE INDEX IF NOT EXISTS idx_gene_sets_beta_unc ON gene_sets (run_id, beta_uncorrected);
CREATE INDEX IF NOT EXISTS idx_loadings_gene ON gene_gene_sets (run_id, gene);
"""


def open_database(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# Columns of dig-portal-data-models' versions/phenotype/v*/portal_phenotypes_flat.tsv (one row per
# ontology mapping; phenotype-level fields repeat on every row of the same legacy id).
PHENOTYPE_ID_COLUMNS = ["phenotype", "legacy_phenotype_id"]


def load_phenotypes(conn: sqlite3.Connection, path: Path, legacy_ids: set[str]) -> dict:
    """
    Load portal phenotype names / ids / ontology mappings for the given legacy ids.

    Args:
        conn: Open writable connection.
        path: `portal_phenotypes_flat.tsv` from dig-portal-data-models.
        legacy_ids: Legacy phenotype ids (the pipeline's trait names) to keep.

    Returns:
        dict: `{"n_phenotypes": int, "n_mappings": int, "missing": [ids not found]}`.
    """
    found: set[str] = set()
    pheno_rows: dict[str, tuple] = {}
    map_rows: list[tuple] = []
    for raw in _iter_rows(path):
        legacy = _first(raw, PHENOTYPE_ID_COLUMNS, "")
        if legacy not in legacy_ids:
            continue
        found.add(legacy)
        if legacy not in pheno_rows:
            pheno_rows[legacy] = (
                legacy, raw.get("portal_id", ""), _first(raw, ["phenotype_name", "name"], legacy),
                raw.get("description", ""), raw.get("trait_group", ""), raw.get("legacy_trait_group", ""),
                raw.get("trait_type", ""), raw.get("is_dichotomous", ""), int(parse_float(raw.get("mapping_count"), 0) or 0),
            )
        target = raw.get("target_id", "")
        if target:
            map_rows.append((legacy, target, raw.get("target_label", ""), raw.get("target_ontology", ""),
                             raw.get("mapping_predicate", ""), parse_float(raw.get("confidence")),
                             raw.get("mapping_justification", ""), raw.get("source", "")))
    for legacy in found:
        conn.execute("DELETE FROM phenotypes WHERE legacy_id=?", (legacy,))
        conn.execute("DELETE FROM phenotype_mappings WHERE legacy_id=?", (legacy,))
    conn.executemany("INSERT OR REPLACE INTO phenotypes VALUES (?,?,?,?,?,?,?,?,?)", list(pheno_rows.values()))
    conn.executemany("INSERT OR REPLACE INTO phenotype_mappings VALUES (?,?,?,?,?,?,?,?)", map_rows)
    return {"n_phenotypes": len(pheno_rows), "n_mappings": len(map_rows), "missing": sorted(legacy_ids - found)}


@dataclass
class BuildOptions:
    gene_filters: list[Filter] = field(default_factory=list)
    gene_set_filters: list[Filter] = field(default_factory=list)
    loading_filters: list[Filter] = field(default_factory=list)
    filter_mode: str = "any"
    append: bool = False
    # keep loadings only for retained gene sets; genes in loadings need not pass gene filters
    restrict_loadings_to_gene_sets: bool = True
    # dig-portal-data-models portal_phenotypes_flat.tsv: names / portal ids / mappings per trait
    phenotype_file: Optional[Path] = None


def _load_run(conn: sqlite3.Connection, files: RunFiles, options: BuildOptions) -> dict:
    """Read, filter and insert one run. Returns the summary row written to `runs`."""
    warnings = list(files.warnings)
    run_id = files.run_id
    conn.execute("DELETE FROM genes WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM gene_sets WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM gene_gene_sets WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM run_params WHERE run_id=?", (run_id,))
    if files.params is not None:
        param_rows = []
        for raw in _iter_rows(files.params):
            key = _first(raw, ["Parameter", "parameter", "key"], "")
            if key:
                param_rows.append((run_id, key, _first(raw, ["Version", "version"], "1"), _first(raw, ["Value", "value"], "")))
        conn.executemany("INSERT OR REPLACE INTO run_params VALUES (?,?,?,?)", param_rows)

    gene_ranks, n_ranked_genes = compute_ranks(
        files.gene_stats, GENE_COLUMN_ALIASES["gene"], {m: GENE_COLUMN_ALIASES[m] for m in GENE_RANK_METRICS})
    n_genes_in = n_genes = 0
    gene_rows = []
    for raw in _iter_rows(files.gene_stats):
        n_genes_in += 1
        row = _normalize(raw, GENE_COLUMN_ALIASES, NUMERIC_GENE_COLUMNS)
        if not row["gene"] or not passes_filters(row, options.gene_filters, options.filter_mode):
            continue
        gene_rows.append((run_id, row["gene"], row["prior"], row["combined"], row["log_bf"], row["huge_score"],
                          row["n"], row["chrom"], row["start"], row["end"], _extra_json(raw),
                          *(gene_ranks[m].get(row["gene"]) for m in GENE_RANK_METRICS)))
    conn.executemany(
        "INSERT OR REPLACE INTO genes (run_id, gene, prior, combined, log_bf, huge_score, n, chrom, start, end, "
        "extra_json, rank_combined, rank_prior, rank_log_bf) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", gene_rows)
    n_genes = len(gene_rows)
    if n_genes_in and not n_genes:
        warnings.append(f"run '{run_id}': no genes passed the gene filters")

    gs_ranks, n_ranked_gene_sets = compute_ranks(
        files.gene_set_stats, GENE_SET_COLUMN_ALIASES["gene_set"], {m: GENE_SET_COLUMN_ALIASES[m] for m in GENE_SET_RANK_METRICS})
    n_gene_sets_in = 0
    gene_set_rows = []
    kept_gene_sets: set[str] = set()
    for raw in _iter_rows(files.gene_set_stats):
        n_gene_sets_in += 1
        row = _normalize(raw, GENE_SET_COLUMN_ALIASES, NUMERIC_GENE_SET_COLUMNS)
        if not row["gene_set"] or not passes_filters(row, options.gene_set_filters, options.filter_mode):
            continue
        kept_gene_sets.add(row["gene_set"])
        gene_set_rows.append((run_id, row["gene_set"], row["label"] or "", row["n"], row["beta"],
                              row["beta_uncorrected"], row["p_orig"], row["z_orig"], _extra_json(raw),
                              *(gs_ranks[m].get(row["gene_set"]) for m in GENE_SET_RANK_METRICS)))
    conn.executemany(
        "INSERT OR REPLACE INTO gene_sets (run_id, gene_set, label, n, beta, beta_uncorrected, p_orig, z_orig, extra_json, "
        "rank_beta, rank_beta_uncorrected) VALUES (?,?,?,?,?,?,?,?,?,?,?)", gene_set_rows)
    n_gene_sets = len(gene_set_rows)
    if n_gene_sets_in and not n_gene_sets:
        warnings.append(f"run '{run_id}': no gene sets passed the gene-set filters")

    n_loadings_in = n_loadings = 0
    if files.gene_gene_set_stats is not None:
        batch = []
        for raw in _iter_rows(files.gene_gene_set_stats):
            n_loadings_in += 1
            row = _normalize(raw, LOADING_COLUMN_ALIASES, NUMERIC_LOADING_COLUMNS)
            if not row["gene"] or not row["gene_set"]:
                continue
            if options.restrict_loadings_to_gene_sets and row["gene_set"] not in kept_gene_sets:
                continue
            if not passes_filters(row, options.loading_filters, options.filter_mode):
                continue
            batch.append((run_id, row["gene_set"], row["gene"], row["beta"], row["weight"], row["prior"],
                          row["combined"], row["log_bf"], row["huge_score"]))
            if len(batch) >= 50000:
                conn.executemany("INSERT OR REPLACE INTO gene_gene_sets VALUES (?,?,?,?,?,?,?,?,?)", batch)
                n_loadings += len(batch)
                batch = []
        if batch:
            conn.executemany("INSERT OR REPLACE INTO gene_gene_sets VALUES (?,?,?,?,?,?,?,?,?)", batch)
            n_loadings += len(batch)

    filters_json = json.dumps({
        "mode": options.filter_mode,
        "genes": [f.expr for f in options.gene_filters],
        "gene_sets": [f.expr for f in options.gene_set_filters],
        "loadings": [f.expr for f in options.loading_filters],
    })
    summary = {
        "run_id": run_id, "title": files.title or run_id,
        "model": files.model, "model_title": files.model_title, "trait": files.trait, "seed": files.seed,
        "params_path": str(files.params) if files.params else "",
        "gene_stats_path": str(files.gene_stats), "gene_set_stats_path": str(files.gene_set_stats),
        "gene_gene_set_stats_path": str(files.gene_gene_set_stats) if files.gene_gene_set_stats else "",
        "n_genes": n_genes, "n_gene_sets": n_gene_sets, "n_loadings": n_loadings,
        "n_genes_input": n_genes_in, "n_gene_sets_input": n_gene_sets_in, "n_loadings_input": n_loadings_in,
        "n_ranked_genes": n_ranked_genes, "n_ranked_gene_sets": n_ranked_gene_sets,
        "filters_json": filters_json, "warnings_json": json.dumps(warnings),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id,title,model,trait,seed,gene_stats_path,gene_set_stats_path,"
        "gene_gene_set_stats_path,n_genes,n_gene_sets,n_loadings,n_genes_input,n_gene_sets_input,n_loadings_input,"
        "filters_json,warnings_json,built_at,model_title,params_path,n_ranked_genes,n_ranked_gene_sets) VALUES (:run_id,:title,:model,:trait,:seed,"
        ":gene_stats_path,:gene_set_stats_path,:gene_gene_set_stats_path,:n_genes,:n_gene_sets,:n_loadings,"
        ":n_genes_input,:n_gene_sets_input,:n_loadings_input,:filters_json,:warnings_json,:built_at,:model_title,:params_path,"
        ":n_ranked_genes,:n_ranked_gene_sets)",
        summary,
    )
    summary["warnings"] = warnings
    return summary


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after schema v1 to an existing (appended-to) database."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    for column in ("model", "trait", "seed", "model_title", "params_path"):
        if column not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT DEFAULT ''")
    for column in ("n_ranked_genes", "n_ranked_gene_sets"):
        if column not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column} INTEGER")
    # schema v5: ranks over the full (pre-threshold) PIGEAN output, used by the Comparer
    for table, columns in (("genes", ("rank_combined", "rank_prior", "rank_log_bf")),
                           ("gene_sets", ("rank_beta", "rank_beta_uncorrected"))):
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column in columns:
            if column not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} INTEGER")


def build_database(db_path: Path, runs: list[RunFiles], options: BuildOptions) -> list[dict]:
    """Create (or append to) the portal SQLite file. Returns one summary dict per run."""
    if db_path.exists() and not options.append:
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_database(db_path)
    try:
        conn.executescript(DDL)
        _migrate(conn)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        summaries = []
        for files in runs:
            summaries.append(_load_run(conn, files, options))
            conn.commit()
        if options.phenotype_file is not None:
            # cover every trait in the database, including runs kept from an earlier --append build
            traits = {row[0] for row in conn.execute("SELECT DISTINCT trait FROM runs WHERE trait != ''")}
            info = load_phenotypes(conn, options.phenotype_file, traits)
            conn.execute("INSERT OR REPLACE INTO meta VALUES ('phenotype_file', ?)", (str(options.phenotype_file),))
            conn.commit()
            for legacy in info["missing"]:
                warning = f"trait '{legacy}' not found in {options.phenotype_file}"
                for row in conn.execute("SELECT run_id, warnings_json FROM runs WHERE trait=?", (legacy,)).fetchall():
                    existing = json.loads(row["warnings_json"] or "[]")
                    if warning not in existing:
                        existing.append(warning)
                        conn.execute("UPDATE runs SET warnings_json=? WHERE run_id=?", (json.dumps(existing), row["run_id"]))
                for summary in summaries:
                    if summary["trait"] == legacy and warning not in summary["warnings"]:
                        summary["warnings"].append(warning)
            conn.commit()
            if summaries:
                summaries[-1]["phenotypes"] = info
        conn.execute("VACUUM")
        conn.commit()
        return summaries
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# Query helpers (used by the HTTP API and directly testable)

def _rows(cursor: sqlite3.Cursor) -> list[dict]:
    return [dict(row) for row in cursor.fetchall()]


def list_phenotypes(conn: sqlite3.Connection) -> dict[str, dict]:
    """Phenotype metadata keyed by legacy id, each with its ontology `mappings` list."""
    phenotypes = {row["legacy_id"]: dict(row, mappings=[]) for row in conn.execute("SELECT * FROM phenotypes")}
    for row in conn.execute("SELECT * FROM phenotype_mappings ORDER BY legacy_id, confidence DESC, target_ontology, target_id"):
        entry = phenotypes.get(row["legacy_id"])
        if entry is not None:
            mapping = dict(row)
            mapping.pop("legacy_id")
            entry["mappings"].append(mapping)
    return phenotypes


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    runs = _rows(conn.execute("SELECT * FROM runs ORDER BY model, trait, seed, run_id"))
    phenotypes = list_phenotypes(conn)
    for run in runs:
        run["filters"] = json.loads(run.pop("filters_json") or "{}")
        run["warnings"] = json.loads(run.pop("warnings_json") or "[]")
        run["phenotype"] = phenotypes.get(run.get("trait") or "")
    return runs


def _like(value: str) -> str:
    return f"%{value}%"


def query_genes(conn: sqlite3.Connection, run_id: str, *, min_prior: Optional[float] = None,
                min_log_bf: Optional[float] = None, min_combined: Optional[float] = None,
                search: str = "", sort: str = "combined", limit: int = 5000) -> list[dict]:
    sort_col = sort if sort in ("combined", "prior", "log_bf", "huge_score", "gene") else "combined"
    sql = "SELECT gene, prior, combined, log_bf, huge_score, n, chrom, start, end, rank_combined, rank_prior, rank_log_bf FROM genes WHERE run_id=?"
    params: list = [run_id]
    for col, value in (("prior", min_prior), ("log_bf", min_log_bf), ("combined", min_combined)):
        if value is not None:
            sql += f" AND {col} >= ?"
            params.append(value)
    if search:
        sql += " AND gene LIKE ?"
        params.append(_like(search))
    direction = "ASC" if sort_col == "gene" else "DESC"
    sql += f" ORDER BY {sort_col} {direction} LIMIT ?"
    params.append(max(1, min(int(limit), 100000)))
    return _rows(conn.execute(sql, params))


def query_gene_sets(conn: sqlite3.Connection, run_id: str, *, min_beta: Optional[float] = None,
                    min_beta_uncorrected: Optional[float] = None, search: str = "",
                    sort: str = "beta", limit: int = 500) -> list[dict]:
    sort_col = sort if sort in ("beta", "beta_uncorrected", "n", "gene_set") else "beta"
    sql = "SELECT gene_set, label, n, beta, beta_uncorrected, p_orig, z_orig, rank_beta, rank_beta_uncorrected FROM gene_sets WHERE run_id=?"
    params: list = [run_id]
    if min_beta is not None:
        sql += " AND beta >= ?"
        params.append(min_beta)
    if min_beta_uncorrected is not None:
        sql += " AND beta_uncorrected >= ?"
        params.append(min_beta_uncorrected)
    if search:
        sql += " AND (gene_set LIKE ? OR label LIKE ?)"
        params += [_like(search), _like(search)]
    direction = "ASC" if sort_col == "gene_set" else "DESC"
    sql += f" ORDER BY {sort_col} {direction} LIMIT ?"
    params.append(max(1, min(int(limit), 100000)))
    return _rows(conn.execute(sql, params))


def run_params(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    return _rows(conn.execute("SELECT parameter, version, value FROM run_params WHERE run_id=? ORDER BY parameter, version", (run_id,)))


_ACROSS_RUN_COLUMNS = ("r.run_id, r.model, r.model_title, r.trait, r.seed, r.title AS run_title, "
                       "p.name AS phenotype_name, p.portal_id, p.trait_group")


def gene_across_runs(conn: sqlite3.Connection, gene: str, *, model: str = "") -> list[dict]:
    """The gene's scores in every run of the database (only runs where it passed the build thresholds)."""
    sql = (f"SELECT {_ACROSS_RUN_COLUMNS}, g.prior, g.combined, g.log_bf, g.huge_score, g.n FROM genes g "
           "JOIN runs r ON r.run_id = g.run_id LEFT JOIN phenotypes p ON p.legacy_id = r.trait WHERE g.gene=?")
    params: list = [gene]
    if model:
        sql += " AND r.model=?"
        params.append(model)
    sql += " ORDER BY p.trait_group, r.trait, r.model, r.seed"
    return _rows(conn.execute(sql, params))


def gene_set_across_runs(conn: sqlite3.Connection, gene_set: str, *, model: str = "") -> list[dict]:
    """The gene set's effects in every run of the database (only runs where it passed the build thresholds)."""
    sql = (f"SELECT {_ACROSS_RUN_COLUMNS}, s.beta, s.beta_uncorrected, s.n, s.p_orig FROM gene_sets s "
           "JOIN runs r ON r.run_id = s.run_id LEFT JOIN phenotypes p ON p.legacy_id = r.trait WHERE s.gene_set=?")
    params: list = [gene_set]
    if model:
        sql += " AND r.model=?"
        params.append(model)
    sql += " ORDER BY p.trait_group, r.trait, r.model, r.seed"
    return _rows(conn.execute(sql, params))


def gene_set_detail(conn: sqlite3.Connection, run_id: str, gene_set: str, *, limit: int = 500) -> Optional[dict]:
    row = conn.execute("SELECT * FROM gene_sets WHERE run_id=? AND gene_set=?", (run_id, gene_set)).fetchone()
    if row is None:
        return None
    detail = dict(row)
    detail["extra"] = json.loads(detail.pop("extra_json") or "{}")
    detail["loadings"] = _rows(conn.execute(
        "SELECT gene, beta, weight, prior, combined, log_bf, huge_score FROM gene_gene_sets "
        "WHERE run_id=? AND gene_set=? ORDER BY weight DESC, combined DESC LIMIT ?",
        (run_id, gene_set, max(1, min(int(limit), 100000))),
    ))
    detail["n_loadings"] = conn.execute(
        "SELECT COUNT(*) FROM gene_gene_sets WHERE run_id=? AND gene_set=?", (run_id, gene_set)
    ).fetchone()[0]
    return detail


def gene_detail(conn: sqlite3.Connection, run_id: str, gene: str, *, limit: int = 500) -> Optional[dict]:
    row = conn.execute("SELECT * FROM genes WHERE run_id=? AND gene=?", (run_id, gene)).fetchone()
    if row is None:
        return None
    detail = dict(row)
    detail["extra"] = json.loads(detail.pop("extra_json") or "{}")
    detail["gene_sets"] = _rows(conn.execute(
        "SELECT l.gene_set, s.label, l.beta, l.weight, s.beta_uncorrected FROM gene_gene_sets l "
        "LEFT JOIN gene_sets s ON s.run_id = l.run_id AND s.gene_set = l.gene_set "
        "WHERE l.run_id=? AND l.gene=? ORDER BY l.beta DESC, l.weight DESC LIMIT ?",
        (run_id, gene, max(1, min(int(limit), 100000))),
    ))
    return detail
