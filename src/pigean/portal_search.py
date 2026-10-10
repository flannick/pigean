"""Cross-run entry points for the portal's Gene, Geneset and Factor searches.

Factor identities are scoped to their source run. Only exported EAGGL evidence is
read; linking a graph to a PIGEAN run does not itself imply a factor/trait association.
"""

from __future__ import annotations

import json
import math
import sqlite3
from functools import lru_cache
from html.parser import HTMLParser

from . import portal_db


def search_entities(conn: sqlite3.Connection, kind: str, query: str, *, model: str = "", limit: int = 20) -> list[dict]:
    if kind not in ("gene", "gene_set", "factor"):
        raise ValueError("kind must be gene, gene_set or factor")
    limit = max(1, min(limit, 50))
    query = query.strip()
    if not query:
        return []
    if kind == "factor":
        q = query.casefold()
        rows = [row for row in factor_catalog(conn, model=model)
                if q in row["id"].casefold() or q in row["label"].casefold()]
        rows.sort(key=lambda row: (row["id"].casefold() != q, row["label"].casefold() != q,
                                   row["label"].casefold(), row["run_id"], row["id"]))
        return rows[:limit]
    table, column = ("genes", "gene") if kind == "gene" else ("gene_sets", "gene_set")
    label = "''" if kind == "gene" else "MIN(e.label)"
    # instr gives literal substring matching: '_' and '%' in gene-set IDs are not wildcards.
    sql = (f"SELECT e.{column} AS id, {label} AS label, COUNT(DISTINCT r.run_id) AS n_runs, "
           "COUNT(DISTINCT COALESCE(NULLIF(r.trait, ''), r.run_id)) AS n_traits "
           f"FROM {table} e JOIN runs r ON r.run_id=e.run_id WHERE "
           f"(instr(lower(e.{column}), lower(?)) > 0")
    params: list = [query]
    if kind == "gene_set":
        sql += " OR instr(lower(e.label), lower(?)) > 0"
        params.append(query)
    sql += ")"
    if model:
        sql += " AND r.model=?"
        params.append(model)
    sql += (f" GROUP BY e.{column} ORDER BY lower(e.{column})=lower(?) DESC, "
            f"instr(lower(e.{column}), lower(?))=1 DESC, length(e.{column}), e.{column} LIMIT ?")
    params += [query, query, limit]
    return [dict(row) for row in conn.execute(sql, params)]


class _GraphData(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inside = False
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.inside = dict(attrs).get("id") == "eaggl-factor-graph-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.inside = False

    def handle_data(self, data):
        if self.inside:
            self.parts.append(data)


FACTOR_METRICS = ("beta", "beta_uncorrected", "nnls_loading", "joint_fraction", "joint_coefficient",
                  "marginal_fraction", "marginal_coefficient", "p_value", "trait_neff", "trait_n_eff", "graph_weight")


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


@lru_cache(maxsize=8)
def _graph_factors(html: str) -> list[dict]:
    """Read JSON, never execute HTML. Cache bounded, with content-based invalidation."""
    parser = _GraphData()
    parser.feed(html)
    try:
        graph = json.loads("".join(parser.parts))
    except (ValueError, RecursionError):
        return []
    if not isinstance(graph, dict) or graph.get("schema") != "eaggl_factor_graph/v1":
        return []
    def records(key):
        value = graph.get(key)
        return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []
    nodes = records("nodes") + records("candidate_nodes")
    trait_nodes = {str(n["id"]): n for n in nodes if n.get("kind") == "trait" and n.get("id")}
    edges = records("edges") + records("candidate_edges")
    factors = []
    for node in records("nodes"):
        if node.get("kind") != "factor" or not isinstance(node.get("id"), str):
            continue
        ident = node["id"]
        traits: dict[str, dict] = {}
        provenance = node.get("provenance")
        details = provenance.get("relevance_by_anchor", []) if isinstance(provenance, dict) else []
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict) or not isinstance(detail.get("anchor"), str) or not detail["anchor"]:
                continue
            trait = detail["anchor"]
            traits[trait] = {"trait": trait, **{m: _number(detail.get(m)) for m in FACTOR_METRICS}}
        for edge in edges:
            if edge.get("kind") != "factor_trait" or edge.get("from") != ident:
                continue
            target = str(edge.get("to", ""))
            if target not in trait_nodes:
                continue
            traits.setdefault(target, {"trait": target})["graph_weight"] = _number(edge.get("weight"))
        factors.append({"id": ident, "label": str(node.get("label") or ident),
                        "relevance": _number(node.get("relevance")), "traits": list(traits.values())})
    return factors


def factor_catalog(conn: sqlite3.Connection, *, model: str = "") -> list[dict]:
    if not portal_db._has_factor_graphs(conn):
        return []
    sql = ("SELECT r.run_id, r.model, r.trait, r.seed, g.html FROM run_factor_graphs g "
           "JOIN runs r ON r.run_id=g.run_id")
    params = []
    if model:
        sql += " WHERE r.model=?"
        params.append(model)
    rows = []
    for run in conn.execute(sql, params):
        for factor in _graph_factors(run["html"]):
            rows.append({k: run[k] for k in ("run_id", "model", "trait", "seed")} |
                        {k: factor[k] for k in ("id", "label", "relevance")} |
                        {"n_traits": len(factor["traits"])})
    return rows


def factor_detail(conn: sqlite3.Connection, run_id: str, ident: str) -> dict | None:
    graph = portal_db.factor_graph(conn, run_id)
    factor = next((f for f in _graph_factors(graph["html"]) if f["id"] == ident), None) if graph else None
    if factor is None:
        return None
    runs = portal_db.list_runs(conn)
    source = next(r for r in runs if r["run_id"] == run_id)
    phenotypes = portal_db.list_phenotypes(conn)
    rows = []
    for row in factor["traits"]:
        ph = phenotypes.get(row["trait"], {})
        rows.append({**row, "phenotype_name": ph.get("name", ""), "portal_id": ph.get("portal_id", ""),
                     "trait_group": ph.get("trait_group", ""),
                     "run_ids": [r["run_id"] for r in runs if r["trait"] == row["trait"] and r["model"] == source["model"]]})
    return {"id": ident, "label": factor["label"], "relevance": factor["relevance"], "run": run_id,
            "model": source["model"], "rows": rows}
