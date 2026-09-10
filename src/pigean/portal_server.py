"""HTTP server for the PIGEAN results portal (standard-library `http.server`).

Serves the single-page UI from `pigean.portal_assets` at `/` and a small read-only JSON
API under `/api/` backed by the SQLite file built by `pigean.portal_db`.

API:
    GET /api/runs
    GET /api/genes?run=ID[&min_prior=&min_log_bf=&min_combined=&search=&sort=&limit=]
    GET /api/gene_sets?run=ID[&min_beta=&min_beta_uncorrected=&search=&sort=&limit=]
    GET /api/gene_set?run=ID&id=GENE_SET[&limit=]
    GET /api/gene?run=ID&id=GENE[&limit=]
    GET /api/gene_across?id=GENE[&model=]          the gene in every run (runs where it passed thresholds)
    GET /api/gene_set_across?id=GENE_SET[&model=]  the gene set in every run
    GET /api/run_params?run=ID                     PIGEAN params recorded for the run
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from . import portal_db
from .portal_assets import render_portal_html

LOGGER = logging.getLogger("pigean.portal")


class PortalState:
    """Per-server state: the database path plus a thread-local read-only connection."""

    def __init__(self, db_path: Path, *, title: str, plotly_src: str, cors_origin: str = "*") -> None:
        self.db_path = db_path
        self.title = title
        self.plotly_src = plotly_src
        # Static (bucket-hosted) copies of the page live on another origin, so the read-only
        # API answers cross-origin requests. Empty string disables the header.
        self.cors_origin = cors_origin
        self._local = threading.local()

    def connection(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = portal_db.open_database(self.db_path, readonly=True)
            self._local.conn = conn
        return conn


def _float(params: dict[str, list[str]], key: str) -> Optional[float]:
    values = params.get(key)
    if not values or values[0] in ("", None):
        return None
    try:
        return float(values[0])
    except ValueError:
        raise ValueError(f"query parameter '{key}' must be a number")


def _str(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key)
    return values[0] if values else default


def _int(params: dict[str, list[str]], key: str, default: int) -> int:
    values = params.get(key)
    if not values or values[0] == "":
        return default
    try:
        return int(values[0])
    except ValueError:
        raise ValueError(f"query parameter '{key}' must be an integer")


def handle_api(state: PortalState, path: str, params: dict[str, list[str]]) -> tuple[int, dict]:
    """Route one API request. Returns (status, json-serialisable body); bad parameters give 400."""
    try:
        return _handle_api(state, path, params)
    except ValueError as exc:
        return 400, {"error": str(exc)}


def _handle_api(state: PortalState, path: str, params: dict[str, list[str]]) -> tuple[int, dict]:
    conn = state.connection()
    if path == "/api/runs":
        return 200, {"runs": portal_db.list_runs(conn)}

    if path in ("/api/gene_across", "/api/gene_set_across"):
        ident = _str(params, "id")
        if not ident:
            return 400, {"error": "missing 'id' parameter"}
        fn = portal_db.gene_across_runs if path == "/api/gene_across" else portal_db.gene_set_across_runs
        rows = fn(conn, ident, model=_str(params, "model"))
        return 200, {"id": ident, "model": _str(params, "model"), "rows": rows}

    run_id = _str(params, "run")
    if not run_id:
        return 400, {"error": "missing 'run' parameter"}
    if conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone() is None:
        return 404, {"error": f"unknown run '{run_id}'"}

    if path == "/api/genes":
        rows = portal_db.query_genes(
            conn, run_id, min_prior=_float(params, "min_prior"), min_log_bf=_float(params, "min_log_bf"),
            min_combined=_float(params, "min_combined"), search=_str(params, "search"),
            sort=_str(params, "sort", "combined"), limit=_int(params, "limit", 5000),
        )
        return 200, {"run": run_id, "genes": rows}
    if path == "/api/gene_sets":
        rows = portal_db.query_gene_sets(
            conn, run_id, min_beta=_float(params, "min_beta"),
            min_beta_uncorrected=_float(params, "min_beta_uncorrected"), search=_str(params, "search"),
            sort=_str(params, "sort", "beta"), limit=_int(params, "limit", 500),
        )
        return 200, {"run": run_id, "gene_sets": rows}
    if path == "/api/gene_set":
        detail = portal_db.gene_set_detail(conn, run_id, _str(params, "id"), limit=_int(params, "limit", 500))
        return (200, detail) if detail is not None else (404, {"error": "unknown gene set"})
    if path == "/api/gene":
        detail = portal_db.gene_detail(conn, run_id, _str(params, "id"), limit=_int(params, "limit", 500))
        return (200, detail) if detail is not None else (404, {"error": "unknown gene"})
    if path == "/api/run_params":
        return 200, {"run": run_id, "params": portal_db.run_params(conn, run_id)}
    return 404, {"error": f"unknown endpoint {path}"}


def make_handler(state: PortalState):
    class PortalHandler(BaseHTTPRequestHandler):
        server_version = "pigean-portal/1"

        def log_message(self, fmt: str, *args) -> None:  # route to logging, not stderr
            LOGGER.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def _cors_headers(self) -> None:
            if state.cors_origin:
                self.send_header("Access-Control-Allow-Origin", state.cors_origin)
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:  # noqa: N802 (CORS preflight)
            self.send_response(204)
            self._cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload, allow_nan=False, default=_json_default).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query, keep_blank_values=True)
            if parsed.path in ("/", "/index.html"):
                html = render_portal_html(title=state.title, plotly_src=state.plotly_src)
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/healthz":
                self._send_json(200, {"ok": True, "db": str(state.db_path)})
                return
            if parsed.path.startswith("/api/"):
                try:
                    status, payload = handle_api(state, parsed.path, params)
                except Exception as exc:  # report, never crash the server thread
                    LOGGER.exception("API error for %s", self.path)
                    status, payload = 500, {"error": f"internal error: {exc}"}
                self._send_json(status, payload)
                return
            self._send_json(404, {"error": "not found"})

    return PortalHandler


def _json_default(value):
    if isinstance(value, float):
        return None
    return str(value)


def serve(db_path: Path, *, host: str = "127.0.0.1", port: int = 8765, title: str = "PIGEAN Portal",
          plotly_src: str = "", cors_origin: str = "*", server_ready=None) -> None:
    """Block serving the portal until interrupted. `server_ready(httpd)` is called once bound."""
    if not db_path.exists():
        raise FileNotFoundError(f"database not found: {db_path}")
    state = PortalState(db_path, title=title, plotly_src=plotly_src, cors_origin=cors_origin)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.daemon_threads = True
    bound_host, bound_port = httpd.server_address[:2]
    LOGGER.info("serving %s at http://%s:%d/", db_path, bound_host, bound_port)
    if server_ready is not None:
        server_ready(httpd)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutting down")
    finally:
        httpd.server_close()
