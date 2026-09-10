from __future__ import annotations

import gzip
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pigean import portal, portal_db, portal_server  # noqa: E402
from pigean.portal_assets import render_portal_html  # noqa: E402

GENE_STATS = (
    "Gene\tprior\tcombined\tlog_bf\thuge_score_gwas\tN\tChrom\tStart\tEnd\n"
    "GENE1\t1.8\t3.0\t1.2\t0.5\t5\t1\t100\t200\n"
    "GENE2\t1.0\t1.4\t0.4\t0.1\t4\t1\t300\t400\n"
    "GENE3\t0.2\t1.3\t1.1\t0.9\t4\t2\t500\t600\n"
    "GENELOW\t0.1\t0.2\t0.1\t0.0\t4\t2\t700\t800\n"
)
GENE_SET_STATS = (
    "Gene_Set\tlabel\tN\tbeta\tbeta_uncorrected\tP_orig\tZ_orig\n"
    "SET_A\tset a\t10\t0.5\t1.5\t1e-5\t4.4\n"
    "SET_B\tset b\t20\t0.02\t0.9\t1e-3\t3.1\n"
    "SET_LOW\tset low\t30\t0.001\t0.1\t0.5\t0.6\n"
)
LOADINGS = (
    "Gene\tprior\tcombined\tlog_bf\thuge_score_gwas\tgene_set\tbeta\tweight\n"
    "GENE1\t1.8\t3.0\t1.2\t0.5\tSET_A\t0.5\t1\n"
    "GENE2\t1.0\t1.4\t0.4\t0.1\tSET_A\t0.5\t0.5\n"
    "GENELOW\t0.1\t0.2\t0.1\t0.0\tSET_A\t0.5\t1\n"
    "GENE3\t0.2\t1.3\t1.1\t0.9\tSET_B\t0.02\t1\n"
    "GENE1\t1.8\t3.0\t1.2\t0.5\tSET_LOW\t0.001\t1\n"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")


class PortalFilterTest(unittest.TestCase):
    def test_parse_filter(self) -> None:
        f = portal_db.parse_filter("prior > 1")
        self.assertEqual((f.column, f.op, f.value), ("prior", ">", 1.0))
        self.assertEqual(portal_db.parse_filter("beta>=0.01").expr, "beta>=0.01")
        self.assertEqual(portal_db.parse_filter("z_orig<-1e-3").value, -0.001)
        with self.assertRaises(ValueError):
            portal_db.parse_filter("prior >> 1")
        with self.assertRaises(ValueError):
            portal_db.parse_filter("prior > abc")

    def test_filter_modes(self) -> None:
        filters = [portal_db.parse_filter("prior>1"), portal_db.parse_filter("log_bf>1")]
        row = {"prior": 0.2, "log_bf": 1.1}
        self.assertTrue(portal_db.passes_filters(row, filters, "any"))
        self.assertFalse(portal_db.passes_filters(row, filters, "all"))
        self.assertFalse(portal_db.passes_filters({"prior": None, "log_bf": None}, filters, "any"))
        self.assertTrue(portal_db.passes_filters(row, [], "all"))


class PortalBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "run"
        _write(self.run_dir / "pigean.gene_stats.out.gz", GENE_STATS)
        _write(self.run_dir / "pigean.gene_set_stats.out.gz", GENE_SET_STATS)
        _write(self.run_dir / "pigean.gene_gene_set_stats.out.gz", LOADINGS)
        self.lap_dir = self.root / "lap"
        _write(self.lap_dir / "x__T2D.gene_stats.tsv", GENE_STATS)
        _write(self.lap_dir / "x__T2D.gene_set_stats.tsv", GENE_SET_STATS)
        _write(self.lap_dir / "x__T2D.gene_gene_set_stats.tsv", LOADINGS)
        self.db = self.root / "portal.sqlite"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, *extra: str) -> int:
        return portal.main([
            "build", "--db", str(self.db), "--run", f"demo:{self.run_dir}",
            "--gene-filter", "prior>1", "--gene-filter", "log_bf>1", "--gene-set-filter", "beta>0.01", *extra,
        ])

    def test_resolve_run_dir_dashboard_and_lap_names(self) -> None:
        files = portal_db.resolve_run_dir("a", self.run_dir)
        self.assertEqual(files.gene_stats.name, "pigean.gene_stats.out.gz")
        self.assertEqual(files.gene_gene_set_stats.name, "pigean.gene_gene_set_stats.out.gz")
        files = portal_db.resolve_run_dir("b", self.lap_dir)
        self.assertEqual(files.gene_stats.name, "x__T2D.gene_stats.tsv")
        self.assertEqual(files.gene_set_stats.name, "x__T2D.gene_set_stats.tsv")
        self.assertEqual(files.gene_gene_set_stats.name, "x__T2D.gene_gene_set_stats.tsv")
        with self.assertRaises(FileNotFoundError):
            portal_db.resolve_run_dir("c", self.root / "nowhere")

    def test_build_applies_thresholds(self) -> None:
        self.assertEqual(self._build(), 0)
        conn = portal_db.open_database(self.db, readonly=True)
        runs = portal_db.list_runs(conn)
        self.assertEqual([r["run_id"] for r in runs], ["demo"])
        run = runs[0]
        # GENE1 (prior 1.8) and GENE3 (log_bf 1.1) pass with mode=any; GENE2/GENELOW do not.
        self.assertEqual((run["n_genes"], run["n_genes_input"]), (2, 4))
        self.assertEqual((run["n_gene_sets"], run["n_gene_sets_input"]), (2, 3))
        # loadings kept only for SET_A / SET_B, including genes that failed the gene filter
        self.assertEqual((run["n_loadings"], run["n_loadings_input"]), (4, 5))
        self.assertEqual(run["filters"]["genes"], ["prior>1", "log_bf>1"])

        genes = portal_db.query_genes(conn, "demo")
        self.assertEqual([g["gene"] for g in genes], ["GENE1", "GENE3"])
        self.assertEqual([g["gene"] for g in portal_db.query_genes(conn, "demo", min_prior=1.5)], ["GENE1"])
        gene_sets = portal_db.query_gene_sets(conn, "demo")
        self.assertEqual([g["gene_set"] for g in gene_sets], ["SET_A", "SET_B"])
        self.assertEqual([g["gene_set"] for g in portal_db.query_gene_sets(conn, "demo", search="set b")], ["SET_B"])

        detail = portal_db.gene_set_detail(conn, "demo", "SET_A")
        self.assertEqual([l["gene"] for l in detail["loadings"]], ["GENE1", "GENELOW", "GENE2"])
        self.assertEqual(detail["extra"]["label"], "set a")
        gene = portal_db.gene_detail(conn, "demo", "GENE1")
        self.assertEqual([g["gene_set"] for g in gene["gene_sets"]], ["SET_A"])
        self.assertEqual(gene["extra"]["Chrom"], 1.0)
        self.assertIsNone(portal_db.gene_detail(conn, "demo", "MISSING"))
        conn.close()

    def test_filter_mode_all_and_keep_all_loadings(self) -> None:
        self.assertEqual(self._build("--filter-mode", "all", "--keep-all-loadings"), 0)
        conn = portal_db.open_database(self.db, readonly=True)
        run = portal_db.list_runs(conn)[0]
        self.assertEqual(run["n_genes"], 1)  # only GENE1 has prior>1 AND log_bf>1
        self.assertEqual(run["n_loadings"], 5)
        conn.close()

    def test_append_adds_second_run(self) -> None:
        self.assertEqual(self._build(), 0)
        rc = portal.main([
            "build", "--db", str(self.db), "--append", "--run", f"lap:{self.lap_dir}",
            "--run-title", "lap:LAP run",
        ])
        self.assertEqual(rc, 0)
        conn = portal_db.open_database(self.db, readonly=True)
        runs = {r["run_id"]: r for r in portal_db.list_runs(conn)}
        self.assertEqual(set(runs), {"demo", "lap"})
        self.assertEqual(runs["lap"]["title"], "LAP run")
        self.assertEqual(runs["lap"]["n_genes"], 4)  # no filters on the appended run
        conn.close()

    def test_run_files_spec_and_missing_loadings_warning(self) -> None:
        rc = portal.main([
            "build", "--db", str(self.db),
            "--run-files", f"only:gene_stats={self.lap_dir / 'x__T2D.gene_stats.tsv'},"
                           f"gene_set_stats={self.lap_dir / 'x__T2D.gene_set_stats.tsv'}",
        ])
        self.assertEqual(rc, 0)
        conn = portal_db.open_database(self.db, readonly=True)
        run = portal_db.list_runs(conn)[0]
        self.assertEqual(run["n_loadings"], 0)
        self.assertTrue(any("loadings will be empty" in w for w in run["warnings"]))
        conn.close()

    def test_build_without_runs_fails(self) -> None:
        self.assertEqual(portal.main(["build", "--db", str(self.db)]), 2)

    def test_api_and_http_round_trip(self) -> None:
        self.assertEqual(self._build(), 0)
        state = portal_server.PortalState(self.db, title="t", plotly_src="about:blank")
        status, body = portal_server.handle_api(state, "/api/runs", {})
        self.assertEqual(status, 200)
        self.assertEqual(body["runs"][0]["run_id"], "demo")
        status, body = portal_server.handle_api(state, "/api/genes", {"run": ["demo"], "min_prior": ["1.5"]})
        self.assertEqual((status, [g["gene"] for g in body["genes"]]), (200, ["GENE1"]))
        status, body = portal_server.handle_api(state, "/api/genes", {"run": ["nope"]})
        self.assertEqual(status, 404)
        status, body = portal_server.handle_api(state, "/api/genes", {"run": ["demo"], "min_prior": ["x"]})
        self.assertEqual(status, 400)

        ready = threading.Event()
        holder: dict = {}

        def on_ready(httpd) -> None:
            holder["httpd"] = httpd
            ready.set()

        thread = threading.Thread(
            target=portal_server.serve,
            kwargs={"db_path": self.db, "host": "127.0.0.1", "port": 0, "title": "t", "plotly_src": "about:blank", "server_ready": on_ready},
            daemon=True,
        )
        thread.start()
        self.assertTrue(ready.wait(10))
        httpd = holder["httpd"]
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(f"{base}/api/gene_set?run=demo&id=SET_A") as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(payload["gene_set"], "SET_A")
            self.assertEqual(len(payload["loadings"]), 3)
            with urllib.request.urlopen(f"{base}/") as resp:
                html = resp.read().decode("utf-8")
                self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
            self.assertIn("<title>t</title>", html)
            self.assertIn("/api/genes", html)
            self.assertIn('window.PIGEAN_PORTAL_API_BASE = ""', html)
            req = urllib.request.Request(f"{base}/api/runs", method="OPTIONS")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 204)
                self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
        finally:
            httpd.shutdown()
            thread.join(5)

    def test_html_subcommand_writes_static_page(self) -> None:
        self.assertEqual(self._build(), 0)
        out = self.root / "static" / "portal.html"
        rc = portal.main(["html", "--api-url", "http://localhost:8765/", "--out", str(out), "--title", "static",
                          "--db", str(self.db)])
        self.assertEqual(rc, 0)
        html = out.read_text(encoding="utf-8")
        self.assertIn('window.PIGEAN_PORTAL_API_BASE = "http://localhost:8765/"', html)
        self.assertIn("<title>static</title>", html)
        self.assertIn("API: <code>http://localhost:8765/</code>", html)
        # bad URL scheme and missing --db are errors
        self.assertEqual(portal.main(["html", "--api-url", "localhost:8765", "--out", str(out)]), 1)
        self.assertEqual(portal.main(["html", "--api-url", "http://x", "--out", str(out), "--db", str(self.root / "no.sqlite")]), 1)

    def test_render_html_escapes_title(self) -> None:
        html = render_portal_html(title="<x>", plotly_src="https://example/plotly.js")
        self.assertIn("&lt;x&gt;", html)
        self.assertIn('src="https://example/plotly.js"', html)


if __name__ == "__main__":
    unittest.main()
