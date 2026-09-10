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

from pigean import portal, portal_compare, portal_db, portal_server  # noqa: E402
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

PHENOTYPES_FLAT = (
    "portal_id\tgwas_source_category\tlegacy_trait_group\ttrait_group\tphenotype\tphenotype_name\tdescription\ttrait_type\tis_dichotomous\tis_complex\tpigean_id\tmapping_count\ttarget_id\ttarget_label\ttarget_ontology\tmapping_predicate\tconfidence\tmapping_justification\tsource\n"
    "PORTAL:0000398\tportal\tGLYCEMIC\tmetabolic\tT2D\tType 2 diabetes (T2D)\tType 2 diabetes (T2D)\tphenotype\t1\tfalse\t\t2\tMESH:D003924\tDiabetes Mellitus, Type 2\tMESH\tskos:exactMatch\t0.85\tinherited\tcurated.tsv\n"
    "PORTAL:0000398\tportal\tGLYCEMIC\tmetabolic\tT2D\tType 2 diabetes (T2D)\tType 2 diabetes (T2D)\tphenotype\t1\tfalse\t\t2\tMONDO:0005148\ttype 2 diabetes mellitus\tMONDO\tskos:exactMatch\t0.9\tcurated\tamp.csv\n"
    "PORTAL:0000001\tportal\tCV\tcardiovascular\tAF\tAtrial fibrillation\tAtrial fibrillation\tphenotype\t1\tfalse\t\t0\t\t\t\t\t\t\t\n"
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

    def test_run_meta_explicit_and_inferred(self) -> None:
        rc = portal.main([
            "build", "--db", str(self.db),
            "--run", f"large__T2D__s2:{self.run_dir}",
            "--run", f"custom:{self.lap_dir}", "--run-meta", "custom:model=pathways,trait=IBD,seed=7,title=IBD pathways",
            "--run", f"plain:{self.lap_dir}",
        ])
        self.assertEqual(rc, 0)
        conn = portal_db.open_database(self.db, readonly=True)
        runs = {r["run_id"]: r for r in portal_db.list_runs(conn)}
        self.assertEqual((runs["large__T2D__s2"]["model"], runs["large__T2D__s2"]["trait"], runs["large__T2D__s2"]["seed"]), ("large", "T2D", "2"))
        self.assertEqual((runs["custom"]["model"], runs["custom"]["trait"], runs["custom"]["seed"], runs["custom"]["title"]), ("pathways", "IBD", "7", "IBD pathways"))
        self.assertEqual((runs["plain"]["model"], runs["plain"]["trait"], runs["plain"]["seed"]), ("", "", ""))
        conn.close()
        with self.assertRaises(SystemExit):
            portal.main(["build", "--db", str(self.db), "--run", f"x:{self.run_dir}", "--run-meta", "x:colour=red"])

    def test_phenotype_file_adds_names_and_mappings(self) -> None:
        pheno = self.root / "portal_phenotypes_flat.tsv"
        _write(pheno, PHENOTYPES_FLAT)
        rc = portal.main([
            "build", "--db", str(self.db), "--phenotype-file", str(pheno),
            "--run", f"large__T2D__s1:{self.run_dir}", "--run", f"large__AF__s1:{self.lap_dir}",
            "--run", f"large__NOPE__s1:{self.lap_dir}",
        ])
        self.assertEqual(rc, 0)
        conn = portal_db.open_database(self.db, readonly=True)
        runs = {r["run_id"]: r for r in portal_db.list_runs(conn)}
        t2d = runs["large__T2D__s1"]["phenotype"]
        self.assertEqual((t2d["name"], t2d["portal_id"], t2d["trait_group"]), ("Type 2 diabetes (T2D)", "PORTAL:0000398", "metabolic"))
        self.assertEqual([m["target_id"] for m in t2d["mappings"]], ["MONDO:0005148", "MESH:D003924"])  # confidence desc
        self.assertEqual(t2d["mappings"][0]["predicate"], "skos:exactMatch")
        af = runs["large__AF__s1"]["phenotype"]
        self.assertEqual((af["name"], af["mappings"]), ("Atrial fibrillation", []))
        self.assertIsNone(runs["large__NOPE__s1"]["phenotype"])
        self.assertTrue(any("NOPE" in w for w in runs["large__NOPE__s1"]["warnings"]))
        conn.close()
        self.assertEqual(portal.main(["build", "--db", str(self.db), "--phenotype-file", str(self.root / "missing.tsv"),
                                      "--run", f"x:{self.run_dir}"]), 1)

    def test_package_inputs_construct_run_ids_and_store_params(self) -> None:
        params = self.root / "params.tsv"
        _write(params, "Parameter\tVersion\tValue\nnum_chains\t1\t10\noption_seed\t1\t3\n")
        gs, gss, ggss = (self.lap_dir / f"x__T2D.{k}.tsv" for k in ("gene_stats", "gene_set_stats", "gene_gene_set_stats"))
        pkg = lambda model, trait, **kw: ",".join([f"model={model}", f"trait={trait}", f"gene_stats={gs}", f"gene_set_stats={gss}", f"gene_gene_set_stats={ggss}"] + [f"{k}={v}" for k, v in kw.items()])
        rc = portal.main([
            "build", "--db", str(self.db),
            "--package", pkg("large", "T2D", params=str(params), model_title="Large model"),
            "--package", pkg("large", "IBD"), "--package", pkg("large", "IBD"),
            "--package", pkg("small", "IBD", run="s7", title="custom title"),
        ])
        self.assertEqual(rc, 0)
        conn = portal_db.open_database(self.db, readonly=True)
        runs = {r["run_id"]: r for r in portal_db.list_runs(conn)}
        self.assertEqual(set(runs), {"large__T2D__main", "large__IBD__run1", "large__IBD__run2", "small__IBD__s7"})
        self.assertEqual((runs["large__T2D__main"]["model_title"], runs["large__T2D__main"]["seed"], runs["large__T2D__main"]["title"]), ("Large model", "main", "T2D / large"))
        self.assertEqual(runs["small__IBD__s7"]["title"], "custom title")
        self.assertEqual([p["parameter"] for p in portal_db.run_params(conn, "large__T2D__main")], ["num_chains", "option_seed"])
        self.assertEqual(portal_db.run_params(conn, "large__IBD__run1"), [])

        across = portal_db.gene_across_runs(conn, "GENE1")
        self.assertEqual(len(across), 4)
        self.assertEqual({r["run_id"] for r in portal_db.gene_across_runs(conn, "GENE1", model="small")}, {"small__IBD__s7"})
        self.assertEqual({r["run_id"] for r in portal_db.gene_set_across_runs(conn, "SET_A")}, set(runs))
        self.assertEqual(portal_db.gene_set_across_runs(conn, "NOPE"), [])
        conn.close()

        state = portal_server.PortalState(self.db, title="t", plotly_src="about:blank")
        status, body = portal_server.handle_api(state, "/api/gene_across", {"id": ["GENE1"], "model": ["large"]})
        self.assertEqual((status, len(body["rows"])), (200, 3))
        status, body = portal_server.handle_api(state, "/api/gene_set_across", {})
        self.assertEqual(status, 400)
        status, body = portal_server.handle_api(state, "/api/run_params", {"run": ["large__T2D__main"]})
        self.assertEqual((status, body["params"][0]["value"]), (200, "10"))
        with self.assertRaises(SystemExit):
            portal.main(["build", "--db", str(self.db), "--package", "model=a,trait=b"])

    def test_ranks_are_over_full_input_not_kept_rows(self) -> None:
        self.assertEqual(self._build(), 0)  # keeps GENE1 and GENE3 only
        conn = portal_db.open_database(self.db, readonly=True)
        ranks = {r["gene"]: r for r in portal_db.query_genes(conn, "demo")}
        # combined: GENE1 3.0 (1), GENE2 1.4 (2), GENE3 1.3 (3), GENELOW 0.2 (4) -> GENE3 is rank 3 even though GENE2 was filtered out
        self.assertEqual((ranks["GENE1"]["rank_combined"], ranks["GENE3"]["rank_combined"]), (1, 3))
        self.assertEqual((ranks["GENE1"]["rank_log_bf"], ranks["GENE3"]["rank_log_bf"]), (1, 2))
        gs = {r["gene_set"]: r for r in portal_db.query_gene_sets(conn, "demo")}
        self.assertEqual((gs["SET_A"]["rank_beta"], gs["SET_B"]["rank_beta"]), (1, 2))
        run = portal_db.list_runs(conn)[0]
        self.assertEqual((run["n_ranked_genes"], run["n_ranked_gene_sets"]), (4, 3))
        conn.close()

    def test_compare_queries(self) -> None:
        # run A: the fixture as-is; run B: same genes with GENE1 halved and GENE2 boosted so ranks flip, plus a B-only gene
        b_dir = self.root / "b"
        _write(b_dir / "pigean.gene_stats.out.gz",
               "Gene\tprior\tcombined\tlog_bf\thuge_score_gwas\tN\tChrom\tStart\tEnd\n"
               "GENE1\t0.9\t1.5\t0.6\t0.5\t5\t1\t100\t200\n"
               "GENE2\t2.0\t3.5\t1.5\t0.1\t4\t1\t300\t400\n"
               "GENE3\t0.2\t1.3\t1.1\t0.9\t4\t2\t500\t600\n"
               "GENENEW\t1.5\t2.0\t0.5\t0.0\t4\t2\t700\t800\n")
        _write(b_dir / "pigean.gene_set_stats.out.gz", GENE_SET_STATS)
        _write(b_dir / "pigean.gene_gene_set_stats.out.gz", LOADINGS)
        params_a, params_b = self.root / "pa.tsv", self.root / "pb.tsv"
        _write(params_a, "Parameter\tVersion\tValue\nnum_chains\t1\t10\nseed\t1\t1\nonly_a\t1\tx\n")
        _write(params_b, "Parameter\tVersion\tValue\nnum_chains\t1\t10.0\nseed\t1\t2\n")
        rc = portal.main([
            "build", "--db", str(self.db), "--gene-filter", "combined>1", "--gene-set-filter", "beta>0.01",
            "--run", f"A:{self.run_dir}", "--run-meta", f"A:model=m,trait=T2D,title=A", "--run-files",
            f"B:gene_stats={b_dir / 'pigean.gene_stats.out.gz'},gene_set_stats={b_dir / 'pigean.gene_set_stats.out.gz'}",
            "--run-meta", "B:model=m,trait=T2D",
        ])
        self.assertEqual(rc, 0)
        conn = portal_db.open_database(self.db)
        # attach params by hand (the --run forms have no params= key)
        conn.executemany("INSERT INTO run_params VALUES (?,?,?,?)", [("A", "num_chains", "1", "10"), ("A", "seed", "1", "1"), ("A", "only_a", "1", "x"),
                                                                   ("B", "num_chains", "1", "10.0"), ("B", "seed", "1", "2")])
        conn.commit()

        genes = portal_compare.compare_genes(conn, "A", "B", metric="combined")
        by = {r["gene"]: r for r in genes["rows"]}
        self.assertEqual(genes["counts"], {"both": 3, "a_only": 0, "b_only": 1})  # combined>1 keeps GENE1-3 in A, all four in B
        self.assertEqual((by["GENE1"]["a_rank_combined"], by["GENE1"]["b_rank_combined"], by["GENE1"]["delta_rank_combined"]), (1, 3, 2))
        self.assertAlmostEqual(by["GENE1"]["delta_combined"], -1.5)
        self.assertEqual(by["GENENEW"]["status"], "b_only")
        self.assertIsNone(by["GENENEW"]["a_combined"])
        self.assertEqual(genes["rows"][0]["gene"], "GENE1")  # biggest |Δrank| first
        self.assertEqual([r["gene"] for r in portal_compare.compare_genes(conn, "A", "B", search="new")["rows"]], ["GENENEW"])
        self.assertEqual([r["gene"] for r in portal_compare.compare_genes(conn, "A", "B", status="both", sort="id")["rows"]], ["GENE1", "GENE2", "GENE3"])

        summary = portal_compare.compare_summary(conn, "A", "B", top_n=2)
        self.assertTrue(summary["ranks_available"])
        combined = next(m for m in summary["genes"]["metrics"] if m["metric"] == "combined")
        # top-2 by combined: A = {GENE1, GENE2}, B = {GENE2, GENENEW}; union ∩ both-present = {GENE1, GENE2}
        self.assertEqual((combined["n_top_a"], combined["n_top_b"], combined["overlap"], combined["n"]), (2, 2, 1, 2))
        self.assertAlmostEqual(combined["jaccard"], 1 / 3)
        self.assertAlmostEqual(combined["pearson"], -1.0)  # A (3.0, 1.4) vs B (1.5, 3.5)
        whole = next(m for m in portal_compare.compare_summary(conn, "A", "B", top_n=0)["genes"]["metrics"] if m["metric"] == "combined")
        self.assertEqual(whole["n"], 3)
        self.assertAlmostEqual(portal_compare.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(portal_compare.spearman([1, 2, 3], [10, 100, 1000]), 1.0)
        self.assertAlmostEqual(portal_compare.spearman([1, 2, 3], [3, 2, 1]), -1.0)

        params = portal_compare.compare_params(conn, "A", "B")
        diff = {r["parameter"]: r for r in params["rows"]}
        self.assertFalse(diff["num_chains"]["differs"])   # 10 vs 10.0 normalise equal
        self.assertTrue(diff["seed"]["differs"])
        self.assertTrue(diff["only_a"]["differs"] and diff["only_a"]["b_value"] is None)
        self.assertEqual(params["n_differ"], 2)

        self.assertEqual(portal_compare.lookup(conn, "A", "B", kind="gene", ident="GENE3")["status"], "both")
        self.assertIsNone(portal_compare.lookup(conn, "A", "B", kind="gene_set", ident="NOPE"))
        self.assertEqual(portal_compare.validate_pair(conn, "A", "A"), (400, "'a' and 'b' must be different runs"))
        self.assertEqual(portal_compare.validate_pair(conn, "A", "zzz")[0], 404)
        conn.close()

        state = portal_server.PortalState(self.db, title="t", plotly_src="about:blank")
        status, body = portal_server.handle_api(state, "/api/compare/summary", {"a": ["A"], "b": ["B"], "top_n": ["2"]})
        self.assertEqual((status, body["top_n"]), (200, 2))
        self.assertEqual(portal_server.handle_api(state, "/api/compare/genes", {"a": ["A"], "b": ["A"]})[0], 400)
        self.assertEqual(portal_server.handle_api(state, "/api/compare/lookup", {"a": ["A"], "b": ["B"], "kind": ["x"], "id": ["GENE1"]})[0], 400)
        status, body = portal_server.handle_api(state, "/api/compare/params", {"a": ["A"], "b": ["B"]})
        self.assertEqual((status, body["n_differ"]), (200, 2))

    def test_comparer_page_and_static_html(self) -> None:
        self.assertEqual(self._build(), 0)
        out = self.root / "compare.html"
        rc = portal.main(["html", "--page", "comparer", "--api-url", "http://localhost:8765", "--out", str(out), "--db", str(self.db)])
        self.assertEqual(rc, 0)
        page = out.read_text(encoding="utf-8")
        self.assertIn("<title>PIGEAN Comparer</title>", page)
        self.assertIn("/api/compare/summary", page)
        self.assertIn('window.PIGEAN_PORTAL_API_BASE = "http://localhost:8765"', page)

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
            self.assertIn('id="sheet"', html)
            self.assertIn("Build details", html)
            self.assertIn('window.PIGEAN_PORTAL_API_BASE = ""', html)
            with urllib.request.urlopen(f"{base}/compare") as resp:
                self.assertIn("PIGEAN Comparer", resp.read().decode("utf-8"))
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
