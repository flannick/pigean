from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pigean import dashboard  # noqa: E402


def _write_gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


class DashboardPayloadTest(unittest.TestCase):
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_ROOT) if not env.get("PYTHONPATH") else str(SRC_ROOT) + os.pathsep + env["PYTHONPATH"]
        return env

    def _write_pigean_outputs(self, root: Path) -> None:
        _write_gz(
            root / "pigean.gene_stats.out.gz",
            "Gene\tcombined\tlog_bf\tprior\tN\tChrom\tStart\tEnd\n"
            "GENE1\t3.0\t1.2\t1.8\t5\t1\t100\t200\n"
            "GENE2\t1.4\t0.4\t1.0\t4\t1\t300\t400\n",
        )
        _write_gz(
            root / "pigean.gene_set_stats.out.gz",
            "Gene_Set\tlabel\tbeta\tbeta_uncorrected\tP_orig\tZ_orig\tN\n"
            "SET1\timmune\t0.8\t0.9\t0.01\t2.5\t2\n"
            "SET2\tmetabolic\t0.2\t0.3\t0.2\t1.0\t2\n",
        )

    def _write_eaggl_outputs(self, root: Path) -> None:
        _write_gz(
            root / "factors.out.gz",
            "Factor\tlabel\tlambda\tfactor_tier\tcombined_mass_fraction\ttop_genes\ttop_gene_sets\n"
            "Factor1\timmune\t1.2\tprimary\t0.2\tGENE1\tSET1\n",
        )
        _write_gz(
            root / "gene_clusters.out.gz",
            "Gene\tcombined\tlog_bf\tprior\tcluster\tlabel\tFactor1\tRelative_Factor1\n"
            "GENE1\t3.0\t1.2\t1.8\tFactor1\timmune\t0.9\t1.0\n"
            "GENE2\t1.4\t0.4\t1.0\tFactor1\timmune\t0.01\t0.1\n",
        )
        _write_gz(
            root / "gene_clusters_full.out.gz",
            "Gene\tcombined\tlog_bf\tprior\tcluster\tlabel\tFactor1\tRelative_Factor1\n"
            "GENE1\t3.0\t1.2\t1.8\tFactor1\timmune\t0.8\t1.0\n"
            "GENE3\t0.2\t0.1\t0.1\tFactor1\timmune\t0.7\t1.0\n",
        )
        _write_gz(
            root / "gene_clusters_full_via_gene_sets.out.gz",
            "Gene\tcombined\tlog_bf\tprior\tcluster\tlabel\tFactor1\tRelative_Factor1\n"
            "GENE1\t3.0\t1.2\t1.8\tFactor1\timmune\t0.6\t1.0\n"
            "GENE4\t0.3\t0.1\t0.2\tFactor1\timmune\t0.5\t1.0\n",
        )
        _write_gz(
            root / "gene_set_clusters.out.gz",
            "Gene_Set\tlabel\tbeta\tbeta_uncorrected\tcluster\tFactor1\tRelative_Factor1\n"
            "SET1\timmune\t0.8\t0.9\tFactor1\t0.8\t1.0\n",
        )
        _write_gz(
            root / "trait_factor_links.out.gz",
            "trait\tfactor\tis_anchor\tjoint_fraction\tmarginal_fraction\ttrait_neff\n"
            "TraitA\tFactor1\t1\t0.7\t0.8\t300\n"
            "TraitLow\tFactor1\t0\t0.9\t0.9\t10\n",
        )
        _write_gz(
            root / "factor_metrics.out.gz",
            "Factor\tlabel\tfactor_gene_mass\tfactor_size_score\tfactor_coherence_score\tgene_effective_support\tgene_max_jaccard\n"
            "Factor1\timmune\t42\t0.85\t0.77\t12.5\t0.18\n",
        )
        _write_gz(
            root / "learn_phi_report.out.gz",
            "selected\tselection_reason\tphi\tphi_composite_score\tfactor_size_score\tnonoverlap_score\tcoverage_score\treconstruction_score\tcoherence_score\tfactor_balance_score\n"
            "1\tcomposite_score\t0.01\t0.91\t0.85\t0.8\t0.95\t0.7\t0.77\t0.6\n",
        )
        (root / "factor_graph.html").write_text("<html><body>factor graph</body></html>", encoding="utf-8")

    def test_payload_loads_complete_pigean_and_eaggl_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdir = root / "pigean"
            edir = root / "eaggl"
            self._write_pigean_outputs(pdir)
            self._write_eaggl_outputs(edir)
            x_in = root / "sets.gmt"
            x_in.write_text("SET1\tdesc\tGENE1\tGENE2\n", encoding="utf-8")
            parser = dashboard.build_parser()
            args = parser.parse_args([
                "--pigean-run", f"run1:{pdir}",
                "--eaggl-run", f"run1:gene_x_gene:{edir}",
                "--x-input", str(x_in),
                "--json-out", str(root / "dashboard.json"),
            ])
            args.run_titles = {}
            args.trait_ids = {}
            payload = dashboard.build_payload(args)

        self.assertEqual(payload["schema"], "pigean_dashboard/v1")
        self.assertEqual(len(payload["pigean_runs"]), 1)
        self.assertEqual(len(payload["eaggl_runs"]), 1)
        self.assertEqual(payload["pigean_runs"][0]["genes"][0]["gene"], "GENE1")
        self.assertIn("GENE1", payload["pigean_runs"][0]["gene_expansions"])
        factor = payload["eaggl_runs"]["run1::gene_x_gene"]["factors"][0]
        self.assertEqual(factor["factor"], "Factor1")
        self.assertEqual(len(factor["genes"]), 1)
        self.assertEqual(len(factor["phenotypes"]), 1)
        self.assertIn("anchor_traits", payload["eaggl_runs"]["run1::gene_x_gene"])
        self.assertEqual(payload["eaggl_runs"]["run1::gene_x_gene"]["anchor_traits"][0]["trait"], "TraitA")
        self.assertEqual(factor[payload["eaggl_runs"]["run1::gene_x_gene"]["anchor_traits"][0]["column"]], 0.7)
        self.assertEqual(factor["factor_gene_mass"], 42)
        self.assertEqual(factor["factor_size_score"], 0.85)
        self.assertEqual(payload["eaggl_runs"]["run1::gene_x_gene"]["selected_phi_metrics"]["phi_composite_score"], 0.91)
        eaggl_run = payload["eaggl_runs"]["run1::gene_x_gene"]
        self.assertTrue(eaggl_run["factor_graph_available"])
        self.assertIn("eaggl_groups", payload)
        self.assertEqual(payload["eaggl_groups"]["run1"][0]["mode_ids"], ["gene_x_gene"])
        self.assertEqual(set(eaggl_run["gene_loading_sources"]), {"discovery", "full_direct", "full_via_gene_sets"})
        self.assertEqual(eaggl_run["gene_loading_sources"]["full_direct"]["by_factor"]["Factor1"][0]["gene"], "GENE1")
        self.assertEqual(eaggl_run["gene_loading_sources"]["full_via_gene_sets"]["by_factor"]["Factor1"][0]["gene"], "GENE1")

    def test_dashboard_merges_separate_trait_projection_and_enrichment_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            edir = root / "eaggl"
            self._write_eaggl_outputs(edir)
            (edir / "trait_factor_links.out.gz").unlink()
            _write_gz(
                edir / "trait_factor_links.nnls.out.gz",
                "trait\tfactor\tnnls_loading\tcosine_loading\teuclidean_distance\ttrait_neff\n"
                "TraitA\tFactor1\t0.7\t0.8\t0.2\t300\n"
                "TraitB\tFactor1\t0.1\t0.2\t0.9\t300\n",
            )
            _write_gz(
                edir / "factor_trait_pigean_enrichments.out.gz",
                "Trait\tGene_Set\tbeta\tbeta_uncorrected\tbeta_tilde\tse\tz\tp_value\n"
                "TraitA\tFactor1\t0.2\t0.6\t0.9\t0.1\t9\t1e-6\n"
                "TraitC\tFactor1\t0.3\t0.7\t1.1\t0.2\t5.5\t2e-5\n",
            )
            parser = dashboard.build_parser()
            args = parser.parse_args([
                "--eaggl-run", f"run1:gene_x_gene:{edir}",
                "--json-out", str(root / "dashboard.json"),
            ])
            args.run_titles = {}
            args.trait_ids = {}
            payload = dashboard.build_payload(args)

        eaggl_run = payload["eaggl_runs"]["run1::gene_x_gene"]
        factor = eaggl_run["factors"][0]
        traits = {row["trait"]: row for row in factor["phenotypes"]}
        self.assertEqual(traits["TraitA"]["nnls_loading"], 0.7)
        self.assertEqual(traits["TraitA"]["cosine_loading"], 0.8)
        self.assertEqual(traits["TraitA"]["beta"], 0.2)
        self.assertEqual(traits["TraitA"]["beta_uncorrected"], 0.6)
        self.assertEqual(traits["TraitA"]["beta_tilde"], 0.9)
        self.assertEqual(traits["TraitA"]["p_value"], 1e-6)
        self.assertEqual(traits["TraitC"]["trait_enrichment_only"], "1")
        self.assertEqual(traits["TraitC"]["beta"], 0.3)
        self.assertNotIn("TraitB", traits)
        self.assertTrue(eaggl_run["paths"]["trait_factor_projection"].endswith("trait_factor_links.nnls.out.gz"))
        self.assertTrue(eaggl_run["paths"]["factor_trait_pigean_enrichments"].endswith("factor_trait_pigean_enrichments.out.gz"))

    def test_missing_outputs_are_recorded_as_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parser = dashboard.build_parser()
            args = parser.parse_args([
                "--pigean-run", f"missing:{root / 'missing_pigean'}",
                "--eaggl-run", f"missing:gene_x_gene:{root / 'missing_eaggl'}",
                "--json-out", str(root / "dashboard.json"),
            ])
            args.run_titles = {}
            args.trait_ids = {}
            payload = dashboard.build_payload(args)

        self.assertTrue(payload["pigean_runs"][0]["warnings"])
        self.assertTrue(payload["eaggl_runs"]["missing::gene_x_gene"]["warnings"])
        self.assertEqual(payload["pigean_runs"][0]["genes"], [])
        self.assertEqual(payload["eaggl_runs"]["missing::gene_x_gene"]["factors"], [])

    def test_eaggl_only_input_gets_placeholder_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            edir = root / "eaggl"
            self._write_eaggl_outputs(edir)
            parser = dashboard.build_parser()
            args = parser.parse_args([
                "--eaggl-run", f"run1:gene_x_gene:{edir}",
                "--json-out", str(root / "dashboard.json"),
            ])
            args.run_titles = {}
            args.trait_ids = {}
            payload = dashboard.build_payload(args)

        self.assertEqual(payload["pigean_runs"][0]["run_id"], "run1")
        self.assertTrue(payload["pigean_runs"][0]["warnings"])
        self.assertEqual(payload["eaggl_runs"]["run1::gene_x_gene"]["factors"][0]["factor"], "Factor1")

    def test_phi_sweep_bundle_loads_per_phi_eaggl_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep = root / "sweep"
            for tag in ("0p02", "0p04"):
                edir = sweep / f"phi_{tag}" / "eaggl"
                self._write_eaggl_outputs(edir)
            (sweep / "summary.tsv").write_text("phi\tselected\n0.02\t0\n0.04\t1\n", encoding="utf-8")
            parser = dashboard.build_parser()
            args = parser.parse_args([
                "--eaggl-phi-sweep", f"run1:gene_x_gene:{sweep}",
                "--json-out", str(root / "dashboard.json"),
            ])
            args.run_titles = {}
            args.trait_ids = {}
            payload = dashboard.build_payload(args)

        self.assertEqual(len(payload["eaggl_runs"]), 2)
        keys = list(payload["eaggl_runs"])
        self.assertEqual(keys[0], "run1::gene_x_gene_phi_0p04")
        self.assertEqual(payload["eaggl_runs"][keys[0]]["phi"], 0.04)
        groups = payload["eaggl_groups"]["run1"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["group_id"], "gene_x_gene")
        self.assertEqual(set(groups[0]["mode_ids"]), {"gene_x_gene_phi_0p02", "gene_x_gene_phi_0p04"})

    def test_phi_sweep_bundle_loads_aggregate_factor_phi_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep = root / "sweep"
            sweep.mkdir()
            _write_gz(
                sweep / "factor_phi_factors.out.gz",
                "phi\tFactor\tlabel\tlambda\tcombined_mass_fraction\ttop_genes\ttop_gene_sets\n"
                "0.02\tFactor1\timmune low\t1.0\t0.1\tGENE1\tSET1\n"
                "0.04\tFactor1\timmune high\t2.0\t0.2\tGENE2\tSET2\n",
            )
            _write_gz(
                sweep / "factor_phi_metrics.out.gz",
                "phi\tFactor\tfactor_gene_mass\tfactor_size_score\n"
                "0.02\tFactor1\t20\t0.5\n"
                "0.04\tFactor1\t40\t0.9\n",
            )
            _write_gz(
                sweep / "factor_phi_gene_clusters.out.gz",
                "phi\tGene\tcombined\tcluster\tlabel\tFactor1\tCosine_Factor1\tEuclidean_Factor1\n"
                "0.02\tGENE1\t3.0\tFactor1\timmune low\t0.8\t0.9\t0.1\n"
                "0.04\tGENE2\t4.0\tFactor1\timmune high\t0.7\t0.8\t0.2\n",
            )
            _write_gz(
                sweep / "factor_phi_gene_set_clusters.out.gz",
                "phi\tGene_Set\tlabel\tbeta\tbeta_uncorrected\tcluster\tFactor1\tCosine_Factor1\tEuclidean_Factor1\n"
                "0.02\tSET1\timmune\t0.2\t0.3\tFactor1\t0.6\t0.7\t0.3\n"
                "0.04\tSET2\tmetabolic\t0.4\t0.5\tFactor1\t0.9\t0.95\t0.05\n",
            )
            _write_gz(
                sweep / "phi_selection_metrics_wide.out.gz",
                "phi\tselected\tphi_composite_score\tcoverage_score\n"
                "0.02\t0\t0.2\t0.3\n"
                "0.04\t1\t0.8\t0.9\n",
            )
            parser = dashboard.build_parser()
            args = parser.parse_args([
                "--eaggl-phi-sweep", f"run1:gene_x_gene:{sweep}",
                "--json-out", str(root / "dashboard.json"),
            ])
            args.run_titles = {}
            args.trait_ids = {}
            payload = dashboard.build_payload(args)

        self.assertEqual(len(payload["eaggl_runs"]), 2)
        first_key = list(payload["eaggl_runs"])[0]
        self.assertEqual(first_key, "run1::gene_x_gene_phi_0p04")
        selected_run = payload["eaggl_runs"][first_key]
        self.assertEqual(selected_run["phi"], 0.04)
        self.assertEqual(selected_run["selected_phi_metrics"]["phi_composite_score"], 0.8)
        factor = selected_run["factors"][0]
        self.assertEqual(factor["label"], "immune high")
        self.assertEqual(factor["factor_gene_mass"], 40)
        self.assertEqual(factor["genes"][0]["gene"], "GENE2")
        self.assertEqual(factor["gene_sets"][0]["gene_set"], "SET2")
        self.assertFalse(selected_run["factor_graph_available"])
        self.assertEqual(set(selected_run["gene_loading_sources"]), {"discovery"})

    def test_explicit_eaggl_group_combines_standalone_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for mode in ("alpha", "beta"):
                self._write_eaggl_outputs(root / mode)
            parser = dashboard.build_parser()
            args = parser.parse_args([
                "--eaggl-run", f"run1:alpha:{root / 'alpha'}",
                "--eaggl-run", f"run1:beta:{root / 'beta'}",
                "--eaggl-group", "run1:alpha:comparison:Comparison",
                "--eaggl-group", "run1:beta:comparison:Comparison",
                "--json-out", str(root / "dashboard.json"),
            ])
            args.run_titles = {}
            args.trait_ids = {}
            payload = dashboard.build_payload(args)

        groups = payload["eaggl_groups"]["run1"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["group_id"], "comparison")
        self.assertEqual(set(groups[0]["mode_ids"]), {"alpha", "beta"})

    def test_pigean_group_assigns_runs_to_trait_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            t2d = root / "t2d"
            mody = root / "mody"
            for pdir in (t2d, mody):
                self._write_pigean_outputs(pdir)
            self._write_eaggl_outputs(root / "eaggl")
            parser = dashboard.build_parser()
            args = parser.parse_args([
                "--pigean-run", f"t2d_pre:{t2d}",
                "--pigean-run", f"mody_pre:{mody}",
                "--pigean-group", "multi_pre:t2d_pre:T2D + MODY pre-exclusion",
                "--pigean-group", "multi_pre:mody_pre:T2D + MODY pre-exclusion",
                "--eaggl-run", f"multi_pre:gene_x_gene:{root / 'eaggl'}",
                "--json-out", str(root / "dashboard.json"),
            ])
            args.run_titles = {}
            args.trait_ids = {}
            payload = dashboard.build_payload(args)

        self.assertEqual(payload["pigean_groups"][0]["group_id"], "multi_pre")
        self.assertEqual(payload["pigean_groups"][0]["title"], "T2D + MODY pre-exclusion")
        self.assertEqual(payload["pigean_groups"][0]["run_ids"], ["t2d_pre", "mody_pre"])
        self.assertIn("multi_pre", payload["eaggl_groups"])

    def test_cli_rejects_malformed_run_specs(self) -> None:
        parser = dashboard.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--pigean-run", "not_a_spec", "--json-out", "x.json"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--eaggl-run", "run:mode_only", "--json-out", "x.json"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--pigean-group", "missing_run_id", "--json-out", "x.json"])

    def test_module_cli_writes_json_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdir = root / "pigean"
            self._write_pigean_outputs(pdir)
            html_out = root / "dashboard.html"
            json_out = root / "dashboard.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pigean.dashboard",
                    "--pigean-run",
                    f"run1:{pdir}",
                    "--html-out",
                    str(html_out),
                    "--json-out",
                    str(json_out),
                ],
                cwd=REPO_ROOT,
                env=self._env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            html = html_out.read_text(encoding="utf-8")

        self.assertEqual(payload["schema"], "pigean_dashboard/v1")
        self.assertIn("PIGEAN/EAGGL Dashboard", html)
        self.assertIn("DATA_PAYLOAD_GZIP_BASE64", html)
        self.assertIn("PIGEAN genes", html)
        self.assertIn("Rows <select", html)
        self.assertIn("data-open-row", html)
        self.assertIn("data-column-info", html)
        self.assertIn("groupSelect", html)
        self.assertIn("phi-metric-heatmap", html)
        self.assertIn("Phi composite columns are run-level selected-candidate metrics", html)
        self.assertIn("restoreFocus", html)
        self.assertIn("data-column-filter-table", html)
        self.assertIn("numeric-filter", html)
        self.assertIn("loading-heatmap", html)
        self.assertIn("heatmapMetricSelect", html)
        self.assertIn("data-heatmap-regex-table", html)
        self.assertIn("heatmap-regex-label", html)
        self.assertIn("function regexMatch", html)
        self.assertIn("new RegExp(String(raw), \"i\")", html)
        self.assertIn("data-heatmap-tip", html)
        self.assertIn("bindHeatmapTooltips", html)
        self.assertIn("factor-tabs", html)
        self.assertIn("refreshEagglTable", html)
        self.assertIn("data-open-row-tab", html)
        self.assertIn("pigeanGroupSelect", html)
        self.assertIn("selectedEagglRunId", html)


if __name__ == "__main__":
    unittest.main()
