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
        eaggl_run = payload["eaggl_runs"]["run1::gene_x_gene"]
        self.assertTrue(eaggl_run["factor_graph_available"])
        self.assertEqual(set(eaggl_run["gene_loading_sources"]), {"discovery", "full_direct", "full_via_gene_sets"})
        self.assertEqual(eaggl_run["gene_loading_sources"]["full_direct"]["by_factor"]["Factor1"][0]["gene"], "GENE1")
        self.assertEqual(eaggl_run["gene_loading_sources"]["full_via_gene_sets"]["by_factor"]["Factor1"][0]["gene"], "GENE1")

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

    def test_cli_rejects_malformed_run_specs(self) -> None:
        parser = dashboard.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--pigean-run", "not_a_spec", "--json-out", "x.json"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--eaggl-run", "run:mode_only", "--json-out", "x.json"])

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
        self.assertIn("restoreFocus", html)
        self.assertIn("data-column-filter-table", html)
        self.assertIn("numeric-filter", html)
        self.assertIn("loading-heatmap", html)
        self.assertIn("heatmapMetricSelect", html)
        self.assertIn("refreshEagglTable", html)
        self.assertIn("data-open-row-tab", html)


if __name__ == "__main__":
    unittest.main()
