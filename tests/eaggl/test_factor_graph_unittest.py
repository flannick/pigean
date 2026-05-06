from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl import factor_graph  # noqa: E402


def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)


class EagglFactorGraphTest(unittest.TestCase):
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_ROOT) if not env.get("PYTHONPATH") else str(SRC_ROOT) + os.pathsep + env["PYTHONPATH"]
        env["PYTHONHASHSEED"] = "0"
        return env

    def _write_example_outputs(self, root: Path) -> None:
        _write_gz(
            root / "factors.out.gz",
            "Factor\tlabel\trelevance\n"
            "Factor1\timmune\t0.8\n"
            "Factor2\tmetabolic\t0.4\n",
        )
        _write_gz(
            root / "gene_clusters.out.gz",
            "Gene\tcombined\tlog_bf\tprior\tcluster\tlabel\tFactor1\tFactor2\n"
            "GENE1\t2.0\t1.0\t1.0\tFactor1\timmune\t0.9\t0.1\n"
            "GENE2\t1.5\t0.2\t1.3\tFactor2\tmetabolic\t0.1\t0.8\n"
            "GENE3\t0.5\t0.1\t0.4\tFactor2\tmetabolic\t0.001\t0.002\n",
        )
        _write_gz(
            root / "trait_factor_links.out.gz",
            "trait\tfactor\tis_anchor\tjoint_fraction\tmarginal_fraction\ttrait_neff\n"
            "TraitA\tFactor1\t0\t0.75\t0.9\t30\n"
            "TraitA\tFactor2\t0\t0.05\t0.1\t30\n"
            "TraitB\tFactor1\t0\t0.01\t0.1\t40\n"
            "TraitB\tFactor2\t0\t0.8\t0.9\t40\n"
            "TraitLowNeff\tFactor1\t0\t0.9\t0.9\t20\n"
            "TraitLowNeff\tFactor2\t0\t0.1\t0.1\t20\n",
        )

    def _write_params(self, root: Path, *, num_anchor_traits: int) -> None:
        _write_gz(
            root / "params.out.gz",
            "Parameter\tVersion\tValue\n"
            "num_anchor_traits\t1\t%d\n"
            "anchor_trait_names\t1\tTraitA,TraitB\n" % num_anchor_traits,
        )

    def test_discover_inputs_prefers_standard_eaggl_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            discovered = factor_graph.discover_inputs(root)

        self.assertTrue(discovered["factors"].endswith("factors.out.gz"))
        self.assertTrue(discovered["genes"].endswith("gene_clusters.out.gz"))
        self.assertTrue(discovered["traits"].endswith("trait_factor_links.out.gz"))

    def test_build_graph_from_standard_outputs_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            parser = factor_graph.build_parser()
            args = parser.parse_args(["--eaggl-dir", str(root), "--json-out", str(root / "graph.json"), "--seed", "3"])
            graph1 = factor_graph.build_graph_from_files(args)
            graph2 = factor_graph.build_graph_from_files(args)

        self.assertEqual(graph1, graph2)
        self.assertEqual(graph1["schema"], "eaggl_factor_graph/v1")
        self.assertEqual(graph1["factors"], ["Factor1", "Factor2"])
        node_by_id = {node["id"]: node for node in graph1["nodes"]}
        self.assertEqual(node_by_id["Factor1"]["shape"], "square")
        self.assertEqual(node_by_id["GENE1"]["shape"], "circle")
        self.assertEqual(node_by_id["TraitA"]["shape"], "diamond")
        self.assertNotIn("GENE3", node_by_id)
        self.assertNotIn("TraitLowNeff", node_by_id)
        edge_keys = {(edge["from"], edge["to"]) for edge in graph1["edges"]}
        self.assertIn(("Factor1", "GENE1"), edge_keys)
        self.assertIn(("Factor2", "TraitB"), edge_keys)

    def test_trait_neff_filter_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            parser = factor_graph.build_parser()
            args = parser.parse_args(["--eaggl-dir", str(root), "--json-out", str(root / "graph.json"), "--trait-min-neff", "0"])
            graph = factor_graph.build_graph_from_files(args)

        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertIn("TraitLowNeff", node_by_id)

    def test_module_cli_writes_json_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            json_out = root / "graph.json"
            html_out = root / "graph.html"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eaggl.factor_graph",
                    "--eaggl-dir",
                    str(root),
                    "--json-out",
                    str(json_out),
                    "--html-out",
                    str(html_out),
                    "--seed",
                    "7",
                ],
                cwd=REPO_ROOT,
                env=self._env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            graph = json.loads(json_out.read_text(encoding="utf-8"))
            html_text = html_out.read_text(encoding="utf-8")

        self.assertEqual(graph["schema"], "eaggl_factor_graph/v1")
        self.assertIn("EAGGL factor graph", html_text)
        self.assertIn("eaggl-factor-graph-data", html_text)
        self.assertIn("togglePhysicsButton", html_text)
        self.assertIn("resetLayoutButton", html_text)
        self.assertIn("zoomInButton", html_text)
        self.assertIn("zoomOutButton", html_text)
        self.assertIn("use +/- to zoom", html_text)
        self.assertNotIn('addEventListener("wheel"', html_text)
        self.assertIn("nodeFilterInput", html_text)
        self.assertIn("node-type-filter", html_text)
        self.assertIn("hideUnmatchedCheckbox", html_text)
        self.assertIn('id="hideUnmatchedCheckbox" type="checkbox">', html_text)
        self.assertIn("filterChips", html_text)
        self.assertIn("phenotypes", html_text)
        self.assertIn("addTextFilters", html_text)
        self.assertIn("addNodeInput", html_text)
        self.assertIn("addNodeOptions", html_text)
        self.assertIn("addCandidateNode", html_text)
        self.assertIn("candidate_nodes", html_text)
        self.assertIn("detailsPanel", html_text)
        self.assertIn("showNodeDetails", html_text)
        self.assertIn("showEdgeTooltip", html_text)
        self.assertIn("Near-Top Factor Loadings", html_text)
        self.assertIn("visibleColumns", html_text)
        self.assertIn("valueIsPresent", html_text)
        self.assertIn("display_label", html_text)
        self.assertIn('"nodes"', html_text)
        self.assertNotIn("&quot;nodes&quot;", html_text)
        self.assertIn("GENE1", html_text)

    def test_gene_nodes_default_to_gene_id_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            parser = factor_graph.build_parser()
            args = parser.parse_args(["--eaggl-dir", str(root), "--json-out", str(root / "graph.json")])
            graph = factor_graph.build_graph_from_files(args)

        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(node_by_id["GENE1"]["label"], "GENE1")
        self.assertEqual(node_by_id["Factor1"]["label"], "immune")

    def test_display_labels_are_truncated_but_full_labels_remain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            parser = factor_graph.build_parser()
            args = parser.parse_args(["--eaggl-dir", str(root), "--json-out", str(root / "graph.json"), "--label-max-chars", "6"])
            graph = factor_graph.build_graph_from_files(args)

        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(node_by_id["Factor2"]["label"], "metabolic")
        self.assertEqual(node_by_id["Factor2"]["display_label"], "met...")
        self.assertEqual(node_by_id["GENE1"]["display_label"], "GENE1")

    def test_display_label_truncation_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            parser = factor_graph.build_parser()
            args = parser.parse_args(["--eaggl-dir", str(root), "--json-out", str(root / "graph.json"), "--label-max-chars", "0"])
            graph = factor_graph.build_graph_from_files(args)

        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(node_by_id["Factor2"]["display_label"], "metabolic")

    def test_graph_records_trait_layout_scales(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            parser = factor_graph.build_parser()
            args = parser.parse_args(["--eaggl-dir", str(root), "--trait-coordinate-scale", "0.3", "--trait-edge-length-scale", "0.4"])
            graph = factor_graph.build_graph_from_files(args)

        self.assertEqual(graph["layout"]["trait_coordinate_scale"], 0.3)
        self.assertEqual(graph["layout"]["trait_edge_length_scale"], 0.4)

    def test_auto_color_by_uses_trait_weights_for_multi_anchor_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            self._write_params(root, num_anchor_traits=2)
            parser = factor_graph.build_parser()
            args = parser.parse_args(["--eaggl-dir", str(root), "--json-out", str(root / "graph.json")])
            graph = factor_graph.build_graph_from_files(args)

        self.assertEqual(graph["coloring"]["color_by"], "auto")
        self.assertEqual(graph["coloring"]["resolved_color_by"], "trait")
        self.assertEqual(graph["coloring"]["trait_count_for_coloring"], 2)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertNotEqual(node_by_id["Factor1"]["color"], node_by_id["Factor2"]["color"])

    def test_color_by_factor_overrides_multi_anchor_trait_coloring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            self._write_params(root, num_anchor_traits=2)
            parser = factor_graph.build_parser()
            args = parser.parse_args(["--eaggl-dir", str(root), "--json-out", str(root / "graph.json"), "--color-by", "factor"])
            graph = factor_graph.build_graph_from_files(args)

        self.assertEqual(graph["coloring"]["resolved_color_by"], "factor")
        self.assertEqual(graph["coloring"]["trait_count_for_coloring"], 0)

    def test_trait_coordinate_scale_pulls_traits_toward_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            parser = factor_graph.build_parser()
            compressed_args = parser.parse_args(["--eaggl-dir", str(root), "--trait-min-neff", "0"])
            raw_args = parser.parse_args(["--eaggl-dir", str(root), "--trait-min-neff", "0", "--trait-coordinate-scale", "1.0"])
            compressed_graph = factor_graph.build_graph_from_files(compressed_args)
            raw_graph = factor_graph.build_graph_from_files(raw_args)

        def trait_factor_distance(graph: dict) -> float:
            node_by_id = {node["id"]: node for node in graph["nodes"]}
            trait = node_by_id["TraitA"]
            factors = [node_by_id["Factor1"], node_by_id["Factor2"]]
            center_x = sum(node["x"] for node in factors) / len(factors)
            center_y = sum(node["y"] for node in factors) / len(factors)
            return ((trait["x"] - center_x) ** 2 + (trait["y"] - center_y) ** 2) ** 0.5

        self.assertLess(trait_factor_distance(compressed_graph), trait_factor_distance(raw_graph))

    def test_hidden_gene_candidates_are_embedded_for_interactive_add_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            _write_gz(
                root / "gene_clusters.out.gz",
                "Gene\tcombined\tlog_bf\tprior\tcluster\tlabel\tFactor1\tFactor2\n"
                "GENE1\t2.0\t1.0\t1.0\tFactor1\timmune\t0.9\t0.1\n"
                "GENE2\t1.5\t0.2\t1.3\tFactor2\tmetabolic\t0.1\t0.8\n"
                "GENE4\t1.2\t0.2\t1.0\tFactor1\timmune\t0.7\t0.0\n",
            )
            parser = factor_graph.build_parser()
            args = parser.parse_args(
                [
                    "--eaggl-dir",
                    str(root),
                    "--json-out",
                    str(root / "graph.json"),
                    "--max-num-gene-nodes-per-factor",
                    "1",
                ]
            )
            graph = factor_graph.build_graph_from_files(args)

        node_by_id = {node["id"]: node for node in graph["nodes"]}
        candidate_by_id = {node["id"]: node for node in graph["candidate_nodes"]}
        self.assertIn("GENE1", node_by_id)
        self.assertIn("GENE2", node_by_id)
        self.assertNotIn("GENE4", node_by_id)
        self.assertIn("GENE4", candidate_by_id)
        self.assertEqual(candidate_by_id["GENE4"]["kind"], "gene")
        self.assertEqual(candidate_by_id["GENE4"]["provenance"]["near_top_factor_loadings"][0]["factor"], "Factor1")
        self.assertEqual(candidate_by_id["GENE4"]["provenance"]["near_top_factor_loadings"][0]["factor_display_label"], "immune")
        candidate_edge_keys = {(edge["from"], edge["to"]) for edge in graph["candidate_edges"]}
        self.assertIn(("Factor1", "GENE4"), candidate_edge_keys)

    def test_provenance_inputs_are_embedded_in_nodes_and_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            _write_gz(
                root / "gene_phewas.tsv.gz",
                "Gene\tTrait\tCombined\tDirect\tIndirect\n"
                "GENE1\tTraitA\t2.5\t1.2\t0.8\n"
                "GENE1\tTraitB\t0.4\t0.1\t0.3\n",
            )
            _write_gz(
                root / "gene_set_clusters.out.gz",
                "Gene_Set\tcombined\tlog_bf\tcluster\tlabel\tFactor1\tFactor2\n"
                "GS1\t1.0\t1.0\tFactor1\tGS label\t0.9\t0.0\n",
            )
            parser = factor_graph.build_parser()
            args = parser.parse_args(
                [
                    "--eaggl-dir",
                    str(root),
                    "--gene-phewas-stats-in",
                    str(root / "gene_phewas.tsv.gz"),
                ]
            )
            graph = factor_graph.build_graph_from_files(args)

        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(node_by_id["GENE1"]["provenance"]["anchor_support"][0]["anchor"], "TraitA")
        self.assertAlmostEqual(node_by_id["GENE1"]["provenance"]["anchor_support"][0]["combined"], 2.5)
        near_top = node_by_id["GENE1"]["provenance"]["near_top_factor_loadings"]
        self.assertEqual([row["factor"] for row in near_top], ["Factor1"])
        self.assertEqual([row["factor_display_label"] for row in near_top], ["immune"])
        self.assertEqual(node_by_id["GENE1"]["provenance"]["near_top_factor_loading_rule"], "loading >= max_loading - 0.01")
        factor1 = node_by_id["Factor1"]["provenance"]
        self.assertIn("relevance_by_anchor", factor1)
        self.assertEqual(factor1["top_gene_loadings"][0]["id"], "GENE1")
        self.assertEqual(factor1["top_gene_set_loadings"][0]["id"], "GS1")
        edge = next(edge for edge in graph["edges"] if edge["from"] == "Factor1" and edge["to"] == "GENE1")
        self.assertEqual(edge["provenance"]["source_table"], "gene_clusters")
        self.assertEqual(edge["provenance"]["weight_field"], "Factor1")

    def test_module_cli_can_start_html_with_physics_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            html_out = root / "graph.html"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eaggl.factor_graph",
                    "--eaggl-dir",
                    str(root),
                    "--html-out",
                    str(html_out),
                    "--html-physics",
                ],
                cwd=REPO_ROOT,
                env=self._env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            html_text = html_out.read_text(encoding="utf-8")

        self.assertIn("let physicsEnabled = true", html_text)
        self.assertIn("requestAnimationFrame(tick)", html_text)

    def test_module_cli_can_write_static_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_example_outputs(root)
            html_out = root / "graph.html"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eaggl.factor_graph",
                    "--eaggl-dir",
                    str(root),
                    "--html-out",
                    str(html_out),
                    "--no-html-interactive",
                ],
                cwd=REPO_ROOT,
                env=self._env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            html_text = html_out.read_text(encoding="utf-8")

        self.assertNotIn("togglePhysicsButton", html_text)
        self.assertIn("<circle", html_text)

    def test_module_cli_fails_without_outputs(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "eaggl.factor_graph", "--eaggl-dir", "/no/such/dir"],
            cwd=REPO_ROOT,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Need at least one of --html-out, --json-out, or --pdf-out", (proc.stderr or "") + (proc.stdout or ""))


if __name__ == "__main__":
    unittest.main()
