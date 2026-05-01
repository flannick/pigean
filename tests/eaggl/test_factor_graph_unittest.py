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
            "TraitA\tFactor1\t0\t0.75\t0.9\t10\n"
            "TraitA\tFactor2\t0\t0.05\t0.1\t10\n"
            "TraitB\tFactor1\t0\t0.01\t0.1\t8\n"
            "TraitB\tFactor2\t0\t0.8\t0.9\t8\n",
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
        edge_keys = {(edge["from"], edge["to"]) for edge in graph1["edges"]}
        self.assertIn(("Factor1", "GENE1"), edge_keys)
        self.assertIn(("Factor2", "TraitB"), edge_keys)

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
