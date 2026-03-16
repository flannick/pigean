from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sparse
import scipy.stats


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

sys.argv = ["eaggl.py", "factor"]
import eaggl.main_support as eaggl  # noqa: E402
from eaggl import gene_list_inputs as eaggl_gene_list_inputs  # noqa: E402


def _options(**overrides):
    defaults = dict(
        gene_list_in=None,
        gene_list=None,
        gene_list_id_col=1,
        gene_list_no_header=False,
        gene_list_max_fdr_q=0.05,
        positive_controls_in=None,
        positive_controls_list=None,
        positive_controls_all_in=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class StandaloneGeneListInputsTest(unittest.TestCase):
    def _build_runtime(self):
        runtime = eaggl.EagglState(background_prior=0.05, batch_size=100)
        genes = [f"G{i}" for i in range(1, 21)]
        gene_sets = ["SET_A", "SET_B", "SET_C", "SET_D"]
        dense = np.zeros((len(genes), len(gene_sets)), dtype=float)
        memberships = {
            "SET_A": {"G1", "G2", "G3", "G4"},
            "SET_B": {"G1", "G5", "G6", "G7"},
            "SET_C": {"G8", "G9", "G10", "G11", "G12", "G13"},
            "SET_D": {"G14", "G15", "G16", "G17", "G18", "G19", "G20"},
        }
        gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
        for col_index, gene_set in enumerate(gene_sets):
            for gene in memberships[gene_set]:
                dense[gene_to_index[gene], col_index] = 1.0
        runtime._set_X(sparse.csc_matrix(dense), genes, gene_sets, skip_V=True, skip_scale_factors=False, skip_N=False)
        return runtime

    def test_builds_standalone_gene_list_inputs_from_inline_genes(self) -> None:
        runtime = self._build_runtime()
        domain = eaggl.build_main_domain()
        options = _options(gene_list=["G1", "G2", "G3"])

        eaggl_gene_list_inputs.build_standalone_gene_list_inputs(domain, runtime, options)

        self.assertEqual(runtime.gene_sets, ["SET_A"])
        self.assertEqual(runtime.genes, ["G1", "G2", "G3", "G4"])
        np.testing.assert_allclose(runtime.Y, np.array([1.0, 1.0, 1.0, 0.0]))
        np.testing.assert_allclose(runtime.Y_positive_controls, np.array([1.0, 1.0, 1.0, 0.0]))
        self.assertEqual(set(runtime.gene_to_positive_controls), {"G1", "G2", "G3"})
        expected_p = scipy.stats.hypergeom.sf(2, 20, 3, 4)
        expected_weight = -math.log(expected_p) / math.sqrt(4.0)
        observed_weight = float(runtime.betas_uncorrected[0] / runtime.scale_factors[0])
        self.assertAlmostEqual(runtime.p_values[0], expected_p)
        self.assertAlmostEqual(runtime.q_values[0], expected_p * 4.0)
        self.assertAlmostEqual(observed_weight, expected_weight)

    def test_positive_controls_file_alias_uses_standalone_mode(self) -> None:
        runtime = self._build_runtime()
        domain = eaggl.build_main_domain()
        with tempfile.TemporaryDirectory() as td:
            gene_list_path = Path(td) / "positive_controls.tsv"
            gene_list_path.write_text("Gene\nG1\nG2\nG3\n", encoding="utf-8")
            options = _options(positive_controls_in=str(gene_list_path))
            eaggl_gene_list_inputs.build_standalone_gene_list_inputs(domain, runtime, options)

        self.assertEqual(runtime.gene_sets, ["SET_A"])
        self.assertEqual(runtime.genes, ["G1", "G2", "G3", "G4"])

    def test_gene_set_stats_writer_includes_hypergeom_p_and_q(self) -> None:
        runtime = self._build_runtime()
        domain = eaggl.build_main_domain()
        options = _options(gene_list=["G1", "G2", "G3"])
        eaggl_gene_list_inputs.build_standalone_gene_list_inputs(domain, runtime, options)

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "gene_set_stats.tsv"
            runtime.write_gene_set_statistics(str(output_path))
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertGreaterEqual(len(lines), 2)
        header = lines[0].split("\t")
        self.assertIn("P", header)
        self.assertIn("Q", header)
        self.assertTrue(lines[1].startswith("SET_A\t"))
