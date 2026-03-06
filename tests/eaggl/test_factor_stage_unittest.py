from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = ["eaggl.py", "factor"]
import src.eaggl as eaggl  # noqa: E402


def _options(**overrides):
    defaults = dict(
        max_num_factors=10,
        phi=1.0,
        alpha0=10.0,
        beta0=1.0,
        gene_set_filter_value=0.0,
        gene_set_pheno_filter_value=0.25,
        pheno_filter_value=0.2,
        gene_filter_value=0.1,
        factor_prune_phenos_val=None,
        factor_prune_phenos_num=None,
        factor_prune_genes_val=None,
        factor_prune_genes_num=None,
        factor_prune_gene_sets_val=None,
        factor_prune_gene_sets_num=None,
        anchor_any_pheno=False,
        anchor_any_gene=False,
        anchor_gene_set=False,
        anchor_genes=None,
        anchor_phenos=None,
        positive_controls_in=None,
        positive_controls_list=None,
        gene_set_phewas_stats_in=None,
        gene_phewas_bfs_in=None,
        run_phewas_from_gene_phewas_stats_in=None,
        no_transpose=False,
        min_lambda_threshold=1e-3,
        lmm_auth_key=None,
        lmm_model=None,
        lmm_provider="openai",
        label_gene_sets_only=False,
        label_include_phenos=False,
        label_individually=False,
        project_phenos_from_gene_sets=False,
        factors_out=None,
        factors_anchor_out=None,
        gene_set_clusters_out=None,
        gene_clusters_out=None,
        pheno_clusters_out=None,
        gene_set_anchor_clusters_out=None,
        gene_anchor_clusters_out=None,
        pheno_anchor_clusters_out=None,
        gene_pheno_stats_out=None,
        max_no_write_gene_pheno=0.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _RuntimeStub:
    def __init__(self) -> None:
        self.run_factor_kwargs = None
        self.calls = []

    def run_factor(self, **kwargs):
        self.run_factor_kwargs = kwargs

    def write_matrix_factors(self, out, write_anchor_specific=False):
        self.calls.append(("write_matrix_factors", out, write_anchor_specific))

    def write_clusters(self, gene_set_out, gene_out, pheno_out, write_anchor_specific=False):
        self.calls.append(("write_clusters", gene_set_out, gene_out, pheno_out, write_anchor_specific))

    def write_gene_pheno_statistics(self, out, min_value_to_print=0):
        self.calls.append(("write_gene_pheno_statistics", out, min_value_to_print))


class FactorStageHelpersTest(unittest.TestCase):
    def test_extract_factor_workflow_defaults(self) -> None:
        workflow = eaggl._extract_factor_workflow({})
        self.assertIsNone(workflow.workflow_id)
        self.assertFalse(workflow.factor_gene_set_x_pheno)

    def test_resolve_filter_value_prefers_gene_set_anchor(self) -> None:
        workflow = eaggl.FactorWorkflow(workflow_id="F9", factor_gene_set_x_pheno=True)
        options = _options(anchor_gene_set=True, gene_set_pheno_filter_value=0.7, pheno_filter_value=0.6, gene_filter_value=0.5)
        val = eaggl._resolve_factor_gene_or_pheno_filter_value(options, workflow)
        self.assertEqual(val, 0.7)

    def test_resolve_filter_value_uses_pheno_for_gene_set_x_pheno(self) -> None:
        workflow = eaggl.FactorWorkflow(workflow_id="F4", factor_gene_set_x_pheno=True)
        options = _options(pheno_filter_value=0.6, gene_filter_value=0.5)
        val = eaggl._resolve_factor_gene_or_pheno_filter_value(options, workflow)
        self.assertEqual(val, 0.6)

    def test_resolve_filter_value_uses_gene_default(self) -> None:
        workflow = eaggl.FactorWorkflow(workflow_id="F1", factor_gene_set_x_pheno=False)
        options = _options(pheno_filter_value=0.6, gene_filter_value=0.5)
        val = eaggl._resolve_factor_gene_or_pheno_filter_value(options, workflow)
        self.assertEqual(val, 0.5)

    def test_build_factor_execution_config_carries_masks(self) -> None:
        workflow = eaggl.FactorWorkflow(workflow_id="F6", factor_gene_set_x_pheno=True)
        factor_inputs = eaggl.FactorInputs(anchor_gene_mask=[True, False], anchor_pheno_mask=[False, True])
        options = _options(anchor_any_gene=True)
        cfg = eaggl._build_factor_execution_config(options, workflow, factor_inputs)
        self.assertEqual(cfg.anchor_gene_mask, [True, False])
        self.assertEqual(cfg.anchor_pheno_mask, [False, True])
        self.assertTrue(cfg.anchor_any_gene)
        self.assertEqual(cfg.gene_or_pheno_filter_value, options.pheno_filter_value)

    def test_run_main_factor_stage_executes_runtime_and_reports_workflow(self) -> None:
        runtime = _RuntimeStub()
        options = _options()
        mode_state = {"factor_workflow": {"id": "F1", "factor_gene_set_x_pheno": False}}
        factor_input_state = {"anchor_gene_mask": [True], "anchor_pheno_mask": [False]}
        result = eaggl._run_main_factor_stage(runtime, options, mode_state, factor_input_state)
        self.assertTrue(result.ran)
        self.assertEqual(result.workflow_id, "F1")
        self.assertIsNotNone(runtime.run_factor_kwargs)
        self.assertIn("max_num_factors", runtime.run_factor_kwargs)

    def test_write_factor_outputs_emits_only_requested_targets(self) -> None:
        runtime = _RuntimeStub()
        options = _options(
            factors_out="factors.tsv",
            factors_anchor_out="factors_anchor.tsv",
            gene_set_clusters_out="gs_cluster.tsv",
            gene_clusters_out="g_cluster.tsv",
            pheno_clusters_out="p_cluster.tsv",
            gene_set_anchor_clusters_out="gs_anchor_cluster.tsv",
            gene_anchor_clusters_out="g_anchor_cluster.tsv",
            pheno_anchor_clusters_out="p_anchor_cluster.tsv",
            gene_pheno_stats_out="gene_pheno.tsv",
            max_no_write_gene_pheno=0.2,
        )
        eaggl._write_main_factor_outputs(runtime, options)
        self.assertEqual(len(runtime.calls), 5)
        self.assertEqual(runtime.calls[0], ("write_matrix_factors", "factors.tsv", False))
        self.assertEqual(runtime.calls[1], ("write_matrix_factors", "factors_anchor.tsv", True))
        self.assertEqual(
            runtime.calls[2],
            ("write_clusters", "gs_cluster.tsv", "g_cluster.tsv", "p_cluster.tsv", False),
        )
        self.assertEqual(
            runtime.calls[3],
            ("write_clusters", "gs_anchor_cluster.tsv", "g_anchor_cluster.tsv", "p_anchor_cluster.tsv", True),
        )
        self.assertEqual(runtime.calls[4], ("write_gene_pheno_statistics", "gene_pheno.tsv", 0.2))

    def test_workflow_required_inputs_contract_for_f1_to_f9(self) -> None:
        cases = [
            ("F1", _options(), []),
            ("F2", _options(positive_controls_list=["INS"]), []),
            ("F3", _options(gene_phewas_bfs_in="gene_phewas.tsv"), []),
            ("F4", _options(anchor_phenos=["T2D"], gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F5", _options(anchor_any_pheno=True, gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F6", _options(anchor_genes=["INS"], gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F7", _options(anchor_genes=["INS", "GCK"], gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F8", _options(anchor_any_gene=True, gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F9", _options(anchor_gene_set=True, run_phewas_from_gene_phewas_stats_in="g.tsv"), []),
        ]
        for workflow_id, options, expected_missing in cases:
            with self.subTest(workflow=workflow_id):
                workflow = eaggl._classify_factor_workflow(options)
                self.assertEqual(workflow["id"], workflow_id)
                self.assertEqual(workflow["missing_required_inputs"], expected_missing)
                self.assertEqual(
                    workflow["factor_gene_set_x_pheno"],
                    eaggl._FACTOR_WORKFLOW_STRATEGY_META[workflow_id]["factor_gene_set_x_pheno"],
                )

    def test_workflow_required_inputs_missing_for_f6(self) -> None:
        workflow = eaggl._classify_factor_workflow(_options(anchor_genes=["INS"]))
        self.assertEqual(workflow["id"], "F6")
        self.assertEqual(
            workflow["missing_required_inputs"],
            ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        )


if __name__ == "__main__":
    unittest.main()
