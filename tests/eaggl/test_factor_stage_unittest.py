from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

sys.argv = ["eaggl.py", "factor"]
import eaggl.main_support as eaggl  # noqa: E402


def _options(**overrides):
    defaults = dict(
        max_num_factors=10,
        phi=1.0,
        alpha0=10.0,
        beta0=1.0,
        seed=0,
        factor_runs=1,
        consensus_nmf=False,
        consensus_min_factor_cosine=0.7,
        consensus_min_run_support=0.5,
        consensus_aggregation="median",
        consensus_stats_out=None,
        learn_phi=False,
        learn_phi_max_redundancy=0.5,
        learn_phi_max_redundancy_q90=0.35,
        learn_phi_runs_per_step=1,
        learn_phi_min_run_support=0.6,
        learn_phi_min_stability=0.85,
        learn_phi_max_fit_loss_frac=0.05,
        learn_phi_k_band_frac=0.9,
        learn_phi_max_steps=8,
        learn_phi_expand_factor=2.0,
        learn_phi_weight_floor=None,
        learn_phi_report_out=None,
        factor_phi_metrics_out=None,
        factor_backend="full",
        learn_phi_backend="sentinel_pruned",
        blockwise_gene_set_block_size=5000,
        blockwise_epochs=3,
        blockwise_shuffle_blocks=True,
        blockwise_warm_start=True,
        blockwise_max_blocks=None,
        blockwise_report_out=None,
        learn_phi_prune_genes_num=1000,
        learn_phi_prune_gene_sets_num=1000,
        learn_phi_max_num_iterations=None,
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
        factor_phewas_full_output=False,
        anchor_genes=None,
        anchor_phenos=None,
        gene_list_in=None,
        gene_list=None,
        gene_list_id_col=1,
        gene_list_no_header=False,
        gene_list_max_fdr_q=0.05,
        positive_controls_in=None,
        positive_controls_list=None,
        positive_controls_all_in=None,
        gene_set_phewas_stats_in=None,
        gene_phewas_bfs_in=None,
        run_phewas=False,
        run_phewas_input=None,
        run_factor_phewas=False,
        run_factor_phewas_input=None,
        no_transpose=False,
        min_lambda_threshold=1e-3,
        lmm_auth_key=None,
        lmm_model=None,
        lmm_provider="openai",
        label_gene_sets_only=False,
        label_include_phenos=False,
        label_individually=False,
        keep_original_loadings=False,
        project_phenos_from_gene_sets=False,
        pheno_capture_input="weighted_thresholded",
        factors_out=None,
        factor_metrics_out=None,
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

    def write_factor_metrics(self, out):
        self.calls.append(("write_factor_metrics", out))

    def write_consensus_factor_diagnostics(self, out):
        self.calls.append(("write_consensus_factor_diagnostics", out))

    def write_clusters(self, gene_set_out, gene_out, pheno_out, write_anchor_specific=False):
        self.calls.append(("write_clusters", gene_set_out, gene_out, pheno_out, write_anchor_specific))

    def write_gene_pheno_statistics(self, out, min_value_to_print=0):
        self.calls.append(("write_gene_pheno_statistics", out, min_value_to_print))


class _FactorPhewasRuntimeStub:
    def __init__(self) -> None:
        self.recorded_params = None
        self.output_path = None
        self.gene_pheno_Y = None
        self.gene_pheno_combined_prior_Ys = None
        self.gene_pheno_priors = None
        self.num_gene_phewas_filtered = 0

    def num_factors(self):
        return 2

    def _record_params(self, params, overwrite=False):
        self.recorded_params = (dict(params), overwrite)

    def write_factor_phewas_statistics(self, path):
        self.output_path = path


class FactorStageHelpersTest(unittest.TestCase):
    def test_zero_uncorrected_filter_hook_accepts_shared_runtime_kwargs(self) -> None:
        runtime = SimpleNamespace(p_values=None, gene_sets=[])
        sort_rank = [1.0, 0.5]
        result = eaggl._maybe_filter_zero_uncorrected_betas_after_x_read(
            runtime_state=runtime,
            sort_rank=sort_rank,
            skip_betas=False,
            filter_gene_set_p=1.0,
            filter_using_phewas=False,
            retain_all_beta_uncorrected=False,
            independent_betas_only=False,
            max_num_burn_in=10,
            max_num_iter_betas=20,
            min_num_iter_betas=5,
            num_chains_betas=2,
            r_threshold_burn_in_betas=1.01,
            use_max_r_for_convergence_betas=True,
            max_frac_sem_betas=0.01,
            max_allowed_batch_correlation=None,
            sparse_solution=False,
            sparse_frac_betas=0.001,
        )
        self.assertIs(result, sort_rank)
        self.assertIsNone(
            eaggl._maybe_reduce_gene_sets_to_max_after_x_read(
                runtime_state=runtime,
                skip_betas=False,
                max_num_gene_sets=5000,
                sort_rank=sort_rank,
                retain_all_beta_uncorrected=False,
                independent_betas_only=False,
            )
        )

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
        self.assertEqual(cfg.factor_runs, 1)
        self.assertFalse(cfg.consensus_nmf)
        self.assertEqual(cfg.gene_set_filter_type, "betas_uncorrected")
        self.assertEqual(cfg.gene_or_pheno_filter_type, "gene_phewas_combined")
        self.assertEqual(cfg.max_num_iterations, 100)
        self.assertEqual(cfg.rel_tol, 1e-4)

    def test_build_factor_execution_config_carries_phi_learning_controls(self) -> None:
        workflow = eaggl.FactorWorkflow(workflow_id="F1", factor_gene_set_x_pheno=False)
        factor_inputs = eaggl.FactorInputs(anchor_gene_mask=None, anchor_pheno_mask=None)
        options = _options(
            learn_phi=True,
            learn_phi_max_redundancy=0.55,
            learn_phi_max_redundancy_q90=0.25,
            learn_phi_runs_per_step=7,
            learn_phi_min_run_support=0.7,
            learn_phi_min_stability=0.9,
            learn_phi_max_fit_loss_frac=0.03,
            learn_phi_k_band_frac=0.8,
            learn_phi_max_steps=6,
            learn_phi_expand_factor=5.0,
            learn_phi_weight_floor=0.02,
            learn_phi_report_out="phi.tsv",
            factor_phi_metrics_out="phi_factor_metrics.tsv",
            factor_backend="blockwise_global_w",
            learn_phi_backend="blockwise_global_w",
            blockwise_gene_set_block_size=123,
            blockwise_epochs=4,
            blockwise_shuffle_blocks=False,
            blockwise_warm_start=False,
            blockwise_max_blocks=7,
            blockwise_report_out="blockwise.tsv",
            learn_phi_prune_genes_num=900,
            learn_phi_prune_gene_sets_num=1000,
            learn_phi_max_num_iterations=25,
        )
        cfg = eaggl._build_factor_execution_config(options, workflow, factor_inputs)
        self.assertTrue(cfg.learn_phi)
        self.assertEqual(cfg.learn_phi_max_redundancy, 0.55)
        self.assertEqual(cfg.learn_phi_max_redundancy_q90, 0.25)
        self.assertEqual(cfg.learn_phi_runs_per_step, 7)
        self.assertEqual(cfg.learn_phi_min_run_support, 0.7)
        self.assertEqual(cfg.learn_phi_min_stability, 0.9)
        self.assertEqual(cfg.learn_phi_max_fit_loss_frac, 0.03)
        self.assertEqual(cfg.learn_phi_k_band_frac, 0.8)
        self.assertEqual(cfg.learn_phi_max_steps, 6)
        self.assertEqual(cfg.learn_phi_expand_factor, 5.0)
        self.assertEqual(cfg.learn_phi_weight_floor, 0.02)
        self.assertEqual(cfg.learn_phi_report_out, "phi.tsv")
        self.assertEqual(cfg.factor_phi_metrics_out, "phi_factor_metrics.tsv")
        self.assertEqual(cfg.factor_backend, "blockwise_global_w")
        self.assertEqual(cfg.learn_phi_backend, "blockwise_global_w")
        self.assertEqual(cfg.blockwise_gene_set_block_size, 123)
        self.assertEqual(cfg.blockwise_epochs, 4)
        self.assertFalse(cfg.blockwise_shuffle_blocks)
        self.assertFalse(cfg.blockwise_warm_start)
        self.assertEqual(cfg.blockwise_max_blocks, 7)
        self.assertEqual(cfg.blockwise_report_out, "blockwise.tsv")
        self.assertEqual(cfg.learn_phi_prune_genes_num, 900)
        self.assertEqual(cfg.learn_phi_prune_gene_sets_num, 1000)
        self.assertEqual(cfg.learn_phi_max_num_iterations, 25)

    def test_build_factor_execution_config_defaults_phi_search_prune_to_1000(self) -> None:
        workflow = eaggl.FactorWorkflow(workflow_id="F1", factor_gene_set_x_pheno=False)
        factor_inputs = eaggl.FactorInputs(anchor_gene_mask=None, anchor_pheno_mask=None)
        options = _options(learn_phi=True, learn_phi_prune_genes_num=1000, learn_phi_prune_gene_sets_num=1000)
        cfg = eaggl._build_factor_execution_config(options, workflow, factor_inputs)
        self.assertEqual(cfg.learn_phi_prune_genes_num, 1000)
        self.assertEqual(cfg.learn_phi_prune_gene_sets_num, 1000)

    def test_build_factor_execution_config_tracks_keep_original_loadings(self) -> None:
        workflow = eaggl.FactorWorkflow(workflow_id="F1", factor_gene_set_x_pheno=False)
        factor_inputs = eaggl.FactorInputs(anchor_gene_mask=None, anchor_pheno_mask=None)
        options = _options(keep_original_loadings=True, anchor_gene_set=True, pheno_capture_input="binary_thresholded")
        cfg = eaggl._build_factor_execution_config(options, workflow, factor_inputs)
        self.assertTrue(cfg.keep_original_loadings)
        self.assertEqual(cfg.gene_or_pheno_filter_type, "gene_set_phewas_betas_uncorrected")
        self.assertEqual(cfg.pheno_capture_input, "binary_thresholded")

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
            factor_metrics_out="factor_metrics.tsv",
            factors_anchor_out="factors_anchor.tsv",
            gene_set_clusters_out="gs_cluster.tsv",
            gene_clusters_out="g_cluster.tsv",
            pheno_clusters_out="p_cluster.tsv",
            gene_set_anchor_clusters_out="gs_anchor_cluster.tsv",
            gene_anchor_clusters_out="g_anchor_cluster.tsv",
            pheno_anchor_clusters_out="p_anchor_cluster.tsv",
            gene_pheno_stats_out="gene_pheno.tsv",
            consensus_stats_out="consensus.tsv",
            max_no_write_gene_pheno=0.2,
        )
        eaggl._write_main_factor_outputs(runtime, options)
        self.assertEqual(len(runtime.calls), 7)
        self.assertEqual(runtime.calls[0], ("write_matrix_factors", "factors.tsv", False))
        self.assertEqual(runtime.calls[1], ("write_factor_metrics", "factor_metrics.tsv"))
        self.assertEqual(runtime.calls[2], ("write_matrix_factors", "factors_anchor.tsv", True))
        self.assertEqual(runtime.calls[3], ("write_consensus_factor_diagnostics", "consensus.tsv"))
        self.assertEqual(
            runtime.calls[4],
            ("write_clusters", "gs_cluster.tsv", "g_cluster.tsv", "p_cluster.tsv", False),
        )
        self.assertEqual(
            runtime.calls[5],
            ("write_clusters", "gs_anchor_cluster.tsv", "g_anchor_cluster.tsv", "p_anchor_cluster.tsv", True),
        )
        self.assertEqual(runtime.calls[6], ("write_gene_pheno_statistics", "gene_pheno.tsv", 0.2))

    def test_write_clusters_gene_set_writer_uses_betas_uncorrected_without_stale_alias(self) -> None:
        runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
        runtime.exp_lambdak = [1.0, 1.0]
        runtime.exp_gene_set_factors = np.array(
            [
                [0.2, 0.8],
                [0.9, 0.1],
            ]
        )
        runtime.exp_gene_factors = None
        runtime.exp_pheno_factors = None
        runtime.betas = None
        runtime.betas_uncorrected = np.array([0.1, 0.9])
        runtime.gene_set_factor_gene_set_mask = None
        runtime.factor_labels = ["label1", "label2"]
        runtime.gene_sets = ["gs1", "gs2"]
        runtime.anchor_pheno_mask = None
        runtime.anchor_gene_mask = None

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "gene_set_clusters.tsv"
            runtime.write_clusters(gene_set_clusters_output_file=str(out_path))
            content = out_path.read_text()

        self.assertIn("beta_uncorrected", content)
        lines = content.strip().splitlines()
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("gs2\t0.9"))

    def test_workflow_required_inputs_contract_for_f1_to_f9(self) -> None:
        cases = [
            ("F1", _options(), []),
            ("F2", _options(gene_list=["INS"]), []),
            ("F3", _options(gene_phewas_bfs_in="gene_phewas.tsv"), []),
            ("F4", _options(anchor_phenos=["T2D"], gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F5", _options(anchor_any_pheno=True, gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F6", _options(anchor_genes=["INS"], gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F7", _options(anchor_genes=["INS", "GCK"], gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F8", _options(anchor_any_gene=True, gene_set_phewas_stats_in="gs.tsv", gene_phewas_bfs_in="g.tsv"), []),
            ("F9", _options(anchor_gene_set=True, run_phewas=True, run_phewas_input="g.tsv", gene_phewas_bfs_in="g.tsv"), []),
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

    def test_workflow_classifies_positive_control_aliases_as_f2(self) -> None:
        workflow = eaggl._classify_factor_workflow(_options(positive_controls_list=["INS"]))
        self.assertEqual(workflow["id"], "F2")

    def test_run_main_factor_phewas_stage_invokes_eaggl_phewas_runner(self) -> None:
        runtime = _FactorPhewasRuntimeStub()
        options = _options(
            run_factor_phewas=True,
            run_factor_phewas_input="factor_phewas.tsv",
            factor_phewas_stats_out="factor_phewas_stats.tsv",
            gene_phewas_bfs_in="loaded_gene_phewas.tsv",
            run_phewas=True,
            run_phewas_input="other_gene_phewas.tsv",
            gene_phewas_bfs_id_col="Gene",
            gene_phewas_bfs_pheno_col="Trait",
            gene_phewas_bfs_log_bf_col="Direct",
            gene_phewas_bfs_combined_col="Combined",
            gene_phewas_bfs_prior_col="Prior",
            max_num_burn_in=20,
            max_num_iter_betas=25,
            min_num_iter_betas=5,
            num_chains_betas=3,
            r_threshold_burn_in_betas=1.02,
            use_max_r_for_convergence_betas=True,
            max_frac_sem_betas=0.1,
            gauss_seidel_betas=False,
            sparse_solution=False,
            sparse_frac_betas=0.01,
            factor_phewas_mode="marginal_anchor_adjusted_binary",
            factor_phewas_modes=["marginal_anchor_adjusted_binary", "joint_anchor_adjusted_binary"],
            factor_phewas_anchor_covariate="direct",
            factor_phewas_thresholded_combined_cutoff=1.0,
            factor_phewas_se="robust",
            factor_phewas_min_gene_factor_weight=0.01,
        )
        domain = eaggl.build_main_domain()
        with mock.patch.object(eaggl.eaggl_factor.eaggl_phewas, "run_phewas") as mocked_run:
            result = eaggl.eaggl_factor.run_main_factor_phewas_stage(domain, runtime, options)
        self.assertTrue(result.ran)
        self.assertEqual(result.output_path, "factor_phewas_stats.tsv")
        mocked_run.assert_called_once()
        args, kwargs = mocked_run.call_args
        self.assertIs(args[0], runtime)
        self.assertEqual(kwargs["gene_phewas_bfs_in"], "factor_phewas.tsv")
        self.assertTrue(kwargs["run_for_factors"])
        self.assertEqual(kwargs["min_gene_factor_weight"], 0.0)
        self.assertEqual(kwargs["options"], options)
        self.assertEqual(runtime.output_path, "factor_phewas_stats.tsv")
        self.assertEqual(runtime.recorded_params[0]["factor_phewas_mode"], "marginal_anchor_adjusted_binary")
        self.assertEqual(
            runtime.recorded_params[0]["factor_phewas_modes"],
            "marginal_anchor_adjusted_binary,joint_anchor_adjusted_binary",
        )
        self.assertTrue(runtime.recorded_params[1])


if __name__ == "__main__":
    unittest.main()
