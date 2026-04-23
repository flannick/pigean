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
        learn_phi_target_gene_effective_support=None,
        learn_phi_size_tolerance_frac=0.25,
        learn_phi_min_primary_factors=3,
        learn_phi_max_primary_gene_max_weight_q90=None,
        learn_phi_max_steps=8,
        learn_phi_expand_factor=2.0,
        learn_phi_weight_floor=None,
        learn_phi_metric_factor_scope="primary",
        learn_phi_report_out=None,
        factor_phi_metrics_out=None,
        factor_phi_factors_out=None,
        factor_phi_gene_set_clusters_out=None,
        factor_phi_gene_clusters_out=None,
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
        max_num_discovery_gene_sets=None,
        no_auto_discovery_subset=False,
        discovery_redundancy_weighting_mode="effective_size",
        no_discovery_redundancy_weighting=False,
        discovery_similarity_threshold=0.35,
        anchor_any_pheno=False,
        anchor_any_gene=False,
        anchor_gene_set=False,
        factor_phewas_full_output=False,
        factor_gene_clusters_in=None,
        factor_gene_set_clusters_in=None,
        min_gene_set_read_beta=1e-20,
        min_gene_set_read_beta_uncorrected=1e-20,
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
        gene_set_phewas_stats_id_col="Gene_Set",
        gene_set_phewas_stats_pheno_col=None,
        gene_set_phewas_stats_beta_col=None,
        gene_set_phewas_stats_beta_uncorrected_col=None,
        gene_phewas_bfs_in=None,
        gene_phewas_bfs_id_col=None,
        gene_phewas_bfs_pheno_col=None,
        gene_phewas_bfs_log_bf_col=None,
        gene_phewas_bfs_combined_col=None,
        gene_phewas_bfs_prior_col=None,
        gene_phewas_id_to_X_id=None,
        min_gene_phewas_read_value=0.0,
        max_read_entries_at_once=None,
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
        trait_linkage_source="combined",
        trait_linkage_threshold=1.0,
        trait_linkage_computation_mode="sparse_full",
        no_trait_linkage=False,
        factors_out=None,
        factor_metrics_out=None,
        factors_anchor_out=None,
        gene_set_clusters_out=None,
        gene_clusters_out=None,
        cluster_row_min_max_loading=0.01,
        factor_output_scope="primary",
        trait_factor_links_out=None,
        pheno_clusters_out=None,
        gene_set_anchor_clusters_out=None,
        gene_anchor_clusters_out=None,
        pheno_anchor_clusters_out=None,
        gene_pheno_stats_out=None,
        max_no_write_gene_pheno=0.0,
        clustering_params_out=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _RuntimeStub:
    def __init__(self) -> None:
        self.run_factor_kwargs = None
        self.calls = []

    def run_factor(self, **kwargs):
        self.run_factor_kwargs = kwargs

    def write_matrix_factors(self, out, write_anchor_specific=False, factor_output_scope="primary"):
        self.calls.append(("write_matrix_factors", out, write_anchor_specific, factor_output_scope))

    def write_factor_metrics(self, out):
        self.calls.append(("write_factor_metrics", out))

    def write_consensus_factor_diagnostics(self, out):
        self.calls.append(("write_consensus_factor_diagnostics", out))

    def write_clusters(
        self,
        gene_set_out,
        gene_out,
        pheno_out,
        write_anchor_specific=False,
        cluster_row_min_max_loading=0.01,
        factor_output_scope="primary",
    ):
        self.calls.append(
            (
                "write_clusters",
                gene_set_out,
                gene_out,
                pheno_out,
                write_anchor_specific,
                cluster_row_min_max_loading,
                factor_output_scope,
            )
        )

    def write_trait_factor_links(self, out):
        self.calls.append(("write_trait_factor_links", out))

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


class _ProjectionOnlyRuntimeStub:
    def __init__(self) -> None:
        self.recorded_params = []

    def _record_params(self, params, overwrite=False):
        self.recorded_params.append((dict(params), overwrite))

    def num_factors(self):
        return 0 if getattr(self, "exp_lambdak", None) is None else len(self.exp_lambdak)


class FactorStageHelpersTest(unittest.TestCase):
    def test_eaggl_state_run_factor_filters_legacy_wrapper_kwargs(self) -> None:
        runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
        captured = {}

        def _stub_run_factor(
            state,
            *,
            factor_backend="full",
            blockwise_gene_set_block_size=5000,
            blockwise_epochs=3,
            blockwise_shuffle_blocks=True,
            blockwise_warm_start=True,
            blockwise_max_blocks=None,
            blockwise_report_out=None,
            trait_linkage_source="combined",
            trait_linkage_threshold=1.0,
            trait_linkage_computation_mode="sparse_full",
            no_trait_linkage=False,
            bail_fn=None,
            warn_fn=None,
            log_fn=None,
            info_level=None,
            debug_level=None,
            trace_level=None,
            labeling_module=None,
        ):
            captured.update(
                {
                    "state": state,
                    "factor_backend": factor_backend,
                    "blockwise_gene_set_block_size": blockwise_gene_set_block_size,
                    "blockwise_epochs": blockwise_epochs,
                    "blockwise_shuffle_blocks": blockwise_shuffle_blocks,
                    "blockwise_warm_start": blockwise_warm_start,
                    "blockwise_max_blocks": blockwise_max_blocks,
                    "blockwise_report_out": blockwise_report_out,
                    "trait_linkage_source": trait_linkage_source,
                    "trait_linkage_threshold": trait_linkage_threshold,
                    "trait_linkage_computation_mode": trait_linkage_computation_mode,
                    "no_trait_linkage": no_trait_linkage,
                }
            )
            return "stubbed"

        with mock.patch.object(eaggl.eaggl_state._eaggl_factor_runtime, "run_factor", new=_stub_run_factor):
            result = runtime.run_factor(
                factor_backend="blockwise_global_w",
                max_num_gene_sets=1234,
                gene_set_budget_mode="pruned",
                learn_phi_gene_set_budget_mode="sentinel_pruned",
                online_block_size=777,
                online_epochs=6,
                online_shuffle_blocks=False,
                online_warm_start=False,
                online_max_blocks=4,
                online_report_out="legacy_report.tsv.gz",
                trait_linkage_source="combined",
                trait_linkage_threshold=0.75,
                trait_linkage_computation_mode="sparse_full",
                no_trait_linkage=True,
            )

        self.assertEqual(result, "stubbed")
        self.assertIs(captured["state"], runtime)
        self.assertEqual(captured["factor_backend"], "blockwise_global_w")
        self.assertEqual(captured["blockwise_gene_set_block_size"], 777)
        self.assertEqual(captured["blockwise_epochs"], 6)
        self.assertFalse(captured["blockwise_shuffle_blocks"])
        self.assertFalse(captured["blockwise_warm_start"])
        self.assertEqual(captured["blockwise_max_blocks"], 4)
        self.assertEqual(captured["blockwise_report_out"], "legacy_report.tsv.gz")
        self.assertEqual(captured["trait_linkage_source"], "combined")
        self.assertEqual(captured["trait_linkage_threshold"], 0.75)
        self.assertEqual(captured["trait_linkage_computation_mode"], "sparse_full")
        self.assertTrue(captured["no_trait_linkage"])

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
            track_filtered_beta_uncorrected=False,
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
                track_filtered_beta_uncorrected=False,
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
            learn_phi_target_gene_effective_support=25.0,
            learn_phi_size_tolerance_frac=0.2,
            learn_phi_min_primary_factors=4,
            learn_phi_max_primary_gene_max_weight_q90=0.6,
            learn_phi_max_steps=6,
            learn_phi_expand_factor=5.0,
            learn_phi_weight_floor=0.02,
            learn_phi_metric_factor_scope="all",
            learn_phi_report_out="phi.tsv",
            factor_phi_metrics_out="phi_factor_metrics.tsv",
            factor_phi_factors_out="phi_factors.tsv",
            factor_phi_gene_set_clusters_out="phi_gene_set_clusters.tsv",
            factor_phi_gene_clusters_out="phi_gene_clusters.tsv",
            cluster_row_min_max_loading=0.02,
            factor_output_scope="all",
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
        self.assertEqual(cfg.learn_phi_target_gene_effective_support, 25.0)
        self.assertEqual(cfg.learn_phi_size_tolerance_frac, 0.2)
        self.assertEqual(cfg.learn_phi_min_primary_factors, 4)
        self.assertEqual(cfg.learn_phi_max_primary_gene_max_weight_q90, 0.6)
        self.assertEqual(cfg.learn_phi_max_steps, 6)
        self.assertEqual(cfg.learn_phi_expand_factor, 5.0)
        self.assertEqual(cfg.learn_phi_weight_floor, 0.02)
        self.assertEqual(cfg.learn_phi_metric_factor_scope, "all")
        self.assertEqual(cfg.learn_phi_report_out, "phi.tsv")
        self.assertEqual(cfg.factor_phi_metrics_out, "phi_factor_metrics.tsv")
        self.assertEqual(cfg.factor_phi_factors_out, "phi_factors.tsv")
        self.assertEqual(cfg.factor_phi_gene_set_clusters_out, "phi_gene_set_clusters.tsv")
        self.assertEqual(cfg.factor_phi_gene_clusters_out, "phi_gene_clusters.tsv")
        self.assertEqual(cfg.cluster_row_min_max_loading, 0.02)
        self.assertEqual(cfg.factor_output_scope, "all")
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

    def test_build_factor_execution_config_defaults_to_effective_size_weighting(self) -> None:
        workflow = eaggl.FactorWorkflow(workflow_id="F1", factor_gene_set_x_pheno=False)
        factor_inputs = eaggl.FactorInputs(anchor_gene_mask=None, anchor_pheno_mask=None)
        cfg = eaggl._build_factor_execution_config(_options(), workflow, factor_inputs)
        self.assertEqual(cfg.discovery_redundancy_weighting_mode, "effective_size")
        self.assertEqual(cfg.discovery_similarity_threshold, 0.35)
        self.assertTrue(cfg.discovery_redundancy_weighting)
        self.assertEqual(cfg.factor_output_scope, "primary")


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
            trait_factor_links_out="trait_links.tsv",
            pheno_clusters_out="p_cluster.tsv",
            gene_set_anchor_clusters_out="gs_anchor_cluster.tsv",
            gene_anchor_clusters_out="g_anchor_cluster.tsv",
            pheno_anchor_clusters_out="p_anchor_cluster.tsv",
            gene_pheno_stats_out="gene_pheno.tsv",
            consensus_stats_out="consensus.tsv",
            max_no_write_gene_pheno=0.2,
            cluster_row_min_max_loading=0.02,
            factor_output_scope="primary_secondary",
        )
        eaggl._write_main_factor_outputs(runtime, options)
        self.assertEqual(len(runtime.calls), 9)
        self.assertEqual(runtime.calls[0], ("write_matrix_factors", "factors.tsv", False, "primary_secondary"))
        self.assertEqual(runtime.calls[1], ("write_factor_metrics", "factor_metrics.tsv"))
        self.assertEqual(runtime.calls[2], ("write_matrix_factors", "factors_anchor.tsv", True, "primary_secondary"))
        self.assertEqual(runtime.calls[3], ("write_consensus_factor_diagnostics", "consensus.tsv"))
        self.assertEqual(
            runtime.calls[4],
            ("write_clusters", "gs_cluster.tsv", "g_cluster.tsv", None, False, 0.02, "primary_secondary"),
        )
        self.assertEqual(runtime.calls[5], ("write_trait_factor_links", "trait_links.tsv"))
        self.assertEqual(runtime.calls[6], ("write_trait_factor_links", "p_cluster.tsv"))
        self.assertEqual(
            runtime.calls[7],
            (
                "write_clusters",
                "gs_anchor_cluster.tsv",
                "g_anchor_cluster.tsv",
                "p_anchor_cluster.tsv",
                True,
                0.02,
                "primary_secondary",
            ),
        )
        self.assertEqual(runtime.calls[8], ("write_gene_pheno_statistics", "gene_pheno.tsv", 0.2))

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
        runtime.gene_set_in_discovery_mask = None
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

    def test_write_clusters_skips_gene_and_gene_set_rows_below_loading_threshold(self) -> None:
        runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
        runtime.exp_lambdak = [1.0, 1.0]
        runtime.exp_gene_set_factors = np.array(
            [
                [0.001, 0.002],
                [0.2, 0.001],
            ],
            dtype=float,
        )
        runtime.exp_gene_factors = np.array(
            [
                [0.003, 0.004],
                [0.001, 0.3],
            ],
            dtype=float,
        )
        runtime.exp_pheno_factors = None
        runtime.betas = None
        runtime.betas_uncorrected = np.array([0.1, 0.2])
        runtime.combined_prior_Ys = np.array([1.0, 2.0])
        runtime.gene_set_in_discovery_mask = None
        runtime.gene_in_discovery_mask = None
        runtime.factor_labels = ["label1", "label2"]
        runtime.gene_sets = ["low_gs", "high_gs"]
        runtime.genes = ["low_gene", "high_gene"]
        runtime.anchor_pheno_mask = None
        runtime.anchor_gene_mask = None

        with tempfile.TemporaryDirectory() as tmpdir:
            gene_set_path = Path(tmpdir) / "gene_set_clusters.tsv"
            gene_path = Path(tmpdir) / "gene_clusters.tsv"
            runtime.write_clusters(
                gene_set_clusters_output_file=str(gene_set_path),
                gene_clusters_output_file=str(gene_path),
                cluster_row_min_max_loading=0.01,
            )
            gene_set_content = gene_set_path.read_text()
            gene_content = gene_path.read_text()

            unfiltered_gene_set_path = Path(tmpdir) / "gene_set_clusters_unfiltered.tsv"
            unfiltered_gene_path = Path(tmpdir) / "gene_clusters_unfiltered.tsv"
            runtime.write_clusters(
                gene_set_clusters_output_file=str(unfiltered_gene_set_path),
                gene_clusters_output_file=str(unfiltered_gene_path),
                cluster_row_min_max_loading=0.0,
            )
            unfiltered_gene_set_content = unfiltered_gene_set_path.read_text()
            unfiltered_gene_content = unfiltered_gene_path.read_text()

        self.assertNotIn("low_gs\t", gene_set_content)
        self.assertIn("high_gs\t", gene_set_content)
        self.assertNotIn("low_gene\t", gene_content)
        self.assertIn("high_gene\t", gene_content)
        self.assertIn("low_gs\t", unfiltered_gene_set_content)
        self.assertIn("low_gene\t", unfiltered_gene_content)

    def test_write_clustering_params_writes_json_and_tsv_siblings(self) -> None:
        runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
        payload = {
            "workflow_id": "F1",
            "routing_family": "default_stats",
            "trait_linkage": {"enabled": True, "basis": "gene", "source": "combined"},
            "inputs": {"gene_stats_in": "gene_stats.tsv.gz"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "clustering_params"
            written = runtime.write_clustering_params(str(base_path), payload)

            import gzip

            with gzip.open(written["json"], "rt", encoding="utf-8") as fh:
                json_content = fh.read()
            with gzip.open(written["tsv"], "rt", encoding="utf-8") as fh:
                tsv_content = fh.read()

        self.assertEqual(written["json"], str(base_path) + ".json.gz")
        self.assertEqual(written["tsv"], str(base_path) + ".tsv.gz")
        self.assertIn('"workflow_id": "F1"', json_content)
        self.assertIn("Field\tValue", tsv_content)
        self.assertIn("trait_linkage.source\tcombined", tsv_content)

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

    def test_load_existing_factor_phewas_gene_clusters_sets_gene_factor_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_clusters = Path(tmpdir) / "gene_clusters.out"
            gene_clusters.write_text(
                "\t".join(
                    [
                        "Gene",
                        "combined",
                        "log_bf",
                        "prior",
                        "in_discovery",
                        "cluster",
                        "label",
                        "Factor1",
                        "Factor2",
                        "Relative_Factor1",
                        "Relative_Factor2",
                        "Combined_Factor1",
                        "Combined_Factor2",
                    ]
                )
                + "\n"
                + "GENE1\t1.5\t0.7\t0.1\tTrue\tFactor1\timmune\t0.8\t0.2\t1\t0.25\t0.7\t0.1\n"
                + "GENE2\t0.5\t0.2\t0.05\tTrue\tFactor2\tmetabolic\t0.1\t0.9\t0.125\t1\t0.05\t0.8\n",
                encoding="utf-8",
            )

            runtime = _ProjectionOnlyRuntimeStub()
            domain = eaggl.build_main_domain()
            result = eaggl.eaggl_factor.load_existing_factor_phewas_gene_clusters(
                domain,
                runtime,
                str(gene_clusters),
            )

        self.assertEqual(result["num_genes"], 2)
        self.assertEqual(result["num_factors"], 2)
        self.assertEqual(runtime.genes, ["GENE1", "GENE2"])
        self.assertEqual(runtime.gene_to_ind["GENE2"], 1)
        np.testing.assert_allclose(runtime.exp_gene_factors, [[0.8, 0.2], [0.1, 0.9]])
        np.testing.assert_allclose(runtime.Y, [0.7, 0.2])
        np.testing.assert_allclose(runtime.combined_prior_Ys, [1.5, 0.5])
        np.testing.assert_allclose(runtime.priors, [0.1, 0.05])
        self.assertEqual(runtime.factor_labels, ["immune", "metabolic"])
        self.assertEqual(runtime.num_factors(), 2)

    def test_load_existing_factor_gene_clusters_sets_gene_factor_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_clusters = Path(tmpdir) / "gene_clusters.out"
            gene_clusters.write_text(
                "\t".join(["Gene", "in_discovery", "cluster", "label", "Factor1", "Factor2"])
                + "\n"
                + "GENE1\tTrue\tFactor1\timmune\t0.8\t0.2\n"
                + "GENE2\tFalse\tFactor2\tmetabolic\t0.1\t0.9\n",
                encoding="utf-8",
            )

            runtime = _ProjectionOnlyRuntimeStub()
            domain = eaggl.build_main_domain()
            result = eaggl.eaggl_factor.load_existing_factor_gene_clusters(
                domain,
                runtime,
                str(gene_clusters),
            )

        self.assertEqual(result["num_genes"], 2)
        self.assertEqual(result["num_factors"], 2)
        np.testing.assert_allclose(runtime.exp_gene_factors, [[0.8, 0.2], [0.1, 0.9]])
        np.testing.assert_array_equal(runtime.gene_in_discovery_mask, [True, False])

    def test_projection_only_pheno_cluster_stage_projects_from_gene_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gene_clusters = tmpdir_path / "gene_clusters.out"
            gene_clusters.write_text(
                "\t".join(["Gene", "in_discovery", "cluster", "label", "Factor1", "Factor2"])
                + "\n"
                + "GENE1\tTrue\tFactor1\timmune\t1.0\t0.0\n"
                + "GENE2\tTrue\tFactor2\tmetabolic\t0.0\t1.0\n"
                + "GENE3\tFalse\tFactor1\timmune\t0.1\t0.1\n",
                encoding="utf-8",
            )
            gene_phewas = tmpdir_path / "gene_phewas.tsv"
            gene_phewas.write_text(
                "Gene\tPheno\tcombined\tlog_bf\tprior\n"
                "GENE1\tTraitA\t2.0\t1.0\t0.1\n"
                "GENE2\tTraitA\t0.0\t0.0\t0.0\n"
                "GENE3\tTraitA\t8.0\t4.0\t0.5\n"
                "GENE1\tTraitB\t0.0\t0.0\t0.0\n"
                "GENE2\tTraitB\t3.0\t1.5\t0.2\n"
                "GENE3\tTraitB\t0.0\t0.0\t0.0\n",
                encoding="utf-8",
            )
            pheno_clusters = tmpdir_path / "pheno_clusters.out.gz"

            runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
            domain = eaggl.build_main_domain()
            eaggl.eaggl_factor.load_existing_factor_gene_clusters(domain, runtime, str(gene_clusters))
            options = _options(
                factor_gene_clusters_in=str(gene_clusters),
                gene_phewas_bfs_in=str(gene_phewas),
                pheno_clusters_out=str(pheno_clusters),
                pheno_capture_input="weighted_thresholded",
            )
            result = eaggl.eaggl_factor.run_main_pheno_projection_stage(domain, runtime, options)
            runtime.write_trait_factor_links(str(pheno_clusters))

            import gzip

            with gzip.open(pheno_clusters, "rt", encoding="utf-8") as fh:
                content = fh.read()

        self.assertTrue(result.ran)
        self.assertEqual(result.output_path, str(pheno_clusters))
        self.assertEqual(runtime.phenos, ["TraitA", "TraitB"])
        self.assertEqual(runtime.trait_linkage_joint.shape, (2, 2))
        self.assertGreater(runtime.trait_linkage_joint[0, 0], runtime.trait_linkage_joint[0, 1])
        self.assertGreater(runtime.trait_linkage_joint[1, 1], runtime.trait_linkage_joint[1, 0])
        np.testing.assert_allclose(runtime.trait_linkage_factor_total_mass, [1.0, 1.0], atol=1e-8)
        np.testing.assert_allclose(runtime.trait_linkage_strength, [10.0, 3.0], atol=1e-8)
        np.testing.assert_allclose(runtime.trait_linkage_retained_strength, [2.0, 3.0], atol=1e-8)
        np.testing.assert_allclose(runtime.trait_linkage_retained_fraction, [0.2, 1.0], atol=1e-8)
        np.testing.assert_allclose(runtime.trait_linkage_n_eff, [100.0 / 68.0, 1.0], atol=1e-8)
        np.testing.assert_allclose(runtime.trait_linkage_retained_n_eff, [1.0, 1.0], atol=1e-8)
        np.testing.assert_array_equal(runtime.trait_linkage_total_feature_count, [2, 1])
        np.testing.assert_array_equal(runtime.trait_linkage_retained_feature_count, [1, 1])
        np.testing.assert_array_equal(runtime.trait_linkage_low_retention_flag, [True, True])
        self.assertLess(runtime.trait_linkage_joint[0, 0], 0.25)
        self.assertIn("trait\tfactor\tis_anchor\tjoint_coefficient\tmarginal_coefficient", content)
        self.assertIn("trait_total_support", content)
        self.assertIn("retained_trait_support", content)
        self.assertIn("retained_fraction", content)
        self.assertIn("total_feature_count", content)
        self.assertIn("retained_feature_count", content)
        self.assertIn("trait_n_eff", content)
        self.assertIn("retained_n_eff", content)
        self.assertIn("low_retention_flag", content)
        self.assertIn("joint_coefficient_support_mass", content)
        self.assertIn("marginal_coefficient_support_mass", content)
        self.assertIn("TraitA\tFactor1", content)
        self.assertIn("TraitB\tFactor2", content)

    def test_projection_only_anchor_and_external_trait_linkage_share_normalization_logic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gene_clusters = tmpdir_path / "gene_clusters.out"
            gene_clusters.write_text(
                "\t".join(["Gene", "used_to_factor", "cluster", "label", "Factor1", "Factor2"])
                + "\n"
                + "GENE1\tTrue\tFactor1\timmune\t1.0\t0.0\n"
                + "GENE2\tTrue\tFactor2\tmetabolic\t0.0\t1.0\n"
                + "GENE3\tFalse\tFactor1\timmune\t0.1\t0.1\n",
                encoding="utf-8",
            )
            gene_phewas = tmpdir_path / "gene_phewas.tsv"
            gene_phewas.write_text(
                "Gene\tPheno\tcombined\tlog_bf\tprior\n"
                "GENE1\tTraitA\t2.0\t1.0\t0.1\n"
                "GENE2\tTraitA\t0.0\t0.0\t0.0\n"
                "GENE3\tTraitA\t8.0\t4.0\t0.5\n"
                "GENE1\tTraitB\t0.0\t0.0\t0.0\n"
                "GENE2\tTraitB\t3.0\t1.5\t0.2\n"
                "GENE3\tTraitB\t0.0\t0.0\t0.0\n",
                encoding="utf-8",
            )

            runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
            domain = eaggl.build_main_domain()
            eaggl.eaggl_factor.load_existing_factor_gene_clusters(domain, runtime, str(gene_clusters))
            options = _options(
                factor_gene_clusters_in=str(gene_clusters),
                gene_phewas_bfs_in=str(gene_phewas),
                anchor_phenos=["TraitA"],
                pheno_capture_input="weighted_thresholded",
            )
            result = eaggl.eaggl_factor.run_main_pheno_projection_stage(domain, runtime, options)

        self.assertTrue(result.ran)
        np.testing.assert_array_equal(runtime.trait_linkage_is_anchor, [True, False])
        np.testing.assert_allclose(runtime.factor_anchor_relevance[:, 0], runtime.trait_linkage_joint[0, :], atol=1e-8)
        np.testing.assert_allclose(runtime.factor_anchor_marginal_relevance[:, 0], runtime.trait_linkage_marginal[0, :], atol=1e-8)
        np.testing.assert_allclose(runtime.factor_relevance, runtime.trait_linkage_joint[0, :], atol=1e-8)
        np.testing.assert_allclose(runtime.factor_marginal_relevance, runtime.trait_linkage_marginal[0, :], atol=1e-8)

    def test_load_existing_factor_gene_set_clusters_sets_gene_set_factor_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_set_clusters = Path(tmpdir) / "gene_set_clusters.out"
            gene_set_clusters.write_text(
                "\t".join(["Gene_Set", "beta_uncorrected", "in_discovery", "discovery_family_id", "discovery_representative", "discovery_family_size", "discovery_weight", "discovery_family_mean_similarity", "discovery_family_effective_size", "cluster", "label", "Factor1", "Factor2"])
                + "\n"
                + "GS1\t0.8\tTrue\t0\tTrue\t2\t0.6\t1.0\t1.0\tFactor1\timmune\t0.9\t0.1\n"
                + "GS2\t0.4\tFalse\t0\tFalse\t2\t0.6\t1.0\t1.0\tFactor2\tmetabolic\t0.2\t0.8\n",
                encoding="utf-8",
            )

            runtime = _ProjectionOnlyRuntimeStub()
            domain = eaggl.build_main_domain()
            result = eaggl.eaggl_factor.load_existing_factor_gene_set_clusters(
                domain,
                runtime,
                str(gene_set_clusters),
            )

        self.assertEqual(result["num_gene_sets"], 2)
        self.assertEqual(result["num_factors"], 2)
        self.assertEqual(runtime.gene_sets, ["GS1", "GS2"])
        self.assertEqual(runtime.gene_set_to_ind["GS2"], 1)
        np.testing.assert_allclose(runtime.exp_gene_set_factors, [[0.9, 0.1], [0.2, 0.8]])
        np.testing.assert_array_equal(runtime.gene_set_in_discovery_mask, [True, False])
        np.testing.assert_allclose(runtime.betas_uncorrected, [0.8, 0.4])
        np.testing.assert_allclose(runtime.gene_set_discovery_family_mean_similarity, [1.0, 1.0])
        np.testing.assert_allclose(runtime.gene_set_discovery_family_effective_size, [1.0, 1.0])
        self.assertEqual(runtime.factor_labels, ["immune", "metabolic"])

    def test_projection_only_pheno_cluster_stage_projects_from_gene_set_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gene_set_clusters = tmpdir_path / "gene_set_clusters.out"
            gene_set_clusters.write_text(
                "\t".join(["Gene_Set", "in_discovery", "discovery_family_id", "discovery_representative", "discovery_family_size", "discovery_weight", "discovery_family_mean_similarity", "discovery_family_effective_size", "cluster", "label", "Factor1", "Factor2"])
                + "\n"
                + "GS1\tTrue\t0\tTrue\t1\t1.0\tNA\t1.0\tFactor1\timmune\t1.0\t0.0\n"
                + "GS2\tTrue\t1\tTrue\t1\t1.0\tNA\t1.0\tFactor2\tmetabolic\t0.0\t1.0\n",
                encoding="utf-8",
            )
            gene_set_phewas = tmpdir_path / "gene_set_phewas.tsv"
            gene_set_phewas.write_text(
                "Gene_Set\tPheno\tbeta_uncorrected\tbeta\n"
                "GS1\tTraitA\t2.0\t1.0\n"
                "GS2\tTraitA\t0.0\t0.0\n"
                "GS1\tTraitB\t0.0\t0.0\n"
                "GS2\tTraitB\t3.0\t1.5\n",
                encoding="utf-8",
            )
            pheno_clusters = tmpdir_path / "pheno_clusters.out.gz"

            runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
            domain = eaggl.build_main_domain()
            eaggl.eaggl_factor.load_existing_factor_gene_set_clusters(domain, runtime, str(gene_set_clusters))
            options = _options(
                factor_gene_set_clusters_in=str(gene_set_clusters),
                gene_set_phewas_stats_in=str(gene_set_phewas),
                project_phenos_from_gene_sets=True,
                pheno_clusters_out=str(pheno_clusters),
                pheno_capture_input="weighted_thresholded",
            )
            result = eaggl.eaggl_factor.run_main_pheno_projection_stage(domain, runtime, options)
            runtime.write_trait_factor_links(str(pheno_clusters))

            import gzip

            with gzip.open(pheno_clusters, "rt", encoding="utf-8") as fh:
                content = fh.read()

        self.assertTrue(result.ran)
        self.assertEqual(result.output_path, str(pheno_clusters))
        self.assertEqual(runtime.phenos, ["TraitA", "TraitB"])
        self.assertEqual(runtime.trait_linkage_joint.shape, (2, 2))
        self.assertGreater(runtime.trait_linkage_joint[0, 0], runtime.trait_linkage_joint[0, 1])
        self.assertGreater(runtime.trait_linkage_joint[1, 1], runtime.trait_linkage_joint[1, 0])
        self.assertIn("gene_sets", content)
        self.assertIn("TraitA\tFactor1", content)
        self.assertIn("TraitB\tFactor2", content)

    def test_write_matrix_factors_reports_factor_total_mass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factors.out.gz"
            runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
            runtime.exp_lambdak = np.array([1.0, 2.0])
            runtime.factor_labels = ["immune", "metabolic"]
            runtime.factor_relevance = np.array([0.25, 0.5])
            runtime.trait_linkage_factor_total_mass = np.array([3.0, 7.0])
            runtime.factor_top_gene_sets = [["GS1"], ["GS2"]]
            runtime.factor_top_genes = [["GENE1"], ["GENE2"]]
            runtime.write_matrix_factors(str(output_path), factor_output_scope="all")

            import gzip

            with gzip.open(output_path, "rt", encoding="utf-8") as fh:
                content = fh.read()

        self.assertIn("factor_total_mass", content.splitlines()[0])
        self.assertIn("lambda\tfactor_tier\tcombined_mass_fraction\tany_relevance", content.splitlines()[0])
        self.assertIn("Factor1\timmune\t1\tfiltered\t0\t0.25\t3", content)

    def test_write_factor_outputs_default_to_primary_factor_scope(self) -> None:
        runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
        runtime.exp_lambdak = np.array([1.0, 2.0, 3.0])
        runtime.factor_labels = ["primary_label", "secondary_label", "filtered_label"]
        runtime.factor_relevance = np.array([1.0, 1.0, 1.0])
        runtime.exp_gene_factors = np.array([[10.0, 0.02, 0.001]], dtype=float)
        runtime.exp_gene_set_factors = np.array([[10.0, 0.02, 0.001]], dtype=float)
        runtime.exp_pheno_factors = None
        runtime.factor_top_genes = [["GENE1"], ["GENE2"], ["GENE3"]]
        runtime.factor_top_gene_sets = [["GS1"], ["GS2"], ["GS3"]]
        runtime.genes = ["gene_row"]
        runtime.gene_sets = ["gene_set_row"]
        runtime.combined_prior_Ys = np.array([1.0])
        runtime.betas_uncorrected = np.array([1.0])
        runtime.gene_in_discovery_mask = None
        runtime.gene_set_in_discovery_mask = None
        runtime.anchor_pheno_mask = None
        runtime.anchor_gene_mask = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            factors_primary = tmpdir_path / "factors_primary.tsv.gz"
            factors_primary_secondary = tmpdir_path / "factors_primary_secondary.tsv.gz"
            factors_all = tmpdir_path / "factors_all.tsv.gz"
            gene_set_clusters_primary = tmpdir_path / "gene_set_clusters_primary.tsv.gz"
            gene_clusters_primary = tmpdir_path / "gene_clusters_primary.tsv.gz"
            gene_set_clusters_all = tmpdir_path / "gene_set_clusters_all.tsv.gz"
            gene_clusters_all = tmpdir_path / "gene_clusters_all.tsv.gz"

            runtime.write_matrix_factors(str(factors_primary))
            runtime.write_matrix_factors(str(factors_primary_secondary), factor_output_scope="primary_secondary")
            runtime.write_matrix_factors(str(factors_all), factor_output_scope="all")
            runtime.write_clusters(
                gene_set_clusters_output_file=str(gene_set_clusters_primary),
                gene_clusters_output_file=str(gene_clusters_primary),
            )
            runtime.write_clusters(
                gene_set_clusters_output_file=str(gene_set_clusters_all),
                gene_clusters_output_file=str(gene_clusters_all),
                factor_output_scope="all",
            )

            import gzip

            with gzip.open(factors_primary, "rt", encoding="utf-8") as fh:
                primary_text = fh.read()
            with gzip.open(factors_primary_secondary, "rt", encoding="utf-8") as fh:
                primary_secondary_text = fh.read()
            with gzip.open(factors_all, "rt", encoding="utf-8") as fh:
                all_text = fh.read()
            with gzip.open(gene_set_clusters_primary, "rt", encoding="utf-8") as fh:
                primary_gene_set_clusters = fh.read()
            with gzip.open(gene_clusters_primary, "rt", encoding="utf-8") as fh:
                primary_gene_clusters = fh.read()
            with gzip.open(gene_set_clusters_all, "rt", encoding="utf-8") as fh:
                all_gene_set_clusters = fh.read()
            with gzip.open(gene_clusters_all, "rt", encoding="utf-8") as fh:
                all_gene_clusters = fh.read()

        self.assertIn("Factor1\tprimary_label\t1\tprimary", primary_text)
        self.assertNotIn("Factor2\tsecondary_label", primary_text)
        self.assertNotIn("Factor3\tfiltered_label", primary_text)
        self.assertIn("Factor2\tsecondary_label\t2\tsecondary", primary_secondary_text)
        self.assertNotIn("Factor3\tfiltered_label", primary_secondary_text)
        self.assertIn("Factor3\tfiltered_label\t3\tfiltered", all_text)
        self.assertIn("Factor1", primary_gene_set_clusters.splitlines()[0])
        self.assertNotIn("Factor2", primary_gene_set_clusters.splitlines()[0])
        self.assertIn("Factor1", primary_gene_clusters.splitlines()[0])
        self.assertNotIn("Factor2", primary_gene_clusters.splitlines()[0])
        self.assertIn("Factor3", all_gene_set_clusters.splitlines()[0])
        self.assertIn("Factor3", all_gene_clusters.splitlines()[0])

    def test_projection_only_factor_phewas_stage_gate_does_not_require_factor_fit(self) -> None:
        self.assertTrue(
            eaggl.eaggl_factor.should_run_main_factor_phewas_stage(
                {
                    "run_factor": False,
                    "run_factor_phewas": True,
                    "factor_phewas_projection_only": True,
                }
            )
        )

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
