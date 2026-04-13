from __future__ import annotations

import csv
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pigean import state as pigean_state
from pegs_shared import output_tables as pegs_output_tables


class _StubState(SimpleNamespace):
    def get_gene_N(self, get_missing: bool = False):
        if get_missing:
            return np.array([], dtype=float)
        return self.gene_N


class GibbsSummaryTest(unittest.TestCase):
    def test_final_priors_remain_direct_chain_summary(self) -> None:
        stub_state = _StubState(
            X_orig=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            X_orig_missing_genes=None,
            mean_shifts=np.zeros(2, dtype=float),
            scale_factors=np.ones(2, dtype=float),
            genes_missing=None,
            gene_N=np.array([0.0, 1.0], dtype=float),
            Y_for_regression=np.zeros(2, dtype=float),
            Y=np.zeros(2, dtype=float),
            background_log_bf=0.0,
        )

        sum_betas_m = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        num_sum_beta_m = np.ones_like(sum_betas_m, dtype=float)
        sum_priors_m = np.array(
            [
                [10.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
            ],
            dtype=float,
        )
        num_sum_y_m = np.ones_like(sum_priors_m, dtype=float)
        zeros_gene = np.zeros_like(sum_priors_m, dtype=float)
        zeros_beta = np.zeros_like(sum_betas_m, dtype=float)

        final_summary = pigean_state._summarize_gibbs_chain_aggregates(
            stub_state,
            sum_Ys_m=zeros_gene,
            sum_Y_raws_m=zeros_gene,
            sum_log_pos_m=zeros_gene,
            sum_log_po_raws_m=zeros_gene,
            sum_log_po_raws2_m=zeros_gene,
            sum_priors_m=sum_priors_m,
            sum_priors2_m=np.square(sum_priors_m),
            sum_Ds_m=zeros_gene,
            sum_D_raws_m=zeros_gene,
            sum_bf_orig_m=zeros_gene,
            sum_bf_orig_raw_m=zeros_gene,
            sum_bf_orig_raw2_m=zeros_gene,
            sum_betas_m=sum_betas_m,
            sum_betas2_m=np.square(sum_betas_m),
            sum_betas_uncorrected_m=sum_betas_m,
            sum_betas_uncorrected2_m=np.square(sum_betas_m),
            sum_postp_m=zeros_beta,
            sum_beta_tildes_m=zeros_beta,
            sum_z_scores_m=zeros_beta,
            num_sum_Y_m=num_sum_y_m,
            num_sum_beta_m=num_sum_beta_m,
            num_chains_effective=sum_betas_m.shape[0],
            num_mad=100,
            adjust_priors=False,
        )

        self.assertFalse(np.allclose(final_summary["avg_priors_v"], final_summary["final_avg_priors_v"]))
        np.testing.assert_allclose(final_summary["final_avg_priors_v"], final_summary["avg_betas_v"])
        self.assertGreater(final_summary["prior_beta_summary_rel_diff_q90"], 0.0)

        pigean_state._apply_gibbs_final_state(stub_state, final_summary, adjust_priors=False)

        np.testing.assert_allclose(stub_state.priors, final_summary["avg_priors_v"])
        np.testing.assert_allclose(stub_state.priors_missing, final_summary["avg_priors_missing_v"])
        np.testing.assert_allclose(stub_state.betas, final_summary["avg_betas_v"])
        np.testing.assert_allclose(
            stub_state.combined_prior_Ys_for_regression,
            final_summary["avg_log_pos_v"] - stub_state.background_log_bf,
        )
        np.testing.assert_allclose(
            stub_state.combined_prior_Ys,
            final_summary["avg_log_po_raws_v"] - stub_state.background_log_bf,
        )

    def test_post_burn_precision_streak_requires_rhat_and_consistency(self) -> None:
        streak, min_post_burn_reached = pigean_state._update_gibbs_post_burn_precision_streak(
            stop_pass_streak=1,
            beta_ratio_q=0.05,
            D_mcse_q=0.01,
            beta_rhat_q_post=1.40,
            prior_beta_rel_inconsistency_q=0.10,
            max_rel_mcse_beta=0.20,
            max_abs_mcse_d=0.05,
            max_post_beta_rhat=1.25,
            max_rel_prior_beta_inconsistency=0.50,
            num_post_burn_D=20,
            min_num_post_burn_in_for_epoch=10,
        )
        self.assertTrue(min_post_burn_reached)
        self.assertEqual(streak, 0)

        streak, min_post_burn_reached = pigean_state._update_gibbs_post_burn_precision_streak(
            stop_pass_streak=1,
            beta_ratio_q=0.05,
            D_mcse_q=0.01,
            beta_rhat_q_post=1.10,
            prior_beta_rel_inconsistency_q=0.80,
            max_rel_mcse_beta=0.20,
            max_abs_mcse_d=0.05,
            max_post_beta_rhat=1.25,
            max_rel_prior_beta_inconsistency=0.50,
            num_post_burn_D=20,
            min_num_post_burn_in_for_epoch=10,
        )
        self.assertTrue(min_post_burn_reached)
        self.assertEqual(streak, 0)

        streak, min_post_burn_reached = pigean_state._update_gibbs_post_burn_precision_streak(
            stop_pass_streak=1,
            beta_ratio_q=0.05,
            D_mcse_q=0.01,
            beta_rhat_q_post=1.10,
            prior_beta_rel_inconsistency_q=0.10,
            max_rel_mcse_beta=0.20,
            max_abs_mcse_d=0.05,
            max_post_beta_rhat=1.25,
            max_rel_prior_beta_inconsistency=0.50,
            num_post_burn_D=20,
            min_num_post_burn_in_for_epoch=10,
        )
        self.assertTrue(min_post_burn_reached)
        self.assertEqual(streak, 2)

    def test_raw_common_mask_summary_preserves_gene_local_beta_mass(self) -> None:
        stub_state = _StubState(
            X_orig=np.array([[1.0, 1.0]], dtype=float),
            X_orig_missing_genes=None,
            mean_shifts=np.zeros(2, dtype=float),
            scale_factors=np.ones(2, dtype=float),
            genes_missing=None,
            gene_N=np.array([0.0], dtype=float),
            Y_for_regression=np.zeros(1, dtype=float),
            Y=np.zeros(1, dtype=float),
            background_log_bf=0.0,
        )

        sum_betas_m = np.array(
            [
                [10.0, 0.0],
                [0.0, 10.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        num_sum_beta_m = np.ones_like(sum_betas_m, dtype=float)
        sum_priors_m = np.array([[10.0], [10.0], [0.0]], dtype=float)
        num_sum_y_m = np.ones_like(sum_priors_m, dtype=float)
        zeros_gene = np.zeros_like(sum_priors_m, dtype=float)
        zeros_beta = np.zeros_like(sum_betas_m, dtype=float)

        legacy_mask, legacy_avg_betas_v = pigean_state._outlier_resistant_mean(
            sum_betas_m,
            num_sum_beta_m,
            num_mad=1,
        )
        self.assertLess(np.sum(legacy_avg_betas_v), 1.0)
        self.assertTrue(np.any(legacy_mask))

        final_summary = pigean_state._summarize_gibbs_chain_aggregates(
            stub_state,
            sum_Ys_m=zeros_gene,
            sum_Y_raws_m=zeros_gene,
            sum_log_pos_m=zeros_gene,
            sum_log_po_raws_m=zeros_gene,
            sum_log_po_raws2_m=zeros_gene,
            sum_priors_m=sum_priors_m,
            sum_priors2_m=np.square(sum_priors_m),
            sum_Ds_m=zeros_gene,
            sum_D_raws_m=zeros_gene,
            sum_bf_orig_m=zeros_gene,
            sum_bf_orig_raw_m=zeros_gene,
            sum_bf_orig_raw2_m=zeros_gene,
            sum_betas_m=sum_betas_m,
            sum_betas2_m=np.square(sum_betas_m),
            sum_betas_uncorrected_m=sum_betas_m,
            sum_betas_uncorrected2_m=np.square(sum_betas_m),
            sum_postp_m=zeros_beta,
            sum_beta_tildes_m=zeros_beta,
            sum_z_scores_m=zeros_beta,
            num_sum_Y_m=num_sum_y_m,
            num_sum_beta_m=num_sum_beta_m,
            num_chains_effective=sum_betas_m.shape[0],
            num_mad=1,
            adjust_priors=False,
            gibbs_summary_mode="raw_common_mask",
            write_gibbs_global_filtered_summaries=False,
            gene_set_p_active_threshold=0.5,
        )

        np.testing.assert_allclose(final_summary["avg_betas_v"], np.array([10.0 / 3.0, 10.0 / 3.0]))
        self.assertAlmostEqual(float(np.sum(final_summary["avg_betas_v"])), 20.0 / 3.0)
        self.assertAlmostEqual(float(final_summary["avg_priors_v"][0]), 20.0 / 3.0)

    def test_gene_set_writer_emits_ci_activity_and_global_filtered_columns(self) -> None:
        runtime = SimpleNamespace(
            gene_sets=["GS1"],
            gene_set_labels=None,
            X_orig=sparse.csc_matrix(np.array([[1.0], [0.0]], dtype=float)),
            get_col_sums=lambda matrix, axis=0: np.array(matrix.sum(axis=axis)).ravel(),
            beta_tildes=None,
            p_values=None,
            q_values=None,
            betas=np.array([2.0]),
            betas_r_hat=None,
            betas_mcse=None,
            betas_ci_lower=np.array([0.10]),
            betas_ci_upper=np.array([0.30]),
            betas_p_active=np.array([0.40]),
            betas_global_filtered=np.array([1.5]),
            betas_uncorrected=np.array([4.0]),
            betas_uncorrected_r_hat=None,
            betas_uncorrected_mcse=None,
            betas_uncorrected_ci_lower=np.array([0.20]),
            betas_uncorrected_ci_upper=np.array([0.60]),
            betas_uncorrected_global_filtered=np.array([3.0]),
            non_inf_avg_cond_betas=None,
            non_inf_avg_postps=np.array([0.25]),
            beta_tildes_orig=None,
            betas_orig=None,
            betas_uncorrected_orig=None,
            non_inf_avg_cond_betas_orig=None,
            non_inf_avg_postps_orig=None,
            ps=None,
            p=None,
            sigma2s=None,
            sigma2=None,
            sigma_threshold_k=None,
            sigma_threshold_xo=None,
            X_osc=None,
            total_qc_metrics=None,
            mean_qc_metrics=None,
            scale_factors=np.array([10.0]),
            gene_sets_missing=None,
            gene_sets_ignored=None,
            debug_only_avg_huge=False,
        )

        out = io.StringIO()
        pegs_output_tables.write_gene_set_statistics(
            runtime,
            "unused",
            open_text_fn=lambda _path, _flag: contextlib.nullcontext(out),
            log_fn=lambda *_args, **_kwargs: None,
        )
        lines = out.getvalue().strip().splitlines()
        header = lines[0].split("\t")
        row = lines[1].split("\t")
        self.assertIn("beta_ci_lower", header)
        self.assertIn("beta_ci_upper", header)
        self.assertIn("p_active_beta_gt_eps", header)
        self.assertIn("beta_global_filtered", header)
        self.assertIn("beta_uncorrected_ci_lower", header)
        self.assertIn("beta_uncorrected_global_filtered", header)
        self.assertEqual(row[header.index("beta_ci_lower")], "0.1")
        self.assertEqual(row[header.index("beta_ci_upper")], "0.3")
        self.assertEqual(row[header.index("p_active_beta_gt_eps")], "0.4")
        self.assertEqual(row[header.index("beta_global_filtered")], "0.15")

    def test_gene_set_writer_main_detail_uses_curated_columns(self) -> None:
        runtime = SimpleNamespace(
            gene_sets=["GS1"],
            gene_set_labels=["LIB:GS1"],
            X_orig=sparse.csc_matrix(np.array([[1.0], [0.0]], dtype=float)),
            get_col_sums=lambda matrix, axis=0: np.array(matrix.sum(axis=axis)).ravel(),
            beta_tildes=np.array([5.0]),
            p_values=np.array([1e-4]),
            q_values=np.array([2e-4]),
            z_scores=np.array([4.0]),
            ses=np.array([0.5]),
            betas=np.array([2.0]),
            betas_r_hat=np.array([1.1]),
            betas_mcse=np.array([0.01]),
            betas_ci_lower=np.array([0.10]),
            betas_ci_upper=np.array([0.30]),
            betas_p_active=np.array([0.40]),
            betas_global_filtered=np.array([1.5]),
            betas_uncorrected=np.array([4.0]),
            betas_uncorrected_r_hat=np.array([1.2]),
            betas_uncorrected_mcse=np.array([0.02]),
            betas_uncorrected_ci_lower=np.array([0.20]),
            betas_uncorrected_ci_upper=np.array([0.60]),
            betas_uncorrected_global_filtered=np.array([3.0]),
            non_inf_avg_cond_betas=np.array([0.2]),
            non_inf_avg_postps=np.array([0.25]),
            beta_tildes_orig=np.array([6.0]),
            p_values_orig=np.array([2e-4]),
            z_scores_orig=np.array([3.0]),
            ses_orig=np.array([0.75]),
            betas_orig=np.array([2.5]),
            betas_uncorrected_orig=np.array([4.5]),
            non_inf_avg_cond_betas_orig=np.array([0.4]),
            non_inf_avg_postps_orig=np.array([0.5]),
            ps=np.array([0.01]),
            p=None,
            sigma2s=np.array([0.2]),
            sigma2=None,
            sigma_power=0,
            sigma_threshold_k=None,
            sigma_threshold_xo=None,
            get_scaled_sigma2=lambda scale_factor, sigma2, sigma_power, _k, _xo: sigma2,
            X_osc=None,
            total_qc_metrics=None,
            mean_qc_metrics=None,
            scale_factors=np.array([10.0]),
            gene_sets_missing=None,
            gene_sets_ignored=None,
            debug_only_avg_huge=False,
        )

        out = io.StringIO()
        pegs_output_tables.write_gene_set_statistics(
            runtime,
            "unused",
            output_detail="main",
            open_text_fn=lambda _path, _flag: contextlib.nullcontext(out),
            log_fn=lambda *_args, **_kwargs: None,
        )
        header = out.getvalue().strip().splitlines()[0].split("\t")
        self.assertEqual(
            header,
            [
                "Gene_Set",
                "label",
                "filter_reason",
                "N",
                "scale",
                "beta",
                "beta_ci_lower",
                "beta_ci_upper",
                "p_active_beta_gt_eps",
                "beta_uncorrected",
                "beta_uncorrected_ci_lower",
                "beta_uncorrected_ci_upper",
                "avg_postp",
                "beta_tilde_orig",
                "P_orig",
                "Z_orig",
                "SE_orig",
                "beta_orig",
                "beta_uncorrected_orig",
                "p_used",
                "sigma2_used",
            ],
        )
