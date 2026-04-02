from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eaggl import factor_runtime as eaggl_factor_runtime  # noqa: E402


class _TinyState:
    def __init__(self) -> None:
        self.params = {}
        self.param_history = {}
        self.exp_gene_set_factors = None
        self.exp_gene_factors = None
        self.exp_pheno_factors = None
        self.exp_lambdak = None
        self.factor_anchor_relevance = None
        self.factor_relevance = None
        self.consensus_mode = None
        self.consensus_reference_run = None
        self.consensus_run_diagnostics = None
        self.consensus_factor_support = None
        self.consensus_stats_out = None

    def _record_params(self, values, overwrite=False) -> None:
        if overwrite:
            self.params.update(values)
        else:
            for key, value in values.items():
                self.params[key] = value

    def _record_param(self, name, value) -> None:
        self.param_history.setdefault(name, []).append(value)

    def num_factors(self) -> int:
        if self.exp_lambdak is None:
            return 0
        return int(len(self.exp_lambdak))


class _CapturedFactorCall(RuntimeError):
    pass


class PhiAutoFactorRuntimeTest(unittest.TestCase):
    def test_compute_factor_mass_profile_ignores_nonfinite_loadings(self) -> None:
        state = SimpleNamespace(
            exp_gene_set_factors=np.array(
                [
                    [1.0, np.nan, 0.0],
                    [2.0, np.inf, 1.0],
                ],
                dtype=float,
            ),
            exp_gene_factors=None,
            exp_pheno_factors=None,
        )

        profile = eaggl_factor_runtime._compute_factor_mass_profile(state, mass_floor_frac=0.1)

        self.assertTrue(np.isfinite(profile["effective_factor_count"]))
        self.assertGreater(profile["effective_factor_count"], 0.0)
        self.assertEqual(profile["mass_ge_floor_factor_count"], 2)
        self.assertTrue(np.isfinite(profile["max_mass_fraction"]))
        self.assertTrue(np.isfinite(profile["top5_mass_fraction"]))

    def test_run_factor_single_forwards_max_num_iterations_to_nmf(self) -> None:
        calls = {}

        class _ForwardingState:
            def __init__(self) -> None:
                self.X_orig = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
                self.X_phewas_beta = None
                self.X_phewas_beta_uncorrected = None
                self.gene_pheno_combined_prior_Ys = None
                self.gene_pheno_priors = None
                self.gene_pheno_Y = None
                self.combined_prior_Ys = None
                self.priors = None
                self.Y = np.array([1.0, 0.5], dtype=float)
                self.betas = np.array([0.2, 0.3], dtype=float)
                self.betas_uncorrected = np.array([0.2, 0.3], dtype=float)
                self.scale_factors = np.ones(2, dtype=float)
                self.background_log_bf = 0.0
                self.gene_sets = ["gs1", "gs2"]
                self.genes = ["g1", "g2"]
                self.phenos = []
                self.default_pheno = "default"
                self.params = {}

            def _record_params(self, values, overwrite=False):
                self.params.update(values)

            def _bayes_nmf_l2_extension(self, *args, **kwargs):
                calls.update(kwargs)
                raise _CapturedFactorCall()

        state = _ForwardingState()

        with self.assertRaises(_CapturedFactorCall):
            eaggl_factor_runtime._run_factor_single(
                state,
                phi=0.1,
                max_num_iterations=7,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda *args, **kwargs: None,
                log_fn=lambda *args, **kwargs: None,
                info_level=1,
                debug_level=2,
                trace_level=3,
                labeling_module=np,
            )

        self.assertEqual(calls["n_iter"], 7)

    def test_run_factor_learn_phi_only_skips_final_factorization(self) -> None:
        state = _TinyState()

        with mock.patch.object(
            eaggl_factor_runtime,
            "_learn_phi",
            return_value={"phi": 0.025},
        ) as learn_phi_mock, mock.patch.object(
            eaggl_factor_runtime,
            "_run_factor_with_seed",
            side_effect=AssertionError("final factorization should not run"),
        ):
            eaggl_factor_runtime.run_factor(
                state,
                phi=0.05,
                learn_phi=True,
                learn_phi_only=True,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda *args, **kwargs: None,
                log_fn=lambda *args, **kwargs: None,
                info_level=1,
                debug_level=2,
                trace_level=3,
                labeling_module=np,
            )

        learn_phi_mock.assert_called_once()
        self.assertEqual(state.params["phi"], 0.025)

    def test_run_factor_single_skips_empty_gene_prune_without_crashing(self) -> None:
        class _ForwardingState:
            def __init__(self) -> None:
                self.X_orig = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
                self.X_phewas_beta = None
                self.X_phewas_beta_uncorrected = None
                self.gene_pheno_combined_prior_Ys = None
                self.gene_pheno_priors = None
                self.gene_pheno_Y = None
                self.combined_prior_Ys = None
                self.priors = None
                self.Y = np.array([1.0, 0.5], dtype=float)
                self.betas = np.array([0.2, 0.3], dtype=float)
                self.betas_uncorrected = np.array([0.2, 0.3], dtype=float)
                self.scale_factors = np.ones(2, dtype=float)
                self.background_log_bf = 0.0
                self.gene_sets = ["gs1", "gs2"]
                self.genes = ["g1", "g2"]
                self.phenos = []
                self.default_pheno = "default"
                self.params = {}
                self.exp_gene_set_factors = None
                self.exp_gene_factors = None
                self.exp_pheno_factors = None
                self.exp_lambdak = None
                self.factor_anchor_relevance = None
                self.factor_relevance = None

            def _record_params(self, values, overwrite=False):
                self.params.update(values)

        state = _ForwardingState()

        eaggl_factor_runtime._run_factor_single(
            state,
            gene_or_pheno_filter_value=10.0,
            gene_prune_number=2,
            bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
            warn_fn=lambda *args, **kwargs: None,
            log_fn=lambda *args, **kwargs: None,
            info_level=1,
            debug_level=2,
            trace_level=3,
            labeling_module=np,
        )

        self.assertIsNone(state.exp_lambdak)

    def test_gene_set_sort_rank_uses_local_uncorrected_betas_when_state_betas_absent(self) -> None:
        state = SimpleNamespace(
            X_phewas_beta_uncorrected=None,
            X_orig=np.zeros((2, 3), dtype=float),
        )
        betas_uncorrected = np.array([[1.0], [3.0], [2.0]])
        rank = eaggl_factor_runtime._gene_set_sort_rank_for_pruning(
            state,
            betas=None,
            betas_uncorrected=betas_uncorrected,
        )
        np.testing.assert_allclose(rank, np.array([-1.0, -3.0, -2.0]))

    def test_gene_set_prune_number_masks_forward_requested_limit(self) -> None:
        state = SimpleNamespace(
            X_orig=np.zeros((2, 4), dtype=float),
            mean_shifts=np.arange(4, dtype=float),
            scale_factors=np.ones(4, dtype=float),
        )
        calls = {}

        def _stub_compute_gene_set_batches(**kwargs):
            calls.update(kwargs)
            return []

        state._compute_gene_set_batches = _stub_compute_gene_set_batches
        gene_set_mask = np.array([True, False, True, True])
        gene_set_sort_rank = np.array([-1.0, -2.0, -3.0, -4.0])

        eaggl_factor_runtime._compute_gene_set_prune_number_masks(
            state,
            gene_set_mask=gene_set_mask,
            gene_set_sort_rank=gene_set_sort_rank,
            gene_set_prune_number=5,
        )

        self.assertEqual(calls["stop_at"], 5)
        self.assertEqual(calls["tag"], "gene sets")
        np.testing.assert_array_equal(calls["sort_values"], gene_set_sort_rank[gene_set_mask])

    def test_combine_prune_masks_does_not_depend_on_state_phenos(self) -> None:
        prune_masks = [
            np.array([True, False, False, False]),
            np.array([False, True, True, False]),
        ]
        sort_rank = np.array([-4.0, -3.0, -2.0, -1.0])
        logged = []
        combined = eaggl_factor_runtime._combine_prune_masks(
            prune_masks,
            prune_number=2,
            sort_rank=sort_rank,
            tag="gene set",
            log_fn=lambda message, level: logged.append((message, level)),
            trace_level=3,
        )
        np.testing.assert_array_equal(combined, np.array([True, True, False, False]))
        self.assertTrue(any("gene sets remaining after pruning to max number (of 4)" in message for message, _ in logged))

    def test_any_anchor_relevance_matches_single_user_coefficients(self) -> None:
        factor_anchor_relevance = np.array([[0.2], [0.8], [1.0]])
        any_relevance = eaggl_factor_runtime._compute_any_anchor_relevance(factor_anchor_relevance)
        np.testing.assert_allclose(any_relevance, factor_anchor_relevance[:, 0])

    def test_any_anchor_relevance_uses_noisy_or_across_users(self) -> None:
        factor_anchor_relevance = np.array(
            [
                [0.2, 0.5],
                [0.8, 0.1],
                [1.2, -0.5],
            ]
        )
        any_relevance = eaggl_factor_runtime._compute_any_anchor_relevance(factor_anchor_relevance)
        expected = np.array(
            [
                1.0 - (1.0 - 0.2) * (1.0 - 0.5),
                1.0 - (1.0 - 0.8) * (1.0 - 0.1),
                1.0 - (1.0 - 1.0) * (1.0 - 0.0),
            ]
        )
        np.testing.assert_allclose(any_relevance, expected)
        self.assertTrue(np.all(any_relevance >= 0))
        self.assertTrue(np.all(any_relevance <= 1))

    def test_weighted_jaccard_similarity_handles_simple_nonnegative_vectors(self) -> None:
        left = np.array([1.0, 1.0, 0.0])
        right = np.array([1.0, 0.0, 1.0])
        similarity = eaggl_factor_runtime._weighted_jaccard_similarity(left, right, weight_floor=0.0)
        self.assertAlmostEqual(similarity, 1.0 / 3.0)

    def test_redundancy_prefers_gene_loadings_over_gene_set_loadings(self) -> None:
        state = SimpleNamespace(
            exp_gene_set_factors=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            exp_gene_factors=np.array(
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ),
            exp_pheno_factors=None,
        )
        profile = eaggl_factor_runtime._compute_within_run_factor_redundancy_profile(
            state,
            weight_floor=0.0,
        )
        self.assertEqual(profile["redundancy_basis"], "gene")
        self.assertAlmostEqual(profile["redundancy_max"], 1.0)
        self.assertAlmostEqual(profile["redundancy_q90"], 1.0)

    def test_select_phi_candidate_prefers_frontier_elbow_over_nearby_higher_complexity(self) -> None:
        candidates = [
            {
                "phi": 0.05,
                "modal_factor_count": 4,
                "run_support": 1.0,
                "stability": 0.95,
                "stability_defined": True,
                "num_modal_runs": 3,
                "capped": False,
                "redundancy": 0.31,
                "redundancy_max": 0.31,
                "redundancy_q90": 0.23,
                "best_error": 40.0,
                "best_evidence": 8.0,
                "effective_factor_count": 3.8,
                "mass_ge_floor_factor_count": 4,
            },
            {
                "phi": 0.02,
                "modal_factor_count": 5,
                "run_support": 1.0,
                "stability": 0.95,
                "stability_defined": True,
                "num_modal_runs": 3,
                "capped": False,
                "redundancy": 0.18,
                "redundancy_max": 0.18,
                "redundancy_q90": 0.12,
                "best_error": 36.0,
                "best_evidence": 9.0,
                "effective_factor_count": 4.7,
                "mass_ge_floor_factor_count": 5,
            },
            {
                "phi": 1.0,
                "modal_factor_count": 2,
                "run_support": 1.0,
                "stability": 0.95,
                "stability_defined": True,
                "num_modal_runs": 3,
                "capped": False,
                "redundancy": 0.2,
                "redundancy_max": 0.2,
                "redundancy_q90": 0.15,
                "best_error": 100.0,
                "best_evidence": 6.0,
                "effective_factor_count": 2.0,
                "mass_ge_floor_factor_count": 2,
            },
        ]
        selected, reason = eaggl_factor_runtime._select_phi_candidate(
            candidates,
            max_redundancy=0.6,
            max_redundancy_q90=0.35,
            min_run_support=0.6,
            min_stability=0.85,
            max_fit_loss_frac=0.05,
            k_band_frac=0.9,
            runs_per_step=3,
            min_error_gain_per_factor=5.0,
        )
        self.assertEqual(reason, "marginal_gain_frontier")
        self.assertEqual(selected["phi"], 0.05)
        self.assertEqual(selected["selection_pool"], "uncapped")
        self.assertEqual(selected["selection_frontier_size"], 3)
        self.assertIsNotNone(selected["selection_marginal_gain"])

    def test_run_factor_with_learn_phi_selects_less_redundant_candidate_before_final_run(self) -> None:
        state = _TinyState()
        state.uncopyable_module = np
        final_runs = []

        def _stub_single(run_state, **kwargs):
            phi = float(kwargs["phi"])
            if phi < 0.05:
                matrix = np.array(
                    [
                        [0.9, 0.9, 0.1, 0.1],
                        [0.8, 0.8, 0.2, 0.2],
                        [0.0, 0.0, 1.0, 1.0],
                    ]
                )
                error = 35.0
                evidence = 10.0
            elif phi < 0.5:
                matrix = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                error = 40.0
                evidence = 8.0
            else:
                matrix = np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 0.0],
                    ]
                )
                error = 100.0
                evidence = 7.0
            run_state.exp_gene_set_factors = matrix
            run_state.exp_gene_factors = matrix
            run_state.exp_pheno_factors = None
            run_state.exp_lambdak = np.ones(matrix.shape[1], dtype=float)
            run_state.factor_anchor_relevance = np.ones((matrix.shape[1], 1), dtype=float)
            run_state.factor_relevance = np.ones(matrix.shape[1], dtype=float)
            final_runs.append(phi)
            return {
                "run_index": 0,
                "seed": None,
                "evidence": evidence,
                "likelihood": evidence,
                "reconstruction_error": error,
                "num_factors": matrix.shape[1],
                "factor_gene_set_x_pheno": False,
            }

        with mock.patch.object(eaggl_factor_runtime, "_run_factor_single", side_effect=_stub_single):
            eaggl_factor_runtime.run_factor(
                state,
                phi=0.1,
                learn_phi=True,
                learn_phi_runs_per_step=3,
                learn_phi_max_redundancy=0.6,
                learn_phi_min_run_support=0.6,
                learn_phi_min_stability=0.85,
                learn_phi_max_fit_loss_frac=0.05,
                learn_phi_max_steps=3,
                factor_runs=1,
                consensus_nmf=False,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda *args, **kwargs: None,
                log_fn=lambda *args, **kwargs: None,
                info_level=1,
                debug_level=2,
                trace_level=3,
                labeling_module=np,
            )

        self.assertTrue(state.params["learn_phi"])
        self.assertGreaterEqual(state.params["learn_phi_selected_phi"], 0.05)
        self.assertLess(state.params["learn_phi_selected_phi"], 0.5)
        self.assertEqual(state.params["learn_phi_selection_reason"], "marginal_gain_frontier")
        self.assertEqual(state.params["learn_phi_redundancy_basis_target"], "gene")
        self.assertEqual(state.params["learn_phi_redundancy_basis"], "gene")
        self.assertEqual(state.params["learn_phi_selection_pool"], "uncapped")
        self.assertEqual(state.num_factors(), 3)
        np.testing.assert_allclose(state.exp_gene_set_factors, np.eye(3), atol=1e-8)
        self.assertEqual(state.consensus_mode, "single")
        self.assertIn("learn_phi_candidate_phi", state.param_history)
        self.assertGreaterEqual(len(state.param_history["learn_phi_candidate_phi"]), 2)
        self.assertTrue(all(value == "gene" for value in state.param_history["learn_phi_candidate_redundancy_basis"]))
        self.assertIn("learn_phi_candidate_redundancy_q90", state.param_history)
        self.assertAlmostEqual(final_runs[-1], state.params["learn_phi_selected_phi"])
        self.assertEqual(state.params["beta0"], 1.0)
        self.assertEqual(state.params["factor_runs"], 1)
        self.assertEqual(state.params["max_num_iterations"], 100)
        self.assertEqual(state.params["rel_tol"], 1e-4)
        self.assertEqual(state.params["min_lambda_threshold"], 1e-3)
        self.assertEqual(state.params["lmm_provider"], "openai")
        self.assertFalse(state.params["lmm_auth_key_present"])
        self.assertFalse(state.params["anchor_gene_mask_present"])
        self.assertFalse(state.params["anchor_pheno_mask_present"])
        self.assertFalse(state.params["keep_original_loadings"])

    def test_run_factor_with_learn_phi_uses_search_only_prune_and_iteration_overrides(self) -> None:
        state = _TinyState()
        state.uncopyable_module = np
        recorded_kwargs = []

        def _stub_single(run_state, **kwargs):
            phi = float(kwargs["phi"])
            recorded_kwargs.append(dict(kwargs))
            matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
            run_state.exp_gene_set_factors = matrix
            run_state.exp_gene_factors = matrix
            run_state.exp_pheno_factors = None
            run_state.exp_lambdak = np.ones(matrix.shape[1], dtype=float)
            run_state.factor_anchor_relevance = np.ones((matrix.shape[1], 1), dtype=float)
            run_state.factor_relevance = np.ones(matrix.shape[1], dtype=float)
            return {
                "run_index": 0,
                "seed": None,
                "evidence": 1.0 if phi >= 0.1 else 0.5,
                "likelihood": 1.0 if phi >= 0.1 else 0.5,
                "reconstruction_error": 1.0,
                "num_factors": matrix.shape[1],
                "factor_gene_set_x_pheno": False,
            }

        with mock.patch.object(eaggl_factor_runtime, "_run_factor_single", side_effect=_stub_single):
            eaggl_factor_runtime.run_factor(
                state,
                phi=0.1,
                learn_phi=True,
                learn_phi_runs_per_step=1,
                learn_phi_max_steps=1,
                learn_phi_prune_gene_sets_num=11,
                learn_phi_max_num_iterations=7,
                gene_set_prune_number=None,
                max_num_iterations=100,
                factor_runs=1,
                consensus_nmf=False,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda *args, **kwargs: None,
                log_fn=lambda *args, **kwargs: None,
                info_level=1,
                debug_level=2,
                trace_level=3,
                labeling_module=np,
            )

        self.assertGreaterEqual(len(recorded_kwargs), 2)
        search_call = recorded_kwargs[0]
        final_call = recorded_kwargs[-1]
        self.assertEqual(search_call["gene_set_prune_number"], 11)
        self.assertEqual(search_call["max_num_iterations"], 7)
        self.assertIsNone(final_call["gene_set_prune_number"])
        self.assertEqual(final_call["max_num_iterations"], 100)

    def test_run_factor_with_learn_phi_uses_search_only_gene_prune_override(self) -> None:
        state = _TinyState()
        state.uncopyable_module = np
        recorded_kwargs = []

        def _stub_single(run_state, **kwargs):
            recorded_kwargs.append(dict(kwargs))
            matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
            run_state.exp_gene_set_factors = matrix
            run_state.exp_gene_factors = matrix
            run_state.exp_pheno_factors = None
            run_state.exp_lambdak = np.ones(matrix.shape[1], dtype=float)
            run_state.factor_anchor_relevance = np.ones((matrix.shape[1], 1), dtype=float)
            run_state.factor_relevance = np.ones(matrix.shape[1], dtype=float)
            return {
                "run_index": 0,
                "seed": None,
                "evidence": 1.0,
                "likelihood": 1.0,
                "reconstruction_error": 1.0,
                "num_factors": matrix.shape[1],
                "factor_gene_set_x_pheno": False,
            }

        with mock.patch.object(eaggl_factor_runtime, "_run_factor_single", side_effect=_stub_single):
            eaggl_factor_runtime.run_factor(
                state,
                phi=0.1,
                learn_phi=True,
                learn_phi_runs_per_step=1,
                learn_phi_max_steps=1,
                learn_phi_prune_genes_num=13,
                learn_phi_prune_gene_sets_num=11,
                gene_prune_number=None,
                gene_set_prune_number=None,
                factor_runs=1,
                consensus_nmf=False,
                bail_fn=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
                warn_fn=lambda *args, **kwargs: None,
                log_fn=lambda *args, **kwargs: None,
                info_level=1,
                debug_level=2,
                trace_level=3,
                labeling_module=np,
            )

        self.assertGreaterEqual(len(recorded_kwargs), 2)
        search_call = recorded_kwargs[0]
        final_call = recorded_kwargs[-1]
        self.assertEqual(search_call["gene_prune_number"], 13)
        self.assertEqual(search_call["gene_set_prune_number"], 11)
        self.assertIsNone(final_call["gene_prune_number"])

    def test_evaluate_phi_candidate_logs_candidate_summary(self) -> None:
        state = _TinyState()
        messages = []

        def _stub_run_factor_with_seed(run_state, *, seed, run_index, factor_kwargs):
            matrix = np.eye(2, dtype=float)
            run_state.exp_gene_set_factors = matrix
            run_state.exp_gene_factors = matrix
            run_state.exp_pheno_factors = None
            run_state.exp_lambdak = np.ones(2, dtype=float)
            run_state.factor_anchor_relevance = np.ones((2, 1), dtype=float)
            run_state.factor_relevance = np.ones(2, dtype=float)
            return {
                "run_index": run_index,
                "seed": seed,
                "evidence": 3.0,
                "likelihood": 3.0,
                "reconstruction_error": 2.0,
                "num_factors": 2,
                "factor_gene_set_x_pheno": False,
            }

        with mock.patch.object(eaggl_factor_runtime, "_run_factor_with_seed", side_effect=_stub_run_factor_with_seed):
            candidate = eaggl_factor_runtime._evaluate_phi_candidate(
                state,
                phi=0.1,
                seed=0,
                runs_per_step=1,
                factor_kwargs={},
                weight_floor=0.0,
                prune_genes_num=None,
                prune_gene_sets_num=None,
                max_num_iterations=None,
                log_fn=lambda message, level: messages.append((message, level)),
                info_level=1,
            )

        self.assertEqual(candidate["modal_factor_count"], 2)
        self.assertEqual(candidate["redundancy_basis"], "gene")
        self.assertTrue(any("Automatic phi candidate 0.1 summary:" in message for message, _ in messages))
        self.assertTrue(any("redundancy_max[gene]=" in message for message, _ in messages))
        self.assertTrue(any("redundancy_q90=" in message for message, _ in messages))

    def test_write_phi_search_report_handles_string_redundancy_basis(self) -> None:
        from tempfile import TemporaryDirectory

        candidate = {
            "phi": 0.05,
            "modal_factor_count": 3,
            "capped": False,
            "num_modal_runs": 1,
            "run_support": 1.0,
            "stability": None,
            "stability_defined": False,
            "redundancy_basis": "gene",
            "redundancy_max": 0.2,
            "redundancy_q90": 0.1,
            "redundancy_mean": 0.05,
            "redundancy_max_worst": 0.2,
            "best_error": 12.0,
            "best_evidence": 3.0,
            "reference_run_index": 0,
            "modal_run_indices": [0],
            "matched_cosines": [],
        }
        with TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "learn_phi.tsv"
            eaggl_factor_runtime._write_phi_search_report(
                str(report),
                [candidate],
                selected_phi=0.05,
                selection_reason="unit_test",
            )
            text = report.read_text()
        self.assertIn("redundancy_basis", text)
        self.assertIn("\tgene\t", text)

    def test_learn_phi_starts_at_initial_phi_and_caps_total_expansions(self) -> None:
        state = _TinyState()
        evaluated_phis = []

        def _stub_evaluate_phi_candidate(
            _state,
            *,
            phi,
            seed,
            runs_per_step,
            factor_kwargs,
            weight_floor,
            mass_floor_frac,
            prune_genes_num,
            prune_gene_sets_num,
            max_num_iterations,
            log_fn,
            info_level,
        ):
            evaluated_phis.append(float(phi))
            if phi <= 0.02:
                factor_count = 200
                capped = True
            elif phi >= 0.2:
                factor_count = 0
                capped = False
            else:
                factor_count = 71
                capped = False
            return {
                "phi": float(phi),
                "modal_factor_count": factor_count,
                "capped": capped,
                "run_support": 1.0,
                "stability": None,
                "stability_defined": False,
                "num_modal_runs": 1,
                "redundancy_basis": "gene" if factor_count > 0 else "none",
                "redundancy": 0.2 if factor_count > 0 else 0.0,
                "redundancy_max": 0.2 if factor_count > 0 else 0.0,
                "redundancy_q90": 0.1 if factor_count > 0 else 0.0,
                "redundancy_mean": 0.08 if factor_count > 0 else 0.0,
                "redundancy_max_worst": 0.2 if factor_count > 0 else 0.0,
                "best_error": 1.0 + float(phi),
                "best_evidence": 10.0 - float(phi),
                "reference_run_index": 0,
                "modal_run_indices": [0],
                "matched_cosines": [],
            }

        with mock.patch.object(eaggl_factor_runtime, "_evaluate_phi_candidate", side_effect=_stub_evaluate_phi_candidate):
            selected = eaggl_factor_runtime._learn_phi(
                state,
                initial_phi=0.05,
                seed=0,
                runs_per_step=1,
                max_redundancy=0.5,
                max_redundancy_q90=0.35,
                min_run_support=0.6,
                min_stability=0.85,
                max_fit_loss_frac=0.05,
                k_band_frac=0.9,
                max_steps=3,
                expand_factor=2.0,
                weight_floor=0.0,
                report_out=None,
                prune_genes_num=1000,
                prune_gene_sets_num=1000,
                max_num_iterations=None,
                factor_kwargs={"max_num_factors": 200},
                log_fn=lambda *args, **kwargs: None,
                info_level=1,
            )

        self.assertEqual(evaluated_phis[0], 0.05)
        self.assertIn(0.1, evaluated_phis)
        self.assertIn(0.025, evaluated_phis)
        self.assertNotIn(5.0, evaluated_phis)
        self.assertLessEqual(len(evaluated_phis) - 1, 3)
        self.assertEqual(selected["phi"], 0.025)


if __name__ == "__main__":
    unittest.main()
