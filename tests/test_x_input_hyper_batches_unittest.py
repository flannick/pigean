from __future__ import annotations

import unittest

import numpy as np

from pigean import x_inputs_core as pigean_x_inputs_core


class XInputHyperBatchTest(unittest.TestCase):
    def test_max_num_gene_sets_hyper_limits_learning_subset_to_requested_cap(self) -> None:
        class _Runtime:
            def __init__(self) -> None:
                self.p_values = np.ones(100, dtype=float)
                self.gene_set_batches = np.array(["B0"] * 100, dtype=object)
                self.batch_size = 50
                self.sigma_power = 0.0
                self.ps = None
                self.sigma2s = None
                self.p = None
                self.sigma2 = None
                self.sigma_threshold_k = None
                self.sigma_threshold_xo = None
                self.recorded_params = []

            def set_p(self, value):
                self.p = value

            def set_sigma(self, value, sigma_power):
                self.sigma2 = value
                self.sigma_power = sigma_power

            def _record_params(self, values):
                self.recorded_params.append(values)

        runtime = _Runtime()
        seen_sizes: list[int] = []
        original = pigean_x_inputs_core.learn_hyper_for_gene_set_batch
        try:
            def _fake_learn_hyper_for_gene_set_batch(*, gene_sets_for_hyper_mask, **_kwargs):
                seen_sizes.append(int(np.sum(gene_sets_for_hyper_mask)))
                return {
                    "computed_p": 0.01,
                    "computed_sigma2": 1e-6,
                    "computed_sigma_power": 0.0,
                }

            pigean_x_inputs_core.learn_hyper_for_gene_set_batch = _fake_learn_hyper_for_gene_set_batch  # type: ignore[assignment]
            pigean_x_inputs_core.maybe_learn_batch_hyper_after_x_read_for_runtime(
                runtime_state=runtime,
                skip_betas=False,
                update_hyper_p=True,
                update_hyper_sigma=False,
                batches=["B0"],
                num_ignored_gene_sets=[0],
                first_for_hyper=False,
                max_num_gene_sets_hyper=20,
                first_for_sigma_cond=False,
                fixed_sigma_cond=False,
                first_max_p_for_hyper=False,
                max_num_burn_in=5,
                max_num_iter_betas=20,
                min_num_iter_betas=5,
                num_chains_betas=2,
                r_threshold_burn_in_betas=1.01,
                use_max_r_for_convergence_betas=True,
                max_frac_sem_betas=0.01,
                max_allowed_batch_correlation=None,
                sigma_num_devs_to_top=2.0,
                p_noninf_inflate=1.0,
                sparse_solution=False,
                sparse_frac_betas=0.0,
                betas_trace_out=None,
                log_fn=lambda *_args, **_kwargs: None,
                debug_level=1,
            )
        finally:
            pigean_x_inputs_core.learn_hyper_for_gene_set_batch = original  # type: ignore[assignment]

        self.assertEqual(seen_sizes, [20])
        self.assertTrue(np.allclose(runtime.ps, 0.01))
        self.assertTrue(np.allclose(runtime.sigma2s, 1e-6))


if __name__ == "__main__":
    unittest.main()
