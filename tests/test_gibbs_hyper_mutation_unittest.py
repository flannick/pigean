from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

sys.argv = ["pigean.py", "gibbs"]
from pigean import main_support as pigean_main_support  # noqa: E402
from pigean import state as pigean  # noqa: E402

pigean.bind_legacy_namespace(pigean_main_support)


class _StubState:
    def __init__(self) -> None:
        self.p = 0.2
        self.sigma2 = 0.4
        self.sigma_power = 2.0
        self.ps = np.full((2, 3), self.p, dtype=float)
        self.sigma2s = np.full((2, 3), self.sigma2, dtype=float)
        self.scale_factors = np.ones(3, dtype=float)
        self.gene_sets = np.array(["GS1", "GS2", "GS3"])
        self.params: dict[str, object] = {}

    def _record_param(self, param, value, overwrite=False, record_only_first_time=False) -> None:
        if record_only_first_time and param in self.params:
            return
        if overwrite or param not in self.params:
            self.params[param] = value
            return
        current = self.params[param]
        if isinstance(current, list):
            if current[-1] != value:
                current.append(value)
            return
        if current != value:
            self.params[param] = [current, value]

    def set_p(self, p) -> None:
        self.p = p

    def set_sigma(self, sigma2, sigma_power) -> None:
        self.sigma2 = sigma2
        self.sigma_power = sigma_power


def _epoch_inputs():
    return (
        {
            "all_sum_betas_m": np.zeros((2, 3), dtype=float),
            "all_num_sum_m": np.ones((2, 1), dtype=float),
            "num_p_increases": 0,
        },
        {
            "sum_betas_m": np.zeros((2, 3), dtype=float),
            "num_sum_beta_m": np.ones((2, 1), dtype=float),
        },
    )


class GibbsHyperMutationTest(unittest.TestCase):
    def test_default_mode_has_no_hidden_hyper_mutation(self) -> None:
        state = _StubState()
        epoch_runtime, epoch_sums = _epoch_inputs()
        update = pigean._maybe_restart_gibbs_for_low_betas(
            state=state,
            increase_hyper_if_betas_below_for_epoch=None,
            experimental_hyper_mutation=False,
            num_before_checking_p_increase=0,
            p_scale_factor=2.0,
            epoch_runtime=epoch_runtime,
            epoch_sums=epoch_sums,
            num_mad=3,
            num_attempts=1,
            max_num_attempt_restarts=3,
            iteration_num=5,
        )
        self.assertTrue(update.gibbs_good)
        self.assertFalse(update.should_break)
        self.assertNotIn("gibbs_hyper_mutation_event", state.params)

    def test_no_signal_without_experimental_mode_fails_explicitly(self) -> None:
        state = _StubState()
        epoch_runtime, epoch_sums = _epoch_inputs()

        orig_bail = pigean.bail
        try:
            def _raise(msg):
                raise RuntimeError(msg)
            pigean.bail = _raise
            with self.assertRaisesRegex(RuntimeError, "explicit failure without hyper mutation"):
                pigean._maybe_restart_gibbs_for_low_betas(
                    state=state,
                    increase_hyper_if_betas_below_for_epoch=0.01,
                    experimental_hyper_mutation=False,
                    num_before_checking_p_increase=0,
                    p_scale_factor=2.0,
                    epoch_runtime=epoch_runtime,
                    epoch_sums=epoch_sums,
                    num_mad=3,
                    num_attempts=1,
                    max_num_attempt_restarts=3,
                    iteration_num=5,
                )
        finally:
            pigean.bail = orig_bail

        self.assertEqual(state.params.get("gibbs_no_signal_detected"), 1)

    def test_experimental_mode_emits_structured_mutation_event(self) -> None:
        state = _StubState()
        epoch_runtime, epoch_sums = _epoch_inputs()
        update = pigean._maybe_restart_gibbs_for_low_betas(
            state=state,
            increase_hyper_if_betas_below_for_epoch=0.01,
            experimental_hyper_mutation=True,
            num_before_checking_p_increase=0,
            p_scale_factor=2.0,
            epoch_runtime=epoch_runtime,
            epoch_sums=epoch_sums,
            num_mad=3,
            num_attempts=1,
            max_num_attempt_restarts=3,
            iteration_num=5,
        )
        self.assertFalse(update.gibbs_good)
        self.assertTrue(update.should_break)
        self.assertEqual(state.params.get("gibbs_hyper_mutation_event_count"), 1)

        event_value = state.params["gibbs_hyper_mutation_event"]
        if isinstance(event_value, list):
            event_value = event_value[-1]
        event = json.loads(event_value)
        self.assertEqual(event["event"], "gibbs_hyper_mutation_restart")
        self.assertEqual(event["trigger"], "all_betas_below_threshold")


if __name__ == "__main__":
    unittest.main()
