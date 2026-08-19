"""Guard the flag recipe that gives every seed an identical Gibbs budget.

Why this exists: PIGEAN's outer Gibbs has three independent early exits, and
each of them can fire at a different iteration for a different seed. When they
do, two seeds of the same run differ in *how much sampling they got* -- not
only in which draws they got -- which is a large part of the run-to-run spread
seen in `scripts/seed_sweep/`. Measured on the real T2D bottom-line GWAS, seeds
0/1/2 completed 2/6/3 epochs, 204/520/284 iterations and aggregated 30/70/40
effective chains.

The three exits and what closes each:

* stall detectors and restart epochs -- ``--disable-stall-detection``
* the MCSE/R-hat stop (``_GIBBS_STOPPING_PRESETS`` in ``pigean/cli.py``) --
  only bounded by making ``--min-num-post-burn-in`` equal
  ``--max-num-post-burn-in``
* the adaptive burn-in, otherwise free within ``burn=[10,400]`` -- only bounded
  by making ``--min-num-burn-in`` equal ``--max-num-burn-in``

``--disable-stall-detection`` alone is *not* sufficient; it leaves the other
two free, and a run under it was observed stopping at 492 iterations for one
seed and 500 for another. That near-miss is exactly what this test is here to
catch, so both cases are asserted: under the partial recipe each phase must
still have room to exit early, and under the full recipe every seed must land
on the same budget.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SEEDS = (0, 1)

# params.tsv keys describing how much sampling a run actually did.
BUDGET_KEYS = (
    "num_gibbs_epochs_completed",
    "num_gibbs_iter_total",
    "gibbs_global_summary_chain_keep_count",
)

# params.tsv keys describing the room each phase had to stop early in. A phase
# whose min is below its max can end at a seed-dependent iteration.
PHASE_BOUND_KEYS = (
    ("min_num_burn_in", "max_num_burn_in"),
    ("min_num_post_burn_in", "max_num_post_burn_in"),
)

RECORDED_KEYS = BUDGET_KEYS + tuple(k for pair in PHASE_BOUND_KEYS for k in pair)

PARTIAL_RECIPE = [
    "--disable-stall-detection",
    "--max-num-iter", "60",
]

FULL_RECIPE = PARTIAL_RECIPE + [
    "--min-num-burn-in", "20",
    "--max-num-burn-in", "20",
    "--min-num-post-burn-in", "40",
    "--max-num-post-burn-in", "40",
]


class GibbsStoppingPinnedBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.fixture_root = cls.repo_root / "tests" / "data" / "t2d_smoke"
        cls.model_data = cls.repo_root / "tests" / "data" / "model_small"
        required = [
            cls.fixture_root / "T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz",
            cls.fixture_root / "gene_set_list_mouse_t2d_toy.txt",
            cls.model_data / "portal_gencode.gene.map",
            cls.model_data / "NCBI37.3.plink.gene.loc",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest("Missing bundled T2D toy fixtures: " + ", ".join(missing))

        cls._tmpdir_ctx = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_ctx.name)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir_ctx"):
            cls._tmpdir_ctx.cleanup()

    def _run_seed(self, label: str, seed: int, recipe: list[str]) -> dict[str, str]:
        out_dir = self.tmpdir / ("%s_seed_%d" % (label, seed))
        out_dir.mkdir(parents=True, exist_ok=True)
        params_out = out_dir / "params.tsv"
        cmd = [
            sys.executable,
            "-m",
            "pigean",
            "gibbs",
            "--hide-opts",
            "--hide-progress",
            # Both flags, deliberately: --seed alone leaves the pre-Gibbs
            # default_rng draws unpinned, so the comparison would not be
            # measuring the stopping rule in isolation.
            "--deterministic",
            "--seed",
            str(seed),
            "--X-in",
            str(self.fixture_root / "gene_set_list_mouse_t2d_toy.txt"),
            "--gene-map-in",
            str(self.model_data / "portal_gencode.gene.map"),
            "--gene-loc-file",
            str(self.model_data / "NCBI37.3.plink.gene.loc"),
            "--gene-loc-file-huge",
            str(self.model_data / "NCBI37.3.plink.gene.loc"),
            "--gwas-in",
            str(self.fixture_root / "T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz"),
            "--gwas-chrom-col", "CHROM",
            "--gwas-pos-col", "POS",
            "--gwas-p-col", "P",
            "--gwas-n-col", "N",
            "--params-out",
            str(params_out),
            *recipe,
        ]
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        src_root = str(self.repo_root / "src")
        env["PYTHONPATH"] = (
            src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        )
        proc = subprocess.run(
            cmd, cwd=self.repo_root, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(
            proc.returncode,
            0,
            "pigean failed for %s seed %d\nSTDERR:\n%s" % (label, seed, proc.stderr[-4000:]),
        )

        budget = {}
        with params_out.open() as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) >= 3 and row[0] in RECORDED_KEYS:
                    budget[row[0]] = row[2]
        return budget

    def test_full_recipe_gives_every_seed_the_same_budget(self) -> None:
        budgets = {seed: self._run_seed("full", seed, FULL_RECIPE) for seed in SEEDS}

        # First that the flags were honoured at all: with min == max a phase
        # has no room to stop early, which is the mechanism the recipe relies
        # on. If PIGEAN silently reinterpreted them, the equality assertions
        # below could still pass for the wrong reason.
        for seed, budget in budgets.items():
            for low_key, high_key in PHASE_BOUND_KEYS:
                self.assertEqual(
                    budget.get(low_key),
                    budget.get(high_key),
                    "seed %d resolved %s=%s but %s=%s; the phase can still end at a "
                    "seed-dependent iteration"
                    % (seed, low_key, budget.get(low_key), high_key, budget.get(high_key)),
                )

        for key in BUDGET_KEYS:
            recorded = {seed: budget.get(key) for seed, budget in budgets.items()}
            self.assertNotIn(
                None,
                recorded.values(),
                "params.tsv did not record %s; the budget can no longer be verified "
                "and scripts/seed_sweep/compare.py will stop reporting it" % key,
            )
            self.assertEqual(
                len(set(recorded.values())),
                1,
                "seeds disagree on %s (%s) under the fully pinned recipe -- an early "
                "exit has been reintroduced, so seeds again differ in how much "
                "sampling they got" % (key, recorded),
            )

    def test_disable_stall_detection_alone_leaves_both_phases_free(self) -> None:
        """The partial recipe is documented as insufficient; prove it still is.

        Asserting that two seeds *do* end up with different budgets would be a
        coin flip -- on a toy fixture they can coincide. The mechanism is what
        is deterministic: under the partial recipe each phase keeps a min below
        its max, so an early exit remains reachable. If this ever fails, the
        phase bounds are being closed by something else and the guidance in
        scripts/seed_sweep/README.md is stale rather than wrong.
        """
        budget = self._run_seed("partial", SEEDS[0], PARTIAL_RECIPE)
        self.assertTrue(budget, "params.tsv recorded no budget keys at all")

        for low_key, high_key in PHASE_BOUND_KEYS:
            low, high = budget.get(low_key), budget.get(high_key)
            self.assertIsNotNone(low, "params.tsv stopped recording %s" % low_key)
            self.assertIsNotNone(high, "params.tsv stopped recording %s" % high_key)
            self.assertLess(
                int(float(low)),
                int(float(high)),
                "%s == %s under --disable-stall-detection alone, so the phase is "
                "already pinned and the README's three-exit guidance is out of date"
                % (low_key, high_key),
            )


if __name__ == "__main__":
    unittest.main()
