# Experimental corrected-inner effort

`--gibbs-enable inner-early-check` halves the configured minimum iteration count (floor 1) before checking corrected-inner convergence. Burn-in/maximum caps, R-hat threshold, chain count and initial hyperparameter learning are unchanged. All presets leave it off. It can be disabled independently.

This changes finite-run inference, not just arithmetic. A translation test verifies the scope; scientific and runtime screening is still required. Earlier stopping is not evidence of better uncertainty calibration or posterior mixing.

## Conditional-mean experiment

`--gibbs-enable inner-mean` selects the existing Gauss-Seidel conditional-mean iteration for corrected fits only. This is a fixed-point approximation inside the stochastic outer loop, not a derived joint variational posterior. It does not implement expected-D outer warmup. It is independently selectable, off in all presets, and can be combined with earlier checks or arithmetic options. Results and uncertainty may change; no statistical target equivalence is claimed.

## Screening decision

Neither probe is recommended. Matched 101-update T2D screens with the seven arithmetic options measured only 1.079x for earlier checks and 1.037x for conditional means. Prior/uncorrected-beta slopes shifted roughly 6–7% and 4%, respectively; corrected-beta correlations were 0.801 and 0.734. R-hat q90 remained above 4 in all three screens. Keep the switches off by default as reproducible experiments; neither warranted a full follow-up run. These results do not rule out a separately derived and calibrated inference approximation.
