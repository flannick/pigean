# Experimental corrected-inner effort

`--gibbs-enable inner-early-check` halves the configured minimum iteration count (floor 1) before checking corrected-inner convergence. Burn-in/maximum caps, R-hat threshold, chain count and initial hyperparameter learning are unchanged. All presets leave it off. It can be disabled independently.

This changes finite-run inference, not just arithmetic. A translation test verifies the scope; scientific and runtime screening is still required. Earlier stopping is not evidence of better uncertainty calibration or posterior mixing.
