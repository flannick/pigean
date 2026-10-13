# Inner approximation target audit

Inspected the actual corrected-beta caller and conditional update, then the captured T2D outer-25 input. This is an audit of a proposed replacement, not a new sampler or a declaration that the entire production pipeline has a particular joint posterior.

For a coefficient included in the slab, the current conditional uses residual `beta_tilde[j] - sum(V[k,j] * beta[k], k != j)` and sampling variance based on `se[j]^2`. With fixed independent Gaussian slab priors, a compatible Gaussian likelihood precision would require off-diagonal entries `Q[j,k] = V[k,j] / se[j]^2`. Such a joint Gaussian precision must be symmetric. Symmetric V alone does not ensure this when SEs differ.

Checked 100 evenly spaced coefficients from the real 1503-coefficient captured problem. V is exactly symmetric on that subset. SE-squared ranges from 0.0001959 to 0.0901646. The maximum absolute precision asymmetry divided by the maximum absolute precision entry is 0.16847; 99% of sampled coefficient pairs differ beyond rtol 1e-8 / atol 1e-10. Evidence: `results/autonomous_20261012/inner_target_audit.json`; source inputs are checksum-bound in `inner_T2D_25/capture.json`.

Consequence: a standard joint Gaussian spike-and-slab variational objective is not a justified drop-in replacement for these fixed inner conditionals. Symmetrizing/scaling the likelihood would be an explicit model change, and deterministic iteration of current conditional means would be a fixed-point approximation rather than a derived variational posterior. Prior-dependent outer feedback adds a further target-definition requirement. Persistent joint chains and valid support-MH moves likewise require a specified compatible target.

Do not silently label an approximate conditional-mean procedure as VI. The obvious next bounded inference probes are reduced minimum inner effort and the already-defined deterministic inner mean iteration, clearly labelled approximations and benchmarked against the retained sampler. A calibrated joint VI/collapsed/support-sampling implementation is a separate research design task, not an obvious overnight arithmetic extension.
