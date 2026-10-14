# Methods-aligned conditional target and practical comparison

The scientific target for the first candidate is the coefficient conditional
given binary gene indicators, with fixed intercept and hyperparameters:

`P(beta | D, X, alpha, pi, tau)` is proportional to the Bernoulli logistic
likelihood with `eta = alpha + X beta`, times independent spike-and-normal-slab
priors. Coefficient units and the centering/scaling of X must be recorded by any
production adapter. This matches the conditional-model intent in `methods.tex`;
it replaces the marginal-summary LDpred analogy rather than claiming that
analogy is exact. It does not define or validate the full HuGE outer posterior.

Pólya–Gamma augmentation introduces one latent variable per gene. Conditional on
these variables, coefficient precision is `X.T omega X + diag(1/tau)` and the
linear term is `X.T (D - 1/2 - omega alpha)`. The augmented variables are sampled,
not replaced with their expectations. Integrating the active coefficients gives
a Gaussian determinant/quadratic-form support weight. After support updates,
coefficients must be redrawn before the next augmentation update. The baseline
coordinate kernel, collapsed support kernel, and optional reversible swaps can
therefore be compared against the same logistic target. Independent quadrature
uses the original logistic likelihood, not the augmentation formulas.

Algorithm source: [Polson, Scott and Windle (2013), technical supplement,
Algorithms 1–3](https://arxiv.org/abs/1205.0310). This research implementation
uses existing NumPy/SciPy/Numba dependencies and its own implementation of the
published rejection algorithm; no package was installed. Distribution moments,
Laplace transforms and tail arithmetic are tested separately.

## Heuristics and integration boundaries

| Component | Initial treatment | Interpretation |
|---|---|---|
| Fixed signal probabilities and evolving HuGE competition | Retain actual production helper behavior in any outer screen | Required biology; no fixed-evidence substitution |
| HuGE covariate correction | Retain in the production outer path | Practical evidence correction; not certified joint likelihood |
| Initial annotation selection / 5,000 maximum | Fix a common panel for each comparison; reduced panels labelled | Conditional inference within a selected model, not all-annotation uncertainty |
| Marginal SE inflation / annotation covariate correction | Cannot insert arbitrary marginal SEs into the full logistic likelihood unchanged | Must evaluate a documented alternative or request guidance; no silent equivalence claim |
| Setting small sampled coefficients to zero | Exclude from exact conditional correctness tests; keep as separate heuristic if screened | Projection during sampling changes the target; output-only thresholding is distinct |
| Mean coefficients fed to outer priors | Retain for an initial empirical adapter if used | Approximate outer estimator; exact inner sampling does not make outer exact |
| Intercept / slab / inclusion updates | Freeze in reference tests; record production choices separately | Reference validation is conditional on nuisance values |

Prespecified checks: quadrature support probabilities stable to 1e-7 under node
refinement; easy-case sampling errors <=.02 in inclusion and <=.03 in coefficient
means; difficult competing-support checks report both error and classical split
R-hat (<=1.05), not just legacy agreement. Count JIT/setup and warm runtime
separately. Start with two coefficients; then test realistic captured design
dimensions/cost before expanding outer budgets. Full biological non-inferiority
requires matched T2D/NAFLD data and reviewed positive/negative controls; tiny
synthetic checks cannot establish it.
