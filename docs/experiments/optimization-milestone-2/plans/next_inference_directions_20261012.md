# Directions after optimization milestone 1

The user accepted the measured T2D gain as the first major improvement and authorized commit/tagging. Preserve the implementation as a baseline with its selectable legacy engine. This plan proposes subsequent experiments; none are launched by the review.

## Runtime budget

The optimized 500-update T2D run used 893.26 seconds for inference, including 589.63 seconds in inner fits (66.0%). Other inference work took 303.63 seconds. This full-run fraction supersedes the earlier short-profile estimate for this configuration. It does not describe NAFLD or isolate inner kernel cost from other inner-fit work.

At unchanged outer/inner workloads, Amdahl's law gives the following hypothetical savings. These are calculations, not measured kernel speedups:

| Inner-fit acceleration | Additional total inference speedup | Approximate inference time |
|---|---:|---:|
| 2x | 1.49x | 10.0 minutes |
| 5x | 2.12x | 7.0 minutes |
| Inner cost entirely removed | 2.94x ceiling | 5.1 minutes |

Another order of magnitude over the current implementation would therefore need to reduce other work or the number of outer updates as well. A 2x inner improvement could bring cumulative T2D inference gain to about 10.4x versus legacy at fixed workload; there is no benchmark establishing that improvement yet.

## Proposed order

1. **One bounded inner arithmetic experiment.** Capture representative early and late inner-fit inputs during one short targeted replay only if the existing evidence is insufficient. First separate residual matrix products, elementwise conditional arithmetic, allocation and diagnostic costs. Compile/fuse the measured expensive arithmetic or maintain residuals incrementally. Preserve existing coefficient batches, random draws, hyperparameter updates and stopping. Use fixed inputs/random draws for conditional checks so sampler trajectory divergence does not hide an arithmetic error. Do not revive the abandoned blockwise redesign. Advance only if the measured fraction and kernel gain imply a material overall benefit.
2. **An inner variational approximation as the smallest inference experiment.** Derive the conditional Gaussian spike-and-slab objective and how empirical hyperparameter adaptation should be handled. Replace inner sampling with fitted coefficient moments/inclusion probabilities while keeping the outer loop initially. This directly targets the current bottleneck and allows comparison against long inner reference fits on fixed inputs. It is an approximation and has the same inner-only speed ceiling at fixed outer workload. Test interchangeable correlated annotations and uncertainty as well as means. Logistic spike-and-slab variational methods are established in [Carbonetto and Stephens](https://pcarbo.github.io/bvs.pdf), but this paper is not a ready-made derivation for the full PIGEAN pipeline.
3. **Choose a broader architecture based on the scientific priority.** For maximum speed, consider joint variational inference or an objective with gene indicators summed out. For posterior exploration, prioritize valid support swaps/collapsed support sampling or a persistent joint chain. A persistent chain may avoid repeated inner equilibration, but simply shortening the current refit-and-average procedure does not establish a correct joint sampler.

For logistic sampling, [Pólya–Gamma augmentation](https://arxiv.org/abs/1205.0310) provides conditional Gaussian structure. [Jankowiak's Bayesian variable selection work](https://arxiv.org/abs/2208.01180), which supersedes the earlier [binomial regression paper](https://arxiv.org/abs/2106.14981), and [Zanella and Roberts](https://arxiv.org/abs/1805.00541) motivate tempered support exploration. Published gains are evidence for research directions, not PIGEAN timing predictions. The current inner updates already operate on Gaussian effect estimates; adding logistic augmentation would alter the broader formulation rather than merely accelerate that inner arithmetic.

The production procedure includes prior-dependent HuGE refresh, marginal effect corrections and feedback of averaged inner coefficients. Define which joint target or approximation each alternative represents before implementation. A fixed-independent-evidence prototype is useful, but its equivalence to the dynamic production procedure cannot be assumed.

## Validation

Keep arithmetic validation distinct from statistical validation. Use deterministic fixed-input conditional checks for arithmetic. Compare final scientific summaries against legacy repeatability or reduced reference targets for inference changes; a single non-converged legacy run is not posterior truth. Record priors, uncorrected/corrected betas, slope, rank stability, annotation-family contributions, calibration, uncertainty and cost. For samplers include mode movement, R-hat and MCSE/ESS. For deterministic approximations include cross-start stability and uncertainty calibration rather than reporting MCMC convergence. Start with one bounded experiment and inspect its result before expanding; no unattended trait/seed matrix.
