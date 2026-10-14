# Frozen inner conditional compatibility gate

This research diagnostic asks whether a proposed joint support-mixture or
Metropolis support swap can preserve the existing frozen coefficient
conditionals. It does not modify production inference.

For the all-included support, production draws coefficient j with mean
`shrink[j] * (beta_tilde[j] - sum(V[k,j]*beta[k], k != j))` and variance
`variance[j] = shrink[j] * se[j]^2`. Independent slab-prior precision only
changes the diagonal. The implied off-diagonal precision is therefore
`Q[j,k] = shrink[j] * V[k,j] / variance[j] = V[k,j] / se[j]^2`.
A twice differentiable joint log density must have equal mixed partials:
`Q[j,k] = Q[k,j]`. A symmetric V with heterogeneous SEs need not satisfy this.
This is a necessary check, not a complete spike/slab or outer-feedback proof.

The implementation checks symmetry, reports a witness pair, and checks positive
definiteness only when symmetry holds. It never silently symmetrizes a rejected
precision. Subsampling coefficients can find a decisive counterexample; passing
subsets does not establish compatibility of the full problem. Tests include a
weighted asymmetric V that is compatible, a symmetric but indefinite precision,
invalid inputs, and finite differences of actual production slab means.

This failure does not imply that the implemented Markov chain has no stationary
distribution. It means one cannot treat these update rules as the full
conditionals of a common joint posterior, then derive support weights or
Metropolis ratios from that nonexistent compatible density. Approximating the
algorithm's stationary behavior would be a different, explicitly defined task.

Background: [Arnold and Press, Compatible Conditional Distributions (1989)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478750).
The application to PIGEAN is the source-derived calculation above.

Run `scripts/research/audit_frozen_inner.py` with a trusted captured conditional
and a compressed output path. It validates the capture checksum before reading
and leaves the captured arrays untouched. Source capture and audit provenance
are retained in its output. No new inference flags or defaults are introduced.
