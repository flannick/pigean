# Lean regression allocations

`--gibbs-engine lean-regression` extends `compact-covariance` at `6299d2e`. All preceding engines remain selectable and legacy remains default. This dependent arithmetic branch is independent of fPS and variance-threshold work.

Grouped regression evaluates moments and covariance from a single annotation matrix, but the previous path still constructs repeated annotation stacks in epoch setup, chain-batch selection, and pseudocount augmentation. The lean path omits these unused stacks, including the tiled column selection mask. It derives zero counts from the grouped design and computes fixed response sums once per regression instead of once per IRLS iteration. Chain-specific coefficients and responses remain separate.

Batch-size decisions still use the previous annotation-size budget even though the repeated matrix is omitted. In particular, partial final batches, processing order, and random-number consumption do not change. This is not the abandoned blockwise sampler redesign. No iteration limit, tolerance, posterior target, or conditional update changes.

Tests forbid sparse horizontal stacking on the lean path and compare complete fits and partial batches exactly to the compact parent. Memory benefit is expected to grow with replicated nonzero entries and chains per batch; measured process peak RSS can include other setup/allocator effects and must be reported separately from traced array allocations. More outer updates repeat the saved allocation work without making the speedup ratio grow indefinitely. The inherited singular-column SE limitation remains a separate promotion issue.
