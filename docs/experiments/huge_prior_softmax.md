# Equivalent HuGE prior normalization

`--gibbs-enable huge-prior-softmax` replaces repeated sparse matrix operations used to normalize gene priors within a signal with a stable softmax of their log odds. For a signal with exactly one causal gene, the product of the other genes' absence probabilities is common to all candidates and cancels in normalization. The implementation uses the caller's clipped probabilities, preserves the sparse candidate pattern (including stored zeros), and changes no downstream regional cap/scale, phantom-gene, evidence-of-absence or covariate correction rules.

This is an arithmetic optimization, not a new model or inference approximation. Floating-point reduction order changes. It is independent of other optimization flags and off in all presets; Numba is optional unless enabled. Tests cover all 16 combinations of downstream regional/absence modes with ordinary/extreme priors, empty and singleton signals, and stored zero entries. Full production-size timing and sampler validation are required before retention.

The fPS branch remains separate. A future merge may need to reconcile the guarded prior-normalization block in HuGE distillation; this feature should preserve whichever downstream regional evidence semantics that branch selects. No fPS behavior is adopted here.
