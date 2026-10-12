# Compiled covariance preparation

Opt in with `--gibbs-engine compiled-covariance`. This includes all preceding
optimization engines; `cached-inner` selects the uncompiled parent and `legacy`
remains the default. The optional `compiled` dependency group declares Numba
0.63.x. It is imported only when this engine needs covariance edge preparation.
Missing Numba raises a clear error; legacy engines do not require it.

Compile the compact covariance edge traversal and exact group-pair reduction
using serial float64 arithmetic without fast-math. Keep row/edge traversal and
pair output order, separate signed and absolute sums, diagonal exclusion,
pseudocount exclusion, and the 65,536-bin reduction limit. Cancellation checks
and their reference fallback remain in the existing regression code. Sparse
index types int32 and int64 are supported. Unusual scalar representations use
the existing Python preparation path.

The first call imports the compiler and compiles or loads machine code from
Numba's disk cache. Later calls in that process reuse it. This one-time cost can
outweigh savings in short runs. The analysis screen records preparation and
warmed inference separately, checks that no new type signature compiled during
the inference timer, and includes preparation in whole-screen time. Compiler
runtime and machine-code cache consume additional memory/storage. No compiler
installation, default change, sampler change or approximation is part of this
experiment.
