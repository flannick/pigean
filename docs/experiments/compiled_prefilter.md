# Direct compiled linear-prefilter covariance

`--gibbs-enable compiled-prefilter` evaluates the same sparse quadratic forms directly from CSC annotation columns and CSR correlation rows, avoiding the intermediate correlation-times-design matrix. It requires `sparse-regression`; unmet prerequisites are reported inactive. All presets leave it off. `parallel-prefilter` independently controls the existing two-worker split, and each compiled worker releases the GIL.

This is an arithmetic change, with possible rounding differences, not an inference approximation. Logistic and operator-correlation paths remain on the reference implementation. Near cancellation (expanded-term ratio at most 1e-7), the calculation delegates to the original sparse implementation, preserving its own fallback policy. Testing caught why this is necessary: altered reductions could move a borderline coefficient across the original cancellation threshold. Tests compare ordinary cases against dense formulas and cancellation cases against the previous sparse behavior, plus full linear fits, pseudocounts, signed correlations, RNG and worker lifecycle.

Expected memory saving: avoid sparse matrix-product intermediates, with one gene-length lookup per worker. Runtime is proportional to correlation edges scanned for each annotation's nonzero rows; dense correlations can be expensive. Cold compilation is additional setup. Production timing is pending.

## Completed validation

See [optimization milestone 2](optimization-milestone-2/README.md) for the completed full T2D timings, NAFLD sensitivity checks, memory measurements and retained configuration. This feature remains independently selectable and off in all historical presets. Kernel gains must not be interpreted as whole-run gains. Long-run NAFLD outputs are sensitive despite fixed-input arithmetic checks; no convergence improvement is claimed.
