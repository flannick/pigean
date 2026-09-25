# HDL gene-dependence robustness audit

The maintained narrative is the gene-dependence subsection in [the EAGGL methods supplement](../../methods.tex), authored in [its included TeX section](../../gene_dependence_robustness.tex). This is separate from the mechanism-specific SNP-score validation note.

These compact outputs contain no GWAS records or gene-level score arrays:

- `baseline.tsv.gz`: current, tightly converged two-group logistic fits with the existing independent pseudocount convention. Historical dashboard files are not overwritten.
- `covariance_stress.tsv.gz`: fixed structural-overlap correlation assumptions and shrinkage grid, with explicit necessary binary-correlation bounds. Invalid binary-bound settings retain diagnostic values only.
- `annotation_source_stress.tsv.gz`: one-source-at-a-time omission at shrinkage 0.1, retaining duplicate membership sets supported by other sources.
- `group_deletions.tsv.gz`: fixed-hit-label deletion of major overlapping annotation groups and supplied 1-Mb gene-start windows. These are influence checks, not full annotation refits or LD-block jackknife estimates.
- `summary.json`: input SHA-256 values, dimensions, scope, missing prerequisites, and compact summaries.

No HDL annotation coefficients are present in the supplied bundle. Consequently, the overlap matrix is a declared target-independent stress model, not an inferred HDL residual correlation. Passing pairwise binary bounds is necessary but not sufficient for a valid multivariate binary distribution. None of these P values is claimed to be empirically calibrated.

Reproduction in the analysis workspace uses `../.venv/bin/python scripts/audit_hdl_gene_dependence_20260907.py`; validation uses `../.venv/bin/python -m unittest discover -s scripts -p test_hdl_gene_dependence_audit_20260907.py -v`. Inputs are read from the saved run and reference paths recorded in `summary.json`; no simulations or target GWAS downloads occur. The diagnostic script and its tests are copied here to preserve an inspectable implementation; the archived copy can be run by setting `PIGEAN_AUDIT_ROOT` to the original analysis workspace. The data/resource locations otherwise follow the paths recorded in the manifest.
