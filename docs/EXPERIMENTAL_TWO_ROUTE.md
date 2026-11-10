# Experimental two-route sampler (local review)

This is an isolated research implementation, not a replacement for
`pigean gibbs`. It does not change defaults, production formats, or EAGGL.
There are two separate changes to review: a new model relative to native
PIGEAN, and a new transition that preserves the same posterior as the
centered sampler within that experimental model.

## Target

Let `S` denote which SNP signals are real. Each mask is drawn from the
external SNP-only distribution. Fit the conditional model below per mask
and combine conditional expectations with the external mask weights.
Annotation evidence does not update those weights. This is a modular
model, not a claim that the native approximate Gibbs loop has this target.

For adjusted annotation row `z_g`, define `eta_g = z_g @ beta` and
`q_g = logistic(logit(p0) + eta_g)`. Coefficients have independent
spike-and-normal-slab priors. A separate trait-wide involvement state
`D_g` has Bernoulli probability `q_g` before conditioning on regional
evidence. Each active signal has exactly one assigned nearby effector,
with prior weight `exp(eta_g) * coupling**D_g` within the candidate region.
Conditional on assignment, the observed local data have relative
likelihood equal to the supplied SNP-only candidate weight `w_kg`.
Summing assignments gives the per-region likelihood

```text
sum_g w_kg exp(eta_g) coupling**D_g / sum_g exp(eta_g) coupling**D_g
```

Region likelihoods multiply conditional on the shared gene states. This
counts each modeled region's evidence once and uses a shared `D_g` when
a gene appears in several regions. It does not reconstruct dependence
between the original SNP summaries. Genes without an active candidate
link are integrated out during fitting and can still receive indirect
support through `q_g`. Fivefold coupling is a test assumption, not a
calibrated constant or a claim about the original model. Setting coupling
to one retains the indirect readout but removes its feedback to the fit.

The outputs deliberately distinguish:

- `direct_without_annotations`: probability of at least one assignment
  in this mask using SNP-only candidate weights.
- `direct_with_annotations`: posterior probability of at least one
  assignment with the fitted model.
- `indirect`: posterior average of the annotation-only probability `q_g`.
- `conditional_involvement`: posterior `P(D_g=1)` after conditioning on
  regional data; differs from `indirect` for candidate genes.
- `either`: posterior probability of involvement or at least one
  assignment, computed jointly. It is not native `combined_D`.

## Transition

The centered baseline holds `D` fixed when proposing a coefficient. The
proposal can be rejected simply because changing `beta` would imply a
different configuration of latent gene states. This can happen even
when those states add no marginal evidence about `beta`.

The candidate uses an equivalent uniform representation, `U_g` uniform
on `[0,1]` and `D_g = 1[U_g < q_g]`. Draw `U_g` conditional on the current
state: below `q_g` for `D_g=1`, above it otherwise. Propose a change in
`beta` while leaving the uniforms fixed, letting `D` change at the same
time. The implementation samples just the induced switches, using
`q_new / q_old` or `(q_new - q_old) / (1 - q_old)` when a switch is
possible. The acceptance ratio contains the regional likelihood change
and the coefficient prior/proposal terms. There is no separate Bernoulli
prior ratio for a move holding `U` fixed. Centered moves use the Bernoulli
ratio. Both moves, and the exact single-gene conditional updates, preserve
the same joint posterior. Interleave centered and joint moves with a
fixed probability, default 0.75. Probability zero is the centered control.

## Running And Testing

From the repository root, with Python, NumPy, SciPy, Numba (the `compiled`
extra), and pytest available:

```sh
PYTHONPATH=src python scripts/run_experimental_two_route_example.py results/two_route_example
PYTHONPATH=src python -m pytest tests/test_experimental_two_route.py
```

The example writes its own synthetic input and runs both samplers. It
needs no GWAS, data bundle, or network access. The example's G4 has no
candidate signal, so its direct scores are zero but its indirect and
either-route scores are not. The tests compare both transitions against
independent enumeration and numerical quadrature, include a deliberately
hard synthetic case, and cover the off-locus output and CLI.

To run a single fit with prepared arrays:

```sh
PYTHONPATH=src python -m pigean.experimental_two_route input.npz output.npz --random-seed 11
```

`mat` is annotations by candidate genes; `initial`, `pi`, `sd` are
annotation-aligned vectors. `ids`, `ptr`, and `weights` pack the candidate
list and normalized weights of each active region; indices are zero-based.
`panel` lists monitored candidate indices and may use `-1` to mark absent
candidates. For full-universe output, provide `all_gene_mat` and `keep`,
where `all_gene_mat[:, keep]` equals `mat`. Reported panel traces at `-1`
are placeholders, not zero indirect support; full-universe outputs use
the actual annotation prediction. The command processes one chain and
one fixed mask. Use independent seeds and dispersed starts to diagnose
mixing, then enough independent masks to assess marginal precision.
This is not yet a native-format adapter or a biological calibration.

## Benchmark Summary

Real-data results below are summaries of local runs, not outputs of the
synthetic example. No raw GWAS, prepared cache, or real-data results with
individual gene scores are included. The review base is native revision
`e1cb1fb02fbd23d16b1916031d93b73f2476289b`. T2D and schizophrenia reused
archived native-GWAS pilot inputs; NAFLD was prepared at the review base.
All experimental comparisons used background `p0=0.05`, coupling five,
and a joint-move probability of 0.75 versus zero. Within a SNP draw, the
annotation design, priors, candidate links, chain starts, and sweep
budget were held fixed. R-hat below is rank-normalized split/folded.

| Trait | Annotations | Paired SNP draws | Sweeps / discarded / stride | Worst coefficient R-hat, centered to joint | Indirect-score bulk ESS per CPU minute, joint / centered |
| --- | ---: | ---: | --- | --- | --- |
| T2D | 1,079 | 8 | 3,200 / 800 / 5 | 1.454 to 1.022 | 3.77x on one timing draw |
| Schizophrenia | 686 | 8 | 3,200 / 800 / 5 | 1.076 to 1.013 | 1.83x on one timing draw |
| NAFLD | 728 | 4 | 4,800 / 1,200 / 5 | 1.038 to 1.005 | 1.15-1.73x across four draws |

T2D and schizophrenia used two starts per draw in the main short-run
comparison. NAFLD used four starts per draw and the public European
ancestry [Knowledge Portal bottom-line release](https://registry.opendata.aws/dig-open-analysis-data/)
with the `mouse_2024` and `msigdb_nohp` libraries. It is not the earlier
six-library NAFLD qualification. The joint move cost about 3% more
sampling CPU at the same number of sweeps in the timing comparisons;
this does not measure the end-to-end production workflow.

Increasing independent SNP draws from 8 to 32 per trait raised the
designated split's annotation-only top-200 overlap from 142 to 160
(T2D) and 117 to 175 (schizophrenia). Alternative 16-versus-16 splits
had median overlaps of 165 and 162, respectively. The corrected
either-route estimator shared 198 of 200 in both designated splits,
but its pooled top list shared 184 and 197 genes, respectively, with
the exact SNP-only reference. The correction replaces each draw's
SNP-only baseline by its analytical expectation under the independent
SNP mask distribution; it does not change the model or improve the
indirect-only estimator. The weak-direct-evidence recovery benchmarks
did not show a consistent biological advantage.

For NAFLD, disjoint two-draw averages shared 96-104 of the top 200
indirect genes. The correction raised either-route overlap from 40-45
to 194-196; the pooled corrected list shared 195 genes with the SNP-only
reference. The short stock NAFLD reference was included as context only;
its native diagnostics measure different quantities and are not a
head-to-head comparison with the experimental sampler.

The evidence supports a faster-mixing transition for this conditional
model. It does not show that native PIGEAN's convergence issue is
fixed, that the indirect shortlist is stable, or that the model is
calibrated. The main modeling decision for review is whether annotation
effects should enter both the within-region assignment weights and the
trait-wide involvement prior in this way, and whether the assumed
coupling between the two routes is appropriate.
