# Methods-aligned logistic sampling: results and disposition

Retain the logistic reference, collapsed support updates, optional swaps and
validation tools as research components. **Do not promote the current outer
adapter.** No production source, defaults or remote main changed. The current
implementation does not demonstrate reliable full-trait estimation, a full-trait
speedup, or biological non-inferiority. Both favorable and unfavorable experiments
are retained. This investigation advances the logistic conditional beyond the
previous Gaussian-only reference; it does not resolve every outer heuristic.

## What was implemented

| Component | Separate control | Evidence / disposition |
|---|---|---|
| Logistic spike-and-slab conditional with fixed intercept and nuisance parameters | `--method coordinate` | Pólya–Gamma augmentation; original logistic-likelihood quadrature independently validates tiny cases. Retain as a reference, not a production baseline replacement |
| Collapse coefficient values when updating annotation support, conditional on augmentation variables | `--method collapsed` | Same specified logistic target; generally improves the competing-mode example, but alone misses the mean-error gate on one seed |
| Symmetric included/excluded annotation exchanges, followed by coefficient restoration | `--swaps N`, default 0; works with either research method | Exact acceptance for the augmented support target; combined method passes both competing-example seeds. Retain independently |
| Research adapter into the existing outer estimator | Separate `screen_logistic_outer.py`; `--method legacy` control | Full genes, reduced fixed annotation panel, original HuGE refresh. Slow and not demonstrated biologically non-inferior; do not promote |
| Biological comparison outputs | `compare_logistic_outer.py` | Correlation/slope, rank changes, top-set overlap and fixed-panel diagnostics. Review aid; not a calibration certificate |

The scientific mapping and explicit heuristic treatment are in
[the target specification](logistic_target.md). The augmentation uses the
published rejection algorithm, not a truncated sum or mean latent variable.
Its moment/variance/Laplace-transform checks, direct integration checks and
empirical implemented-kernel transition checks pass. Independent review found
no defect in offset units, production slab scaling or partial-collapse ordering.

The two new inference controls are additive: collapse changes the conditional
update strategy, while swaps can be enabled independently. Neither is a flag on
production `pigean gibbs`; they remain research commands until promotion is
justified. The production arithmetic recommendations remain unchanged.

## Known-answer logistic experiments

Four chains, 6,000 sweeps with 1,000 discarded, seeds 117 and 118, fixed nuisance
parameters. The challenging model has two identical annotation columns and two
single-annotation explanations with posterior probability .497044 each; the
both-included support has probability .005913. Initial chains occupy different
explanations. All algorithms use the same target. Timings below are warm sampler
time; reference calculation, cache/JIT loading and file writes are separate and
recorded. Fresh compiler cost was not benchmarked as a cross-method speed gain.

| Method | Seconds, seeds 117 / 118 | Support TV error | Maximum inclusion split R-hat | Maximum coefficient mean error |
|---|---|---|---|---|
| Coordinate | .831 / .803 | .24731 / .00156 | 8.431 / 11.548 | 1.05549 / .00171 |
| Collapsed | 1.241 / 1.214 | .01814 / .00136 | 1.01241 / 1.00313 | .07620 / .00731 |
| Collapsed plus one half-probability swap attempt | 1.422 / 1.417 | .00121 / .00071 | .99998 / 1.00017 | .00318 / .00284 |
| Longer coordinate control, 12,000 sweeps / 2,000 discarded | 1.694 / 1.633 | .05114 / .00134 | 1.44192 / 11.82637 | .21284 / .00244 |

The longer coordinate controls cost more than the combined method and still
fail chain agreement. Seed 118 also illustrates why aggregate agreement is
insufficient: opposing trapped chains average to approximately the right mean.
Collapse alone improves mixing but is not sufficient to meet all error gates on
both seeds. The combined method improves accuracy at the tested time budget;
there is no defensible universal fold improvement in time to convergence.

Easy logistic and synthetic null cases agree with direct quadrature in all
three configurations: support TV errors are below .006 and .0015 respectively.
On these easy problems, extra support work increases runtime without a useful
mixing improvement. The null example is a sampler/model check, not a simulated
GWAS passed through the complete HuGE and covariate pipeline.

Initial tensor quadrature did not resolve the challenging narrow ridge: the
192/256-node support difference was 1.06e-4. That initial case was correctly
marked unscored. The retained follow-up uses adaptive integration of the sum of
identical-column coefficients, with an independent tensor check on a milder
case and tolerance refinement on the challenging case. Initial and corrected
evidence directories are retained; no failed check was silently discarded.

## Reduced real-data screens

Every run uses the verified saved T2D/NAFLD fixture, all genes, a fixed panel of
12 annotations, four outer and four inner chains, seed 117 and 20 executed outer
updates. The driver receives an iteration limit of 21 because of the production
loop's initial-state convention. The initial panel uses the largest absolute
coefficients from the saved baseline setup. Both arms disable per-iteration
annotation removal and inner coefficient projection so panel membership stays
fixed. The production inner stopping rule remains active in the control;
the research fit uses 120 inner sweeps
with 40 discarded. These are comparisons of configured estimators, not equal
numbers of inner updates or estimates at matched accuracy.

Production HuGE competition, fixed signal probabilities, strongest-signal rule,
gene covariate correction, learned nuisance settings and mean-coefficient outer
feedback are retained. The full logistic candidate explicitly omits marginal
annotation coefficient/SE corrections, including dependence-based SE inflation.
Those quantities still feed the production uncorrected-output path; they do not
define the candidate's logistic likelihood. Differences cannot be attributed to
swaps alone. The T2D first conditional has bitwise-identical binary responses,
design, offset, inclusion probabilities and slab variances across arms.

| Trait / scope | Control / candidate inference seconds | Candidate slowdown | Peak RSS control / candidate | Interpretation |
|---|---|---|---|---|
| T2D initial panel | 5.783 / 144.234 | 24.94x | .903 / .872 GB | No demonstrated outer convergence gain; substantial biological ranking changes |
| NAFLD initial panel | 1.122 / 133.758 | 119.24x | .780 / .850 GB | Both effectively null for annotation effects; cannot assess original multimodality |
| NAFLD panel selected from full 500-update baseline | 1.115 / not run | Not measured | See control receipt | Largest control beta only .000522; failed signal-retention gate, so no candidate queued |

Times include profiling and compilation/load during inference, exclude fixture
preparation/export, and represent single sequential pairs. Peak RSS is process
memory, not an isolated kernel allocation. T2D profile assigns 121.55 of 144.23 s
to PG draws versus about 1.38 s to collapse and swaps together. Per-gene
augmentation repeated inside nested fits dominates this prototype.

T2D active-panel production beta R-hat q90 is 5.43 / 6.26. Panels differ (three
versus two active coefficients), so this is not a matched-panel worsening claim.
An additional fixed-12-annotation split diagnostic on the last ten outer mean
states has three / two values above 1.25 and zero / one undefined values. With
only five draws per split, neither establishes reliable estimation. Undefined
diagnostics are not treated as successful convergence. The candidate's inner
finite R-hat q90 is 1.060 and maximum 1.329; many sparse coordinates have no
variation. The short inner budget is not certified converged either.

### Output agreement

Each entry is Pearson correlation / candidate-on-control slope with an
intercept, aligned by identifier, using exported values. Annotation comparisons
include only the twelve fitted entries; missing/ignored outputs are excluded.
Rounded exports and rank ties limit the precision of ranking comparisons.

| Trait | Gene prior | Corrected beta | Uncorrected beta | Log BF | Combined gene score |
|---|---|---|---|---|---|
| T2D | .7245 / .4427 | .4385 / .3566 | .4600 / .3154 | .9927 / .9820 | .9412 / .8357 |
| NAFLD initial, near-null annotation panel | .6411 / .3985 | .8088 / .3324 | .9114 / .1947 | approximately 1 / 1 | approximately 1 / 1 |

T2D combined-score top-20/top-200 overlap is 65%/87.5%; prior top-200 overlap is
60%. Annotation top-200 overlap is trivially 100% on a shared twelve-entry panel
and is not evidence of agreement. NAFLD's combined-score agreement principally
reflects unchanged direct evidence when annotation effects are nearly absent.

### Biological review

T2D redistributes support rather than eliminating every familiar signal.
`MIR500A_3P` beta increases .0012 to .925, while
`Hepatocyte_Nuclear_Factor_1-beta` falls .912 to .0221 and Proinsulin .195 to
.000873. Incretins remains prominent, 1.23 to .824.

| Gene | Control combined-score rank | Candidate rank |
|---|---|---|
| WFS1 | 2 | 28 |
| HNF1A | 6 | 48 |
| PAX4 | 7 | 51 |
| GCK | 8 | 57 |
| KCNJ11 | 16 | 63 |
| ADCY5 | 1 | 1 |
| HNF1B | 4 | 5 |
| SLC30A8 | 5 | 6 |
| TCF7L2 | 11 | 13 |

These changes warrant expert review; they neither prove inferiority nor establish
that newly promoted genes/annotations are false positives. No agreed independent
positive/negative biological control list was supplied during this run. The
named examples are post hoc ranking observations, not a prespecified validation
set. Biological non-inferiority remains unestablished. Full GWAS null/permuted
annotation controls and full-panel trait comparisons were not run because the
prototype failed the cost gate and reduced NAFLD failed signal retention.

## Decision and next boundary

Keep the two modular sampling components and independent references. Keep the
outer adapter as an explicitly unsuccessful/inconclusive feasibility experiment,
not as a recommended flag. No full 500-update job is justified by these results.
Before another production candidate, specify how practical marginal corrections
should enter the intended joint logistic inference (or an explicitly calibrated
approximation), and address the repeated PG cost. A representative NAFLD test
must retain the original non-null support problem. Merely increasing this inner
budget or swapping more often does not address these issues.

`docs/methods.tex` now has an evidence-grounded Performance optimizations section
and an explicit Sampling improvements placeholder. Source/brace/label checks
passed; a rendered LaTeX build was not verified, and the old `methods.pdf` was
not updated. Native LaTeX editor/compiler tools were not available in this session.

## Reproduction and provenance

Run from this code checkout using the shared analysis venv. All result paths
below are relative to the analysis project; refuse overwrite and run sequentially.

```sh
bash scripts/research/validate_logistic.sh
../../../.venv/bin/python scripts/research/benchmark_logistic.py --out ../../results/NEW_REFERENCE
../../../.venv/bin/python scripts/research/benchmark_logistic.py --out ../../results/NEW_TIME_CONTROL --case competing --methods coordinate --sweeps 12000 --burn 2000
../../../.venv/bin/python scripts/research/screen_logistic_outer.py --fixture ../../results/feature_foundation_20261011/fixtures/T2D/fixture.pkl.gz --out ../../results/NEW_T2D_CONTROL --method legacy --profile
../../../.venv/bin/python scripts/research/screen_logistic_outer.py --fixture ../../results/feature_foundation_20261011/fixtures/T2D/fixture.pkl.gz --out ../../results/NEW_T2D_CANDIDATE --method collapsed --swaps 6 --profile
../../../.venv/bin/python scripts/research/compare_logistic_outer.py --baseline ../../results/NEW_T2D_CONTROL --candidate ../../results/NEW_T2D_CANDIDATE --out ../../results/NEW_COMPARISON
```

Run receipts, profiles, raw retained reference draws, first-condition captures,
complete compressed exports and review tables are under analysis
`results/methods_aligned_sampling_20261013/`. Compact compressed evidence is
copied beside this document under `logistic_sampling_evidence/`. Core tests plus
existing production default/feature checks passed: 297 tests. A first combined
test invocation lacked the source import path; the committed validation wrapper
sets it explicitly. Five real-data jobs completed sequentially; no sampler jobs
remain. The uninformative second NAFLD control did not trigger a sixth job.
