# Choosing factor resolution in the research workflow

Start with automatic phi selection and the default **two-percentage-point error allowance**. Phi controls how strongly fitting regularizes the factors. The allowance controls how much detail the selector may give up in exchange for fewer leaves. If the best eligible fit has 6.5% normalized relationship error, an allowance of 2 admits eligible fits through 8.5%. The selector picks the fewest fitted leaves in that range, breaking ties by lower error and then lower phi.

## Recommended workflow

1. Supply the captured discovery inputs and the qualified fits for the same trait. The four-trait research grid used phi **0.005, 0.01, 0.02 and 0.04**, with five starts at each value and a joint consensus refit. The selector scores the candidate directories supplied on the command line. Fitting an additional phi is a separate operation.
2. Run the selector with its defaults. It flags individual leaves first. A candidate is eligible when at most 10% of its leaves and at most 10% of its squared loading mass are flagged. Relationship error is the larger of the normalized error across gene pairs and the mean normalized neighborhood error across genes with signal.
3. Read `selection.json` for the chosen phi, error and policy. Compare candidates in `phi_summary.tsv.gz` and inspect the winning leaves in `selected_factor_qc.tsv.gz`. If no candidate qualifies, the decision records `no_usable_candidate`; inspect the per-phi QC tables before changing a threshold.
4. For the hierarchy, take the passing columns of `W` in `selected_fit.npz`. Keep the complete fit and QC table alongside the tree so flagged leaves remain inspectable. Gene and annotation labels are in `inputs/metadata.npz` under the matched capture. Its `genes` array indexes rows of `W`, and `gene_sets` indexes rows of `H`. QC field `factor` is the one-based exported column number; `original_consensus_factor` equals the corresponding `original_factor_indices` value plus one. Factor numbers are local to each trait and fit.

| Qualified research leaves | Selected phi | Pass / fitted |
|---|---:|---:|
| T2D | 0.02 | 84 / 84 |
| CAD | 0.01 | 74 / 75 |
| NAFLD | 0.01 | 35 / 36 |
| Asthma | 0.04 | 42 / 42 |

## What to try, in order

| Priority | When to use it | Control and expected effect |
|---|---|---|
| 1 | The result looks too coarse or too fragmented | Keep automatic selection. Try `--phi-error-allowance-points 1` for finer resolution. A larger value permits a coarser inventory; 5 is a useful sensitivity check. Reuses the saved fits. |
| 2 | The winner is at an edge and the available fits do not cover the resolution you need | Fit one adjacent phi (halve the low end or double the high end), include that directory in `--candidates`, and rescore. A wider grid costs additional fits. An interior winner ordinarily ends the search. |
| 3 | A comparison calls for a specific regularization strength, or you need one known resolution for a sensitivity analysis | Supply just the matching qualified candidate directory. It still has to pass QC. With one candidate there is no across-phi fidelity comparison. In the released Main fitting CLI, explicit `--phi VALUE` fixes phi, and `--no-learn-phi` uses fixed 0.01 in fresh gene-by-gene discovery. A common phi can still yield different numbers of leaves across traits. |
| 4 | A *specific kind* of questionable leaf recurs | Inspect the per-factor reasons and loadings. Adjust only the screen that measures the problem (size, contrast, duplication or restart support). Use the same policy for every candidate. Changes to these settings can alter individual flags even if they leave the winning phi unchanged. |
| Expert | No candidates pass, independent ensembles disagree, or the factorization itself is being studied | Inspect which screen failed and how much loading mass it affects. Evaluate extra independent starts or ensembles before relaxing the evidence requirements. Changes to consensus construction, fitting capacity or the model require refitting and separate validation. |

From the branch root, run the research selector in a Python environment with NumPy, SciPy, pandas and Numba. Set `study` to the existing CAD study directory on your machine. Each selection needs a fresh output directory. Omitting the allowance uses the same default shown here.

```bash
study=/path/to/analysis/pigean_optimize/results/EAGGL_four_trait_leaf_validation/CAD
python research/eaggl_leaf_selection/scripts/select_research_phi_with_qc.py \
  --inputs "$study/main" \
  --candidates "$study"/research/candidates/{0.005,0.01,0.02,0.04} \
  --output "$study/selection_example" \
  --phi-error-allowance-points 2
```

## Complete selector controls

Ordinary `--help` shows the resolution control. Use `--help-expert` to see all QC and replication options. Advanced flags can be passed directly.

| Level | Option | Default | Meaning |
|---|---|---:|---|
| Routine | `--phi-error-allowance-points` | 2 | Absolute percentage points above the best eligible error. Enter 2 for two points. |
| Targeted QC | `--qc-minimum-effective-genes` | 3 | Flag a factor dominated by too few genes. Raising the minimum is stricter. |
| Targeted QC | `--qc-minimum-contrast` | 0.20 | Flag near-uniform loadings using Hoyer contrast. Raising the minimum is stricter. |
| Targeted QC | `--qc-duplicate-cosine` | 0.90 | Flag both factors in a highly similar pair. Lowering this value flags more pairs. |
| Targeted QC | `--qc-minimum-support` | 3 of 5 | Number of raw starts that must match a final leaf. Four is a useful stricter reproducibility check. |
| Expert QC | `--qc-support-cosine` | 0.70 | Similarity needed for a raw start to count as support for the refitted leaf. |
| Expert QC | `--qc-maximum-flagged-fraction` | 0.10 | Maximum flagged fraction of leaves allowed for a phi to be eligible. |
| Expert QC | `--qc-maximum-flagged-energy` | 0.10 | Maximum fraction of squared loading mass on flagged leaves. Both eligibility limits must pass. |
| Expert replication | `--qc-replication-cosine` | 0.90 | Minimum similarity of matched pair-contribution profiles across independent ensembles. |
| Expert replication | `--qc-replication-amplitude-min`, `--qc-replication-amplitude-max` | 0.80, 1.25 | Both directional amplitude ratios must fall inside these bounds. |

| Input, output and help | Requirement | Argument |
|---|---|---|
| `--inputs` | Required | Matched Main discovery capture directory used by the research fits |
| `--candidates` | Required | One or more qualified candidate directories named by phi, with the same input and model contract |
| `--output` | Required | New directory for the decision, per-phi QC, selected fit and its flags |
| `--replicates` | Optional | A second qualified directory for every candidate phi, with independent seeds. Omitting it records independent replication as unassessed. |
| `-h`, `--help` | Optional | Routine usage and resolution control |
| `--help-expert` | Optional | Full argument list, including every advanced default |

The decision records the effective policy, with the allowance stored as a fraction (`epsilon=0.02`).

## Expert fitting settings

The research fits used 120 starting slots, five independent starts, pooled matching at cosine 0.70 with support in three starts, median aggregation, and joint refitting under the same union objective. The post-refit QC thresholds above can be changed on saved fits. Changing how the consensus is **built** requires generating new fits. Use another master seed for independent replication. Revisit capacity if fits keep hitting the limit. Solver stopping criteria, scientific ARD scale, loading bounds, union link, additional penalties and projection rules belong to method development.

Released Main uses a different selector. Its fitting controls include `--learn-phi-values`, `--phi-search-extension`, `--phi-search-max-extra`, `--max-num-factors`, `--factor-runs`, `--consensus-matching`, `--consensus-min-factor-cosine`, `--consensus-min-run-support`, `--consensus-aggregation` and `--no-consensus-nmf`. Its qualified automatic workflow restricts custom fitting recipes. The points and QC options above belong to the research selector. [Main workflow and supported overrides](../../docs/eaggl/RECOMMENDED_DEFAULTS.md)

In the four-trait qualification, one point gave finer resolution in every trait. Five points weakened recognizable NAFLD substructure. None of the tested structural QC threshold changes moved the selected phi. Four traits support this ordering for troubleshooting; performance on arbitrary new traits remains to be measured. [Validated leaf inputs and SHA-256 checksums](hierarchy_inputs.json)

## Build a mechanism hierarchy

From the package root, point the command at the analysis directory that contains the files listed in the manifest. Run once per trait (T2D, CAD, NAFLD or Asthma). The command verifies the recorded hashes and builds one tree per trait from the selected, jointly refitted consensus. Flagged leaves remain available in the original factor and QC tables.

```bash
python research/eaggl_leaf_selection/scripts/build_factor_hierarchy.py \
  --input-root /path/to/analysis/pigean_optimize --trait CAD \
  --output /path/to/new/CAD_hierarchy \
  --min-leaf-cosine 0.30 0.50 0.70 --min-shared-mass 0.50
```

Both cut options are optional. The primary flat factors are the passing leaves. A cut picks the broadest nodes on each branch whose least similar pair of descendant leaves meets the requested floor. Every passing leaf appears exactly once. The binary tree uses complete linkage on cosine distances of the original discovery profiles. Input columns are ordered by exported leaf ID before tie resolution. Biological labels are used only during review.

For gene membership, a parent of nodes A and B has loading `A + (1 − A) × B` for each gene, the heuristic union under independence. A parent represents membership through any descendant. The same-leaf gene-pair score still uses the original union of descendant pair contributions. An internal node can span distinct submechanisms, and the root summarizes the trait.

| Output | Contents |
|---|---|
| `factor_hierarchy.tsv.gz` | Fit-local zero-based node and parent IDs, children, one-based source leaf IDs, original consensus IDs, number of descendants, minimum cosine and common loading mass, effective number of genes and shared-leaf pair fraction. The root has an empty parent field. |
| `hierarchy_loadings.npz` | Gene-by-node matrix W with explicit gene and node axes. Leaf columns copy the original weights. |
| `hierarchy_cuts.tsv.gz` | Chosen nodes and their source leaf IDs at each requested cut. |
| `hierarchy.json` | Trait, phi, method version, source hashes, supplied cut settings and excluded flagged leaf IDs. |

**Overlap in gene usage.** To compare two leaves, scale each to one unit of total loading and sum the smaller of their two weights at each gene. A value of 0.50 means half their loading mass can be assigned to the same genes, after adjusting for overall strength. The optional shared-mass floor cuts the existing cosine tree using its least-overlapping pair of descendants. A ceiling on overlap **between displayed families** is a different criterion. Two families can share genes even when they differ in their overall makeup. Neither measure estimates the probability that two factors have the same underlying mechanism.

Effective gene count uses squared loading mass, as in leaf QC. The shared-leaf pair fraction measures the off-diagonal mass explained by shared descendants divided by the mass obtained by multiplying the two parent memberships. It is missing when the denominator is zero. Outputs directory creation is atomic and requires a fresh destination.

On the pinned T2D, CAD, NAFLD and Asthma fits, the 0.50 cosine cut produced 53, 51, 20 and 28 displayed families. The same trees cut at 50% minimum shared gene mass gave 77, 68, 26 and 36. CAD cardiac-development and Asthma immune-signaling pairs regrouped in both archived masked fits; broader metabolic and immune families had more variable boundaries. Use 0.50 cosine as an exploratory view. Run `scripts/assess_four_trait_hierarchies.py --analysis-root PATH` to describe the four saved consensus trees. The historical masked-fit and per-start tree comparisons were exploratory and remain in the original study archive.

### Inspecting one join

Use the optional command below to examine a specific join on the saved tree. Supply the directory from the build command and its fit-local internal node ID. The command checks the fit and raw-start receipts and reproduces the saved support counts across all consensus leaves before inspecting the join.

```bash
python research/eaggl_leaf_selection/scripts/inspect_hierarchy_join.py \
  --input-root /path/to/analysis/pigean_optimize --hierarchy /path/to/new/CAD_hierarchy \
  --node 80 --output /path/to/CAD_join.tsv.gz
```

Each row represents one start and one pair of leaves on opposite sides of the join. Compare their consensus cosine and shared loading mass with those of the two matched raw profiles. Missing either match leaves the corresponding similarity fields blank. Count the starts with both matches when reporting a range. The five starts reuse the same input relationships. Agreement measures sensitivity to initialization. A pair can match in every start and still vary in how strongly its members resemble each other. These observations annotate the interpretation of the consensus join without changing its membership or adding a second QC stage.

### Choosing a flat view

Use the fitted leaves to see the most detail. For a coarser overview, cosine 0.50 is the current research reference. Use `--min-shared-mass 0.40` if you want each pair of descendant leaves to share at least 40% of its normalized loading mass. On the four saved traits that floor sometimes separates leaves with a recognizable common core. Two families displayed separately can share genes.

For a size-limited review, use `--max-gene-equivalents 30` with the hierarchy build command. A ceiling of 30 means that the summed memberships of the discovery genes in a displayed family can reach 30. A weight of 0.2 contributes 0.2. The output `hierarchy_size_cuts.tsv.gz` identifies an indivisible leaf above the requested ceiling. This option can group small, biologically distinct profiles while splitting large families. Interpret it as a way to limit the breadth of a display. The original fit, tree and flat leaf output are common to all of these views.
