# SNP factor weights and PheWAS

This optional EAGGL stage maps PIGEAN signals through gene effectors to continuous factor loadings. Weight construction, file export, and phenotype association are separate operations. Existing gene-based PheWAS and projection outputs are unchanged.

## Quick start

Use the Python interpreter configured for your environment. These examples use `eaggl` and `pigean` entry points; `python -m eaggl` and `python -m pigean` are equivalent when the package is installed. For a source checkout, ensure its `src` directory is on the module search path before using those module entry points.

Metadata collection and additional effector columns require `--snp-factor-metadata`; omitting it preserves legacy behavior. First generate allele-resolved PIGEAN effectors, retaining the build and new conditional allocation weight:

```sh
pigean huge --snp-factor-metadata --gwas-in discovery.tsv.gz --gene-loc-file-huge genes.loc \
  --gwas-build GRCh37 --gwas-effect-allele-col ALT --gwas-other-allele-col REF \
  --gene-effectors-out effectors.tsv.gz --huge-statistics-out huge.tar.gz
```

Generate reusable unsigned weights from saved EAGGL gene factors, without reading target traits:

```sh
eaggl factor --factor-gene-clusters-in gene_clusters.tsv.gz \
  --snp-factor-gene-effectors-in effectors.tsv.gz \
  --snp-factor-weights-out factor_weights.tsv.gz
```

The same output arguments can be appended to an ordinary EAGGL factor-fitting run. Supply the effector input alongside your existing factor inputs. PIGEAN's new `signal_gene_weight` and allele metadata survive HuGE cache reloads.

Reload weights and test local GWAS files plus DIG traits:

```sh
eaggl factor --snp-factor-weights-in factor_weights.tsv.gz \
  --run-snp-factor-phewas \
  --snp-factor-phewas-gwas-in target.tsv.gz \
  --snp-factor-phewas-gwas-list targets.tsv \
  --snp-factor-phewas-dig-id T2D \
  --snp-factor-phewas-stats-out factor_phewas.tsv.gz
```

Include all DIG phenotypes using `--snp-factor-phewas-dig-all`. Ancestry defaults to `Mixed`; repeat `--snp-factor-phewas-dig-ancestry` to include multiple ancestries. All selectors are additive, and identical source/ancestry pairs are deduplicated. DIG identifiers may be explicitly qualified as `dig:Mixed:T2D`.

To fit factors, construct weights, and run PheWAS in one invocation, append `--run-snp-factor-phewas`, target selectors, and the association output to your usual EAGGL command. Weight exports are optional and do not change the association calculations.

## Sign-only and effect-weighted scores

No anchor means a direction-free quadratic association test plus unsigned evidence regressions. Factor membership is not an allele's signed biological effect.

`--snp-factor-sign-anchor SOURCE` uses only the signs of finite nonzero anchor effects. Changing their magnitudes does not change the weights. `--snp-factor-effect-weight-anchor SOURCE` explicitly uses signed effect magnitudes instead. The arguments are mutually exclusive and neither filters by anchor significance. Sources can be local GWAS files or `dig:ANCESTRY:ID`.

Signed scores require usable allele frequencies to define standardized genotypes. A supplied LD reference is preferred; effector or anchor frequencies are accepted with explicit reference-approximation provenance. Palindromic alleles require frequency-based resolution. Missing anchor directions count against coverage.

```sh
eaggl factor --factor-gene-clusters-in gene_clusters.tsv.gz \
  --snp-factor-gene-effectors-in effectors.tsv.gz \
  --snp-factor-sign-anchor anchor.tsv.gz \
  --snp-factor-ld-bfile reference_EUR \
  --snp-factor-weights-out signed_weights.tsv.gz \
  --snp-factor-weights-plink-out scores.plink.tsv.gz \
  --snp-factor-weights-pgs-out scores
```

PLINK exports contain `ID`, `A1`, then one dosage-coefficient column per factor. Select columns 3 onward and request score sums, for example `--score scores.plink.tsv.gz 1 2 header-read cols=+scoresums --score-col-nums 3-4` for two factors. The JSON companion records centering offsets: internal centered score = external dosage-weight sum + offset. Do not variance-standardize dosage again. Scale by the relevant reference score SD for per-SD reporting.

PGS-style exports contain chromosome, position, effect allele, other allele, and dosage coefficient. Their manifest identifies each factor and centering offset. These are research scores, not PGS Catalog registrations. Both standard linear exports require signed weights and a documented genome build. Unsigned membership weights are available through native export only.

Native weights preserve a complete factor roster, sparse membership, coefficients, allele frequencies, build, anchor, effector sources, and factor-basis provenance. Reloaded weights are authoritative; reconstruction options cannot be combined with `--snp-factor-weights-in`.

## Methods and units

Membership is `a(signal,factor) = sum_gene q(signal,gene) W(gene,factor)`. The recommended `signal_gene_weight` averages gene linkage allocations over conditional variant localization, before posterior-odds conversion. It excludes the signal-existence probability and gene priors. Unassigned mass remains unassigned. An expert can select another effector column using `--snp-factor-gene-weight`. Legacy `cond_prob_huge` has a different, normalized-evidence interpretation and produces a warning. Legacy position-only leads must first be annotated with allele-resolved identities.

Both marginal and joint regressions run by default. `--snp-factor-phewas-regression-scope marginal` or `joint` restricts the computation. The joint roster contains all supplied factors; it is not selected by target significance.

### Signed scores

For standardized-genotype weights U and selected-variant LD R, the score Gram matrix is G = U' R U. Target effects and SEs are transformed to this genotype scale. The numerator is m = U' b and its covariance is V = U' D R D U.

Marginal coefficients are m[k]/sqrt(G[k,k]). Joint coefficients are H solve(G,m), where H contains the score SDs. Their covariance is H solve(G,V) solve(G,H). Effects are target units per reference-population score SD. Odds ratios are output only when a manifest documents log-odds units. The summary covariance assumes compatible samples and reference LD; per-SNP sample-size ranges and uncertain overlap are recorded.

### Unsigned tests and regressions

The quadratic test Q = sum a*z^2 has a weighted chi-square null. Signed LD supports eigenvalue-based saddlepoint evaluation, with analytic equal-weight and single-eigenvalue cases. R-squared-only input uses a labeled moment approximation. Failed numerical tails are missing. Upper tails are evaluated in log space; a P value below floating-point range is reported with finite `log_p` (natural logarithm), `neg_log10_p`, and `p_underflow=True`. This preserves the existing saddlepoint approximation, rather than substituting a minimum P value. Multiple-testing output includes `log_q`, `log_bonferroni`, `q_underflow`, and `bonferroni_underflow`; correction uses log probabilities so extreme tails remain distinguishable. Unavailable tests remain in the family denominator but receive no corrected primary P value.

Separate evidence regressions fit z^2 - 1 against an intercept and factor membership. Marginal fits use one factor; the joint fit uses all factors on the same selected-variant union. GLS uses working null covariance 2*(R squared elementwise); delete-one-LD-block jackknife supplies uncertainty. Coefficients measure excess chi-square evidence per membership unit among the selected loci. They can be negative. This new descriptive regression is not a signed phenotype effect, genome-wide competitive enrichment, or stratified heritability estimator. It remains power-dependent.

Rank-deficient or ill-conditioned systems do not receive arbitrary ridge or pseudoinverse coefficients. Identifiable coordinates can still be reported; others have `not_identifiable` status. Rank-changing jackknife deletions yield `jackknife_unavailable`. Numerical condition limits are 1e10, with the matrix scale recorded through condition diagnostics. Single-locus or low-block results need sensitivity interpretation even when a quadratic P value is available.

## LD and ranked proxies

Use `--snp-factor-ld-bfile PREFIX` for variant-major BED/BIM/FAM, or `--snp-factor-ld-in FILE` for a pair table, matrix, or PLINK clump report. `--snp-factor-ld-variants-in FILE` supplies a BIM or TSV mapping for rsIDs. Mapping TSV columns are `id`, `chrom`, `pos`, `ref`, `alt`, and optional `frequency`. Alleles define the dosage orientation of signed correlations. Signed pair-table r values must correspond to those allele definitions; r-squared alone never establishes phase.

Canonical pair tables have `index`, `proxy`, and `r` and/or `r2`, optionally `rank`. PLINK `SNP_A/SNP_B/R/R2` and `ID_A/ID_B/UNPHASED_R/UNPHASED_R2` tables are also recognized. Square matrices use an adjacent `.vars` (or `.ids`) list. Legacy matrices require explicit `--snp-factor-ld-matrix-kind r` or `r2`; `.vcor1`/`.vcor2` filenames determine the type automatically. Document the reference build with `--snp-factor-ld-build`; a mismatch with the weights is rejected. PLINK `.vcor1` is signed and `.vcor2` is squared LD. Binary matrices are little-endian float64; other encodings must be converted before use. Sparse tables should represent all relevant within-block correlations; omitted pairs are assumed independent and external-reference calculations are explicitly labeled approximate.

The original QC-passing SNP takes priority. Otherwise proxies are ranked by decreasing r-squared, then distance and ID, without using target P values. Defaults are r-squared >=0.8 and <=1 Mb; override with `--snp-factor-proxy-r2` and `--snp-factor-proxy-window`. Clump membership without correlation supplies an explicitly unknown-quality ranking only. An unsigned substitution retains the membership weight. A signed substitution transfers standardized coefficients using signed r. Reused proxies are collapsed before covariance calculation, including in joint regressions.

European is the default reference ancestry and Mixed use is flagged. Override with `--snp-factor-ld-ancestry`. The stage never downloads reference panels automatically or reclumps HuGE signals. Without LD, independence-based estimates are approximate. BED readers retain metadata only for requested windows, use a bounded genotype-vector cache, and compute original-to-proxy links without materializing all candidate-pair correlations. Covariance is calculated lazily for selected target variants. Use regional LD inputs if an irreducible selected block cannot fit.

## Target manifests, QC, and bounded access

A list file contains one source per line. A TSV manifest starts with `source` and may add `trait`, `ancestry`, `build`, `units`, `study`, `n`, `index`, and field overrides such as `chrom_col`, `pos_col`, `p_col`, `beta_col`, `se_col`, `alt_col`, `ref_col`, `freq_col`, `info_col`, or `id_col`. Selectors accept column names or one-based indices. Existing GWAS column selectors are inherited when not overridden by a manifest. Unit values `log_odds`, `logor`, or `log(OR)` permit odds-ratio output; input effects must already be log odds, not ORs.

QC validates identity, build when documented, duplicate records, alleles, and usable statistics. Negative SEs are not repaired. Valid P values can support unsigned tests despite unusable effect fields. Position-only target rows can support unsigned analysis only when a requested coordinate has a unique candidate identity; allele verification is then explicitly unavailable, and such rows cannot support signed testing.

Default MAF and INFO thresholds are 0.01 and 0.8 where present. Missing fields are recorded as unassessed. Target significance is never a QC filter. Coverage below 90% suppresses primary estimates while retaining diagnostics; a joint result cannot claim complete adjustment if any factor fails coverage. Override with `--snp-factor-min-coverage`.

Weight construction precedes target access. Files with usable tabix indexes are queried at requested coordinates. The optional pysam package or a native tabix executable provides indexed access. Remote adjacent indexes are discovered by HEAD requests; an installed DIG indexed extension can resolve companion files from a manifest. Missing/unusable random access produces a warning and a single streaming scan. Only selected records enter a temporary SQLite store, and it is deleted afterward.

`--snp-factor-memory-mb` defaults to 512 MiB and `--snp-factor-trait-batch-size` to 32. The stage reduces the trait batch according to SNP and factor-Gram working sets and processes LD by components. Matrices are vectorized across compatible trait masks. An irreducible LD or genotype working set exceeding budget fails explicitly rather than exceeding the bound.

Association outputs include method, scope, beta/SE/statistic/P, status, coverage, LD calibration, and joint roster. Quadratic statistics remain separate from evidence regressions. Sensitivity columns report strongest-locus contribution and leave-one-locus-out changes. Companion files record resolved trait metadata, proxy substitutions, and run provenance. Benjamini-Hochberg and Bonferroni corrections use separate method/scope/calibration/dependence families and retain failed tests in each family's denominator where the family is known.

Use repeated `--snp-factor-training-source` identifiers to record factor-training overlap. Identical anchor/target sources and declared training matches are labeled discovery-dependent; otherwise overlap is unknown, not asserted absent. These calculations are research associations, with approximation and power dependence visible in the outputs.

## DIG deployment handoff

See `contrib/dig-open-data/README.md` and its additive indexed module and companion-preparation tool. The PIGEAN stage works through streaming before the separate DIG library/S3 rollout. No S3 objects are modified by this implementation.


## Method and format references

- [PLINK scoring and score sums](https://www.cog-genomics.org/plink/2.0/score).
- [PLINK LD correlation formats](https://www.cog-genomics.org/plink/2.0/ld).
- [PGS Catalog scoring-file columns](https://www.pgscatalog.org/downloads/).
- [HTSlib remote tabix access](https://www.htslib.org/doc/tabix.html).
- [PascalX weighted chi-square methods](https://academic.oup.com/bioinformatics/article/39/5/btad296/7151067).

## Validation and deployment limits

The regression tests cover native round trips, weight-only isolation, saved-weight export, full EAGGL fitting with export on/off, signed coefficients against individual-level least squares, correlated unsigned GLS against a separate dense calculation, delete-block uncertainty, allele recoding, shared proxies, BED decoding, streaming memory, and method/scope correction families. The unsigned saddlepoint is checked against independent Imhof inversion and seeded null simulations. The evidence-regression extension remains descriptive; these simulations do not establish universal calibration under heterogeneous GWAS ascertainment or sample overlap.

Live HTSlib indexed/streaming parity is conditional on a usable native tabix/bgzip installation. The pysam and DIG region adapters are also tested with controlled fixtures. Live indexed S3 testing and the external DIG companion publication require the separate deployment described above. No dependency is installed automatically.

The September 2026 diagnostic validation directly checked 20 moderate-tail cases against [CompQuadForm's Davies implementation](https://github.com/cran/CompQuadForm/blob/master/src/qfc.cpp), compiled for validation with only R-specific includes replaced. All Davies fault codes were zero; the maximum saddlepoint absolute error was 0.00476 across four spectra. Portable tests retain selected Davies reference values and independent Imhof/null checks. Analytic extreme-tail tests and 60-digit evaluations of the LR formula verify underflow handling; they do not certify exact extreme mixture-tail probabilities. PLINK export is verified algebraically against dosage scores; no PLINK executable is invoked by those tests.

## Joint models with coverage-selected factors

The default `--snp-factor-joint-factor-policy all` retains the original complete-roster requirement. With `--snp-factor-joint-factor-policy covered`, the joint model includes exactly those factors whose represented SNP weight passes `--snp-factor-min-coverage` for that trait. For example, use `--snp-factor-min-coverage 0.85 --snp-factor-joint-factor-policy covered` for an explicitly labeled sensitivity analysis. Selection never uses target P values. Factors excluded for coverage remain in the output, with unavailable joint estimates.

Marginal and joint evidence regressions retain the same target-specific SNP rows and an intercept. Removing a factor removes its design column, not the SNPs associated with that factor. Joint coefficients and jackknife uncertainty are recomputed from the reduced simultaneous model. Rank problems do not trigger further silent factor removal. If no factors pass coverage, no joint factor model is reported.

Outputs record `joint_factor_count`, `joint_factors`, `joint_full_roster`, `original_factor_count`, `joint_factor_policy`, `minimum_coverage`, and `n_selected_variants`. The joint adjustment set can differ across traits, so conditional coefficients should be interpreted alongside that roster. `analysis_label` distinguishes marginal evidence regression, joint evidence regression and the separate quadratic evidence test. Only the evidence regressions have a fitted intercept, which absorbs a common baseline but is not a guarantee of ascertainment-bias correction.

Current-main port: metadata collection is opt-in with `--snp-factor-metadata`; `--no-snp-factor-metadata` explicitly selects legacy effector columns. The port preserves sample-size and p-source fields added upstream.
