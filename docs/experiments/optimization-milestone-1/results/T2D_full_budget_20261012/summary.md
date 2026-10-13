# T2D matched 500-budget validation

Review: **User-accepted first optimization milestone.** Gibbs inference improved 6.94x and peak RSS fell 40.4%. The original 10x and strict numerical gates were unmet; the user accepted the practical gain and broad agreement and authorized commit/tagging. Statistical equivalence remains unquantified, and both runs remain non-converged by beta R-hat. Inner fits occupy 66% of optimized inference. Legacy remains the default.

One seed; ten outer/four inner chains. Completion is not convergence. Times start from a saved pre-Gibbs state and exclude upstream input construction.

| Measure | Legacy | Optimized | Change |
|---|---:|---:|---:|
| Gibbs inference (s) | 6201.423 | 893.261 | 6.942x |
| Cold compiler preparation (s) | 0.000 | 2.399 | 0.000x |
| Fixture-to-output elapsed (s) | 6215.168 | 909.348 | 6.835x |
| Inference CPU (s) | 6179.919 | 919.233 | 6.723x |
| Inner fits (s) | 554.976 | 589.629 | 0.941x |
| Peak process RSS (MiB) | 2538.4 | 1512.0 | -40.4% |

Numerical agreement, sparse trajectory checkpoints and final diagnostics are recorded in [comparison.json](comparison.json). No automatic promotion, default change, push or additional trait run follows.

## Final output agreement

```json
{
  "prior": {
    "n_union": 19746,
    "n_finite": 19746,
    "n_missing": 0,
    "n_shared_missing": 0,
    "n_unmatched_missing": 0,
    "n_invalid_nonfinite": 0,
    "same_universe": true,
    "exact": false,
    "valid": true,
    "pearson": 0.9870348447667523,
    "spearman": 0.9611695035429192,
    "slope_zero": 0.9806831687312247,
    "slope": 0.9792485146852482,
    "intercept": -0.003021693922102012,
    "max_abs_error": 0.979,
    "mean_abs_error": 0.031584562309834904,
    "strict_pass": false,
    "top200_union": {
      "n_union": 212,
      "n_finite": 212,
      "n_missing": 0,
      "n_shared_missing": 0,
      "n_unmatched_missing": 0,
      "n_invalid_nonfinite": 0,
      "same_universe": true,
      "exact": false,
      "valid": true,
      "pearson": 0.9893418744772328,
      "spearman": 0.9705814374651502,
      "slope_zero": 0.9887832696313145,
      "slope": 0.9773949355270106,
      "intercept": 0.03943907110716561,
      "max_abs_error": 0.6599999999999999,
      "mean_abs_error": 0.13358490566037737,
      "strict_pass": false
    }
  },
  "beta_uncorrected": {
    "n_union": 299344,
    "n_finite": 4912,
    "n_missing": 294432,
    "n_shared_missing": 294432,
    "n_unmatched_missing": 0,
    "n_invalid_nonfinite": 0,
    "same_universe": true,
    "exact": false,
    "valid": true,
    "pearson": 0.9964588963634502,
    "spearman": 0.960157455792456,
    "slope_zero": 0.9949778894419645,
    "slope": 0.9905265142062806,
    "intercept": 0.00566306237284092,
    "max_abs_error": 0.49978,
    "mean_abs_error": 0.013554471457654723,
    "strict_pass": false,
    "top200_union": {
      "n_union": 208,
      "n_finite": 208,
      "n_missing": 0,
      "n_shared_missing": 0,
      "n_unmatched_missing": 0,
      "n_invalid_nonfinite": 0,
      "same_universe": true,
      "exact": false,
      "valid": true,
      "pearson": 0.9928042910186121,
      "spearman": 0.9795799745859302,
      "slope_zero": 0.9938555329212825,
      "slope": 0.9717562133785911,
      "intercept": 0.041953242861802895,
      "max_abs_error": 0.20999999999999996,
      "mean_abs_error": 0.03572115384615385,
      "strict_pass": false
    }
  },
  "beta": {
    "n_union": 299344,
    "n_finite": 4912,
    "n_missing": 294432,
    "n_shared_missing": 294432,
    "n_unmatched_missing": 0,
    "n_invalid_nonfinite": 0,
    "same_universe": true,
    "exact": false,
    "valid": true,
    "pearson": 0.8756788898484842,
    "spearman": 0.9209559396718039,
    "slope_zero": 0.8595098117525566,
    "slope": 0.857278424627192,
    "intercept": 0.0004970121148289006,
    "max_abs_error": 0.40234,
    "mean_abs_error": 0.0013430815798045601,
    "strict_pass": false,
    "top200_union": {
      "n_union": 247,
      "n_finite": 247,
      "n_missing": 0,
      "n_shared_missing": 0,
      "n_unmatched_missing": 0,
      "n_invalid_nonfinite": 0,
      "same_universe": true,
      "exact": false,
      "valid": true,
      "pearson": 0.8475974988734154,
      "spearman": 0.4903899849836767,
      "slope_zero": 0.859470117454286,
      "slope": 0.8188040906326008,
      "intercept": 0.010329255583176825,
      "max_abs_error": 0.40234,
      "mean_abs_error": 0.023115688259109313,
      "strict_pass": false
    }
  }
}
```

## Latest post-burn diagnostics

```json
[
  {
    "epoch_iter_num": 500,
    "total_iter_num": 500,
    "stop_mcse_quantile": 0.9,
    "beta_rhat_q_post": 1.678687121324981,
    "beta_ratio_q": 0.09456770047863812,
    "max_rel_mcse_beta": 0.2,
    "beta_rel_mcse_denom_floor": 0.1,
    "top_gene_k": 200,
    "stop_min_gene_d": 0.3,
    "num_monitored_genes": 200,
    "num_eligible_genes": 341,
    "D_mcse_q": 0.05610847211934571,
    "max_abs_mcse_d": 0.1,
    "max_post_beta_rhat": 1.25,
    "prior_beta_rel_inconsistency_q": 0.004787442270563457,
    "max_rel_prior_beta_inconsistency": 0.5,
    "num_active_betas": 50,
    "active_beta_panel_saturated": false,
    "num_full_gene_sets": 4912,
    "num_chains_effective_for_diag": 10,
    "burn_in_pass_streak": 0,
    "burn_in_patience": 2,
    "stop_pass_streak": 0,
    "stop_patience": 1000
  },
  {
    "epoch_iter_num": 500,
    "total_iter_num": 500,
    "stop_mcse_quantile": 0.9,
    "beta_rhat_q_post": 1.840822382185644,
    "beta_ratio_q": 0.10481002820464058,
    "max_rel_mcse_beta": 0.2,
    "beta_rel_mcse_denom_floor": 0.1,
    "top_gene_k": 200,
    "stop_min_gene_d": 0.3,
    "num_monitored_genes": 200,
    "num_eligible_genes": 333,
    "D_mcse_q": 0.06335565231360321,
    "max_abs_mcse_d": 0.1,
    "max_post_beta_rhat": 1.25,
    "prior_beta_rel_inconsistency_q": 0.0024299862992930047,
    "max_rel_prior_beta_inconsistency": 0.5,
    "num_active_betas": 54,
    "active_beta_panel_saturated": false,
    "num_full_gene_sets": 4912,
    "num_chains_effective_for_diag": 10,
    "burn_in_pass_streak": 0,
    "burn_in_patience": 2,
    "stop_pass_streak": 0,
    "stop_patience": 1000
  }
]
```
