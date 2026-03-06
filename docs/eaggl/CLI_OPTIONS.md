# CLI Option Inventory

This document is generated from parser definitions in `src/eaggl.py`.
Do not edit manually; run `scripts/generate_cli_manifest.py`.

## Summary

- Total options: `184`
- `core_model`: `158`
- `engineering`: `26`

## Core Model

| Flag | Dest | Type | Default | Action | Usage refs | Notes |
|---|---|---|---|---|---:|---|
| `--V-in` | `V_in` | `-` | `None` | `-` | 4 | - |
| `--X-in` | `X_in` | `-` | `None` | `append` | 19 | - |
| `--X-list` | `X_list` | `-` | `None` | `append` | 7 | - |
| `--Xd-in` | `Xd_in` | `-` | `None` | `append` | 7 | - |
| `--Xd-list` | `Xd_list` | `-` | `None` | `append` | 5 | - |
| `--add-all-genes` | `add_all_genes` | `-` | `False` | `store_true` | 4 | - |
| `--add-ext` | `add_ext` | `-` | `False` | `store_true` | 4 | - |
| `--add-gene-sets-by-enrichment-p` | `add_gene_sets_by_enrichment_p` | `float` | `None` | `-` | 6 | - |
| `--add-gene-sets-by-fraction` | `add_gene_sets_by_fraction` | `float` | `None` | `-` | 6 | - |
| `--adjust-priors` | `adjust_priors` | `-` | `None` | `store_true` | 4 | - |
| `--alpha0` | `alpha0` | `float` | `10` | `-` | 4 | - |
| `--anchor-any-gene` | `anchor_any_gene` | `-` | `False` | `store_true` | 9 | - |
| `--anchor-any-pheno` | `anchor_any_pheno` | `-` | `False` | `store_true` | 17 | - |
| `--anchor-gene` | `anchor_genes` | `string` | `None` | `callback` | 31 | - |
| `--anchor-gene-set` | `anchor_gene_set` | `-` | `False` | `store_true` | 9 | - |
| `--anchor-genes` | `anchor_genes` | `string` | `None` | `callback` | 12 | - |
| `--anchor-pheno` | `anchor_phenos` | `string` | `None` | `callback` | 16 | - |
| `--anchor-phenos` | `anchor_phenos` | `string` | `None` | `callback` | 12 | - |
| `--background-prior` | `background_prior` | `float` | `0.05` | `-` | 5 | - |
| `--beta0` | `beta0` | `float` | `1` | `-` | 4 | - |
| `--betas-from-phewas` | `betas_from_phewas` | `-` | `False` | `store_true` | 4 | - |
| `--betas-uncorrected-from-phewas` | `betas_uncorrected_from_phewas` | `-` | `False` | `store_true` | 4 | - |
| `--correct-betas-mean` | `correct_betas_mean` | `-` | `None` | `store_true` | 4 | - |
| `--factor-phewas-from-gene-phewas-stats-in` | `factor_phewas_from_gene_phewas_stats_in` | `-` | `None` | `-` | 5 | - |
| `--factor-phewas-min-gene-factor-weight` | `factor_phewas_min_gene_factor_weight` | `float` | `0.01` | `-` | 4 | - |
| `--factor-phewas-stats-out` | `factor_phewas_stats_out` | `-` | `None` | `-` | 4 | - |
| `--factor-prune-gene-sets-num` | `factor_prune_gene_sets_num` | `int` | `None` | `-` | 4 | - |
| `--factor-prune-gene-sets-val` | `factor_prune_gene_sets_val` | `float` | `None` | `-` | 4 | - |
| `--factor-prune-genes-num` | `factor_prune_genes_num` | `int` | `None` | `-` | 4 | - |
| `--factor-prune-genes-val` | `factor_prune_genes_val` | `float` | `None` | `-` | 4 | - |
| `--factor-prune-phenos-num` | `factor_prune_phenos_num` | `int` | `None` | `-` | 6 | - |
| `--factor-prune-phenos-val` | `factor_prune_phenos_val` | `float` | `None` | `-` | 6 | - |
| `--factors-anchor-out` | `factors_anchor_out` | `-` | `None` | `-` | 4 | - |
| `--factors-out` | `factors_out` | `-` | `None` | `-` | 17 | - |
| `--file-separator` | `file_separator` | `-` | `None` | `-` | 4 | - |
| `--filter-gene-set-p` | `filter_gene_set_p` | `float` | `None` | `-` | 4 | - |
| `--filter-negative` | `filter_negative` | `-` | `None` | `store_true` | 4 | - |
| `--gauss-seidel-betas` | `gauss_seidel_betas` | `-` | `-` | `store_true` | 4 | - |
| `--gene-anchor-clusters-out` | `gene_anchor_clusters_out` | `-` | `None` | `-` | 4 | - |
| `--gene-clusters-out` | `gene_clusters_out` | `-` | `None` | `-` | 5 | - |
| `--gene-cor-file` | `gene_cor_file` | `-` | `None` | `-` | 8 | - |
| `--gene-covs-in` | `gene_covs_in` | `-` | `None` | `-` | 7 | - |
| `--gene-covs-out` | `gene_covs_out` | `-` | `None` | `-` | 4 | - |
| `--gene-effectors-out` | `gene_effectors_out` | `-` | `None` | `-` | 4 | - |
| `--gene-filter-value` | `gene_filter_value` | `float` | `1` | `-` | 4 | - |
| `--gene-gene-set-stats-out` | `gene_gene_set_stats_out` | `-` | `None` | `-` | 4 | - |
| `--gene-loc-file` | `gene_loc_file` | `-` | `None` | `-` | 10 | - |
| `--gene-map-in` | `gene_map_in` | `-` | `None` | `-` | 5 | - |
| `--gene-map-orig-gene-col` | `gene_map_orig_gene_col` | `-` | `1` | `-` | 5 | - |
| `--gene-pheno-stats-out` | `gene_pheno_stats_out` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-bfs-combined-col` | `gene_phewas_bfs_combined_col` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-bfs-id-col` | `gene_phewas_bfs_id_col` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-bfs-in` | `gene_phewas_bfs_in` | `-` | `None` | `-` | 7 | - |
| `--gene-phewas-bfs-pheno-col` | `gene_phewas_bfs_pheno_col` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-bfs-prior-col` | `gene_phewas_bfs_prior_col` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-id-to-X-id` | `gene_phewas_id_to_X_id` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-stats-combined-col` | `gene_phewas_bfs_combined_col` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-stats-id-col` | `gene_phewas_bfs_id_col` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-stats-in` | `gene_phewas_bfs_in` | `-` | `None` | `-` | 49 | - |
| `--gene-phewas-stats-pheno-col` | `gene_phewas_bfs_pheno_col` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-stats-prior-col` | `gene_phewas_bfs_prior_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-anchor-clusters-out` | `gene_set_anchor_clusters_out` | `-` | `None` | `-` | 4 | - |
| `--gene-set-clusters-out` | `gene_set_clusters_out` | `-` | `None` | `-` | 6 | - |
| `--gene-set-filter-value` | `gene_set_filter_value` | `float` | `0.01` | `-` | 4 | - |
| `--gene-set-overlap-stats-out` | `gene_set_overlap_stats_out` | `-` | `None` | `-` | 4 | - |
| `--gene-set-pheno-filter-value` | `gene_set_pheno_filter_value` | `float` | `0.01` | `-` | 4 | - |
| `--gene-set-phewas-stats-beta-col` | `gene_set_phewas_stats_beta_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-phewas-stats-beta-uncorrected-col` | `gene_set_phewas_stats_beta_uncorrected_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-phewas-stats-id-col` | `gene_set_phewas_stats_id_col` | `-` | `Gene_Set` | `-` | 4 | - |
| `--gene-set-phewas-stats-in` | `gene_set_phewas_stats_in` | `-` | `None` | `-` | 48 | - |
| `--gene-set-phewas-stats-pheno-col` | `gene_set_phewas_stats_pheno_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-stats-beta-col` | `gene_set_stats_beta_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-stats-beta-tilde-col` | `gene_set_stats_beta_tilde_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-stats-beta-uncorrected-col` | `gene_set_stats_beta_uncorrected_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-stats-exp-beta-tilde-col` | `gene_set_stats_exp_beta_tilde_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-stats-id-col` | `gene_set_stats_id_col` | `-` | `Gene_Set` | `-` | 4 | - |
| `--gene-set-stats-in` | `gene_set_stats_in` | `-` | `None` | `-` | 14 | - |
| `--gene-set-stats-out` | `gene_set_stats_out` | `-` | `None` | `-` | 6 | - |
| `--gene-set-stats-p-col` | `gene_set_stats_p_col` | `-` | `None` | `-` | 4 | - |
| `--gene-set-stats-se-col` | `gene_set_stats_se_col` | `-` | `None` | `-` | 4 | - |
| `--gene-stats-combined-col` | `gene_stats_combined_col` | `-` | `None` | `-` | 5 | - |
| `--gene-stats-id-col` | `gene_stats_id_col` | `-` | `None` | `-` | 5 | - |
| `--gene-stats-in` | `gene_stats_in` | `-` | `None` | `-` | 21 | - |
| `--gene-stats-out` | `gene_stats_out` | `-` | `None` | `-` | 6 | - |
| `--gene-stats-prior-col` | `gene_stats_prior_col` | `-` | `None` | `-` | 5 | - |
| `--gene-stats-prob-col` | `gene_stats_prob_col` | `-` | `None` | `-` | 6 | - |
| `--gibbs-max-mb-X-h` | `gibbs_max_mb_X_h` | `int` | `100` | `-` | 4 | - |
| `--hide-progress` | `hide_progress` | `-` | `False` | `store_true` | 4 | - |
| `--hold-out-chrom` | `hold_out_chrom` | `string` | `None` | `-` | 5 | - |
| `--ignore-genes` | `ignore_genes` | `-` | `["NA"]` | `append` | 4 | - |
| `--ignore-negative-exp-beta` | `ignore_negative_exp_beta` | `-` | `-` | `store_true` | 5 | - |
| `--label-gene-sets-only` | `label_gene_sets_only` | `-` | `False` | `store_true` | 5 | - |
| `--label-include-phenos` | `label_include_phenos` | `-` | `False` | `store_true` | 4 | - |
| `--label-individually` | `label_individually` | `-` | `False` | `store_true` | 4 | - |
| `--linear` | `linear` | `-` | `None` | `store_true` | 4 | - |
| `--lmm-auth-key` | `lmm_auth_key` | `str` | `None` | `-` | 5 | - |
| `--lmm-model` | `lmm_model` | `str` | `gpt-4o-mini` | `-` | 4 | - |
| `--lmm-provider` | `lmm_provider` | `str` | `openai` | `-` | 7 | - |
| `--max-for-linear` | `max_for_linear` | `float` | `None` | `-` | 4 | - |
| `--max-frac-sem-betas` | `max_frac_sem_betas` | `float` | `0.01` | `-` | 4 | - |
| `--max-gene-set-read-p` | `max_gene_set_read_p` | `float` | `0.05` | `-` | 4 | - |
| `--max-gene-set-size` | `max_gene_set_size` | `int` | `30000` | `-` | 4 | - |
| `--max-no-write-gene-gene-set-beta` | `max_no_write_gene_gene_set_beta` | `float` | `0` | `-` | 4 | - |
| `--max-no-write-gene-pheno` | `max_no_write_gene_pheno` | `float` | `0` | `-` | 4 | - |
| `--max-no-write-gene-set-beta` | `max_no_write_gene_set_beta` | `float` | `None` | `-` | 8 | - |
| `--max-no-write-gene-set-beta-uncorrected` | `max_no_write_gene_set_beta_uncorrected` | `float` | `None` | `-` | 4 | - |
| `--max-num-burn-in` | `max_num_burn_in` | `int` | `None` | `-` | 4 | - |
| `--max-num-factors` | `max_num_factors` | `int` | `30` | `-` | 4 | - |
| `--max-num-gene-sets` | `max_num_gene_sets` | `int` | `5000` | `-` | 13 | - |
| `--max-num-gene-sets-hyper` | `max_num_gene_sets_hyper` | `int` | `5000` | `-` | 4 | - |
| `--max-num-gene-sets-initial` | `max_num_gene_sets_initial` | `int` | `None` | `-` | 5 | - |
| `--max-num-iter-betas` | `max_num_iter_betas` | `int` | `1100` | `-` | 4 | - |
| `--min-gene-phewas-read-value` | `min_gene_phewas_read_value` | `float` | `1` | `-` | 4 | - |
| `--min-gene-set-read-beta` | `min_gene_set_read_beta` | `float` | `1e-20` | `-` | 8 | - |
| `--min-gene-set-read-beta-uncorrected` | `min_gene_set_read_beta_uncorrected` | `float` | `1e-20` | `-` | 4 | - |
| `--min-gene-set-size` | `min_gene_set_size` | `int` | `None` | `-` | 4 | - |
| `--min-lambda-threshold` | `min_lambda_threshold` | `float` | `0.001` | `-` | 4 | - |
| `--min-num-iter-betas` | `min_num_iter_betas` | `int` | `10` | `-` | 4 | - |
| `--no-add-bottom` | `add_bottom` | `-` | `True` | `store_false` | 4 | - |
| `--no-add-top` | `add_top` | `-` | `True` | `store_false` | 4 | - |
| `--no-adjust-priors` | `adjust_priors` | `-` | `None` | `store_false` | 4 | - |
| `--no-cap-weights` | `cap_weights` | `-` | `True` | `store_false` | 4 | - |
| `--no-correct-betas-mean` | `correct_betas_mean` | `-` | `None` | `store_false` | 4 | - |
| `--no-filter-negative` | `filter_negative` | `-` | `None` | `store_false` | 4 | - |
| `--no-linear` | `linear` | `-` | `None` | `store_false` | 4 | - |
| `--no-sparse-solution` | `sparse_solution` | `-` | `None` | `store_false` | 4 | - |
| `--no-transpose` | `no_transpose` | `-` | `-` | `store_true` | 4 | - |
| `--num-chains` | `num_chains` | `int` | `10` | `-` | 8 | - |
| `--num-chains-betas` | `num_chains_betas` | `int` | `4` | `-` | 4 | - |
| `--num-gene-sets-for-prior` | `num_gene_sets_for_prior` | `int` | `None` | `-` | 4 | - |
| `--ols` | `ols` | `-` | `-` | `store_true` | 6 | - |
| `--p-noninf` | `p_noninf` | `float` | `None` | `append` | 6 | - |
| `--params-out` | `params_out` | `-` | `None` | `-` | 5 | - |
| `--permute-gene-sets` | `permute_gene_sets` | `-` | `None` | `store_true` | 4 | - |
| `--pheno-anchor-clusters-out` | `pheno_anchor_clusters_out` | `-` | `None` | `-` | 4 | - |
| `--pheno-clusters-out` | `pheno_clusters_out` | `-` | `None` | `-` | 5 | - |
| `--pheno-filter-value` | `pheno_filter_value` | `float` | `1` | `-` | 4 | - |
| `--phewas-gene-set-stats-out` | `phewas_gene_set_stats_out` | `-` | `None` | `-` | 4 | - |
| `--phewas-stats-out` | `phewas_stats_out` | `-` | `None` | `-` | 4 | - |
| `--phi` | `phi` | `float` | `0.05` | `-` | 4 | - |
| `--positive-controls-all-in` | `positive_controls_all_in` | `-` | `None` | `-` | 4 | - |
| `--positive-controls-in` | `positive_controls_in` | `-` | `None` | `-` | 7 | - |
| `--positive-controls-list` | `positive_controls_list` | `string` | `None` | `callback` | 10 | - |
| `--project-phenos-from-gene-sets` | `project_phenos_from_gene_sets` | `-` | `False` | `store_true` | 4 | - |
| `--prune-gene-sets` | `prune_gene_sets` | `float` | `None` | `-` | 8 | - |
| `--r-threshold-burn-in-betas` | `r_threshold_burn_in_betas` | `float` | `1.01` | `-` | 4 | - |
| `--run-phewas-from-gene-phewas-stats-in` | `run_phewas_from_gene_phewas_stats_in` | `-` | `None` | `-` | 12 | - |
| `--sigma-power` | `sigma_power` | `float` | `None` | `-` | 4 | - |
| `--sparse-frac-betas` | `sparse_frac_betas` | `float` | `None` | `-` | 4 | - |
| `--sparse-solution` | `sparse_solution` | `-` | `None` | `store_true` | 4 | - |
| `--threshold-weights` | `threshold_weights` | `float` | `0.5` | `-` | 4 | - |
| `--top-gene-set-prior` | `top_gene_set_prior` | `float` | `None` | `-` | 4 | - |
| `--update-hyper` | `update_hyper` | `string` | `None` | `-` | 6 | - |
| `--use-beta-uncorrected-for-gene-gene-set-write-filter` | `use_beta_uncorrected_for_gene_gene_set_write_filter` | `-` | `False` | `store_true` | 4 | - |
| `--use-max-r-for-convergence-betas` | `use_max_r_for_convergence_betas` | `-` | `-` | `store_true` | 4 | - |
| `--warnings-file` | `warnings_file` | `-` | `None` | `-` | 4 | - |
| `--weighted-prune-gene-sets` | `weighted_prune_gene_sets` | `float` | `None` | `-` | 4 | - |
| `--x-sparsify` | `x_sparsify` | `string` | `[50,100,250,1000]` | `callback` | 5 | - |

## Engineering

| Flag | Dest | Type | Default | Action | Usage refs | Notes |
|---|---|---|---|---|---:|---|
| `--batch-separator` | `batch_separator` | `-` | `@` | `-` | 4 | - |
| `--batch-size` | `batch_size` | `int` | `5000` | `-` | 4 | - |
| `--config` | `config` | `-` | `None` | `-` | 12 | - |
| `--debug-just-check-header` | `debug_just_check_header` | `-` | `-` | `store_true` | 4 | - |
| `--debug-level` | `debug_level` | `int` | `None` | `-` | 4 | - |
| `--debug-old-batch` | `debug_old_batch` | `-` | `-` | `store_true` | 4 | - |
| `--debug-only-avg-huge` | `debug_only_avg_huge` | `-` | `-` | `store_true` | 4 | - |
| `--debug-skip-correlation` | `debug_skip_correlation` | `-` | `-` | `store_true` | 4 | - |
| `--debug-skip-huber` | `debug_skip_huber` | `-` | `-` | `store_true` | 4 | - |
| `--debug-skip-phewas-covs` | `debug_skip_phewas_covs` | `-` | `-` | `store_true` | 4 | - |
| `--deterministic` | `deterministic` | `-` | `False` | `store_true` | 10 | - |
| `--eaggl-bundle-in` | `eaggl_bundle_in` | `-` | `None` | `-` | 23 | - |
| `--gene-phewas-bfs-log-bf-col` | `gene_phewas_bfs_log_bf_col` | `-` | `None` | `-` | 4 | - |
| `--gene-phewas-stats-log-bf-col` | `gene_phewas_bfs_log_bf_col` | `-` | `None` | `-` | 4 | - |
| `--gene-stats-log-bf-col` | `gene_stats_log_bf_col` | `-` | `None` | `-` | 6 | - |
| `--gibbs-num-batches-parallel` | `gibbs_num_batches_parallel` | `int` | `10` | `-` | 4 | - |
| `--hide-opts` | `hide_opts` | `-` | `False` | `store_true` | 6 | - |
| `--log-file` | `log_file` | `-` | `None` | `-` | 4 | - |
| `--max-gb` | `max_gb` | `float` | `2.0` | `-` | 8 | - |
| `--max-read-entries-at-once` | `max_read_entries_at_once` | `int` | `None` | `-` | 4 | - |
| `--pre-filter-batch-size` | `pre_filter_batch_size` | `int` | `None` | `-` | 4 | - |
| `--pre-filter-small-batch-size` | `pre_filter_small_batch_size` | `int` | `500` | `-` | 4 | - |
| `--print-effective-config` | `print_effective_config` | `-` | `False` | `store_true` | 17 | - |
| `--priors-num-gene-batches` | `priors_num_gene_batches` | `int` | `20` | `-` | 4 | - |
| `--prune-deterministically` | `prune_deterministically` | `-` | `-` | `store_true` | 4 | - |
| `--seed` | `seed` | `int` | `None` | `-` | 7 | - |

