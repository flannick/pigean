# CLI Option Inventory

This document is generated from parser definitions in `src/pigean/cli.py` and CLI metadata in `pigean.cli`.
Do not edit manually; run `scripts/generate_cli_manifest.py`.

## Summary

- Total options: `329`
- `method_required`: `19`
- `method_optional`: `180`
- `engineering`: `110`
- `experimental`: `2`
- `compat_alias`: `10`
- `debug_only`: `8`
- visibility `expert`: `301`
- visibility `normal`: `28`

## Method Required

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--X-in` | `normal` | `yes` | `core_help` | `X_in` | `None` | load one or more sparse gene-set matrix files directly |
| `--X-list` | `normal` | `yes` | `core_help` | `X_list` | `None` | load a file listing sparse gene-set matrix inputs |
| `--Xd-in` | `normal` | `yes` | `core_help` | `Xd_in` | `None` | load one or more dense gene-set matrix files directly |
| `--Xd-list` | `normal` | `yes` | `core_help` | `Xd_list` | `None` | load a file listing dense gene-set matrix inputs |
| `--add-all-genes` | `expert` | `yes` | `expert_help` | `add_all_genes` | `False` | - |
| `--case-counts-in` | `normal` | `yes` | `core_help` | `case_counts_in` | `None` | load case variant-count evidence for gene-level support |
| `--credible-sets-in` | `expert` | `yes` | `expert_help` | `credible_sets_in` | `None` | - |
| `--ctrl-counts-in` | `normal` | `yes` | `core_help` | `ctrl_counts_in` | `None` | load control variant-count evidence paired with --case-counts-in |
| `--exomes-in` | `normal` | `yes` | `core_help` | `exomes_in` | `None` | load exome burden statistics as an additional HuGE evidence source |
| `--exons-loc-file-huge` | `expert` | `yes` | `expert_help` | `exons_loc_file_huge` | `None` | - |
| `--gene-covs-in` | `expert` | `yes` | `expert_help` | `gene_covs_in` | `None` | - |
| `--gene-list` | `normal` | `yes` | `core_help` | `positive_controls_list` | `None` | specify gene-list genes directly on the command line |
| `--gene-list-all-in` | `normal` | `yes` | `core_help` | `positive_controls_all_in` | `None` | load the full background gene universe for the gene-list input |
| `--gene-list-in` | `normal` | `yes` | `core_help` | `positive_controls_in` | `None` | load gene-list inputs with optional probabilities from a file |
| `--gene-loc-file` | `normal` | `yes` | `core_help` | `gene_loc_file` | `None` | gene location table used for correlation and locus-aware operations |
| `--gene-loc-file-huge` | `normal` | `yes` | `core_help` | `gene_loc_file_huge` | `None` | gene location table used during HuGE score construction |
| `--gene-map-in` | `expert` | `yes` | `expert_help` | `gene_map_in` | `None` | - |
| `--gwas-in` | `normal` | `yes` | `core_help` | `gwas_in` | `None` | load GWAS summary statistics as the primary HuGE input |
| `--s2g-in` | `normal` | `yes` | `core_help` | `s2g_in` | `None` | load SNP-to-gene mappings used during HuGE score construction |

## Method Optional

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--V-in` | `expert` | `yes` | `expert_help` | `V_in` | `None` | - |
| `--active-beta-min-abs` | `expert` | `yes` | `expert_help` | `active_beta_min_abs` | `0.005` | - |
| `--active-beta-top-k` | `expert` | `yes` | `expert_help` | `active_beta_top_k` | `200` | - |
| `--add-ext` | `expert` | `yes` | `expert_help` | `add_ext` | `False` | - |
| `--adjust-priors` | `expert` | `yes` | `expert_help` | `adjust_priors` | `None` | - |
| `--allow-evidence-of-absence` | `expert` | `yes` | `expert_help` | `allow_evidence_of_absence` | `False` | - |
| `--background-prior` | `expert` | `yes` | `expert_help` | `background_prior` | `0.05` | - |
| `--batch-all-for-hyper` | `expert` | `yes` | `expert_help` | `batch_all_for_hyper` | `-` | - |
| `--beta-rel-mcse-denom-floor` | `expert` | `yes` | `expert_help` | `beta_rel_mcse_denom_floor` | `0.1` | - |
| `--betas-from-phewas` | `expert` | `yes` | `advanced_workflows` | `betas_from_phewas` | `False` | sample betas using loaded gene-phewas statistics instead of default Y |
| `--betas-uncorrected-from-phewas` | `expert` | `yes` | `advanced_workflows` | `betas_uncorrected_from_phewas` | `False` | compute uncorrected beta path from gene-phewas statistics |
| `--burn-in-patience` | `expert` | `yes` | `expert_help` | `burn_in_patience` | `2` | - |
| `--burn-in-rhat-quantile` | `expert` | `yes` | `expert_help` | `burn_in_rhat_quantile` | `0.9` | - |
| `--burn-in-stall-delta` | `expert` | `yes` | `expert_help` | `burn_in_stall_delta` | `0.01` | - |
| `--burn-in-stall-window` | `expert` | `yes` | `expert_help` | `burn_in_stall_window` | `3` | - |
| `--cap-raw-closest-gene` | `expert` | `yes` | `expert_help` | `cap_raw_closest_gene` | `False` | - |
| `--closest-gene-prob` | `expert` | `yes` | `expert_help` | `closest_gene_prob` | `0.7` | - |
| `--const-gene-Y` | `expert` | `yes` | `expert_help` | `const_gene_Y` | `None` | - |
| `--const-gene-set-beta` | `expert` | `yes` | `expert_help` | `const_gene_set_beta` | `None` | - |
| `--const-sigma` | `expert` | `yes` | `expert_help` | `const_sigma` | `-` | - |
| `--correct-betas-mean` | `expert` | `yes` | `expert_help` | `correct_betas_mean` | `None` | - |
| `--correct-betas-var` | `expert` | `yes` | `expert_help` | `correct_betas_var` | `False` | - |
| `--counts-beta` | `expert` | `yes` | `expert_help` | `counts_beta` | `1.0` | - |
| `--counts-max-case-freq` | `expert` | `yes` | `expert_help` | `counts_max_case_freq` | `0.001` | - |
| `--counts-max-ctrl-freq` | `expert` | `yes` | `expert_help` | `counts_max_ctrl_freq` | `0.001` | - |
| `--counts-mean-rrs` | `expert` | `yes` | `expert_help` | `counts_mean_rrs` | `[1.3, 1.6, 2.5, 3.8]` | - |
| `--counts-min-revels` | `expert` | `yes` | `expert_help` | `counts_min_revels` | `[0.4, 0.6, 0.8, 1]` | - |
| `--counts-nu` | `expert` | `yes` | `expert_help` | `counts_nu` | `1.0` | - |
| `--counts-syn-fisher-p` | `expert` | `yes` | `expert_help` | `counts_syn_fisher_p` | `0.0001` | - |
| `--counts-syn-revel` | `expert` | `yes` | `expert_help` | `counts_syn_revel` | `0` | - |
| `--credible-set-span` | `expert` | `yes` | `expert_help` | `credible_set_span` | `25000` | - |
| `--cross-val` | `expert` | `yes` | `advanced_workflows` | `cross_val` | `None` | enable cross-validation tuning of inner beta sampling hyperparameters |
| `--cross-val-folds` | `expert` | `yes` | `advanced_workflows` | `cross_val_folds` | `4` | number of folds for cross-validation tuning |
| `--cross-val-max-num-tries` | `expert` | `yes` | `advanced_workflows` | `cross_val_max_num_tries` | `2` | maximum cross-validation boundary expansions |
| `--cross-val-num-explore-each-direction` | `expert` | `yes` | `advanced_workflows` | `cross_val_num_explore_each_direction` | `3` | cross-validation exploration breadth for sigma tuning |
| `--disable-stall-detection` | `expert` | `yes` | `expert_help` | `disable_stall_detection` | `False` | - |
| `--exomes-high-p` | `expert` | `yes` | `expert_help` | `exomes_high_p` | `0.05` | - |
| `--exomes-high-p-posterior` | `expert` | `yes` | `expert_help` | `exomes_high_p_posterior` | `0.1` | - |
| `--exomes-low-p` | `expert` | `yes` | `expert_help` | `exomes_low_p` | `2.5e-06` | - |
| `--exomes-low-p-posterior` | `expert` | `yes` | `expert_help` | `exomes_low_p_posterior` | `0.95` | - |
| `--exomes-n` | `expert` | `yes` | `expert_help` | `exomes_n` | `None` | - |
| `--exomes-units` | `expert` | `yes` | `expert_help` | `exomes_units` | `None` | - |
| `--filter-gene-set-metric-z` | `expert` | `yes` | `expert_help` | `filter_gene_set_metric_z` | `2.5` | - |
| `--filter-gene-set-p` | `expert` | `yes` | `expert_help` | `filter_gene_set_p` | `None` | - |
| `--filter-negative` | `expert` | `yes` | `expert_help` | `filter_negative` | `None` | - |
| `--first-for-hyper` | `expert` | `yes` | `expert_help` | `first_for_hyper` | `-` | - |
| `--first-for-sigma-cond` | `expert` | `yes` | `expert_help` | `first_for_sigma_cond` | `-` | - |
| `--first-max-p-for-hyper` | `expert` | `yes` | `expert_help` | `first_max_p_for_hyper` | `-` | - |
| `--frac-gene-sets-for-prior` | `expert` | `yes` | `expert_help` | `frac_gene_sets_for_prior` | `1` | - |
| `--gauss-seidel` | `expert` | `yes` | `expert_help` | `gauss_seidel` | `-` | - |
| `--gauss-seidel-betas` | `expert` | `yes` | `expert_help` | `gauss_seidel_betas` | `-` | - |
| `--gene-cor-file` | `expert` | `yes` | `expert_help` | `gene_cor_file` | `None` | - |
| `--gene-list-default-prob` | `expert` | `yes` | `expert_help` | `positive_controls_default_prob` | `0.95` | default inclusion probability used for gene-list inputs without an explicit probability column |
| `--gene-phewas-bfs-in` | `expert` | `yes` | `advanced_workflows` | `gene_phewas_bfs_in` | `None` | input gene-phewas BFS table for advanced phewas workflows |
| `--gene-phewas-id-to-X-id` | `expert` | `yes` | `advanced_workflows` | `gene_phewas_id_to_X_id` | `None` | gene ID remapping table for advanced gene-phewas ingestion |
| `--gene-phewas-stats-in` | `expert` | `yes` | `advanced_workflows` | `gene_phewas_bfs_in` | `None` | input gene-phewas statistics table for advanced phewas workflows |
| `--gene-set-betas-in` | `expert` | `yes` | `expert_help` | `gene_set_betas_in` | `None` | - |
| `--gene-set-stats-in` | `normal` | `yes` | `core_help` | `gene_set_stats_in` | `None` | use precomputed gene-set statistics to bypass beta-tilde recomputation |
| `--gene-stats-in` | `normal` | `yes` | `core_help` | `gene_stats_in` | `None` | use precomputed gene-level statistics as input instead of deriving scores from raw sources |
| `--gene-zs-gws-prob-true` | `expert` | `yes` | `expert_help` | `gene_zs_gws_prob_true` | `None` | - |
| `--gwas-detect-high-power` | `expert` | `yes` | `expert_help` | `gwas_detect_high_power` | `100` | - |
| `--gwas-detect-low-power` | `expert` | `yes` | `expert_help` | `gwas_detect_low_power` | `10` | - |
| `--gwas-detect-no-adjust-huge` | `expert` | `yes` | `expert_help` | `gwas_detect_adjust_huge` | `True` | - |
| `--gwas-filter-value` | `expert` | `yes` | `expert_help` | `gwas_filter_value` | `None` | - |
| `--gwas-high-p` | `expert` | `yes` | `expert_help` | `gwas_high_p` | `0.01` | - |
| `--gwas-high-p-posterior` | `expert` | `yes` | `expert_help` | `gwas_high_p_posterior` | `0.01` | - |
| `--gwas-ignore-p-threshold` | `expert` | `yes` | `expert_help` | `gwas_ignore_p_threshold` | `None` | - |
| `--gwas-low-p` | `expert` | `yes` | `expert_help` | `gwas_low_p` | `5e-08` | - |
| `--gwas-low-p-posterior` | `expert` | `yes` | `expert_help` | `gwas_low_p_posterior` | `0.75` | - |
| `--gwas-n` | `expert` | `yes` | `expert_help` | `gwas_n` | `None` | - |
| `--gwas-units` | `expert` | `yes` | `expert_help` | `gwas_units` | `None` | - |
| `--hold-out-chrom` | `expert` | `yes` | `expert_help` | `hold_out_chrom` | `None` | - |
| `--increase-filter-gene-set-p` | `expert` | `yes` | `expert_help` | `increase_filter_gene_set_p` | `0.01` | - |
| `--learn-window` | `expert` | `yes` | `expert_help` | `learn_window` | `False` | - |
| `--linear` | `expert` | `yes` | `expert_help` | `linear` | `None` | - |
| `--max-abs-mcse-d` | `normal` | `yes` | `core_help` | `max_abs_mcse_d` | `None` | stop Gibbs once monitored gene-probability MCSE is below this absolute threshold |
| `--max-allowed-batch-correlation` | `expert` | `yes` | `expert_help` | `max_allowed_batch_correlation` | `0.5` | - |
| `--max-closest-gene-dist` | `expert` | `yes` | `expert_help` | `max_closest_gene_dist` | `250000.0` | - |
| `--max-closest-gene-prob` | `expert` | `yes` | `expert_help` | `max_closest_gene_prob` | `0.9` | - |
| `--max-clump-ld` | `expert` | `yes` | `expert_help` | `max_clump_ld` | `0.5` | - |
| `--max-for-linear` | `expert` | `yes` | `expert_help` | `max_for_linear` | `None` | - |
| `--max-frac-sem-betas` | `expert` | `yes` | `expert_help` | `max_frac_sem_betas` | `0.01` | - |
| `--max-gene-set-read-p` | `expert` | `yes` | `expert_help` | `max_gene_set_read_p` | `0.05` | - |
| `--max-gene-set-size` | `expert` | `yes` | `expert_help` | `max_gene_set_size` | `30000` | - |
| `--max-no-write-gene-gene-set-beta` | `expert` | `yes` | `expert_help` | `max_no_write_gene_gene_set_beta` | `0` | - |
| `--max-no-write-gene-set-beta` | `expert` | `yes` | `expert_help` | `max_no_write_gene_set_beta` | `None` | - |
| `--max-no-write-gene-set-beta-uncorrected` | `expert` | `yes` | `expert_help` | `max_no_write_gene_set_beta_uncorrected` | `None` | - |
| `--max-num-burn-in` | `expert` | `yes` | `expert_help` | `max_num_burn_in` | `None` | - |
| `--max-num-gene-sets` | `expert` | `yes` | `expert_help` | `max_num_gene_sets` | `5000` | - |
| `--max-num-gene-sets-hyper` | `expert` | `yes` | `expert_help` | `max_num_gene_sets_hyper` | `5000` | - |
| `--max-num-gene-sets-initial` | `expert` | `yes` | `expert_help` | `max_num_gene_sets_initial` | `None` | - |
| `--max-num-iter` | `normal` | `yes` | `core_help` | `max_num_iter` | `500` | legacy per-epoch outer Gibbs iteration cap used when phase-specific bounds are not set |
| `--max-num-iter-betas` | `expert` | `yes` | `expert_help` | `max_num_iter_betas` | `1100` | - |
| `--max-num-post-burn-in` | `expert` | `yes` | `expert_help` | `max_num_post_burn_in` | `None` | - |
| `--max-num-restarts` | `expert` | `yes` | `expert_help` | `max_num_restarts` | `10` | - |
| `--max-rel-mcse-beta` | `normal` | `yes` | `core_help` | `max_rel_mcse_beta` | `None` | stop Gibbs once active beta MCSE is below this relative threshold |
| `--min-gene-phewas-read-value` | `expert` | `yes` | `advanced_workflows` | `min_gene_phewas_read_value` | `1` | minimum value filter for advanced gene-phewas ingestion |
| `--min-gene-set-read-beta` | `expert` | `yes` | `expert_help` | `min_gene_set_read_beta` | `1e-20` | - |
| `--min-gene-set-read-beta-uncorrected` | `expert` | `yes` | `expert_help` | `min_gene_set_read_beta_uncorrected` | `1e-20` | - |
| `--min-gene-set-size` | `expert` | `yes` | `expert_help` | `min_gene_set_size` | `None` | - |
| `--min-n-ratio` | `expert` | `yes` | `expert_help` | `min_n_ratio` | `0.5` | - |
| `--min-num-burn-in` | `expert` | `yes` | `expert_help` | `min_num_burn_in` | `10` | - |
| `--min-num-gene-sets` | `expert` | `yes` | `expert_help` | `min_num_gene_sets` | `1` | - |
| `--min-num-iter-betas` | `expert` | `yes` | `expert_help` | `min_num_iter_betas` | `10` | - |
| `--min-num-post-burn-in` | `expert` | `yes` | `expert_help` | `min_num_post_burn_in` | `10` | - |
| `--min-var-posterior` | `expert` | `yes` | `expert_help` | `min_var_posterior` | `0.01` | - |
| `--multi-y-in` | `expert` | `yes` | `advanced_workflows` | `multi_y_in` | `None` | run the current pigean pipeline once per trait from a long-form multi-Y table and append trait-labeled outputs |
| `--no-add-bottom` | `expert` | `yes` | `expert_help` | `add_bottom` | `True` | - |
| `--no-add-top` | `expert` | `yes` | `expert_help` | `add_top` | `True` | - |
| `--no-adjust-priors` | `expert` | `yes` | `expert_help` | `adjust_priors` | `None` | - |
| `--no-cap-region-posterior` | `expert` | `yes` | `expert_help` | `cap_region_posterior` | `True` | - |
| `--no-cap-weights` | `expert` | `yes` | `expert_help` | `cap_weights` | `True` | - |
| `--no-correct-betas-mean` | `expert` | `yes` | `expert_help` | `correct_betas_mean` | `None` | - |
| `--no-correct-huge` | `expert` | `yes` | `expert_help` | `correct_huge` | `True` | - |
| `--no-cross-val` | `expert` | `yes` | `advanced_workflows` | `cross_val` | `None` | explicitly disable cross-validation tuning |
| `--no-filter-negative` | `expert` | `yes` | `expert_help` | `filter_negative` | `None` | - |
| `--no-initial-linear-filter` | `expert` | `yes` | `expert_help` | `initial_linear_filter` | `True` | - |
| `--no-linear` | `expert` | `yes` | `expert_help` | `linear` | `None` | - |
| `--no-scale-raw-closest-gene` | `expert` | `yes` | `expert_help` | `scale_raw_closest_gene` | `True` | - |
| `--no-sparse-solution` | `expert` | `yes` | `expert_help` | `sparse_solution` | `None` | - |
| `--no-update-huge-scores` | `expert` | `yes` | `expert_help` | `update_huge_scores` | `True` | - |
| `--no-warm-start` | `normal` | `yes` | `core_help` | `warm_start` | `-` | disable warm-starting and restart outer Gibbs updates from default initialization each iteration |
| `--num-chains` | `normal` | `yes` | `core_help` | `num_chains` | `10` | number of parallel outer Gibbs chains to run |
| `--num-chains-betas` | `expert` | `yes` | `expert_help` | `num_chains_betas` | `4` | - |
| `--num-gene-sets-for-prior` | `expert` | `yes` | `expert_help` | `num_gene_sets_for_prior` | `None` | - |
| `--num-mad` | `expert` | `yes` | `expert_help` | `num_mad` | `10` | - |
| `--ols` | `expert` | `yes` | `expert_help` | `ols` | `-` | - |
| `--p-noninf` | `expert` | `yes` | `expert_help` | `p_noninf` | `None` | - |
| `--p-noninf-inflate` | `expert` | `yes` | `expert_help` | `p_noninf_inflate` | `1.0` | - |
| `--permute-gene-sets` | `expert` | `yes` | `expert_help` | `permute_gene_sets` | `None` | - |
| `--phantom-region-posterior` | `expert` | `yes` | `expert_help` | `phantom_region_posterior` | `False` | - |
| `--phewas-comparison-set` | `expert` | `yes` | `advanced_workflows` | `phewas_comparison_set` | `matched` | choose gene-level phewas output surface: matched or diagnostic |
| `--prune-deterministically` | `expert` | `yes` | `expert_help` | `prune_deterministically` | `-` | - |
| `--prune-gene-sets` | `expert` | `yes` | `expert_help` | `prune_gene_sets` | `None` | - |
| `--r-threshold-burn-in` | `expert` | `yes` | `expert_help` | `r_threshold_burn_in` | `1.1` | - |
| `--r-threshold-burn-in-betas` | `expert` | `yes` | `expert_help` | `r_threshold_burn_in_betas` | `1.01` | - |
| `--run-phewas-from-gene-phewas-stats-in` | `expert` | `yes` | `advanced_workflows` | `run_phewas_from_gene_phewas_stats_in` | `None` | run gene-level phewas output stage from precomputed gene-phewas stats |
| `--s2g-normalize-values` | `expert` | `yes` | `expert_help` | `s2g_normalize_values` | `None` | - |
| `--scale-region-posterior` | `expert` | `yes` | `expert_help` | `scale_region_posterior` | `False` | - |
| `--sigma-num-devs-to-top` | `expert` | `yes` | `expert_help` | `sigma_num_devs_to_top` | `2.0` | - |
| `--sigma-power` | `expert` | `yes` | `expert_help` | `sigma_power` | `None` | - |
| `--sigma-soft-threshold-5` | `expert` | `yes` | `expert_help` | `sigma_soft_threshold_5` | `None` | - |
| `--sigma-soft-threshold-95` | `expert` | `yes` | `expert_help` | `sigma_soft_threshold_95` | `None` | - |
| `--sigma2` | `expert` | `yes` | `expert_help` | `sigma2` | `None` | - |
| `--sigma2-cond` | `expert` | `yes` | `expert_help` | `sigma2_cond` | `None` | - |
| `--sigma2-ext` | `expert` | `yes` | `expert_help` | `sigma2_ext` | `None` | - |
| `--signal-max-logp-ratio` | `expert` | `yes` | `expert_help` | `signal_max_logp_ratio` | `None` | - |
| `--signal-min-sep` | `expert` | `yes` | `expert_help` | `signal_min_sep` | `100000` | - |
| `--signal-window-size` | `expert` | `yes` | `expert_help` | `signal_window_size` | `250000` | - |
| `--sim-log-bf-noise-sigma-mult` | `expert` | `yes` | `advanced_workflows` | `sim_log_bf_noise_sigma_mult` | `0` | simulation-only noise scale for generated log Bayes factors |
| `--sim-only-positive` | `expert` | `yes` | `advanced_workflows` | `sim_only_positive` | `-` | simulation-only: constrain synthetic effects to positive values |
| `--sparse-frac-betas` | `expert` | `yes` | `expert_help` | `sparse_frac_betas` | `None` | - |
| `--sparse-frac-gibbs` | `expert` | `yes` | `expert_help` | `sparse_frac_gibbs` | `0.01` | - |
| `--sparse-max-gibbs` | `expert` | `yes` | `expert_help` | `sparse_max_gibbs` | `0.001` | - |
| `--sparse-solution` | `expert` | `yes` | `expert_help` | `sparse_solution` | `None` | - |
| `--stall-delta-mcse` | `expert` | `yes` | `expert_help` | `stall_delta_mcse` | `0.002` | - |
| `--stall-delta-rhat` | `expert` | `yes` | `expert_help` | `stall_delta_rhat` | `0.01` | - |
| `--stall-min-burn-in` | `expert` | `yes` | `expert_help` | `stall_min_burn_in` | `10` | - |
| `--stall-min-post-burn-samples` | `expert` | `yes` | `expert_help` | `stall_min_post_burn_in` | `50` | - |
| `--stall-recent-eps` | `expert` | `yes` | `expert_help` | `stall_recent_eps` | `0.05` | - |
| `--stall-recent-window` | `expert` | `yes` | `expert_help` | `stall_recent_window` | `4` | - |
| `--stall-window` | `expert` | `yes` | `expert_help` | `stall_window` | `3` | - |
| `--stop-mcse-quantile` | `expert` | `yes` | `expert_help` | `stop_mcse_quantile` | `None` | - |
| `--stop-min-gene-d` | `expert` | `yes` | `expert_help` | `stop_min_gene_d` | `0.3` | - |
| `--stop-patience` | `expert` | `yes` | `expert_help` | `stop_patience` | `2` | - |
| `--stop-top-gene-k` | `expert` | `yes` | `expert_help` | `stop_top_gene_k` | `200` | - |
| `--strict-stopping` | `normal` | `yes` | `core_help` | `strict_stopping` | `False` | tighten Gibbs stopping thresholds relative to the default lenient preset |
| `--threshold-weights` | `expert` | `yes` | `expert_help` | `threshold_weights` | `0.5` | - |
| `--top-gene-prior` | `expert` | `yes` | `expert_help` | `top_gene_prior` | `None` | - |
| `--top-gene-set-prior` | `expert` | `yes` | `expert_help` | `top_gene_set_prior` | `None` | - |
| `--total-num-iter-gibbs` | `normal` | `yes` | `core_help` | `total_num_iter_gibbs` | `None` | total outer Gibbs iteration budget across all restart epochs |
| `--update-hyper` | `normal` | `yes` | `core_help` | `update_hyper` | `None` | choose whether outer Gibbs updates p, sigma, both, or neither during adaptation |
| `--use-beta-uncorrected-for-gene-gene-set-write-filter` | `expert` | `yes` | `expert_help` | `use_beta_uncorrected_for_gene_gene_set_write_filter` | `False` | - |
| `--use-max-r-for-convergence` | `expert` | `yes` | `expert_help` | `use_max_r_for_convergence` | `-` | - |
| `--use-max-r-for-convergence-betas` | `expert` | `yes` | `expert_help` | `use_max_r_for_convergence_betas` | `-` | - |
| `--use-sampled-betas-in-gibbs` | `expert` | `yes` | `expert_help` | `use_sampled_betas_in_gibbs` | `-` | - |
| `--use-sampling-for-betas` | `expert` | `yes` | `expert_help` | `use_sampling_for_betas` | `None` | - |
| `--warm-start` | `normal` | `yes` | `core_help` | `warm_start` | `True` | reuse previous-iteration beta state when warm-starting outer Gibbs updates |
| `--weighted-prune-gene-sets` | `expert` | `yes` | `expert_help` | `weighted_prune_gene_sets` | `None` | - |
| `--x-sparsify` | `expert` | `yes` | `expert_help` | `x_sparsify` | `[50,100,250,1000]` | - |

## Engineering

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--V-out` | `expert` | `no` | `expert_help` | `V_out` | `None` | - |
| `--X-out` | `expert` | `no` | `expert_help` | `X_out` | `None` | - |
| `--Xd-out` | `expert` | `no` | `expert_help` | `Xd_out` | `None` | - |
| `--batch-separator` | `expert` | `no` | `expert_help` | `batch_separator` | `@` | - |
| `--batch-size` | `expert` | `no` | `expert_help` | `batch_size` | `5000` | - |
| `--betas-trace-out` | `expert` | `no` | `expert_help` | `betas_trace_out` | `None` | - |
| `--case-counts-count-col` | `expert` | `no` | `expert_help` | `case_counts_count_col` | `None` | - |
| `--case-counts-gene-col` | `expert` | `no` | `expert_help` | `case_counts_gene_col` | `None` | - |
| `--case-counts-max-freq-col` | `expert` | `no` | `expert_help` | `case_counts_max_freq_col` | `None` | - |
| `--case-counts-revel-col` | `expert` | `no` | `expert_help` | `case_counts_revel_col` | `None` | - |
| `--case-counts-tot-col` | `expert` | `no` | `expert_help` | `case_counts_tot_col` | `None` | - |
| `--config` | `expert` | `no` | `core_help` | `config` | `None` | load a JSON config file; explicit CLI flags override config values |
| `--credible-sets-chrom-col` | `expert` | `no` | `expert_help` | `credible_sets_chrom_col` | `None` | - |
| `--credible-sets-id-col` | `expert` | `no` | `expert_help` | `credible_sets_id_col` | `None` | - |
| `--credible-sets-pos-col` | `expert` | `no` | `expert_help` | `credible_sets_pos_col` | `None` | - |
| `--credible-sets-ppa-col` | `expert` | `no` | `expert_help` | `credible_sets_ppa_col` | `None` | - |
| `--ctrl-counts-count-col` | `expert` | `no` | `expert_help` | `ctrl_counts_count_col` | `None` | - |
| `--ctrl-counts-gene-col` | `expert` | `no` | `expert_help` | `ctrl_counts_gene_col` | `None` | - |
| `--ctrl-counts-max-freq-col` | `expert` | `no` | `expert_help` | `ctrl_counts_max_freq_col` | `None` | - |
| `--ctrl-counts-revel-col` | `expert` | `no` | `expert_help` | `ctrl_counts_revel_col` | `None` | - |
| `--ctrl-counts-tot-col` | `expert` | `no` | `expert_help` | `ctrl_counts_tot_col` | `None` | - |
| `--debug-level` | `expert` | `no` | `core_help` | `debug_level` | `None` | set logging verbosity for progress and diagnostic output |
| `--deterministic` | `expert` | `no` | `core_help` | `deterministic` | `False` | force deterministic random seed behavior (seed=0 unless --seed is set) |
| `--diag-every` | `expert` | `no` | `expert_help` | `diag_every` | `4` | - |
| `--eaggl-bundle-out` | `expert` | `no` | `expert_help` | `eaggl_bundle_out` | `None` | write bundled PIGEAN outputs for direct eaggl.py consumption |
| `--exomes-beta-col` | `expert` | `no` | `expert_help` | `exomes_beta_col` | `None` | - |
| `--exomes-gene-col` | `expert` | `no` | `expert_help` | `exomes_gene_col` | `None` | - |
| `--exomes-n-col` | `expert` | `no` | `expert_help` | `exomes_n_col` | `None` | - |
| `--exomes-p-col` | `expert` | `no` | `expert_help` | `exomes_p_col` | `None` | - |
| `--exomes-se-col` | `expert` | `no` | `expert_help` | `exomes_se_col` | `None` | - |
| `--file-separator` | `expert` | `no` | `expert_help` | `file_separator` | `None` | - |
| `--gene-cor-file-cor-start-col` | `expert` | `no` | `expert_help` | `gene_cor_file_cor_start_col` | `10` | - |
| `--gene-cor-file-gene-col` | `expert` | `no` | `expert_help` | `gene_cor_file_gene_col` | `1` | - |
| `--gene-covs-out` | `expert` | `no` | `expert_help` | `gene_covs_out` | `None` | - |
| `--gene-effectors-out` | `expert` | `no` | `expert_help` | `gene_effectors_out` | `None` | - |
| `--gene-gene-set-stats-out` | `expert` | `no` | `expert_help` | `gene_gene_set_stats_out` | `None` | - |
| `--gene-list-all-id-col` | `expert` | `no` | `expert_help` | `positive_controls_all_id_col` | `None` | ID column in the full background gene-universe file for gene-list inputs |
| `--gene-list-all-no-header` | `expert` | `no` | `expert_help` | `positive_controls_all_has_header` | `True` | declare that the background gene-universe file for gene-list inputs has no header row |
| `--gene-list-id-col` | `expert` | `no` | `expert_help` | `positive_controls_id_col` | `None` | gene ID column for the gene-list input file |
| `--gene-list-no-header` | `expert` | `no` | `expert_help` | `positive_controls_has_header` | `True` | declare that the gene-list input file has no header row |
| `--gene-list-prob-col` | `expert` | `no` | `expert_help` | `positive_controls_prob_col` | `None` | probability column for the gene-list input file |
| `--gene-map-new-gene-col` | `expert` | `no` | `expert_help` | `gene_map_new_gene_col` | `2` | - |
| `--gene-map-orig-gene-col` | `expert` | `no` | `expert_help` | `gene_map_orig_gene_col` | `1` | - |
| `--gene-phewas-bfs-combined-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_combined_col` | `None` | combined column for advanced gene-phewas input |
| `--gene-phewas-bfs-id-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_id_col` | `None` | gene ID column for advanced gene-phewas input |
| `--gene-phewas-bfs-log-bf-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_log_bf_col` | `None` | log BF column for advanced gene-phewas input |
| `--gene-phewas-bfs-pheno-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_pheno_col` | `None` | phenotype column for advanced gene-phewas input |
| `--gene-phewas-bfs-prior-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_prior_col` | `None` | prior column for advanced gene-phewas input |
| `--gene-phewas-stats-combined-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_combined_col` | `None` | combined column for advanced gene-phewas input |
| `--gene-phewas-stats-id-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_id_col` | `None` | gene ID column for advanced gene-phewas input |
| `--gene-phewas-stats-log-bf-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_log_bf_col` | `None` | log BF column for advanced gene-phewas input |
| `--gene-phewas-stats-pheno-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_pheno_col` | `None` | phenotype column for advanced gene-phewas input |
| `--gene-phewas-stats-prior-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_prior_col` | `None` | prior column for advanced gene-phewas input |
| `--gene-set-overlap-stats-out` | `expert` | `no` | `expert_help` | `gene_set_overlap_stats_out` | `None` | - |
| `--gene-set-stats-beta-col` | `expert` | `no` | `expert_help` | `gene_set_stats_beta_col` | `None` | beta column mapping for advanced gene-set stats ingestion |
| `--gene-set-stats-beta-tilde-col` | `expert` | `no` | `expert_help` | `gene_set_stats_beta_tilde_col` | `None` | beta-tilde column mapping for advanced gene-set stats ingestion |
| `--gene-set-stats-beta-uncorrected-col` | `expert` | `no` | `expert_help` | `gene_set_stats_beta_uncorrected_col` | `None` | uncorrected beta column mapping for advanced gene-set stats ingestion |
| `--gene-set-stats-exp-beta-tilde-col` | `expert` | `no` | `expert_help` | `gene_set_stats_exp_beta_tilde_col` | `None` | exp(beta-tilde) column mapping for advanced gene-set stats ingestion |
| `--gene-set-stats-id-col` | `expert` | `no` | `expert_help` | `gene_set_stats_id_col` | `Gene_Set` | column mapping for advanced --gene-set-stats-in ingestion |
| `--gene-set-stats-out` | `normal` | `no` | `core_help` | `gene_set_stats_out` | `None` | write the final gene-set statistics table |
| `--gene-set-stats-p-col` | `expert` | `no` | `expert_help` | `gene_set_stats_p_col` | `None` | p-value column mapping for advanced gene-set stats ingestion |
| `--gene-set-stats-se-col` | `expert` | `no` | `expert_help` | `gene_set_stats_se_col` | `None` | SE column mapping for advanced gene-set stats ingestion |
| `--gene-set-stats-trace-out` | `expert` | `no` | `expert_help` | `gene_set_stats_trace_out` | `None` | - |
| `--gene-stats-combined-col` | `expert` | `no` | `expert_help` | `gene_stats_combined_col` | `None` | combined column mapping for advanced --gene-stats-in ingestion |
| `--gene-stats-id-col` | `expert` | `no` | `expert_help` | `gene_stats_id_col` | `None` | column mapping for advanced --gene-stats-in ingestion |
| `--gene-stats-log-bf-col` | `expert` | `no` | `expert_help` | `gene_stats_log_bf_col` | `None` | log BF column mapping for advanced --gene-stats-in ingestion |
| `--gene-stats-out` | `normal` | `no` | `core_help` | `gene_stats_out` | `None` | write the final gene-level statistics table |
| `--gene-stats-prior-col` | `expert` | `no` | `expert_help` | `gene_stats_prior_col` | `None` | prior column mapping for advanced --gene-stats-in ingestion |
| `--gene-stats-prob-col` | `expert` | `no` | `expert_help` | `gene_stats_prob_col` | `None` | probability column mapping for advanced --gene-stats-in ingestion |
| `--gene-stats-trace-out` | `expert` | `no` | `expert_help` | `gene_stats_trace_out` | `None` | - |
| `--gibbs-max-mb-X-h` | `expert` | `no` | `expert_help` | `gibbs_max_mb_X_h` | `100` | - |
| `--gibbs-num-batches-parallel` | `expert` | `no` | `expert_help` | `gibbs_num_batches_parallel` | `10` | - |
| `--gwas-beta-col` | `expert` | `no` | `expert_help` | `gwas_beta_col` | `None` | - |
| `--gwas-chrom-col` | `expert` | `no` | `expert_help` | `gwas_chrom_col` | `None` | - |
| `--gwas-filter-col` | `expert` | `no` | `expert_help` | `gwas_filter_col` | `None` | - |
| `--gwas-freq-col` | `expert` | `no` | `expert_help` | `gwas_freq_col` | `None` | - |
| `--gwas-locus-col` | `expert` | `no` | `expert_help` | `gwas_locus_col` | `None` | - |
| `--gwas-n-col` | `expert` | `no` | `expert_help` | `gwas_n_col` | `None` | - |
| `--gwas-p-col` | `expert` | `no` | `expert_help` | `gwas_p_col` | `None` | - |
| `--gwas-pos-col` | `expert` | `no` | `expert_help` | `gwas_pos_col` | `None` | - |
| `--gwas-se-col` | `expert` | `no` | `expert_help` | `gwas_se_col` | `None` | - |
| `--help-expert` | `expert` | `no` | `expert_help` | `help_expert` | `False` | show expert, advanced, and debug flags in addition to the normal public interface |
| `--hide-opts` | `expert` | `no` | `core_help` | `hide_opts` | `False` | suppress printing resolved options at startup |
| `--hide-progress` | `expert` | `no` | `core_help` | `hide_progress` | `False` | reduce progress logging noise during long runs |
| `--huge-statistics-in` | `expert` | `no` | `expert_help` | `huge_statistics_in` | `None` | read precomputed HuGE statistics cache instead of raw --gwas-in processing |
| `--huge-statistics-out` | `expert` | `no` | `expert_help` | `huge_statistics_out` | `None` | write HuGE statistics cache for faster reruns |
| `--ignore-genes` | `expert` | `no` | `expert_help` | `ignore_genes` | `["NA"]` | - |
| `--ignore-negative-exp-beta` | `expert` | `no` | `expert_help` | `ignore_negative_exp_beta` | `-` | - |
| `--log-file` | `expert` | `no` | `core_help` | `log_file` | `None` | write structured run logs to this file |
| `--max-gb` | `expert` | `no` | `expert_help` | `max_gb` | `2.0` | - |
| `--max-read-entries-at-once` | `expert` | `no` | `expert_help` | `max_read_entries_at_once` | `None` | - |
| `--multi-y-combined-col` | `expert` | `no` | `expert_help` | `multi_y_combined_col` | `None` | combined-support column for --multi-y-in |
| `--multi-y-id-col` | `expert` | `no` | `expert_help` | `multi_y_id_col` | `None` | gene ID column for --multi-y-in |
| `--multi-y-log-bf-col` | `expert` | `no` | `expert_help` | `multi_y_log_bf_col` | `None` | log BF column for --multi-y-in |
| `--multi-y-max-phenos-per-batch` | `expert` | `no` | `expert_help` | `multi_y_max_phenos_per_batch` | `None` | expert override for the number of traits loaded per native multi-Y batch |
| `--multi-y-pheno-col` | `expert` | `no` | `expert_help` | `multi_y_pheno_col` | `None` | trait column for --multi-y-in |
| `--multi-y-prior-col` | `expert` | `no` | `expert_help` | `multi_y_prior_col` | `None` | prior-support column for --multi-y-in |
| `--params-out` | `normal` | `no` | `core_help` | `params_out` | `None` | write learned hyperparameters and runtime settings |
| `--phewas-gene-set-stats-out` | `expert` | `no` | `expert_help` | `phewas_gene_set_stats_out` | `None` | - |
| `--phewas-stats-out` | `expert` | `no` | `advanced_workflows` | `phewas_stats_out` | `None` | write optional advanced gene-level phewas output table |
| `--pre-filter-batch-size` | `expert` | `no` | `expert_help` | `pre_filter_batch_size` | `None` | - |
| `--pre-filter-small-batch-size` | `expert` | `no` | `expert_help` | `pre_filter_small_batch_size` | `500` | - |
| `--print-effective-config` | `expert` | `no` | `core_help` | `print_effective_config` | `False` | print the fully resolved mode/options JSON and exit |
| `--priors-num-gene-batches` | `expert` | `no` | `expert_help` | `priors_num_gene_batches` | `20` | - |
| `--s2g-chrom-col` | `expert` | `no` | `expert_help` | `s2g_chrom_col` | `None` | - |
| `--s2g-gene-col` | `expert` | `no` | `expert_help` | `s2g_gene_col` | `None` | - |
| `--s2g-pos-col` | `expert` | `no` | `expert_help` | `s2g_pos_col` | `None` | - |
| `--s2g-prob-col` | `expert` | `no` | `expert_help` | `s2g_prob_col` | `None` | - |
| `--seed` | `expert` | `no` | `core_help` | `seed` | `None` | set explicit random seed for deterministic reproducibility checks |
| `--warnings-file` | `expert` | `no` | `core_help` | `warnings_file` | `None` | write warning messages to this file |

## Experimental

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--experimental-hyper-mutation` | `expert` | `yes` | `expert_help` | `experimental_hyper_mutation` | `False` | - |
| `--experimental-increase-hyper-if-betas-below` | `expert` | `yes` | `expert_help` | `experimental_increase_hyper_if_betas_below` | `None` | - |

## Compat Alias

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--increase-hyper-if-betas-below` | `expert` | `no` | `expert_help` | `increase_hyper_if_betas_below` | `None` | - |
| `--positive-controls-all-id-col` | `expert` | `no` | `expert_help` | `positive_controls_all_id_col` | `None` | compatibility alias for --gene-list-all-id-col |
| `--positive-controls-all-in` | `expert` | `no` | `expert_help` | `positive_controls_all_in` | `None` | compatibility alias for --gene-list-all-in |
| `--positive-controls-all-no-header` | `expert` | `no` | `expert_help` | `positive_controls_all_has_header` | `True` | compatibility alias for --gene-list-all-no-header |
| `--positive-controls-default-prob` | `expert` | `no` | `expert_help` | `positive_controls_default_prob` | `0.95` | compatibility alias for --gene-list-default-prob |
| `--positive-controls-id-col` | `expert` | `no` | `expert_help` | `positive_controls_id_col` | `None` | compatibility alias for --gene-list-id-col |
| `--positive-controls-in` | `expert` | `no` | `expert_help` | `positive_controls_in` | `None` | compatibility alias for --gene-list-in |
| `--positive-controls-list` | `expert` | `no` | `expert_help` | `positive_controls_list` | `None` | compatibility alias for --gene-list |
| `--positive-controls-no-header` | `expert` | `no` | `expert_help` | `positive_controls_has_header` | `True` | compatibility alias for --gene-list-no-header |
| `--positive-controls-prob-col` | `expert` | `no` | `expert_help` | `positive_controls_prob_col` | `None` | compatibility alias for --gene-list-prob-col |

## Debug Only

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--debug-just-check-header` | `expert` | `no` | `internal_only` | `debug_just_check_header` | `-` | - |
| `--debug-max-gene-sets-for-hyper` | `expert` | `no` | `internal_only` | `debug_max_gene_sets_for_hyper` | `-` | - |
| `--debug-old-batch` | `expert` | `no` | `internal_only` | `debug_old_batch` | `-` | - |
| `--debug-only-avg-huge` | `expert` | `no` | `internal_only` | `debug_only_avg_huge` | `-` | - |
| `--debug-skip-correlation` | `expert` | `no` | `internal_only` | `debug_skip_correlation` | `-` | - |
| `--debug-skip-huber` | `expert` | `no` | `internal_only` | `debug_skip_huber` | `-` | - |
| `--debug-skip-phewas-covs` | `expert` | `no` | `internal_only` | `debug_skip_phewas_covs` | `-` | - |
| `--debug-zero-sparse` | `expert` | `no` | `internal_only` | `debug_zero_sparse` | `-` | - |

