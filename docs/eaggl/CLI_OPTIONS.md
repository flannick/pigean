# CLI Option Inventory

This document is generated from parser definitions in `src/eaggl/cli.py` and CLI metadata in `eaggl.cli`.
Do not edit manually; run `scripts/eaggl/generate_cli_manifest.py`.

## Summary

- Total options: `214`
- `method_required`: `16`
- `method_optional`: `113`
- `engineering`: `68`
- `compat_alias`: `11`
- `debug_only`: `6`
- visibility `expert`: `176`
- visibility `hidden`: `8`
- visibility `normal`: `30`

## Method Required

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--X-in` | `normal` | `yes` | `core_help` | `X_in` | `None` | load one or more sparse gene-set matrix files directly |
| `--X-list` | `normal` | `yes` | `core_help` | `X_list` | `None` | load a file listing sparse gene-set matrix inputs |
| `--Xd-in` | `normal` | `yes` | `core_help` | `Xd_in` | `None` | load one or more dense gene-set matrix files directly |
| `--Xd-list` | `normal` | `yes` | `core_help` | `Xd_list` | `None` | load a file listing dense gene-set matrix inputs |
| `--anchor-any-gene` | `normal` | `yes` | `core_help` | `anchor_any_gene` | `False` | anchor factorization to any gene in the loaded gene-phewas inputs |
| `--anchor-any-pheno` | `normal` | `yes` | `core_help` | `anchor_any_pheno` | `False` | anchor factorization to any phenotype in the loaded phewas inputs |
| `--anchor-gene-set` | `normal` | `yes` | `core_help` | `anchor_gene_set` | `False` | run gene-set anchoring using the loaded phenotype evidence |
| `--anchor-genes` | `normal` | `yes` | `core_help` | `anchor_genes` | `None` | anchor factorization to one or more genes |
| `--anchor-phenos` | `normal` | `yes` | `core_help` | `anchor_phenos` | `None` | anchor factorization to one or more phenotypes |
| `--eaggl-bundle-in` | `normal` | `yes` | `core_help` | `eaggl_bundle_in` | `None` | load bundled PIGEAN outputs as default EAGGL inputs |
| `--gene-list` | `normal` | `yes` | `core_help` | `gene_list` | `None` | provide a standalone EAGGL input gene list directly on the command line |
| `--gene-list-in` | `normal` | `yes` | `core_help` | `gene_list_in` | `None` | read a standalone EAGGL input gene list from a file and synthesize enrichment weights internally |
| `--gene-list-max-fdr-q` | `normal` | `yes` | `core_help` | `gene_list_max_fdr_q` | `0.05` | retain enriched gene sets up to this Benjamini-Hochberg FDR threshold in standalone EAGGL gene-list mode |
| `--gene-loc-file` | `expert` | `yes` | `expert_help` | `gene_loc_file` | `None` | - |
| `--gene-set-stats-in` | `normal` | `yes` | `core_help` | `gene_set_stats_in` | `None` | load gene-set statistics exported from PIGEAN |
| `--gene-stats-in` | `normal` | `yes` | `core_help` | `gene_stats_in` | `None` | load gene-level statistics exported from PIGEAN |

## Method Optional

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--V-in` | `expert` | `yes` | `expert_help` | `V_in` | `None` | - |
| `--add-all-genes` | `expert` | `yes` | `expert_help` | `add_all_genes` | `False` | - |
| `--add-ext` | `expert` | `yes` | `expert_help` | `add_ext` | `False` | - |
| `--add-gene-sets-by-enrichment-p` | `expert` | `yes` | `expert_help` | `add_gene_sets_by_enrichment_p` | `None` | - |
| `--add-gene-sets-by-fraction` | `expert` | `yes` | `expert_help` | `add_gene_sets_by_fraction` | `None` | - |
| `--adjust-priors` | `expert` | `yes` | `expert_help` | `adjust_priors` | `None` | - |
| `--alpha0` | `normal` | `yes` | `core_help` | `alpha0` | `10` | - |
| `--background-prior` | `expert` | `yes` | `expert_help` | `background_prior` | `0.05` | - |
| `--beta0` | `normal` | `yes` | `core_help` | `beta0` | `1` | - |
| `--betas-from-phewas` | `expert` | `yes` | `advanced_workflows` | `betas_from_phewas` | `False` | - |
| `--betas-uncorrected-from-phewas` | `expert` | `yes` | `advanced_workflows` | `betas_uncorrected_from_phewas` | `False` | - |
| `--consensus-aggregation` | `normal` | `yes` | `core_help` | `consensus_aggregation` | `median` | choose how matched factors are aggregated across restarts in consensus mode |
| `--consensus-min-factor-cosine` | `normal` | `yes` | `core_help` | `consensus_min_factor_cosine` | `0.7` | minimum cosine similarity needed to align a restart factor to the reference factor |
| `--consensus-min-run-support` | `normal` | `yes` | `core_help` | `consensus_min_run_support` | `0.5` | minimum restart support fraction required to keep a consensus factor |
| `--consensus-nmf` | `normal` | `yes` | `core_help` | `consensus_nmf` | `False` | build a consensus factorization from multiple random restarts instead of keeping only the best run |
| `--correct-betas-mean` | `expert` | `yes` | `expert_help` | `correct_betas_mean` | `None` | - |
| `--factor-phewas-anchor-covariate` | `expert` | `yes` | `advanced_workflows` | `factor_phewas_anchor_covariate` | `direct` | choose the anchor covariate for binary factor-phewas modes: direct, combined, or none |
| `--factor-phewas-full-output` | `expert` | `yes` | `advanced_workflows` | `factor_phewas_full_output` | `False` | expose the full expert factor-phewas surface, including combined and huber variants |
| `--factor-phewas-min-gene-factor-weight` | `expert` | `yes` | `advanced_workflows` | `factor_phewas_min_gene_factor_weight` | `0.0` | - |
| `--factor-phewas-mode` | `expert` | `yes` | `advanced_workflows` | `factor_phewas_mode` | `marginal_anchor_adjusted_binary` | choose the factor-phewas model surface; the default is thresholded binary enrichment with direct anchor adjustment |
| `--factor-phewas-modes` | `expert` | `yes` | `advanced_workflows` | `factor_phewas_modes` | `None` | expert override: run multiple factor-phewas model surfaces in one pass and append them into one output table |
| `--factor-phewas-se` | `expert` | `yes` | `advanced_workflows` | `factor_phewas_se` | `robust` | choose the uncertainty estimator for binary factor-phewas: robust or none |
| `--factor-phewas-thresholded-combined-cutoff` | `expert` | `yes` | `advanced_workflows` | `factor_phewas_thresholded_combined_cutoff` | `1.0` | set the combined-support cutoff used to define thresholded phenotype hits for binary factor-phewas |
| `--factor-prune-gene-sets-num` | `expert` | `yes` | `advanced_workflows` | `factor_prune_gene_sets_num` | `None` | - |
| `--factor-prune-gene-sets-val` | `expert` | `yes` | `advanced_workflows` | `factor_prune_gene_sets_val` | `None` | - |
| `--factor-prune-genes-num` | `expert` | `yes` | `advanced_workflows` | `factor_prune_genes_num` | `None` | - |
| `--factor-prune-genes-val` | `expert` | `yes` | `advanced_workflows` | `factor_prune_genes_val` | `None` | - |
| `--factor-prune-phenos-num` | `expert` | `yes` | `advanced_workflows` | `factor_prune_phenos_num` | `None` | - |
| `--factor-prune-phenos-val` | `expert` | `yes` | `advanced_workflows` | `factor_prune_phenos_val` | `None` | - |
| `--factor-runs` | `normal` | `yes` | `core_help` | `factor_runs` | `1` | run repeated random restarts for factorization; without consensus keep only the best run |
| `--filter-gene-set-p` | `expert` | `yes` | `expert_help` | `filter_gene_set_p` | `None` | - |
| `--filter-negative` | `expert` | `yes` | `expert_help` | `filter_negative` | `None` | - |
| `--gauss-seidel-betas` | `expert` | `yes` | `expert_help` | `gauss_seidel_betas` | `-` | - |
| `--gene-cor-file` | `expert` | `yes` | `expert_help` | `gene_cor_file` | `None` | - |
| `--gene-covs-in` | `expert` | `yes` | `expert_help` | `gene_covs_in` | `None` | - |
| `--gene-filter-value` | `expert` | `yes` | `expert_help` | `gene_filter_value` | `1` | - |
| `--gene-map-in` | `expert` | `yes` | `expert_help` | `gene_map_in` | `None` | - |
| `--gene-phewas-id-to-X-id` | `expert` | `yes` | `advanced_workflows` | `gene_phewas_id_to_X_id` | `None` | - |
| `--gene-phewas-stats-in` | `expert` | `yes` | `advanced_workflows` | `gene_phewas_bfs_in` | `None` | - |
| `--gene-set-filter-value` | `expert` | `yes` | `expert_help` | `gene_set_filter_value` | `0.01` | - |
| `--gene-set-pheno-filter-value` | `expert` | `yes` | `expert_help` | `gene_set_pheno_filter_value` | `0.01` | - |
| `--gene-set-phewas-stats-in` | `expert` | `yes` | `advanced_workflows` | `gene_set_phewas_stats_in` | `None` | load gene-set phewas statistics for projection and anchor workflows |
| `--hold-out-chrom` | `expert` | `yes` | `expert_help` | `hold_out_chrom` | `None` | - |
| `--label-gene-sets-only` | `expert` | `yes` | `advanced_workflows` | `label_gene_sets_only` | `False` | - |
| `--label-include-phenos` | `expert` | `yes` | `advanced_workflows` | `label_include_phenos` | `False` | - |
| `--label-individually` | `expert` | `yes` | `advanced_workflows` | `label_individually` | `False` | - |
| `--learn-phi` | `normal` | `yes` | `core_help` | `learn_phi` | `False` | automatically tune phi by structural model selection before the final factorization |
| `--learn-phi-expand-factor` | `expert` | `yes` | `advanced_workflows` | `learn_phi_expand_factor` | `10.0` | set the multiplicative expansion factor used to bracket phi during automatic phi tuning |
| `--learn-phi-max-fit-loss-frac` | `expert` | `yes` | `advanced_workflows` | `learn_phi_max_fit_loss_frac` | `0.05` | maximum allowed reconstruction-error loss relative to the best tested phi during automatic tuning |
| `--learn-phi-max-redundancy` | `normal` | `yes` | `core_help` | `learn_phi_max_redundancy` | `0.6` | maximum allowed weighted Jaccard overlap between retained factors during automatic phi tuning |
| `--learn-phi-max-steps` | `expert` | `yes` | `advanced_workflows` | `learn_phi_max_steps` | `8` | maximum number of log-space phi search steps after bracketing |
| `--learn-phi-min-run-support` | `expert` | `yes` | `advanced_workflows` | `learn_phi_min_run_support` | `0.6` | minimum run-support fraction required for a phi candidate during automatic tuning |
| `--learn-phi-min-stability` | `expert` | `yes` | `advanced_workflows` | `learn_phi_min_stability` | `0.85` | minimum matched-factor cosine stability required for a phi candidate during automatic tuning |
| `--learn-phi-runs-per-step` | `expert` | `yes` | `advanced_workflows` | `learn_phi_runs_per_step` | `5` | number of repeated restarts used to score each candidate phi |
| `--learn-phi-weight-floor` | `expert` | `yes` | `advanced_workflows` | `learn_phi_weight_floor` | `None` | weights below this are treated as zero when measuring factor redundancy during phi tuning |
| `--linear` | `expert` | `yes` | `expert_help` | `linear` | `None` | - |
| `--lmm-auth-key` | `expert` | `yes` | `advanced_workflows` | `lmm_auth_key` | `None` | enable optional LLM-based factor labeling |
| `--lmm-model` | `expert` | `yes` | `advanced_workflows` | `lmm_model` | `gpt-4o-mini` | choose the LLM model used for optional labeling |
| `--lmm-provider` | `expert` | `yes` | `advanced_workflows` | `lmm_provider` | `openai` | choose the LLM provider used for optional labeling |
| `--max-for-linear` | `expert` | `yes` | `expert_help` | `max_for_linear` | `None` | - |
| `--max-frac-sem-betas` | `expert` | `yes` | `expert_help` | `max_frac_sem_betas` | `0.01` | - |
| `--max-gene-set-read-p` | `expert` | `yes` | `expert_help` | `max_gene_set_read_p` | `0.05` | - |
| `--max-gene-set-size` | `expert` | `yes` | `expert_help` | `max_gene_set_size` | `30000` | - |
| `--max-no-write-gene-gene-set-beta` | `expert` | `yes` | `expert_help` | `max_no_write_gene_gene_set_beta` | `0` | - |
| `--max-no-write-gene-pheno` | `expert` | `yes` | `expert_help` | `max_no_write_gene_pheno` | `0` | - |
| `--max-no-write-gene-set-beta` | `expert` | `yes` | `expert_help` | `max_no_write_gene_set_beta` | `None` | - |
| `--max-no-write-gene-set-beta-uncorrected` | `expert` | `yes` | `expert_help` | `max_no_write_gene_set_beta_uncorrected` | `None` | - |
| `--max-num-burn-in` | `expert` | `yes` | `expert_help` | `max_num_burn_in` | `None` | - |
| `--max-num-factors` | `normal` | `yes` | `core_help` | `max_num_factors` | `30` | - |
| `--max-num-gene-sets` | `expert` | `yes` | `expert_help` | `max_num_gene_sets` | `5000` | - |
| `--max-num-gene-sets-hyper` | `expert` | `yes` | `expert_help` | `max_num_gene_sets_hyper` | `5000` | - |
| `--max-num-gene-sets-initial` | `expert` | `yes` | `expert_help` | `max_num_gene_sets_initial` | `None` | - |
| `--max-num-iter-betas` | `expert` | `yes` | `expert_help` | `max_num_iter_betas` | `1100` | - |
| `--min-gene-phewas-read-value` | `expert` | `yes` | `advanced_workflows` | `min_gene_phewas_read_value` | `1` | - |
| `--min-gene-set-read-beta` | `expert` | `yes` | `expert_help` | `min_gene_set_read_beta` | `1e-20` | - |
| `--min-gene-set-read-beta-uncorrected` | `expert` | `yes` | `expert_help` | `min_gene_set_read_beta_uncorrected` | `1e-20` | - |
| `--min-gene-set-size` | `expert` | `yes` | `expert_help` | `min_gene_set_size` | `None` | - |
| `--min-lambda-threshold` | `normal` | `yes` | `core_help` | `min_lambda_threshold` | `0.001` | - |
| `--min-num-iter-betas` | `expert` | `yes` | `expert_help` | `min_num_iter_betas` | `10` | - |
| `--no-add-bottom` | `expert` | `yes` | `expert_help` | `add_bottom` | `True` | - |
| `--no-add-top` | `expert` | `yes` | `expert_help` | `add_top` | `True` | - |
| `--no-adjust-priors` | `expert` | `yes` | `expert_help` | `adjust_priors` | `None` | - |
| `--no-cap-weights` | `expert` | `yes` | `expert_help` | `cap_weights` | `True` | - |
| `--no-correct-betas-mean` | `expert` | `yes` | `expert_help` | `correct_betas_mean` | `None` | - |
| `--no-filter-negative` | `expert` | `yes` | `expert_help` | `filter_negative` | `None` | - |
| `--no-linear` | `expert` | `yes` | `expert_help` | `linear` | `None` | - |
| `--no-sparse-solution` | `expert` | `yes` | `expert_help` | `sparse_solution` | `None` | - |
| `--no-transpose` | `expert` | `yes` | `expert_help` | `no_transpose` | `-` | - |
| `--num-chains` | `expert` | `yes` | `expert_help` | `num_chains` | `10` | - |
| `--num-chains-betas` | `expert` | `yes` | `expert_help` | `num_chains_betas` | `4` | - |
| `--num-gene-sets-for-prior` | `expert` | `yes` | `expert_help` | `num_gene_sets_for_prior` | `None` | - |
| `--ols` | `expert` | `yes` | `expert_help` | `ols` | `-` | - |
| `--p-noninf` | `expert` | `yes` | `expert_help` | `p_noninf` | `None` | - |
| `--permute-gene-sets` | `expert` | `yes` | `expert_help` | `permute_gene_sets` | `None` | - |
| `--pheno-capture-input` | `expert` | `yes` | `advanced_workflows` | `pheno_capture_input` | `weighted_thresholded` | choose the phenotype-capture input profile: weighted thresholded support by default or binary thresholded hits for expert sensitivity checks |
| `--pheno-filter-value` | `expert` | `yes` | `expert_help` | `pheno_filter_value` | `1` | - |
| `--phi` | `normal` | `yes` | `core_help` | `phi` | `0.05` | - |
| `--project-phenos-from-gene-sets` | `expert` | `yes` | `advanced_workflows` | `project_phenos_from_gene_sets` | `False` | project phenotype loadings from gene-set scores instead of gene scores |
| `--prune-deterministically` | `expert` | `yes` | `expert_help` | `prune_deterministically` | `-` | - |
| `--prune-gene-sets` | `expert` | `yes` | `expert_help` | `prune_gene_sets` | `None` | - |
| `--r-threshold-burn-in-betas` | `expert` | `yes` | `expert_help` | `r_threshold_burn_in_betas` | `1.01` | - |
| `--run-factor-phewas` | `expert` | `yes` | `advanced_workflows` | `run_factor_phewas` | `False` | run the optional factor-level phewas stage |
| `--run-phewas` | `expert` | `yes` | `advanced_workflows` | `run_phewas` | `False` | run the optional gene-level phewas output stage |
| `--sigma-power` | `expert` | `yes` | `expert_help` | `sigma_power` | `None` | - |
| `--sparse-frac-betas` | `expert` | `yes` | `expert_help` | `sparse_frac_betas` | `None` | - |
| `--sparse-solution` | `expert` | `yes` | `expert_help` | `sparse_solution` | `None` | - |
| `--threshold-weights` | `expert` | `yes` | `expert_help` | `threshold_weights` | `0.5` | - |
| `--top-gene-set-prior` | `expert` | `yes` | `expert_help` | `top_gene_set_prior` | `None` | - |
| `--update-hyper` | `expert` | `yes` | `expert_help` | `update_hyper` | `None` | - |
| `--use-beta-uncorrected-for-gene-gene-set-write-filter` | `expert` | `yes` | `expert_help` | `use_beta_uncorrected_for_gene_gene_set_write_filter` | `False` | - |
| `--use-max-r-for-convergence-betas` | `expert` | `yes` | `expert_help` | `use_max_r_for_convergence_betas` | `-` | - |
| `--weighted-prune-gene-sets` | `expert` | `yes` | `expert_help` | `weighted_prune_gene_sets` | `None` | - |
| `--x-sparsify` | `expert` | `yes` | `expert_help` | `x_sparsify` | `[50,100,250,1000]` | - |

## Engineering

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--batch-separator` | `expert` | `no` | `expert_help` | `batch_separator` | `@` | - |
| `--batch-size` | `expert` | `no` | `expert_help` | `batch_size` | `5000` | - |
| `--config` | `expert` | `no` | `core_help` | `config` | `None` | load a JSON config file; explicit CLI flags override config values |
| `--consensus-stats-out` | `normal` | `no` | `core_help` | `consensus_stats_out` | `None` | write per-run and per-factor diagnostics for restart or consensus factorization |
| `--debug-level` | `expert` | `no` | `core_help` | `debug_level` | `None` | set logging verbosity for progress and diagnostic output |
| `--deterministic` | `expert` | `no` | `core_help` | `deterministic` | `False` | force deterministic random seed behavior (seed=0 unless --seed is set) |
| `--factor-phewas-stats-out` | `expert` | `no` | `advanced_workflows` | `factor_phewas_stats_out` | `None` | - |
| `--factors-anchor-out` | `normal` | `no` | `core_help` | `factors_anchor_out` | `None` | write anchor-specific factorization outputs |
| `--factors-out` | `normal` | `no` | `core_help` | `factors_out` | `None` | write the main factor loading output table |
| `--file-separator` | `expert` | `no` | `expert_help` | `file_separator` | `None` | - |
| `--gene-anchor-clusters-out` | `expert` | `no` | `advanced_workflows` | `gene_anchor_clusters_out` | `None` | - |
| `--gene-clusters-out` | `expert` | `no` | `advanced_workflows` | `gene_clusters_out` | `None` | - |
| `--gene-covs-out` | `expert` | `no` | `expert_help` | `gene_covs_out` | `None` | - |
| `--gene-effectors-out` | `expert` | `no` | `expert_help` | `gene_effectors_out` | `None` | - |
| `--gene-gene-set-stats-out` | `expert` | `no` | `expert_help` | `gene_gene_set_stats_out` | `None` | - |
| `--gene-list-id-col` | `expert` | `no` | `expert_help` | `gene_list_id_col` | `1` | select the gene column from a standalone EAGGL gene-list file when it has multiple columns |
| `--gene-list-no-header` | `expert` | `no` | `expert_help` | `gene_list_no_header` | `False` | treat the standalone EAGGL gene-list file as headerless |
| `--gene-map-new-gene-col` | `expert` | `no` | `expert_help` | `gene_map_new_gene_col` | `2` | - |
| `--gene-map-orig-gene-col` | `expert` | `no` | `expert_help` | `gene_map_orig_gene_col` | `1` | - |
| `--gene-pheno-stats-out` | `expert` | `no` | `advanced_workflows` | `gene_pheno_stats_out` | `None` | - |
| `--gene-phewas-stats-combined-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_combined_col` | `None` | - |
| `--gene-phewas-stats-id-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_id_col` | `None` | - |
| `--gene-phewas-stats-log-bf-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_log_bf_col` | `None` | - |
| `--gene-phewas-stats-pheno-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_pheno_col` | `None` | - |
| `--gene-phewas-stats-prior-col` | `expert` | `no` | `expert_help` | `gene_phewas_bfs_prior_col` | `None` | - |
| `--gene-set-anchor-clusters-out` | `expert` | `no` | `advanced_workflows` | `gene_set_anchor_clusters_out` | `None` | - |
| `--gene-set-clusters-out` | `expert` | `no` | `advanced_workflows` | `gene_set_clusters_out` | `None` | - |
| `--gene-set-overlap-stats-out` | `expert` | `no` | `expert_help` | `gene_set_overlap_stats_out` | `None` | - |
| `--gene-set-phewas-stats-beta-col` | `expert` | `no` | `expert_help` | `gene_set_phewas_stats_beta_col` | `None` | - |
| `--gene-set-phewas-stats-beta-uncorrected-col` | `expert` | `no` | `expert_help` | `gene_set_phewas_stats_beta_uncorrected_col` | `None` | - |
| `--gene-set-phewas-stats-id-col` | `expert` | `no` | `expert_help` | `gene_set_phewas_stats_id_col` | `Gene_Set` | - |
| `--gene-set-phewas-stats-pheno-col` | `expert` | `no` | `expert_help` | `gene_set_phewas_stats_pheno_col` | `None` | - |
| `--gene-set-stats-beta-col` | `expert` | `no` | `expert_help` | `gene_set_stats_beta_col` | `None` | - |
| `--gene-set-stats-beta-tilde-col` | `expert` | `no` | `expert_help` | `gene_set_stats_beta_tilde_col` | `None` | - |
| `--gene-set-stats-beta-uncorrected-col` | `expert` | `no` | `expert_help` | `gene_set_stats_beta_uncorrected_col` | `None` | - |
| `--gene-set-stats-exp-beta-tilde-col` | `expert` | `no` | `expert_help` | `gene_set_stats_exp_beta_tilde_col` | `None` | - |
| `--gene-set-stats-id-col` | `expert` | `no` | `expert_help` | `gene_set_stats_id_col` | `Gene_Set` | - |
| `--gene-set-stats-out` | `expert` | `no` | `expert_help` | `gene_set_stats_out` | `None` | - |
| `--gene-set-stats-p-col` | `expert` | `no` | `expert_help` | `gene_set_stats_p_col` | `None` | - |
| `--gene-set-stats-se-col` | `expert` | `no` | `expert_help` | `gene_set_stats_se_col` | `None` | - |
| `--gene-stats-combined-col` | `expert` | `no` | `expert_help` | `gene_stats_combined_col` | `None` | - |
| `--gene-stats-id-col` | `expert` | `no` | `expert_help` | `gene_stats_id_col` | `None` | - |
| `--gene-stats-log-bf-col` | `expert` | `no` | `expert_help` | `gene_stats_log_bf_col` | `None` | - |
| `--gene-stats-out` | `expert` | `no` | `expert_help` | `gene_stats_out` | `None` | - |
| `--gene-stats-prior-col` | `expert` | `no` | `expert_help` | `gene_stats_prior_col` | `None` | - |
| `--gene-stats-prob-col` | `expert` | `no` | `expert_help` | `gene_stats_prob_col` | `None` | - |
| `--gibbs-max-mb-X-h` | `expert` | `no` | `expert_help` | `gibbs_max_mb_X_h` | `100` | - |
| `--gibbs-num-batches-parallel` | `expert` | `no` | `expert_help` | `gibbs_num_batches_parallel` | `10` | - |
| `--help-expert` | `expert` | `no` | `expert_help` | `help_expert` | `False` | show expert workflow, projection, and debug flags in addition to the normal public interface |
| `--hide-opts` | `expert` | `no` | `core_help` | `hide_opts` | `False` | suppress printing resolved options at startup |
| `--hide-progress` | `expert` | `no` | `core_help` | `hide_progress` | `False` | reduce progress logging noise during long runs |
| `--ignore-genes` | `expert` | `no` | `expert_help` | `ignore_genes` | `["NA"]` | - |
| `--ignore-negative-exp-beta` | `expert` | `no` | `expert_help` | `ignore_negative_exp_beta` | `-` | - |
| `--learn-phi-report-out` | `expert` | `no` | `advanced_workflows` | `learn_phi_report_out` | `None` | write per-candidate phi search diagnostics |
| `--log-file` | `expert` | `no` | `core_help` | `log_file` | `None` | write structured run logs to this file |
| `--max-gb` | `expert` | `no` | `expert_help` | `max_gb` | `2.0` | - |
| `--max-read-entries-at-once` | `expert` | `no` | `expert_help` | `max_read_entries_at_once` | `None` | - |
| `--params-out` | `expert` | `no` | `expert_help` | `params_out` | `None` | - |
| `--pheno-anchor-clusters-out` | `expert` | `no` | `advanced_workflows` | `pheno_anchor_clusters_out` | `None` | - |
| `--pheno-clusters-out` | `expert` | `no` | `advanced_workflows` | `pheno_clusters_out` | `None` | - |
| `--phewas-gene-set-stats-out` | `expert` | `no` | `expert_help` | `phewas_gene_set_stats_out` | `None` | - |
| `--phewas-stats-out` | `expert` | `no` | `advanced_workflows` | `phewas_stats_out` | `None` | - |
| `--pre-filter-batch-size` | `expert` | `no` | `expert_help` | `pre_filter_batch_size` | `None` | - |
| `--pre-filter-small-batch-size` | `expert` | `no` | `expert_help` | `pre_filter_small_batch_size` | `500` | - |
| `--print-effective-config` | `expert` | `no` | `core_help` | `print_effective_config` | `False` | print the fully resolved mode/options JSON and exit |
| `--priors-num-gene-batches` | `expert` | `no` | `expert_help` | `priors_num_gene_batches` | `20` | - |
| `--seed` | `expert` | `no` | `core_help` | `seed` | `None` | set explicit random seed for deterministic reproducibility checks |
| `--warnings-file` | `expert` | `no` | `core_help` | `warnings_file` | `None` | write warning messages to this file |

## Compat Alias

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--factor-phewas-from-gene-phewas-stats-in` | `hidden` | `yes` | `internal_only` | `factor_phewas_legacy_input` | `None` | compatibility alias for --run-factor-phewas plus --gene-phewas-stats-in |
| `--gene-phewas-bfs-combined-col` | `hidden` | `yes` | `internal_only` | `gene_phewas_bfs_combined_col` | `None` | - |
| `--gene-phewas-bfs-id-col` | `hidden` | `yes` | `internal_only` | `gene_phewas_bfs_id_col` | `None` | - |
| `--gene-phewas-bfs-in` | `hidden` | `yes` | `internal_only` | `gene_phewas_bfs_in` | `None` | load gene-phewas statistics for projection and anchor workflows |
| `--gene-phewas-bfs-log-bf-col` | `hidden` | `yes` | `internal_only` | `gene_phewas_bfs_log_bf_col` | `None` | - |
| `--gene-phewas-bfs-pheno-col` | `hidden` | `yes` | `internal_only` | `gene_phewas_bfs_pheno_col` | `None` | - |
| `--gene-phewas-bfs-prior-col` | `hidden` | `yes` | `internal_only` | `gene_phewas_bfs_prior_col` | `None` | - |
| `--positive-controls-all-in` | `expert` | `yes` | `expert_help` | `positive_controls_all_in` | `None` | compatibility alias for standalone EAGGL gene-list background handling |
| `--positive-controls-in` | `expert` | `yes` | `expert_help` | `positive_controls_in` | `None` | compatibility alias for --gene-list-in |
| `--positive-controls-list` | `expert` | `yes` | `expert_help` | `positive_controls_list` | `None` | compatibility alias for --gene-list |
| `--run-phewas-from-gene-phewas-stats-in` | `hidden` | `yes` | `internal_only` | `run_phewas_legacy_input` | `None` | compatibility alias for --run-phewas plus --gene-phewas-stats-in |

## Debug Only

| Flag | Visibility | Semantic | Doc target | Dest | Default | Notes |
|---|---|---|---|---|---|---|
| `--debug-just-check-header` | `expert` | `no` | `internal_only` | `debug_just_check_header` | `-` | - |
| `--debug-old-batch` | `expert` | `no` | `internal_only` | `debug_old_batch` | `-` | - |
| `--debug-only-avg-huge` | `expert` | `no` | `internal_only` | `debug_only_avg_huge` | `-` | - |
| `--debug-skip-correlation` | `expert` | `no` | `internal_only` | `debug_skip_correlation` | `-` | - |
| `--debug-skip-huber` | `expert` | `no` | `internal_only` | `debug_skip_huber` | `-` | - |
| `--debug-skip-phewas-covs` | `expert` | `no` | `internal_only` | `debug_skip_phewas_covs` | `-` | - |

