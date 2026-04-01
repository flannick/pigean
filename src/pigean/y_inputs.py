from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class YPrimaryInputsContract:
    gwas_in: str | None = None
    huge_statistics_in: str | None = None
    exomes_in: str | None = None
    positive_controls_in: str | None = None
    positive_controls_list: list | None = None
    case_counts_in: str | None = None

    def has_any_source(self) -> bool:
        return any([
            self.gwas_in is not None,
            self.huge_statistics_in is not None,
            self.exomes_in is not None,
            self.positive_controls_in is not None,
            self.positive_controls_list is not None,
            self.case_counts_in is not None,
        ])

    def has_only_positive_controls(self) -> bool:
        return (
            self.positive_controls_in is not None
            or self.positive_controls_list is not None
        ) and (
            self.gwas_in is None
            and self.huge_statistics_in is None
            and self.exomes_in is None
            and self.case_counts_in is None
        )


@dataclass
class YReadContract:
    primary_inputs: YPrimaryInputsContract = field(default_factory=YPrimaryInputsContract)
    read_kwargs: dict = field(default_factory=dict)

    def has_any_source(self) -> bool:
        return self.primary_inputs.has_any_source()

    def has_only_positive_controls(self) -> bool:
        return self.primary_inputs.has_only_positive_controls()

    def to_read_kwargs(self) -> dict:
        return dict(self.read_kwargs)


def build_main_y_read_contract(options):
    primary_inputs = YPrimaryInputsContract(
        gwas_in=options.gwas_in,
        huge_statistics_in=options.huge_statistics_in,
        exomes_in=options.exomes_in,
        positive_controls_in=options.positive_controls_in,
        positive_controls_list=options.positive_controls_list,
        case_counts_in=options.case_counts_in,
    )
    read_kwargs = dict(
        gwas_in=options.gwas_in,
        huge_statistics_in=options.huge_statistics_in,
        huge_statistics_out=options.huge_statistics_out,
        show_progress=not options.hide_progress,
        gwas_chrom_col=options.gwas_chrom_col,
        gwas_pos_col=options.gwas_pos_col,
        gwas_p_col=options.gwas_p_col,
        gwas_beta_col=options.gwas_beta_col,
        gwas_se_col=options.gwas_se_col,
        gwas_n_col=options.gwas_n_col,
        gwas_n=options.gwas_n,
        gwas_units=options.gwas_units,
        gwas_freq_col=options.gwas_freq_col,
        gwas_filter_col=options.gwas_filter_col,
        gwas_filter_value=options.gwas_filter_value,
        gwas_locus_col=options.gwas_locus_col,
        gwas_ignore_p_threshold=options.gwas_ignore_p_threshold,
        gwas_low_p=options.gwas_low_p,
        gwas_high_p=options.gwas_high_p,
        gwas_low_p_posterior=options.gwas_low_p_posterior,
        gwas_high_p_posterior=options.gwas_high_p_posterior,
        detect_low_power=options.gwas_detect_low_power,
        detect_high_power=options.gwas_detect_high_power,
        detect_adjust_huge=options.gwas_detect_adjust_huge,
        learn_window=options.learn_window,
        closest_gene_prob=options.closest_gene_prob,
        max_closest_gene_prob=options.max_closest_gene_prob,
        scale_raw_closest_gene=options.scale_raw_closest_gene,
        cap_raw_closest_gene=options.cap_raw_closest_gene,
        cap_region_posterior=options.cap_region_posterior,
        scale_region_posterior=options.scale_region_posterior,
        phantom_region_posterior=options.phantom_region_posterior,
        allow_evidence_of_absence=options.allow_evidence_of_absence,
        correct_huge=options.correct_huge,
        gws_prob_true=options.gene_zs_gws_prob_true,
        max_closest_gene_dist=options.max_closest_gene_dist,
        signal_window_size=options.signal_window_size,
        signal_min_sep=options.signal_min_sep,
        signal_max_logp_ratio=options.signal_max_logp_ratio,
        credible_set_span=options.credible_set_span,
        min_n_ratio=options.min_n_ratio,
        max_clump_ld=options.max_clump_ld,
        exomes_in=options.exomes_in,
        exomes_gene_col=options.exomes_gene_col,
        exomes_p_col=options.exomes_p_col,
        exomes_beta_col=options.exomes_beta_col,
        exomes_se_col=options.exomes_se_col,
        exomes_n_col=options.exomes_n_col,
        exomes_n=options.exomes_n,
        exomes_units=options.exomes_units,
        exomes_low_p=options.exomes_low_p,
        exomes_high_p=options.exomes_high_p,
        exomes_low_p_posterior=options.exomes_low_p_posterior,
        exomes_high_p_posterior=options.exomes_high_p_posterior,
        positive_controls_in=options.positive_controls_in,
        positive_controls_id_col=options.positive_controls_id_col,
        positive_controls_prob_col=options.positive_controls_prob_col,
        positive_controls_default_prob=options.positive_controls_default_prob,
        positive_controls_has_header=options.positive_controls_has_header,
        positive_controls_list=options.positive_controls_list,
        positive_controls_all_in=options.positive_controls_all_in,
        positive_controls_all_id_col=options.positive_controls_all_id_col,
        positive_controls_all_has_header=options.positive_controls_all_has_header,
        gene_universe_in=options.gene_universe_in,
        gene_universe_id_col=options.gene_universe_id_col,
        gene_universe_has_header=options.gene_universe_has_header,
        gene_universe_from_y=options.gene_universe_from_y,
        gene_universe_from_x=options.gene_universe_from_x,
        case_counts_in=options.case_counts_in,
        case_counts_gene_col=options.case_counts_gene_col,
        case_counts_revel_col=options.case_counts_revel_col,
        case_counts_count_col=options.case_counts_count_col,
        case_counts_tot_col=options.case_counts_tot_col,
        case_counts_max_freq_col=options.case_counts_max_freq_col,
        min_revels=options.counts_min_revels,
        mean_rrs=options.counts_mean_rrs,
        max_case_freq=options.counts_max_case_freq,
        ctrl_counts_in=options.ctrl_counts_in,
        ctrl_counts_gene_col=options.ctrl_counts_gene_col,
        ctrl_counts_revel_col=options.ctrl_counts_revel_col,
        ctrl_counts_count_col=options.ctrl_counts_count_col,
        ctrl_counts_tot_col=options.ctrl_counts_tot_col,
        ctrl_counts_max_freq_col=options.ctrl_counts_max_freq_col,
        max_ctrl_freq=options.counts_max_ctrl_freq,
        syn_revel_threshold=options.counts_syn_revel,
        syn_fisher_p=options.counts_syn_fisher_p,
        nu=options.counts_nu,
        beta=options.counts_beta,
        gene_loc_file=options.gene_loc_file_huge if options.gene_loc_file_huge is not None else options.gene_loc_file,
        exons_loc_file=options.exons_loc_file_huge,
        gene_covs_in=options.gene_covs_in,
        hold_out_chrom=options.hold_out_chrom,
        min_var_posterior=options.min_var_posterior,
        s2g_in=options.s2g_in,
        s2g_chrom_col=options.s2g_chrom_col,
        s2g_pos_col=options.s2g_pos_col,
        s2g_gene_col=options.s2g_gene_col,
        s2g_prob_col=options.s2g_prob_col,
        s2g_normalize_values=options.s2g_normalize_values,
        credible_sets_in=options.credible_sets_in,
        credible_sets_id_col=options.credible_sets_id_col,
        credible_sets_chrom_col=options.credible_sets_chrom_col,
        credible_sets_pos_col=options.credible_sets_pos_col,
        credible_sets_ppa_col=options.credible_sets_ppa_col,
    )
    return YReadContract(primary_inputs=primary_inputs, read_kwargs=read_kwargs)


def mode_requires_gene_scores(mode_state):
    return (
        mode_state["run_huge"]
        or mode_state["run_beta_tilde"]
        or mode_state["run_beta"]
        or mode_state["run_priors"]
        or mode_state["run_naive_priors"]
        or mode_state["run_gibbs"]
    )


def load_advanced_set_b_y_inputs(state, options, read_gene_phewas_bfs_fn, bail_fn):
    if not options.betas_uncorrected_from_phewas:
        return False
    if not options.gene_phewas_bfs_in:
        bail_fn("Require --gene-phewas-bfs-in for --betas-from-phewas option")
    read_gene_phewas_bfs_fn(
        state,
        gene_phewas_bfs_in=options.gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=options.gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=options.gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=options.gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=options.gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=options.gene_phewas_bfs_prior_col,
        phewas_gene_to_X_gene_in=options.gene_phewas_id_to_X_id,
        min_value=options.min_gene_phewas_read_value,
        max_num_entries_at_once=options.max_read_entries_at_once,
    )
    return True


def load_main_y_inputs(
    state,
    options,
    mode_state,
    run_read_y_stage_fn,
    run_read_y_contract_stage_fn,
    read_gene_phewas_bfs_fn,
    bail_fn,
):
    if not mode_requires_gene_scores(mode_state):
        return True

    y_read_contract = build_main_y_read_contract(options)

    if options.gene_stats_in:
        run_read_y_stage_fn(
            state,
            gene_bfs_in=options.gene_stats_in,
            show_progress=not options.hide_progress,
            gene_bfs_id_col=options.gene_stats_id_col,
            gene_bfs_log_bf_col=options.gene_stats_log_bf_col,
            gene_bfs_combined_col=options.gene_stats_combined_col,
            gene_bfs_prob_col=options.gene_stats_prob_col,
            gene_bfs_prior_col=options.gene_stats_prior_col,
            gene_covs_in=options.gene_covs_in,
            hold_out_chrom=options.hold_out_chrom,
            gene_universe_in=options.gene_universe_in,
            gene_universe_id_col=options.gene_universe_id_col,
            gene_universe_has_header=options.gene_universe_has_header,
            gene_universe_from_y=options.gene_universe_from_y,
            gene_universe_from_x=options.gene_universe_from_x,
        )
        return False

    if y_read_contract.has_any_source():
        if y_read_contract.has_only_positive_controls():
            options.ols = True
        run_read_y_contract_stage_fn(state, y_read_contract)
        return False

    if load_advanced_set_b_y_inputs(
        state,
        options,
        read_gene_phewas_bfs_fn=read_gene_phewas_bfs_fn,
        bail_fn=bail_fn,
    ):
        return False

    return True
