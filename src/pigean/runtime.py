from __future__ import annotations

import contextlib


# State-field groups used to make temporary overrides explicit in hot paths.
STATE_FIELDS_X_INDEXING = (
    "X_orig",
    "X_orig_missing_genes",
    "X_orig_missing_gene_sets",
    "genes",
    "genes_missing",
    "gene_sets",
    "gene_sets_missing",
    "scale_factors",
    "mean_shifts",
)

STATE_FIELDS_Y_SOURCES = (
    "Y",
    "Y_for_regression",
    "Y_uncorrected",
    "gene_to_huge_score",
    "gene_to_gwas_huge_score",
    "gene_to_gwas_huge_score_uncorrected",
    "gene_to_exomes_huge_score",
)

STATE_FIELDS_COVARIATE_CORRECTION = (
    "gene_covariates",
    "gene_covariates_mask",
    "gene_covariate_names",
    "gene_covariate_directions",
    "gene_covariate_intercept_index",
    "gene_covariates_mat_inv",
    "gene_covariate_zs",
    "gene_covariate_adjustments",
)

STATE_FIELDS_SAMPLER_HYPER = (
    "p",
    "ps",
    "sigma2",
    "sigma2s",
    "sigma_power",
)

HYPERPARAMETER_PROXY_FIELDS = (
    "p",
    "sigma2",
    "sigma_power",
    "sigma2_osc",
    "sigma2_se",
    "sigma2_p",
    "sigma2_total_var",
    "sigma2_total_var_lower",
    "sigma2_total_var_upper",
    "ps",
    "sigma2s",
    "sigma2s_missing",
)


def build_runtime_state(state_cls, options):
    state = state_cls(background_prior=options.background_prior, batch_size=options.batch_size)
    state.debug_old_batch = options.debug_old_batch
    state.debug_skip_correlation = options.debug_skip_correlation
    state.debug_skip_phewas_covs = options.debug_skip_phewas_covs
    state.debug_only_avg_huge = options.debug_only_avg_huge
    state.debug_just_check_header = options.debug_just_check_header
    return state


def snapshot_state_fields(state, field_names):
    return {field_name: getattr(state, field_name) for field_name in field_names}


def restore_state_fields(state, snapshot):
    for field_name, field_value in snapshot.items():
        setattr(state, field_name, field_value)


@contextlib.contextmanager
def temporary_state_fields(state, overrides, restore_fields):
    snapshot = snapshot_state_fields(state, restore_fields)
    for field_name, field_value in overrides.items():
        setattr(state, field_name, field_value)
    try:
        yield snapshot
    finally:
        restore_state_fields(state, snapshot)


@contextlib.contextmanager
def open_optional_gibbs_trace_files(gene_set_stats_trace_out, gene_stats_trace_out, open_gz):
    with contextlib.ExitStack() as stack:
        gene_set_stats_trace_fh = None
        gene_stats_trace_fh = None
        if gene_set_stats_trace_out is not None:
            gene_set_stats_trace_fh = stack.enter_context(open_gz(gene_set_stats_trace_out, "w"))
            gene_set_stats_trace_fh.write(
                "It\tChain\tGene_Set\tbeta_tilde\tP\tZ\tSE\tbeta_uncorrected\tbeta\tpostp\tbeta_tilde_outlier_z\tR\tSEM\n"
            )
        if gene_stats_trace_out is not None:
            gene_stats_trace_fh = stack.enter_context(open_gz(gene_stats_trace_out, "w"))
            gene_stats_trace_fh.write("It\tChain\tGene\tprior\tcombined\tlog_bf\tD\tpercent_top\tadjust\n")
        yield (gene_set_stats_trace_fh, gene_stats_trace_fh)


def open_optional_inner_betas_trace_file(betas_trace_out, open_gz):
    if betas_trace_out is None:
        return None
    betas_trace_fh = open_gz(betas_trace_out, "w")
    betas_trace_fh.write(
        "It\tParallel\tChain\tGene_Set\tbeta_post\tbeta\tpostp\tres_beta_hat\tbeta_tilde\tbeta_internal\tres_beta_hat_internal\tbeta_tilde_internal\tse_internal\tsigma2\tp\tR\tR_weighted\tSEM\n"
    )
    return betas_trace_fh


def close_optional_inner_betas_trace_file(betas_trace_fh):
    if betas_trace_fh is not None:
        betas_trace_fh.close()


def return_inner_betas_result(betas_trace_fh, result):
    close_optional_inner_betas_trace_file(betas_trace_fh)
    return result


def maybe_unsubset_gene_sets(state, enabled, skip_V=False, skip_scale_factors=False):
    if not enabled:
        return None
    return state._unsubset_gene_sets(skip_V=skip_V, skip_scale_factors=skip_scale_factors)


def restore_subset_gene_sets(state, subset_mask, keep_missing=True, skip_V=False, skip_scale_factors=False):
    if subset_mask is None:
        return
    state.subset_gene_sets(
        subset_mask,
        keep_missing=keep_missing,
        skip_V=skip_V,
        skip_scale_factors=skip_scale_factors,
    )


@contextlib.contextmanager
def temporary_unsubset_gene_sets(
    state,
    enabled,
    keep_missing=True,
    skip_V=False,
    skip_scale_factors=False,
):
    subset_mask = maybe_unsubset_gene_sets(
        state,
        enabled,
        skip_V=skip_V,
        skip_scale_factors=skip_scale_factors,
    )
    try:
        yield subset_mask
    finally:
        restore_subset_gene_sets(
            state,
            subset_mask,
            keep_missing=keep_missing,
            skip_V=skip_V,
            skip_scale_factors=skip_scale_factors,
        )


def bind_hyperparameter_properties(state_cls, field_names=HYPERPARAMETER_PROXY_FIELDS):
    for field_name in field_names:
        private_name = "_%s" % field_name

        def _getter(self, _field=field_name, _private=private_name):
            hyper_state = self.__dict__.get("hyperparameter_state")
            if hyper_state is not None:
                return getattr(hyper_state, _field)
            return self.__dict__.get(_private, None)

        def _setter(self, value, _field=field_name, _private=private_name):
            self.__dict__[_private] = value
            hyper_state = self.__dict__.get("hyperparameter_state")
            if hyper_state is not None:
                setattr(hyper_state, _field, value)

        setattr(state_cls, field_name, property(_getter, _setter))
