import copy

import numpy as np
import scipy.sparse as sparse

from pegs_shared.types import HyperparameterData, PhewasRuntimeState, YData


def y_data_from_runtime(runtime):
    return YData(
        Y=getattr(runtime, "Y", None),
        Y_for_regression=getattr(runtime, "Y_for_regression", None),
        Y_exomes=getattr(runtime, "Y_exomes", None),
        Y_positive_controls=getattr(runtime, "Y_positive_controls", None),
        Y_case_counts=getattr(runtime, "Y_case_counts", None),
        y_var=getattr(runtime, "y_var", None),
        y_corr=getattr(runtime, "y_corr", None),
        y_corr_sparse=getattr(runtime, "y_corr_sparse", None),
    )


def build_y_data_from_inputs(
    runtime,
    Y,
    Y_for_regression=None,
    Y_exomes=None,
    Y_positive_controls=None,
    Y_case_counts=None,
    Y_corr_m=None,
    store_corr_sparse=False,
    min_correlation=0,
):
    y_data = y_data_from_runtime(runtime)
    if Y_corr_m is not None:
        y_corr_m = copy.copy(Y_corr_m)
        if min_correlation is not None:
            y_corr_m[y_corr_m <= 0] = 0

        keep_mask = np.array([True] * len(y_corr_m))
        for i in range(len(y_corr_m) - 1, -1, -1):
            if np.sum(y_corr_m[i] != 0) == 0:
                keep_mask[i] = False
            else:
                break
        if np.sum(keep_mask) > 0:
            y_corr_m = y_corr_m[keep_mask]

        y_data.y_corr = copy.copy(y_corr_m)

        y_corr_diags = [y_data.y_corr[i, :(len(y_data.y_corr[i, :]) - i)] for i in range(len(y_data.y_corr))]
        y_corr_sparse = sparse.csc_matrix(
            sparse.diags(
                y_corr_diags + y_corr_diags[1:],
                list(range(len(y_corr_diags))) + list(range(-1, -len(y_corr_diags), -1)),
            )
        )

        if store_corr_sparse:
            y_data.y_corr_sparse = y_corr_sparse

    if Y is not None:
        na_mask = ~np.isnan(Y)
        y_data.y_var = np.var(Y[na_mask])
    else:
        y_data.y_var = None
    y_data.Y = Y
    y_data.Y_for_regression = Y_for_regression
    y_data.Y_exomes = Y_exomes
    y_data.Y_positive_controls = Y_positive_controls
    y_data.Y_case_counts = Y_case_counts
    return y_data


def apply_y_data_to_runtime(runtime, y_data):
    runtime.Y = y_data.Y
    runtime.Y_for_regression = y_data.Y_for_regression
    runtime.Y_exomes = y_data.Y_exomes
    runtime.Y_positive_controls = y_data.Y_positive_controls
    runtime.Y_case_counts = y_data.Y_case_counts
    runtime.y_var = y_data.y_var
    runtime.y_corr = y_data.y_corr
    runtime.y_corr_sparse = y_data.y_corr_sparse


def set_runtime_y_from_inputs(
    runtime,
    Y,
    Y_for_regression=None,
    Y_exomes=None,
    Y_positive_controls=None,
    Y_case_counts=None,
    Y_corr_m=None,
    store_corr_sparse=False,
    min_correlation=0,
):
    y_data = build_y_data_from_inputs(
        runtime=runtime,
        Y=Y,
        Y_for_regression=Y_for_regression,
        Y_exomes=Y_exomes,
        Y_positive_controls=Y_positive_controls,
        Y_case_counts=Y_case_counts,
        Y_corr_m=Y_corr_m,
        store_corr_sparse=store_corr_sparse,
        min_correlation=min_correlation,
    )
    apply_y_data_to_runtime(runtime, y_data)
    return y_data


def hyperparameter_data_from_runtime(runtime):
    return HyperparameterData.from_runtime(runtime)


def apply_hyperparameter_data_to_runtime(runtime, hyper_data):
    hyper_data.apply_to_runtime(runtime)


def ensure_hyperparameter_state(runtime):
    hyper_state = getattr(runtime, "hyperparameter_state", None)
    if isinstance(hyper_state, HyperparameterData):
        return hyper_state
    hyper_state = hyperparameter_data_from_runtime(runtime)
    runtime.hyperparameter_state = hyper_state
    return hyper_state


def phewas_runtime_state_from_runtime(runtime):
    return PhewasRuntimeState(
        phenos=getattr(runtime, "phenos", None),
        pheno_to_ind=getattr(runtime, "pheno_to_ind", None),
        gene_pheno_Y=getattr(runtime, "gene_pheno_Y", None),
        gene_pheno_combined_prior_Ys=getattr(runtime, "gene_pheno_combined_prior_Ys", None),
        gene_pheno_priors=getattr(runtime, "gene_pheno_priors", None),
        X_phewas_beta=getattr(runtime, "X_phewas_beta", None),
        X_phewas_beta_uncorrected=getattr(runtime, "X_phewas_beta_uncorrected", None),
        num_gene_phewas_filtered=getattr(runtime, "num_gene_phewas_filtered", 0),
        anchor_gene_mask=getattr(runtime, "anchor_gene_mask", None),
        anchor_pheno_mask=getattr(runtime, "anchor_pheno_mask", None),
    )


def apply_phewas_runtime_state_to_runtime(runtime, phewas_state):
    runtime.phenos = phewas_state.phenos
    runtime.pheno_to_ind = phewas_state.pheno_to_ind
    runtime.gene_pheno_Y = phewas_state.gene_pheno_Y
    runtime.gene_pheno_combined_prior_Ys = phewas_state.gene_pheno_combined_prior_Ys
    runtime.gene_pheno_priors = phewas_state.gene_pheno_priors
    runtime.X_phewas_beta = phewas_state.X_phewas_beta
    runtime.X_phewas_beta_uncorrected = phewas_state.X_phewas_beta_uncorrected
    runtime.num_gene_phewas_filtered = phewas_state.num_gene_phewas_filtered
    runtime.anchor_gene_mask = phewas_state.anchor_gene_mask
    runtime.anchor_pheno_mask = phewas_state.anchor_pheno_mask


def sync_y_state(runtime):
    y_state = y_data_from_runtime(runtime)
    apply_y_data_to_runtime(runtime, y_state)
    return y_state


def sync_hyperparameter_state(runtime):
    hyper_state = ensure_hyperparameter_state(runtime)
    apply_hyperparameter_data_to_runtime(runtime, hyper_state)
    return hyper_state


def sync_phewas_runtime_state(runtime):
    phewas_state = phewas_runtime_state_from_runtime(runtime)
    apply_phewas_runtime_state_to_runtime(runtime, phewas_state)
    return phewas_state
