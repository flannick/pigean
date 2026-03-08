from __future__ import annotations

import numpy as np


def apply_loaded_gene_covariates(domain, runtime, gene_covs_in, **kwargs):
    if gene_covs_in is None:
        return

    (cov_names, gene_covs, _, _) = runtime._read_gene_covs(gene_covs_in, **kwargs)
    cov_dirs = np.array([0] * len(cov_names))

    col_means = np.nanmean(gene_covs, axis=0)
    nan_indices = np.where(np.isnan(gene_covs))
    gene_covs[nan_indices] = np.take(col_means, nan_indices[1])

    if runtime.gene_covariates is not None:
        assert gene_covs.shape[0] == runtime.gene_covariates.shape[0]
        runtime.gene_covariates = np.hstack((runtime.gene_covariates, gene_covs))
        runtime.gene_covariate_names = runtime.gene_covariate_names + cov_names
        runtime.gene_covariate_directions = np.append(runtime.gene_covariate_directions, cov_dirs)
    else:
        runtime.gene_covariates = gene_covs
        runtime.gene_covariate_names = cov_names
        runtime.gene_covariate_directions = cov_dirs

    _finalize_gene_covariates(domain, runtime)


def _finalize_gene_covariates(domain, runtime):
    if runtime.gene_covariates is None:
        return

    constant_features = np.isclose(np.var(runtime.gene_covariates, axis=0), 0)
    if np.sum(constant_features) > 0:
        runtime.gene_covariates = runtime.gene_covariates[:, ~constant_features]
        runtime.gene_covariate_names = [runtime.gene_covariate_names[i] for i in np.where(~constant_features)[0]]
        runtime.gene_covariate_directions = np.array(
            [runtime.gene_covariate_directions[i] for i in np.where(~constant_features)[0]]
        )

    prune_threshold = 0.95
    cor_mat = np.abs(np.corrcoef(runtime.gene_covariates.T))
    np.fill_diagonal(cor_mat, 0)

    while True:
        if np.max(cor_mat) < prune_threshold:
            try:
                np.linalg.inv(runtime.gene_covariates.T.dot(runtime.gene_covariates))
                break
            except np.linalg.LinAlgError:
                pass

        max_index = np.unravel_index(np.argmax(cor_mat), cor_mat.shape)
        if np.max(max_index) == runtime.gene_covariate_intercept_index:
            max_index = np.min(max_index)
        else:
            max_index = np.max(max_index)

        domain.log("Removing feature %s" % runtime.gene_covariate_names[max_index], domain.TRACE)
        runtime.gene_covariates = np.delete(runtime.gene_covariates, max_index, axis=1)
        del runtime.gene_covariate_names[max_index]
        runtime.gene_covariate_directions = np.delete(runtime.gene_covariate_directions, max_index)
        cor_mat = np.delete(np.delete(cor_mat, max_index, axis=1), max_index, axis=0)
        if len(runtime.gene_covariates) == 0:
            domain.bail(
                "Error: something went wrong with matrix inversion. Still couldn't invert after removing all but one column"
            )

    runtime.gene_covariate_intercept_index = np.where(np.isclose(np.var(runtime.gene_covariates, axis=0), 0))[0]
    if len(runtime.gene_covariate_intercept_index) == 0:
        runtime.gene_covariates = np.hstack(
            (runtime.gene_covariates, np.ones(runtime.gene_covariates.shape[0])[:, np.newaxis])
        )
        runtime.gene_covariate_names.append("intercept")
        runtime.gene_covariate_directions = np.append(runtime.gene_covariate_directions, 0)
        runtime.gene_covariate_intercept_index = len(runtime.gene_covariate_names) - 1
    else:
        runtime.gene_covariate_intercept_index = runtime.gene_covariate_intercept_index[0]

    covariate_means = np.mean(runtime.gene_covariates, axis=0)
    covariate_sds = np.std(runtime.gene_covariates, axis=0)
    covariate_sds[covariate_sds == 0] = 1

    runtime.gene_covariates_mask = np.all(runtime.gene_covariates < covariate_means + 5 * covariate_sds, axis=1)
    runtime.gene_covariates_mat_inv = np.linalg.inv(
        runtime.gene_covariates[runtime.gene_covariates_mask, :].T.dot(
            runtime.gene_covariates[runtime.gene_covariates_mask, :]
        )
    )
    gene_covariate_sds = np.std(runtime.gene_covariates, axis=0)
    gene_covariate_sds[gene_covariate_sds == 0] = 1
    runtime.gene_covariate_zs = (runtime.gene_covariates - np.mean(runtime.gene_covariates, axis=0)) / gene_covariate_sds

    _apply_covariate_correction(runtime)


def _apply_covariate_correction(runtime):
    Y_for_regression = runtime.Y_for_regression
    if runtime.Y_for_regression is not None:
        (Y_for_regression, _, _) = runtime._correct_huge(
            runtime.Y_for_regression,
            runtime.gene_covariates,
            runtime.gene_covariates_mask,
            runtime.gene_covariates_mat_inv,
            runtime.gene_covariate_names,
            runtime.gene_covariate_intercept_index,
        )

    (Y, runtime.Y_uncorrected, _) = runtime._correct_huge(
        runtime.Y,
        runtime.gene_covariates,
        runtime.gene_covariates_mask,
        runtime.gene_covariates_mat_inv,
        runtime.gene_covariate_names,
        runtime.gene_covariate_intercept_index,
    )

    runtime._set_Y(Y, Y_for_regression, runtime.Y_exomes, runtime.Y_positive_controls, runtime.Y_case_counts)
    runtime.gene_covariate_adjustments = runtime.Y_for_regression - runtime.Y_uncorrected
