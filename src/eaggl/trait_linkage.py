from __future__ import annotations

import math
import gzip

import numpy as np
from scipy import sparse

from . import phenotype_annotation as eaggl_phenotype_annotation


def _as_dense_2d(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2:
        raise ValueError("Trait linkage expects 2D inputs")
    return matrix


def _sanitize_nonfinite(matrix):
    if matrix is None:
        return None
    return np.nan_to_num(_as_dense_2d(matrix), nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_nonfinite_preserve_sparse(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        sanitized = matrix.tocsr(copy=True).astype(float)
        if sanitized.nnz > 0:
            sanitized.data = np.nan_to_num(sanitized.data, nan=0.0, posinf=0.0, neginf=0.0)
            sanitized.eliminate_zeros()
        return sanitized
    return np.nan_to_num(_as_dense_2d(matrix), nan=0.0, posinf=0.0, neginf=0.0)


def _compute_positive_counts(matrix):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        indicator = matrix.copy()
        if indicator.nnz > 0:
            indicator.data = np.where(indicator.data > 0.0, 1.0, 0.0)
            indicator.eliminate_zeros()
        return np.asarray(indicator.sum(axis=0)).ravel().astype(int)
    dense = _as_dense_2d(matrix)
    return np.asarray(np.sum(dense > 0, axis=0), dtype=int)


def _compute_effective_feature_count(matrix, *, eps=1e-12):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        support_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
        support_square_sums = np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(float)
        n_eff = np.square(support_sums) / np.maximum(support_square_sums, eps)
        n_eff = np.where(support_sums > 0.0, n_eff, 0.0)
        return np.asarray(n_eff, dtype=float)
    dense = _as_dense_2d(matrix)
    support_sums = np.sum(dense, axis=0, dtype=float)
    support_square_sums = np.sum(np.square(dense), axis=0, dtype=float)
    n_eff = np.square(support_sums) / np.maximum(support_square_sums, eps)
    n_eff = np.where(support_sums > 0.0, n_eff, 0.0)
    return np.asarray(n_eff, dtype=float)


def _normalize_profiles_by_strength(feature_by_profile, strengths, *, eps=1e-12):
    if feature_by_profile is None:
        return None
    inv_strengths = 1.0 / np.maximum(np.asarray(strengths, dtype=float), eps)
    if sparse.issparse(feature_by_profile):
        return feature_by_profile @ sparse.diags(inv_strengths, format="csr")
    return np.asarray(feature_by_profile, dtype=float) * inv_strengths[np.newaxis, :]


def _mask_full_profile_rows(feature_by_profile, row_mask):
    if feature_by_profile is None:
        return None
    mask = np.asarray(row_mask, dtype=bool)
    if sparse.issparse(feature_by_profile):
        masked = feature_by_profile.tocsr(copy=True)
        drop_rows = np.flatnonzero(~mask)
        if len(drop_rows) > 0:
            masked[drop_rows, :] = 0.0
            masked.eliminate_zeros()
        return masked
    masked = np.asarray(feature_by_profile, dtype=float).copy()
    masked[~mask, :] = 0.0
    return masked


def _build_full_basis_matrix(dense_basis, basis_mask, *, expected_num_rows):
    if basis_mask is None:
        if dense_basis.shape[0] != expected_num_rows:
            raise ValueError(
                "Trait linkage basis rows %s must match feature rows %s when no basis mask is provided"
                % (dense_basis.shape[0], expected_num_rows)
            )
        return np.asarray(dense_basis, dtype=float), np.full(expected_num_rows, True, dtype=bool)

    mask = np.asarray(basis_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("Trait linkage basis mask must be 1D")
    if mask.shape[0] != expected_num_rows:
        raise ValueError(
            "Trait linkage basis mask length %s must match feature rows %s"
            % (mask.shape[0], expected_num_rows)
        )

    if dense_basis.shape[0] == mask.shape[0]:
        full_basis = np.asarray(dense_basis, dtype=float).copy()
    elif dense_basis.shape[0] == int(np.sum(mask)):
        full_basis = np.zeros((mask.shape[0], dense_basis.shape[1]), dtype=float)
        full_basis[mask, :] = np.asarray(dense_basis, dtype=float)
    else:
        raise ValueError(
            "Trait linkage basis rows %s must match mask length %s or kept rows %s"
            % (dense_basis.shape[0], mask.shape[0], int(np.sum(mask)))
        )

    # Keep linkage on the retained factorized universe while still solving in the full objective space.
    full_basis[~mask, :] = 0.0
    return full_basis, mask


def resolve_trait_linkage_source(
    requested_source,
    *,
    combined=None,
    log_bf=None,
    prior=None,
    log_fn=None,
    info_level=1,
    context_label="trait linkage",
):
    allowed = {"auto", "combined", "log_bf", "prior"}
    if requested_source not in allowed:
        raise ValueError("Unknown trait linkage source: %s" % requested_source)

    candidates = [
        ("combined", combined),
        ("log_bf", log_bf),
        ("prior", prior),
    ]
    if requested_source != "auto":
        candidates = [candidate for candidate in candidates if candidate[0] == requested_source]

    for label, matrix in candidates:
        if matrix is None:
            continue
        dense_matrix = _sanitize_nonfinite(matrix)
        if log_fn is not None and requested_source == "auto":
            log_fn("Using %s support surface for %s" % (label, context_label), info_level)
        return dense_matrix, label
    return None, None


def resolve_trait_linkage_evidence_source(
    requested_source,
    *,
    combined=None,
    log_bf=None,
    prior=None,
    log_fn=None,
    info_level=1,
    context_label="trait linkage evidence",
):
    allowed = {"auto", "combined", "log_bf", "prior"}
    if requested_source not in allowed:
        raise ValueError("Unknown trait linkage evidence source: %s" % requested_source)

    candidates = [
        ("log_bf", log_bf),
        ("combined", combined),
        ("prior", prior),
    ]
    if requested_source != "auto":
        candidates = [candidate for candidate in candidates if candidate[0] == requested_source]

    for label, matrix in candidates:
        if matrix is None:
            continue
        dense_matrix = _sanitize_nonfinite(matrix)
        if log_fn is not None and requested_source == "auto":
            log_fn("Using %s support surface for %s" % (label, context_label), info_level)
        return dense_matrix, label
    return None, None


def _normal_cdf(values):
    values = np.asarray(values, dtype=float)
    erf_values = np.vectorize(math.erf)(values / math.sqrt(2.0))
    return 0.5 * (1.0 + erf_values)


def threshold_factor_gene_basis(basis, threshold=0.05):
    """Return nonnegative factor loadings with small gene weights zeroed."""
    dense = np.maximum(_sanitize_nonfinite(basis), 0.0)
    threshold = 0.0 if threshold is None else float(threshold)
    if threshold > 0.0:
        dense = np.where(dense >= threshold, dense, 0.0)
    return dense


def factor_basis_cosines(basis, *, eps=1e-12):
    """Cosine of each row-loading vector against each factor indicator."""
    dense = np.maximum(_sanitize_nonfinite(basis), 0.0)
    row_norms = np.linalg.norm(dense, axis=1, keepdims=True)
    return np.divide(dense, np.maximum(row_norms, eps), out=np.zeros_like(dense), where=row_norms > eps)


def write_factor_gmt(path, genes, factor_names, basis, threshold=0.05):
    """Export weighted EAGGL factors as a GMT-like factor gene-set file."""
    if path is None:
        return
    W = threshold_factor_gene_basis(basis, threshold)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as out:
        for factor_index, factor_name in enumerate(factor_names):
            keep = np.flatnonzero(W[:, factor_index] > 0.0)
            fields = [str(factor_name), "EAGGL_factor"]
            fields.extend(str(genes[i]) for i in keep)
            out.write("%s\n" % "\t".join(fields))


def _ols_beta_se(y, x, *, eps=1e-12):
    y = np.nan_to_num(np.asarray(y, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(np.asarray(x, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if y.shape[0] != x.shape[0] or y.size < 3:
        return np.nan, np.nan, np.nan, np.nan
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.dot(xc, xc))
    if denom <= eps:
        return 0.0, np.inf, 0.0, 1.0
    beta = float(np.dot(xc, yc) / denom)
    resid = yc - beta * xc
    sigma2 = float(np.dot(resid, resid) / max(y.size - 2, 1))
    se = math.sqrt(max(sigma2, eps) / max(denom, eps))
    z = beta / se if se > eps and np.isfinite(se) else 0.0
    p = float(math.erfc(abs(z) / math.sqrt(2.0)))
    return beta, se, z, p


def _joint_ridge_betas(Y, W, ridge=1e-6):
    Y = np.nan_to_num(np.asarray(Y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    W = np.nan_to_num(np.asarray(W, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if W.size == 0 or Y.size == 0:
        return np.zeros((Y.shape[1] if Y.ndim == 2 else 0, W.shape[1] if W.ndim == 2 else 0), dtype=float)
    X = W - np.mean(W, axis=0, keepdims=True)
    Yc = Y - np.mean(Y, axis=0, keepdims=True)
    XtX = X.T @ X
    penalty = float(ridge) * np.eye(XtX.shape[0], dtype=float)
    try:
        coef = np.linalg.solve(XtX + penalty, X.T @ Yc)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(XtX + penalty) @ (X.T @ Yc)
    return coef.T


def compute_factor_trait_links(
    nnls_project_fn,
    basis,
    feature_by_trait,
    *,
    basis_mask=None,
    threshold_mode="weighted_thresholded",
    threshold_value=1.0,
    trait_response_source_name="combined",
    factor_loading_threshold=0.05,
    computation_mode="sparse_full",
):
    """Compute simplified factor-trait links: PIGEAN-style beta stats and NNLS loadings.

    The beta columns are computed from the factor loadings treated as weighted gene-set
    memberships. `beta_uncorrected` is the marginal slope; `beta` is the joint ridge
    coefficient across all factors. `beta_tilde`, `se`, `z`, and `p_value` are the
    marginal regression statistics.
    """
    if computation_mode not in {"dense_full", "sparse_full"}:
        raise ValueError("Unknown trait linkage computation mode: %s" % computation_mode)
    dense_basis = threshold_factor_gene_basis(basis, factor_loading_threshold)
    target_feature_by_trait = _sanitize_nonfinite_preserve_sparse(feature_by_trait)
    if dense_basis is None or target_feature_by_trait is None:
        return None
    expected_num_rows = target_feature_by_trait.shape[0]
    if basis_mask is None:
        retained_mask = np.full(expected_num_rows, True, dtype=bool)
        if dense_basis.shape[0] != expected_num_rows:
            raise ValueError(
                "Trait linkage basis rows %s must match feature rows %s when no basis mask is provided"
                % (dense_basis.shape[0], expected_num_rows)
            )
        retained_basis = dense_basis
    else:
        retained_mask = np.asarray(basis_mask, dtype=bool)
        if retained_mask.shape[0] != expected_num_rows:
            raise ValueError("Trait linkage basis mask length must match feature rows")
        if dense_basis.shape[0] == expected_num_rows:
            retained_basis = dense_basis[retained_mask, :]
        elif dense_basis.shape[0] == int(np.sum(retained_mask)):
            retained_basis = dense_basis
        else:
            raise ValueError("Trait linkage basis rows do not match mask length or kept rows")

    full_support = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
        target_feature_by_trait,
        threshold_mode,
        threshold_value=threshold_value,
        strict_threshold=True,
    )
    retained_support = full_support[retained_mask, :]
    retained_dense = retained_support.toarray() if sparse.issparse(retained_support) else np.asarray(retained_support, dtype=float)
    retained_dense = np.nan_to_num(retained_dense, nan=0.0, posinf=0.0, neginf=0.0)

    trait_total_support = eaggl_phenotype_annotation.compute_profile_strengths(full_support)
    trait_n_eff = _compute_effective_feature_count(full_support)
    retained_n_eff = _compute_effective_feature_count(retained_support)

    beta_tilde = np.zeros((retained_dense.shape[1], retained_basis.shape[1]), dtype=float)
    se = np.full_like(beta_tilde, np.nan)
    z = np.zeros_like(beta_tilde)
    p_value = np.ones_like(beta_tilde)
    for trait_index in range(retained_dense.shape[1]):
        y = retained_dense[:, trait_index]
        for factor_index in range(retained_basis.shape[1]):
            b, s, zz, pp = _ols_beta_se(y, retained_basis[:, factor_index])
            beta_tilde[trait_index, factor_index] = b
            se[trait_index, factor_index] = s
            z[trait_index, factor_index] = zz
            p_value[trait_index, factor_index] = pp

    beta_uncorrected = beta_tilde.copy()
    beta = _joint_ridge_betas(retained_dense, retained_basis)
    nnls_loadings = np.asarray(
        nnls_project_fn(retained_basis, retained_dense.T, max_sum=None),
        dtype=float,
    )
    factor_weight_sum = np.sum(retained_basis, axis=0)
    factor_num_genes = np.sum(retained_basis > 0.0, axis=0).astype(int)
    return {
        "nnls": np.maximum(nnls_loadings, 0.0),
        "beta": beta,
        "beta_uncorrected": beta_uncorrected,
        "beta_tilde": beta_tilde,
        "se": se,
        "z": z,
        "p_value": p_value,
        "trait_total_support": trait_total_support,
        "trait_n_eff": trait_n_eff,
        "retained_n_eff": retained_n_eff,
        "factor_weight_sum": factor_weight_sum,
        "factor_num_genes": factor_num_genes,
        "factor_loading_threshold": float(factor_loading_threshold),
        "trait_response_source": trait_response_source_name,
    }


def _transform_effect_input(matrix, mode, threshold):
    dense = _sanitize_nonfinite(matrix)
    if dense is None:
        return None
    if mode == "raw":
        return dense
    if mode == "weighted_thresholded":
        return np.where(dense > float(threshold), dense, 0.0)
    if mode == "excess_thresholded":
        return np.maximum(dense - float(threshold), 0.0)
    raise ValueError("Unknown trait-factor linkage effect input transform: %s" % mode)


def _membership_matrix(basis, normalization, cap, *, eps=1e-12):
    W = np.maximum(_sanitize_nonfinite(basis), 0.0)
    cap = float(cap)
    if normalization == "max":
        col_max = np.max(W, axis=0, keepdims=True) if W.size > 0 else np.ones((1, W.shape[1]))
        M = W / np.maximum(col_max, eps)
    elif normalization == "raw_capped":
        M = W
    else:
        raise ValueError("Unknown trait-factor linkage membership normalization: %s" % normalization)
    return np.clip(M, 0.0, cap)


def _ridge_residualize(vector, design, ridge_lambda, *, eps=1e-12):
    y = np.asarray(vector, dtype=float).ravel()
    X = np.asarray(design, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if X.shape[1] == 0:
        return y - float(np.mean(y)) if y.size > 0 else y
    XtX = X.T @ X
    ridge = float(ridge_lambda) * np.eye(X.shape[1], dtype=float)
    if ridge.shape[0] > 0:
        ridge[0, 0] = 0.0
    try:
        coef = np.linalg.solve(XtX + ridge + eps * np.eye(X.shape[1]), X.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(XtX + ridge + eps * np.eye(X.shape[1])) @ (X.T @ y)
    return y - X @ coef


def _one_covariate_bayes_lm(y, x, prior_sd, *, eps=1e-12):
    y = np.nan_to_num(np.asarray(y, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(np.asarray(x, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if y.shape[0] != x.shape[0] or y.shape[0] < 3:
        return {
            "beta_hat": np.nan,
            "beta_var": np.nan,
            "posterior_mean": np.nan,
            "posterior_sd": np.nan,
            "posterior_prob_positive": np.nan,
            "ln_bf": np.nan,
            "available": False,
            "unavailable_reason": "insufficient_rows",
        }
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    sxx = float(np.dot(x_centered, x_centered))
    if sxx <= eps:
        return {
            "beta_hat": 0.0,
            "beta_var": np.nan,
            "posterior_mean": 0.0,
            "posterior_sd": np.nan,
            "posterior_prob_positive": np.nan,
            "ln_bf": 0.0,
            "available": False,
            "unavailable_reason": "zero_covariate_variance",
        }
    beta_hat = float(np.dot(x_centered, y_centered) / sxx)
    residual = y_centered - beta_hat * x_centered
    sigma2 = float(np.dot(residual, residual) / max(1, y.shape[0] - 2))
    sigma2 = max(sigma2, eps)
    V = float(sigma2 / max(sxx, eps))
    W_prior = float(prior_sd) ** 2
    r = float(W_prior / max(W_prior + V, eps))
    z = float(beta_hat / math.sqrt(max(V, eps)))
    ln_bf = 0.5 * (math.log(max(1.0 - r, eps)) + r * z * z)
    posterior_mean = float(r * beta_hat)
    posterior_var = float(max(r * V, eps))
    posterior_sd = math.sqrt(posterior_var)
    posterior_prob_positive = float(_normal_cdf(np.array([posterior_mean / posterior_sd]))[0])
    return {
        "beta_hat": beta_hat,
        "beta_var": V,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "posterior_prob_positive": posterior_prob_positive,
        "ln_bf": float(ln_bf),
        "available": True,
        "unavailable_reason": "",
    }


def _one_covariate_bayes_lm_matrix(Y, x, prior_sd, *, eps=1e-12):
    Y = np.nan_to_num(np.asarray(Y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(np.asarray(x, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.shape[0] != x.shape[0] or Y.shape[0] < 3:
        n_traits = Y.shape[1] if Y.ndim == 2 else 0
        return {
            "beta_hat": np.full(n_traits, np.nan, dtype=float),
            "beta_var": np.full(n_traits, np.nan, dtype=float),
            "posterior_mean": np.full(n_traits, np.nan, dtype=float),
            "posterior_sd": np.full(n_traits, np.nan, dtype=float),
            "posterior_prob_positive": np.full(n_traits, np.nan, dtype=float),
            "ln_bf": np.full(n_traits, np.nan, dtype=float),
            "available": np.full(n_traits, False, dtype=bool),
            "unavailable_reason": np.full(n_traits, "insufficient_rows", dtype=object),
        }
    x_centered = x - float(np.mean(x))
    Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
    sxx = float(np.dot(x_centered, x_centered))
    n_traits = Y.shape[1]
    if sxx <= eps:
        return {
            "beta_hat": np.zeros(n_traits, dtype=float),
            "beta_var": np.full(n_traits, np.nan, dtype=float),
            "posterior_mean": np.zeros(n_traits, dtype=float),
            "posterior_sd": np.full(n_traits, np.nan, dtype=float),
            "posterior_prob_positive": np.full(n_traits, np.nan, dtype=float),
            "ln_bf": np.zeros(n_traits, dtype=float),
            "available": np.full(n_traits, False, dtype=bool),
            "unavailable_reason": np.full(n_traits, "zero_covariate_variance", dtype=object),
        }
    beta_hat = np.asarray(x_centered @ Y_centered / sxx, dtype=float)
    residual = Y_centered - x_centered[:, np.newaxis] * beta_hat[np.newaxis, :]
    sigma2 = np.sum(np.square(residual), axis=0) / max(1, Y.shape[0] - 2)
    sigma2 = np.maximum(sigma2, eps)
    V = sigma2 / max(sxx, eps)
    W_prior = float(prior_sd) ** 2
    r = W_prior / np.maximum(W_prior + V, eps)
    z = beta_hat / np.sqrt(np.maximum(V, eps))
    ln_bf = 0.5 * (np.log(np.maximum(1.0 - r, eps)) + r * np.square(z))
    posterior_mean = r * beta_hat
    posterior_var = np.maximum(r * V, eps)
    posterior_sd = np.sqrt(posterior_var)
    posterior_prob_positive = _normal_cdf(posterior_mean / posterior_sd)
    return {
        "beta_hat": beta_hat,
        "beta_var": V,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "posterior_prob_positive": posterior_prob_positive,
        "ln_bf": ln_bf,
        "available": np.full(n_traits, True, dtype=bool),
        "unavailable_reason": np.full(n_traits, "", dtype=object),
    }


def _one_covariate_bayes_lm_from_crossproducts(xTy, yTy, sxx, n_obs, prior_sd, *, df=None, eps=1e-12):
    xTy = np.nan_to_num(np.asarray(xTy, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    yTy = np.nan_to_num(np.asarray(yTy, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    n_traits = int(max(xTy.size, yTy.size))
    if xTy.size != n_traits:
        xTy = np.resize(xTy, n_traits)
    if yTy.size != n_traits:
        yTy = np.resize(yTy, n_traits)
    if int(n_obs) < 3:
        return {
            "beta_hat": np.full(n_traits, np.nan, dtype=float),
            "beta_var": np.full(n_traits, np.nan, dtype=float),
            "posterior_mean": np.full(n_traits, np.nan, dtype=float),
            "posterior_sd": np.full(n_traits, np.nan, dtype=float),
            "posterior_prob_positive": np.full(n_traits, np.nan, dtype=float),
            "ln_bf": np.full(n_traits, np.nan, dtype=float),
            "available": np.full(n_traits, False, dtype=bool),
            "unavailable_reason": np.full(n_traits, "insufficient_rows", dtype=object),
        }
    sxx = float(sxx)
    if sxx <= eps:
        return {
            "beta_hat": np.zeros(n_traits, dtype=float),
            "beta_var": np.full(n_traits, np.nan, dtype=float),
            "posterior_mean": np.zeros(n_traits, dtype=float),
            "posterior_sd": np.full(n_traits, np.nan, dtype=float),
            "posterior_prob_positive": np.full(n_traits, np.nan, dtype=float),
            "ln_bf": np.zeros(n_traits, dtype=float),
            "available": np.full(n_traits, False, dtype=bool),
            "unavailable_reason": np.full(n_traits, "zero_covariate_variance", dtype=object),
        }
    beta_hat = xTy / sxx
    residual_sse = np.maximum(yTy - np.square(xTy) / max(sxx, eps), eps)
    sigma2 = residual_sse / max(1, int(df) if df is not None else int(n_obs) - 2)
    sigma2 = np.maximum(sigma2, eps)
    V = sigma2 / max(sxx, eps)
    W_prior = float(prior_sd) ** 2
    r = W_prior / np.maximum(W_prior + V, eps)
    z = beta_hat / np.sqrt(np.maximum(V, eps))
    ln_bf = 0.5 * (np.log(np.maximum(1.0 - r, eps)) + r * np.square(z))
    posterior_mean = r * beta_hat
    posterior_var = np.maximum(r * V, eps)
    posterior_sd = np.sqrt(posterior_var)
    posterior_prob_positive = _normal_cdf(posterior_mean / posterior_sd)
    return {
        "beta_hat": beta_hat,
        "beta_var": V,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "posterior_prob_positive": posterior_prob_positive,
        "ln_bf": ln_bf,
        "available": np.full(n_traits, True, dtype=bool),
        "unavailable_reason": np.full(n_traits, "", dtype=object),
    }


def _residual_crossproducts_against_design(Y, x, design, ridge_lambda, *, eps=1e-12):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    x = np.asarray(x, dtype=float).ravel()
    X = np.asarray(design, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_obs = Y.shape[0]
    if X.shape[1] == 0:
        x_resid = x - float(np.mean(x))
        y_centered = Y - np.mean(Y, axis=0, keepdims=True)
        return x_resid @ y_centered, np.sum(np.square(y_centered), axis=0), float(x_resid @ x_resid), max(1, n_obs - 2)
    XtX = X.T @ X
    ridge = float(ridge_lambda) * np.eye(X.shape[1], dtype=float)
    if ridge.shape[0] > 0:
        ridge[0, 0] = 0.0
    system = XtX + ridge + eps * np.eye(X.shape[1])
    XtY = X.T @ Y
    Xtx = X.T @ x
    try:
        inv_XtY = np.linalg.solve(system, XtY)
        inv_Xtx = np.linalg.solve(system, Xtx)
    except np.linalg.LinAlgError:
        pinv_system = np.linalg.pinv(system)
        inv_XtY = pinv_system @ XtY
        inv_Xtx = pinv_system @ Xtx
    yTy = (
        np.sum(np.square(Y), axis=0)
        - 2.0 * np.sum(XtY * inv_XtY, axis=0)
        + np.sum(inv_XtY * (XtX @ inv_XtY), axis=0)
    )
    sxx = float(x @ x - 2.0 * Xtx.T @ inv_Xtx + inv_Xtx.T @ XtX @ inv_Xtx)
    xTy = x @ Y - Xtx.T @ inv_XtY - inv_Xtx.T @ XtY + inv_Xtx.T @ XtX @ inv_XtY
    df = max(1, n_obs - X.shape[1] - 1)
    return np.asarray(xTy, dtype=float), np.asarray(yTy, dtype=float), max(sxx, 0.0), df


def _ridge_residualize_matrix(matrix, design, ridge_lambda, *, eps=1e-12):
    Y = np.asarray(matrix, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    X = np.asarray(design, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if X.shape[1] == 0:
        return Y - np.mean(Y, axis=0, keepdims=True)
    XtX = X.T @ X
    ridge = float(ridge_lambda) * np.eye(X.shape[1], dtype=float)
    if ridge.shape[0] > 0:
        ridge[0, 0] = 0.0
    system = XtX + ridge + eps * np.eye(X.shape[1])
    try:
        coef = np.linalg.solve(system, X.T @ Y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(system) @ (X.T @ Y)
    return Y - X @ coef


def _effect_matrix_shape(num_traits, num_factors, fill=np.nan):
    return np.full((int(num_traits), int(num_factors)), fill, dtype=float)


def _compute_effect_scores(
    full_basis,
    evidence_feature_by_trait,
    *,
    anchor_feature_by_covariate=None,
    effect_input_transform="weighted_thresholded",
    effect_threshold=1.0,
    effect_prior_sd=1.0,
    effect_min_trait_neff=10.0,
    effect_min_retained_fraction=0.1,
    notable_ln_bf=3.0,
    notable_ln_bf_scale=5.0,
    membership_normalization="max",
    membership_cap=1.0,
    trait_n_eff=None,
    retained_fraction=None,
    low_retention_flag=None,
    ridge_lambda=1e-6,
    eps=1e-12,
):
    Y = _transform_effect_input(evidence_feature_by_trait, effect_input_transform, effect_threshold)
    if full_basis is None or Y is None:
        return {}
    M = _membership_matrix(full_basis, membership_normalization, membership_cap, eps=eps)
    if M.shape[0] != Y.shape[0]:
        raise ValueError("Trait-factor linkage evidence rows must match factor basis rows")
    num_features, num_factors = M.shape
    num_traits = Y.shape[1]
    ones = np.ones((num_features, 1), dtype=float)
    trait_n_eff = np.zeros(num_traits, dtype=float) if trait_n_eff is None else np.asarray(trait_n_eff, dtype=float)
    retained_fraction = np.ones(num_traits, dtype=float) if retained_fraction is None else np.asarray(retained_fraction, dtype=float)
    low_retention_flag = np.zeros(num_traits, dtype=bool) if low_retention_flag is None else np.asarray(low_retention_flag, dtype=bool)

    results = {
        "effect_membership": M,
        "marginal_mean_in": _effect_matrix_shape(num_traits, num_factors),
        "marginal_mean_out": _effect_matrix_shape(num_traits, num_factors),
        "marginal_lift": _effect_matrix_shape(num_traits, num_factors),
        "marginal_posterior_lift": _effect_matrix_shape(num_traits, num_factors),
        "marginal_posterior_sd": _effect_matrix_shape(num_traits, num_factors),
        "marginal_posterior_prob_positive": _effect_matrix_shape(num_traits, num_factors),
        "marginal_ln_bf": _effect_matrix_shape(num_traits, num_factors),
        "joint_lift": _effect_matrix_shape(num_traits, num_factors),
        "joint_posterior_lift": _effect_matrix_shape(num_traits, num_factors),
        "joint_posterior_sd": _effect_matrix_shape(num_traits, num_factors),
        "joint_posterior_prob_positive": _effect_matrix_shape(num_traits, num_factors),
        "joint_ln_bf": _effect_matrix_shape(num_traits, num_factors),
        "joint_conditioning_num_factors": np.full((num_traits, num_factors), max(0, num_factors - 1), dtype=int),
        "joint_conditioning_ridge_lambda": np.full((num_traits, num_factors), float(ridge_lambda), dtype=float),
        "joint_model_available": np.full((num_traits, num_factors), True, dtype=bool),
        "joint_unavailable_reason": np.full((num_traits, num_factors), "", dtype=object),
        "anchor_conditional_lift": _effect_matrix_shape(num_traits, num_factors),
        "anchor_conditional_posterior_lift": _effect_matrix_shape(num_traits, num_factors),
        "anchor_conditional_posterior_sd": _effect_matrix_shape(num_traits, num_factors),
        "anchor_conditional_posterior_prob_positive": _effect_matrix_shape(num_traits, num_factors),
        "anchor_conditional_ln_bf": _effect_matrix_shape(num_traits, num_factors),
        "anchor_conditional_available": np.full((num_traits, num_factors), False, dtype=bool),
        "anchor_conditional_unavailable_reason": np.full((num_traits, num_factors), "anchor_support_unavailable", dtype=object),
    }

    anchor_design = None
    if anchor_feature_by_covariate is not None:
        anchor_dense = _sanitize_nonfinite(anchor_feature_by_covariate)
        if anchor_dense is not None and anchor_dense.shape[0] == num_features and anchor_dense.shape[1] > 0:
            anchor_design = np.column_stack([np.ones(num_features, dtype=float), anchor_dense])

    for factor_index in range(num_factors):
        m = M[:, factor_index]
        m_sum = float(np.sum(m))
        out = 1.0 - m
        out_sum = float(np.sum(out))
        mean_in = np.asarray(m @ Y / max(m_sum, eps), dtype=float)
        mean_out = np.asarray(out @ Y / max(out_sum, eps), dtype=float)
        results["marginal_mean_in"][:, factor_index] = mean_in
        results["marginal_mean_out"][:, factor_index] = mean_out
        results["marginal_lift"][:, factor_index] = mean_in - mean_out
        marginal_fit = _one_covariate_bayes_lm_matrix(Y, m, effect_prior_sd, eps=eps)
        results["marginal_posterior_lift"][:, factor_index] = marginal_fit["posterior_mean"]
        results["marginal_posterior_sd"][:, factor_index] = marginal_fit["posterior_sd"]
        results["marginal_posterior_prob_positive"][:, factor_index] = marginal_fit["posterior_prob_positive"]
        results["marginal_ln_bf"][:, factor_index] = marginal_fit["ln_bf"]

        other = np.delete(M, factor_index, axis=1)
        joint_design = np.column_stack([ones, other])
        xTy, yTy, sxx, df = _residual_crossproducts_against_design(
            Y,
            m,
            joint_design,
            ridge_lambda,
            eps=eps,
        )
        joint_fit = _one_covariate_bayes_lm_from_crossproducts(
            xTy,
            yTy,
            sxx,
            num_features,
            effect_prior_sd,
            df=df,
            eps=eps,
        )
        results["joint_lift"][:, factor_index] = joint_fit["beta_hat"]
        results["joint_posterior_lift"][:, factor_index] = joint_fit["posterior_mean"]
        results["joint_posterior_sd"][:, factor_index] = joint_fit["posterior_sd"]
        results["joint_posterior_prob_positive"][:, factor_index] = joint_fit["posterior_prob_positive"]
        results["joint_ln_bf"][:, factor_index] = joint_fit["ln_bf"]
        results["joint_model_available"][:, factor_index] = joint_fit["available"]
        results["joint_unavailable_reason"][:, factor_index] = joint_fit["unavailable_reason"]

        if anchor_design is not None:
            xTy, yTy, sxx, df = _residual_crossproducts_against_design(
                Y,
                m,
                anchor_design,
                ridge_lambda,
                eps=eps,
            )
            anchor_fit = _one_covariate_bayes_lm_from_crossproducts(
                xTy,
                yTy,
                sxx,
                num_features,
                effect_prior_sd,
                df=df,
                eps=eps,
            )
            results["anchor_conditional_lift"][:, factor_index] = anchor_fit["beta_hat"]
            results["anchor_conditional_posterior_lift"][:, factor_index] = anchor_fit["posterior_mean"]
            results["anchor_conditional_posterior_sd"][:, factor_index] = anchor_fit["posterior_sd"]
            results["anchor_conditional_posterior_prob_positive"][:, factor_index] = anchor_fit["posterior_prob_positive"]
            results["anchor_conditional_ln_bf"][:, factor_index] = anchor_fit["ln_bf"]
            results["anchor_conditional_available"][:, factor_index] = anchor_fit["available"]
            results["anchor_conditional_unavailable_reason"][:, factor_index] = anchor_fit["unavailable_reason"]

    for prefix in ("marginal", "joint", "anchor_conditional"):
        posterior = np.asarray(results["%s_posterior_lift" % prefix], dtype=float)
        ln_bf = np.asarray(results["%s_ln_bf" % prefix], dtype=float)
        available = np.isfinite(posterior) & np.isfinite(ln_bf)
        if prefix == "anchor_conditional":
            available &= np.asarray(results["anchor_conditional_available"], dtype=bool)
        notable = (
            available
            & (posterior > 0.0)
            & (ln_bf >= float(notable_ln_bf))
            & (trait_n_eff[:, np.newaxis] >= float(effect_min_trait_neff))
            & (retained_fraction[:, np.newaxis] >= float(effect_min_retained_fraction))
            & (~low_retention_flag[:, np.newaxis])
        )
        score = np.maximum(0.0, np.nan_to_num(posterior, nan=0.0)) * np.minimum(
            1.0,
            np.maximum(0.0, np.nan_to_num(ln_bf, nan=0.0)) / max(float(notable_ln_bf_scale), eps),
        )
        results["%s_notable" % prefix] = notable
        results["%s_notable_score" % prefix] = score
    return results


def compute_trait_linkage(
    nnls_project_fn,
    basis,
    feature_by_trait,
    *,
    full_feature_by_trait=None,
    evidence_feature_by_trait=None,
    anchor_feature_by_covariate=None,
    basis_mask=None,
    threshold_mode="weighted_thresholded",
    threshold_value=1.0,
    evidence_source_name=None,
    effect_input_transform="weighted_thresholded",
    effect_threshold=1.0,
    effect_prior_sd=1.0,
    effect_min_trait_neff=10.0,
    effect_min_retained_fraction=0.1,
    notable_ln_bf=3.0,
    notable_ln_bf_scale=5.0,
    membership_normalization="max",
    membership_cap=1.0,
    anchor_source_name="auto",
    strict_threshold=True,
    computation_mode="sparse_full",
    eps=1e-12,
):
    if computation_mode not in {"dense_full", "sparse_full"}:
        raise ValueError("Unknown trait linkage computation mode: %s" % computation_mode)
    dense_basis = _sanitize_nonfinite(basis)
    target_feature_by_trait = _sanitize_nonfinite_preserve_sparse(feature_by_trait)
    full_target_feature_by_trait = _sanitize_nonfinite_preserve_sparse(
        full_feature_by_trait if full_feature_by_trait is not None else feature_by_trait
    )

    if dense_basis is None or target_feature_by_trait is None or full_target_feature_by_trait is None:
        return None
    if full_target_feature_by_trait.shape[1] != target_feature_by_trait.shape[1]:
        raise ValueError(
            "Trait linkage full/target column mismatch: %s vs %s"
            % (full_target_feature_by_trait.shape, target_feature_by_trait.shape)
        )

    expected_num_rows = full_target_feature_by_trait.shape[0]
    if basis_mask is None:
        retained_basis_mask = np.full(expected_num_rows, True, dtype=bool)
        if dense_basis.shape[0] != expected_num_rows:
            raise ValueError(
                "Trait linkage basis rows %s must match feature rows %s when no basis mask is provided"
                % (dense_basis.shape[0], expected_num_rows)
            )
        retained_basis = np.asarray(dense_basis, dtype=float)
        full_basis = np.asarray(dense_basis, dtype=float)
    else:
        retained_basis_mask = np.asarray(basis_mask, dtype=bool)
        if retained_basis_mask.ndim != 1 or retained_basis_mask.shape[0] != expected_num_rows:
            raise ValueError(
                "Trait linkage basis mask length %s must match feature rows %s"
                % (retained_basis_mask.shape[0] if retained_basis_mask.ndim == 1 else "invalid", expected_num_rows)
            )
        if dense_basis.shape[0] == expected_num_rows:
            retained_basis = np.asarray(dense_basis[retained_basis_mask, :], dtype=float)
        elif dense_basis.shape[0] == int(np.sum(retained_basis_mask)):
            retained_basis = np.asarray(dense_basis, dtype=float)
        else:
            raise ValueError(
                "Trait linkage basis rows %s must match mask length %s or kept rows %s"
                % (dense_basis.shape[0], expected_num_rows, int(np.sum(retained_basis_mask)))
            )
        full_basis, retained_basis_mask = _build_full_basis_matrix(
            dense_basis,
            retained_basis_mask,
            expected_num_rows=expected_num_rows,
        )

    full_trait_support = eaggl_phenotype_annotation.prepare_thresholded_profile_input(
        full_target_feature_by_trait,
        threshold_mode,
        threshold_value=threshold_value,
        strict_threshold=strict_threshold,
    )
    if computation_mode == "dense_full":
        masked_full_trait_support = np.asarray(full_trait_support, dtype=float).copy()
        masked_full_trait_support[~retained_basis_mask, :] = 0.0
        masked_trait_support = np.asarray(masked_full_trait_support[retained_basis_mask, :], dtype=float)
    else:
        masked_full_trait_support = _mask_full_profile_rows(full_trait_support, retained_basis_mask)
        masked_trait_support = masked_full_trait_support[retained_basis_mask, :]

    total_trait_support = eaggl_phenotype_annotation.compute_profile_strengths(full_trait_support)
    retained_trait_support = eaggl_phenotype_annotation.compute_profile_strengths(masked_trait_support)
    total_feature_counts = _compute_positive_counts(full_trait_support)
    retained_feature_counts = _compute_positive_counts(masked_trait_support)
    trait_n_eff = _compute_effective_feature_count(full_trait_support, eps=eps)
    retained_n_eff = _compute_effective_feature_count(masked_trait_support, eps=eps)

    factor_total_mass = np.sum(retained_basis, axis=0, keepdims=True)
    clipped_factor_basis = np.maximum(np.asarray(retained_basis, dtype=float), 0.0)
    factor_positive_mass = np.sum(clipped_factor_basis, axis=0)
    factor_square_mass = np.sum(np.square(clipped_factor_basis), axis=0)
    factor_n_eff = np.zeros_like(factor_positive_mass, dtype=float)
    np.divide(
        np.square(factor_positive_mass),
        np.maximum(factor_square_mass, eps),
        out=factor_n_eff,
        where=factor_positive_mass > eps,
    )
    factor_top_share = np.zeros_like(factor_positive_mass, dtype=float)
    if clipped_factor_basis.shape[0] > 0:
        np.divide(
            np.max(clipped_factor_basis, axis=0),
            np.maximum(factor_positive_mass, eps),
            out=factor_top_share,
            where=factor_positive_mass > eps,
        )
    factor_top10_share = np.zeros_like(factor_positive_mass, dtype=float)
    if clipped_factor_basis.shape[0] > 0:
        top_count = min(10, clipped_factor_basis.shape[0])
        top10_mass = np.sum(np.sort(clipped_factor_basis, axis=0)[-top_count:, :], axis=0)
        np.divide(
            top10_mass,
            np.maximum(factor_positive_mass, eps),
            out=factor_top10_share,
            where=factor_positive_mass > eps,
        )
    broad_factor_flag = np.logical_and(factor_n_eff >= 500.0, factor_top_share <= 0.01)
    normalized_factor_basis_retained = retained_basis / np.maximum(factor_total_mass, eps)
    normalized_trait_support_for_projection = _normalize_profiles_by_strength(
        masked_trait_support,
        total_trait_support,
        eps=eps,
    )
    normalized_factor_basis = full_basis / np.maximum(factor_total_mass, eps)
    if computation_mode == "dense_full":
        normalized_trait_support = masked_full_trait_support / np.maximum(total_trait_support[np.newaxis, :], eps)
    else:
        normalized_trait_support = _normalize_profiles_by_strength(
            masked_full_trait_support,
            total_trait_support,
            eps=eps,
        )

    joint = np.asarray(
        nnls_project_fn(normalized_factor_basis_retained, normalized_trait_support_for_projection.T, max_sum=1.0),
        dtype=float,
    )
    marginal_numerator = normalized_trait_support_for_projection.T @ normalized_factor_basis_retained
    if sparse.issparse(marginal_numerator):
        marginal_numerator = marginal_numerator.toarray()
    marginal_numerator = np.asarray(marginal_numerator, dtype=float)
    marginal_overlap = np.maximum(marginal_numerator, 0.0)
    marginal_denominator = np.sum(np.square(normalized_factor_basis_retained), axis=0, dtype=float)
    marginal = np.zeros_like(marginal_numerator, dtype=float)
    np.divide(
        marginal_numerator,
        marginal_denominator[np.newaxis, :],
        out=marginal,
        where=marginal_denominator[np.newaxis, :] > eps,
    )
    marginal = np.clip(marginal, 0.0, 1.0)

    residual = np.maximum(0.0, 1.0 - np.sum(joint, axis=1))
    retained_fraction = retained_trait_support / np.maximum(total_trait_support, eps)
    low_retention_flag = np.logical_or.reduce((
        retained_feature_counts < 5,
        retained_fraction < 0.1,
        retained_n_eff < 5.0,
    ))

    result = {
        "prepared_feature_by_trait": masked_trait_support,
        "masked_trait_support": masked_trait_support,
        "full_trait_support": full_trait_support,
        "normalized_trait_support": normalized_trait_support,
        "normalized_factor_basis": np.asarray(normalized_factor_basis, dtype=float),
        "trait_total_support": np.asarray(total_trait_support, dtype=float),
        "retained_trait_support": np.asarray(retained_trait_support, dtype=float),
        "retained_fraction": np.asarray(retained_fraction, dtype=float),
        "total_feature_count": np.asarray(total_feature_counts, dtype=int),
        "retained_feature_count": np.asarray(retained_feature_counts, dtype=int),
        "trait_n_eff": np.asarray(trait_n_eff, dtype=float),
        "retained_n_eff": np.asarray(retained_n_eff, dtype=float),
        "low_retention_flag": np.asarray(low_retention_flag, dtype=bool),
        "factor_total_mass": np.asarray(np.ravel(factor_total_mass), dtype=float),
        "factor_n_eff": np.asarray(factor_n_eff, dtype=float),
        "factor_top_share": np.asarray(factor_top_share, dtype=float),
        "factor_top10_share": np.asarray(factor_top10_share, dtype=float),
        "broad_factor_flag": np.asarray(broad_factor_flag, dtype=bool),
        "joint": joint,
        "marginal": marginal,
        "marginal_overlap": marginal_overlap,
        "residual": residual,
        "evidence_source": evidence_source_name,
        "effect_input_transform": effect_input_transform,
        "effect_threshold": float(effect_threshold),
        "effect_prior_sd": float(effect_prior_sd),
        "anchor_source": anchor_source_name,
    }
    evidence_matrix = evidence_feature_by_trait if evidence_feature_by_trait is not None else full_target_feature_by_trait
    effect_scores = _compute_effect_scores(
        full_basis,
        evidence_matrix,
        anchor_feature_by_covariate=anchor_feature_by_covariate,
        effect_input_transform=effect_input_transform,
        effect_threshold=effect_threshold,
        effect_prior_sd=effect_prior_sd,
        effect_min_trait_neff=effect_min_trait_neff,
        effect_min_retained_fraction=effect_min_retained_fraction,
        notable_ln_bf=notable_ln_bf,
        notable_ln_bf_scale=notable_ln_bf_scale,
        membership_normalization=membership_normalization,
        membership_cap=membership_cap,
        trait_n_eff=trait_n_eff,
        retained_fraction=retained_fraction,
        low_retention_flag=low_retention_flag,
        eps=eps,
    )
    result.update(effect_scores)
    return result
