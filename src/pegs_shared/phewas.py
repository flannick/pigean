import os
from dataclasses import dataclass

import numpy as np


def _default_bail(message):
    raise ValueError(message)


@dataclass
class PhewasRuntimeState:
    phenos: object = None
    pheno_to_ind: object = None
    gene_pheno_Y: object = None
    gene_pheno_combined_prior_Ys: object = None
    gene_pheno_priors: object = None
    X_phewas_beta: object = None
    X_phewas_beta_uncorrected: object = None
    num_gene_phewas_filtered: int = 0
    anchor_gene_mask: object = None
    anchor_pheno_mask: object = None


@dataclass
class FactorInputData:
    anchor_gene_mask: object = None
    anchor_pheno_mask: object = None
    loaded_gene_set_phewas_stats: bool = False
    loaded_gene_phewas_bfs: bool = False


@dataclass
class PhewasStageConfig:
    gene_phewas_bfs_in: object = None
    gene_phewas_bfs_id_col: object = None
    gene_phewas_bfs_pheno_col: object = None
    gene_phewas_bfs_log_bf_col: object = None
    gene_phewas_bfs_combined_col: object = None
    gene_phewas_bfs_prior_col: object = None
    max_num_burn_in: int = 1000
    max_num_iter: int = 1100
    min_num_iter: int = 10
    num_chains: int = 10
    r_threshold_burn_in: float = 1.01
    use_max_r_for_convergence: bool = True
    max_frac_sem: float = 0.01
    gauss_seidel: bool = False
    sparse_solution: bool = False
    sparse_frac_betas: object = None
    run_for_factors: bool = False
    batch_size: int | None = None
    min_gene_factor_weight: float = 0.0

    def to_run_kwargs(self):
        run_kwargs = {
            "gene_phewas_bfs_in": self.gene_phewas_bfs_in,
            "gene_phewas_bfs_id_col": self.gene_phewas_bfs_id_col,
            "gene_phewas_bfs_pheno_col": self.gene_phewas_bfs_pheno_col,
            "gene_phewas_bfs_log_bf_col": self.gene_phewas_bfs_log_bf_col,
            "gene_phewas_bfs_combined_col": self.gene_phewas_bfs_combined_col,
            "gene_phewas_bfs_prior_col": self.gene_phewas_bfs_prior_col,
            "max_num_burn_in": self.max_num_burn_in,
            "max_num_iter": self.max_num_iter,
            "min_num_iter": self.min_num_iter,
            "num_chains": self.num_chains,
            "r_threshold_burn_in": self.r_threshold_burn_in,
            "use_max_r_for_convergence": self.use_max_r_for_convergence,
            "max_frac_sem": self.max_frac_sem,
            "gauss_seidel": self.gauss_seidel,
            "sparse_solution": self.sparse_solution,
            "sparse_frac_betas": self.sparse_frac_betas,
        }
        if self.run_for_factors:
            run_kwargs["run_for_factors"] = True
            if self.batch_size is not None:
                run_kwargs["batch_size"] = self.batch_size
            run_kwargs["min_gene_factor_weight"] = self.min_gene_factor_weight
        return run_kwargs


@dataclass
class PhewasInputResolution:
    requested_input: object = None
    resolved_input: object = None
    mode: str = "skip"
    reason: str = "no_input_requested"

    @property
    def should_reuse_loaded_matrix(self):
        return self.mode == "reuse_loaded_matrix"

    @property
    def should_reread_file(self):
        return self.mode == "re_read_file"


def derive_factor_anchor_masks(genes, phenos, anchor_genes=None, anchor_phenos=None, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    anchor_gene_mask = None
    anchor_pheno_mask = None

    if anchor_genes is not None:
        anchor_gene_mask = np.array([x in anchor_genes for x in genes])
        if np.sum(anchor_gene_mask) == 0:
            bail_fn("None of the anchor genes are in X")

    if anchor_phenos is not None:
        anchor_pheno_mask = np.array([x in anchor_phenos for x in phenos])
        if np.sum(anchor_pheno_mask) == 0:
            bail_fn("None of the anchor phenos are in gene pheno matrix")

    return FactorInputData(
        anchor_gene_mask=anchor_gene_mask,
        anchor_pheno_mask=anchor_pheno_mask,
    )


def resolve_gene_phewas_input_for_stage(
    requested_input,
    reusable_inputs,
    *,
    read_gene_phewas,
    num_gene_phewas_filtered,
):
    decision = resolve_gene_phewas_input_decision_for_stage(
        requested_input=requested_input,
        reusable_inputs=reusable_inputs,
        read_gene_phewas=read_gene_phewas,
        num_gene_phewas_filtered=num_gene_phewas_filtered,
    )
    return decision.resolved_input


def _normalize_optional_path(path):
    if path is None:
        return None
    return os.path.realpath(os.path.abspath(path))


def _paths_match(a, b):
    if a is None or b is None:
        return False
    return _normalize_optional_path(a) == _normalize_optional_path(b)


def resolve_gene_phewas_input_decision_for_stage(
    requested_input,
    reusable_inputs,
    *,
    read_gene_phewas,
    num_gene_phewas_filtered,
):
    if requested_input is None:
        return PhewasInputResolution()

    if not read_gene_phewas:
        return PhewasInputResolution(
            requested_input=requested_input,
            resolved_input=requested_input,
            mode="re_read_file",
            reason="matrix_not_loaded",
        )

    if num_gene_phewas_filtered != 0:
        return PhewasInputResolution(
            requested_input=requested_input,
            resolved_input=requested_input,
            mode="re_read_file",
            reason="loaded_matrix_filtered",
        )

    for candidate in reusable_inputs:
        if _paths_match(requested_input, candidate):
            return PhewasInputResolution(
                requested_input=requested_input,
                resolved_input=None,
                mode="reuse_loaded_matrix",
                reason="requested_input_matches_loaded_source",
            )

    return PhewasInputResolution(
        requested_input=requested_input,
        resolved_input=requested_input,
        mode="re_read_file",
        reason="requested_input_not_reusable",
    )


def build_phewas_stage_config(
    *,
    gene_phewas_bfs_in,
    gene_phewas_bfs_id_col,
    gene_phewas_bfs_pheno_col,
    gene_phewas_bfs_log_bf_col,
    gene_phewas_bfs_combined_col,
    gene_phewas_bfs_prior_col,
    max_num_burn_in,
    max_num_iter,
    min_num_iter,
    num_chains,
    r_threshold_burn_in,
    use_max_r_for_convergence,
    max_frac_sem,
    gauss_seidel,
    sparse_solution,
    sparse_frac_betas,
    run_for_factors=False,
    batch_size=None,
    min_gene_factor_weight=0.0,
):
    return PhewasStageConfig(
        gene_phewas_bfs_in=gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        max_num_burn_in=max_num_burn_in,
        max_num_iter=max_num_iter,
        min_num_iter=min_num_iter,
        num_chains=num_chains,
        r_threshold_burn_in=r_threshold_burn_in,
        use_max_r_for_convergence=use_max_r_for_convergence,
        max_frac_sem=max_frac_sem,
        gauss_seidel=gauss_seidel,
        sparse_solution=sparse_solution,
        sparse_frac_betas=sparse_frac_betas,
        run_for_factors=run_for_factors,
        batch_size=batch_size,
        min_gene_factor_weight=min_gene_factor_weight,
    )
