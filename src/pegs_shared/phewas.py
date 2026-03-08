import copy
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse

from pegs_shared.io_common import construct_map_to_ind, resolve_column_index
from pegs_shared.types import ParsedGenePhewasBfs, PhewasFileColumnInfo


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


def resolve_phewas_file_columns(
    header_cols,
    *,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
    get_col_fn=None,
):
    if get_col_fn is None:
        get_col_fn = resolve_column_index
    id_col_name = gene_phewas_bfs_id_col if gene_phewas_bfs_id_col is not None else "Gene"
    pheno_col_name = gene_phewas_bfs_pheno_col if gene_phewas_bfs_pheno_col is not None else "Pheno"
    return PhewasFileColumnInfo(
        id_col=get_col_fn(id_col_name, header_cols),
        pheno_col=get_col_fn(pheno_col_name, header_cols),
        bf_col=get_col_fn(gene_phewas_bfs_log_bf_col, header_cols) if gene_phewas_bfs_log_bf_col is not None else get_col_fn("log_bf", header_cols, False),
        combined_col=get_col_fn(gene_phewas_bfs_combined_col, header_cols, True) if gene_phewas_bfs_combined_col is not None else get_col_fn("combined", header_cols, False),
        prior_col=get_col_fn(gene_phewas_bfs_prior_col, header_cols, True) if gene_phewas_bfs_prior_col is not None else get_col_fn("prior", header_cols, False),
    )


def expand_phewas_state_for_added_phenos(runtime, num_added_phenos):
    if num_added_phenos <= 0:
        return
    if runtime.X_phewas_beta is not None:
        runtime.X_phewas_beta = sparse.csc_matrix(
            sparse.vstack((runtime.X_phewas_beta, sparse.csc_matrix((num_added_phenos, runtime.X_phewas_beta.shape[1]))))
        )
    if runtime.X_phewas_beta_uncorrected is not None:
        runtime.X_phewas_beta_uncorrected = sparse.csc_matrix(
            sparse.vstack((runtime.X_phewas_beta_uncorrected, sparse.csc_matrix((num_added_phenos, runtime.X_phewas_beta_uncorrected.shape[1]))))
        )
    if runtime.gene_pheno_Y is not None:
        runtime.gene_pheno_Y = sparse.csc_matrix(
            sparse.hstack((runtime.gene_pheno_Y, sparse.csc_matrix((runtime.gene_pheno_Y.shape[0], num_added_phenos))))
        )
    if runtime.gene_pheno_combined_prior_Ys is not None:
        runtime.gene_pheno_combined_prior_Ys = sparse.csc_matrix(
            sparse.hstack((runtime.gene_pheno_combined_prior_Ys, sparse.csc_matrix((runtime.gene_pheno_combined_prior_Ys.shape[0], num_added_phenos))))
        )
    if runtime.gene_pheno_priors is not None:
        runtime.gene_pheno_priors = sparse.csc_matrix(
            sparse.hstack((runtime.gene_pheno_priors, sparse.csc_matrix((runtime.gene_pheno_priors.shape[0], num_added_phenos))))
        )


def prepare_phewas_phenos_from_file(
    runtime,
    gene_phewas_bfs_in,
    *,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
    open_text_fn=None,
    get_col_fn=None,
    construct_map_to_ind_fn=None,
    expand_state_fn=None,
    warn_fn=None,
    log_fn=None,
    debug_level=0,
):
    if open_text_fn is None:
        open_text_fn = lambda path: open(path)
    if get_col_fn is None:
        get_col_fn = resolve_column_index
    if construct_map_to_ind_fn is None:
        construct_map_to_ind_fn = construct_map_to_ind
    if expand_state_fn is None:
        expand_state_fn = expand_phewas_state_for_added_phenos
    if warn_fn is None:
        warn_fn = lambda _msg: None
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None

    if runtime.phenos is not None:
        phenos = copy.copy(runtime.phenos)
        pheno_to_ind = copy.copy(runtime.pheno_to_ind)
    else:
        phenos = []
        pheno_to_ind = {}

    runtime.num_gene_phewas_filtered = 0
    with open_text_fn(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
        log_fn("Fetching phenotypes to use", debug_level)
        header_cols = gene_phewas_bfs_fh.readline().strip("\n").split()
        col_info = resolve_phewas_file_columns(
            header_cols=header_cols,
            gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
            gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
            gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
            gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
            gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
            get_col_fn=get_col_fn,
        )
        for line in gene_phewas_bfs_fh:
            cols = line.strip("\n").split()
            if (
                col_info.id_col >= len(cols)
                or col_info.pheno_col >= len(cols)
                or (col_info.bf_col is not None and col_info.bf_col >= len(cols))
                or (col_info.combined_col is not None and col_info.combined_col >= len(cols))
                or (col_info.prior_col is not None and col_info.prior_col >= len(cols))
            ):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            gene = cols[col_info.id_col]
            if runtime.gene_label_map is not None and gene in runtime.gene_label_map:
                gene = runtime.gene_label_map[gene]
            if gene not in runtime.gene_to_ind:
                continue

            pheno = cols[col_info.pheno_col]
            if pheno not in pheno_to_ind:
                pheno_to_ind[pheno] = len(phenos)
                phenos.append(pheno)

    prior_num_phenos = len(runtime.phenos) if runtime.phenos is not None else 0
    expand_state_fn(runtime, len(phenos) - prior_num_phenos)
    runtime.phenos = phenos
    runtime.pheno_to_ind = construct_map_to_ind_fn(phenos)
    return phenos, runtime.pheno_to_ind, col_info


def read_phewas_file_batch(
    runtime,
    gene_phewas_bfs_in,
    *,
    begin,
    cur_batch_size,
    pheno_to_ind,
    col_info,
    open_text_fn=None,
    warn_fn=None,
):
    if open_text_fn is None:
        open_text_fn = lambda path: open(path)
    if warn_fn is None:
        warn_fn = lambda _msg: None
    if isinstance(col_info, dict):
        col_info = PhewasFileColumnInfo(
            id_col=col_info["id_col"],
            pheno_col=col_info["pheno_col"],
            bf_col=col_info["bf_col"],
            combined_col=col_info["combined_col"],
            prior_col=col_info["prior_col"],
        )

    gene_pheno_Y = np.zeros((len(runtime.genes), cur_batch_size)) if col_info.bf_col is not None else None
    gene_pheno_combined_prior_Ys = np.zeros((len(runtime.genes), cur_batch_size)) if col_info.combined_col is not None else None
    gene_pheno_priors = np.zeros((len(runtime.genes), cur_batch_size)) if col_info.prior_col is not None else None

    with open_text_fn(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
        for line in gene_phewas_bfs_fh:
            cols = line.strip("\n").split()
            if (
                col_info.id_col >= len(cols)
                or col_info.pheno_col >= len(cols)
                or (col_info.bf_col is not None and col_info.bf_col >= len(cols))
                or (col_info.combined_col is not None and col_info.combined_col >= len(cols))
                or (col_info.prior_col is not None and col_info.prior_col >= len(cols))
            ):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            gene = cols[col_info.id_col]
            if runtime.gene_label_map is not None and gene in runtime.gene_label_map:
                gene = runtime.gene_label_map[gene]
            if gene not in runtime.gene_to_ind:
                continue

            pheno = cols[col_info.pheno_col]
            if pheno not in pheno_to_ind:
                continue
            pheno_ind = pheno_to_ind[pheno] - begin
            if pheno_ind < 0 or pheno_ind >= cur_batch_size:
                continue

            gene_ind = runtime.gene_to_ind[gene]
            if col_info.combined_col is not None:
                try:
                    combined = float(cols[col_info.combined_col])
                except ValueError:
                    if cols[col_info.combined_col] != "NA":
                        warn_fn("Skipping unconvertible value %s for gene %s" % (cols[col_info.combined_col], gene))
                    continue
                gene_pheno_combined_prior_Ys[gene_ind, pheno_ind] = combined

            if col_info.bf_col is not None:
                try:
                    bf = float(cols[col_info.bf_col])
                except ValueError:
                    if cols[col_info.bf_col] != "NA":
                        warn_fn("Skipping unconvertible value %s for gene %s and pheno %s" % (cols[col_info.bf_col], gene, pheno))
                    continue
                gene_pheno_Y[gene_ind, pheno_ind] = bf

            if col_info.prior_col is not None:
                try:
                    prior = float(cols[col_info.prior_col])
                except ValueError:
                    if cols[col_info.prior_col] != "NA":
                        warn_fn("Skipping unconvertible value %s for gene %s" % (cols[col_info.prior_col], gene))
                    continue
                gene_pheno_priors[gene_ind, pheno_ind] = prior

    return gene_pheno_Y, gene_pheno_combined_prior_Ys, gene_pheno_priors


def append_phewas_metric_block(
    current_beta,
    current_beta_tilde,
    current_se,
    current_z,
    current_p_value,
    current_one_sided_p_value,
    beta,
    beta_tilde,
    se,
    z_score,
    p_value,
    one_sided_p_value,
):
    if current_beta_tilde is None:
        return beta, beta_tilde, se, z_score, p_value, one_sided_p_value
    beta_append = np.hstack((current_beta, beta)) if current_beta is not None else None
    one_sided_append = np.hstack((current_one_sided_p_value, one_sided_p_value)) if current_one_sided_p_value is not None else None
    return (
        beta_append,
        np.hstack((current_beta_tilde, beta_tilde)),
        np.hstack((current_se, se)),
        np.hstack((current_z, z_score)),
        np.hstack((current_p_value, p_value)),
        one_sided_append,
    )


def accumulate_standard_phewas_outputs(runtime, output_prefix, beta, beta_tilde, se, z_score, p_value):
    input_axes = [
        ("Y", runtime.Y is not None, 0),
        ("combined_prior_Ys", runtime.combined_prior_Ys is not None, 1),
        ("priors", runtime.priors is not None, 2),
    ]
    for axis_name, axis_enabled, axis_index in input_axes:
        if not axis_enabled:
            continue
        output_base = "%s_vs_input_%s" % (output_prefix, axis_name)
        (
            updated_beta,
            updated_beta_tilde,
            updated_se,
            updated_z,
            updated_p_value,
            _,
        ) = append_phewas_metric_block(
            getattr(runtime, "%s_beta" % output_base),
            getattr(runtime, "%s_beta_tilde" % output_base),
            getattr(runtime, "%s_se" % output_base),
            getattr(runtime, "%s_Z" % output_base),
            getattr(runtime, "%s_p_value" % output_base),
            None,
            beta[axis_index, :],
            beta_tilde[axis_index, :],
            se[axis_index, :],
            z_score[axis_index, :],
            p_value[axis_index, :],
            None,
        )
        setattr(runtime, "%s_beta" % output_base, updated_beta)
        setattr(runtime, "%s_beta_tilde" % output_base, updated_beta_tilde)
        setattr(runtime, "%s_se" % output_base, updated_se)
        setattr(runtime, "%s_Z" % output_base, updated_z)
        setattr(runtime, "%s_p_value" % output_base, updated_p_value)


def accumulate_factor_phewas_outputs(runtime, output_prefix, beta_tilde, se, z_score, p_value, one_sided_p_value, *, huber=False):
    output_base = "factor_phewas_%s" % output_prefix
    if huber:
        output_base += "_huber"
    (
        updated_beta,
        updated_beta_tilde,
        updated_se,
        updated_z,
        updated_p_value,
        updated_one_sided,
    ) = append_phewas_metric_block(
        None,
        getattr(runtime, "%s_betas" % output_base),
        getattr(runtime, "%s_ses" % output_base),
        getattr(runtime, "%s_zs" % output_base),
        getattr(runtime, "%s_p_values" % output_base),
        getattr(runtime, "%s_one_sided_p_values" % output_base),
        None,
        beta_tilde,
        se,
        z_score,
        p_value,
        one_sided_p_value,
    )
    assert updated_beta is None
    setattr(runtime, "%s_betas" % output_base, updated_beta_tilde)
    setattr(runtime, "%s_ses" % output_base, updated_se)
    setattr(runtime, "%s_zs" % output_base, updated_z)
    setattr(runtime, "%s_p_values" % output_base, updated_p_value)
    setattr(runtime, "%s_one_sided_p_values" % output_base, updated_one_sided)


def parse_gene_phewas_bfs_file(
    gene_phewas_bfs_in,
    *,
    gene_phewas_bfs_id_col,
    gene_phewas_bfs_pheno_col,
    gene_phewas_bfs_log_bf_col,
    gene_phewas_bfs_combined_col,
    gene_phewas_bfs_prior_col,
    min_value,
    max_num_entries_at_once,
    existing_phenos,
    existing_pheno_to_ind,
    gene_to_ind,
    gene_label_map,
    phewas_gene_to_x_gene,
    open_text_fn,
    get_col_fn,
    bail_fn=None,
    warn_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if max_num_entries_at_once is None:
        max_num_entries_at_once = 200 * 10000

    success = False
    num_filtered = 0
    final_phenos = list(existing_phenos) if existing_phenos is not None else []
    final_pheno_to_ind = copy.copy(existing_pheno_to_ind) if existing_pheno_to_ind is not None else {}
    final_row = np.array([], dtype=np.int32)
    final_col = np.array([], dtype=np.int32)
    final_Ys = None
    final_combineds = None
    final_priors = None

    for delim in [None, "\t"]:
        success = True
        Ys = None
        combineds = None
        priors = None

        row = []
        col = []
        row_chunks = []
        col_chunks = []
        Y_chunks = []
        combined_chunks = []
        prior_chunks = []

        with open_text_fn(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
            header_cols = gene_phewas_bfs_fh.readline().strip("\n").split(delim)
            id_col_name = gene_phewas_bfs_id_col if gene_phewas_bfs_id_col is not None else "Gene"
            pheno_col_name = gene_phewas_bfs_pheno_col if gene_phewas_bfs_pheno_col is not None else "Pheno"

            id_col = get_col_fn(id_col_name, header_cols)
            pheno_col = get_col_fn(pheno_col_name, header_cols)

            if gene_phewas_bfs_log_bf_col is not None:
                bf_col = get_col_fn(gene_phewas_bfs_log_bf_col, header_cols)
            else:
                bf_col = get_col_fn("log_bf", header_cols, False)

            if gene_phewas_bfs_combined_col is not None:
                combined_col = get_col_fn(gene_phewas_bfs_combined_col, header_cols, True)
            else:
                combined_col = get_col_fn("combined", header_cols, False)

            if gene_phewas_bfs_prior_col is not None:
                prior_col = get_col_fn(gene_phewas_bfs_prior_col, header_cols, True)
            else:
                prior_col = get_col_fn("prior", header_cols, False)

            if bf_col is not None:
                Ys = []
            if combined_col is not None:
                combineds = []
            if prior_col is not None:
                priors = []

            def _flush_chunks():
                if len(row) == 0:
                    return
                row_chunks.append(np.array(row, dtype=np.int32))
                col_chunks.append(np.array(col, dtype=np.int32))
                if Ys is not None:
                    Y_chunks.append(np.array(Ys, dtype=np.float64))
                    Ys[:] = []
                if combineds is not None:
                    combined_chunks.append(np.array(combineds, dtype=np.float64))
                    combineds[:] = []
                if priors is not None:
                    prior_chunks.append(np.array(priors, dtype=np.float64))
                    priors[:] = []
                row[:] = []
                col[:] = []

            phenos = list(existing_phenos) if existing_phenos is not None else []
            pheno_to_ind = (
                copy.copy(existing_pheno_to_ind) if existing_pheno_to_ind is not None else {}
            )
            num_filtered = 0

            for line in gene_phewas_bfs_fh:
                cols = line.strip("\n").split(delim)
                if len(cols) != len(header_cols):
                    success = False
                    continue

                if (
                    id_col >= len(cols)
                    or pheno_col >= len(cols)
                    or (bf_col is not None and bf_col >= len(cols))
                    or (combined_col is not None and combined_col >= len(cols))
                    or (prior_col is not None and prior_col >= len(cols))
                ):
                    warn_fn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                pheno = cols[pheno_col]

                cur_combined = None
                if combined_col is not None:
                    try:
                        combined = float(cols[combined_col])
                    except ValueError:
                        if cols[combined_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene_set %s"
                                % (cols[combined_col], gene)
                            )
                        continue

                    if min_value is not None and combined < min_value:
                        num_filtered += 1
                        continue
                    cur_combined = combined

                cur_Y = None
                if bf_col is not None:
                    try:
                        bf = float(cols[bf_col])
                    except ValueError:
                        if cols[bf_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene %s and pheno %s"
                                % (cols[bf_col], gene, pheno)
                            )
                        continue

                    if min_value is not None and combined_col is None and bf < min_value:
                        num_filtered += 1
                        continue
                    cur_Y = bf

                cur_prior = None
                if prior_col is not None:
                    try:
                        prior = float(cols[prior_col])
                    except ValueError:
                        if cols[prior_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene %s"
                                % (cols[prior_col], gene)
                            )
                        continue

                    if min_value is not None and combined_col is None and bf_col is None and prior < min_value:
                        num_filtered += 1
                        continue
                    cur_prior = prior

                if pheno not in pheno_to_ind:
                    pheno_to_ind[pheno] = len(phenos)
                    phenos.append(pheno)
                pheno_ind = pheno_to_ind[pheno]

                if gene_label_map is not None and gene in gene_label_map:
                    gene = gene_label_map[gene]

                mapped_genes = [gene]
                if phewas_gene_to_x_gene is not None and gene in phewas_gene_to_x_gene:
                    mapped_genes = list(phewas_gene_to_x_gene[gene])

                for cur_gene in mapped_genes:
                    if cur_gene not in gene_to_ind:
                        continue
                    if combineds is not None:
                        combineds.append(cur_combined)
                    if Ys is not None:
                        Ys.append(cur_Y)
                    if priors is not None:
                        priors.append(cur_prior)

                    col.append(pheno_ind)
                    row.append(gene_to_ind[cur_gene])
                    if len(row) >= max_num_entries_at_once:
                        _flush_chunks()

            _flush_chunks()

        if success:
            final_phenos = phenos
            final_pheno_to_ind = pheno_to_ind
            if len(row_chunks) > 0:
                row = np.concatenate(row_chunks)
                col = np.concatenate(col_chunks)
            else:
                row = np.array([], dtype=np.int32)
                col = np.array([], dtype=np.int32)

            if len(row) > 0:
                key = row.astype(np.int64) * int(len(phenos)) + col.astype(np.int64)
                _, unique_indices = np.unique(key, return_index=True)
            else:
                unique_indices = np.array([], dtype=np.int64)

            if len(unique_indices) < len(row):
                warn_fn("Found %d duplicate values; ignoring duplicates" % (len(row) - len(unique_indices)))

            final_row = row[unique_indices]
            final_col = col[unique_indices]

            if combineds is not None:
                if len(combined_chunks) > 0:
                    final_combineds = np.concatenate(combined_chunks)[unique_indices]
                else:
                    final_combineds = np.array([], dtype=np.float64)
            else:
                final_combineds = None

            if Ys is not None:
                if len(Y_chunks) > 0:
                    final_Ys = np.concatenate(Y_chunks)[unique_indices]
                else:
                    final_Ys = np.array([], dtype=np.float64)
            else:
                final_Ys = None

            if priors is not None:
                if len(prior_chunks) > 0:
                    final_priors = np.concatenate(prior_chunks)[unique_indices]
                else:
                    final_priors = np.array([], dtype=np.float64)
            else:
                final_priors = None
            break

    if not success:
        bail_fn("Error: different number of columns in header row and non header rows")

    return ParsedGenePhewasBfs(
        phenos=final_phenos,
        pheno_to_ind=final_pheno_to_ind,
        row=final_row,
        col=final_col,
        Ys=final_Ys,
        combineds=final_combineds,
        priors=final_priors,
        num_filtered=num_filtered,
    )
