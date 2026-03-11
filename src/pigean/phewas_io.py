from __future__ import annotations


def read_gene_phewas_bfs(
    state,
    gene_phewas_bfs_in,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    anchor_genes=None,
    anchor_phenos=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
    phewas_gene_to_X_gene_in=None,
    min_value=None,
    max_num_entries_at_once=None,
    *,
    parse_gene_map_file_fn,
    load_and_apply_gene_phewas_bfs_fn,
    sync_phewas_runtime_state_fn,
    construct_map_to_ind_fn,
    open_text_fn,
    get_col_fn,
    warn_fn,
    bail_fn,
    log_fn,
    info_level,
    debug_level,
    **kwargs
):
    cached = dict(locals())
    for key in (
        "state",
        "parse_gene_map_file_fn",
        "load_and_apply_gene_phewas_bfs_fn",
        "sync_phewas_runtime_state_fn",
        "construct_map_to_ind_fn",
        "open_text_fn",
        "get_col_fn",
        "warn_fn",
        "bail_fn",
        "log_fn",
        "info_level",
        "debug_level",
        "kwargs",
    ):
        cached.pop(key, None)
    state.cached_gene_phewas_call = cached

    if gene_phewas_bfs_in is None:
        bail_fn("Require --gene-stats-in or --gene-phewas-bfs-in for this operation")

    log_fn("Reading --gene-phewas-bfs-in file %s" % gene_phewas_bfs_in, info_level)
    if state.genes is None:
        bail_fn("Need to initialixe --X before reading gene_phewas")

    phewas_gene_to_X_gene = None
    if phewas_gene_to_X_gene_in is not None:
        phewas_gene_to_X_gene = parse_gene_map_file_fn(
            phewas_gene_to_X_gene_in,
            allow_multi=True,
            bail_fn=bail_fn,
        )

    load_and_apply_gene_phewas_bfs_fn(
        state,
        gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        anchor_genes=anchor_genes,
        anchor_phenos=anchor_phenos,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        phewas_gene_to_x_gene=phewas_gene_to_X_gene,
        min_value=min_value,
        max_num_entries_at_once=max_num_entries_at_once,
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        construct_map_to_ind_fn=construct_map_to_ind_fn,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
        log_fn=lambda message: log_fn(message, debug_level),
    )
    state.phewas_state = sync_phewas_runtime_state_fn(state)


def reread_gene_phewas_bfs(state, *, read_gene_phewas_bfs_fn):
    cached_call = getattr(state, "cached_gene_phewas_call", None)
    if cached_call is None:
        return
    read_gene_phewas_bfs_fn(state, **cached_call)
