"""Versioned sidecar for reproducible phenotype projection from supplied factors."""
import hashlib
import json
import numpy as np
from scipy import sparse

SETTINGS=('trait_linkage_source','trait_linkage_threshold','trait_linkage_computation_mode',
          'pheno_capture_input','trait_factor_linkage_factor_gene_threshold',
          'trait_factor_linkage_nnls_min_loading','trait_factor_linkage_nnls_max_value')


def fingerprint(arrays):
    h=hashlib.sha256()
    for name in sorted(arrays):
        a=np.ascontiguousarray(arrays[name])
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def save_context(state,path,options):
    if options.project_phenos_from_gene_sets or state.exp_gene_factors is None:
        raise ValueError('Internal projection context currently supports the gene basis only')
    if options.factor_output_scope!='all' or options.cluster_row_min_max_loading!=0:
        raise ValueError('Reproducible context requires all factors and unfiltered gene rows')
    if not hasattr(state,'internal_last_projection_phi'):
        raise ValueError('No phenotype projection context is available to save')
    n=len(state.genes);basis=np.asarray(state.exp_gene_factors,dtype=float)
    weights=getattr(state,'gene_prob_factor_vector',None)
    weights=weights.toarray() if sparse.issparse(weights) else weights
    weights=np.empty((n,0)) if weights is None else np.asarray(weights,dtype=float)
    mask=getattr(state,'gene_factor_gene_mask',None)
    ids=getattr(state,'supplied_factor_ids',tuple(range(1,basis.shape[1]+1)))
    arrays=dict(genes=np.array(state.genes,dtype=str),basis=basis,weights=weights,
                mask=np.ones(n,dtype=bool) if mask is None else np.asarray(mask,dtype=bool),
                factor_ids=np.asarray(ids,dtype=int))
    if basis.shape[0]!=n or weights.shape[0]!=n or arrays['mask'].shape!=(n,):
        raise ValueError('Projection context row dimensions disagree')
    metadata=dict(schema='eaggl_gene_projection_context/v1',sha256=fingerprint(arrays),
                  background_prior=float(state.background_prior),phi=state.internal_last_projection_phi,
                  projection_tol=state.internal_last_projection_tol,
                  solver=state.internal_projection_method,solver_tol=state.internal_projection_tol,
                  max_cycles=state.internal_projection_max_cycles,
                  settings={key:getattr(options,key) for key in SETTINGS},
                  scope='Exact gene basis and phenotype projection settings; not discovery restart state')
    with open(path,'wb') as stream:
        np.savez_compressed(stream,metadata=np.array(json.dumps(metadata)),**arrays)
    state._record_params({'internal_projection_context_out':path,'internal_projection_context_hash':metadata['sha256']},overwrite=True)


def restore_context(state,path,options):
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        arrays={name:archive[name].copy() for name in ('genes','basis','weights','mask','factor_ids')}
    if metadata.get('schema')!='eaggl_gene_projection_context/v1' or metadata.get('sha256')!=fingerprint(arrays):
        raise ValueError('Projection context schema or integrity check failed')
    genes=arrays['genes'];basis=arrays['basis'];weights=arrays['weights'];mask=arrays['mask']
    if len(set(genes))!=len(genes) or basis.ndim!=2 or weights.ndim not in (1,2):
        raise ValueError('Invalid projection context dimensions or gene identities')
    if basis.shape[0]!=len(genes) or weights.shape[0]!=len(genes) or mask.shape!=(len(genes),):
        raise ValueError('Projection context dimensions disagree')
    if not np.isfinite(basis).all() or np.any(basis<0) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Projection context contains invalid loadings or weights')
    ids=tuple(int(x) for x in arrays['factor_ids'])
    if tuple(getattr(state,'supplied_factor_ids',()))!=ids:
        raise ValueError('Projection context factor IDs disagree with supplied factors')
    if state.exp_gene_factors is None or set(state.genes)!=set(genes):
        raise ValueError('Projection context gene universe disagrees with supplied factors')
    lookup={gene:i for i,gene in enumerate(genes)}
    ix=[lookup[g] for g in state.genes]
    exact=basis[ix]
    if exact.shape!=state.exp_gene_factors.shape or not np.allclose(exact,state.exp_gene_factors,atol=6e-5,rtol=0):
        raise ValueError('Projection context basis disagrees with supplied factors beyond export rounding')
    if not np.isclose(metadata['background_prior'],state.background_prior,atol=1e-14,rtol=0):
        raise ValueError('Projection context background prior disagrees with this run')
    for key,value in metadata['settings'].items():
        if getattr(options,key)!=value:
            raise ValueError('Projection context setting disagrees: '+key)
    if metadata['solver']!=state.internal_projection_method or metadata['solver_tol']!=state.internal_projection_tol or metadata['max_cycles']!=state.internal_projection_max_cycles:
        raise ValueError('Projection context solver settings disagree with this run')
    if not np.isfinite(metadata['phi']) or metadata['phi']<0 or metadata['projection_tol']<=0:
        raise ValueError('Projection context contains invalid regularization/tolerance')
    # Restore exact values only after the explicitly supplied rounded table is verified.
    state.exp_gene_factors=exact.copy()
    state.gene_prob_factor_vector=None if weights.ndim==2 and weights.shape[1]==0 else weights[ix].copy()
    state.gene_factor_gene_mask=mask[ix].astype(bool)
    state.gene_in_discovery_mask=state.gene_factor_gene_mask.copy()
    state.internal_loaded_projection_phi=float(metadata['phi'])
    state.internal_loaded_projection_tol=float(metadata['projection_tol'])
    state._record_params({'internal_projection_context_in':path,'internal_projection_context_hash':metadata['sha256'],
                          'internal_projection_exact_basis_restored':True},overwrite=True)
