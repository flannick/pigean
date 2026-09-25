"""Deterministic reviewer-facing sensitivity audit; no simulations or GWAS downloads."""
from pathlib import Path
import csv, gzip, hashlib, json, os, sys
import numpy as np
from scipy import sparse, special, stats
ROOT=Path(os.environ.get('PIGEAN_AUDIT_ROOT',str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0,str(ROOT/'worktrees/snp-factor-phewas/src'))
from pegs_shared.regression import compute_logistic_beta_tildes
BASE=ROOT/'results/validation_20260905/T2D/post_suggested_exclusions'
REF=Path('/private/tmp/pigean-validation-20260905/bundles/model_large-2026.02.22/data')
OUT=ROOT/'results/hdl_gene_dependence_robustness_20260907'


def grouped_fit(x,y):
    """Exact two-group fit retaining the production independent-pseudocount convention."""
    x=np.asarray(x,bool);y=np.asarray(y,float)
    n1=x.sum();n0=(~x).sum()
    if min(n1,n0)==0:return dict(status='missing_predictor_group')
    p1=(y[x].sum()+y.mean())/(n1+1);p0=y[~x].mean()
    if not (0<p1<1 and 0<p0<1):return dict(status='separation')
    beta=special.logit(p1)-special.logit(p0)
    se=np.sqrt(1/((n1+1)*p1*(1-p1))+1/(n0*p0*(1-p0)))
    return dict(status='ok',beta=float(beta),se=float(se),p=float(2*stats.norm.sf(abs(beta/se))),
                member_genes=int(n1),member_hits=int(y[x].sum()),nonmember_genes=int(n0),nonmember_hits=int(y[~x].sum()),p1=float(p1),p0=float(p0))


def covariance_stress(x,y,B,weights=(0,.1,.25,.5,.75)):
    """K is normalized annotation overlap, a declared stress model, not inferred HDL covariance."""
    fit=grouped_fit(x,y)
    if fit['status']!='ok':return []
    x=np.asarray(x,bool);v=np.where(x,fit['p1']*(1-fit['p1']),fit['p0']*(1-fit['p0']))
    vp=fit['p1']*(1-fit['p1']);mean=(v[x].sum()+vp)/(v.sum()+vp)
    c=np.sqrt(v)*(x-mean);cp=np.sqrt(vp)*(1-mean);den=c@c+cp*cp
    degree=np.asarray(B.multiply(B).sum(axis=1)).ravel();scale=np.divide(1,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
    L=B.multiply(scale[:,None]).tocsr()
    Kform=float(np.sum(np.asarray(L.T@c)**2)+np.sum(c[degree==0]**2))
    # With fitted group-specific Bernoulli means, correlation has a pairwise upper bound.
    bound=(min(fit['p1'],fit['p0'])-fit['p1']*fit['p0'])/np.sqrt(vp*fit['p0']*(1-fit['p0']))
    members=np.flatnonzero(x);other=L[~x].T;max_cross=0.
    for start in range(0,len(members),32):
        block=L[members[start:start+32]]@other
        if block.nnz:max_cross=max(max_cross,float(block.data.max()))
    rows=[]
    for weight in weights:
        ratio=np.sqrt(((1-weight)*(c@c)+weight*Kform+cp*cp)/den)
        se=fit['se']*ratio
        rows.append(dict(**fit,shrinkage_weight=weight,stress_se=float(se),se_ratio=float(ratio),
          stress_p=float(2*stats.norm.sf(abs(fit['beta']/se))),max_cross_group_correlation=weight*max_cross,
          bernoulli_cross_group_upper_bound=float(bound),pairwise_binary_bound_passes=bool(weight*max_cross<=bound+1e-12),
          interpretation='Working structural-overlap stress only; pairwise feasibility is necessary, not sufficient for a binary joint distribution.'))
    return rows


def read_annotations(genes):
    index={g:i for i,g in enumerate(genes)};rr=[];cc=[];names=[];sources=[];seen={};duplicates=0
    aliases={}
    for line in (REF/'portal_gencode.gene.map').open():
        a,b=line.rstrip().split('\t');aliases.setdefault(a,set()).add(b)
    paths=sorted(REF.glob('gene_set_list_*.txt'))
    for path in paths:
        for line in path.open():
            fields=line.rstrip().split('\t')
            if len(fields)==1:fields=line.rstrip().split(r'\t')
            members=set()
            for token in fields[1:]:
                gene=token.split(':')[0]
                for g in aliases.get(gene,{gene}):
                    if g in index:members.add(index[g])
            if not members:continue
            signature=tuple(sorted(members))
            if signature in seen:
                duplicates+=1;sources[seen[signature]].add(path.name);continue
            seen[signature]=len(names)
            rr.extend(signature);cc.extend([len(names)]*len(signature));names.append(fields[0]);sources.append({path.name})
    B=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(genes),len(names)))
    return B,names,sources,paths,duplicates,aliases


def genomic_groups(genes,aliases):
    locations={g:set() for g in genes}
    for filename,col in [('NCBI37.3.plink.gene.loc',5),('refGene_hg19_TSS.subset.loc',0)]:
        for line in (REF/filename).open():
            fields=line.split();chrom=fields[1].removeprefix('chr');chrom={'23':'X','24':'Y','26':'MT','M':'MT'}.get(chrom,chrom)
            for g in aliases.get(fields[col],{fields[col]}):
                if g in locations:locations[g].add((chrom,int(fields[2])//1000000))
    if any(not values for values in locations.values()):raise ValueError('Missing gene coordinates')
    groups={}
    for gene,values in locations.items():
        for chrom,window in values:groups.setdefault(f'{chrom}:{window}Mb',set()).add(gene)
    return {name:np.array([g in members for g in genes]) for name,members in groups.items()}


def write_table(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with gzip.open(OUT/(name+'.tsv.gz'),'wt') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    universe=BASE/'pigean/rerun_bundle_extract/gene_universe.tsv.gz';gmtpath=BASE/'eaggl/factors_as_gene_sets.gmt.gz'
    with gzip.open(universe,'rt') as f:genes=[r['Gene'] for r in csv.DictReader(f,delimiter='\t')]
    with gzip.open(gmtpath,'rt') as f:gmt={a[0]:set(a[2:]) for a in (line.rstrip().split('\t') for line in f)}
    gene_ref=REF/'all.gene_stats.large.gt1.out.gz'
    with gzip.open(gene_ref,'rt') as f:observed={r['Gene']:r for r in csv.DictReader(f,delimiter='\t') if r['Trait_Internal']=='HDL'}
    combined=np.array([float(observed[g]['Combined']) if g in observed else 0 for g in genes]);p=special.expit(combined+special.logit(.05))
    y=(p>np.sort(p)[::-1][int(p.sum())]).astype(float)
    np.testing.assert_array_equal(y,[g in observed for g in genes])
    B,names,sources,paths,duplicates,aliases=read_annotations(genes)
    genomic=genomic_groups(genes,aliases)
    rows=[];deletions=[];baseline=[];source_scans=[]
    for factor in ['Factor4','Factor7']:
        x=np.array([g in gmt[factor] for g in genes]);fit=grouped_fit(x,y);baseline.append(dict(factor=factor,**fit))
        production=compute_logistic_beta_tildes(sparse.csc_matrix(x[:,None],dtype=float),p.copy(),rel_tol=1e-10)
        np.testing.assert_allclose([fit['beta'],fit['se']],[production[0][0]/x.std(),production[1][0]/x.std()],rtol=1e-9)
        for row in covariance_stress(x,y,B):rows.append(dict(factor=factor,annotation_basis='all_four_reference_sources_deduplicated',**row))
        # Source omission changes covariance only, never response, weights, or fitted beta.
        for source in sorted(set().union(*sources)):
            keep=np.array([bool(s-{source}) for s in sources])
            for row in covariance_stress(x,y,B[:,keep],weights=(.1,)):
                source_scans.append(dict(factor=factor,omitted_source=source,**row))
        # Annotation removal is ranked without target evidence: overlap with fixed factor membership.
        overlap=np.asarray(B.T@x.astype(float)).ravel();size=np.asarray(B.sum(axis=0)).ravel()
        chosen=sorted(np.flatnonzero(overlap>0),key=lambda j:(-overlap[j],size[j],names[j]))[:10]
        masks=[('annotation',names[j],np.asarray(B[:,j].toarray()).ravel()>0) for j in chosen]
        masks += [('genomic_window',name,mask) for name,mask in genomic.items() if np.any(mask & x)]
        for kind,name,mask in masks:
            deleted=grouped_fit(x[~mask],y[~mask])
            row=dict(factor=factor,group_type=kind,group=name,deleted_genes=int(mask.sum()),deleted_members=int(np.sum(mask&x)),deleted_member_hits=int(np.sum(mask&x&(y>0))),remaining_member_fraction=float(np.sum(x&~mask)/x.sum()),**deleted)
            if deleted['status']=='ok':row['beta_change']=deleted['beta']-fit['beta']
            deletions.append(row)
    for name,data in [('baseline',baseline),('covariance_stress',rows),('annotation_source_stress',source_scans),('group_deletions',deletions)]:write_table(name,data)
    manifest=[]
    for path in [universe,gmtpath,gene_ref,*paths,REF/'portal_gencode.gene.map',REF/'NCBI37.3.plink.gene.loc',REF/'refGene_hg19_TSS.subset.loc']:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        manifest.append(dict(path=str(path),sha256=h.hexdigest()))
    summary=dict(genes=len(genes),hits=int(y.sum()),annotations=B.shape[1],annotation_nnz=B.nnz,duplicate_annotation_memberships_removed=duplicates,
      annotation_sources=[p.name for p in paths],annotation_gene_coverage=float(np.mean(np.asarray(B.sum(axis=1)).ravel()>0)),
      baseline=baseline,covariance_stress=rows,source_stress=source_scans,
      input_hashes=manifest,scope='Deterministic sensitivity audit; no simulations, no GWAS access, no gene-level arrays saved.',
      missing_prerequisite='The supplied reference bundle contains HDL gene scores but no HDL annotation-coefficient table. Structural overlap is a target-independent stress model, not a reconstruction of the proposed HDL-specific annotation covariance.',
      genomic_scope='1-Mb start-coordinate windows from supplied hg19 files; ambiguous mappings enter every applicable deletion. Overlapping windows are not jackknife replicates or validated LD blocks.',
      annotation_scope='Top ten unique membership sets ranked by factor-gene overlap then size/name, without HDL evidence. Deletion holds gene-hit labels fixed; it does not refit annotation effects or recompute reference inclusion.')
    summary['deletion_summary']=[]
    for factor in ['Factor4','Factor7']:
        for kind in ['annotation','genomic_window']:
            selected=[r for r in deletions if r['factor']==factor and r['group_type']==kind];valid=[r for r in selected if r['status']=='ok']
            summary['deletion_summary'].append(dict(factor=factor,group_type=kind,groups=len(selected),unavailable=len(selected)-len(valid),
               min_beta=min(r['beta'] for r in valid),max_beta=max(r['beta'] for r in valid),max_p=max(r['p'] for r in valid),
               weakest=min(valid,key=lambda r:r['beta']),largest_change=max(valid,key=lambda r:abs(r['beta_change']))))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['genes','hits','annotations','duplicate_annotation_memberships_removed','deletion_summary']},indent=2))

if __name__=='__main__':main()
