"""Bounded trait batches, shared proxy masks, and incremental result storage."""
from __future__ import annotations
import csv
import json
import math
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy import sparse
from .data import Source,Weights,fetch_selected,resolve_sources,text_open,warn
from .ld import load_ld
from .weights import construct,apply_anchor,export_scores
from .statistics import signed_associations,unsigned_associations


def requested(o):
    return any(getattr(o,k,None) for k in ('snp_factor_weights_in','snp_factor_weights_out','snp_factor_weights_plink_out','snp_factor_weights_pgs_out','run_snp_factor_phewas','snp_factor_phewas_stats_out'))


def validate(o):
    if o.snp_factor_phewas_stats_out and not o.run_snp_factor_phewas:raise ValueError('--snp-factor-phewas-stats-out requires --run-snp-factor-phewas')
    sign=getattr(o,'snp_factor_sign_anchor',None); effect=getattr(o,'snp_factor_effect_weight_anchor',None)
    if sign and effect: raise ValueError('Sign-only and effect-weight anchors are mutually exclusive')
    if getattr(o,'snp_factor_weights_in',None) and any(getattr(o,k,None) for k in ('snp_factor_sign_anchor','snp_factor_effect_weight_anchor','snp_factor_gene_effectors_in','factor_gene_clusters_in','snp_factor_genome_build','snp_factor_training_source','factor_gene_set_clusters_in','X_in','gene_stats_in','gene_set_stats_in')):
        raise ValueError('Saved weights are authoritative; do not combine weights-in with construction/anchor options')
    if o.snp_factor_weights_in and o.snp_factor_gene_weight!='signal_gene_weight':
        raise ValueError('Saved weights cannot be combined with --snp-factor-gene-weight')
    if getattr(o,'snp_factor_ld_in',None) and getattr(o,'snp_factor_ld_bfile',None): raise ValueError('Choose LD input or BED reference, not both')
    for name in ('snp_factor_proxy_r2','snp_factor_min_coverage','snp_factor_min_info'):
        if not 0<=getattr(o,name)<=1:raise ValueError(f'{name} must be in [0,1]')
    if not 0<=o.snp_factor_min_maf<.5:raise ValueError('MAF must be in [0,.5)')
    if o.snp_factor_memory_mb<1 or o.snp_factor_trait_batch_size<1:raise ValueError('Memory and batch size must be positive')


def load_reference(o,weights):
    if o.snp_factor_ld_build and weights.metadata.get("build") and o.snp_factor_ld_build!=weights.metadata["build"]:
        raise ValueError("LD reference genome build disagrees with score weights")
    return load_ld(getattr(o,'snp_factor_ld_in',None),getattr(o,'snp_factor_ld_variants_in',None),getattr(o,'snp_factor_ld_bfile',None),
                   weights.variants,o.snp_factor_memory_mb,o.snp_factor_proxy_window,o.snp_factor_ld_ancestry,o.snp_factor_ld_matrix_kind,o.snp_factor_proxy_r2)


def prepare_target(source,weights,ld,candidates,coordinates,o,audit):
    build=weights.metadata.get('build')
    if build and source.build and build!=source.build:raise ValueError('Target genome build disagrees with weights')
    signed=weights.metadata.get('mode','unsigned')!='unsigned'
    chosen=[]; selected={}; records=[]; transfer=[]; covered=[]; qualities=[]
    with fetch_selected(source,coordinates,(v for group in candidates for v,r,r2 in group)) as store:
        if build and source.build and build!=source.build:raise ValueError('Target genome build disagrees with weights')
        for i,original in enumerate(weights.variants):
            result=None
            if signed and weights.coefficients.getrow(i).nnz==0:
                audit.writerow([source.trait,source.ancestry,original.id,'','','','anchor_missing']);continue
            for proxy,r,r2 in candidates[i]:
                if signed and r is None:continue
                f=ld.frequencies.get(proxy.id)
                if f is None and proxy==original and np.isfinite(weights.frequencies[i]):f=float(weights.frequencies[i])
                record=store.lookup(proxy,signed=signed,freq=f,maf=o.snp_factor_min_maf,info=o.snp_factor_min_info)
                if record is None:continue
                if signed and f is None and not record.get('freq_is_maf'):f=record['freq']
                if signed and (f is None or not 0<f<1):continue
                result=(proxy,r,r2,record,f);break
            if result is None:
                audit.writerow([source.trait,source.ancestry,original.id,'','','','unavailable']);continue
            proxy,r,r2,record,f=result
            if proxy not in selected:
                selected[proxy]=len(records);records.append((record,f))
            chosen.append((selected[proxy],i));transfer.append(r if signed else 1.)
            covered.append(i);qualities.append(1. if proxy==original else (r2 if r2 is not None else np.nan))
            audit.writerow([source.trait,source.ancestry,original.id,proxy.id,r,r2,'direct' if proxy==original else 'proxy'])
        qc=dict(store.qc,access=store.access)
    n=len(records);rows=[p[0] for p in chosen];cols=[p[1] for p in chosen]
    mapping=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,len(weights.variants)))
    A=(mapping@weights.membership).tocsr()
    U=(sparse.csr_matrix((transfer,(rows,cols)),shape=mapping.shape)@weights.coefficients).tocsr()
    total=np.asarray(weights.membership.sum(axis=0)).ravel()
    represented=np.asarray(weights.membership[covered].sum(axis=0)).ravel() if covered else np.zeros(len(total))
    coverage=np.divide(represented,total,out=np.zeros_like(total),where=total>0)
    variants=list(selected)
    z2=np.array([r['z2'] for r,f in records])
    b=np.array([r['beta']*np.sqrt(2*f*(1-f)) if signed else 0. for r,f in records])
    se=np.array([r['se']*np.sqrt(2*f*(1-f)) if signed else 0. for r,f in records])
    signature=(tuple((v.id) for v in variants),tuple(chosen),tuple(transfer),tuple(f for r,f in records) if signed else ())
    sample_sizes=[r['n'] for r,f in records if r['n'] is not None and r['n']>0]
    qc.update(n_min=min(sample_sizes) if sample_sizes else None,n_max=max(sample_sizes) if sample_sizes else None,
              covariance_sample_overlap='assumed_common_samples',
              direct=sum(1 for row,i in chosen if variants[row]==weights.variants[i]),proxies=sum(1 for row,i in chosen if variants[row]!=weights.variants[i]),
              missing=len(weights.variants)-len(covered),tagging_quality_mean=float(np.nanmean(qualities)) if np.any(np.isfinite(qualities)) else None,
              alleles_not_verified=sum(not r.get('allele_verified',True) for r,f in records),
              frequency_not_assessed=sum(r['freq'] is None for r,f in records),info_not_assessed=sum(r['info'] is None for r,f in records))
    return dict(source=source,A=A,U=U,variants=variants,z2=z2,b=b,se=se,coverage=coverage,signature=signature,qc=qc)


def joint_roster(weights,coverage,o):
    if getattr(o,'snp_factor_joint_factor_policy','all')=='covered':
        return [i for i,value in enumerate(coverage) if value>=o.snp_factor_min_coverage]
    return list(range(len(weights.factors)))


def associate_group(items,weights,ld,o):
    """Fit covered-factor joint models on exactly the original selected SNP rows."""
    first=items[0];scope=o.snp_factor_phewas_regression_scope
    factors=weights.factors; signed=weights.metadata.get('mode','unsigned')!='unsigned'
    blocks=lambda:ld.blocks(first['variants'],max(1,o.snp_factor_memory_mb/4))
    method='signed_score' if signed else 'unsigned_evidence'
    if not first['variants']:
        scopes=(['marginal','joint'] if scope=='both' else [scope])+([] if signed else ['quadratic'])
        return [dict(factor=f,trait_index=t,method='unsigned_quadratic' if s=='quadratic' else method,
                     scope=s,status='no_variants',p=None) for t in range(len(items)) for f in factors for s in scopes]
    def fit(indices,fit_scope,quadratic=True):
        names=[factors[i] for i in indices]
        if signed:
            return signed_associations(first['U'][:,indices],np.column_stack([x['b'] for x in items]),
                                       np.column_stack([x['se'] for x in items]),blocks,names,fit_scope)
        return unsigned_associations(first['A'][:,indices],np.column_stack([x['z2'] for x in items]),
                                     blocks,names,fit_scope,quadratic=quadratic)
    indices=joint_roster(weights,first['coverage'],o)
    if getattr(o,'snp_factor_joint_factor_policy','all')=='all' or scope=='marginal':
        return fit(list(range(len(factors))),scope)
    # Marginal fits and quadratic tests retain the full factor roster and SNP
    # universe; only the joint design columns change. No SNP row is removed.
    rows=fit(list(range(len(factors))),'marginal' if scope=='both' else 'none')
    if indices:rows.extend(fit(indices,'joint',quadratic=False))
    for t in range(len(items)):
        for i,factor in enumerate(factors):
            if i not in indices:
                rows.append(dict(factor=factor,trait_index=t,method=method,scope='joint',
                                 status='excluded_joint_coverage' if indices else 'no_eligible_joint_factors',p=None))
    return rows


class Results:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA cache_size=-2048')
        self.db.execute('CREATE TABLE results(id INTEGER PRIMARY KEY,p REAL,family TEXT,payload TEXT,q REAL,bonferroni REAL,log_p REAL,log_q REAL,log_bonferroni REAL)')
        self.fields=set()
    def add(self,row):
        self.fields.update(row)
        p=row.get('p');p=float(p) if p is not None and np.isfinite(p) else None
        family='|'.join(str(row.get(k,'')) for k in ('method','scope','calibration','dependence'))
        clean={k: (None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in row.items()}
        logp=row.get('log_p') if p is not None else None
        if logp is None or not np.isfinite(logp):
            logp=float(np.log(p)) if p is not None and p>0 else (-np.inf if p==0 else None)
        self.db.execute('INSERT INTO results(p,family,payload,log_p) VALUES (?,?,?,?)',(p,family,json.dumps(clean,allow_nan=False),logp))
    def write(self,path):
        self.db.commit()
        for family, in self.db.execute('SELECT DISTINCT family FROM results').fetchall():
            total=self.db.execute('SELECT COUNT(*) FROM results WHERE family=?',(family,)).fetchone()[0]
            valid=self.db.execute('SELECT COUNT(p) FROM results WHERE family=?',(family,)).fetchone()[0]
            previous=0.
            cursor=self.db.execute('SELECT id,log_p FROM results WHERE family=? AND p IS NOT NULL ORDER BY log_p DESC,id DESC',(family,))
            for offset,(id,logp) in enumerate(cursor):
                rank=valid-offset;previous=min(previous,logp+np.log(total/rank))
                logbonf=min(0.,logp+np.log(total))
                self.db.execute('UPDATE results SET q=?,bonferroni=?,log_q=?,log_bonferroni=? WHERE id=?',
                    (float(np.exp(previous)),float(np.exp(logbonf)),previous,logbonf,id))
        fields=['trait','ancestry','factor','method','scope','beta','se','statistic','p','q','bonferroni','status']
        fields+=sorted((self.fields|{'log_q','log_bonferroni','q_underflow','bonferroni_underflow'})-set(fields))
        with text_open(path,'wt') as fh:
            writer=csv.DictWriter(fh,fieldnames=fields,delimiter='\t');writer.writeheader()
            for payload,q,bonf,logq,logbonf in self.db.execute('SELECT payload,q,bonferroni,log_q,log_bonferroni FROM results ORDER BY id'):
                row=json.loads(payload);row.update(q=q,bonferroni=bonf,
                    log_q=logq,log_bonferroni=logbonf,
                    q_underflow=bool(q==0 and logq is not None and np.isfinite(logq)),
                    bonferroni_underflow=bool(bonf==0 and logbonf is not None and np.isfinite(logbonf)))
                writer.writerow({k: 'NA' if v is None else v for k,v in row.items()})
    def close(self):self.db.close()


def run(o,state=None):
    validate(o)
    weights=Weights.read(o.snp_factor_weights_in) if o.snp_factor_weights_in else construct(o,state)
    ld=load_reference(o,weights)
    anchor=o.snp_factor_sign_anchor or o.snp_factor_effect_weight_anchor
    if anchor:weights=apply_anchor(weights,anchor,'sign' if o.snp_factor_sign_anchor else 'effect',ld)
    if o.snp_factor_weights_out:weights.write(o.snp_factor_weights_out)
    export_scores(weights,o.snp_factor_weights_plink_out,o.snp_factor_weights_pgs_out)
    if not o.run_snp_factor_phewas:return weights
    sources=resolve_sources(o)
    if not sources:raise ValueError('PheWAS requires at least one target source selector')
    for source in sources:
        for field,key in (('chrom','chrom'),('pos','pos'),('locus','id'),('p','p'),('beta','beta'),('se','se'),('freq','freq'),('n','n')):
            value=getattr(o,'gwas_'+field+'_col',None)
            if value is not None:source.columns.setdefault(key,value)
    path=o.snp_factor_phewas_stats_out or 'snp_factor_phewas.tsv.gz'
    # Sidecars carry complete resolved denominator and target-specific substitutions.
    Path(str(path)+'.traits.json').write_text(json.dumps([asdict(s) for s in sources],indent=2)+'\n')
    signed=weights.metadata.get('mode','unsigned')!='unsigned'
    candidates=[ld.candidates(v,o.snp_factor_proxy_r2,o.snp_factor_proxy_window) for v in weights.variants]
    coordinates={v.coordinate for group in candidates for v,r,r2 in group}
    n=len(coordinates);k=len(weights.factors);budget=o.snp_factor_memory_mb*1024**2
    base=(weights.membership.data.nbytes+weights.coefficients.data.nbytes)*3+8*k*k*8
    per_trait=8*(n*8+k*k*4)+weights.membership.nnz*32
    if base+per_trait>budget:raise MemoryError('Minimum selected-variant/joint working set exceeds memory budget')
    batch=max(1,min(o.snp_factor_trait_batch_size,int((budget-base)/max(per_trait,1))))
    warn(f'SNP factor PheWAS: {len(sources)} traits, batch {batch}, {n} requested coordinates; LD={ld.provenance}')
    with tempfile.TemporaryDirectory(prefix='eaggl-results-') as temp, text_open(str(path)+'.proxies.tsv.gz','wt') as audit_fh:
        results=Results(str(Path(temp)/'results.sqlite'))
        audit=csv.writer(audit_fh,delimiter='\t');audit.writerow(['trait','ancestry','original','selected','r','r2','status'])
        try:
            for start in range(0,len(sources),batch):
                groups={}
                for source in sources[start:start+batch]:
                    try:
                        item=prepare_target(source,weights,ld,candidates,coordinates,o,audit)
                        groups.setdefault(item['signature'],[]).append(item)
                    except (OSError,ValueError,ImportError) as exc:
                        warn(f'{source.trait}: {exc}')
                        scopes=['marginal','joint'] if o.snp_factor_phewas_regression_scope=='both' else [o.snp_factor_phewas_regression_scope]
                        if not signed:scopes=scopes+['quadratic']
                        for factor in weights.factors:
                            for scope in scopes:
                                results.add(dict(trait=source.trait,source=source.source,ancestry=source.ancestry,factor=factor,
                                    method='signed_score' if signed else ('unsigned_quadratic' if scope=='quadratic' else 'unsigned_evidence'),
                                    scope=scope,status='source_unavailable',error=str(exc),p=None,
                                    calibration=ld.provenance+('_Mixed_European' if source.ancestry.lower()=='mixed' and ld.ancestry.lower() in ('european','eur','eu') else ''),
                                    dependence='discovery_dependent' if source.source in weights.metadata.get('training_sources',[]) or source.source==weights.metadata.get('anchor') else 'overlap_unknown',
                                    joint_factors=json.dumps(weights.factors)))
                for items in groups.values():
                    rows=associate_group(items,weights,ld,o)
                    for row in rows:
                        item=items[row.pop('trait_index')];source=item['source'];j=weights.factors.index(row['factor'])
                        coverage=float(item['coverage'][j])
                        roster=joint_roster(weights,item['coverage'],o)
                        complete_required=getattr(o,'snp_factor_joint_factor_policy','all')=='all'
                        if row.get('status') not in ('excluded_joint_coverage','no_eligible_joint_factors') and (coverage<o.snp_factor_min_coverage or (row['scope']=='joint' and complete_required and np.any(item['coverage']<o.snp_factor_min_coverage))):
                            row.update(status='insufficient_coverage',p=None,beta=None,se=None,statistic=None)
                        dependence='discovery_dependent' if source.source in weights.metadata.get('training_sources',[]) or source.source==weights.metadata.get('anchor') else 'overlap_unknown'
                        calibration=ld.provenance+('_Mixed_European' if source.ancestry.lower()=='mixed' and ld.ancestry.lower() in ('european','eur','eu') else '')
                        row.update(trait=source.trait,source=source.source,ancestry=source.ancestry,coverage=coverage,calibration=calibration,dependence=dependence,
                                   units=(source.units or 'target_units_unknown')+'_per_reference_score_SD' if signed else ('dimensionless_chi_square' if row['method']=='unsigned_quadratic' else 'excess_chi_square_per_membership'),
                                   build_status='verified' if source.build and source.build==weights.metadata.get('build') else 'not_verified',
                                   joint_factors=json.dumps([weights.factors[i] for i in roster]),joint_factor_count=len(roster),
                                   joint_full_roster=len(roster)==len(weights.factors),original_factor_count=len(weights.factors),
                                   joint_factor_policy=getattr(o,'snp_factor_joint_factor_policy','all'),minimum_coverage=o.snp_factor_min_coverage,
                                   n_selected_variants=len(item['variants']),intercept_included=row['method']=='unsigned_evidence',
                                   analysis_label=('Quadratic evidence test' if row['scope']=='quadratic' else ('Marginal' if row['scope']=='marginal' else 'Joint')+' '+('signed-score regression' if signed else 'evidence regression')),
                                   qc=json.dumps(item['qc'],sort_keys=True),target_n=source.n)
                        if signed and source.units.lower() in ('log_odds','logor','log(or)') and row.get('beta') is not None:
                            row['odds_ratio']=float(np.exp(np.clip(row['beta'],-700,700)))
                        results.add(row)
            results.write(path)
        finally:results.close()
    Path(str(path)+'.traits.json').write_text(json.dumps([asdict(s) for s in sources],indent=2)+'\n')
    Path(str(path)+'.run.json').write_text(json.dumps(dict(weights=weights.metadata,ld=ld.provenance,ld_ancestry=ld.ancestry,
        minimum_coverage=o.snp_factor_min_coverage,joint_factor_policy=getattr(o,'snp_factor_joint_factor_policy','all'),
        ld_build=o.snp_factor_ld_build or 'not_verified',trait_batch_size=batch,memory_mb=o.snp_factor_memory_mb,factors=weights.factors,regression_scope=o.snp_factor_phewas_regression_scope,
        note='Summary-data approximations; unsigned evidence regression is descriptive and power dependent.'),indent=2)+'\n')
    return weights


def run_cli(options,state=None):
    """Translate anticipated input/resource failures into normal CLI diagnostics."""
    from pegs_cli_errors import CliUsageError
    try:
        return run(options,state)
    except (ValueError,OSError,ImportError,MemoryError) as exc:
        raise CliUsageError(str(exc)) from exc
