"""Sequential, bounded target/correctness/mixing/outer-feedback research audit."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'src')]
import numpy as np
from scipy import sparse
from scipy.special import logit
from conditional_compatibility import audit_slab_precision
from reliability.quadratic import QuadraticTarget,enumerate_posterior
from reliability.observed_logistic import ObservedLogisticTarget
from reliability.fixed_evidence import FixedEvidenceTarget
from reliability.checks import (gaussian_scan_stationary,production_draws,summarize_draws,
    collapsed_draws,support_transition,transition_metrics,finite_outer_reference,
    finite_outer_metrics,mean_feedback_metrics,production_huge_conditionals)


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def write(out,name,payload):
    with gzip.open(out/(name+'.json.gz'),'wt') as f:json.dump(clean(payload),f,indent=2,allow_nan=False)


def audit(capture_root):
    from pegs_shared.regression import compute_logistic_beta_tildes
    rng=np.random.default_rng(185)
    X=rng.choice([0.,0.,1.],size=(60,2));X[:15,1]=X[:15,0]
    y=(rng.random(60)<1/(1+np.exp(-(-1+1.2*X[:,0]-.4*X[:,1])))).astype(float)
    def regress(design):
        return compute_logistic_beta_tildes(sparse.csc_matrix(design),y,append_pseudo=False,rel_tol=1e-10)
    b,se,*_=regress(X);b2,se2,*_=regress(X*np.array([2.,.3]))
    V=np.corrcoef(X.T);tau=np.ones(2);shrink=tau/(tau+se**2)
    toy=audit_slab_precision(V,shrink,shrink*se**2)
    out=dict(logistic_before_covariate_correction=dict(beta_tilde=b,se=se,V=V,compatibility=toy,
        column_units_beta_error=float(np.max(np.abs(b-b2))),column_units_se_error=float(np.max(np.abs(se-se2)))))
    out['capture_checks']={}
    for trait in ('NAFLD','T2D'):
        source=capture_root/(trait+'_conditional')/'conditional.pkl.gz'
        meta=json.loads((source.parent/'capture.json').read_text());digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=meta['conditional_sha256']:raise ValueError('Capture checksum mismatch')
        with gzip.open(source,'rb') as f:k=pickle.load(f)
        if k['use_X'] or k['assume_independent'] or k['gauss_seidel']:raise ValueError('Unsupported capture')
        V=k['V'].toarray() if k['sparse_V'] else k['V'];rows=[]
        for c in range(k['se2s_m'].shape[0]):
            var=k['norm_scale_m'][c]**2;s=k['hdmp_hdmpn_m'][c]
            ids=np.flatnonzero((var>0)&(s>0)&np.isfinite(var)&np.isfinite(s))
            ids=ids[np.linspace(0,len(ids)-1,min(128,len(ids)),dtype=int)]
            matrix=V[c] if k['multiple_V'] else V
            if k['sparse_V']:matrix=matrix.T
            matrix=matrix[np.ix_(ids,ids)]
            row=audit_slab_precision(matrix,s[ids],var[ids]);row['chain']=c
            row['se_min']=float(np.sqrt(k['se2s_m'][c,ids]).min());row['se_max']=float(np.sqrt(k['se2s_m'][c,ids]).max())
            row['V_symmetry_error']=float(np.max(np.abs(matrix-matrix.T)))
            rows.append(row)
        out['capture_checks'][trait]=dict(sha256=digest,capture=meta,checks=rows)
        del k,V
    args=(np.array([.3,.4]),np.array([.2,.8]),np.array([[1.,.8],[.8,1.]]),np.ones(2))
    forward=gaussian_scan_stationary(*args,[0,1]);reverse=gaussian_scan_stationary(*args,[1,0])
    out['incompatible_but_stationary']=dict(forward=forward,reverse=reverse,
        covariance_order_difference=float(np.max(np.abs(forward['covariance']-reverse['covariance']))))
    return out


def reference():
    X=np.array([[-1.,-.3],[-.7,-.9],[0.,.2],[.2,.7],[.8,.4],[1.,1.]])
    y=np.array([0,1,0,0,1,1]);offset=np.full(len(y),-.7)
    observed=ObservedLogisticTarget(X,y,offset,np.full(2,.2),np.ones(2)*.8)
    a=observed.quadrature(48);b=observed.quadrature(96)
    evidence=FixedEvidenceTarget(X,np.array([-.2,1.,-.5,.3,1.2,.8]),offset,np.full(2,.2),np.ones(2)*.8)
    c=evidence.quadrature(48);d=evidence.quadrature(96)
    return dict(observed_logistic=b,observed_refinement_error=float(np.max(np.abs(a['weights']-b['weights']))),
                fixed_evidence_logistic=d,fixed_evidence_refinement_error=float(np.max(np.abs(c['weights']-d['weights']))),
                scope='Independent support enumeration and Gaussian quadrature; continuous two-variable references only.')


def mixing(a,out):
    cases=dict(easy=dict(V=np.array([[1.,.2],[.2,1.]]),se=np.full(2,.5),b=np.array([.4,.6]),pi=np.full(2,.2),tau=np.ones(2)),
               competing=dict(V=np.array([[1.,.999],[.999,1.]]),se=np.full(2,.2),b=np.full(2,1.5),pi=np.full(2,.01),tau=np.ones(2)))
    result={}
    for name,k in cases.items():
        target=QuadraticTarget(k['V']/k['se'][0]**2,k['b']/k['se']**2,k['pi'],k['tau']);truth=enumerate_posterior(target)
        results={};modes=[('sequential',False)]
        if name=='easy':modes += [('production',False),('sequential',True)]
        for batching,sparsify in modes:
            start=time.perf_counter()
            draws,masks=production_draws(k['b'],k['se'],k['V'],k['pi'],k['tau'],sweeps=a.sweeps,burn=a.burn,chains=8,
                                        batching=batching,sparsify=sparsify)
            label=batching+('_sparsity_projection' if sparsify else '')
            results[label]=dict(**summarize_draws(draws,truth),seconds=time.perf_counter()-start,batches=masks)
            np.savez_compressed(out/(name+'_'+label+'.npz'),draws=draws)
        if name=='competing':
            start=time.perf_counter();draws,metrics=collapsed_draws(target,sweeps=a.sweeps,burn=a.burn,chains=8)
            results['collapsed_with_swap']=dict(**summarize_draws(draws,truth),seconds=time.perf_counter()-start,transition=metrics)
            results['collapsed_without_swap_transition']=transition_metrics(support_transition(truth['weights']),truth['weights'])
            np.savez_compressed(out/'competing_collapsed_with_swap.npz',draws=draws)
        result[name]=dict(inputs=k,truth=truth,results=results)
    return result


def outer():
    design=np.array([[1.,.2],[.3,1.]])
    logbf=np.array([2.,3.]);offset=np.full(2,logit(.05))
    r=finite_outer_reference(design,offset,logbf,levels=(-2.,0.,2.))
    dgb,checks=production_huge_conditionals(r)
    # Fixed BF is a valid control; refreshed HuGE is tested against compatibility,
    # not called inaccurate merely because it differs from the fixed-BF model.
    refreshed=finite_outer_metrics(r,dgb)
    refreshed.pop('joint_tv_from_fixed_evidence');refreshed.pop('max_D_probability_error')
    return dict(fixed_evidence_exact_Gibbs=finite_outer_metrics(r,r['D_given_beta']),
                fixed_evidence_mean_feedback=mean_feedback_metrics(r,logbf),
                production_HuGE_contract=checks,refreshed_HuGE_compatibility=refreshed,
                scope='Exact finite discrete-slab control, with actual HuGE and D-probability helpers. No full-trait posterior validation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['audit','reference','mixing','outer','all'],default='all')
    p.add_argument('--capture-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sweeps',type=int,default=10000);p.add_argument('--burn',type=int,default=2000)
    a=p.parse_args()
    if not 0<=a.burn<a.sweeps:raise ValueError('Require 0 <= burn < sweeps')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Require clean source')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    a.out.mkdir(parents=True,exist_ok=False)
    meta=dict(commit=commit,arguments=vars(a),started=time.time(),production_changes=False,
              gates=dict(reference_probability_error=1e-7,support_tv=.02,beta_rhat=1.05,support_rhat=1.05),
              purpose='Correctness and mixing audit, not runtime optimization or production promotion')
    meta['arguments']={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    write(a.out,'receipt',meta)
    for name,fn in [('audit',lambda:audit(a.capture_root)),('reference',reference),('mixing',lambda:mixing(a,a.out)),('outer',outer)]:
        if a.stage in (name,'all'):
            start=time.perf_counter();report=fn();report['seconds']=time.perf_counter()-start
            write(a.out,name,report);print(name+' complete',flush=True)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Source changed')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=commit:raise ValueError('Commit changed')
    meta['completed']=time.time();meta['status']='COMPLETE_RESEARCH_AUDIT_NOT_PRODUCTION_VALIDATION'
    write(a.out,'receipt',meta)

if __name__=='__main__':main()
