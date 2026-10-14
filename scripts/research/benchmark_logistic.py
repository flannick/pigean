"""Sequential known-answer logistic benchmarks with complete costs and draws."""
import argparse
import gzip
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
import numpy as np
from logistic_sampling.model import LogisticTarget
from logistic_sampling.sampler import sample
from logistic_sampling.diagnostics import summarize,clean


def cases():
    X=np.array([[-1.,-.3],[-.7,-.9],[0.,.2],[.2,.7],[.8,.4],[1.,1.]])
    easy=LogisticTarget(X,np.array([0,1,0,0,1,1]),np.full(6,-.7),np.full(2,.2),np.full(2,.8))
    x=np.r_[np.ones(50),np.zeros(50)];X=np.column_stack((x,x))
    y=np.zeros(100);y[:30]=1;y[50]=1
    difficult=LogisticTarget(X,y,np.full(100,np.log(.02/.98)),np.full(2,.01),np.full(2,9.))
    # Null has exactly equal outcome rates in annotation members/nonmembers.
    y0=np.zeros(100);y0[0]=1;y0[50]=1
    null=LogisticTarget(X,y0,np.full(100,np.log(.02/.98)),np.full(2,.01),np.full(2,9.))
    return dict(easy=easy,competing=difficult,null=null)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--sweeps',type=int,default=6000)
    p.add_argument('--burn',type=int,default=1000);p.add_argument('--seeds',type=int,nargs='+',default=[117,118])
    p.add_argument('--case',choices=list(cases()),nargs='+',default=list(cases()))
    p.add_argument('--methods',choices=['coordinate','collapsed','collapsed_swap'],nargs='+',default=['coordinate','collapsed','collapsed_swap'])
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    # Tracked draft documentation may coexist; record exactly what is dirty.
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True)
    all_start=time.perf_counter();results=dict(args=vars(args),commit=commit,dirty=dirty,cases={})
    start=time.perf_counter()
    for collapse,swaps in ((False,0),(True,0),(True,1)):
        sample(cases()['easy'],sweeps=2,burn=1,collapse=collapse,swaps=swaps)
    results['compile_setup_seconds']=time.perf_counter()-start
    for name in args.case:
        target=cases()[name];start=time.perf_counter()
        a=target.quadrature(128);ref=target.quadrature(192)
        err=float(np.max(abs(a['weights']-ref['weights'])))
        # For sharply concentrated cases refine further, never silently pass.
        if err>1e-7:
            a=ref;ref=target.quadrature(256);err=float(np.max(abs(a['weights']-ref['weights'])))
        oracle='tensor_gauss_hermite'
        if name=='competing':
            a=target.duplicate_reference(1e-9);ref=target.duplicate_reference(1e-11)
            err=float(np.max(abs(a['weights']-ref['weights'])))
            oracle='adaptive_duplicate_sum'
        row=dict(reference=ref,oracle=oracle,refinement_error=err,reference_seconds=time.perf_counter()-start,runs=[])
        results['cases'][name]=row
        if err>1e-7:
            row['status']='REFERENCE_REFINEMENT_FAILED'
            continue
        initial=np.zeros((4,2))
        if name=='competing':initial[:2,0]=4.;initial[2:,1]=4.
        for seed in args.seeds:
            for collapse,swaps,label in ((False,0,'coordinate'),(True,0,'collapsed'),(True,1,'collapsed_swap')):
                if label not in args.methods:continue
                start=time.perf_counter()
                draws=sample(target,sweeps=args.sweeps,burn=args.burn,chains=4,seed=seed,
                             initial=initial,collapse=collapse,swaps=swaps)
                elapsed=time.perf_counter()-start
                metrics=summarize(draws,ref)
                run=dict(method=label,seed=seed,seconds=elapsed,metrics=metrics)
                row['runs'].append(run)
                np.savez_compressed(out/f'{name}_{label}_{seed}.npz',beta=draws)
                print(json.dumps(clean(dict(case=name,method=label,seed=seed,seconds=elapsed,
                    support_tv=metrics['support_tv'],rhat=metrics['inclusion_rhat']))),flush=True)
                with gzip.open(out/'results.json.gz','wt') as f:json.dump(clean(results),f,indent=2,allow_nan=False)
    results['total_seconds']=time.perf_counter()-all_start
    results['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    with gzip.open(out/'results.json.gz','wt') as f:json.dump(clean(results),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
