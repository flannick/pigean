"""Bounded real-data outer-loop feasibility screen, not convergence validation."""
import argparse
import cProfile
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import resource
import signal
import subprocess
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'src')]
import numpy as np
from logistic_sampling.outer_adapter import outer_backend,conditional_prior
from logistic_sampling.diagnostics import clean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',required=True);p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['legacy','coordinate','collapsed'],default='legacy')
    p.add_argument('--swaps',type=int,default=0);p.add_argument('--panel',type=int,default=12)
    p.add_argument('--panel-table',help='Existing baseline gene-set export to rank the fixed panel; hash recorded')
    p.add_argument('--iterations',type=int,default=21);p.add_argument('--outer-chains',type=int,default=4)
    p.add_argument('--sweeps',type=int,default=120);p.add_argument('--burn',type=int,default=40)
    p.add_argument('--seed',type=int,default=117);p.add_argument('--seconds',type=int,default=240)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    if not 1<=args.panel<=64 or not 0<=args.burn<args.sweeps or args.iterations<2 or args.outer_chains<2:
        p.error('Invalid bounded screen settings')
    if args.method=='legacy' and args.swaps:p.error('Legacy control does not accept research swaps')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True)
    if dirty:raise ValueError('Commit source before running a trait screen')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    started=time.perf_counter();receipt=dict(args=vars(args),commit=commit,status='RUNNING',updates=0)
    fixture=Path(args.fixture);digest=hashlib.sha256(fixture.read_bytes()).hexdigest()
    meta=json.loads(fixture.with_suffix(fixture.suffix+'.json').read_text())
    if digest!=meta['sha256']:raise ValueError('Fixture checksum mismatch')
    receipt['fixture_sha256']=digest
    with gzip.open(fixture,'rb') as f:data=pickle.load(f)
    state=data['state']
    ids=np.argsort(-np.abs(state.betas),kind='stable')[:args.panel]
    if args.panel_table:
        import pandas as pd
        table_path=Path(args.panel_table)
        table=pd.read_csv(table_path,sep='\t',compression='infer')
        names=table.assign(rank_value=table['beta'].abs()).sort_values('rank_value',ascending=False,kind='stable')['Gene_Set']
        mapping={name:i for i,name in enumerate(state.gene_sets)}
        ids=np.array([mapping[name] for name in names if name in mapping][:args.panel])
        receipt['panel_table_sha256']=hashlib.sha256(table_path.read_bytes()).hexdigest()
    mask=np.zeros(len(state.gene_sets),bool);mask[ids]=True
    receipt['panel_selection']='largest absolute baseline beta; original order retained; table' if args.panel_table else 'largest absolute initial baseline beta; original order retained'
    receipt['gene_sets']=[g for j,g in enumerate(state.gene_sets) if mask[j]]
    state.subset_gene_sets(mask,keep_missing=False,skip_V=True)
    receipt['shape']=state.X_orig.shape
    kwargs=dict(data['kwargs'])
    kwargs.update(max_num_iter=args.iterations,total_num_iter=args.iterations,num_chains=args.outer_chains,
                  num_batches_parallel=args.outer_chains,initial_linear_filter=False,sparse_solution=False,
                  sparse_frac_gibbs=None,sparse_max_gibbs=None,max_num_iter_betas=args.sweeps,
                  min_num_iter_betas=10,max_num_burn_in=10,min_num_burn_in=10)
    receipt['kwargs']=kwargs
    receipt['heuristics']=dict(huge_refresh='unchanged',huge_covariates='unchanged',signal_probabilities='unchanged',
        fixed_panel=True,inner_projection=False,outer_mean_feedback=True,
        marginal_corrections_in_logistic_candidate=False,inner_candidate_budget='fixed; no convergence claim')
    import pigean.state as sm
    original=sm._compute_gibbs_iteration_betas_and_priors
    diag_original=sm._log_gibbs_post_burn_diagnostics
    arrays={k:[] for k in ('prior','beta','beta_uncorrected','log_bf')}
    def progress(*pos,**inputs):
        result=original(*pos,**inputs)
        arrays['prior'].append(inputs['epoch_priors']['priors_mean_m'].copy())
        arrays['beta'].append(result['full_betas_mean_m'].copy())
        arrays['beta_uncorrected'].append(inputs['iter_state']['uncorrected_betas_mean_m'].copy())
        arrays['log_bf'].append(result['log_bf_m'].copy())
        receipt['updates']+=1
        with gzip.open(out/'progress.json.gz','wt') as f:json.dump(clean(dict(updates=receipt['updates'],elapsed=time.perf_counter()-started)),f)
        return result
    def diagnostics(*pos,**inputs):
        receipt['last_diagnostics']=inputs
        return diag_original(*pos,**inputs)
    def capture(s,it,offsets):
        pi,tau=conditional_prior(s,it,0)
        X=(s.X_orig.toarray()-it['full_mean_shifts_m'][0])/it['full_scale_factors_m'][0]
        with gzip.open(out/'first_conditional.pkl.gz','wb') as f:
            pickle.dump(dict(X=X,y=it['p_sample_m'][0],offset=np.full(X.shape[0],np.asarray(offsets).ravel()[0]),
                inclusion=pi,variance=tau,gene_sets=s.gene_sets,genes=s.genes),f,protocol=5)
    def timeout(*_):raise TimeoutError('Bounded screen time limit')
    np.random.seed(args.seed);random.seed(args.seed)
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,args.seconds)
    profiler=cProfile.Profile() if args.profile else None
    sm._compute_gibbs_iteration_betas_and_priors=progress;sm._log_gibbs_post_burn_diagnostics=diagnostics
    t0=time.perf_counter()
    signal_before=hashlib.sha256(pickle.dumps(state.huge_signal_posteriors_for_regression,protocol=5)).hexdigest()
    try:
        from pegs_shared.sparse_regression import marginal_backend
        features=['inner-matmul','inner-fused','inner-batching','compiled-prefilter','inner-cache64','huge-prior-softmax','static-correlation','compiled-moments']
        if profiler:profiler.enable()
        with marginal_backend('parallel-prefilter',enable=features),outer_backend(method=args.method,swaps=args.swaps,
                sweeps=args.sweeps,burn=args.burn,seed=args.seed,records=receipt,capture=capture):
            state.run_gibbs(**kwargs)
        receipt['status']='COMPLETE_REDUCED_SCREEN'
        receipt['inference_seconds']=time.perf_counter()-t0
        receipt['signal_probabilities_unchanged']=signal_before==hashlib.sha256(pickle.dumps(state.huge_signal_posteriors_for_regression,protocol=5)).hexdigest()
        if not receipt['signal_probabilities_unchanged']:raise ValueError('Signal probabilities mutated')
        state.write_gene_statistics(str(out/'gene_stats.out.gz'),output_detail='full')
        state.write_gene_set_statistics(str(out/'gene_set_stats.out.gz'),output_detail='full')
        if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=commit or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
            raise ValueError('Source changed during screen')
    except TimeoutError as e:receipt.update(status='TIME_LIMIT',error=str(e))
    except BaseException as e:
        receipt.update(status='ERROR',error=repr(e));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        sm._compute_gibbs_iteration_betas_and_priors=original;sm._log_gibbs_post_burn_diagnostics=diag_original
        if profiler:
            profiler.disable();profiler.dump_stats(str(out/'profile.pstats'))
        receipt.setdefault('inference_seconds',time.perf_counter()-t0)
        receipt['wall_seconds']=time.perf_counter()-started
        receipt['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        with gzip.open(out/'screen.json.gz','wt') as f:json.dump(clean(receipt),f,indent=2,allow_nan=False)
        np.savez_compressed(out/'iteration_arrays.npz',**{k:np.array(v) for k,v in arrays.items()},genes=state.genes,gene_sets=state.gene_sets)
    print(json.dumps(clean({k:receipt[k] for k in ('status','updates','inference_seconds','peak_rss_bytes')})))
    if receipt['status']!='COMPLETE_REDUCED_SCREEN':raise SystemExit(1)


if __name__=='__main__':main()
