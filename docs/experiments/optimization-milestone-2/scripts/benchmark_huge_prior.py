"""Complete HuGE distillation on real fixture signals with fixed synthetic priors."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','fixture','out'):p.add_argument('--'+key,required=True)
    args=p.parse_args();repo,fixture,out=(Path(x).resolve() for x in (args.repo,args.fixture,args.out))
    out.mkdir(parents=True,exist_ok=False)
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    from pegs_shared.sparse_regression import marginal_backend
    digest=hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert digest==json.loads(fixture.with_suffix(fixture.suffix+'.json').read_text())['sha256']
    with gzip.open(fixture,'rb') as f:state=pickle.load(f)['state']
    prior=np.random.default_rng(781).normal(size=(10,len(state.genes)))
    kw={k:getattr(state,k) for k in ('huge_signal_bfs','huge_signal_posteriors','huge_signal_sum_gene_cond_probabilities',
         'huge_signal_mean_gene_pos','huge_signal_max_closest_gene_prob','gene_covariates','gene_covariates_mask',
         'gene_covariates_mat_inv','gene_covariate_names','gene_covariate_intercept_index')}
    for k in ('cap_region_posterior','scale_region_posterior','phantom_region_posterior','allow_evidence_of_absence'):
        kw[k]=getattr(state,'huge_'+k)
    kw.update(gene_prob_genes=state.genes,rel_prior_log_bf=prior)
    def run(compiled):
        with marginal_backend('legacy',enable=['huge-prior-softmax'] if compiled else []):
            start=time.perf_counter();result=state._distill_huge_signal_bfs(**kw)
            return time.perf_counter()-start,result
    cold,_=run(True)
    times={'reference':[],'candidate':[]};difference=0.
    for repeat in range(5):
        pair={}
        for label in (list(times) if repeat%2==0 else list(times)[::-1]):
            elapsed,result=run(label=='candidate');times[label].append(elapsed);pair[label]=result
        for a,b in zip(pair['reference'][:2],pair['candidate'][:2]):
            np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11,equal_nan=True)
            difference=max(difference,float(np.max(np.abs(a-b))))
        assert pair['reference'][2:]==pair['candidate'][2:]
    med={k:float(np.median(v)) for k,v in times.items()}
    report=dict(status='PRODUCTION_SIGNALS_SYNTHETIC_PRIORS',seconds=times,medians=med,speedup=med['reference']/med['candidate'],
                max_abs_difference=difference,cold_compile_and_call_seconds=cold,fixture_sha256=digest,
                prior_sha256=hashlib.sha256(prior.tobytes()).hexdigest(),shape=list(kw['huge_signal_bfs'].shape),
                source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
