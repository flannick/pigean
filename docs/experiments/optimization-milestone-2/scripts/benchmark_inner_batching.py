"""Exact partition check and paired timing on a captured dense correlation."""
import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--capture',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    repo,capture,out=(Path(x).resolve() for x in (args.repo,args.capture,args.out))
    out.mkdir(parents=True,exist_ok=False)
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[k]='1'
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    from pigean.state import PigeanState
    from pegs_shared.sparse_regression import marginal_backend
    with gzip.open(capture/'conditional.pkl.gz','rb') as f:V=pickle.load(f)['V']
    state=PigeanState(background_prior=.05,batch_size=10)
    def run(features):
        with marginal_backend('legacy',enable=features):
            start=time.perf_counter();result=state._compute_gene_set_batches(V)
            return time.perf_counter()-start,result
    cold,_=run(['inner-batching'])
    timings={'baseline':[],'candidate':[]}
    for repeat in range(5):
        pairs={}
        for label in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
            elapsed,batches=run([] if label=='baseline' else ['inner-batching'])
            timings[label].append(elapsed);pairs[label]=batches
        np.testing.assert_array_equal(pairs['baseline'],pairs['candidate'])
    medians={k:float(np.median(v)) for k,v in timings.items()}
    report={'status':'BATCH_CONSTRUCTION_ONLY','exact_partition':True,'batches':len(batches),
            'cold_compile_and_call_seconds':cold,'seconds':timings,'medians':medians,
            'speedup':medians['baseline']/medians['candidate']}
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
