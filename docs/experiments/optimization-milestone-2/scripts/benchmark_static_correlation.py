"""Real-design correlation cache: cold setup, changing subsets, exact agreement."""
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
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    from pigean.static_correlation import StaticCorrelationCache
    digest=hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert digest==json.loads(fixture.with_suffix(fixture.suffix+'.json').read_text())['sha256']
    with gzip.open(fixture,'rb') as f:state=pickle.load(f)['state']
    assert state.y_corr_cholesky is None
    rng=np.random.default_rng(316)
    sizes=[500,1503,2000];masks=[]
    for size in sizes:
        mask=np.zeros(state.X_orig.shape[1],dtype=bool)
        mask[rng.choice(len(mask),size=min(size,len(mask)),replace=False)]=True;masks.append(mask)
    cache=StaticCorrelationCache()
    start=time.perf_counter();cache.get(state,masks[0]);cold=time.perf_counter()-start
    times={'reference':[],'candidate':[]}
    for repeat in range(5):
        pair={}
        for label in (list(times) if repeat%2==0 else list(times)[::-1]):
            start=time.perf_counter()
            pair[label]=[cache.get(state,m) if label=='candidate' else state._calculate_V_internal(
                state.X_orig[:,m],None,state.mean_shifts[m],state.scale_factors[m]) for m in masks]
            times[label].append(time.perf_counter()-start)
        for a,b in zip(pair['reference'],pair['candidate']):np.testing.assert_array_equal(a,b)
    med={k:float(np.median(v)) for k,v in times.items()}
    report=dict(status='REAL_DESIGN_SYNTHETIC_SUBSETS',cold_seconds=cold,subset_sizes=sizes,seconds=times,medians=med,
                speedup=med['reference']/med['candidate'],exact_arrays=True,retained_bytes=cache.matrix.nbytes,
                max_bytes=cache.max_bytes,builds=cache.builds,hits=cache.hits,fixture_sha256=digest,
                source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
