"""Paired direct-prefilter timing on one checksum-bound production input."""
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
    for key in ('repo','capture','out'):p.add_argument('--'+key,required=True)
    args=p.parse_args();repo,capture,out=(Path(x).resolve() for x in (args.repo,args.capture,args.out))
    out.mkdir(parents=True,exist_ok=False)
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    from pegs_shared.covariance import marginal_inflation
    from pegs_shared.sparse_regression import marginal_backend
    payload=capture/'conditional.pkl.gz';digest=hashlib.sha256(payload.read_bytes()).hexdigest()
    assert digest==json.loads((capture/'capture.json').read_text())['conditional_sha256']
    with gzip.open(payload,'rb') as f:kw=pickle.load(f)
    kw['correlation']=kw.pop('r');kw['backend']='sparse'
    def run(label):
        with marginal_backend('parallel-prefilter',enable=[] if label=='reference' else ['compiled-prefilter'],
                              disable=['parallel-prefilter'] if label=='compiled_serial' else []):
            start=time.perf_counter();result=marginal_inflation(**kw)
            return time.perf_counter()-start,result
    cold,_=run('compiled_serial')
    timings={k:[] for k in ('reference','compiled_serial','compiled_parallel')};difference=0.
    for repeat in range(5):
        pair={}
        for label in (list(timings) if repeat%2==0 else list(timings)[::-1]):
            elapsed,result=run(label);timings[label].append(elapsed);pair[label]=result
        for label in list(timings)[1:]:
            np.testing.assert_allclose(pair[label],pair['reference'],rtol=1e-10,atol=1e-12)
            difference=max(difference,float(np.max(np.abs(pair[label]-pair['reference']))))
    medians={k:float(np.median(v)) for k,v in timings.items()}
    report=dict(status='PREFILTER_KERNEL_ONLY',seconds=timings,medians=medians,cold_compile_and_call_seconds=cold,
                speedup={k:medians['reference']/v for k,v in medians.items()},max_abs_difference=difference,
                capture_sha256=digest,source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
