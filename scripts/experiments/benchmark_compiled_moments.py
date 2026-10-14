"""Bounded exactness/scaling microbenchmark; includes no inference or convergence claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();sys.path.insert(0,str(Path(a.repo).resolve()/'src'))
    import numpy as np
    from pegs_shared.compiled_moments import grouped_moment_sums
    rng=np.random.default_rng(734);results=[]
    t=time.perf_counter()
    grouped_moment_sums(np.array([0]),np.ones(1),np.ones(1),np.ones(1),np.zeros(1),1)
    cold=time.perf_counter()-t
    for annotations in (512,2048,8192):
        width=10*annotations;size=64*width
        local=np.repeat(np.arange(width),64)
        values=rng.normal(size=size);counts=rng.integers(1,100,size=size).astype(float)
        prob=rng.random(size=size);var=prob*(1-prob)
        def old():return tuple(np.bincount(local,weights=counts*w,minlength=width)
                              for w in (prob,var,values*prob,values*var,values*values*var))
        def new():return grouped_moment_sums(local,values,counts,prob,var,width)
        reference=old();candidate=new()
        for x,y in zip(reference,candidate):np.testing.assert_array_equal(x,y)
        times={'old':[],'new':[]}
        for rep in range(5):
            for key in (('old','new') if rep%2==0 else ('new','old')):
                start=time.perf_counter();out=(old if key=='old' else new)();times[key].append(time.perf_counter()-start)
        med={k:float(np.median(v)) for k,v in times.items()}
        results.append(dict(annotations=annotations,chains=10,groups_per_annotation=64,
            seconds=times,median=med,speedup=med['old']/med['new'],exact=True,
            removed_simultaneous_weight_arrays_bytes=4*size*8,
            caveat='Synthetic grouped reductions only; whole-process memory and full inference measured separately.'))
    report=dict(first_call_seconds=cold, first_call_note='Process-first call; existing Numba disk cache may be used. Full sampler uses a fresh cache directory.', scaling=results, source_sha256={name:hashlib.sha256((Path(a.repo)/name).read_bytes()).hexdigest() for name in ('src/pegs_shared/compiled_moments.py','src/pegs_shared/grouped_regression.py')})
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
