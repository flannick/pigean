"""Compare full conditional sweeps on a captured problem with matched draws."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True)
    p.add_argument('--capture',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--enable',action='append',default=[])
    p.add_argument('--sweeps',type=int,default=6)
    args=p.parse_args()
    repo,capture,out=(Path(x).resolve() for x in (args.repo,args.capture,args.out))
    out.mkdir(parents=True,exist_ok=False)
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[k]='1'
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    from pigean.state import PigeanState,_update_inner_beta_gene_set_batch as update
    from pigean.inner_cache import InnerCorrelationCache
    from pegs_shared.sparse_regression import marginal_backend
    with gzip.open(capture/'conditional.pkl.gz','rb') as f:source=pickle.load(f)
    meta=json.loads((capture/'capture.json').read_text())
    assert hashlib.sha256((capture/'conditional.pkl.gz').read_bytes()).hexdigest()==meta['conditional_sha256']
    state=PigeanState(background_prior=.05,batch_size=10)
    batches=state._compute_gene_set_batches(source['V'])
    rng=np.random.default_rng(98)
    shape=source['sampled_beta_t'].shape
    draws=[(rng.random(shape),rng.normal(size=shape)) for _ in range(args.sweeps)]
    base=['inner-matmul','inner-fused']
    def run(features):
        kw=copy.deepcopy(source)
        kw['correlation_cache']=InnerCorrelationCache(kw['V'])
        if 'inner-residual' in features:
            from pigean.inner_residual import InnerResidualCache
            kw['residual_cache']=InnerResidualCache(kw['V'])
        masks=[np.broadcast_to(mask,(shape[1],shape[2])).copy() for mask in batches]
        snapshots=[]
        with marginal_backend('parallel-prefilter',enable=features):
            start=time.perf_counter()
            for uniforms,normals in draws:
                kw['rand_ps_t'],kw['rand_norms_t']=uniforms,normals
                for mask in masks:
                    kw['compute_mask_m']=mask
                    update(**kw)
                snapshots.append(kw['posterior_beta_mean_t'].copy())
            elapsed=time.perf_counter()-start
        return elapsed,kw,snapshots
    # Warm all selected signatures, then reset the complete problem per pair.
    run(base);run(base+args.enable)
    timings={'baseline':[],'candidate':[]};difference=0.
    for repeat in range(3):
        pair={}
        for label in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
            t,kw,trace=run(base if label=='baseline' else base+args.enable)
            timings[label].append(t);pair[label]=(kw,trace)
        for a,b in zip(pair['baseline'][1],pair['candidate'][1]):
            np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-11)
            difference=max(difference,float(np.max(np.abs(a-b))))
    medians={k:float(np.median(v)) for k,v in timings.items()}
    report={'status':'CAPTURED_SWEEPS_ONLY','features':args.enable,'sweeps':args.sweeps,
            'batches':len(batches),'seconds':timings,'medians':medians,
            'speedup':medians['baseline']/medians['candidate'],'max_abs_difference':difference,
            'source_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [repo/'src/pigean/state.py',*sorted((repo/'src/pigean').glob('inner*.py'))]}}
    cache=pair['candidate'][0].get('residual_cache')
    if cache is not None:
        report['residual_cache']={k:getattr(cache,k) for k in ('calls','rebuilds','delta_updates')}
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
