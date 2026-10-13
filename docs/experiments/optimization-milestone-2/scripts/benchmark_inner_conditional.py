"""Paired fixed-input conditional benchmarks; no outer inference claims."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import resource
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True)
    p.add_argument('--capture',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--enable',action='append',default=[])
    p.add_argument('--baseline-enable',action='append',default=[])
    p.add_argument('--calls',type=int,default=30)
    args=p.parse_args()
    repo, capture, out=(Path(x).resolve() for x in (args.repo,args.capture,args.out))
    out.mkdir(parents=True,exist_ok=False)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    from pigean.state import _update_inner_beta_gene_set_batch as update
    from pigean.inner_cache import InnerCorrelationCache
    from pegs_shared.sparse_regression import marginal_backend
    with gzip.open(capture/'conditional.pkl.gz','rb') as f: original=pickle.load(f)
    meta=json.loads((capture/'capture.json').read_text())
    assert hashlib.sha256((capture/'conditional.pkl.gz').read_bytes()).hexdigest()==meta['conditional_sha256']
    def fresh(features):
        kw=copy.deepcopy(original)
        kw['correlation_cache']=InnerCorrelationCache(kw['V'],kw['multiple_V'],kw['sparse_V'])
        if 'inner-residual' in features:
            from pigean.inner_residual import InnerResidualCache
            kw['residual_cache']=InnerResidualCache(kw['V'])
        return kw
    draw_rng=np.random.default_rng(736)
    draws=[(draw_rng.random(original['rand_ps_t'].shape),
            draw_rng.normal(size=original['rand_norms_t'].shape)) for _ in range(max(6,args.calls))]
    def run(kw,features,n):
        with marginal_backend('parallel-prefilter',enable=features):
            for i in range(n):
                kw['rand_ps_t'],kw['rand_norms_t']=draws[i]
                update(**kw)
    a,b=fresh(args.baseline_enable),fresh(args.enable)
    np_rng=pickle.dumps(np.random.get_state());py_rng=random.getstate()
    start=time.perf_counter();run(a,args.baseline_enable,1);run(b,args.enable,1)
    preparation=time.perf_counter()-start
    agreement={}
    fields=('sampled_beta_t','posterior_nonzero_prob_t','posterior_beta_mean_t','residual_beta_tilde_t')
    for step in range(5):
        for field in fields:
            np.testing.assert_allclose(a[field],b[field],rtol=1e-10,atol=1e-12)
            agreement[field]=max(agreement.get(field,0.),float(np.max(np.abs(a[field]-b[field]))))
        run(a,args.baseline_enable,1);run(b,args.enable,1)
    assert pickle.dumps(np.random.get_state())==np_rng and random.getstate()==py_rng
    timing={'baseline':[],'candidate':[]}
    for repeat in range(5):
        for label in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
            features=args.baseline_enable if label=='baseline' else args.enable
            kw=fresh(features)
            run(kw,features,1)
            start=time.perf_counter();run(kw,features,args.calls)
            timing[label].append(time.perf_counter()-start)
    medians={k:float(np.median(v)) for k,v in timing.items()}
    report=dict(status='CONDITIONAL_ONLY',source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        dirty_source=bool(subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()),
        features=args.enable,baseline_features=args.baseline_enable,capture=meta,calls_per_measurement=args.calls,
        preparation_seconds=preparation,max_absolute_difference=agreement,
        seconds=timing,median_seconds=medians,speedup=medians['baseline']/medians['candidate'])
    if 'inner-fused' in args.enable:
        from pigean.inner_compiled import conditional_kernel
        report['compiled_signatures'] = [str(x) for x in conditional_kernel.signatures]
        assert conditional_kernel.signatures, 'Requested kernel never dispatched'
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
