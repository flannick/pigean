"""Capture one trusted real inner conditional near a selected outer update."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import time


class Captured(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--outer', type=int, default=25)
    args = p.parse_args()
    repo, fixture, out = (Path(x).resolve() for x in (args.repo, args.fixture, args.out))
    out.mkdir(parents=True, exist_ok=False)
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[k] = '1'
    os.environ['NUMBA_CACHE_DIR'] = str(out/'numba_cache')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(Path.home()/'codex-workspace/.cache/matplotlib')
    sys.path.insert(0, str(repo/'src'))
    import numpy as np
    import pigean.state as module
    from pegs_shared.sparse_regression import marginal_backend
    from pegs_shared.compiled_covariance import prepare_compiled_covariance
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert digest == json.loads(fixture.with_suffix(fixture.suffix+'.json').read_text())['sha256']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    with gzip.open(fixture,'rb') as f:
        data = pickle.load(f)
    started = time.perf_counter()
    def timeout(*_):
        raise TimeoutError('Capture exceeded 300 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 300)
    count = 0
    original_update = module._compute_gibbs_iteration_betas_and_priors
    original_batch = module._update_inner_beta_gene_set_batch
    selected = None
    def update(*positional, **kw):
        nonlocal count
        count += 1
        return original_update(*positional, **kw)
    def batch(**kw):
        nonlocal selected
        if count == args.outer and not kw['assume_independent'] and not kw['use_X']:
            selected = {k:v for k,v in kw.items() if k != 'correlation_cache'}
            with gzip.open(out/'conditional.pkl.gz','wb') as f:
                pickle.dump(selected,f,protocol=5)
            raise Captured()
        return original_batch(**kw)
    module._compute_gibbs_iteration_betas_and_priors = update
    module._update_inner_beta_gene_set_batch = batch
    np.random.seed(0); random.seed(0)
    prepare_compiled_covariance()
    try:
        with marginal_backend('parallel-prefilter'):
            data['state'].run_gibbs(**dict(data['kwargs'],max_num_iter=500,total_num_iter=500))
    except Captured:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        module._compute_gibbs_iteration_betas_and_priors = original_update
        module._update_inner_beta_gene_set_batch = original_batch
    assert selected is not None
    receipt = dict(outer=count,capture_seconds=time.perf_counter()-started,
                   source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
                   fixture_sha256=digest,conditional_sha256=hashlib.sha256((out/'conditional.pkl.gz').read_bytes()).hexdigest(),
                   shapes={k:list(v.shape) for k,v in selected.items() if hasattr(v,'shape')})
    (out/'capture.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
