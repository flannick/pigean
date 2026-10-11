"""Capture/replay trusted local pre-Gibbs state; never load untrusted pickle files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import time
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['capture','replay'])
    parser.add_argument('--fixture',required=True)
    parser.add_argument('--out')
    parser.add_argument('--seed',type=int)
    args,command=parser.parse_known_args()
    fixture=Path(args.fixture)
    if args.mode=='capture':
        if fixture.exists():raise ValueError('Refusing to overwrite fixture')
        fixture.parent.mkdir(parents=True,exist_ok=True)
        from pigean.state import PigeanState
        def capture(state,**kwargs):
            with gzip.open(fixture,'wb') as handle:
                pickle.dump({'state':state,'kwargs':kwargs,'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},handle,protocol=5)
            fixture.with_suffix(fixture.suffix+'.json').write_text(json.dumps({'sha256':sha(fixture),'format':1,'trust':'local trusted fixture only','kwargs':kwargs},indent=2)+'\n')
            raise SystemExit(0)
        PigeanState.run_gibbs=capture
        import pigean
        if command and command[0]=='--':command=command[1:]
        rc=pigean.main(command)
        if not fixture.exists():raise ValueError('No Gibbs boundary reached')
        return rc
    if command:raise ValueError('Replay does not accept unrecorded model overrides')
    manifest=json.loads(fixture.with_suffix(fixture.suffix+'.json').read_text())
    if manifest['format']!=1 or sha(fixture)!=manifest['sha256']:raise ValueError('Fixture identity mismatch')
    with gzip.open(fixture,'rb') as handle: data=pickle.load(handle)
    np.random.set_state(data['numpy_rng']);random.setstate(data['python_rng'])
    if args.seed is not None:np.random.seed(args.seed);random.seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();data['state'].run_gibbs(**data['kwargs']);elapsed=time.perf_counter()-started
    data['state'].write_gene_statistics(str(out/'gene_stats.out.gz'),output_detail='full')
    data['state'].write_gene_set_statistics(str(out/'gene_set_stats.out.gz'),output_detail='full')
    (out/'inference_metrics.json').write_text(json.dumps({'inference_seconds':elapsed,'fixture_sha256':manifest['sha256'],'seed':args.seed})+'\n')

if __name__=='__main__':sys.exit(main())
