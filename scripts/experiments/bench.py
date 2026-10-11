"""Commit-bound experimental runs. Invoke with the project's Python interpreter."""
from __future__ import annotations
import argparse
import contextlib
import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

VERSION = 1

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

def identity(repo):
    repo = Path(repo).resolve()
    if git(repo, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError(f'Uncommitted source: {repo}')
    return {'repo': str(repo), 'commit': git(repo, 'rev-parse', 'HEAD')}

def inputs(spec):
    result = {}
    for name, item in spec['inputs'].items():
        path = Path(item['path']).resolve()
        actual = digest(path)
        if actual != item['sha256']: raise ValueError(f'Input changed: {name}')
        result[name] = {'path': str(path), 'sha256': actual}
    return result

def inventory(repos):
    result = []
    for repo in repos:
        for line in git(repo, 'for-each-ref', '--format=%(refname:short)\t%(objectname)', 'refs/heads').splitlines():
            name, sha = line.split('\t')
            abandoned = any(x in name.lower() for x in ('debug', 'diagnostic/', 'diag/', 'blockwise', 'adaptive-gibbs', 'gibbs-adaptive', 'gibbs-fixed500-default-20260826'))
            result.append({'repo': str(Path(repo).resolve()), 'branch': name, 'commit': sha,
                           'status': 'ABANDONED' if abandoned else 'UNCLASSIFIED'})
    return result

def validate_receipt(path):
    p = Path(path); r = read(p)
    if r['status'] != 'COMPLETE': raise ValueError('Incomplete run')
    if identity(r['source']['repo']) != r['source']: raise ValueError('Source changed')
    if digest(r['spec_path']) != r['spec_sha256']: raise ValueError('Specification changed')
    if digest(p.parent/'runner.py') != r['runner_sha256']: raise ValueError('Archived runner changed')
    inputs(read(r['spec_path']))
    for name, sha in r['outputs'].items():
        if digest(p.parent / name) != sha: raise ValueError(f'Output changed: {name}')
    return r

def run(spec_path, repo, output, seed, lock):
    spec_path = Path(spec_path).resolve(); output = Path(output).resolve()
    spec = read(spec_path); source = identity(repo); input_ids = inputs(spec)
    if spec.get('schema_version') != VERSION: raise ValueError('Unsupported specification')
    bound = {'source': source, 'spec_path': str(spec_path), 'spec_sha256': digest(spec_path),
             'runner_sha256': digest(__file__), 'seed': seed, 'inputs': input_ids}
    receipt = output / 'receipt.json'
    Path(lock).parent.mkdir(parents=True, exist_ok=True)
    with open(lock, 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            old = validate_receipt(receipt)
            if any(old[k] != v for k, v in bound.items()): raise ValueError('Resume identity mismatch')
            print('REUSED', output); return
        if output.exists() and any(output.iterdir()): raise ValueError('Unfinished output directory; use a new run ID')
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for k in list(env):
            if k.startswith('PIGEAN_') or k.startswith('DIG_OPEN_DATA_CACHE'): del env[k]
        env.update(PYTHONPATH=str(Path(repo).resolve() / 'src'), PYTHONHASHSEED='0',
                   PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MPLBACKEND='Agg',
                   MPLCONFIGDIR=str(Path.home() / 'codex-workspace/.cache/matplotlib'))
        values = {'python': sys.executable, 'output': str(output), 'seed': str(seed),
                  'repo': source['repo']}
        cmd = [x.format(**values) for x in spec['command']]
        write(output / 'command.json', {'argv': cmd, 'source': source, 'environment': {
            k: env[k] for k in ('PYTHONPATH','PYTHONHASHSEED','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}})
        (output/'runner.py').write_bytes(Path(__file__).read_bytes())
        write(output / 'running.json', {**bound, 'pid': os.getpid(), 'status': 'RUNNING'})
        started = time.perf_counter()
        with open(output / 'console.txt', 'wb') as console:
            proc = subprocess.run(cmd, cwd=repo, env=env, stdout=console, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if platform.system() != 'Darwin': rss *= 1024
        if proc.returncode:
            write(receipt, {**bound, 'status': 'FAILED', 'returncode': proc.returncode, 'wall_seconds': elapsed})
            raise ValueError(f'Run failed ({proc.returncode}): {output}')
        outputs = {}
        for name in spec['outputs']:
            path = output / name
            if not path.is_file() or not path.stat().st_size: raise ValueError(f'Missing output: {name}')
            if name.endswith('.gz'):
                if path.read_bytes()[:2] != b'\x1f\x8b':
                    raw = path.read_bytes()
                    with gzip.open(path, 'wb') as f: f.write(raw)
                with gzip.open(path, 'rb') as f:
                    for _ in iter(lambda: f.read(1024*1024), b''): pass
            outputs[name] = digest(path)
        if identity(repo) != source: raise ValueError('Source changed during execution')
        inputs(spec)
        totals = []
        for name in spec['outputs']:
            if 'log' in name:
                with (gzip.open(output/name, 'rt') if name.endswith('.gz') else open(output/name)) as f:
                    for line in f:
                        m = re.search(r'iteration_total=([\d.]+)s', line)
                        if m: totals.append(float(m[1]))
        write(receipt, {**bound, 'status': 'COMPLETE', 'returncode': 0, 'wall_seconds': elapsed,
                       'peak_rss_bytes': rss, 'outer_iteration_seconds': sum(totals) if totals else None,
                       'outer_iteration_timer_count': len(totals), 'outputs': outputs,
                       'platform': platform.platform(), 'python': sys.version})
        (output/'running.json').unlink()
        print('COMPLETE', output, round(elapsed, 2))

def load_metric(paths, filename, key, value):
    import pandas as pd
    series = []
    for p in paths:
        validate_receipt(Path(p)/'receipt.json')
        frame = pd.read_csv(Path(p)/filename, sep='\t')
        if frame[key].isna().any() or frame[key].duplicated().any(): raise ValueError('Missing or duplicate identifiers')
        series.append(pd.to_numeric(frame.set_index(key)[value], errors='coerce'))
    if not series: raise ValueError('Empty run panel')
    if any(set(s.index) != set(series[0].index) for s in series): raise ValueError('Within-panel universe changed')
    # Require finite entries in every replicate; averaging must not hide missingness.
    return pd.concat(series, axis=1).mean(axis=1, skipna=False)

def metric(a, b):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    union = a.index.union(b.index); x = a.reindex(union).to_numpy(); y = b.reindex(union).to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    valid = len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0
    result = {'n_union': len(union), 'n_finite': len(x), 'n_missing': int((~finite).sum()),
              'same_universe': set(a.index)==set(b.index), 'exact': bool(np.array_equal(x,y) and len(x)==len(union)),
              'valid': bool(valid)}
    if valid:
        slope, intercept = np.polyfit(x,y,1)
        result.update(pearson=float(pearsonr(x,y).statistic), spearman=float(spearmanr(x,y).statistic),
                      slope_zero=float(x@y/(x@x)), slope=float(slope), intercept=float(intercept),
                      max_abs_error=float(np.max(abs(x-y))), mean_abs_error=float(np.mean(abs(x-y))))
    result['strict_pass'] = bool(valid and result['same_universe'] and not result['n_missing'] and
                                result['pearson']>=.99 and result['spearman']>=.99 and .95<=result['slope_zero']<=1.05)
    return result

def compare(base, candidate, track, repeatability=None):
    import numpy as np
    rows = []
    for file, key, value in [('gene_stats.out.gz','Gene','prior'),('gene_set_stats.out.gz','Gene_Set','beta_uncorrected')]:
        a = load_metric(base,file,key,value); b = load_metric(candidate,file,key,value)
        top = a.abs().nlargest(200).index.union(b.abs().nlargest(200).index)
        for label, aa, bb in [('all',a,b),('signal',a.reindex(top),b.reindex(top))]:
            rows.append({'metric': value, 'panel': label, **metric(aa,bb)})
    bt = [validate_receipt(Path(p)/'receipt.json')['wall_seconds'] for p in base]
    ct = [validate_receipt(Path(p)/'receipt.json')['wall_seconds'] for p in candidate]
    ratio = float(np.median(bt)/np.median(ct))
    if track == 'neutral': passed = all(r['exact'] for r in rows) and ratio >= 1/1.05
    elif track == 'speed': passed = all(r['strict_pass'] for r in rows) and ratio >= 10
    elif track in ('agreement','repeatability'): passed = all(r['strict_pass'] for r in rows)
    elif track == 'inference': passed = False  # Requires an additional predeclared scientific validation receipt.
    else: raise ValueError('Unknown track')
    status = 'PASS' if passed else 'FAIL'
    if track in ('speed','agreement') and (repeatability is None or read(repeatability)['status'] != 'PASS'):
        status = 'INCONCLUSIVE'
    if track == 'inference': status = 'INCONCLUSIVE'
    return {'schema_version': VERSION, 'track': track, 'status': status, 'metrics': rows,
            'median_speedup': ratio, 'baseline': [str(Path(p).resolve()) for p in base],
            'candidate': [str(Path(p).resolve()) for p in candidate],
            'receipt_hashes': {str(Path(p).resolve()/'receipt.json'):digest(Path(p)/'receipt.json') for p in base+candidate},
            'repeatability': str(Path(repeatability).resolve()) if repeatability else None}

def eligible(manifest, repo):
    m = read(manifest); src = identity(repo)
    if m['commit'] != src['commit']: raise ValueError('Publication commit mismatch')
    if m.get('status') != 'PASS': raise ValueError('Not qualified')
    if set(m['traits']) != {'T2D','NAFLD'}: raise ValueError('Both traits required')
    if not m.get('tests') or not m.get('comparisons'): raise ValueError('Tests and comparisons required')
    for item in m['tests']+m['comparisons']:
        if digest(item['path']) != item['sha256']: raise ValueError('Evidence changed')
        r = read(item['path'])
        if r['status'] != 'PASS': raise ValueError('Evidence did not pass')
        if item in m['tests'] and r['commit'] != src['commit']: raise ValueError('Tests bound to another commit')
        for p, sha in r.get('receipt_hashes',{}).items():
            if digest(p) != sha: raise ValueError('Receipt changed')
            validate_receipt(p)
        for p in r.get('candidate',[]):
            if read(Path(p)/'receipt.json')['source']['commit'] != src['commit']: raise ValueError('Candidate commit mismatch')
    return {'status':'PASS','commit':src['commit'],'note':'Push feature branch only; upstream integration must also be current.'}


def test_run(repo, output, pytest_args):
    source=identity(repo); output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ, PYTHONPATH=str(Path(repo).resolve()/'src'), PYTHONHASHSEED='0',
             OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1', MPLBACKEND='Agg',
             MPLCONFIGDIR=str(Path.home()/'codex-workspace/.cache/matplotlib'))
    cmd=[sys.executable,'-m','pytest',*pytest_args]
    with open(output/'pytest.log','wb') as f:
        proc=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
    if identity(repo)!=source:raise ValueError('Source changed during tests')
    write(output/'tests.json',{'status':'PASS' if proc.returncode==0 else 'FAIL',
          'commit':source['commit'],'argv':cmd,'returncode':proc.returncode,
          'log':str(output/'pytest.log'),'log_sha256':digest(output/'pytest.log')})
    return proc.returncode

def integrate(repo, path, branch, base, features):
    identity(repo)
    if Path(path).exists():raise ValueError('Integration path already exists')
    subprocess.run(['git','-C',str(repo),'worktree','add','-b',branch,str(Path(path).resolve()),base],check=True)
    for feature in features:
        subprocess.run(['git','-C',str(Path(path).resolve()),'merge','--no-edit',feature],check=True)
    return identity(path)

def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('inventory');a.add_argument('--repo',action='append',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('freeze');a.add_argument('--spec',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('run');a.add_argument('--spec',required=True);a.add_argument('--repo',required=True);a.add_argument('--out',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--lock',required=True)
    a=sub.add_parser('compare');a.add_argument('--baseline',nargs='+',required=True);a.add_argument('--candidate',nargs='+',required=True);a.add_argument('--track',choices=['neutral','speed','agreement','repeatability','inference'],required=True);a.add_argument('--repeatability');a.add_argument('--out',required=True)
    a=sub.add_parser('verify');a.add_argument('--receipt',required=True)
    a=sub.add_parser('publication');a.add_argument('--manifest',required=True);a.add_argument('--repo',required=True)
    a=sub.add_parser('test');a.add_argument('--repo',required=True);a.add_argument('--out',required=True);a.add_argument('pytest_args',nargs=argparse.REMAINDER)
    a=sub.add_parser('integrate');a.add_argument('--repo',required=True);a.add_argument('--out',required=True);a.add_argument('--branch',required=True);a.add_argument('--base',required=True);a.add_argument('--feature',action='append',required=True)
    args=p.parse_args()
    if args.op=='test':
        rest=args.pytest_args[1:] if args.pytest_args[:1]==['--'] else args.pytest_args
        sys.exit(test_run(args.repo,args.out,rest))
    elif args.op=='integrate':print(json.dumps(integrate(args.repo,args.out,args.branch,args.base,args.feature)))
    elif args.op=='inventory':write(args.out,inventory(args.repo))
    elif args.op=='freeze':
        spec=read(args.spec)
        for v in spec['inputs'].values():v['sha256']=digest(v['path'])
        if Path(args.out).exists():raise ValueError('Refusing to overwrite frozen specification')
        write(args.out,spec)
    elif args.op=='run':run(args.spec,args.repo,args.out,args.seed,args.lock)
    elif args.op=='compare':write(args.out,compare(args.baseline,args.candidate,args.track,args.repeatability))
    elif args.op=='verify':print(json.dumps(validate_receipt(args.receipt),indent=2))
    elif args.op=='publication':print(json.dumps(eligible(args.manifest,args.repo),indent=2))

if __name__=='__main__': main()
