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
        started_at=time.time(); started = time.perf_counter()
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
        totals = []; diagnostics = None
        for name in spec['outputs']:
            if 'log' in name:
                with (gzip.open(output/name, 'rt') if name.endswith('.gz') else open(output/name)) as f:
                    for line in f:
                        m = re.search(r'iteration_total=([\d.]+)s', line)
                        if m: totals.append(float(m[1]))
                        if 'beta_Rhat_q' in line: diagnostics = line.strip()
        inference_seconds = None
        if spec['level'] == 'fixed_state_inference':
            inference_seconds = read(output/'inference_metrics.json')['inference_seconds']
        write(receipt, {**bound, 'status': 'COMPLETE', 'returncode': 0, 'wall_seconds': elapsed,
                       'started_at':started_at,'finished_at':time.time(),
                       'peak_rss_bytes': rss, 'outer_iteration_seconds': sum(totals) if totals else None,
                       'outer_iteration_timer_count': len(totals), 'outputs': outputs,
                       'inference_seconds': inference_seconds, 'last_diagnostics': diagnostics,
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
    exact = set(a.index)==set(b.index) and bool(np.array_equal(x,y,equal_nan=True))
    x, y = x[finite], y[finite]
    valid = len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0
    result = {'n_union': len(union), 'n_finite': len(x), 'n_missing': int((~finite).sum()),
              'same_universe': set(a.index)==set(b.index), 'exact': exact,
              'valid': bool(valid)}
    if valid:
        slope, intercept = np.polyfit(x,y,1)
        result.update(pearson=float(pearsonr(x,y).statistic), spearman=float(spearmanr(x,y).statistic),
                      slope_zero=float(x@y/(x@x)), slope=float(slope), intercept=float(intercept),
                      max_abs_error=float(np.max(abs(x-y))), mean_abs_error=float(np.mean(abs(x-y))))
    result['strict_pass'] = bool(valid and result['same_universe'] and not result['n_missing'] and
                                result['pearson']>=.99 and result['spearman']>=.99 and .95<=result['slope_zero']<=1.05)
    return result

def compare(base, candidate, track, repeatability=None, timing_base=None, timing_candidate=None):
    import numpy as np
    records = [validate_receipt(Path(p)/'receipt.json') for p in base+candidate]
    specs = [read(r['spec_path']) for r in records]
    if len({s['trait'] for s in specs}) != 1 or len({s['level'] for s in specs}) != 1:
        raise ValueError('Trait or benchmark level mismatch')
    if any(r['inputs'] != records[0]['inputs'] for r in records):
        raise ValueError('Cannot compare different input fixtures')
    if len({(r['platform'],r['python']) for r in records}) != 1:
        raise ValueError('Benchmark environment mismatch')
    rows = []; seed_rows = []; schemas_equal = True; full_outputs_equal = True
    for file, key, value in [('gene_stats.out.gz','Gene','prior'),('gene_set_stats.out.gz','Gene_Set','beta_uncorrected')]:
        a = load_metric(base,file,key,value); b = load_metric(candidate,file,key,value)
        top = a.abs().nlargest(200).index.union(b.abs().nlargest(200).index)
        for label, aa, bb in [('all',a,b),('signal',a.reindex(top),b.reindex(top))]:
            rows.append({'metric': value, 'panel': label, **metric(aa,bb)})
        if [r['seed'] for r in records[:len(base)]] == [r['seed'] for r in records[len(base):]]:
            import pandas as pd
            for bp,cp,r in zip(base,candidate,records):
                aa=load_metric([bp],file,key,value);bb=load_metric([cp],file,key,value)
                seed_rows.append({'metric':value,'seed':r['seed'],**metric(aa,bb)})
                af=pd.read_csv(Path(bp)/file,sep='\t').set_index(key).sort_index()
                bf=pd.read_csv(Path(cp)/file,sep='\t').set_index(key).sort_index()
                schemas_equal &= list(af.columns)==list(bf.columns)
                full_outputs_equal &= af.equals(bf)
    bt = [validate_receipt(Path(p)/'receipt.json')['wall_seconds'] for p in base]
    ct = [validate_receipt(Path(p)/'receipt.json')['wall_seconds'] for p in candidate]
    ratio = float(np.median(bt)/np.median(ct))
    if track == 'neutral': passed = all(r['exact'] for r in rows+seed_rows) and full_outputs_equal
    elif track == 'speed': passed = all(r['strict_pass'] for r in rows)
    elif track in ('agreement','repeatability'): passed = all(r['strict_pass'] for r in rows)
    elif track == 'inference': passed = False  # Requires an additional predeclared scientific validation receipt.
    else: raise ValueError('Unknown track')
    status = 'PASS' if passed else 'FAIL'
    reasons=[]
    baseline_seeds=[r['seed'] for r in records[:len(base)]]
    candidate_seeds=[r['seed'] for r in records[len(base):]]
    expected_candidate=list(range(5,10)) if track=='repeatability' else list(range(5))
    if sorted(baseline_seeds)!=list(range(5)) or sorted(candidate_seeds)!=expected_candidate:
        reasons.append('Required five-seed panels are incomplete')
    if len({r['source']['commit'] for r in records[:len(base)]})!=1 or len({r['source']['commit'] for r in records[len(base):]})!=1:
        raise ValueError('Mixed source commits within a panel')
    if any(r['source']['commit']!=specs[0]['baseline_commit'] for r in records[:len(base)]):
        raise ValueError('Baseline is not the protected baseline commit')
    if track=='repeatability' and any(r['source']['commit']!=specs[0]['baseline_commit'] for r in records):
        raise ValueError('Repeatability requires two baseline panels')
    repeatability_hash=digest(repeatability) if repeatability else None
    if track in ('speed','agreement'):
        if repeatability is None:
            reasons.append('Baseline repeatability evidence is required')
        else:
            repeat=validate_comparison(repeatability)
            repeatability_hash=digest(repeatability)
            if repeat['track']!='repeatability' or repeat['trait']!=specs[0]['trait'] or repeat['level']!=specs[0]['level'] or repeat['baseline'] != [str(Path(p).resolve()) for p in base]:
                raise ValueError('Repeatability evidence does not match this baseline')
            if repeat['status']!='PASS':reasons.append('Baseline repeatability did not pass')
    timing_ratio=None; timing_paths=[]
    if track in ('neutral','speed'):
        if not timing_base or not timing_candidate or len(timing_base)!=3 or len(timing_candidate)!=3:
            reasons.append('Three separate alternating timing pairs are required')
        else:
            timing_paths=list(timing_base)+list(timing_candidate)
            if len(set(map(str,timing_paths)))!=6 or set(map(str,timing_paths)) & set(map(str,base+candidate)):
                raise ValueError('Timing confirmations must be separate runs')
            times=[validate_receipt(Path(p)/'receipt.json') for p in timing_paths]
            for i,t in enumerate(times):
                ref=records[0 if i<3 else len(base)]
                if t['source']!=ref['source'] or t['spec_sha256']!=ref['spec_sha256'] or t['seed']!=0:
                    raise ValueError('Timing confirmation identity mismatch')
                if (t['platform'],t['python'])!=(ref['platform'],ref['python']):
                    raise ValueError('Timing confirmation environment mismatch')
            ordered=[times[i] for i in (0,3,1,4,2,5)]
            if any(a['finished_at']>b['started_at'] for a,b in zip(ordered,ordered[1:])):
                raise ValueError('Timing confirmations must run in alternating serial order')
            clock='inference_seconds' if specs[0]['level']=='fixed_state_inference' else 'wall_seconds'
            timing_ratio=float(np.median([t[clock] for t in times[:3]])/np.median([t[clock] for t in times[3:]]))
            if timing_ratio < (10 if track=='speed' else 1/1.05):status='FAIL'
    if reasons:status='INCONCLUSIVE'
    if track == 'inference': status = 'INCONCLUSIVE'
    return {'schema_version': VERSION, 'track': track, 'status': status, 'metrics': rows,
            'median_speedup': ratio, 'confirmed_speedup':timing_ratio, 'reasons':reasons,
            'seed_metrics':seed_rows,'schemas_equal':schemas_equal,'full_outputs_equal':full_outputs_equal,
            'trait': specs[0]['trait'], 'level': specs[0]['level'],
            'baseline_seeds':[r['seed'] for r in records[:len(base)]],
            'candidate_seeds':[r['seed'] for r in records[len(base):]], 'baseline': [str(Path(p).resolve()) for p in base],
            'candidate': [str(Path(p).resolve()) for p in candidate],
            'receipt_hashes': {str(Path(p).resolve()/'receipt.json'):digest(Path(p)/'receipt.json') for p in base+candidate+timing_paths},
            'timing_baseline':list(map(str,timing_base or [])), 'timing_candidate':list(map(str,timing_candidate or [])),
            'repeatability': str(Path(repeatability).resolve()) if repeatability else None,
            'repeatability_sha256':repeatability_hash}

def validate_comparison(path):
    r=read(path)
    for p,sha in r['receipt_hashes'].items():
        if digest(p)!=sha:raise ValueError('Comparison receipt changed')
        validate_receipt(p)
    if r.get('repeatability'):
        if digest(r['repeatability'])!=r['repeatability_sha256']:raise ValueError('Repeatability evidence changed')
        validate_comparison(r['repeatability'])
    return r

def eligible(manifest, repo):
    m = read(manifest); src = identity(repo)
    if m['commit'] != src['commit']: raise ValueError('Publication commit mismatch')
    if m.get('status') != 'PASS': raise ValueError('Not qualified')
    if set(m['traits']) != {'T2D','NAFLD'}: raise ValueError('Both traits required')
    if m.get('change_kind') not in ('foundation','speed'):
        raise ValueError('Statistical features require a separately implemented preregistered acceptance gate')
    if not m.get('tests') or not m.get('comparisons'): raise ValueError('Tests and comparisons required')
    seen_traits=set()
    for item in m['tests']+m['comparisons']:
        if digest(item['path']) != item['sha256']: raise ValueError('Evidence changed')
        r = read(item['path'])
        if r['status'] != 'PASS': raise ValueError('Evidence did not pass')
        if item in m['comparisons']:
            validate_comparison(item['path'])
            seen_traits.add(r['trait'])
            if r['track'] not in ('neutral','speed'): raise ValueError('Non-promotion comparison track')
            if r['track']!=('neutral' if m['change_kind']=='foundation' else 'speed'):
                raise ValueError('Comparison track does not match the declared change kind')
            if r.get('confirmed_speedup') is None:raise ValueError('Timing confirmation missing')
            if sorted(r['baseline_seeds'])!=list(range(5)) or sorted(r['candidate_seeds'])!=list(range(5)):
                raise ValueError('Incomplete seed panels')
        if item in m['tests'] and digest(r['log'])!=r['log_sha256']:raise ValueError('Test log changed')
        if item in m['tests'] and r['commit'] != src['commit']: raise ValueError('Tests bound to another commit')
        for p, sha in r.get('receipt_hashes',{}).items():
            if digest(p) != sha: raise ValueError('Receipt changed')
            validate_receipt(p)
        for p in r.get('candidate',[]):
            if read(Path(p)/'receipt.json')['source']['commit'] != src['commit']: raise ValueError('Candidate commit mismatch')
    if seen_traits != {'T2D','NAFLD'}:raise ValueError('Missing trait evidence')
    integration=m.get('integration')
    if not integration:raise ValueError('Current upstream integration evidence required')
    if digest(integration['tests'])!=integration['tests_sha256']:raise ValueError('Integration evidence changed')
    integrated=identity(integration['repo']);tests=read(integration['tests'])
    if tests['status']!='PASS' or tests['commit']!=integrated['commit'] or digest(tests['log'])!=tests['log_sha256']:
        raise ValueError('Integration tests invalid')
    upstream=git(repo,'rev-parse','origin/main')
    if upstream!=integration['upstream_commit']:raise ValueError('Upstream advanced; repeat integration validation')
    for commit in (upstream,src['commit']):
        subprocess.run(['git','-C',integration['repo'],'merge-base','--is-ancestor',commit,integrated['commit']],check=True)
    return {'status':'PASS','commit':src['commit'],'upstream_commit':upstream}

def publish(manifest,repo):
    branch=git(repo,'branch','--show-current')
    if not branch.startswith('codex/'):raise ValueError('Publication requires a codex feature branch')
    proc=subprocess.run(['git','-C',str(repo),'fetch','--quiet','origin','main'],capture_output=True)
    if proc.returncode:raise ValueError('Upstream fetch failed; no publication attempted')
    result=eligible(manifest,repo)
    proc=subprocess.run(['git','-C',str(repo),'push','--quiet','--set-upstream','origin',f'HEAD:refs/heads/{branch}'],capture_output=True)
    if proc.returncode:raise ValueError('Feature branch push failed')
    return result


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
    registry=read(Path(__file__).parent.parent.parent/'docs/experiments/registry.json')
    abandoned={e['branch'] for e in registry['entries'] if e['status']=='ABANDONED'}
    if any(f in abandoned or re.search(r'debug|diagnostic/|diag/|blockwise',f) for f in features):
        raise ValueError('Abandoned experiment cannot be an integration dependency')
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
    a=sub.add_parser('compare');a.add_argument('--baseline',nargs='+',required=True);a.add_argument('--candidate',nargs='+',required=True);a.add_argument('--track',choices=['neutral','speed','agreement','repeatability','inference'],required=True);a.add_argument('--repeatability');a.add_argument('--timing-baseline',nargs='+');a.add_argument('--timing-candidate',nargs='+');a.add_argument('--out',required=True)
    a=sub.add_parser('verify');a.add_argument('--receipt',required=True)
    a=sub.add_parser('publication');a.add_argument('--manifest',required=True);a.add_argument('--repo',required=True)
    a=sub.add_parser('publish');a.add_argument('--manifest',required=True);a.add_argument('--repo',required=True)
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
    elif args.op=='compare':write(args.out,compare(args.baseline,args.candidate,args.track,args.repeatability,args.timing_baseline,args.timing_candidate))
    elif args.op=='verify':print(json.dumps(validate_receipt(args.receipt),indent=2))
    elif args.op=='publication':print(json.dumps(eligible(args.manifest,args.repo),indent=2))
    elif args.op=='publish':print(json.dumps(publish(args.manifest,args.repo),indent=2))

if __name__=='__main__': main()
