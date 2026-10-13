"""Run exactly one serial T2D legacy/cumulative pair and summarize it."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/T2D_full_budget_20261012'


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=False)
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'RUNNING', 'pid': os.getpid(), 'started_unix': time.time(),
                 'scope': 'Two serial T2D fits only; 500 iteration budget; no convergence claim',
                 'completed': []}
        drivers = [ROOT/'scripts/screen_inference.py', ROOT/'scripts/compare_inference_screen.py']
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in drivers}
        state['driver_sha256'] = hashes
        for p in drivers:
            (OUT/p.name).write_bytes(p.read_bytes())
        (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(('PIGEAN_', 'DIG_OPEN_DATA_CACHE')):
                del env[key]
        env.update(PYTHONHASHSEED='0', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                   MPLBACKEND='Agg', MPLCONFIGDIR=str(Path.home()/'codex-workspace/.cache/matplotlib'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1')
        write(OUT/'status.json', state)
        try:
            for label, repo, engine in (
                ('legacy', 'experiment-foundation', 'legacy'),
                ('optimized', 'parallel-prefilter', 'parallel-prefilter'),
            ):
                if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
                    raise RuntimeError('Validation driver changed; refusing unmatched pair')
                command = [sys.executable, str(drivers[0]), '--repo', str(ROOT/'worktrees'/repo),
                           '--fixture', str(ROOT/'results/feature_foundation_20261011/fixtures/T2D/fixture.pkl.gz'),
                           '--out', str(OUT/label), '--iterations', '500', '--seconds', '10800',
                           '--seed', '0', '--engine', engine, '--prepare-engine', '--write-outputs',
                           '--fixed-budget-validation', '--trace-stride', '100', '--monitor']
                write(OUT/(label+'.command.json'), {'argv': command, 'environment': {
                    key: env[key] for key in ('PYTHONHASHSEED', 'PYTHONNOUSERSITE', 'MPLBACKEND',
                        'MPLCONFIGDIR', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS')}})
                state['active_run'] = label
                with (OUT/(label+'.console.log')).open('xb') as console:
                    proc = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT, env=env)
                    state['child_pid'] = proc.pid
                    write(OUT/'status.json', state)
                    try:
                        rc = proc.wait(timeout=10920)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        raise RuntimeError(label+' exceeded hard timeout')
                if rc:
                    raise RuntimeError(label+' failed with exit code '+str(rc))
                receipt = json.loads((OUT/label/'screen.json').read_text())
                if receipt['status'] != 'COMPLETE_FIXED_BUDGET':
                    raise RuntimeError(label+' did not complete')
                state['completed'].append(label)
                write(OUT/'status.json', state)
            with (OUT/'comparison.console.log').open('xb') as console:
                subprocess.run([sys.executable, str(drivers[1]), '--baseline', str(OUT/'legacy'),
                    '--candidate', str(OUT/'optimized'), '--out', str(OUT/'comparison.json')],
                    env=env, stdout=console, stderr=subprocess.STDOUT, check=True)
            report = json.loads((OUT/'comparison.json').read_text())
            a, b = [json.loads((OUT/label/'screen.json').read_text()) for label in ('legacy', 'optimized')]
            lines = ['# T2D matched 500-budget validation', '',
                'One seed; ten outer/four inner chains. Completion is not convergence. '
                'Times start from a saved pre-Gibbs state and exclude upstream input construction.', '',
                '| Measure | Legacy | Optimized | Change |', '|---|---:|---:|---:|']
            for title, key in [('Gibbs inference (s)', 'inference_seconds'),
                               ('Cold compiler preparation (s)', 'engine_preparation_seconds'),
                               ('Fixture-to-output elapsed (s)', 'wall_seconds'),
                               ('Inference CPU (s)', 'inference_cpu_seconds'),
                               ('Inner fits (s)', 'inner_fit_seconds')]:
                x, y = a[key], b[key]
                lines.append(f'| {title} | {x:.3f} | {y:.3f} | {x/y:.3f}x |')
            x, y = a['peak_rss_bytes']/2**20, b['peak_rss_bytes']/2**20
            lines.append(f'| Peak process RSS (MiB) | {x:.1f} | {y:.1f} | {(y/x-1)*100:+.1f}% |')
            lines += ['', 'Numerical agreement, sparse trajectory checkpoints and final diagnostics '
                      'are recorded in [comparison.json](comparison.json). '
                      'No automatic promotion, default change, push or additional trait run follows.', '',
                      '## Final output agreement', '', '```json',
                      json.dumps(report['exports'], indent=2), '```', '',
                      '## Latest post-burn diagnostics', '', '```json',
                      json.dumps(report['latest_post_burn_diagnostics'], indent=2), '```', '']
            (OUT/'summary.md').write_text('\n'.join(lines))
            state.update(status='COMPLETE', active_run=None, child_pid=None,
                         comparison=str(OUT/'comparison.json'), summary=str(OUT/'summary.md'),
                         meets_speed_and_agreement_gates=report['meets_speed_and_agreement_gates'])
        except BaseException as error:
            state.update(status='FAILED', error=repr(error))
            raise
        finally:
            state['updated_unix'] = time.time()
            write(OUT/'status.json', state)


if __name__ == '__main__':
    main()
