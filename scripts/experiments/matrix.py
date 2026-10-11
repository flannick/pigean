"""Run a serial, resumable two-trait matrix from frozen benchmark specifications."""
import argparse
from pathlib import Path
import subprocess
import sys

import bench


def freeze(path, value):
    if path.exists():
        if bench.read(path) != value:
            raise ValueError(f'Matrix changed; use a new output root: {path}')
    else:
        bench.write(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', action='append', required=True)
    parser.add_argument('--baseline-repo', required=True)
    parser.add_argument('--candidate-repo', required=True)
    parser.add_argument('--candidate-arg', action='append', default=[])
    parser.add_argument('--out', required=True)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--track', choices=['neutral', 'speed', 'agreement', 'inference'], default='neutral')
    parser.add_argument('--phase', choices=['baseline', 'candidate', 'timing', 'compare', 'all'], default='all')
    args = parser.parse_args()
    root = Path(args.out).resolve()
    baseline = bench.identity(args.baseline_repo)
    candidate = bench.identity(args.candidate_repo)
    specs = [bench.read(p) for p in args.spec]
    if {s['trait'] for s in specs} != {'T2D', 'NAFLD'} or len(specs) != 2:
        raise ValueError('Exactly one T2D and one NAFLD specification required')
    if any(s['baseline_commit'] != baseline['commit'] for s in specs):
        raise ValueError('Baseline must be the protected main commit')
    freeze(root/'matrix.json', {
        'baseline': baseline, 'candidate': candidate, 'track': args.track,
        'runner_sha256': bench.digest(bench.__file__),
        'specifications': {str(Path(p).resolve()): bench.digest(p) for p in args.spec},
        'candidate_args': args.candidate_arg,
    })

    def job(spec, repo, output, seed):
        # A new process per job gives each resource-usage receipt its own lifetime.
        subprocess.run([sys.executable, bench.__file__, 'run', '--spec', str(spec),
                        '--repo', repo, '--out', str(output), '--seed', str(seed),
                        '--lock', str(Path(args.lock).resolve())], check=True)

    for spec in specs:
        trait = root/spec['trait']
        bs = trait/'baseline_spec.json'
        cs = trait/'candidate_spec.json'
        freeze(bs, spec)
        modified = dict(spec, command=spec['command'] + args.candidate_arg)
        freeze(cs, modified)
        bp = [trait/'baseline'/f'seed{i}' for i in range(10)]
        cp = [trait/'candidate'/f'seed{i}' for i in range(5)]
        bt = [trait/'timing'/f'pair{i}'/'baseline' for i in range(3)]
        ct = [trait/'timing'/f'pair{i}'/'candidate' for i in range(3)]
        if args.phase in ('baseline', 'all'):
            for seed, output in enumerate(bp):
                job(bs, baseline['repo'], output, seed)
        if args.phase in ('candidate', 'all'):
            for seed, output in enumerate(cp):
                job(cs, candidate['repo'], output, seed)
        if args.phase in ('timing', 'all'):
            for base_out, candidate_out in zip(bt, ct):
                job(bs, baseline['repo'], base_out, 0)
                job(cs, candidate['repo'], candidate_out, 0)
        if args.phase in ('compare', 'all'):
            repeat = trait/'repeatability.json'
            bench.write(repeat, bench.compare(bp[:5], bp[5:], 'repeatability'))
            comparison = bench.compare(bp[:5], cp, args.track, repeat, bt, ct)
            bench.write(trait/'comparison.json', comparison)
            print(spec['trait'], comparison['status'], flush=True)


if __name__ == '__main__':
    main()
