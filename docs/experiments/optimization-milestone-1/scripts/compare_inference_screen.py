"""Compare paired short replays; never issue convergence or promotion PASS."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'worktrees/experiment-foundation/scripts/experiments'))
from bench import metric


def table(path):
    with path.open('rb') as f:
        compressed = f.read(2) == b'\x1f\x8b'
    return pd.read_csv(path, sep='\t', compression='gzip' if compressed else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    base, candidate = Path(args.baseline), Path(args.candidate)
    a, b = [json.loads((path/'screen.json').read_text()) for path in (base, candidate)]
    if a['status'] not in ('COMPLETE_SCREEN', 'COMPLETE_FIXED_BUDGET') or a['status'] != b['status']:
        raise ValueError('Both screens must complete')
    for key in ('fixture_sha256', 'inference_kwargs', 'script_sha256', 'outer_updates_executed'):
        if a[key] != b[key]:
            raise ValueError('Mismatched '+key)
    for key in ('seed', 'iterations', 'profile', 'write_outputs'):
        if a['args'][key] != b['args'][key]:
            raise ValueError('Mismatched '+key)
    if a['args'].get('memory_stages', False) != b['args'].get('memory_stages', False):
        raise ValueError('Mismatched allocation tracing')
    if a['args'].get('prepare_engine', False) != b['args'].get('prepare_engine', False):
        raise ValueError('Mismatched engine preparation policy')
    for key, default in (('trace_stride', 1), ('monitor', False), ('fixed_budget_validation', False)):
        if a['args'].get(key, default) != b['args'].get(key, default):
            raise ValueError('Mismatched '+key)
    report = {'status': 'SCREEN_ONLY', 'baseline': str(base), 'candidate': str(candidate),
              'commits': [a['commit'], b['commit']],
              'outer_updates_executed': a['outer_updates_executed'],
              'inference_seconds': [a['inference_seconds'], b['inference_seconds']],
              'inference_speedup': a['inference_seconds']/b['inference_seconds'],
              'wall_seconds': [a['wall_seconds'], b['wall_seconds']],
              'exports': {}, 'iteration_arrays': {}}
    long_validation = a['status'] == 'COMPLETE_FIXED_BUDGET'
    if long_validation:
        report['status'] = 'FIXED_BUDGET_COMPARISON_NOT_CONVERGENCE_OR_PROMOTION'
    report['whole_replay_speedup'] = a['wall_seconds'] / b['wall_seconds']
    report['cold_inference_seconds'] = [x['inference_seconds'] + x.get('engine_preparation_seconds', 0) for x in (a, b)]
    report['cold_inference_speedup'] = report['cold_inference_seconds'][0] / report['cold_inference_seconds'][1]
    for key in ('peak_rss_bytes', 'peak_rss_through_inference_bytes', 'inference_cpu_seconds',
                'inner_fit_seconds', 'inner_fit_calls', 'latest_post_burn_diagnostics', 'final_summary_scalars'):
        report[key] = [x.get(key) for x in (a, b)]
    report['peak_rss_change_percent'] = 100 * (b['peak_rss_bytes'] / a['peak_rss_bytes'] - 1)
    for filename, key, value in [('gene_stats.out.gz', 'Gene', 'prior'),
                                  ('gene_set_stats.out.gz', 'Gene_Set', 'beta_uncorrected'),
                                  ('gene_set_stats.out.gz', 'Gene_Set', 'beta')]:
        x, y = [table(path/filename) for path in (base, candidate)]
        if any(frame[key].isna().any() or frame[key].duplicated().any() for frame in (x,y)):
            raise ValueError('Invalid identifiers')
        sx, sy = x.set_index(key)[value], y.set_index(key)[value]
        report['exports'][value] = metric(sx, sy)
        top = sx.abs().nlargest(200).index.union(sy.abs().nlargest(200).index)
        report['exports'][value]['top200_union'] = metric(sx.reindex(top), sy.reindex(top))
    # Locally generated, trusted diagnostic arrays; these include burn-in draws.
    arrays = []
    for path in (base, candidate):
        with gzip.open(path/'iteration_arrays.pkl.gz','rb') as f:
            arrays.append(pickle.load(f))
    for key in ('genes', 'gene_sets'):
        if list(arrays[0][key]) != list(arrays[1][key]):
            raise ValueError('Changed iteration-array identifiers')
    if arrays[0].get('update_indices') != arrays[1].get('update_indices'):
        raise ValueError('Changed trajectory checkpoint indices')
    report['trace_update_indices'] = arrays[0].get('update_indices')
    for key in ('prior','beta_uncorrected','beta'):
        x, y = arrays[0][key], arrays[1][key]
        if x.shape != y.shape:
            raise ValueError('Changed iteration-array dimensions')
        report['iteration_arrays'][key] = metric(pd.Series(x.ravel()), pd.Series(y.ravel()))
        report['iteration_arrays'][key]['allclose_1e_8'] = bool(np.allclose(x, y, rtol=1e-8, atol=1e-10, equal_nan=True))
    report['proceed_to_full_runs'] = bool(
        report['inference_speedup'] >= 10 and
        all(x['strict_pass'] and x['top200_union']['strict_pass'] for x in report['exports'].values()) and
        all(x['strict_pass'] and x['allclose_1e_8'] for x in report['iteration_arrays'].values()))
    if long_validation:
        report['meets_speed_and_agreement_gates'] = report.pop('proceed_to_full_runs')
        report['promotion_authorized'] = False
    Path(args.out).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
