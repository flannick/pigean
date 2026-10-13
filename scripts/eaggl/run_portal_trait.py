"""Run the reviewed gene-by-gene configuration with final-factor LLM labels.

Inputs may be local or downloaded read-only from explicit S3 paths. Results are
local only; no upload, overwrite, or automatic rerun of failed labeling.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def run(command, log, env):
    with log.open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def validate_outputs(out):
    names = ('factors.out.gz', 'factor_metrics.out.gz', 'consensus_stats.out.gz',
             'gene_clusters.out.gz', 'gene_set_clusters.out.gz',
             'gene_clusters_full_via_gene_sets.out.gz')
    for name in names:
        with gzip.open(out/name, 'rt') as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        if not rows:
            raise RuntimeError('Empty required output: '+name)
    with gzip.open(out/'consensus_stats.out.gz', 'rt') as stream:
        records = list(csv.DictReader(stream, delimiter='\t'))
    runs = [r for r in records if r['record_type'] == 'run']
    if len(runs) != 5 or len({r['seed'] for r in runs}) != 5:
        raise RuntimeError('Expected five distinct consensus restart seeds')
    if any(r['update_method'] != 'projected_gradient' for r in runs):
        raise RuntimeError('Discovery did not use the new solver')
    if any(r['stopping_status'] == 'line_search_failed' for r in runs):
        raise RuntimeError('A discovery restart failed line search')
    numeric = json.loads((out/'numerical_diagnostics.json').read_text())
    if not numeric['projections'] or not all(r['converged'] for r in numeric['projections']):
        raise RuntimeError('A checked projection did not converge; inspect numerical_diagnostics.json')
    return dict(restarts=5, converged_restarts=sum(r['converged'] == 'True' for r in runs),
                iteration_cap_restarts=sum(r['hit_iteration_cap'] == 'True' for r in runs),
                consensus_fallback=any(r['fallback_to_reference'] == 'True' for r in runs),
                checked_projections=len(numeric['projections']))


def publish_labels(out):
    """Replace text labels only; numerical columns and factor IDs are preserved."""
    with (out/'labeling/factor_labels.tsv').open() as stream:
        labels = {r['factor']: r['label'] for r in csv.DictReader(stream, delimiter='\t')}
    for name in ('gene_clusters.out.gz', 'gene_set_clusters.out.gz',
                 'gene_clusters_full_via_gene_sets.out.gz'):
        source = out/name
        temporary = out/(name+'.labeling.tmp')
        try:
            with gzip.open(source, 'rt') as stream, gzip.open(temporary, 'wt') as target:
                reader = csv.DictReader(stream, delimiter='\t')
                if not reader.fieldnames or not {'cluster', 'label'}.issubset(reader.fieldnames):
                    raise RuntimeError('Missing cluster/label columns: '+name)
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames, delimiter='\t')
                writer.writeheader()
                for row in reader:
                    if row['cluster'] == 'NA' and all(float(row[key]) == 0 for key in labels):
                        row['label'] = 'NA'
                    elif row['cluster'] not in labels:
                        raise RuntimeError('Unknown factor ID in '+name+': '+row['cluster'])
                    else:
                        row['label'] = labels[row['cluster']]
                    writer.writerow(row)
            temporary.replace(source)
        finally:
            temporary.unlink(missing_ok=True)
    shutil.copyfile(out/'factors.out.gz', out/'factors.unlabeled.out.gz')
    shutil.copyfile(out/'labeling/factors.labeled.out.gz', out/'factors.out.gz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trait-group', required=True)
    parser.add_argument('--phenotype', required=True)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--gs-s3')
    parser.add_argument('--gss-s3')
    parser.add_argument('--mouse-x-s3', default='s3://dig-analysis-bin/pigean/gene_sets/mouse_2024.txt')
    parser.add_argument('--msigdb-x-s3', default='s3://dig-analysis-bin/pigean/gene_sets/msigdb_nohp.txt')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--phenotype-profiles', type=Path)
    parser.add_argument('--phi', type=float, default=.01)
    parser.add_argument('--max-num-factors', type=int)
    parser.add_argument('--mock-labels', action='store_true', help='Testing only; no biological labels')
    args = parser.parse_args()
    if not args.mock_labels and (not os.environ.get('EAGGL_LLM_MODEL') or not
            (os.environ.get('EAGGL_LLM_AUTH_KEY') or os.environ.get('OPENAI_API_KEY'))):
        parser.error('Live labels are required: set EAGGL_LLM_MODEL and the API key environment variable')
    if not (0 <= args.phi < float('inf')) or (args.max_num_factors is not None and args.max_num_factors < 1):
        parser.error('Phi must be finite/nonnegative and starting rank must be positive')
    if args.input_dir and (args.gs_s3 or args.gss_s3):
        parser.error('Choose local inputs or S3 inputs, not both')
    if not args.input_dir and not (args.gs_s3 and args.gss_s3):
        parser.error('Provide --input-dir or explicit --gs-s3 and --gss-s3 paths')
    if args.phenotype_profiles and not args.phenotype_profiles.is_file():
        parser.error('Phenotype profile file is missing')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONHASHSEED='0')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = os.environ.get('EAGGL_NUM_THREADS', '2')
    metadata = dict(status='started', trait_group=args.trait_group, phenotype=args.phenotype,
                    model='mouse_msigdb', labeling=True, mock_labels=args.mock_labels,
                    s3_uploads=False, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    metadata['launcher_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['label_helper_sha256'] = hashlib.sha256((ROOT/'scripts/eaggl/label_final_factors.py').read_bytes()).hexdigest()
    try:
        inputs = args.input_dir.resolve() if args.input_dir else out/'inputs'
        if not args.input_dir:
            inputs.mkdir()
            if not shutil.which('aws'):
                raise RuntimeError('AWS CLI is required for S3 input downloads')
            for uri, name in ((args.gs_s3, 'gs.out'), (args.gss_s3, 'gss.out'),
                              (args.mouse_x_s3, 'mouse_2024.txt'), (args.msigdb_x_s3, 'msigdb_nohp.txt')):
                if not uri.startswith('s3://'):
                    raise ValueError('Input URI must start with s3://')
                subprocess.run(['aws', 's3', 'cp', uri, str(inputs/name), '--only-show-errors'], check=True)
        expected = ('gs.out', 'gss.out', 'mouse_2024.txt', 'msigdb_nohp.txt')
        for name in expected:
            if not (inputs/name).is_file() or (inputs/name).stat().st_size == 0:
                raise RuntimeError('Missing or empty input: '+name)
        metadata['input_sha256'] = {name: hashlib.sha256((inputs/name).read_bytes()).hexdigest() for name in expected}
        metadata['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((ROOT/'src').rglob('*.py'))}
        command = [sys.executable, '-m', 'eaggl', 'factor',
                   '--X-in', str(inputs/'mouse_2024.txt'), '--X-in', str(inputs/'msigdb_nohp.txt'),
                   '--gene-stats-in', str(inputs/'gs.out'), '--gene-set-stats-in', str(inputs/'gss.out'),
                   '--discovery-model', 'gene_by_gene', '--deterministic', '--seed', '0',
                   '--phi', str(args.phi), '--symnmf-update-method', 'projected_gradient',
                   '--symnmf-stopping', 'adaptive', '--factor-max-iterations', '20000',
                   '--factor-rel-tol', '1e-6', '--factor-runs', '5', '--consensus-nmf',
                   '--consensus-aggregation', 'median', '--factor-output-scope', 'all',
                   '--factor-top-loading-type', 'raw',
                   '--cluster-row-min-max-loading', '0', '--fixed-basis-projection-method', 'verified_quadratic',
                   '--numerical-diagnostics-out', str(out/'numerical_diagnostics.json')]
        for flag, name in (('factors-out', 'factors.out.gz'), ('factor-metrics-out', 'factor_metrics.out.gz'),
            ('gene-clusters-out', 'gene_clusters.out.gz'), ('gene-set-clusters-out', 'gene_set_clusters.out.gz'),
            ('gene-clusters-full-via-gene-sets-out', 'gene_clusters_full_via_gene_sets.out.gz'),
            ('consensus-stats-out', 'consensus_stats.out.gz'), ('params-out', 'params.out'),
            ('warnings-file', 'warnings.txt')):
            command += ['--'+flag, str(out/name)]
        if args.max_num_factors:
            command += ['--max-num-factors', str(args.max_num_factors)]
        if args.phenotype_profiles:
            command += ['--gene-phewas-bfs-in', str(args.phenotype_profiles.resolve()),
                        '--trait-factor-links-out', str(out/'trait_factor_links.out.gz'),
                        '--projection-context-out', str(out/'projection_context.npz')]
        metadata['command'] = command
        run(command, out/'run.log', env)
        metadata['qc'] = validate_outputs(out)
        # Label final consensus raw genes/sets, without phenotype names in the prompt.
        labeling = [sys.executable, str(ROOT/'scripts/eaggl/label_final_factors.py'),
                    '--source', str(ROOT), '--gene-loadings', str(out/'gene_clusters.out.gz'),
                    '--gene-set-loadings', str(out/'gene_set_clusters.out.gz'),
                    '--factors-in', str(out/'factors.out.gz'), '--out', str(out/'labeling')]
        if args.mock_labels:
            labeling.append('--mock')
        run(labeling, out/'labeling.log', env)
        publish_labels(out)
        metadata['status'] = 'mock_test_complete' if args.mock_labels else 'complete'
        if not args.mock_labels:
            if (metadata['qc']['converged_restarts'] != 5 or metadata['qc']['consensus_fallback']):
                metadata['status'] = 'complete_needs_numerical_review'
    except BaseException:
        metadata['status'] = 'failed'
        raise
    finally:
        metadata['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (out/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if metadata['status'] == 'complete_needs_numerical_review':
        (out/'_QC_REVIEW_REQUIRED').touch()
        raise SystemExit(4)
    if metadata['status'] == 'complete':
        (out/'_SUCCESS').touch()


if __name__ == '__main__':
    main()
