"""Native count and fixed-seed parity checks on the bundled compact GWAS."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_native(out, flags, observed=False):
    out.mkdir()
    data = ROOT / 'tests/data'
    args = ['gibbs', '--deterministic', '--seed', '12', '--hide-opts', '--hide-progress',
            '--X-in', str(data / 't2d_smoke/gene_set_list_mouse_t2d_toy.txt'),
            '--gene-map-in', str(data / 'model_small/portal_gencode.gene.map'),
            '--gene-loc-file', str(data / 'model_small/NCBI37.3.plink.gene.loc'),
            '--gene-loc-file-huge', str(data / 'model_small/NCBI37.3.plink.gene.loc'),
            '--gwas-in', str(data / 't2d_smoke/T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz'),
            '--gwas-chrom-col', 'CHROM', '--gwas-pos-col', 'POS', '--gwas-p-col', 'P',
            '--gwas-n-col', 'N', '--num-chains', '2', '--num-chains-betas', '2',
            '--gene-stats-out', str(out / 'genes.tsv.gz'),
            '--gene-set-stats-out', str(out / 'annotations.tsv.gz'),
            '--params-out', str(out / 'params.tsv.gz'),
            '--output-detail', 'full', '--gene-stats-output-scope', 'all', *flags]
    (out / 'command.json').write_text(json.dumps(args))
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONHASHSEED='0',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1')
    with gzip.open(out / 'console.log.gz', 'wb') as log:
        result = subprocess.run([sys.executable, str(Path(__file__)), str(out), str(int(observed))],
                                cwd=ROOT, env=env, capture_output=True, timeout=300)
        log.write(result.stdout + result.stderr)
    assert result.returncode == 0, (result.stdout + result.stderr)[-5000:].decode()
    return json.loads((out / 'receipt.json').read_text())


def test_default_native_counts_and_explicit_recipe_parity(tmp_path):
    implicit = tmp_path / 'default'
    explicit = tmp_path / 'explicit'
    receipt = run_native(implicit, [], observed=True)
    assert receipt['updates'] == 1250
    assert receipt['retained'] == list(range(251, 1251))
    run_native(explicit, ['--preparation-engine', 'grouped-regression',
                         '--gibbs-enable', 'inner-centered-variance',
                         '--gibbs-schedule', 'legacy', '--use-sampled-betas-in-gibbs',
                         '--min-num-burn-in', '251', '--max-num-burn-in', '251',
                         '--min-num-post-burn-in', '1000', '--max-num-post-burn-in', '1000',
                         '--max-num-iter', '1251', '--total-num-iter-gibbs', '1251',
                         '--max-num-restarts', '0', '--gibbs-reruns', '1', '--disable-stall-detection'])
    for name in ['genes.tsv.gz', 'annotations.tsv.gz']:
        with gzip.open(implicit / name, 'rb') as a, gzip.open(explicit / name, 'rb') as b:
            assert a.read() == b.read(), name


if __name__ == '__main__':
    import resource
    import time
    import pigean
    import pigean.state as state

    out = Path(sys.argv[1])
    receipt = {'updates': 0, 'retained': [], 'observed': bool(int(sys.argv[2]))}
    if receipt['observed']:
        original_update = state._compute_gibbs_iteration_betas_and_priors
        original_retain = state._advance_gibbs_post_burn_state

        def update(*a, **kw):
            result = original_update(*a, **kw)
            receipt['updates'] += 1
            return result

        def retain(*a, **kw):
            result = original_retain(*a, **kw)
            receipt['retained'].append(receipt['updates'])
            return result

        state._compute_gibbs_iteration_betas_and_priors = update
        state._advance_gibbs_post_burn_state = retain
    start = time.monotonic()
    rc = pigean.main(json.loads((out / 'command.json').read_text()))
    assert rc in (None, 0), rc
    receipt.update(seconds=time.monotonic()-start,
                   max_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
