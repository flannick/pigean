import csv
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import sparse
from tests.eaggl.test_factor_stage_unittest import eaggl
from eaggl.supplied_factors import read_transposed_factors

ROOT = Path(__file__).resolve().parents[2]


class SuppliedFactorProjectionTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'eaggl', 'factor', *map(str, args)],
                              env=dict(os.environ, PYTHONPATH=str(ROOT/'src')),
                              cwd=ROOT, capture_output=True, text=True)

    def test_numerical_joint_marginal_and_zero_factors(self):
        runtime = eaggl.EagglState(background_prior=0.05, batch_size=10)
        runtime.params = {}
        runtime.genes = ['G1', 'G2', 'G3']
        runtime.gene_to_ind = dict(zip(runtime.genes, range(3)))
        runtime.gene_sets = ['S1', 'S2', 'ZERO']
        runtime.X_orig = sparse.csc_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        basis = np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 0.]])
        domain = eaggl.build_main_domain()
        domain.options = SimpleNamespace(gene_set_projection_mode='both')
        np.random.seed(7)
        eaggl.eaggl_factor._project_gene_set_factors_from_loaded_gene_factors(domain, runtime, runtime.genes, basis)
        np.testing.assert_allclose(runtime.exp_gene_set_factors_marginal,
                                   [[1., .5, 0.], [0., .5, 0.], [0., 0., 0.]])
        np.testing.assert_allclose(runtime.exp_gene_set_factors,
                                   [[1., 0., 0.], [0., .5, 0.], [0., 0., 0.]], atol=.025)
        for k in range(3):
            expected = runtime._project_H_with_fixed_W(basis[:, k:k+1], runtime.X_orig, None, None, cap_genes=True)
            np.testing.assert_allclose(runtime.exp_gene_set_factors_marginal[:, k], expected[:, 0], atol=1e-7)

    def test_transposed_input_validation(self):
        def bail(message):
            raise ValueError(message)
        domain = SimpleNamespace(bail=bail)
        for text in ['Factor\tA\tA\nf\t1\t2\n', 'Factor\tA\nf\t1\nf\t0\n',
                     'Factor\tA\nf\t-1\n', 'Factor\tA\nf\tnan\n', 'Factor\tA\nf\tinf\n',
                     'Factor\tA\nf\n', 'Factor\tA\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_transposed_factors(io.StringIO(text), domain)

    def test_cli_layouts_modes_and_multiple_gmts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wide, transposed = root/'wide.tsv.gz', root/'transposed.tsv.gz'
            with gzip.open(wide, 'wt') as f:
                f.write('Gene\tFactor1\tFactor2\nG1\t1\t1\nG2\t0\t1\nG3\t0\t0\n')
            with gzip.open(transposed, 'wt') as f:
                f.write('Factor\tG1\tG2\tG3\nFirst\t1\t0\t0\nSecond\t1\t1\t0\n')
            (root/'a.gmt').write_text('S1\tna\tG1\nS0\tna\tG3\n')
            (root/'b.gmt').write_text('S2\tna\tG2\n')
            (root/'inputs.txt').write_text(str(root/'a.gmt')+'\n'+str(root/'b.gmt')+'\n')
            observed = {}
            for layout, source in [('genes-by-factors', wide), ('factors-by-genes', transposed)]:
                for mode in ['joint', 'marginal', 'both']:
                    joint = root/f'{layout}.{mode}.joint.tsv.gz'
                    marginal = root/f'{layout}.{mode}.marginal.tsv.gz'
                    args = ['--factor-gene-clusters-in', source, '--factor-gene-clusters-layout', layout,
                            '--X-list', root/'inputs.txt', '--gene-set-projection-mode', mode,
                            '--cluster-row-min-max-loading', '0', '--seed', '7']
                    if mode != 'marginal':
                        args += ['--gene-set-clusters-out', joint]
                    if mode != 'joint':
                        args += ['--gene-set-clusters-marginal-out', marginal]
                    proc = self.run_cli(*args)
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                    for kind, path in [('joint', joint), ('marginal', marginal)]:
                        if not path.exists():
                            continue
                        with gzip.open(path, 'rt') as f:
                            rows = list(csv.DictReader(f, delimiter='\t'))
                        self.assertEqual({r['Gene_Set'] for r in rows}, {'S1', 'S2', 'S0'})
                        self.assertIn('Cosine_Factor1', rows[0])
                        values = {r['Gene_Set']: [float(r['Factor1']), float(r['Factor2'])] for r in rows}
                        if kind in observed:
                            for name in values:
                                np.testing.assert_allclose(values[name], observed[kind][name])
                        observed[kind] = values
            np.testing.assert_allclose(observed['marginal']['S1'], [1., .5])
            self.assertLess(observed['joint']['S1'][1], .025)

    def test_cli_rejects_conflicts(self):
        cases = [['--gene-set-projection-mode', 'both'],
                 ['--factor-gene-clusters-in', 'f', '--gene-set-projection-mode', 'both'],
                 ['--factor-gene-clusters-in', 'f', '--gene-set-clusters-marginal-out', 'm.gz'],
                 ['--factor-gene-clusters-in', 'f', '--gene-set-projection-mode', 'both',
                  '--gene-set-clusters-out', 'same.gz', '--gene-set-clusters-marginal-out', './same.gz'],
                 ['--factor-gene-clusters-layout', 'factors-by-genes']]
        for args in cases:
            with self.subTest(args=args):
                proc = self.run_cli(*args)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)

    def test_explicit_output_filters_override_projection_defaults(self):
        base = ['--factor-gene-clusters-in', 'factors.tsv', '--X-in', 'sets.gmt',
                '--gene-set-clusters-out', 'joint.gz', '--print-effective-config']
        proc = self.run_cli(*base)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        options = json.loads(proc.stdout)['options']
        self.assertEqual(options['factor_output_scope'], 'all')
        self.assertEqual(options['cluster_row_min_max_loading'], 0.)
        proc = self.run_cli(*base, '--factor-output-scope', 'primary', '--cluster-row-min-max-loading', '.2')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        options = json.loads(proc.stdout)['options']
        self.assertEqual(options['factor_output_scope'], 'primary')
        self.assertEqual(options['cluster_row_min_max_loading'], .2)

    def test_thousands_of_factors_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root/'factors.tsv'
            source.write_text('Factor\tA\tB\n' + ''.join(f'f{i}\t1\t0\n' for i in range(1200)))
            (root/'sets.gmt').write_text('S\tna\tA\n')
            out = root/'marginal.tsv.gz'
            proc = self.run_cli('--factor-gene-clusters-in', source,
                                '--factor-gene-clusters-layout', 'factors-by-genes',
                                '--X-in', root/'sets.gmt', '--gene-set-projection-mode', 'marginal',
                                '--gene-set-clusters-marginal-out', out)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            with gzip.open(out, 'rt') as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            self.assertTrue(rows, proc.stdout + proc.stderr)
            self.assertEqual(float(rows[0]['Factor1200']), 1.)
            self.assertEqual(sum(k.startswith('Factor') for k in rows[0]), 1200)


if __name__ == '__main__':
    unittest.main()
