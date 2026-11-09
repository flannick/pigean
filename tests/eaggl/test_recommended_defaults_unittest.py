"""Workflow defaults, explicit overrides and compatibility boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RecommendedDefaultsTest(unittest.TestCase):
    def config(self, *args, mode='factor'):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        p = subprocess.run([sys.executable, '-m', 'eaggl', mode, *args,
                            '--print-effective-config'], env=env, cwd=ROOT,
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        return json.loads(p.stdout)['options']

    def test_minimal_equals_explicit_recommended(self):
        minimal = self.config()
        explicit = self.config(
            '--discovery-model', 'gene_by_gene', '--phi', '0.01',
            '--factor-runs', '5', '--consensus-nmf',
            '--consensus-matching', 'pooled', '--consensus-aggregation', 'median',
            '--consensus-min-factor-cosine', '0.7', '--consensus-min-run-support', '0.5',
            '--factor-retention', 'backend_scale', '--max-num-factors', '30',
            '--symnmf-update-method', 'projected_gradient', '--symnmf-stopping', 'adaptive',
            '--factor-max-iterations', '20000', '--factor-rel-tol', '1e-6',
            '--fixed-basis-projection-method', 'verified_quadratic',
            '--projection-max-cycles', '2000', '--projection-stationarity-tol', '1e-8',
            '--factor-output-scope', 'all', '--cluster-row-min-max-loading', '0',
            '--factor-top-loading-type', 'raw')
        self.assertEqual(minimal, explicit)
        self.assertFalse(minimal['learn_phi'])

    def test_independent_controls_and_single_run(self):
        for retention in ('legacy', 'backend_scale'):
            for matching in ('reference', 'pooled'):
                with self.subTest(retention=retention, matching=matching):
                    c = self.config('--factor-retention', retention, '--consensus-matching', matching)
                    self.assertEqual(c['factor_retention'], retention)
                    self.assertEqual(c['consensus_matching'], matching)
        self.assertFalse(self.config('--factor-runs', '1')['consensus_nmf'])
        c = self.config('--no-consensus-nmf')
        self.assertFalse(c['consensus_nmf'])
        self.assertEqual(c['factor_runs'], 5)

    def test_alternate_modes_keep_legacy_settings(self):
        for mode, flags in [('factor', ['--discovery-model', 'gene_by_annotation']),
                            ('naive_factor', []),
                            ('factor', ['--factor-gene-clusters-in', 'saved.out.gz', '--gene-set-clusters-out', 'projection.out.gz', '--X-in', 'annotations.gmt.gz'])]:
            with self.subTest(mode=mode, flags=flags):
                c = self.config(*flags, mode=mode)
                self.assertEqual(c['discovery_model'], 'gene_by_annotation')
                self.assertEqual(c['phi'], .05)
                self.assertEqual(c['factor_runs'], 1)
                self.assertFalse(c['consensus_nmf'])
                self.assertEqual(c['factor_retention'], 'legacy')
                self.assertEqual(c['consensus_matching'], 'reference')
                self.assertEqual(c['fixed_basis_projection_method'], 'legacy')
        c = self.config('--discovery-model', 'gene_by_annotation')
        self.assertEqual(c['max_num_iterations'], 100)
        self.assertEqual(c['rel_tol'], 1e-4)

    def test_config_false_zero_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'config.json'
            p.write_text(json.dumps(dict(phi=.02, consensus_nmf=False,
                factor_runs=3, cluster_row_min_max_loading=0,
                factor_retention='legacy', consensus_matching='reference')))
            c = self.config('--config', str(p))
            self.assertFalse(c['consensus_nmf'])
            self.assertEqual(c['phi'], .02)
            self.assertEqual(c['factor_retention'], 'legacy')
            self.assertEqual(c['cluster_row_min_max_loading'], 0)
            c = self.config('--config', str(p), '--phi', '.03', '--consensus-nmf',
                            '--factor-retention', 'backend_scale')
            self.assertTrue(c['consensus_nmf'])
            self.assertEqual(c['phi'], .03)
            self.assertEqual(c['factor_retention'], 'backend_scale')

    def test_auto_phi_is_explicit(self):
        c = self.config('--learn-phi')
        self.assertTrue(c['learn_phi'])
        self.assertEqual(c['phi'], .01)
        self.assertIsNone(c['learn_phi_target_gene_effective_support'])


if __name__ == '__main__':
    unittest.main()
