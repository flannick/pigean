import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

class RequirementsCliTest(unittest.TestCase):
    def command(self,*flags):
        return subprocess.run([sys.executable,'-m','eaggl','factor',*flags,'--print-effective-config'],
                              env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)

    def test_requirements_default_and_explicit_equivalence(self):
        normal=self.command();self.assertEqual(normal.returncode,0,normal.stderr)
        opts=json.loads(normal.stdout)['options'];self.assertTrue(opts['learn_phi']);self.assertEqual(opts['phi'],.01)
        self.assertEqual(opts['phi_selection_objective'],'requirements')
        result=self.command('--learn-phi','--phi-selection-objective','requirements')
        self.assertEqual(result.returncode,0,result.stderr)
        opts=json.loads(result.stdout)['options'];self.assertEqual(opts['phi_selection_objective'],'requirements')
        self.assertIsNone(opts['learn_phi_target_gene_mass'])
        self.assertIsNone(opts['learn_phi_target_gene_effective_support'])

    def test_unqualified_overrides_rejected_before_loading(self):
        for flags in [['--learn-phi','--factor-runs','1'],['--learn-phi','--max-num-factors','60'],
                      ['--learn-phi','--learn-phi-values','.01,.02'],['--learn-phi','--learn-phi-only']]:
            with self.subTest(flags=flags):
                p=self.command('--phi-selection-objective','requirements',*flags)
                self.assertNotEqual(p.returncode,0)
                self.assertIn('Requirements selection',p.stderr+p.stdout)

    def test_fixed_phi_opt_out_and_config_precedence(self):
        import tempfile
        for flags in [('--no-learn-phi',), ('--phi', '.02')]:
            p=self.command(*flags);self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(json.loads(p.stdout)['options']['learn_phi'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json';path.write_text(json.dumps({'learn_phi':False}))
            p=self.command('--config',str(path));self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(json.loads(p.stdout)['options']['learn_phi'])
            p=self.command('--config',str(path),'--learn-phi');self.assertEqual(p.returncode,0,p.stderr)
            self.assertTrue(json.loads(p.stdout)['options']['learn_phi'])
            path.write_text(json.dumps({'phi':.01}))
            p=self.command('--config',str(path));self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(json.loads(p.stdout)['options']['learn_phi'])

    def test_previous_selectors_and_explicit_fixed_custom_recipe(self):
        for objective in ['composite','legacy']:
            p=self.command('--learn-phi','--phi-selection-objective',objective)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout)['options']['phi_selection_objective'],objective)
        p=self.command('--no-learn-phi','--factor-runs','1','--max-num-factors','60')
        self.assertEqual(p.returncode,0,p.stderr)
        p=self.command('--factor-runs','1')
        self.assertNotEqual(p.returncode,0)
        self.assertIn('--no-learn-phi',p.stderr+p.stdout)

    def test_extension_options_and_validation(self):
        for mode in ['improvement','noninferior','none']:
            p=self.command('--phi-search-extension',mode,'--phi-search-max-extra','0')
            self.assertEqual(p.returncode,0,p.stderr)
            opts=json.loads(p.stdout)['options']
            self.assertEqual(opts['phi_search_extension'],mode);self.assertEqual(opts['phi_search_max_extra'],0)
        p=self.command();self.assertEqual(p.returncode,0,p.stderr)
        opts=json.loads(p.stdout)['options'];self.assertEqual(opts['phi_search_extension'],'improvement');self.assertEqual(opts['phi_search_max_extra'],2)
        self.assertNotEqual(self.command('--phi-search-max-extra','-1').returncode,0)
        self.assertNotEqual(self.command('--phi-search-extension','bad').returncode,0)

if __name__=='__main__':unittest.main()
