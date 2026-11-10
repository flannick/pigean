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

    def test_opt_in_and_unchanged_defaults(self):
        normal=self.command();self.assertEqual(normal.returncode,0,normal.stderr)
        opts=json.loads(normal.stdout)['options'];self.assertFalse(opts['learn_phi']);self.assertEqual(opts['phi'],.01)
        self.assertEqual(opts['phi_selection_objective'],'composite')
        result=self.command('--learn-phi','--phi-selection-objective','requirements')
        self.assertEqual(result.returncode,0,result.stderr)
        opts=json.loads(result.stdout)['options'];self.assertEqual(opts['phi_selection_objective'],'requirements')
        self.assertIsNone(opts['learn_phi_target_gene_mass'])
        self.assertIsNone(opts['learn_phi_target_gene_effective_support'])

    def test_unqualified_overrides_rejected_before_loading(self):
        for flags in [[],['--learn-phi','--factor-runs','1'],['--learn-phi','--max-num-factors','60'],
                      ['--learn-phi','--learn-phi-values','.01,.02'],['--learn-phi','--learn-phi-only']]:
            with self.subTest(flags=flags):
                p=self.command('--phi-selection-objective','requirements',*flags)
                self.assertNotEqual(p.returncode,0)
                self.assertIn('Requirements selection',p.stderr+p.stdout)

if __name__=='__main__':unittest.main()
