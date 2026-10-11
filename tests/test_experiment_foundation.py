import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bench',ROOT/'scripts/experiments/bench.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

class ComparisonTests(unittest.TestCase):
    def test_scale_shift_not_hidden_by_correlation(self):
        a=pd.Series([1.,2.,3.]);b=2*a
        r=bench.metric(a,b)
        self.assertAlmostEqual(r['pearson'],1)
        self.assertEqual(r['slope_zero'],2)
        self.assertFalse(r['strict_pass'])
    def test_missing_not_zero_filled(self):
        r=bench.metric(pd.Series([1.,2.,3.]),pd.Series([1.,2.],index=[0,1]))
        self.assertEqual(r['n_missing'],1);self.assertFalse(r['strict_pass'])
    def test_constants_inconclusive(self):
        r=bench.metric(pd.Series([0.,0.,0.]),pd.Series([0.,0.,0.]))
        self.assertFalse(r['valid']);self.assertFalse(r['strict_pass'])
    def test_missing_replicate_not_averaged_away(self):
        with tempfile.TemporaryDirectory() as d:
            dirs=[]
            for i,values in enumerate(([1.,2.,3.],[1.,float('nan'),3.])):
                p=Path(d)/str(i);p.mkdir();dirs.append(p)
                pd.DataFrame({'Gene':['A','B','C'],'prior':values}).to_csv(p/'g.tsv.gz',sep='\t',index=False)
            with patch.object(bench,'validate_receipt',return_value={}):
                s=bench.load_metric(dirs,'g.tsv.gz','Gene','prior')
            self.assertTrue(pd.isna(s['B']))
    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({'Gene':['A','A'],'prior':[1,2]}).to_csv(Path(d)/'g.tsv.gz',sep='\t',index=False)
            with patch.object(bench,'validate_receipt',return_value={}):
                with self.assertRaises(ValueError):bench.load_metric([d],'g.tsv.gz','Gene','prior')
    def test_stale_input_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input';p.write_text('old');s={'inputs':{'x':{'path':str(p),'sha256':bench.digest(p)}}}
            p.write_text('new')
            with self.assertRaises(ValueError):bench.inputs(s)
    def test_publication_needs_both_traits(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.json';bench.write(p,{'commit':'a','status':'PASS','traits':['T2D']})
            with patch.object(bench,'identity',return_value={'commit':'a'}):
                with self.assertRaises(ValueError):bench.eligible(p,d)

class EngineTests(unittest.TestCase):
    def test_unknown_engine_rejected(self):
        r=subprocess.run([sys.executable,'-m','pigean','gibbs','--gibbs-engine','unknown'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('--gibbs-engine must be legacy',r.stderr)
    def test_legacy_default(self):
        from pigean.cli import parser
        opts,_=parser.parse_args([])
        self.assertEqual(opts.gibbs_engine,'legacy')
    def test_dispatch_legacy_forwards_without_changing_kwargs(self):
        from pigean import pipeline
        from types import SimpleNamespace
        from unittest.mock import Mock
        state=Mock();cfg=SimpleNamespace(run_kwargs={'num_chains':10},num_chains=10,total_num_iter=500)
        with patch.object(pipeline,'build_main_gibbs_stage_config',return_value=cfg):
            pipeline.run_main_gibbs_stage(None,state,SimpleNamespace(gibbs_engine='legacy'),{'run_gibbs':True})
        state.run_gibbs.assert_called_once_with(num_chains=10)
