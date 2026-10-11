import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import fcntl
from unittest.mock import patch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bench',ROOT/'scripts/experiments/bench.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

class ComparisonTests(unittest.TestCase):
    def test_matching_structural_nan_is_exact_but_not_strict(self):
        a=pd.Series([1.,2.,float('nan')])
        r=bench.metric(a,a)
        self.assertTrue(r['exact']);self.assertFalse(r['strict_pass'])
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

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.repo=self.root/'repo';self.repo.mkdir()
        self.spec=self.root/'spec.json';self.out=self.root/'output';self.lock=self.root/'lock'
        for args in [('init',),('config','user.name','Test'),('config','user.email','test@example.invalid')]:
            subprocess.run(['git','-C',str(self.repo),*args],check=True,capture_output=True)
        (self.repo/'source').write_text('source')
        subprocess.run(['git','-C',str(self.repo),'add','source'],check=True)
        subprocess.run(['git','-C',str(self.repo),'commit','-qm','fixture'],check=True)
        self.input=self.root/'input';self.input.write_text('frozen')
        self.recipe={'schema_version':1,'trait':'T2D','level':'complete_cached_input',
            'inputs':{'x':{'path':str(self.input),'sha256':bench.digest(self.input)}},
            'command':['{python}','-c',"from pathlib import Path; Path(r'{output}/result.gz').write_text('result')"],
            'outputs':['result.gz']}
        bench.write(self.spec,self.recipe)
    def test_completion_resume_and_output_tamper(self):
        bench.run(self.spec,self.repo,self.out,0,self.lock)
        receipt=self.out/'receipt.json';sha=bench.digest(receipt)
        bench.run(self.spec,self.repo,self.out,0,self.lock)
        self.assertEqual(sha,bench.digest(receipt))
        (self.out/'result.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Output changed'):bench.validate_receipt(receipt)
    def test_stale_source_and_failed_run_cannot_resume(self):
        self.recipe['command']=['{python}','-c','raise SystemExit(7)'];bench.write(self.spec,self.recipe)
        with self.assertRaisesRegex(ValueError,'Run failed'):bench.run(self.spec,self.repo,self.out,0,self.lock)
        self.assertEqual(bench.read(self.out/'receipt.json')['status'],'FAILED')
        with self.assertRaisesRegex(ValueError,'Incomplete'):bench.run(self.spec,self.repo,self.out,0,self.lock)
        (self.repo/'source').write_text('dirty')
        with self.assertRaisesRegex(ValueError,'Uncommitted'):bench.run(self.spec,self.repo,self.out,0,self.lock)
    def test_lock_prevents_concurrent_runs(self):
        with self.lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):bench.run(self.spec,self.repo,self.out,0,self.lock)
        self.assertFalse(self.out.exists())
    def test_interrupted_output_is_not_overwritten(self):
        self.out.mkdir();(self.out/'running.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Unfinished'):bench.run(self.spec,self.repo,self.out,0,self.lock)
    def test_abandoned_integration_rejected(self):
        with self.assertRaisesRegex(ValueError,'Abandoned'):
            bench.integrate(self.repo,self.root/'integration','codex/test','HEAD',['codex/blockwise-redesign'])
    def test_two_trait_matrix_and_resume(self):
        recipe=dict(self.recipe,baseline_commit=bench.git(self.repo,'rev-parse','HEAD'),
            outputs=['gene_stats.out.gz','gene_set_stats.out.gz'],
            command=['{python}','-c',"import gzip; from pathlib import Path; p=Path(r'{output}'); gzip.open(p/'gene_stats.out.gz','wt').write('Gene\\tprior\\nA\\t1\\nB\\t2\\nC\\t3\\n'); gzip.open(p/'gene_set_stats.out.gz','wt').write('Gene_Set\\tbeta_uncorrected\\nX\\t1\\nY\\t2\\nZ\\t3\\n')"])
        paths=[]
        for trait in ('T2D','NAFLD'):
            p=self.root/(trait+'.json');bench.write(p,dict(recipe,trait=trait));paths += ['--spec',str(p)]
        cmd=[sys.executable,str(ROOT/'scripts/experiments/matrix.py'),*paths,
            '--baseline-repo',str(self.repo),'--candidate-repo',str(self.repo),
            '--out',str(self.out),'--lock',str(self.lock)]
        for _ in range(2):
            p=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stdout[-1000:]+p.stderr[-2000:])
        for trait in ('T2D','NAFLD'):
            r=bench.validate_comparison(self.out/trait/'comparison.json')
            self.assertEqual(r['baseline_seeds'],list(range(5)))
            self.assertTrue(r['full_outputs_equal'])
            self.assertIsNotNone(r['confirmed_speedup'])
            self.assertEqual(bench.read(self.out/trait/'repeatability.json')['status'],'PASS')

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

class InferenceReplayTests(unittest.TestCase):
    def test_replay_matches_normal_run_with_nonzero_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);normal=root/'normal';normal.mkdir();replay=root/'replay';fixture=root/'fixture.pkl.gz'
            args=['gibbs','--X-in','tests/data/model_small/gene_set_list_mouse_2024.txt',
                '--gene-map-in','tests/data/model_small/portal_gencode.gene.map',
                '--gene-loc-file','tests/data/model_small/NCBI37.3.plink.gene.loc',
                '--positive-controls-list','GCK,HNF1A,HNF4A,PDX1','--gene-universe-from-x',
                '--deterministic','--seed','7','--num-chains','2','--num-chains-betas','2',
                '--max-num-iter','12','--total-num-iter-gibbs','12','--min-num-burn-in','5',
                '--min-num-post-burn-in','5','--max-num-restarts','0','--hide-opts','--hide-progress',
                '--output-detail','full']
            def run(cmd):
                p=subprocess.run([sys.executable,*cmd],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stdout[-2000:]+p.stderr[-2000:])
            tool=str(ROOT/'scripts/experiments/inference.py')
            run([tool,'capture','--fixture',str(fixture),'--',*args])
            manifest=json.loads(Path(str(fixture)+'.json').read_text())
            self.assertEqual(manifest['capture_command'],args)
            run([tool,'replay','--fixture',str(fixture),'--out',str(replay)])
            run(['-m','pigean',*args,'--gene-stats-out',str(normal/'gene_stats.out.gz'),
                 '--gene-set-stats-out',str(normal/'gene_set_stats.out.gz')])
            for name in ('gene_stats.out.gz','gene_set_stats.out.gz'):
                pd.testing.assert_frame_equal(pd.read_csv(normal/name,sep='\t'),pd.read_csv(replay/name,sep='\t'),check_exact=True)
