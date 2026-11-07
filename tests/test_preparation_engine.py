"""Preparation selection must not leak into inner fitting or Gibbs."""
from types import SimpleNamespace
import pytest
from pigean import pipeline
from pegs_shared.gibbs_features import ACTIVE, default_feature_scope
from pegs_shared.sparse_regression import marginal_backend
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def effective(*args):
    result = subprocess.run([sys.executable, '-m', 'pigean', *args, '--print-effective-config'],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT/'src')), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)['options']

@pytest.mark.parametrize('engine',['legacy','sparse-regression','grouped-regression'])
def test_cli_independent(engine):
    o=effective('gibbs','--gwas-in','input','--preparation-engine',engine)
    assert o['preparation_engine']==engine
    assert o['gibbs_engine']=='legacy'
    assert o['use_sampled_betas_in_gibbs']
    assert o['gibbs_schedule_resolved']=='retained'

def test_default_unchanged():
    assert effective('gibbs','--gwas-in','input')['preparation_engine']=='legacy'

@pytest.mark.parametrize('engine',['sparse-regression','grouped-regression'])
def test_context_restored_and_optouts_respected(engine):
    with default_feature_scope(disable=['warm-start-layout']):
        before=ACTIVE.get()
        with pytest.raises(RuntimeError):
            with pipeline.preparation_backend(SimpleNamespace(preparation_engine=engine)):
                assert 'sparse-regression' in ACTIVE.get()
                assert 'warm-start-layout' not in ACTIVE.get()
                raise RuntimeError('sentinel')
        assert ACTIVE.get()==before

def test_legacy_preserves_python_caller_context():
    with marginal_backend('grouped-regression'):
        before=ACTIVE.get()
        with pipeline.preparation_backend(SimpleNamespace()):
            assert ACTIVE.get()==before

def test_pipeline_scope(monkeypatch):
    observed={}
    def record(name):
        def fn(*a,**k):observed[name]=ACTIVE.get()
        return fn
    monkeypatch.setattr(pipeline.pigean_main_support,'run_main_adaptive_read_x',record('read'))
    for stage in ['beta_tilde','beta','priors','gibbs']:
        monkeypatch.setattr(pipeline,'run_main_'+stage+'_stage',record(stage))
    options=SimpleNamespace(preparation_engine='grouped-regression',X_in=['x'],X_out=None,Xd_out=None,V_out=None)
    state=SimpleNamespace(p=.001,has_gene_sets=lambda:True)
    before=ACTIVE.get()
    pipeline.run_main_non_huge_pipeline(None,state,options,{'run_sim':False},None,False)
    for key in ['read','beta_tilde']:assert 'grouped-regression' in observed[key]
    for key in ['beta','priors','gibbs']:assert observed[key]==before
