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

@pytest.mark.parametrize('input_flag',['--gwas-in','--huge-statistics-in'])
def test_genetic_gibbs_defaults(input_flag):
    o=effective('gibbs',input_flag,'input')
    assert o['preparation_engine']=='grouped-regression'
    assert 'inner-centered-variance' in o['gibbs_enable']

@pytest.mark.parametrize('args',[
    ('gibbs','--gene-list','GCK,HNF1A','--gene-universe-from-x'), ('betas','--gwas-in','input'),
    ('huge','--gwas-in','input')])
def test_other_workflows_unchanged(args):
    o=effective(*args)
    assert o['preparation_engine']=='legacy'
    assert 'inner-centered-variance' not in (o['gibbs_enable'] or [])

@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('disable',[False,True])
def test_independent_default_optouts(legacy,disable):
    args=['gibbs','--gwas-in','input']
    if legacy:args+=['--preparation-engine','legacy']
    if disable:args+=['--gibbs-disable','inner-centered-variance']
    o=effective(*args)
    assert o['preparation_engine']==('legacy' if legacy else 'grouped-regression')
    assert ('inner-centered-variance' in o['gibbs_enable'])== (not disable)
    assert o['use_sampled_betas_in_gibbs']
    assert o['gibbs_schedule_resolved']=='retained'

def test_default_variance_stays_out_of_preparation():
    o=effective('gibbs','--gwas-in','input')
    with default_feature_scope(enable=o['gibbs_enable'],disable=o['gibbs_disable']):
        assert 'inner-centered-variance' not in ACTIVE.get()
        with pipeline.preparation_backend(SimpleNamespace(**o)):
            assert 'grouped-regression' in ACTIVE.get()
            assert 'inner-centered-variance' not in ACTIVE.get()
        assert 'inner-centered-variance' not in ACTIVE.get()
        with marginal_backend(o['gibbs_engine'],enable=o['gibbs_enable'],disable=o['gibbs_disable']):
            assert 'inner-centered-variance' in ACTIVE.get()
        assert 'inner-centered-variance' not in ACTIVE.get()

def test_config_optouts_and_cli_override(tmp_path):
    config=tmp_path/'legacy.json'
    config.write_text(json.dumps({'preparation_engine':'legacy',
                                  'gibbs_disable':['inner-centered-variance']}))
    o=effective('gibbs','--gwas-in','input','--config',str(config))
    assert o['preparation_engine']=='legacy'
    assert 'inner-centered-variance' not in o['gibbs_enable']
    o=effective('gibbs','--gwas-in','input','--config',str(config),
                '--preparation-engine','grouped-regression')
    assert o['preparation_engine']=='grouped-regression'
    assert 'inner-centered-variance' not in o['gibbs_enable']

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
