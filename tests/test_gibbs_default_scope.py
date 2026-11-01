"""Numerical opt-outs cover preprocessing without expanding arithmetic scope."""
from types import SimpleNamespace

import pytest

from pegs_shared.gibbs_features import ACTIVE, DEFAULT_FEATURES
from pegs_shared.sparse_regression import marginal_backend
from pigean import dispatch, pipeline


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_dispatch_scopes_preprocessing_and_gibbs_then_restores(monkeypatch, disabled, fail):
    options = SimpleNamespace(
        gibbs_engine='legacy', gibbs_enable=['inner-matmul'],
        gibbs_disable=['inner-rhat-stable'] if disabled else None,
        hide_opts=True, multi_y_in=None, run_phewas=False)
    numerical = set() if disabled else DEFAULT_FEATURES
    calls = []

    def build_state(opts):
        # Explicit arithmetic flags must not alter preprocessing.
        assert ACTIVE.get() == numerical
        calls.append('preprocessing')
        return SimpleNamespace(run_gibbs=sample)

    def sample(**kwargs):
        assert ACTIVE.get() == numerical | {'inner-matmul'}
        calls.append('gibbs')
        if fail:
            raise RuntimeError('test restoration')

    support = dispatch.pigean_main_support
    monkeypatch.setattr(support, 'build_runtime_state', build_state)
    monkeypatch.setattr(support, 'build_mode_state', lambda *a: {'run_huge': False, 'run_gibbs': True})
    monkeypatch.setattr(support, 'configure_hyperparameters_for_main', lambda *a: None)
    monkeypatch.setattr(support, 'load_main_y_inputs', lambda *a: False)
    monkeypatch.setattr(support, 'record_resolved_runtime_options', lambda *a: None)
    monkeypatch.setattr(pipeline, 'build_main_gibbs_stage_config', lambda *a:
                        SimpleNamespace(run_kwargs={}, num_chains=10, total_num_iter=500))
    monkeypatch.setattr(pipeline, 'run_main_non_huge_pipeline', lambda **kw:
                        pipeline.run_main_gibbs_stage(kw['services'], kw['state'],
                                                     kw['options'], kw['mode_state']))
    monkeypatch.setattr(dispatch.pigean_outputs, 'write_main_outputs_and_optional_phewas', lambda **kw: None)
    logs = []
    if fail:
        with pytest.raises(RuntimeError, match='test restoration'):
            dispatch.run_main_pipeline(options, 'gibbs', SimpleNamespace(log=logs.append))
    else:
        dispatch.run_main_pipeline(options, 'gibbs', SimpleNamespace(log=logs.append))
    assert calls == ['preprocessing', 'gibbs']
    assert ('inner-rhat-stable' in logs[0]) == (not disabled)
    assert ACTIVE.get() == DEFAULT_FEATURES


def test_numerical_scope_preserves_callers_arithmetic_context():
    from pegs_shared.gibbs_features import default_feature_scope
    with marginal_backend('legacy', enable=['inner-matmul']):
        with default_feature_scope(disable=['inner-rhat-stable']):
            assert ACTIVE.get() == {'inner-matmul'}
        assert ACTIVE.get() == DEFAULT_FEATURES | {'inner-matmul'}
    assert ACTIVE.get() == DEFAULT_FEATURES
