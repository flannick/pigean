"""The SE schedule must not leak into retained updates or later calls."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pigean import gibbs, model
from pigean.se_warmup import (warmup_schedule, correlation_se_bypass,
    bypass_correlation_se, epoch_warmup_length, validate_warmup)


def final_se():
    return model.finalize_regression(None, np.array([1., 2.]), np.array([1., 1.]),
        np.array([3., .5]), log_fn=lambda *a: None, warn_fn=lambda *a: None,
        trace_level=0)[1]


def test_scope_restores_on_nested_calls_and_exception():
    np.testing.assert_array_equal(final_se(), [3., .5])
    with pytest.raises(RuntimeError):
        with warmup_schedule(125), correlation_se_bypass(True):
            np.testing.assert_array_equal(final_se(), [1., 1.])
            with correlation_se_bypass(False):
                np.testing.assert_array_equal(final_se(), [3., .5])
            assert bypass_correlation_se()
            raise RuntimeError('stop')
    assert not bypass_correlation_se() and epoch_warmup_length(251) == 0
    np.testing.assert_array_equal(final_se(), [3., .5])


@pytest.mark.parametrize('value', [-1, 1.2, True, None])
def test_invalid_count(value):
    with pytest.raises(ValueError): validate_warmup(value, 251)


def test_corrected_discard_interval_and_shortened_epochs():
    validate_warmup(125, 251)
    with pytest.raises(ValueError): validate_warmup(125, 250)
    with pytest.raises(ValueError): validate_warmup(1, 251, True)
    with warmup_schedule(125):
        assert epoch_warmup_length(251) == 125
        for burn in range(1, 252):
            n = epoch_warmup_length(burn)
            assert 2*n < burn
    assert epoch_warmup_length(251) == 0


@pytest.mark.parametrize('iteration,in_burn,expected', [(0, True, 1.),
    (124, True, 1.), (125, True, 3.), (249, True, 3.),
    (250, False, 3.), (0, False, 3.)])
def test_actual_iteration_scope(monkeypatch, iteration, in_burn, expected):
    observed = []
    def prepare(**kw):
        observed.append(final_se()[0])
        return {'epoch_iter_num': iteration+1, 'total_iter_num': iteration+1}, None
    def correction(**kw):
        assert not bypass_correlation_se()
        return {}
    callbacks = SimpleNamespace(prepare_gibbs_iteration_state_fn=prepare,
        run_gibbs_iteration_correction_and_updates_fn=correction,
        log_fn=lambda *a: None, info_level=0)
    monkeypatch.setattr(gibbs, '_extract_gibbs_iteration_update_state', lambda _: ((None, None, None), False))
    monkeypatch.setattr(gibbs, '_finalize_gibbs_iteration_after_correction', lambda **kw: None)
    progress = SimpleNamespace(min_num_burn_in_for_epoch=251, trace_chain_offset=0,
        iteration_progress_config={})
    with warmup_schedule(125):
        gibbs._run_single_gibbs_iteration(
            state=SimpleNamespace(_record_param=lambda *a, **kw: None), run_state=None,
            epoch_control={'in_burn_in': in_burn}, epoch_sums={}, epoch_priors={},
            epoch_runtime={}, correction_config=None, progress_runtime_config=progress,
            iteration_state_config={}, gene_set_stats_trace_fh=None,
            gene_prior_terms_trace_fh=None, iteration_num=iteration,
            log_bf_state=(None, None, None), callbacks=callbacks)
    assert observed == [expected]
    assert not bypass_correlation_se()


def cli(*args):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'))
    return subprocess.run([sys.executable, '-m', 'pigean', 'gibbs',
        '--print-effective-config', *args], cwd=root, env=env, text=True, capture_output=True)


def test_cli_default_opt_in_and_validation():
    p = cli(); assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['options']['gibbs_se_warmup_iterations'] == 0
    p = cli('--gibbs-se-warmup-iterations', '125', '--min-num-burn-in', '251')
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['options']['gibbs_se_warmup_iterations'] == 125
    for args in [('--gibbs-se-warmup-iterations', '-1'),
                 ('--gibbs-se-warmup-iterations', '125', '--min-num-burn-in', '250')]:
        assert cli(*args).returncode != 0


def test_cli_config_override(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'options': {'gibbs_se_warmup_iterations': 3,
                                            'min_num_burn_in': 7}}))
    p = cli('--config', str(config))
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['options']['gibbs_se_warmup_iterations'] == 3
    p = cli('--config', str(config), '--gibbs-se-warmup-iterations', '0')
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['options']['gibbs_se_warmup_iterations'] == 0
