"""Public CLI defaults must preserve explicit settings and unrelated workflows."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def effective(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run([sys.executable, '-m', 'pigean', *args, '--print-effective-config'],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)['options']


@pytest.mark.parametrize('source', ['--gwas-in', '--huge-statistics-in'])
def test_genetic_gibbs_defaults_to_sampled(source):
    assert effective('gibbs', source, 'input')['use_sampled_betas_in_gibbs'] is True


@pytest.mark.parametrize('flags,expected', [
    (['--use-mean-betas-in-gibbs'], False),
    (['--use-sampled-betas-in-gibbs', '--use-mean-betas-in-gibbs'], False),
    (['--use-mean-betas-in-gibbs', '--use-sampled-betas-in-gibbs'], True),
])
def test_feedback_override_and_last_flag_wins(flags, expected):
    assert effective('gibbs', '--gwas-in', 'input', *flags)['use_sampled_betas_in_gibbs'] is expected


@pytest.mark.parametrize('mode,flags', [('gibbs', []), ('gibbs', ['--gene-list', 'GCK,HNF1A', '--gene-universe-from-x']),
                                     ('betas', ['--gwas-in', 'input']), ('huge', ['--gwas-in', 'input'])])
def test_feedback_scope(mode, flags):
    assert not effective(mode, *flags)['use_sampled_betas_in_gibbs']


def test_config_false_and_cli_override(tmp_path):
    config = tmp_path/'mean.json'
    config.write_text(json.dumps({'use_sampled_betas_in_gibbs': False}))
    assert effective('gibbs', '--gwas-in', 'input', '--config', str(config))['use_sampled_betas_in_gibbs'] is False
    assert effective('gibbs', '--gwas-in', 'input', '--config', str(config), '--use-sampled-betas-in-gibbs')['use_sampled_betas_in_gibbs'] is True


@pytest.mark.parametrize('source', ['--gwas-in', '--huge-statistics-in'])
def test_genetic_schedule_matches_qualified_recipe(source):
    from pigean.gibbs_defaults import RETAINED_SCHEDULE
    options = effective('gibbs', source, 'input')
    assert options['gibbs_schedule_resolved'] == 'retained'
    assert {key: options[key] for key in RETAINED_SCHEDULE} == RETAINED_SCHEDULE


@pytest.mark.parametrize('flags', [[], ['--use-mean-betas-in-gibbs']])
def test_legacy_schedule_independent_of_feedback(flags):
    options = effective('gibbs', '--gwas-in', 'input', '--gibbs-schedule', 'legacy', *flags)
    assert options['max_num_iter'] == options['total_num_iter_gibbs'] == 500
    assert options['min_num_burn_in'] == options['min_num_post_burn_in'] == 10
    assert options['use_sampled_betas_in_gibbs'] is (not bool(flags))


def test_mean_feedback_does_not_change_retention():
    options = effective('gibbs', '--gwas-in', 'input', '--use-mean-betas-in-gibbs')
    assert options['gibbs_schedule_resolved'] == 'retained'


@pytest.mark.parametrize('flags', [
    ['--max-num-iter', '20'], ['--min-num-burn-in', '5'],
    ['--gibbs-reruns', '2'], ['--enable-stall-detection'],
])
def test_custom_controller_opts_out_of_auto(flags):
    options = effective('gibbs', '--gwas-in', 'input', *flags)
    assert options['gibbs_schedule_resolved'] == 'legacy'
    assert options['use_sampled_betas_in_gibbs'] is True
    assert options['min_num_post_burn_in'] == 10


def test_config_budget_and_cli_precedence(tmp_path):
    config = tmp_path/'custom.json'
    config.write_text(json.dumps({'max_num_iter': 20}))
    options = effective('gibbs', '--gwas-in', 'input', '--config', str(config))
    assert options['max_num_iter'] == options['total_num_iter_gibbs'] == 20
    assert options['gibbs_schedule_resolved'] == 'legacy'
    config.write_text(json.dumps({'gibbs_schedule': 'legacy'}))
    options = effective('gibbs', '--gwas-in', 'input', '--config', str(config), '--gibbs-schedule', 'retained')
    assert options['max_num_iter'] == 1251


@pytest.mark.parametrize('mode,flags', [('gibbs', []), ('gibbs', ['--gene-list', 'GCK,HNF1A', '--gene-universe-from-x']),
                                     ('betas', ['--gwas-in', 'input']), ('huge', ['--gwas-in', 'input'])])
def test_schedule_scope(mode, flags):
    options = effective(mode, *flags)
    assert options['gibbs_schedule_resolved'] == 'legacy'
    assert options['max_num_iter'] == 500


def test_retained_schedule_rejects_conflicting_budget():
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run([sys.executable, '-m', 'pigean', 'gibbs', '--gwas-in', 'input',
                             '--gibbs-schedule', 'retained', '--max-num-iter', '20',
                             '--print-effective-config'], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'conflicts with explicit controller settings' in result.stderr + result.stdout
