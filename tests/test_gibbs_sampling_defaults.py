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
