"""Fail-closed labeling and local-only handoff checks; no paid/network calls."""
import csv
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts/eaggl'/('%s.py' % name))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


@pytest.mark.parametrize('response', [None, '1. Alone', '2. A\n1. B', '1. A\n1. B',
    'A\nB', '1. One two three four five six\n2. B', '1. \n2. B'])
def test_invalid_llm_responses_fail(response):
    with pytest.raises(RuntimeError):
        module('label_final_factors').check_response(response, 2)


def test_noncontiguous_factors_are_labeled_without_id_loss(tmp_path):
    genes = tmp_path/'genes.tsv'
    sets = tmp_path/'sets.tsv'
    factors = tmp_path/'factors.tsv'
    genes.write_text('Gene\tFactor10\tFactor2\nG10\t.8\t.1\nG2\t.2\t.9\n')
    sets.write_text('Gene_Set\tFactor2\tFactor10\nS2\t.9\t.1\nS10\t.1\t.9\n')
    factors.write_text('Factor\tlabel\tlambda\nFactor10\told10\t.2\nFactor2\told2\t.1\n')
    out = tmp_path/'labels'
    command = [sys.executable, str(ROOT/'scripts/eaggl/label_final_factors.py'),
               '--source', str(ROOT), '--gene-loadings', str(genes),
               '--gene-set-loadings', str(sets), '--factors-in', str(factors), '--out', str(out), '--mock']
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    with (out/'factor_labels.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    assert [r['factor'] for r in rows] == ['Factor2', 'Factor10']
    assert rows[0]['top_genes'].startswith('G2')
    assert rows[1]['top_gene_sets'].startswith('S10')
    with gzip.open(out/'factors.labeled.out.gz', 'rt') as stream:
        labeled = list(csv.DictReader(stream, delimiter='\t'))
    assert [r['Factor'] for r in labeled] == ['Factor10', 'Factor2']
    assert [r['label'] for r in labeled] == ['Mock factor label 2', 'Mock factor label 1']
    assert [r['lambda'] for r in labeled] == ['.2', '.1']
    assert factors.read_text().splitlines()[1] == 'Factor10\told10\t.2'
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0
    assert 'fresh' in again.stderr


def test_mismatched_summary_removes_partial_label_outputs(tmp_path):
    genes = tmp_path/'genes.tsv'
    sets = tmp_path/'sets.tsv'
    factors = tmp_path/'factors.tsv'
    genes.write_text('Gene\tFactor1\nA\t.2\n')
    sets.write_text('Gene_Set\tFactor1\nS\t.3\n')
    factors.write_text('Factor\tlabel\nFactor9\twrong\n')
    out = tmp_path/'labels'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/eaggl/label_final_factors.py'),
        '--source', str(ROOT), '--gene-loadings', str(genes), '--gene-set-loadings', str(sets),
        '--factors-in', str(factors), '--out', str(out), '--mock'], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (out/'factor_labels.tsv').exists()
    assert not (out/'factors.labeled.out.gz').exists()
    assert json.loads((out/'labeling_metadata.json').read_text())['status'] == 'failed_no_validated_labels'


def test_launcher_requires_labels_before_creating_outputs(tmp_path):
    out = tmp_path/'run'
    env = dict(os.environ)
    for key in ('EAGGL_LLM_MODEL', 'EAGGL_LLM_AUTH_KEY', 'OPENAI_API_KEY'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, str(ROOT/'scripts/eaggl/run_portal_trait.py'),
        '--trait-group', 'portal', '--phenotype', 'T2D', '--input-dir', str(tmp_path),
        '--out', str(out)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Live labels are required' in result.stderr
    assert not out.exists()


def test_launcher_has_no_s3_upload_command():
    source = (ROOT/'scripts/eaggl/run_portal_trait.py').read_text()
    assert "str(inputs/name)" in source
    assert 's3_uploads=False' in source
    assert '--recursive' not in source


@pytest.mark.parametrize('value', ['NaN', 'inf', '-.01'])
def test_label_helper_rejects_invalid_raw_loadings(tmp_path, value):
    table = tmp_path/'genes.tsv'
    table.write_text('Gene\tFactor1\nA\t'+value+'\n')
    with pytest.raises(ValueError, match='finite and nonnegative'):
        module('label_final_factors').factor_columns(table)


@pytest.mark.parametrize('content', ['1. Insulin secretion', None, '2. Wrong numbering'])
def test_live_label_path_with_mocked_http_is_fail_closed(tmp_path, monkeypatch, content):
    import urllib.request
    helper = module('label_final_factors')
    genes, sets = tmp_path/'genes.tsv', tmp_path/'sets.tsv'
    genes.write_text('Gene\tFactor7\nINS\t.9\n')
    sets.write_text('Gene_Set\tFactor7\nINSULIN_SECRETION\t.9\n')
    out = tmp_path/'labels'
    payload = {'choices': [{'message': {'content': content}}]}
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        assert timeout == 60
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setenv('EAGGL_LLM_MODEL', 'TEST_MODEL_NO_NETWORK')
    monkeypatch.setenv('EAGGL_LLM_AUTH_KEY', 'TEST_SECRET_NOT_REAL')
    monkeypatch.setattr(sys, 'argv', ['label_helper', '--source', str(ROOT),
        '--gene-loadings', str(genes), '--gene-set-loadings', str(sets), '--out', str(out)])
    if content == '1. Insulin secretion':
        helper.main()
        status = 'llm_labels_validated'
    else:
        with pytest.raises(RuntimeError):
            helper.main()
        status = 'failed_no_validated_labels'
        assert not (out/'factor_labels.tsv').exists()
    assert len(requests) == 1
    metadata = (out/'labeling_metadata.json').read_text()
    assert json.loads(metadata)['status'] == status
    assert 'TEST_SECRET_NOT_REAL' not in metadata


def test_label_publication_preserves_all_numeric_columns_and_unassigned_rows(tmp_path):
    labels_dir = tmp_path/'labeling'
    labels_dir.mkdir()
    (labels_dir/'factor_labels.tsv').write_text('factor\tlabel\nFactor1\tNew label\n')
    for name in ('gene_clusters.out.gz', 'gene_set_clusters.out.gz',
                 'gene_clusters_full_via_gene_sets.out.gz'):
        with gzip.open(tmp_path/name, 'wt') as stream:
            stream.write('Gene\tcluster\tlabel\tFactor1\tCosine_Factor1\n'
                         'A\tFactor1\tOld\t.23456\t.8765\nB\tNA\tNA\t0\t0\n')
    with gzip.open(tmp_path/'factors.out.gz', 'wt') as stream:
        stream.write('Factor\tlabel\tlambda\nFactor1\tOld\t.1\n')
    with gzip.open(labels_dir/'factors.labeled.out.gz', 'wt') as stream:
        stream.write('Factor\tlabel\tlambda\nFactor1\tNew label\t.1\n')
    module('run_portal_trait').publish_labels(tmp_path)
    for name in ('gene_clusters.out.gz', 'gene_set_clusters.out.gz',
                 'gene_clusters_full_via_gene_sets.out.gz'):
        with gzip.open(tmp_path/name, 'rt') as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        assert rows[0]['label'] == 'New label'
        assert rows[0]['Factor1'] == '.23456'
        assert rows[0]['Cosine_Factor1'] == '.8765'
        assert rows[1]['label'] == 'NA'
    assert (tmp_path/'factors.unlabeled.out.gz').exists()
