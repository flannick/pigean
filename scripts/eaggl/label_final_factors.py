"""Label final factor tables without refitting or modifying numeric outputs.

Live calls require an API key in the environment. --mock
tests alignment without making network calls or producing biological labels.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


def factor_columns(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        columns = reader.fieldnames or []
        found = [name for name in columns if re.fullmatch(r'Factor[1-9]\d*', name)]
        if len(found) != len(set(found)) or any(re.fullmatch(r'Factor\d+', name)
                and name not in found for name in columns):
            raise ValueError('Duplicate or invalid factor columns in %s' % path)
        for row in reader:
            for name in found:
                value = float(row[name])
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Factor loadings must be finite and nonnegative: %s' % path)
    return sorted(found, key=lambda name: int(name[6:]))


def write_labeled_factors(source, destination, labels):
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rt') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        fields = reader.fieldnames
        if fields is None or 'Factor' not in fields or 'label' not in fields:
            raise ValueError('Factor summary requires Factor and label columns')
        rows = list(reader)
    ids = [row['Factor'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError('Factor summary IDs do not match the final loading tables')
    for row in rows:
        row.update(labels[row['Factor']])
    fields = list(dict.fromkeys([*fields, *next(iter(labels.values()))]))
    with gzip.open(destination, 'wt') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--gene-loadings', required=True)
    parser.add_argument('--gene-set-loadings', required=True)
    parser.add_argument('--phenotype-loadings')
    parser.add_argument('--label-sources')
    parser.add_argument('--label-strategy', choices=('cross-factor', 'independent'), default='cross-factor')
    parser.add_argument('--label-context', help='Optional study context for all requested labels.')
    parser.add_argument('--additional-label-sources', action='append', default=[])
    parser.add_argument('--out', required=True)
    parser.add_argument('--factors-in')
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    genes = Path(args.gene_loadings).resolve()
    sets = Path(args.gene_set_loadings).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('factor_labels.tsv', 'labeling_metadata.json', 'factors.labeled.out.gz')):
        raise ValueError('Use a fresh labeling output directory to avoid stale labels after a failed run')
    columns = factor_columns(genes)
    if not columns or columns != factor_columns(sets):
        raise ValueError('Final gene and gene-set factor columns must agree exactly')
    inputs = [genes, sets]
    if args.phenotype_loadings:
        phenos = Path(args.phenotype_loadings).resolve()
        if factor_columns(phenos) != columns:
            raise ValueError('Phenotype factor columns must match the gene loadings')
        inputs.append(phenos)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    model = 'MOCK_NOT_BIOLOGICAL' if args.mock else os.environ.get('EAGGL_LLM_MODEL', 'gpt-6-luna')
    key = 'MOCK' if args.mock else (os.environ.get('EAGGL_LLM_AUTH_KEY')
                                   or os.environ.get('OPENAI_API_KEY'))
    if not model or not key:
        raise ValueError('Live labeling requires an API key environment variable')
    sys.path.insert(0, str(source/'src'))
    from eaggl import cli, label, labeling, main_support
    build_client = labeling.build_labeling_client
    calls = []

    class CheckedClient:
        def __init__(self, client):
            self.client = client
            self.model = model

        def query(self, prompt, warn_fn=None):
            calls.append(prompt)
            if args.mock:
                response = 'Mock factor label %d' % len(calls)
            else:
                def sanitized_warning(message):
                    if warn_fn:
                        warn_fn(str(message).replace(key, '[REDACTED]'))
                response = self.client.query(prompt, warn_fn=sanitized_warning)
            value = labeling.parse_label(response)
            if not value:
                raise RuntimeError("LLM returned an invalid mechanism label")
            return value

    def checked_builder(**kwargs):
        return CheckedClient(None if args.mock else build_client(**kwargs))

    labeling.build_labeling_client = checked_builder
    scratch = tempfile.TemporaryDirectory(prefix='label_scratch_', dir=out)
    command = ['label', '--label-gene-clusters-in', str(genes),
               '--label-gene-set-clusters-in', str(sets),
               '--factors-out', str(Path(scratch.name)/'label_only_factors.tsv'),
               '--factor-top-loading-type', 'raw', '--factor-output-scope', 'all',
               '--cluster-row-min-max-loading', '0', '--lmm-provider', 'openai',
               '--lmm-model', model, '--label-strategy', args.label_strategy]
    if args.phenotype_loadings:
        command.extend(['--label-pheno-clusters-in', str(phenos)])
    if args.label_sources:
        command.extend(['--label-sources', args.label_sources])
    if args.label_context:
        command.extend(['--label-context', args.label_context])
    for item in args.additional_label_sources:
        command.extend(['--additional-label-sources', item])
    if not main_support._bootstrap_cli(command):
        raise RuntimeError('Label-only CLI initialization failed')
    # Keep secrets out of command lines, output metadata and logged arguments.
    cli.options.lmm_auth_key = key
    metadata = dict(status='started', mock=args.mock, model=model, provider='openai',
                    label_context=args.label_context, label_strategy=args.label_strategy, inputs=hashes, factor_columns=columns, top_loading_type='raw',
                    evidence='five top genes and gene sets with available loading, coverage, enrichment and supporting genes',
                    numeric_inputs_unchanged=True)
    (out/'labeling_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    try:
        runtime = label.run_label_command(cli.options)
        groups = {"label": runtime.factor_labels}
        primary_column = labeling.label_column(labeling.parse_sources(runtime.factor_label_primary_sources))
        groups.update({column: values for column, values in runtime.factor_labels_by_sources.items()
                       if column != primary_column})
        if (runtime.num_factors() != len(columns)
                or any(len(values) != len(columns) for values in groups.values())
                or len(calls) != len(columns) * len(groups)):
            raise RuntimeError('Missing evidence or provider label for a requested factor/source combination')
        for p in inputs:
            if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]:
                raise RuntimeError('Numeric input was modified during labeling')
        records = {}
        for i, factor in enumerate(columns):
            record = {'label': runtime.factor_labels[i], 'label_sources': runtime.factor_label_primary_sources}
            record.update({column: values[i] for column, values in runtime.factor_labels_by_sources.items()})
            records[factor] = record
        with (out/'factor_labels.tsv').open('w') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['factor', *records[columns[0]], 'top_genes', 'top_gene_sets', 'label_source'])
            for i, factor in enumerate(columns):
                writer.writerow([factor, *records[factor].values(), ','.join(runtime.factor_top_genes[i]),
                    ','.join(runtime.factor_top_gene_sets[i]), 'mock_not_biological' if args.mock else 'llm'])
        prompt_ids = [(column, field) for field in groups for column in columns]
        (out/'label_prompt.txt').write_text('\n\n'.join(f'{factor} {field}\n{prompt}'
            for (factor, field), prompt in zip(prompt_ids, calls))+'\n')
        if args.factors_in:
            write_labeled_factors(Path(args.factors_in), out/'factors.labeled.out.gz', records)
        metadata.update(primary_sources=runtime.factor_label_primary_sources,
                        additional_labels=list(runtime.factor_labels_by_sources), requests=len(calls))
        metadata.update(status='mock_alignment_passed' if args.mock else 'llm_labels_validated',
                        factors=len(columns))
    except BaseException:
        metadata['status'] = 'failed_no_validated_labels'
        for name in ('factor_labels.tsv', 'factors.labeled.out.gz'):
            (out/name).unlink(missing_ok=True)
        raise
    finally:
        labeling.build_labeling_client = build_client
        (out/'labeling_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        scratch.cleanup()
    print(metadata['status'], len(columns), 'factors')


if __name__ == '__main__':
    main()
