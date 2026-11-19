"""Select from completed research candidate directories, optionally with a replicate grid."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from eaggl_capacity_research.candidate_data import SavedInputs, sha
from eaggl_mechanism_research.objectives import union_predict
from eaggl_leaf_quality_research.preselection import factor_qc
from eaggl_leaf_quality_research.selection import Policy, relationship_error, select


def metadata(path):
    return json.loads(path.read_text())


def index_candidates(paths):
    grid = {}
    for path in paths:
        phi = float(path.name)
        if phi in grid or not np.isfinite(phi) or phi <= 0:
            raise ValueError(f"Invalid or duplicate phi {path}")
        grid[phi] = path
    return grid


def load_candidate(path, inputs):
    contract_file = path.parents[1]/'contract.json'
    contract = metadata(contract_file)
    if (contract['target_hash'], contract['genes_hash'], contract['link']) != (
            inputs.target_hash, inputs.genes_hash, 'union') or contract['inputs'] != inputs.hashes:
        raise ValueError(f"Candidate input mismatch at {path}")
    export = path/'consensus/integrated_export'
    q = metadata(export/'qualification.json')
    if (not q['qualified'] or q['key']['phi'] != float(path.name)
            or q['key']['contract_sha256'] != sha(contract_file)
            or q['fit_sha256'] != sha(export/'fit.npz')):
        raise ValueError(f"Invalid consensus export at {path}")
    with np.load(export/'fit.npz') as z:
        w = z['W']
        indices = z['original_factor_indices']
    starts = []
    seeds = contract['restart_seeds']
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('Expected five distinct seeds')
    for i, seed in enumerate(seeds):
        folder = path/f'raw_{i}'
        raw = metadata(folder/'result.json')
        if (not raw['qualified'] or raw['key']['seed'] != seed or
                raw['key']['target_hash'] != inputs.target_hash or
                raw['fit_sha256'] != sha(folder/'fit.npz')):
            raise ValueError(f"Invalid raw fit at {folder}")
        with np.load(folder/'fit.npz') as z:
            v = z['W']
        starts.append(v[:, (v*v).sum(0) > contract['b0']])
    return w, indices, starts, contract


def run(inputs_path, candidates, replicas, output, *, policy=Policy()):
    grid = index_candidates(candidates)
    repeats = index_candidates(replicas)
    if repeats and repeats.keys() != grid.keys():
        raise ValueError('Supply independent ensembles for the whole comparison grid')
    inputs = SavedInputs(inputs_path.resolve())
    records = []
    scientific_contract = None
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp)/'output'
        stage.mkdir()
        for phi, path in sorted(grid.items()):
            w, indices, starts, contract = load_candidate(path, inputs)
            settings = {k:v for k,v in contract.items() if k not in ('seed','restart_seeds')}
            if scientific_contract is not None and settings != scientific_contract:
                raise ValueError('Primary candidate models differ across phi')
            scientific_contract = settings
            other = None
            if repeats:
                other, _, _, comparison = load_candidate(repeats[phi], inputs)
                if settings != {k:v for k,v in comparison.items() if k not in ('seed','restart_seeds')}:
                    raise ValueError('Independent ensemble settings differ')
                if set(contract['restart_seeds']) & set(comparison['restart_seeds']):
                    raise ValueError('Ensembles reuse random seeds')
            rows, summary = factor_qc(w, starts, other, policy)
            folder = stage/f'{phi:g}'
            folder.mkdir()
            for row, original in zip(rows, indices):
                row.update(phi=phi, original_consensus_factor=int(original)+1)
            pd.DataFrame(rows).to_csv(folder/'factor_qc.tsv.gz',sep='\t',index=False)
            records.append(dict(id=f'{phi:g}', phi=phi, numerically_qualified=True, **summary,
                                **relationship_error(inputs.target, union_predict(w)),
                                source=str(path.resolve()), replicate_source=str(repeats[phi].resolve()) if repeats else None))
        decision = select(records, policy)
        pd.DataFrame(records).to_csv(stage/'phi_summary.tsv.gz',sep='\t',index=False)
        if decision['selected'] is not None:
            source = grid[decision['selected_phi']]/'consensus/integrated_export/fit.npz'
            shutil.copyfile(source, stage/'selected_fit.npz')
            qc = pd.read_csv(stage/decision['selected']/'factor_qc.tsv.gz', sep='\t')
            passing = qc.loc[~qc.flagged, 'factor'].to_numpy(dtype=int)
            with np.load(source) as fit:
                leaf_table = pd.DataFrame(fit['W'][:, passing - 1], index=inputs.genes,
                                          columns=[f'Factor{i}' for i in passing])
            leaf_table.to_csv(stage/'leaf_loadings.tsv.gz', sep='\t', index_label='Gene')
            shutil.copyfile(stage/decision['selected']/'factor_qc.tsv.gz',stage/'selected_factor_qc.tsv.gz')
            decision.update(selected_fit_sha256=sha(source), selected_qc_sha256=sha(stage/'selected_factor_qc.tsv.gz'))
        (stage/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
        stage.rename(output)
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--help-expert', action='help', help='Show all factor QC and replication controls.')
    parser.add_argument('--inputs',required=True,type=Path)
    parser.add_argument('--candidates',required=True,type=Path,nargs='+')
    parser.add_argument('--replicates',type=Path,nargs='+',default=[])
    parser.add_argument('--output',required=True,type=Path)
    defaults = Policy()
    parser.add_argument('--phi-error-allowance-points', type=float, default=100*defaults.epsilon,
                        help='Allowed increase in normalized relationship error in percentage points '
                             '(default: %(default)g). Smaller values favor finer resolution.')
    advanced = parser.add_argument_group('Advanced leaf QC (applied before selection)')
    controls = {
        'minimum_effective_genes': 'Minimum effective number of contributing genes',
        'minimum_contrast': 'Minimum Hoyer loading contrast',
        'duplicate_cosine': 'Flag both leaves at or above this profile cosine',
        'support_cosine': 'Minimum post-refit cosine for a matching restart',
        'minimum_support': 'Minimum supporting restarts out of five',
        'maximum_flagged_fraction': 'Largest eligible fraction of flagged leaves',
        'maximum_flagged_energy': 'Largest eligible fraction of squared loading mass in flagged leaves',
        'replication_cosine': 'Minimum contribution cosine across independent ensembles (with --replicates)',
        'replication_amplitude_min': 'Lower directional amplitude ratio (with --replicates)',
        'replication_amplitude_max': 'Upper directional amplitude ratio (with --replicates)',
    }
    for field, description in controls.items():
        default = getattr(defaults, field)
        help_text = description+' (default: %(default)g)' if '--help-expert' in sys.argv[1:] else argparse.SUPPRESS
        advanced.add_argument('--qc-'+field.replace('_','-'), dest=field, default=default,
                              type=type(default), help=help_text)
    args = parser.parse_args()
    try:
        policy = Policy(epsilon=args.phi_error_allowance_points / 100,
                        **{field:getattr(args, field) for field in controls})
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(run(args.inputs,args.candidates,args.replicates,args.output,policy=policy),indent=2))
