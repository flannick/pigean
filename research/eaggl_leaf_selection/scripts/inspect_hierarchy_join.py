"""Inspect a saved consensus join using direct leaf matches in its contributing starts."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from select_research_phi_with_qc import ROOT, SavedInputs, load_candidate, sha
from eaggl_leaf_quality_research.selection import restart_matches


def leaves(tree, node):
    row = tree.loc[node]
    if pd.notna(row.leaf_id):
        return [int(row.leaf_id)]
    return leaves(tree, int(row.left)) + leaves(tree, int(row.right))


def similarities(a, b):
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    shared = np.minimum(a / a.sum(), b / b.sum()).sum()
    return float(cosine), float(shared)


def matched_starts(root, directory, manifest):
    receipt = json.loads((directory/'hierarchy.json').read_text())
    trait = receipt['trait']
    record = json.loads(manifest.read_text())['traits'][trait]
    if record['files'] != receipt['source_files'] or record['phi'] != receipt['phi']:
        raise ValueError('Hierarchy differs from selected fit manifest')
    files = {k: root / v['path'] for k, v in record['files'].items()}
    for key, path in files.items():
        if sha(path) != record['files'][key]['sha256']:
            raise ValueError(f'Selected {key} checksum mismatch')
    decision = json.loads(files['decision'].read_text())
    phi = decision['selected_phi']
    grid = pd.read_csv(files['decision'].with_name('phi_summary.tsv.gz'), sep='\t')
    source = Path(grid.loc[grid.phi == phi, 'source'].item())
    inputs = SavedInputs(files['axes'].parents[1])
    w, indices, starts, contract = load_candidate(source, inputs)
    if (float(source.name) != record['phi'] or
            sha(source/'consensus/integrated_export/fit.npz') != record['files']['loadings']['sha256']):
        raise ValueError('Raw starts do not belong to selected fit')
    qc = pd.read_csv(files['qc'], sep='\t')
    matches = restart_matches(w, starts, decision['policy']['support_cosine'])
    np.testing.assert_array_equal(indices + 1, qc.original_consensus_factor)
    np.testing.assert_array_equal(qc.factor, np.arange(1, w.shape[1] + 1))
    np.testing.assert_array_equal((matches >= 0).sum(0), qc.support)
    tree = pd.read_csv(directory/'factor_hierarchy.tsv.gz', sep='\t').set_index('node')
    np.testing.assert_array_equal(np.sort(tree.leaf_id.dropna()), qc.factor[~qc.flagged])
    return trait, phi, tree, w, starts, matches, contract['restart_seeds']


def inspect(root, directory, manifest, nodes):
    trait, phi, tree, w, starts, matches, seeds = matched_starts(root, directory, manifest)
    rows = []
    for node in dict.fromkeys(nodes):
        if node not in tree.index or pd.notna(tree.loc[node].leaf_id):
            raise ValueError(f'{node} must be an internal node of {trait}')
        for a, b in product(leaves(tree, int(tree.loc[node].left)), leaves(tree, int(tree.loc[node].right))):
            consensus_cos, consensus_shared = similarities(w[:, a-1], w[:, b-1])
            for seed, raw, match in zip(seeds, starts, matches):
                i, j = match[[a-1, b-1]]
                cos, shared = (similarities(raw[:, i], raw[:, j])
                               if i >= 0 and j >= 0 else (np.nan, np.nan))
                rows.append(dict(trait=trait, phi=phi, node=node, left_leaf=a, right_leaf=b,
                                 seed=seed, left_matched=i>=0, right_matched=j>=0,
                                 consensus_cosine=consensus_cos, consensus_shared_mass=consensus_shared,
                                 start_cosine=cos, start_shared_mass=shared))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--hierarchy', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'hierarchy_inputs.json')
    parser.add_argument('--node', type=int, nargs='+', required=True, help='Internal node IDs in the saved tree')
    parser.add_argument('--output', type=Path, required=True, help='New .tsv.gz file')
    args = parser.parse_args()
    rows = inspect(args.input_root.resolve(), args.hierarchy, args.manifest, args.node)
    rows.to_csv(args.output, sep='\t', index=False, mode='x', compression='gzip')
