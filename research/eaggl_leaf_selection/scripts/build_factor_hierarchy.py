"""Build and cut a display hierarchy for one fit in a leaf handoff manifest."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from eaggl_capacity_research.candidate_data import sha
from eaggl_leaf_quality_research.hierarchy import build_tree, cut_tree

FIELDS = ['node', 'parent', 'left', 'right', 'leaf_id', 'n_leaves', 'min_leaf_cosine',
          'min_shared_mass', 'effective_genes', 'shared_leaf_pair_fraction', 'gene_equivalents']


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    record = json.loads(args.manifest.read_text())['traits'][args.trait]
    files = record['files']
    paths = {key:args.input_root / item['path'] for key,item in files.items()}
    for key,path in paths.items():
        if sha(path) != files[key]['sha256']:
            raise ValueError(f'{key} checksum mismatch')
    qc = pd.read_csv(paths['qc'], sep='\t')
    with np.load(paths['axes']) as axes, np.load(paths['loadings']) as fit:
        w, genes = fit['W'], axes['genes']
        original = fit['original_factor_indices'] + 1
    if (w.shape != (len(genes),len(qc)) or not np.array_equal(qc.factor, np.arange(1,len(qc)+1))
            or not np.array_equal(qc.original_consensus_factor,original)
            or (len(qc) and qc.flagged.dtype != bool)):
        raise ValueError('Gene axes or factor QC identities do not match the fit')
    passing = ~qc.flagged.to_numpy(dtype=bool)
    if len(qc) != record['fitted'] or passing.sum() != record['passing']:
        raise ValueError('Unexpected fitted or passing leaf count')
    nodes, values, members = build_tree(w[:,passing], qc.factor[passing])
    tree = pd.DataFrame(nodes, columns=FIELDS)
    for key in ('node','left','right','parent','leaf_id'):
        tree[key] = tree[key].astype('Int64')
    tree['original_consensus_factor'] = tree.leaf_id.map(dict(zip(qc.factor,original))).astype('Int64')
    tree.insert(0,'trait',args.trait)
    rows = []
    for metric,floors in [('min_leaf_cosine',args.min_leaf_cosine),('min_shared_mass',args.min_shared_mass)]:
        for floor in dict.fromkeys(floors):
            for node in cut_tree(nodes,floor,metric):
                rows.append((metric,floor,node,','.join(map(str,sorted(members[node])))))
    output = args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp)/'tree'
        stage.mkdir()
        tree.to_csv(stage/'factor_hierarchy.tsv.gz',sep='\t',index=False)
        pd.DataFrame(rows,columns=['metric','floor','node','leaf_factors']).to_csv(
            stage/'hierarchy_cuts.tsv.gz',sep='\t',index=False)
        if args.max_gene_equivalents:
            size_rows = []
            for cap in dict.fromkeys(args.max_gene_equivalents):
                for i in cut_tree(nodes, cap, 'gene_equivalents'):
                    size = nodes[i]['gene_equivalents']
                    size_rows.append((cap, i, ','.join(map(str, sorted(members[i]))), size, size > cap))
            pd.DataFrame(size_rows, columns=['ceiling', 'node', 'leaf_factors', 'gene_equivalents',
                                             'oversize_leaf']).to_csv(stage/'hierarchy_size_cuts.tsv.gz', sep='\t', index=False)
        np.savez_compressed(stage/'hierarchy_loadings.npz',W=values,genes=genes,node=tree.node.to_numpy(dtype=int))
        (stage/'hierarchy.json').write_text(json.dumps(dict(version=1,trait=args.trait,phi=record['phi'],
            source_files=files,method='complete linkage on original leaf cosine',parent='noisy OR',
            skipped_flagged=qc.factor[~passing].tolist(),cosine_cuts=args.min_leaf_cosine,
            shared_mass_cuts=args.min_shared_mass, size_ceilings=args.max_gene_equivalents),indent=2)+'\n')
        stage.rename(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'hierarchy_inputs.json')
    p.add_argument('--input-root',required=True,type=Path,help='Base directory for paths in the manifest')
    p.add_argument('--trait',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--min-leaf-cosine',nargs='+',type=float,default=[],help='Optional cosine cut floors')
    p.add_argument('--min-shared-mass',nargs='+',type=float,default=[],help='Optional minimum common gene mass on the same tree')
    p.add_argument('--max-gene-equivalents', nargs='+', type=float, default=[],
                   help='Optional ceiling on summed discovery-gene memberships per displayed family')
    run(p.parse_args())
