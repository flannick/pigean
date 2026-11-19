"""Describe four saved consensus trees and their flat cuts."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def main(root, output=None):
    out=root/'results/EAGGL_hierarchy_v1'
    base=root/'results/EAGGL_four_trait_leaf_validation'
    output = out if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    inventories=[];summary=[];controls=[];pair_checks=[]
    for trait in ['T2D','CAD','NAFLD','Asthma']:
        dest=out/trait
        z=np.load(dest/'hierarchy_loadings.npz'); w=z['W']; genes=z['genes']
        tree=pd.read_csv(dest/'factor_hierarchy.tsv.gz',sep='\t').set_index('node')
        cuts=pd.read_csv(dest/'hierarchy_cuts.tsv.gz',sep='\t',dtype={'leaf_factors':str})
        leaves=tree[tree.leaf_id.notna()]
        ids=leaves.leaf_id.to_numpy(int); n=len(ids);v=w[:,:n]
        p=base/trait/'selection';phi=json.loads((p/'selection.json').read_text())['selected_phi']
        panels=pd.read_csv(p/'biological_panels.tsv.gz',sep='\t').query('arm=="Research" and phi==@phi')
        anchors=dict(zip(panels.panel,panels.factor.astype(int)))
        if trait=='NAFLD':
            for gene in ('CYP4A11','UGT1A9','BAAT','ACOX1'):
                anchors[gene]=ids[v[np.flatnonzero(genes==gene)[0]].argmax()]
        shared=1-squareform(pdist((v/v.sum(0)).T,'cityblock'))/2
        refparts={}
        for (metric,floor),rows in cuts.groupby(['metric','floor']):
            lab=np.full(n,-1)
            for r in rows.itertuples():
                children=list(map(int,r.leaf_factors.split(',')));where=np.isin(ids,children)
                assert where.sum()==len(children) and np.all(lab[where]==-1)
                lab[where]=r.node
                top=','.join(genes[np.argsort(-w[:,r.node],kind='stable')[:10]])
                anchor=[name for name,fac in anchors.items() if fac in children]
                item=dict(trait=trait,metric=metric,floor=floor,node=r.node,leaf_factors=r.leaf_factors,
                          n_leaves=len(children),anchors=';'.join(anchor),top_genes=top,
                          **tree.loc[r.node,['effective_genes','min_leaf_cosine','min_shared_mass','shared_leaf_pair_fraction']].to_dict())
                inventories.append(item)
                if anchor:controls.append(item)
            assert np.all(lab>=0)
            i,j=np.triu_indices(n,1);same=lab[i]==lab[j];between=shared[i[~same],j[~same]]
            profiles=w[:,rows.node.to_numpy()].T
            parent_overlap=1-pdist(profiles/profiles.sum(1,keepdims=True),'cityblock')/2
            summary.append(dict(displayed_max_overlap=parent_overlap.max() if len(parent_overlap) else np.nan,
                                displayed_pairs_over_half=int((parent_overlap>=.5).sum()),trait=trait,metric=metric,floor=floor,groups=len(rows),merged_groups=int((rows.leaf_factors.str.contains(',')).sum()),
                                largest=max(np.bincount(lab)),within_min=shared[i[same],j[same]].min() if same.any() else np.nan,
                                between_max=between.max() if between.size else np.nan,between_over_half=int((between>=.5).sum()),
                                between_pairs=len(between)))
            refparts[metric,floor]=lab
        a=refparts['min_leaf_cosine',.5];b=refparts['min_shared_mass',.5]
        x=a[i]==a[j];y=b[i]==b[j]
        pair_checks.append(dict(trait=trait,cosine_pairs=x.sum(),overlap_pairs=y.sum(),common_pairs=(x&y).sum()))
    for name,items in [('families',inventories),('anchor_families',controls),('cut_summary',summary),
                       ('pair_comparison',pair_checks)]:
        pd.DataFrame(items).to_csv(output/f'{name}.tsv.gz',sep='\t',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root',type=Path,required=True)
    parser.add_argument('--output', type=Path, help='Directory for the four descriptive tables')
    args = parser.parse_args()
    main(args.analysis_root.resolve(), args.output)
