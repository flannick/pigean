"""Describe saved four-trait trees and test co-grouping against archived fits."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

FLOORS=(.3,.5,.7)


def compare_partition(w,v,labels,floor,ids,anchors):
    cosine=(w/np.linalg.norm(w,axis=0)).T@(v/np.linalg.norm(v,axis=0))
    a,b=linear_sum_assignment(-cosine)
    keep=cosine[a,b]>=.7
    a,b=a[keep],b[keep]
    other=fcluster(linkage(pdist(v.T,'cosine'),method='complete'),1-floor,criterion='distance')
    i,j=np.triu_indices(len(a),1)
    ref=labels[a[i]]==labels[a[j]]
    alt=other[b[i]]==other[b[j]]
    same=int((ref & alt).sum())
    anchor_checks = {}
    for name,leaf in anchors.items():
        group=labels==labels[np.flatnonzero(ids==leaf)[0]]
        children=other[b[group[a]]]
        total=len(children)*(len(children)-1)//2
        coassigned=int((children[:,None]==children)[np.triu_indices(len(children),1)].sum())
        anchor_checks[name]=dict(size=int(group.sum()),matched_children=len(children),pairs=total,same=coassigned)
    return anchor_checks,dict(matched=len(a),passing=len(labels),other_active=v.shape[1],matched_pairs_in_reference=int(ref.sum()),
                matched_pairs_in_other=int(alt.sum()), shared_pairs=same,
                recall=same/ref.sum() if ref.any() else np.nan,
                precision=same/alt.sum() if alt.any() else np.nan)


def main(root):
    out=root/'results/EAGGL_hierarchy_v1'
    base=root/'results/EAGGL_four_trait_leaf_validation'
    perturbed={x['trait']:x for x in json.loads((root/'results/EAGGL_ARD_cross_trait_20261117/manifest.json').read_text())['traits']}
    inventories=[];summary=[];stability=[];controls=[];pair_checks=[];anchor_stability=[]
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
        grid=pd.read_csv(p/'phi_summary.tsv.gz',sep='\t')
        source=Path(grid[grid.phi==phi].iloc[0].source)
        level=json.loads((source.parents[1]/'contract.json').read_text())['b0']
        comparisons=[]
        for k in range(5):
            raw=np.load(source/f'raw_{k}/fit.npz')['W']
            comparisons.append((f'start_{k}',phi,v,raw[:,(raw*raw).sum(0)>level]))
        case=perturbed[trait]
        level=json.loads((root/case['manifest']).read_text())['hyper']['b0']
        strength=4 if phi==.04 else 1
        for part in range(2):
            zz=np.load(root/case['run_root']/f'p{part}_x{strength}/fit.npz')
            active=(zz['W']**2).sum(0)>level
            _,curr,other=np.intersect1d(genes,zz['genes'],return_indices=True)
            comparisons.append((f'masked_{part}',.01*strength,v[curr],zz['W'][other][:,active]))
        for cmp,other_phi,current,other in comparisons:
            for floor in FLOORS:
                grouped,stats=compare_partition(current,other,refparts['min_leaf_cosine',floor],floor,ids,anchors)
                for name,obs in grouped.items():
                    anchor_stability.append(dict(trait=trait,comparison=cmp,panel=name,floor=floor,**obs))
                stability.append(dict(trait=trait,comparison=cmp,phi=phi,other_phi=other_phi,common_genes=len(current),
                    floor=floor,**stats))
    for name,items in [('families',inventories),('anchor_families',controls),('cut_summary',summary),
                       ('stability',stability),('anchor_stability',anchor_stability),('pair_comparison',pair_checks)]:
        pd.DataFrame(items).to_csv(out/f'{name}.tsv.gz',sep='\t',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root',type=Path,required=True)
    main(parser.parse_args().analysis_root.resolve())
