"""Compare merge scores and scopes on the frozen consensus leaves."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_factor_hierarchy import ROOT, sha
from eaggl_leaf_quality_research.hierarchy import RULE_FIELDS, build_tree, cut_tree


def paired(leaves):
    return set(combinations(sorted(leaves), 2))


def run(root, out, all_methods=False):
    manifest = json.loads((ROOT/'hierarchy_inputs.json').read_text())['traits']
    archive = root/'results/EAGGL_hierarchy_v1'
    cases = pd.read_csv(archive/'resolution_review/cases.tsv.gz', sep='\t')
    summary, observations, families, merge_log = [], [], [], []
    for trait, record in manifest.items():
        paths = {key: root / v['path'] for key,v in record['files'].items()}
        for key, path in paths.items():
            if sha(path) != record['files'][key]['sha256']:
                raise ValueError(f'{trait} {key} checksum mismatch')
        q = pd.read_csv(paths['qc'], sep='\t')
        selected = q.loc[~q.flagged, 'factor'].to_numpy()
        with np.load(paths['loadings']) as fit, np.load(paths['axes']) as axes:
            w, genes = fit['W'][:,selected-1], axes['genes']
        ref = pd.read_csv(archive/trait/'hierarchy_cuts.tsv.gz', sep='\t')
        ref = ref.query("metric=='min_leaf_cosine' and floor==.5")
        reference = {frozenset(map(int, v.split(','))) for v in ref.leaf_factors}
        reference_pairs = set.union(*(paired(group) for group in reference))
        phi = record['phi']
        panels = pd.read_csv(paths['qc'].with_name('biological_panels.tsv.gz'), sep='\t')
        anchors = {int(r.factor):r.panel for r in panels.query("arm=='Research' and phi==@phi").itertuples()}
        rules = list(RULE_FIELDS) if all_methods else ['cosine', 'shared-mass', 'shared-squared-mass']
        scopes = ['leaves','nodes'] if all_methods else ['leaves']
        for rule, scope in product(rules, scopes):
            field = RULE_FIELDS[rule] if scope == 'leaves' else 'merge_prefix_floor'
            nodes, profiles, members = build_tree(w, selected, rule, scope)
            clades = {frozenset(m) for m in members if 1 < len(m) < len(selected)}
            baseline = (rule,scope) == ('cosine','leaves')
            if baseline:
                reference_clades = clades
            last = len(nodes)-len(reference)
            floor = nodes[last][field]
            settings = [('matched', .5 if baseline else floor)]
            if not baseline or all_methods:
                settings.append(('fixed', .4))
            for i in range(len(selected), len(nodes)):
                node = nodes[i]
                merge_log.append(dict(trait=trait, rule=rule, scope=scope, node=i,
                                      at_matched_count=i<=last, larger_child=max(len(members[node['left']]),
                                      len(members[node['right']])), **{k:node[k] for k in ('n_leaves',
                                      'merge_score','merge_prefix_floor','right_given_left','left_given_right',
                                      'left_background','right_background')}))
            for setting, threshold in settings:
                cut = cut_tree(nodes, threshold, field)
                chosen = cut
                if all_methods and setting == 'matched':
                    # Exact-count snapshot; a scalar threshold can skip this state on ties or inversions.
                    chosen = [i for i in range(last+1) if nodes[i]['parent'] is None or nodes[i]['parent']>last]
                groups = [members[i] for i in chosen]
                if baseline and setting=='matched' and {frozenset(m) for m in groups} != reference:
                    raise ValueError(f'{trait} baseline frontier changed')
                pairs = set.union(*(paired(m) for m in groups))
                key = dict(trait=trait, rule=rule, scope=scope, setting=setting, floor=threshold)
                np.testing.assert_array_equal(sorted(k for g in groups for k in g),selected)
                summary.append(dict(**key, groups=len(groups), groups_at_floor=len(cut),
                                    largest_family=max(map(len, groups)),
                                    distinct_anchor_groups=sum(bool(set(g)&anchors.keys()) for g in groups), baseline_groups=len(reference),
                                    common_internal=len(clades & reference_clades), internal=len(clades),
                                    exact_displayed=sum(frozenset(m) in reference for m in groups),
                                    shared_pairs=len(pairs & reference_pairs), pairs=len(pairs),
                                    baseline_pairs=len(reference_pairs)))
                for i, group in zip(chosen, groups):
                    top = np.argsort(-profiles[:,i], kind='stable')[:10]
                    families.append(dict(**key, node=i, leaves=','.join(map(str,sorted(group))),
                                         anchors=';'.join(anchors[k] for k in group if k in anchors),
                                         top=','.join(genes[top])))
                for r in cases[cases.trait==trait].itertuples():
                    a, b = [set(map(int,s.split(','))) for s in (r.left_leaves,r.right_leaves)]
                    cross = {tuple(sorted(pair)) for pair in product(a,b)}
                    observations.append(dict(**key, case=r.case, cross_kept=len(cross & pairs),
                                             cross_total=len(cross), within_kept=len((paired(a)|paired(b)) & pairs),
                                             within_total=len(paired(a)|paired(b))))
    out.mkdir(parents=True,exist_ok=False)
    for name, data in [('summary',summary),('cases',observations),('families',families),('merges',merge_log)]:
        pd.DataFrame(data).to_csv(out/(name+'.tsv.gz'),sep='\t',index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--all-methods',action='store_true',help='Test all six scores under both scopes')
    args = p.parse_args()
    run(args.analysis_root.resolve(),args.output,args.all_methods)
