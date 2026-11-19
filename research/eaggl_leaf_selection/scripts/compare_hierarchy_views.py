"""Compare cuts on four fixed consensus trees and review predeclared extreme examples."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from inspect_hierarchy_join import ROOT, matched_starts
from eaggl_leaf_quality_research.hierarchy import cut_tree


def overlap_vector(w, nodes):
    x = w[:, nodes].T
    return 1 - pdist(x / x.sum(axis=1, keepdims=True), 'cityblock') / 2


def describe_pair(a, b, genes):
    a, b = a / a.sum(), b / b.sum()
    common = np.minimum(a, b)
    def top(x, n=10):
        return np.argsort(-x, kind='stable')[:n]

    left, right = top(a), top(b)
    shared = common.sum()
    core = np.union1d(left, right)
    return dict(shared_mass=shared, common_top10=len(np.intersect1d(left, right)),
                shared_fraction_outside_both_top10=(1-common[core].sum()/shared) if shared else np.nan,
                left_top=','.join(genes[left]), right_top=','.join(genes[right]),
                left_enriched=','.join(genes[top(a-b, 5)]), right_enriched=','.join(genes[top(b-a, 5)]),
                top_common_mass=','.join(genes[top(common, 5)]))


def choose_cut(nodes, metric, target):
    values = {node[metric] for node in nodes}
    # Include the all-leaf frontier when all leaves are smaller than the lowest internal node.
    values.add(1. if metric == 'min_shared_mass' else np.finfo(float).tiny)
    options = []
    for threshold in sorted(values):
        frontier = sorted(cut_tree(nodes, threshold, metric))
        # Prefer more groups on count ties, then the least restrictive equivalent bound
        # among observed node thresholds.
        options.append((abs(len(frontier)-target), -len(frontier),
                        -threshold if metric == 'gene_equivalents' else threshold, threshold, frontier))
    return min(options)[-2:]


def run(root, output):
    archive = root/'results/EAGGL_hierarchy_v1'
    review, pair_stats, start_stats, summary, inventory = [], [], [], [], []
    for trait in ('NAFLD', 'T2D', 'CAD', 'Asthma'):
        directory = archive/trait
        _, phi, tree, consensus, starts, matches, seeds = matched_starts(root, directory, ROOT/'hierarchy_inputs.json')
        with np.load(directory/'hierarchy_loadings.npz') as z:
            w, genes = z['W'], z['genes']
            np.testing.assert_array_equal(z['node'], tree.index)
        nodes = []
        members = []
        for i, row in tree.iterrows():
            assert i == len(nodes)
            rec = row.to_dict()
            if pd.isna(row.left):
                rec['left'] = rec['right'] = None
                ids = [int(row.leaf_id)]
                np.testing.assert_array_equal(w[:,i], consensus[:,ids[0]-1])
            else:
                a, b = int(row.left), int(row.right)
                rec.update(left=a, right=b)
                ids = members[a]+members[b]
                np.testing.assert_allclose(w[:,i], w[:,a]+(1-w[:,a])*w[:,b], rtol=0, atol=1e-14)
            rec['gene_equivalents'] = float(w[:,i].sum())
            nodes.append(rec)
            members.append(ids)
        middle = sorted(cut_tree(nodes, .5))
        pairs = list(combinations(middle, 2))
        a, b = pairs[int(np.argmax(overlap_vector(w, middle)))]
        weak = min((i for i in middle if len(members[i]) > 1), key=lambda i: (nodes[i]['min_leaf_cosine'], i))
        cases = [('most_overlapping_separate', a, b), ('weakest_join_children',nodes[weak]['left'],nodes[weak]['right'])]
        for kind, i, j in cases:
            key = dict(trait=trait, case=kind, left_node=i, right_node=j)
            review.append(dict(**key, left_leaves=','.join(map(str, sorted(members[i]))),
                               right_leaves=','.join(map(str, sorted(members[j])))))
            pair_stats.append(dict(**key, **describe_pair(w[:,i], w[:,j], genes)))
            for raw, match, seed in zip(starts, matches, seeds):
                aa = match[np.array(members[i])-1]
                bb = match[np.array(members[j])-1]
                scored = {}
                if np.all(aa >= 0) and np.all(bb >= 0):
                    ra = 1-np.prod(1-raw[:,aa], axis=1)
                    rb = 1-np.prod(1-raw[:,bb], axis=1)
                    scored = describe_pair(ra, rb, genes)
                    for raw_w, node, side in [(ra, i, 'left'), (rb, j, 'right')]:
                        raw_top = np.argsort(-raw_w, kind='stable')[:10]
                        ref_top = np.argsort(-w[:,node], kind='stable')[:10]
                        scored[side+'_consensus_top10_recovered'] = len(np.intersect1d(raw_top, ref_top))
                start_stats.append(dict(**key, seed=seed, left_matched=int(sum(aa>=0)), left_total=len(aa),
                                        right_matched=int(sum(bb>=0)), right_total=len(bb), **scored))
        panel = pd.read_csv(root/'results/EAGGL_four_trait_leaf_validation'/trait/'selection/biological_panels.tsv.gz', sep='\t')
        anchors = {r.panel:int(r.factor) for r in panel.query("arm=='Research' and phi==@phi").itertuples()}
        if trait == 'NAFLD':
            leaf_nodes = [i for i in range(len(nodes)) if len(members[i])==1]
            for g in ('CYP4A11','UGT1A9','BAAT','ACOX1'):
                best = np.argmax(w[np.flatnonzero(genes==g).item(),leaf_nodes])
                anchors[g] = members[leaf_nodes[best]][0]
        for floor in (.3, .5, .7):
            ref = sorted(cut_tree(nodes, floor))
            views = [('cosine', floor, ref)]
            for label, metric in [('overlap','min_shared_mass'), ('size','gene_equivalents')]:
                threshold, chosen = choose_cut(nodes, metric, len(ref))
                views.append((label, threshold, chosen))
            if floor == .5:
                views.append(('fixed_overlap', .4, sorted(cut_tree(nodes, .4, 'min_shared_mass'))))
            def within(chosen):
                return {pair for i in chosen for pair in combinations(sorted(members[i]), 2)}
            baseline = within(ref)
            for label, threshold, chosen in views:
                np.testing.assert_array_equal(sorted(k for i in chosen for k in members[i]), sorted(members[-1]))
                sizes = [nodes[i]['gene_equivalents'] for i in chosen]
                pairs = within(chosen)
                external = overlap_vector(w, chosen)
                key = dict(trait=trait, reference_cosine=floor, view=label, threshold=threshold)
                summary.append(dict(**key, families=len(chosen), target=len(ref),
                                    identical_nodes=len(set(chosen)&set(ref)), reference_pairs=len(baseline),
                                    paired_leaves=len(pairs), common_pairs=len(pairs & baseline),
                                    pair_jaccard=len(pairs & baseline)/len(pairs | baseline) if pairs | baseline else 1.,
                                    median_size=np.median(sizes), max_size=max(sizes),
                                    oversize_leaves=sum(v>threshold for v in sizes) if label=='size' else 0,
                                    max_external_overlap=external.max(), external_pairs_over_half=sum(external>=.5)))
                for i, size in zip(chosen, sizes):
                    inventory.append(dict(**key, node=i, gene_equivalents=size,
                                          oversized=(label=='size' and size>threshold),
                                          leaf_ids=','.join(map(str, sorted(members[i]))),
                                          anchors=';'.join(k for k,v in anchors.items() if v in members[i]),
                                          top_genes=','.join(genes[np.argsort(-w[:,i],kind='stable')[:10]])))
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('cases',review), ('case_profiles',pair_stats), ('case_starts',start_stats),
                       ('view_summary',summary), ('families',inventory)]:
        pd.DataFrame(rows).to_csv(output/(name+'.tsv.gz'), sep='\t', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.analysis_root.resolve(), args.output)
