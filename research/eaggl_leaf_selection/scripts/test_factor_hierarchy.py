from itertools import combinations
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eaggl_leaf_quality_research.hierarchy import RULE_FIELDS, build_tree, cut_tree


class HierarchyTest(unittest.TestCase):
    def test_known_tree_and_same_leaf_mass(self):
        w=np.array([[.4,.8,0],[.25,.5,0],[0,0,.6],[0,0,.3]])
        nodes,loading,members=build_tree(w,[5,7,2])
        group=next(i for i,m in enumerate(members) if set(m)=={5,7})
        for floor in (.3,.5,.7):
            self.assertEqual({tuple(sorted(members[i])) for i in cut_tree(nodes,floor)}, {(5,7),(2,)})
        self.assertAlmostEqual(nodes[group]['min_shared_mass'],1.)
        self.assertAlmostEqual(nodes[-1]['min_shared_mass'],0.)
        for i,ids in enumerate(members):
            cols=w[:,[[5,7,2].index(k) for k in ids]]
            np.testing.assert_allclose(loading[:,i],1-np.prod(1-cols,axis=1))
            pairs=1-np.prod(1-cols[:,None,:]*cols[None,:,:],axis=2)
            overall=np.outer(loading[:,i],loading[:,i]);np.fill_diagonal(overall,0);np.fill_diagonal(pairs,0)
            self.assertAlmostEqual(nodes[i]['shared_leaf_pair_fraction'],pairs.sum()/overall.sum())
        self.assertEqual(cut_tree(nodes,0),[len(nodes)-1])
        self.assertEqual(members,build_tree(w[::-1,::-1],[2,7,5])[2])

    def test_overlap_and_cut_coverage(self):
        w=np.array([[.6,.4,0],[.4,.3,.3],[0,.3,.7]])
        nodes,_,members=build_tree(w,[1,2,3])
        i=next(i for i,m in enumerate(members) if set(m)=={1,2})
        self.assertAlmostEqual(nodes[i]['min_shared_mass'],.7)
        for metric in ['min_leaf_cosine','min_shared_mass']:
            for floor in (.3,.5,.7):
                chosen=cut_tree(nodes,floor,metric)
                self.assertEqual(sorted(k for i in chosen for k in members[i]),[1,2,3])
                self.assertTrue(all(nodes[i][metric]>=floor for i in chosen))

    def test_rule_controls_every_merge_and_cut(self):
        w = np.random.default_rng(4).uniform(.001, 1, size=(11, 9))
        trees = set()
        for rule, field in RULE_FIELDS.items():
            if rule == 'cosine':
                z = w / np.linalg.norm(w, axis=0)
                score = z.T @ z
            else:
                z = w if rule == 'shared-mass' else w**2
                z = z / z.sum(0)
                score = np.minimum(z[:,:,None], z[:,None,:]).sum(0)
            nodes, _, groups = build_tree(w, range(w.shape[1]), rule)
            active = set(range(w.shape[1]))
            for i in range(w.shape[1], len(nodes)):
                node = nodes[i]
                best = max(score[np.ix_(groups[a], groups[b])].min() for a,b in combinations(active,2))
                self.assertAlmostEqual(node[field], best)
                self.assertAlmostEqual(score[np.ix_(groups[i], groups[i])].min(), best)
                active.difference_update([node['left'], node['right']])
                active.add(i)
            trees.add(frozenset(frozenset(g) for g in groups))
            for floor in (.3, .5, .7):
                chosen = cut_tree(nodes, floor, field)
                self.assertEqual(sorted(k for i in chosen for k in groups[i]), list(range(w.shape[1])))
                for i in chosen:
                    self.assertGreaterEqual(nodes[i][field], floor)
                    parent = nodes[i]['parent']
                    self.assertTrue(parent is None or nodes[parent][field] < floor)
        self.assertGreater(len(trees), 1)

    def test_size_ceiling(self):
        nodes, values, members = build_tree(np.array([[.4,.8,0],[.25,.5,0],[0,0,.6],[0,0,.3]]), [5,7,2])
        for cap in (.5, 1., 2., 3.):
            frontier = cut_tree(nodes, cap, 'gene_equivalents')
            self.assertEqual(sorted(k for i in frontier for k in members[i]), [2,5,7])
            for i in frontier:
                self.assertAlmostEqual(nodes[i]['gene_equivalents'], values[:, i].sum())
                self.assertTrue(values[:, i].sum() <= cap or len(members[i]) == 1)
                parent = nodes[i]['parent']
                self.assertTrue(parent is None or values[:, parent].sum() > cap)
        for cap in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                cut_tree(nodes, cap, 'gene_equivalents')

    def test_edge_cases(self):
        self.assertEqual(cut_tree(build_tree(np.empty((3,0)),[])[0],.5),[])
        n,_,_=build_tree(np.array([[1.],[0.]]),[4])
        self.assertEqual(cut_tree(n,.5),[0])
        self.assertTrue(np.isnan(n[0]['shared_leaf_pair_fraction']))
        for w in [np.array([[1.2]]),np.array([[0.]]),np.array([[np.nan]])]:
            with self.assertRaises(ValueError):build_tree(w,[1])


if __name__=='__main__':
    unittest.main()
