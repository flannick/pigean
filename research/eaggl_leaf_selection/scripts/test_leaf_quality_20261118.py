from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eaggl_leaf_quality_research.selection import Policy,leaf_metrics,relationship_error,select,restart_support

class UsefulLeafTests(unittest.TestCase):
    def setUp(self):
        self.w=np.zeros((100,2));self.w[:10,0]=.6;self.w[5:15,1]=.4
    def test_distinct_overlapping_leaves_accepted(self):
        _,s=leaf_metrics(self.w,[5,5]);self.assertTrue(s['leaf_usable'])
    def test_uniform_and_singleton_rejected(self):
        w=np.column_stack([np.full(100,.2),np.eye(100)[:,0]])
        _,s=leaf_metrics(w,[5,5]);self.assertEqual(s['uniform'],1);self.assertEqual(s['small'],1);self.assertFalse(s['leaf_usable'])
    def test_long_tiny_tail_does_not_rescue_singleton(self):
        w=np.full((100,1),.01);w[0]=1
        _,s=leaf_metrics(w,[5]);self.assertEqual(s['small'],1)
    def test_duplicate_split_not_rewarded(self):
        w=np.column_stack([self.w[:,0]/2,self.w[:,0]/2,self.w[:,1]])
        _,s=leaf_metrics(w,[5,5,5]);self.assertEqual(s['duplicate_pairs'],1);self.assertFalse(s['leaf_usable'])
    def test_weak_coherent_profile_not_penalized_for_amplitude(self):
        a,_=leaf_metrics(self.w,[5,5]);b,s=leaf_metrics(self.w*.0001,[5,5]);self.assertTrue(s['leaf_usable'])
        for i in range(2):
            self.assertAlmostEqual(a[i]['effective_genes'],b[i]['effective_genes']);self.assertAlmostEqual(a[i]['contrast'],b[i]['contrast'])
    def test_permutations(self):
        rng=np.random.default_rng(213);a,s=leaf_metrics(self.w,[5,5]);b,t=leaf_metrics(self.w[rng.permutation(100),::-1],[5,5]);self.assertEqual(s,t)
        np.testing.assert_allclose([r['effective_genes'] for r in a],[r['effective_genes'] for r in b][::-1])
    def test_support_one_to_one(self):
        w=np.column_stack([self.w[:,0],self.w[:,0]])
        counts=restart_support(w,[self.w[:,:1]]*5);self.assertEqual(counts.sum(),5)
        _,s=leaf_metrics(self.w,None);self.assertFalse(s['leaf_usable'])
    def test_balanced_error_protects_weak_region_without_maximum_gate(self):
        w=np.zeros((100,2));w[:50,0]=.8;w[50:,1]=.08
        t=w@w.T;p=w[:,:1]@w[:,:1].T;d=relationship_error(t,p)
        self.assertLess(d['global_error'],.011);self.assertAlmostEqual(d['balanced_error'],.5)
    def test_zero_target_rows_and_diagonal(self):
        t=self.w@self.w.T;p=t.copy();p[99,98]=.5
        d=relationship_error(t,p);self.assertGreater(d['global_error'],0);self.assertEqual(d['balanced_error'],0);self.assertGreater(d['zero_row_prediction_norm'],0)
        np.fill_diagonal(p,100);self.assertEqual(d,relationship_error(t,p))
    def test_relationship_axes_must_be_aligned(self):
        t=self.w@self.w.T;p=t*.99;q=np.random.default_rng(41).permutation(100)
        a=relationship_error(t,p);b=relationship_error(t[np.ix_(q,q)],p[np.ix_(q,q)])
        self.assertAlmostEqual(a['distortion'],b['distortion'])
        self.assertGreater(relationship_error(t[np.ix_(q,q)],p)['distortion'],.1)
    def test_utility_tolerance_not_perfect_preservation(self):
        rs=[dict(id='fine',phi=.01,factors=100,distortion=.03,numerically_qualified=True,leaf_usable=True),dict(id='coarse',phi=.02,factors=70,distortion=.045,numerically_qualified=True,leaf_usable=True)]
        self.assertEqual(select(rs)['selected'],'coarse');self.assertEqual(select(rs,Policy(epsilon=.01))['selected'],'fine')
        rs[1]['distortion']=.20;self.assertEqual(select(rs)['selected'],'fine')
    def test_no_forced_selection(self):
        self.assertIsNone(select([dict(id='x',phi=.01,numerically_qualified=True,leaf_usable=False)])['selected'])

if __name__=='__main__':unittest.main(verbosity=2)
