import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from eaggl.phi_requirements import PHIS, align_candidate, select


class RequirementsTest(unittest.TestCase):
    def fixture(self):
        w = np.zeros((12, 2)); w[:6, 0] = .3; w[6:, 1] = .2
        bases = {p: w.copy() for p in PHIS}
        return bases, {p: [w.copy() for _ in range(5)] for p in PHIS}, w @ w.T, {p: True for p in PHIS}

    def test_representative_view_and_input_immutability(self):
        b, r, t, q = self.fixture(); before = t.copy()
        d = select(b, r, t, q)
        self.assertEqual(d['selected_phi'], .01)
        self.assertTrue(d['selection_qualified'])
        np.testing.assert_array_equal(t, before)

    def test_failed_candidate_cannot_certify_partial_comparison(self):
        b, r, t, q = self.fixture(); q[.005] = False
        d = select(b, r, t, q)
        self.assertEqual(d['status'], 'incomplete_numerical_qualification')
        self.assertFalse(d['selection_qualified']); self.assertTrue(d['fallback_qualified'])
        q[.01] = False
        self.assertFalse(select(b, r, t, q)['fallback_qualified'])

    def test_no_supported_profiles_is_explicitly_unresolved(self):
        b, r, t, q = self.fixture()
        r = {p: [np.zeros_like(w) for w in rows] for p, rows in r.items()}
        d = select(b, r, t, q)
        self.assertFalse(d['selection_qualified'])
        self.assertEqual(d['selected_phi'], .01)

    def test_weak_low_contrast_profile_is_not_excluded(self):
        b, r, _, q = self.fixture(); t = np.full((12, 12), .2)
        d = select(b, r, t, q)
        self.assertEqual(d['supported_profiles'], 6)
        self.assertTrue(any(p['contrast_descriptive_flag'] for p in d['profiles']))

    def test_independent_target_and_gene_permutations(self):
        genes = ['c', 'a', 'b']; w = np.array([[3.], [1.], [2.]])
        canonical_t = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        # Target order is b,c,a while W order is c,a,b.
        indices = np.array([2, 0, 1]); ti = [1, 2, 0]
        aw, at = align_candidate(genes, w, canonical_t[np.ix_(ti, ti)], indices, ['a', 'b', 'c'])
        np.testing.assert_array_equal(aw[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(at, canonical_t)
        with self.assertRaises(ValueError): align_candidate(genes, w, canonical_t, [0, 0, 1])

    def test_invalid_scientific_inputs(self):
        b, r, t, q = self.fixture(); t[0, 1] = np.nan
        with self.assertRaises(ValueError): select(b, r, t, q)
        b, r, t, q = self.fixture(); r[.01].pop()
        with self.assertRaises(ValueError): select(b, r, t, q)


if __name__ == '__main__': unittest.main()
