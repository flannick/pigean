import unittest
import numpy as np
from eaggl.pooled_consensus import collect_pooled_matches


class PooledConsensusTest(unittest.TestCase):
    def groups(self, matrices, ids, **kwargs):
        matches, stats = collect_pooled_matches(matrices, ids, **kwargs)
        used = [(r, c) for group in matches.values() for r, c, _ in group]
        self.assertEqual(len(used), len(set(used)))
        for group in matches.values():
            self.assertEqual(len(group), len({r for r, _, _ in group}))
        return matches, stats

    def test_support_can_recover_factor_absent_from_reference(self):
        basis = np.eye(3)
        matches, _ = self.groups([basis[:, :2], basis, basis[:, :2], basis, basis], list('abcde'))
        self.assertEqual(sorted(map(len, matches.values())), [3, 5, 5])

    def test_final_consensus_can_exceed_per_restart_budget(self):
        basis = np.eye(31)
        matches, _ = self.groups([np.delete(basis, i, axis=1) for i in range(5)], list('abcde'))
        self.assertEqual(len(matches), 31)

    def test_chaining_does_not_satisfy_all_pairs_support(self):
        a = np.array([1., 0., 0.]); b = np.array([np.sqrt(.75), .5, 0.]); c = np.array([.5, np.sqrt(.75), 0.])
        common = np.array([0., 0., 1.])
        matches, _ = self.groups([np.column_stack([x, common]) for x in [a, b, c]], list('abc'))
        self.assertEqual(len(matches), 1)
        self.assertTrue(all(column == 1 for _, column, _ in matches[0]))

    def test_order_invariance_and_empty_inputs(self):
        basis = np.eye(4); matrices = [basis.copy() for _ in range(5)]; ids = list('abcde')
        def signatures(ms, names):
            groups, _ = self.groups(ms, names)
            return sorted(tuple(sorted((names[r], tuple(ms[r][:, c])) for r, c, _ in g)) for g in groups.values())
        expected = signatures(matrices, ids)
        self.assertEqual(expected, signatures([m[:, ::-1] for m in matrices[::-1]], ids[::-1]))
        matches, _ = self.groups([np.zeros((4, 0)), np.zeros((4, 1))], list('ab'))
        self.assertEqual(matches, {})
        matches, _ = self.groups([basis], ['a'], min_runs=3)
        self.assertEqual(matches, {})

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.groups([np.eye(2), np.eye(3)], ['a','b'])
        with self.assertRaises(ValueError):
            self.groups([np.array([[np.nan]])], ['a'])
