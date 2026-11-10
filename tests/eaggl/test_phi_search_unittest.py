import unittest
from eaggl.phi_search import next_step


def decision(errors, accepted=None, qualified=True):
    if accepted is None: accepted=list(errors)
    return dict(all_candidates_qualified=qualified, selection_qualified=bool(accepted) and qualified,
        views=[dict(phi=p,admissible=p in accepted,global_error=e) for p,e in errors.items()],
        errors=[dict(candidate_phi=p,source_phi=.01,source_factor=1,other_phi=.01,
                     other_factor=1,evaluable=True,error=e) for p,e in errors.items()])


class PhiSearchTest(unittest.TestCase):
    def test_flat_or_small_gain_does_not_expand(self):
        for errors in [{.005:1.,.01:1.,.02:1.},{.005:.81,.01:1.,.02:1.}]:
            self.assertIsNone(next_step(decision(errors))['next_phi'])
        # An inadmissible interior caused only by a different gate is insufficient.
        self.assertIsNone(next_step(decision({.005:.9,.01:1.,.02:1.},[.005]))['next_phi'])

    def test_material_gain_in_each_direction(self):
        for errors,good,expected in [({.005:.5,.01:1.,.02:1.},.005,.0025),
                                    ({.005:1.,.01:1.,.02:.5},.02,.04)]:
            d=next_step(decision(errors,[good]));self.assertEqual(d['next_phi'],expected)
            self.assertEqual(d['reason'],'meaningful_improvement')

    def test_both_edges_share_budget_and_shallower_edge_runs_next(self):
        d=decision({.005:1.,.01:1.,.02:1.})
        self.assertEqual(next_step(d,'noninferior')['next_phi'],.0025)
        d=decision({.0025:1.,.005:1.,.01:1.,.02:1.})
        self.assertEqual(next_step(d,'noninferior')['next_phi'],.04)
        d=decision({.0025:1.,.005:1.,.01:1.,.02:1.,.04:1.})
        step=next_step(d,'noninferior');self.assertIsNone(step['next_phi'])
        self.assertEqual(step['reason'],'extra_budget_exhausted')

    def test_new_edge_must_beat_best_interior_not_original_center(self):
        d=decision({.0025:.45,.005:.5,.01:1.,.02:1.},[.0025,.005])
        self.assertIsNone(next_step(d)['next_phi'])
        d=decision({.00125:.1,.0025:.5,.005:1.,.01:1.5,.02:2.},[.00125])
        self.assertEqual(next_step(d)['reason'],'extra_budget_exhausted')
        self.assertEqual(next_step(d,max_extra=3)['next_phi'],.000625)

    def test_rejected_edge_and_numerical_failure_stop(self):
        d=decision({.0025:1.,.005:.5,.01:1.,.02:1.},[.005])
        self.assertIsNone(next_step(d)['next_phi'])
        d=decision({.005:.5,.01:1.,.02:1.},[.005],qualified=False)
        self.assertEqual(next_step(d)['reason'],'numerical_uncertainty')

    def test_none_zero_budget_and_invalid_settings(self):
        d=decision({.005:.5,.01:1.,.02:1.},[.005])
        self.assertEqual(next_step(d,'none')['reason'],'extension_disabled')
        self.assertEqual(next_step(d,max_extra=0)['reason'],'extra_budget_exhausted')
        for args in [('unknown',2),('improvement',-1),('improvement',1.5)]:
            with self.assertRaises(ValueError):next_step(d,*args)

    def test_global_gain_is_sufficient_without_local_gain(self):
        d=decision({.005:.5,.01:1.,.02:1.},[.005])
        for row in d['errors']:row['error']=1.
        self.assertEqual(next_step(d)['next_phi'],.0025)

    def test_threshold_is_strict_and_unresolved_stays_unresolved(self):
        d=decision({.005:.8,.01:1.,.02:1.},[.005])
        self.assertIsNone(next_step(d)['next_phi'])
        d=decision({.005:.5,.01:1.,.02:1.},[])
        self.assertEqual(next_step(d)['reason'],'unresolved_comparison')
