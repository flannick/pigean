import sys
import tempfile
import unittest
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from eaggl import factor_runtime as fr
from eaggl.phi_requirements_runtime import run


class State(SimpleNamespace):
    def _record_params(self, values, overwrite=False): self.params.update(values)


class RuntimeTest(unittest.TestCase):
    def exercise(self, failed=None, bad_target=False, report=None, **search):
        self.calls=[]; self.w=np.zeros((8,2)); self.w[:4,0]=.3; self.w[4:,1]=.2
        state=State(genes=list('abcdefgh'), params={})
        def fit(s, seed, run_index, factor_kwargs):
            phi=factor_kwargs['phi'];self.calls.append((phi,seed))
            s.exp_gene_factors=self.w.copy();s._phi_selection_target_matrix=self.w@self.w.T
            if bad_target and phi==.02:s._phi_selection_target_matrix*=2
            s._phi_selection_target_gene_indices=np.arange(8)
            s.internal_projection_diagnostics=[dict(converged=True)]
            return dict(seed=seed,converged=phi!=failed,hit_iteration_cap=phi==failed)
        def consensus(s, runs, summaries, **kwargs):
            return runs[0], dict(mode='consensus',reference_run_index=0,run_summaries=summaries,factor_support=[])
        with patch.object(fr,'_run_factor_with_seed',side_effect=fit),patch.object(fr,'_apply_consensus_solution',side_effect=consensus),patch.object(fr,'_write_final_factor_gmt'):
            run(state,seed=202,factor_kwargs=dict(log_fn=lambda *a:None,warn_fn=lambda *a:None,info_level=0),consensus_stats_out=None,report_out=report,consensus_min_run_support=.5,consensus_min_factor_cosine=.7, **search)
        return state

    def test_state_wrapper_forwards_explicit_extension_controls(self):
        from eaggl.state import EagglState
        def spy(state, phi_search_extension, phi_search_max_extra):
            return phi_search_extension, phi_search_max_extra
        with patch.object(fr,'run_factor',new=spy):
            self.assertEqual(EagglState.__new__(EagglState).run_factor(
                phi_search_extension='none',phi_search_max_extra=0),('none',0))

    def test_fifteen_restarts_and_no_fourth_fit(self):
        s=self.exercise();self.assertEqual(len(self.calls),15)
        self.assertEqual([x[1] for x in self.calls[:5]],[x[1] for x in self.calls[5:10]])
        self.assertEqual(s.params['phi'],.01);self.assertTrue(s.params['phi_selection_qualified'])
        np.testing.assert_array_equal(s.exp_gene_factors,self.w)

    def test_expansion_fits_each_candidate_once_and_reuses_selected_fit(self):
        s=self.exercise(search_extension='noninferior',search_max_extra=2)
        self.assertEqual(len(self.calls),25)
        self.assertEqual([p for p,_ in self.calls[::5]],[.01,.005,.02,.0025,.04])
        self.assertEqual(s.params['phi'],.01)
        self.assertEqual(s.params['phi_search_stop_reason'],'extra_budget_exhausted')
        np.testing.assert_array_equal(s.exp_gene_factors,self.w)
        for i in range(0,25,5):self.assertEqual([x[1] for x in self.calls[:5]],[x[1] for x in self.calls[i:i+5]])

    def test_none_and_zero_extra_keep_three_candidates(self):
        for controls in [dict(search_extension='none'),dict(search_extension='noninferior',search_max_extra=0)]:
            self.exercise(**controls);self.assertEqual(len(self.calls),15)

    def test_failed_extra_does_not_certify_partial_search(self):
        s=self.exercise(failed=.0025,search_extension='noninferior')
        self.assertEqual(len(self.calls),20)
        self.assertFalse(s.params['phi_selection_qualified'])
        self.assertEqual(s.params['phi_search_stop_reason'],'numerical_uncertainty')

    def test_failure_reports_all_candidates_and_uses_qualified_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'selection.tsv.gz';s=self.exercise(failed=.005,report=p)
            self.assertFalse(s.params['phi_selection_qualified'])
            with gzip.open(str(p)+'.details.json.gz','rt') as f:report=json.load(f)
            self.assertEqual(len(report['candidate_diagnostics']),3)
            self.assertFalse(report['candidate_diagnostics']['0.005']['qualified'])

    def test_unqualified_baseline_stops_after_report(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'selection.tsv.gz'
            with self.assertRaisesRegex(ValueError,'no numerically qualified'):self.exercise(failed=.01,report=p)
            self.assertTrue(Path(str(p)+'.details.json.gz').exists())

    def test_target_change_stops_comparison(self):
        with self.assertRaisesRegex(ValueError,'changed the gene-pair target'):self.exercise(bad_target=True)


if __name__=='__main__':unittest.main()
