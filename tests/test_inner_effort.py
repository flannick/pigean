"""Experimental effort/mean switches affect corrected fits only and compose."""
import inspect
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pegs_shared.sparse_regression import marginal_backend
from pigean.state import _run_gibbs_corrected_beta_sampler


class Recorder:
    gene_sets=['a']
    def _calculate_non_inf_betas(self,**kwargs):
        return kwargs


def test_independent_translation():
    import numpy as np
    base={k:None for k in inspect.signature(_run_gibbs_corrected_beta_sampler).parameters}
    base.update(state=Recorder(),min_num_iter_betas=10,gauss_seidel_betas=False,
                gene_set_mask_m=np.array([[True]]))
    for features,expected_min,expected_mean in (([],10,False),(['inner-early-check'],5,False), (['inner-mean'],10,True), (['inner-early-check','inner-mean'],5,True)):
        with marginal_backend('legacy',enable=features):
            result=_run_gibbs_corrected_beta_sampler(**base)
        assert result['min_num_iter']==expected_min
        assert result['gauss_seidel']==expected_mean
        assert result['update_hyper_sigma'] is False
        assert result['update_hyper_p'] is False
