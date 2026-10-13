from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pigean.inner_batching import pairwise_total
from pigean.state import PigeanState
from pegs_shared.sparse_regression import marginal_backend


@pytest.mark.parametrize('size',[1,7,8,9,15,16,127,128,129,257,1001])
def test_reduction_order(size):
    rng=np.random.default_rng(257)
    for _ in range(5):
        a=rng.normal(size=size)*rng.choice([1e-10,1.,1e10],size=size)
        assert pairwise_total(a)==np.sum(a)


@pytest.mark.parametrize('use_sum',[False,True])
@pytest.mark.parametrize('threshold',[0.,.1,.5,1.])
def test_partitions_and_threshold_boundaries(use_sum,threshold):
    rng=np.random.default_rng(43)
    state=PigeanState(background_prior=.05,batch_size=10)
    for g in (1,9,137):
        V=rng.choice([0.,.01,.1,.5,1.],size=(g,g))
        np.fill_diagonal(V,1.)
        with marginal_backend('legacy'):
            expected=state._compute_gene_set_batches(V,max_allowed_batch_correlation=threshold,use_sum=use_sum)
        with marginal_backend('legacy',enable=['inner-batching']):
            observed=state._compute_gene_set_batches(V,max_allowed_batch_correlation=threshold,use_sum=use_sum)
        np.testing.assert_array_equal(expected,observed)


def test_nondefault_sort_falls_back():
    state=PigeanState(background_prior=.05,batch_size=10)
    V=np.eye(6); order=np.arange(6)[::-1]
    with marginal_backend('legacy'):
        a=state._compute_gene_set_batches(V,sort_values=order,stop_at=3)
    with marginal_backend('legacy',enable=['inner-batching']):
        b=state._compute_gene_set_batches(V,sort_values=order,stop_at=3)
    np.testing.assert_array_equal(a,b)
