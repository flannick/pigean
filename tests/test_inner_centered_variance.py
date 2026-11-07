"""Centered online moments against independent two-pass calculations."""
import pickle
import numpy as np
import pytest

from pigean.inner_diagnostics import CenteredChainMoments
from pegs_shared.gibbs_features import DEFAULT_FEATURES, PRESETS, resolve_features


def reference(draws):
    x = np.asarray(draws, dtype=np.longdouble)
    # Use one common anchor across chains; longdouble is only float64 on
    # some platforms, so reconstructing absolute means before B is unsafe.
    center = x[0, 0:1]
    shifted = x - center
    delta_mean = shifted.mean(axis=0)
    mean = center + delta_mean
    var = ((shifted - delta_mean)**2).sum(axis=0)/(len(x)-1)
    n = len(x)
    B = n*((delta_mean-delta_mean.mean(axis=0))**2).sum(axis=0)/max(x.shape[1]-1, 1)
    W = var.mean(axis=0)
    R = np.ones(W.shape, dtype=np.longdouble)
    positive = W > 0
    R[positive] = np.sqrt(((n-1)*W[positive]+B[positive])/(n*W[positive]))
    return B, W, R, var.mean(axis=2), mean


@pytest.mark.parametrize('offset,scale', [(0.,1.),(.182,1e-9),(1e8,1e-6)])
def test_online_matches_independent_centered_reference(offset, scale):
    x = offset + np.random.default_rng(73).normal(size=(40,4,3,7))*scale
    moments = CenteredChainMoments(x.shape[1:])
    before = pickle.dumps(np.random.get_state())
    for draw in x:
        moments.update(draw, np.ones(3,dtype=bool))
    assert pickle.dumps(np.random.get_state()) == before
    result = moments.statistics()
    expected = reference(x)
    # Between-chain differences should also be centered before reconstructing
    # large absolute means; the returned means themselves remain float64.
    for a,b in zip(result[:4],expected[:4]):
        np.testing.assert_allclose(a,b,rtol=2e-13,atol=1e-30)
    np.testing.assert_allclose(result[4],expected[4],rtol=8*np.finfo(float).eps,atol=8*np.finfo(float).eps*max(scale,abs(offset)))


def test_masked_slots_only_accumulate_when_active():
    x=np.random.default_rng(7).normal(size=(12,4,3,2))
    moments=CenteredChainMoments(x.shape[1:])
    for i,draw in enumerate(x):
        moments.update(draw,np.array([True,i<5,i%2==0]))
    np.testing.assert_array_equal(moments.count,[12,5,6])
    for slot,indices in enumerate([np.arange(12),np.arange(5),np.arange(0,12,2)]):
        expected=reference(x[indices,:,slot:slot+1,:])
        for k,actual in enumerate(moments.statistics()):
            sliced=actual[:,slot:slot+1] if k in [3,4] else actual[slot:slot+1]
            np.testing.assert_allclose(sliced,expected[k],rtol=2e-14,atol=2e-14)


def test_constant_and_single_sample_preserve_existing_convention():
    moments=CenteredChainMoments((4,2,3))
    moments.update(np.ones((4,2,3)),np.ones(2,dtype=bool))
    for v in moments.statistics()[:4]: np.testing.assert_array_equal(v,np.ones_like(v))
    moments.update(np.ones((4,2,3)),np.ones(2,dtype=bool))
    np.testing.assert_array_equal(moments.statistics()[1],0)
    np.testing.assert_array_equal(moments.statistics()[2],1)


def test_outlier_replacement_copies_history_and_future_updates():
    x=np.random.default_rng(9).normal(size=(12,4,2,3))
    moments=CenteredChainMoments(x.shape[1:])
    for draw in x[:7]: moments.update(draw,np.ones(2,dtype=bool))
    moments.replace_chains(np.array([True,False,True,False]),1,np.array([3,1]))
    expected=x.copy()
    expected[:7,0,1]=x[:7,3,1]
    expected[:7,2,1]=x[:7,1,1]
    for draw in x[7:]: moments.update(draw,np.ones(2,dtype=bool))
    for a,b in zip(moments.statistics(),reference(expected)):
        np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-14)


def test_opt_in_does_not_change_defaults_or_presets():
    feature='inner-centered-variance'
    assert feature not in DEFAULT_FEATURES
    assert all(feature not in x for x in PRESETS.values())
    assert feature in resolve_features(enable=[feature])['active']
    assert feature not in resolve_features(disable=[feature])['active']


def test_native_outlier_replacement_also_replaces_centered_history(monkeypatch):
    import pigean.state as sm
    shape=(4,1,16)
    moments=CenteredChainMoments(shape)
    values=np.broadcast_to(np.arange(4)[:,None,None],shape).astype(float).copy()
    for i in range(11): moments.update(values+i*.01,np.array([True]))
    B,W,R,avg,means=moments.statistics()
    R[:]=np.linspace(1.1,2.8,16);R[0,-1]=1000
    monkeypatch.setattr(moments,'statistics',lambda:(B,W,R,avg,means))
    tensors=[values.copy() for _ in range(5)]
    origins=moments.origin.copy();deltas=moments.mean_delta.copy();m2=moments.m2.copy()
    np.random.seed(77)
    sm._update_inner_beta_rhat_and_outliers(
        tensors[0],tensors[1],11,np.array([True]),False,5,
        tensors[2],tensors[3],tensors[4],np.array([True]),1.01,1,
        centered_moments=moments)
    # The last chain is farthest from the mean (ties resolve to chain zero).
    changed=np.flatnonzero(np.any(tensors[2]!=values,axis=(1,2)))
    assert len(changed)==1
    replaced=int(changed[0]);source=int(tensors[2][replaced,0,0])
    assert source!=replaced
    for actual,previous in [(moments.origin,origins),(moments.mean_delta,deltas),(moments.m2,m2)]:
        np.testing.assert_array_equal(actual[replaced],previous[source])


def test_near_constant_draws():
    # The portable tests above always run; real trace validation is performed
    # separately by the analysis assessment, not by local-path-dependent tests.
    x=np.array([.182 + i*1e-10 for i in range(11)])[:,None,None,None]
    x=np.broadcast_to(x,(11,4,1,1)).copy()
    x[:,1] += 2e-10
    moments=CenteredChainMoments((4,1,1))
    for draw in x: moments.update(draw,np.array([True]))
    np.testing.assert_allclose(moments.statistics()[2],reference(x)[2],rtol=1e-8)


def test_actual_inner_fit_uses_centered_draws_and_resets(monkeypatch):
    import random
    from scipy import sparse
    import pigean.state as state_module
    from pegs_shared.sparse_regression import marginal_backend
    X=sparse.csc_matrix(np.random.default_rng(4).random((30,6))>.5,dtype=float)
    mean=np.asarray(X.mean(axis=0)).ravel()
    scale=np.sqrt(np.asarray(X.power(2).mean(axis=0)).ravel()-mean**2)
    C=(X.toarray()-mean)/scale
    V=C.T@C/len(C)
    draws=[];checks=[]
    original_sparse=state_module._apply_inner_beta_sparsity_update
    original_diag=state_module._update_inner_beta_rhat_and_outliers
    def observe_sparse(**kw):
        result=original_sparse(**kw);draws.append(kw['posterior_beta_mean_t'].copy());return result
    def observe_diag(**kw):
        result=original_diag(**kw)
        if kw['centered_moments'] is not None:
            expected=reference(draws)
            np.testing.assert_allclose(result[0][kw['compute_mask_v']],expected[2][kw['compute_mask_v']],rtol=1e-12,atol=1e-12)
            checks.append(kw['iteration_num'])
        return result
    monkeypatch.setattr(state_module,'_apply_inner_beta_sparsity_update',observe_sparse)
    monkeypatch.setattr(state_module,'_update_inner_beta_rhat_and_outliers',observe_diag)
    outcomes=[]
    for enabled in (False,True,True):
        draws.clear();checks.clear()
        state=state_module.PigeanState(background_prior=.05,batch_size=10)
        state.sigma2,state.sigma_power,state.p=.02,None,.1
        state.scale_factors,state.mean_shifts=scale,mean
        state.is_dense_gene_set=np.zeros(6,dtype=bool)
        state.ps,state.sigma2s=None,None;state.X_orig=X
        np.random.seed(1729);random.seed(1729)
        with marginal_backend('legacy',enable=['inner-centered-variance'] if enabled else []):
            fit=state._calculate_non_inf_betas(None,beta_tildes=np.array([.8,.4,-.1,.2,.7,.3]),ses=np.full(6,.2),V=V,ps=np.full(6,.1),sigma2s=np.full(6,.02),num_chains=4,max_num_iter=24,max_num_burn_in=8,min_num_iter=3,update_hyper_p=False,update_hyper_sigma=False)
        assert bool(checks)==enabled
        outcomes.append((fit,pickle.dumps(np.random.get_state()),random.getstate()))
    for a,b in zip(outcomes[1][0],outcomes[2][0]):np.testing.assert_array_equal(a,b)
    assert outcomes[1][1:]==outcomes[2][1:]
