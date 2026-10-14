from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import log_ndtr
from scipy.stats import norm
from logistic_sampling.pg import draw_pg, _log_phi
from logistic_sampling.model import LogisticTarget
from logistic_sampling.sampler import sample


def test_pg_tail_calculation():
    for x in (-500.,-40.,-12.,-10.,-9.,0.,10.):
        assert abs(_log_phi(x)-log_ndtr(x))<1e-9


def test_pg_distribution_moments_and_laplace_transform():
    rng=np.random.default_rng(701)
    for eta in (0.,1.,-4.,20.,100.):
        a=draw_pg(np.full(80000,eta),rng)
        z=abs(eta)
        mean=.25 if z==0 else np.tanh(z/2)/(2*z)
        var=1/24 if z==0 else (np.tanh(z/2)-z/(2*np.cosh(z/2)**2))/(2*z**3)
        assert np.all(a>0)
        assert abs(a.mean()-mean)<5*np.sqrt(var/len(a))
        assert abs(a.var()-var)<.035*var
        for t in (.5,3.):
            expected=np.cosh(z/2)/np.cosh(np.sqrt(z*z+2*t)/2)
            values=np.exp(-t*a)
            assert abs(values.mean()-expected)<5*values.std()/np.sqrt(len(a))+1e-8


def target():
    return LogisticTarget(np.array([[-1.],[0.],[1.],[2.]]),np.array([0.,1.,0.,1.]),np.full(4,-.7),np.array([.2]),np.array([.8]))


def test_reference_independent_integral():
    t=target();r=t.quadrature(96)
    f=lambda b:np.exp(t.log_likelihood(np.array([b])))*norm.pdf(b,scale=np.sqrt(.8))
    m=quad(f,-np.inf,np.inf,epsabs=1e-12)[0];f1=quad(lambda b:b*f(b),-np.inf,np.inf,epsabs=1e-12)[0]
    z=.8*np.exp(t.log_likelihood(np.zeros(1)))+.2*m
    np.testing.assert_allclose(r['inclusion'],[.2*m/z],atol=1e-9)
    np.testing.assert_allclose(r['mean'],[.2*f1/z],atol=1e-9)


def test_coordinate_sampler_matches_nonaugmented_reference():
    t=target();ref=t.quadrature(96);d=sample(t,sweeps=6000,burn=1000)
    assert np.max(abs(d.mean(axis=(0,1))-ref['mean']))<.02
    assert np.max(abs((d!=0).mean(axis=(0,1))-ref['inclusion']))<.02
    a=sample(t,sweeps=12,burn=2);b=sample(t,sweeps=12,burn=2)
    np.testing.assert_array_equal(a,b)


def test_invalid_inputs_fail():
    with pytest.raises(ValueError): draw_pg(np.array([np.nan]),np.random.default_rng(0))
    with pytest.raises(ValueError): sample(target(),sweeps=1,burn=1)


def test_augmented_support_integral_matches_independent_quadrature():
    from logistic_sampling.collapse import support_logmass
    Q=np.array([[2.]]);h=np.array([1.3]);pi=np.array([.2]);tau=np.array([.8])
    mass=quad(lambda x:np.exp(1.3*x-x*x)*norm.pdf(x,scale=np.sqrt(.8)),-np.inf,np.inf)[0]
    score=support_logmass(Q,h,pi,tau,np.array([True]))
    np.testing.assert_allclose(np.exp(score),.2*mass,rtol=1e-10)
    np.testing.assert_allclose(np.exp(support_logmass(Q,h,pi,tau,np.array([False]))),.8)


def test_collapsed_sampler_matches_logistic_reference():
    t=LogisticTarget(np.array([[-1.,-.3],[-.7,-.9],[0.,.2],[.2,.7],[.8,.4],[1.,1.]]),
                     np.array([0,1,0,0,1,1]),np.full(6,-.7),np.full(2,.2),np.full(2,.8))
    ref=t.quadrature(96);d=sample(t,sweeps=7000,burn=1000,collapse=True)
    assert np.max(abs((d!=0).mean(axis=(0,1))-ref['inclusion']))<.02
    assert np.max(abs(d.mean(axis=(0,1))-ref['mean']))<.03


def test_swap_detailed_balance_for_unequal_prior_and_variance():
    from logistic_sampling.collapse import support_logmass
    Q=np.array([[2.,.7],[.7,3.]])
    h=np.array([1.,.4]);pi=np.array([.1,.3]);tau=np.array([.7,1.2])
    a=support_logmass(Q,h,pi,tau,np.array([True,False]))
    b=support_logmass(Q,h,pi,tau,np.array([False,True]))
    np.testing.assert_allclose(np.exp(a)*.5*min(1.,np.exp(b-a)),np.exp(b)*.5*min(1.,np.exp(a-b)),rtol=1e-14)


@pytest.mark.parametrize('collapse',[False,True])
def test_optional_swaps_match_original_logistic_target(collapse):
    t=LogisticTarget(np.array([[-1.,-.3],[-.7,-.9],[0.,.2],[.2,.7],[.8,.4],[1.,1.]]),
                     np.array([0,1,0,0,1,1]),np.full(6,-.7),np.array([.1,.3]),np.array([.7,1.2]))
    ref=t.quadrature(96);d=sample(t,sweeps=7000,burn=1000,collapse=collapse,swaps=1)
    assert np.max(abs((d!=0).mean(axis=(0,1))-ref['inclusion']))<.02
    assert np.max(abs(d.mean(axis=(0,1))-ref['mean']))<.03


def test_duplicate_adaptive_oracle_matches_tensor_integral():
    X=np.tile(np.array([-1.,0.,1.,2.])[:,None],(1,2))
    t=LogisticTarget(X,np.array([0,1,0,1]),np.full(4,-.7),np.array([.1,.3]),np.array([.7,1.2]))
    a=t.quadrature(128);b=t.duplicate_reference()
    for key in ('weights','mean','covariance'):
        np.testing.assert_allclose(a[key],b[key],atol=1e-9)


def test_implemented_swap_transition_matches_symmetric_mh_kernel():
    from itertools import product
    from scipy.special import softmax
    from logistic_sampling.collapse import support_logmass
    from logistic_sampling.swaps import support_swaps
    Q=np.array([[2.,.7,.2],[.7,3.,.1],[.2,.1,1.]])
    h=np.array([1.,.4,-.2]);pi=np.array([.1,.3,.6]);tau=np.array([.7,1.2,.4])
    states=np.array(list(product((False,True),repeat=3)))
    logs=np.array([support_logmass(Q,h,pi,tau,s) for s in states]);weights=softmax(logs)
    T=np.zeros((8,8));rng=np.random.default_rng(30);trials=16000
    for a,s in enumerate(states):
        expected=np.zeros(8)
        for j in range(3):
            for k in range(j+1,3):
                if s[j]==s[k]:continue
                t=s.copy();t[j]=not t[j];t[k]=not t[k]
                b=int(t@np.array([4,2,1]))
                expected[b]=(.5/3)*min(1.,np.exp(logs[b]-logs[a]))
        expected[a]=1-expected.sum()
        for _ in range(trials):
            t=support_swaps(Q,h,pi,tau,s,rng,1)
            T[a,int(t@np.array([4,2,1]))]+=1/trials
        np.testing.assert_allclose(T[a],expected,atol=.012)
    np.testing.assert_allclose(weights@T,weights,atol=.003)


def test_outer_adapter_restores_production_functions_after_error():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    import pigean.state as sm
    from logistic_sampling.outer_adapter import outer_backend
    a=sm._prepare_gibbs_iteration_inputs;b=sm._run_gibbs_corrected_betas_step
    with pytest.raises(RuntimeError):
        with outer_backend():
            assert sm._prepare_gibbs_iteration_inputs is not a
            raise RuntimeError('intentional')
    assert sm._prepare_gibbs_iteration_inputs is a
    assert sm._run_gibbs_corrected_betas_step is b


def test_adapter_uses_production_slab_scaling_and_coefficient_units():
    from types import SimpleNamespace
    from logistic_sampling.outer_adapter import conditional_prior
    s=SimpleNamespace(sigma_power=-2.,sigma_threshold_k=None,sigma_threshold_xo=None,sigma2=.01,
        get_scaled_sigma2=lambda scale,var,power,k,xo:var*scale**power)
    it=dict(full_scale_factors_m=np.array([[.2,.4]]),full_ps_m=np.array([[.1,.2]]),
        full_sigma2s_m=np.array([[.01,.02]]),full_is_dense_gene_set_m=np.array([[False,False]]))
    pi,tau=conditional_prior(s,it,0)
    np.testing.assert_allclose(pi,[.1,.2]);np.testing.assert_allclose(tau,[2.5,.625])
