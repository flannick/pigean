from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from reliability.quadratic import QuadraticTarget,enumerate_posterior
from reliability.observed_logistic import ObservedLogisticTarget
from reliability.checks import (gaussian_scan_stationary,support_transition,transition_metrics,
    finite_outer_reference,finite_outer_metrics,mean_feedback_metrics,production_huge_conditionals,
    conditional_cycle_error,production_draws)


def test_analytic_gaussian_support_integral_matches_independent_quadrature():
    t=QuadraticTarget(np.array([[2.]]),np.array([1.3]),np.array([.15]),np.array([.7]))
    r=enumerate_posterior(t)
    f=lambda b:np.exp(1.3*b-b*b)*norm.pdf(b,scale=np.sqrt(.7))
    mass=quad(f,-np.inf,np.inf,epsabs=1e-11)[0]
    first=quad(lambda b:b*f(b),-np.inf,np.inf,epsabs=1e-11)[0]
    Z=.85+.15*mass
    np.testing.assert_allclose(r['weights'],[.85/Z,.15*mass/Z],rtol=1e-10)
    np.testing.assert_allclose(r['mean'],[.15*first/Z],rtol=1e-10)


def test_logistic_reference_agrees_with_adaptive_quadrature_and_refinement():
    t=ObservedLogisticTarget(np.array([[-1.],[0.],[1.],[2.]]),np.array([0,1,0,1]),
        np.full(4,-.7),np.array([.2]),np.array([.8]))
    a=t.quadrature(64);b=t.quadrature(96)
    f=lambda x:np.exp(t.log_likelihood(np.array([x])))*norm.pdf(x,scale=np.sqrt(.8))
    mass=quad(f,-np.inf,np.inf,epsabs=1e-12)[0];zero=np.exp(t.log_likelihood(np.zeros(1)))
    first=quad(lambda x:x*f(x),-np.inf,np.inf,epsabs=1e-12)[0];Z=.8*zero+.2*mass
    np.testing.assert_allclose(a['weights'],b['weights'],atol=1e-9)
    np.testing.assert_allclose(b['weights'],[.8*zero/Z,.2*mass/Z],atol=1e-9)
    np.testing.assert_allclose(b['mean'],[.2*first/Z],atol=1e-9)


def test_gaussian_scan_compatible_law_matches_joint_and_order():
    V=np.array([[1.,.4],[.4,1.]]);se=np.ones(2)*.5;tau=np.ones(2)
    a=gaussian_scan_stationary(np.array([.3,.4]),se,V,tau,[0,1])
    b=gaussian_scan_stationary(np.array([.3,.4]),se,V,tau,[1,0])
    cov=np.linalg.inv(V/.25+np.eye(2))
    np.testing.assert_allclose(a['covariance'],cov,atol=1e-14)
    np.testing.assert_allclose(b['covariance'],cov,atol=1e-14)
    np.testing.assert_allclose(a['mean'],cov@np.array([.3,.4])/.25,atol=1e-14)


def test_incompatible_gaussian_scans_have_distinct_stationary_laws():
    args=(np.array([.3,.4]),np.array([.2,.8]),np.array([[1.,.8],[.8,1.]]),np.ones(2))
    a=gaussian_scan_stationary(*args,[0,1]);b=gaussian_scan_stationary(*args,[1,0])
    assert a['invariance_error']<1e-12 and b['invariance_error']<1e-12
    assert np.max(np.abs(a['covariance']-b['covariance']))>.05


def test_collapsed_support_transitions_preserve_exact_target():
    t=QuadraticTarget(np.array([[25.,24.975],[24.975,25.]]),np.array([37.5,37.5]),np.full(2,.01),np.ones(2))
    w=enumerate_posterior(t)['weights']
    for swap in (False,True):
        T=support_transition(w,swap=swap)
        np.testing.assert_allclose(T.sum(axis=1),1,atol=1e-14)
        assert np.all(T>=0)
        assert transition_metrics(T,w)['stationarity_error']<1e-14
    assert transition_metrics(support_transition(w,swap=True),w)['spectral_gap']>transition_metrics(support_transition(w),w)['spectral_gap']


def outer():
    return finite_outer_reference(np.array([[1.,.2],[.3,1.]]),np.full(2,np.log(.05/.95)),np.array([2.,3.]),levels=(-2.,0.,2.))


def test_exact_outer_gibbs_recovers_enumerated_joint():
    r=outer();m=finite_outer_metrics(r,r['D_given_beta'])
    assert m['cycle_error']<1e-12 and m['joint_tv_from_fixed_evidence']<1e-12


def test_mean_feedback_is_not_exact_outer_gibbs():
    m=mean_feedback_metrics(outer(),np.array([2.,3.]))
    assert m['stationarity_error']<1e-12
    assert m['joint_tv_from_fixed_evidence']>.001


def test_actual_huge_invariants_and_outer_conditional_compatibility():
    r=outer();dgb,checks=production_huge_conditionals(r)
    assert checks['signal_probabilities_unchanged']
    for k in ('strongest_signal_error','duplicate_signal_error','order_error','probability_formula_error'):
        assert checks[k]<1e-12
    assert min(checks['log_bf_range_across_priors'])>.01
    assert conditional_cycle_error(r['beta_given_D'],dgb)>.01


def test_production_adapter_shapes_and_seed_reproducibility():
    args=(np.array([.3,.4]),np.ones(2),np.eye(2),np.full(2,.2),np.ones(2))
    a,m=production_draws(*args,sweeps=12,burn=2,chains=4)
    b,_=production_draws(*args,sweeps=12,burn=2,chains=4)
    assert a.shape==(10,4,2);np.testing.assert_array_equal(a,b)
