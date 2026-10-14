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
