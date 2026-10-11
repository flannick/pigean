"""Preserve interpretable extreme tails without changing regression estimates."""
import csv,gzip,tempfile
from pathlib import Path
import numpy as np
from scipy import sparse,special
from eaggl.snp_factor.statistics import mixture_sf,mixture_logsf,unsigned_associations
from eaggl.snp_factor.runner import Results

def test_analytic_extreme_chi_square():
    for df in [1,2,4,10]:
        q=10000.;value,method=mixture_logsf(q,np.ones(df))
        if df==1:expected=np.log(2)+special.log_ndtr(-np.sqrt(q))
        else:
            j=np.arange(df//2)
            expected=-q/2+special.logsumexp(j*np.log(q/2)-special.gammaln(j+1))
        np.testing.assert_allclose(value,expected,atol=1e-9,rtol=0)
        assert method=='analytic'
        assert mixture_sf(q,np.ones(df))[0]==0

def test_unequal_extreme_tail_against_high_precision_lr():
    # 60-digit independent evaluation of the LR formula for q=10000,
    # eigenvalues [0.1,0.3,0.8,1.0] (not an exact-mixture reference).
    e=np.array([.1,.3,.8,1.])
    moderate=[mixture_sf(q,e)[0] for q in [3,5,10,20,40]]
    assert all(a>b>0 for a,b in zip(moderate,moderate[1:]))
    logp,method=mixture_logsf(10000,e)
    assert method=='saddlepoint' and np.isfinite(logp)
    np.testing.assert_allclose(logp,-5003.6485747936130155,atol=1e-9,rtol=0)
    # Multiplying Q and the spectrum by the same scale leaves the test unchanged.
    np.testing.assert_allclose(mixture_logsf(10000*.003,e*.003)[0],logp,atol=1e-9,rtol=0)
    for q in [1420.,1450.]:
        subnormal,method=mixture_sf(q,e)
        assert method=='saddlepoint' and 0<subnormal<np.finfo(float).tiny

def test_unsigned_underflow_is_available_with_log_p():
    A=sparse.csr_matrix([[.1],[.3],[.8],[1.]])
    def blocks():
        for i in range(4):yield np.array([i]),np.ones((1,1)),np.ones((1,1)),True
    rows=unsigned_associations(A,np.full((4,1),10000.),blocks,['a'])
    row=next(r for r in rows if r['method']=='unsigned_quadratic')
    assert row['status']=='ok' and row['p']==0 and row['p_underflow']
    assert np.isfinite(row['neg_log10_p']) and row['neg_log10_p']>1000

def test_multiple_testing_distinguishes_underflowed_probabilities():
    with tempfile.TemporaryDirectory() as d:
        store=Results(':memory:');path=Path(d)/'out.gz'
        for i,logp in enumerate([-900.,-1100.,-1000.]):
            store.add(dict(trait=str(i),method='unsigned_quadratic',scope='quadratic',p=0.,log_p=logp))
        store.add(dict(trait='unavailable',method='unsigned_quadratic',scope='quadratic',p=None,log_p=-2000.))
        store.write(path);store.close()
        with gzip.open(path,'rt') as f:rows=list(csv.DictReader(f,delimiter='\t'))
        np.testing.assert_allclose([float(r['log_q']) for r in rows[:3]],[-900+np.log(4/3),-1100+np.log(4),-1000+np.log(2)])
        assert all(r['q_underflow']=='True' for r in rows[:3])
        assert rows[3]['q']=='NA' and rows[3]['log_q']=='NA'


def test_davies_reference_moderate_tails():
    # CompQuadForm AS 155, qfc.cpp: independent CDF inversion, acc=1e-9,
    # lim=1e6, all fault codes zero. Approximation tolerance is deliberate.
    e=np.linspace(.3,1.2,20)
    for q,reference in [(14.437643686855852,.4967051495298539),
                        (24.15199125738422,.05067562720997576),
                        (35.50246079305041,.0012352963623998292)]:
        assert abs(mixture_sf(q,e)[0]-reference)<.0005
