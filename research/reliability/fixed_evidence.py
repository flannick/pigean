"""Fixed-evidence logistic gene model with exact indicator marginalization.

Conditional on beta, D_i ~ Bernoulli(sigmoid(offset_i + X_i beta)); the observed
evidence has likelihood ratio exp(log_bf_i) for D_i=1 versus D_i=0. Evidence is
fixed and conditionally independent. Dynamic HuGE/dependence corrections are NOT
part of this experimental target. Gaussian slab integration below is quadrature,
not an analytic or exact full-pipeline posterior.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.special import expit, logsumexp
from numpy.polynomial.hermite import hermgauss


@dataclass(frozen=True)
class FixedEvidenceTarget:
    design: np.ndarray
    log_bf: np.ndarray
    offset: np.ndarray
    inclusion: np.ndarray
    slab_variance: np.ndarray

    def __post_init__(self):
        for key in ('design','log_bf','offset','inclusion','slab_variance'):
            a=np.array(getattr(self,key),dtype=float,copy=True)
            if not np.all(np.isfinite(a)):raise ValueError('Require finite target inputs')
            a.setflags(write=False);object.__setattr__(self,key,a)
        n,p=self.design.shape
        if self.log_bf.shape!=(n,) or self.offset.shape!=(n,) or self.inclusion.shape!=(p,) or self.slab_variance.shape!=(p,):
            raise ValueError('Incompatible target dimensions')
        if np.any((self.inclusion<=0)|(self.inclusion>=1)) or np.any(self.slab_variance<=0):
            raise ValueError('Invalid spike/slab parameters')

    @property
    def size(self):return self.design.shape[1]

    def log_likelihood(self,beta):
        eta=np.asarray(beta)@self.design.T+self.offset
        return np.sum(np.logaddexp(0.,eta+self.log_bf)-np.logaddexp(0.,eta),axis=-1)

    def expected_indicators(self,beta):
        return expit(self.offset+self.design@beta+self.log_bf)

    def log_augmented_likelihood(self,beta,indicators):
        eta=self.offset+self.design@beta
        return float(np.sum(np.asarray(indicators)*(eta+self.log_bf)-np.logaddexp(0.,eta)))

    def log_prior(self,beta,support):
        beta=np.asarray(beta);support=np.asarray(support,dtype=bool)
        if np.any(beta[~support]!=0):return -np.inf
        slab=self.slab_variance[support]
        return float(np.log(self.inclusion[support]).sum()+np.log1p(-self.inclusion[~support]).sum()
                     -.5*np.sum(np.log(2*np.pi*slab)+beta[support]**2/slab))

    def quadrature(self,nodes=40,max_variables=3):
        """Enumerate supports and integrate Gaussian slabs. Check node refinement."""
        if self.size>max_variables:raise ValueError('Quadrature variable cap exceeded')
        if nodes<2:raise ValueError('Need at least two quadrature nodes')
        x,w=hermgauss(nodes);logw=np.log(w)-.5*np.log(np.pi)
        supports=np.array(list(product([False,True],repeat=self.size)))
        evidence=[];means=[];seconds=[]
        for support in supports:
            idx=np.flatnonzero(support);k=len(idx)
            if k:
                coords=np.array(np.meshgrid(*([np.arange(nodes)]*k),indexing='ij')).reshape(k,-1).T
                beta=np.zeros((len(coords),self.size))
                beta[:,idx]=np.sqrt(2*self.slab_variance[idx])*x[coords]
                log_weights=logw[coords].sum(axis=1)+self.log_likelihood(beta)
                normalizer=logsumexp(log_weights);weights=np.exp(log_weights-normalizer)
                mu=weights@beta;second=(beta.T*weights)@beta
            else:
                normalizer=self.log_likelihood(np.zeros(self.size));mu=np.zeros(self.size);second=np.zeros((self.size,self.size))
            evidence.append(normalizer+np.log(self.inclusion[support]).sum()+np.log1p(-self.inclusion[~support]).sum())
            means.append(mu);seconds.append(second)
        evidence=np.array(evidence);log_z=logsumexp(evidence);weights=np.exp(evidence-log_z)
        mu=weights@np.array(means);second=np.einsum('s,sij->ij',weights,np.array(seconds))
        return dict(supports=supports,weights=weights,log_evidence=float(log_z),inclusion=weights@supports,
                    mean=mu,covariance=second-np.outer(mu,mu),nodes=nodes)
