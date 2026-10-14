"""Binary logistic likelihood with fixed offset and proper spike/slab prior."""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss


@dataclass
class LogisticTarget:
    X: np.ndarray
    y: np.ndarray
    offset: np.ndarray
    inclusion: np.ndarray
    variance: np.ndarray

    def __post_init__(self):
        for key in ('X','y','offset','inclusion','variance'):
            a=np.array(getattr(self,key),dtype=float,copy=True)
            if not np.all(np.isfinite(a)): raise ValueError('Nonfinite target')
            setattr(self,key,a)
        if self.X.ndim!=2: raise ValueError('Design must be a matrix')
        n,p=self.X.shape
        if self.y.shape!=(n,) or self.offset.shape!=(n,) or self.inclusion.shape!=(p,) or self.variance.shape!=(p,):
            raise ValueError('Target shape mismatch')
        if np.any((self.y!=0)&(self.y!=1)): raise ValueError('Binary response required')
        if np.any((self.inclusion<=0)|(self.inclusion>=1)) or np.any(self.variance<=0):
            raise ValueError('Invalid spike/slab prior')

    def log_likelihood(self,beta):
        eta=np.asarray(beta)@self.X.T+self.offset
        return np.sum(self.y*eta-np.logaddexp(0.,eta),axis=-1)

    def quadrature(self,nodes=64):
        """Independent non-augmented quadrature; at most two coefficients."""
        p=self.X.shape[1]
        if p>2: raise ValueError('Independent quadrature capped at two coefficients')
        x,w=hermgauss(nodes);lw=np.log(w)-.5*np.log(np.pi)
        supports=np.array(list(product((False,True),repeat=p)))
        masses=[];means=[];seconds=[]
        for support in supports:
            ids=np.flatnonzero(support)
            if ids.size:
                grid=np.array(list(product(range(nodes),repeat=len(ids))))
                b=np.zeros((len(grid),p));b[:,ids]=np.sqrt(2*self.variance[ids])*x[grid]
                logs=lw[grid].sum(axis=1)+self.log_likelihood(b)
                z=logsumexp(logs);weights=np.exp(logs-z)
                means.append(weights@b);seconds.append((b.T*weights)@b)
            else:
                z=self.log_likelihood(np.zeros(p));means.append(np.zeros(p));seconds.append(np.zeros((p,p)))
            masses.append(z+np.log(self.inclusion[support]).sum()+np.log1p(-self.inclusion[~support]).sum())
        weights=np.exp(masses-logsumexp(masses));mu=weights@np.array(means)
        return dict(supports=supports,weights=weights,mean=mu,inclusion=weights@supports,
                    covariance=np.einsum('s,sij->ij',weights,np.array(seconds))-np.outer(mu,mu))
