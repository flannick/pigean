"""Binary logistic likelihood with fixed offset and proper spike/slab prior."""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


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

    def duplicate_reference(self,tolerance=1e-10):
        """Adaptive 1D oracle when all columns are identical.

        The sum of active independent normal slabs is normal. Conditional
        moments of individual coefficients follow normal conditioning. This
        avoids a large tensor grid for a narrow diagonal likelihood ridge.
        """
        p=self.X.shape[1]
        if p>12 or not np.all(self.X==self.X[:,[0]]):
            raise ValueError('Duplicate-column oracle requires identical columns, p<=12')
        def ll(s):
            eta=self.offset+self.X[:,0]*s
            return np.sum(self.y*eta-np.logaddexp(0,eta))
        supports=np.array(list(product((False,True),repeat=p)))
        masses=[];means=[];seconds=[];errors=[]
        for support in supports:
            tau=self.variance[support];v=tau.sum();mu=np.zeros(p);second=np.zeros((p,p))
            if v:
                def lp(s):return ll(s)-.5*s*s/v-.5*np.log(2*np.pi*v)
                mode=minimize_scalar(lambda s:-lp(s),bounds=(-50*np.sqrt(v),50*np.sqrt(v)),method='bounded').x
                peak=lp(mode)
                # Split at the mode so adaptive integration cannot miss it.
                moments=[]
                for k in range(3):
                    f=lambda s:s**k*np.exp(lp(s)-peak)
                    left,e1=quad(f,-np.inf,mode,epsabs=tolerance,epsrel=tolerance)
                    right,e2=quad(f,mode,np.inf,epsabs=tolerance,epsrel=tolerance)
                    moments.append(left+right);errors.append(e1+e2)
                mass,first,squared=moments
                z=peak+np.log(mass);mu[support]=tau/v*first/mass
                second[np.ix_(support,support)]=np.diag(tau)-np.outer(tau,tau)/v+np.outer(tau,tau)/(v*v)*squared/mass
            else:z=ll(0.)
            masses.append(z+np.log(self.inclusion[support]).sum()+np.log1p(-self.inclusion[~support]).sum())
            means.append(mu);seconds.append(second)
        weights=np.exp(masses-logsumexp(masses));mu=weights@np.array(means)
        return dict(supports=supports,weights=weights,mean=mu,inclusion=weights@supports,
                    covariance=np.einsum('s,sij->ij',weights,np.array(seconds))-np.outer(mu,mu),
                    max_scaled_integral_error=max(errors,default=0.))
