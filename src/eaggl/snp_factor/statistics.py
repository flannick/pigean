"""Blockwise summary-score association and descriptive unsigned GLS.

Every reported SE belongs to its own model. Rank deficient coefficients are
reported only if estimable, never as arbitrary generalized-inverse effects.
"""
from __future__ import annotations
import numpy as np
from scipy import linalg, optimize, special, stats


def solve_estimable(gram,rhs,condition_limit=1e10):
    values,vectors=np.linalg.eigh((gram+gram.T)/2)
    cutoff=max(float(values.max(initial=0))/condition_limit,1e-12)
    keep=values>cutoff
    rank=int(keep.sum())
    null=vectors[:,~keep]
    estimable=np.sum(null*null,axis=1)<1e-10
    if not rank:
        return np.full_like(rhs,np.nan,dtype=float),np.zeros_like(gram),estimable,rank,np.inf
    inverse=(vectors[:,keep]/values[keep])@vectors[:,keep].T
    result=inverse@rhs
    result[~estimable]=np.nan
    condition=float(values.max()/values[keep].min()) if rank==len(values) else np.inf
    return result,inverse,estimable,rank,condition


def _chi2_logsf(q,df):
    """Log survival, including tails below floating-point probability range."""
    if q<=0: return 0.
    value=float(stats.chi2.logsf(q,df))
    if np.isfinite(value): return value
    # Upper incomplete gamma asymptotic series. This branch is reached only
    # deep in the right tail, where successive terms decrease rapidly.
    x=q/2.; shape=df/2.; term=1.; total=1.
    for j in range(1,10000):
        previous=abs(term); term*= (shape-j)/x
        if abs(term)>previous: raise FloatingPointError('Unconverged gamma tail')
        total+=term
        if abs(term)<=abs(total)*2e-16: break
    else: raise FloatingPointError('Unconverged gamma tail')
    if total<=0: raise FloatingPointError('Invalid gamma tail')
    return float((shape-1)*np.log(x)-x-special.gammaln(shape)+np.log(total))


def mixture_logsf(q,eigenvalues):
    """Log Kuonen/Lugannani-Rice tail; exact scaled chi-square special cases.

    Evaluate the upper-tail LR expression after factoring out phi(w). This
    preserves its approximation while avoiding underflow of both summands.
    """
    eig=np.asarray(eigenvalues,dtype=float)
    if not np.isfinite(q) or np.any(~np.isfinite(eig)): return np.nan,'numerical_failure'
    eig=eig[eig>max(eig.max(initial=0)*1e-12,1e-15)]
    if not len(eig): return np.nan,'zero_variance'
    if q<=0: return 0.,'analytic'
    if np.allclose(eig,eig[0],rtol=1e-10,atol=0):
        return _chi2_logsf(q/eig[0],len(eig)),'analytic'
    scale=eig.max(); eig=eig/scale; q=q/scale
    mean=eig.sum(); var=2*np.dot(eig,eig)
    if abs(q-mean)<1e-5*np.sqrt(var):
        p=.5-(8*np.sum(eig**3)/(6*var**1.5))/np.sqrt(2*np.pi)
        return float(np.log(np.clip(p,0,1))),'saddlepoint'
    kp=lambda t: np.sum(eig/(1-2*t*eig))
    if q>mean: lo,hi=0.,np.nextafter(.5,0.)
    else:
        hi=0.; lo=-1.
        while kp(lo)>q: lo*=2
    try:
        t=optimize.brentq(lambda t:kp(t)-q,lo,hi,xtol=1e-14)
        K=-.5*np.log1p(-2*t*eig).sum()
        w=np.sign(t)*np.sqrt(max(2*(t*q-K),0))
        v=t*np.sqrt(2*np.sum((eig/(1-2*t*eig))**2))
        if w>0:
            mills=np.sqrt(np.pi/2)*special.erfcx(w/np.sqrt(2))
            bracket=mills-1/w+1/v
            if w>20:
                # Mills ratio minus 1/w without subtracting near-equal values.
                term=-1/w**3; correction=term
                for j in range(1,100):
                    term*=-(2*j+1)/w**2; correction+=term
                    if abs(term)<abs(correction)*2e-16: break
                bracket=correction+1/v
            logp=-w*w/2-.5*np.log(2*np.pi)+np.log(bracket) if bracket>0 else np.nan
        else:
            p=special.ndtr(-w)+np.exp(-w*w/2)/np.sqrt(2*np.pi)*(1/v-1/w)
            logp=np.log(p) if p>0 else np.nan
        if not np.isfinite(logp) or logp>0: return np.nan,'numerical_failure'
        return float(logp),'saddlepoint'
    except (ValueError,FloatingPointError):
        return np.nan,'numerical_failure'


def mixture_sf(q,eigenvalues):
    """Probability interface; use mixture_logsf to retain underflowed tails."""
    logp,method=mixture_logsf(q,eigenvalues)
    return float(np.exp(logp)),method


def _row(factor,trait,method,scope,beta=np.nan,se=np.nan,p=np.nan,status='ok',**extra):
    z=beta/se if np.isfinite(se) and se>0 else np.nan
    return dict(factor=factor,trait_index=trait,method=method,scope=scope,beta=float(beta),se=float(se),statistic=float(z),p=float(p),status=status,**extra)


def signed_associations(U,b,se,blocks,factors,scope='both'):
    """Matrices b/se are selected SNP x batched trait; U is sparse SNP x factor."""
    k=U.shape[1]; nt=b.shape[1]
    gram=np.zeros((k,k)); m=np.asarray(U.T@b); vm=np.zeros((nt,k,k))
    variance_squares=np.zeros(k);max_block_variance=np.zeros(k);block_count=0
    for ix,R,R2,signed in blocks():
        if not signed: raise ValueError('Signed score testing requires signed LD, not r2 only')
        u=U[ix].toarray()
        block_gram=u.T@R@u;gram+=block_gram
        block_variance=np.maximum(np.diag(block_gram),0)
        variance_squares+=block_variance**2;max_block_variance=np.maximum(max_block_variance,block_variance);block_count+=1
        us=u[:,:,None]*se[ix,None,:]
        vm+=np.einsum('ikt,ij,jlt->tkl',us,R,us,optimize=True)
    h=np.sqrt(np.maximum(np.diag(gram),0)); results=[]
    estimate,inv,estimable,rank,condition=solve_estimable(gram,m)
    max_change=np.zeros((k,nt)); marginal_change=np.zeros((k,nt))
    marginal_sensitivity_valid=np.ones(k,dtype=bool);joint_sensitivity_valid=np.ones(k,dtype=bool)
    for ix,R,R2,signed in blocks():
        u=U[ix].toarray(); bg=u.T@R@u; bm=u.T@b[ix]
        h_without=np.sqrt(np.maximum(np.diag(gram-bg),0))
        marginal=np.divide(m,h[:,None],out=np.full_like(m,np.nan),where=h[:,None]>0)
        without=np.divide(m-bm,h_without[:,None],out=np.full_like(m,np.nan),where=h_without[:,None]>0)
        marginal_sensitivity_valid &= h_without>0
        marginal_change=np.maximum(marginal_change,np.nan_to_num(abs(without-marginal)))
        deleted,_,de,dr,_=solve_estimable(gram-bg,m-bm)
        joint_sensitivity_valid &= de & (dr==rank) & (h_without>0)
        delta=abs(h_without[:,None]*deleted-h[:,None]*estimate)
        max_change=np.maximum(max_change,np.nan_to_num(delta))
    marginal_change[~marginal_sensitivity_valid]=np.nan
    max_change[~joint_sensitivity_valid]=np.nan
    diagnostics=[dict(score_variance=float(h[j]**2),ld_blocks=block_count,
        effective_loci=float(h[j]**4/variance_squares[j]) if variance_squares[j]>0 else 0.,
        strongest_locus_variance_share=float(max_block_variance[j]/h[j]**2) if h[j]>0 else np.nan) for j in range(k)]
    for t in range(nt):
        if scope in ('both','marginal'):
            for j,factor in enumerate(factors):
                beta=m[j,t]/h[j] if h[j]>0 else np.nan
                sigma=np.sqrt(max(vm[t,j,j],0))/h[j] if h[j]>0 else np.nan
                p=2*stats.norm.sf(abs(beta/sigma)) if sigma>0 else np.nan
                results.append(_row(factor,t,'signed_score','marginal',beta,sigma,p,'ok' if np.isfinite(p) else 'zero_variance',joint_rank=rank,condition=condition,max_leave_one_locus_out_beta_change=float(marginal_change[j,t]),**diagnostics[j]))
        if scope in ('both','joint'):
            covariance=inv@vm[t]@inv
            for j,factor in enumerate(factors):
                beta=h[j]*estimate[j,t]
                sigma=h[j]*np.sqrt(max(covariance[j,j],0)) if estimable[j] and h[j]>0 else np.nan
                p=2*stats.norm.sf(abs(beta/sigma)) if sigma>0 else np.nan
                results.append(_row(factor,t,'signed_score','joint',beta,sigma,p,'ok' if np.isfinite(p) else 'not_identifiable',joint_rank=rank,condition=condition,max_leave_one_locus_out_beta_change=float(max_change[j,t]),**diagnostics[j]))
    return results


def _gls_block(A,y,ix,R2):
    x=np.column_stack((np.ones(len(ix)),A[ix].toarray()))
    c=2*R2
    values,vectors=np.linalg.eigh(c)
    if values.min() < -1e-7: raise ValueError('Invalid squared-LD covariance')
    keep=values>max(values.max()*1e-10,1e-12)
    whiten=(vectors[:,keep]/np.sqrt(values[keep])).T
    xw=whiten@x; yw=whiten@y[ix]
    return xw.T@xw,xw.T@yw


def unsigned_associations(A,z2,blocks,factors,scope='both',quadratic=True):
    k=A.shape[1]; nt=z2.shape[1]; y=z2-1
    gram=np.zeros((k+1,k+1)); rhs=np.zeros((k+1,nt)); block_count=0
    q=np.asarray(A.T@z2); mu=np.asarray(A.sum(axis=0)).ravel()
    nullvar=np.zeros(k); max_contrib=np.zeros((k,nt)); removed_max=np.zeros((k,nt))
    for ix,R,R2,signed in blocks():
        g,b=_gls_block(A,y,ix,R2); gram+=g; rhs+=b; block_count+=1
        a=A[ix].toarray()
        nullvar+=2*np.einsum('ik,ij,jk->k',a,R2,a)
        contribution=a.T@z2[ix]
        max_contrib=np.maximum(max_contrib,contribution)
    results=[]
    # Only O(SNPs) eigenvalues held at a time, never SNPs x factors x traits.
    for j,factor in (enumerate(factors) if quadratic else []):
        eigen=[]; exact=True
        for ix,R,R2,signed in blocks():
            a=A[ix,j].toarray().ravel(); keep=a>0
            if not np.any(keep): continue
            if not signed: exact=False; continue
            root=np.sqrt(a[keep]); matrix=root[:,None]*R[np.ix_(keep,keep)]*root[None,:]
            eigen.extend(np.linalg.eigvalsh(matrix))
        for t in range(nt):
            if exact: logp,algorithm=mixture_logsf(q[j,t],eigen); p=float(np.exp(logp))
            elif nullvar[j]>0 and mu[j]>0:
                logp=_chi2_logsf(q[j,t]/(nullvar[j]/(2*mu[j])),2*mu[j]**2/nullvar[j]); p=float(np.exp(logp)); algorithm='r2_moment_approximation'
            else: p,logp,algorithm=np.nan,np.nan,'zero_variance'
            results.append(_row(factor,t,'unsigned_quadratic','quadratic',p=p,status='ok' if np.isfinite(p) else algorithm,
                log_p=float(logp),neg_log10_p=float(-logp/np.log(10)),p_underflow=bool(p==0 and np.isfinite(logp)),
                Q=float(q[j,t]),null_mean=float(mu[j]),null_variance=float(nullvar[j]),tail_method=algorithm,
                effective_loci=float(2*mu[j]**2/nullvar[j]) if nullvar[j]>0 else 0.,
                strongest_locus_share=float(max_contrib[j,t]/q[j,t]) if q[j,t]>0 else 0.,
                Q_without_strongest_locus=float(q[j,t]-max_contrib[j,t])))
    fits=[]
    if scope in ('both','marginal'): fits.extend(('marginal',np.array([0,j+1]),[j]) for j in range(k))
    if scope in ('both','joint'): fits.append(('joint',np.arange(k+1),list(range(k))))
    for fit_scope,cols,js in fits:
        g=gram[np.ix_(cols,cols)]; r=rhs[cols]
        beta,_,estimable,rank,condition=solve_estimable(g,r)
        max_change=np.zeros_like(beta)
        sums=np.zeros_like(beta); sums2=np.zeros_like(beta); valid=np.ones(len(cols),dtype=bool)
        for ix,R,R2,signed in blocks():
            bg,br=_gls_block(A,y,ix,R2)
            deleted,_,de,dr,_=solve_estimable(g-bg[np.ix_(cols,cols)],r-br[cols])
            valid &= de & (dr==rank)
            safe=np.nan_to_num(deleted); sums+=safe; sums2+=safe*safe
            max_change=np.maximum(max_change,np.nan_to_num(abs(deleted-beta)))
        variance=(block_count-1)/max(block_count,1)*(sums2-sums*sums/max(block_count,1))
        for local,j in enumerate(js,1):
            for t in range(nt):
                sigma=np.sqrt(max(variance[local,t],0)) if block_count>2 and valid[local] and estimable[local] else np.nan
                p=2*stats.t.sf(abs(beta[local,t]/sigma),block_count-1) if sigma>0 else np.nan
                status='not_identifiable' if not estimable[local] else ('ok' if np.isfinite(p) else 'jackknife_unavailable')
                results.append(_row(factors[j],t,'unsigned_evidence',fit_scope,beta[local,t],sigma,p,status,
                    joint_rank=rank,condition=condition,jackknife_blocks=block_count,intercept=float(beta[0,t]),max_leave_one_locus_out_beta_change=float(max_change[local,t]) if valid[local] and estimable[local] else np.nan))
    return results
