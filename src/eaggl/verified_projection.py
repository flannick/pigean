"""Opt-in fixed-basis quadratic projection with explicit per-output KKT checks."""
import numpy as np
from scipy.optimize import minimize
from scipy import sparse


def bounded_projection(x, cap=None, mass=None):
    y=np.maximum(x,0.)
    if cap is not None:
        y=np.minimum(y,cap)
    if mass is not None and y.sum()>mass:
        lo,hi=0.,float(x.max())
        for _ in range(70):
            mid=(lo+hi)/2
            y=np.maximum(x-mid,0.)
            if cap is not None:
                y=np.minimum(y,cap)
            if y.sum()>mass:
                lo=mid
            else:
                hi=mid
        y=np.maximum(x-hi,0.)
        if cap is not None:
            y=np.minimum(y,cap)
    return y


def residual(H,Q,z,cap=None,mass=None):
    gradient=np.einsum('mij,mj->mi',Q,H)-z
    L=np.maximum(np.max(np.sum(np.abs(Q),axis=2),axis=1),1e-15)
    proposed=np.maximum(H-gradient/L[:,None],0.)
    if cap is not None:
        proposed=np.minimum(proposed,cap)
    if mass is not None:
        proposed=np.array([bounded_projection(x,cap,mass) for x in H-gradient/L[:,None]])
    return np.linalg.norm(H-proposed,axis=1)/np.maximum(np.linalg.norm(H,axis=1),1e-8)


def refine_active_face(x,Q,z,cap=None,mass=None):
    """Polish the active face found by SLSQP, retaining only feasible improvements."""
    boundary=1e-7*max(1.,float(np.max(x,initial=0)))
    fixed=(x<=boundary)
    if cap is not None:
        fixed|=(x>=cap-boundary)
    free=~fixed
    if not np.any(free):
        return x
    candidate=x.copy()
    candidate[x<=boundary]=0.
    if cap is not None:
        candidate[x>=cap-boundary]=cap
    rhs=z[free]-Q[np.ix_(free,fixed)]@candidate[fixed]
    system=Q[np.ix_(free,free)]
    active_mass=mass is not None and abs(x.sum()-mass)<=boundary
    if active_mass:
        system=np.block([[system,np.ones((free.sum(),1))],
                         [np.ones((1,free.sum())),np.zeros((1,1))]])
        rhs=np.r_[rhs,mass-candidate[fixed].sum()]
    solved=np.linalg.lstsq(system,rhs,rcond=None)[0]
    candidate[free]=solved[:free.sum()]
    if np.min(candidate)<-1e-12 or (cap is not None and np.max(candidate)>cap+1e-12):
        return x
    if mass is not None and candidate.sum()>mass+1e-12:
        return x
    candidate=bounded_projection(candidate,cap,mass)
    old=.5*x@Q@x-z@x
    new=.5*candidate@Q@candidate-z@candidate
    before=residual(x[None],Q[None],z[None],cap,mass)[0]
    after=residual(candidate[None],Q[None],z[None],cap,mass)[0]
    if new<=old+1e-12*max(abs(old),1) and after<before:
        return candidate
    return x


def project(W,V,*,left=None,right=None,phi=0.,lambdas=None,cap=None,mass=None,
            max_cycles=2000,tol=1e-8,block_size=256):
    W=W.toarray() if sparse.issparse(W) else np.asarray(W,dtype=float)
    if W.ndim!=2 or W.shape[1]<1 or V.ndim!=2 or W.shape[0]!=V.shape[0]:
        raise ValueError('Projection basis and response dimensions disagree')
    if not np.isfinite(W).all() or np.any(W<0):
        raise ValueError('Projection basis must be finite and nonnegative')
    if not np.isfinite(tol) or min(tol,max_cycles,block_size)<=0 or not np.isfinite(phi) or phi<0:
        raise ValueError('Invalid projection solver settings')
    if cap is not None and (not np.isfinite(cap) or cap<=0):
        raise ValueError('Projection box cap must be positive')
    if mass is not None and (not np.isfinite(mass) or mass<=0):
        raise ValueError('Projection sum cap must be positive')
    n,k=W.shape;m=V.shape[1]
    la=np.ones(k) if lambdas is None else np.asarray(lambdas,dtype=float)
    if la.shape!=(k,) or not np.isfinite(la).all() or np.any(la<=0):
        raise ValueError('Projection regularization scales must be positive and finite')
    regularizer=phi/la
    gram=W.T@W
    extended=left is not None or right is not None
    if extended:
        if left is not None:
            left=left.toarray() if sparse.issparse(left) else np.asarray(left,dtype=float)
            left=left[:,None] if left.ndim==1 else left
        if right is not None:
            right=right.toarray() if sparse.issparse(right) else np.asarray(right,dtype=float)
            right=right[:,None] if right.ndim==1 else right
        u=left.shape[1] if left is not None else right.shape[1]
        left=np.ones((n,u)) if left is None else left
        right=np.ones((m,u)) if right is None else right
        if left.shape!=(n,u) or right.shape!=(m,u):
            raise ValueError('Projection weight dimensions disagree')
        if not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left<0) or np.any(right<0):
            raise ValueError('Projection weights must be finite and nonnegative')
        grams=np.array([W.T@(W*left[:,i,None]) for i in range(u)])
    answer=np.zeros((m,k));records=[];polished=0
    for start in range(0,m,block_size):
        stop=min(start+block_size,m)
        response=V[:,start:stop]
        response=response.toarray() if sparse.issparse(response) else np.asarray(response,dtype=float)
        if not np.isfinite(response).all() or np.any(response<0):
            raise ValueError('Projection responses must be finite and nonnegative')
        if extended:
            weights=left@right[start:stop].T+1e-10
            Q=np.einsum('mu,uij->mij',right[start:stop],grams)+1e-10*gram[None]
            z=(W.T@(response*weights)).T-1e-10
        else:
            Q=np.broadcast_to(gram,(stop-start,k,k)).copy()
            z=(W.T@response).T-1e-10
        Q+=np.diag(regularizer)[None]
        H=np.zeros_like(z);G=-z.copy();cycles=0
        if mass is None:
            diagonal=np.diagonal(Q,axis1=1,axis2=2)
            for cycles in range(1,max_cycles+1):
                for j in range(k):
                    change=np.divide(G[:,j],diagonal[:,j],out=np.zeros(len(H)),where=diagonal[:,j]>0)
                    updated=np.maximum(H[:,j]-change,0.)
                    if cap is not None:
                        updated=np.minimum(updated,cap)
                    delta=updated-H[:,j];H[:,j]=updated
                    G+=delta[:,None]*Q[:,j,:]
                if cycles%10==0 and np.max(residual(H,Q,z,cap,mass),initial=0)<=tol:
                    break
        mapping=residual(H,Q,z,cap,mass)
        for j in np.where(mapping>tol)[0]:
            scale=max(float(np.linalg.norm(Q[j],ord=2)),1e-15)
            constraints=[] if mass is None else [dict(type='ineq',fun=lambda x:mass-x.sum(),jac=lambda x:-np.ones_like(x))]
            optimized=minimize(lambda x:(.5*x@Q[j]@x-z[j]@x)/scale,
                               bounded_projection(H[j],cap,mass),jac=lambda x:(Q[j]@x-z[j])/scale,
                               method='SLSQP',bounds=[(0,cap)]*k,constraints=constraints,
                               options=dict(ftol=1e-14,maxiter=5000))
            candidate=bounded_projection(optimized.x,cap,mass)
            candidate=refine_active_face(candidate,Q[j],z[j],cap,mass)
            old=.5*H[j]@Q[j]@H[j]-z[j]@H[j]
            new=.5*candidate@Q[j]@candidate-z[j]@candidate
            if new<=old+1e-12*max(abs(old),1):
                H[j]=candidate
            polished+=1
        mapping=residual(H,Q,z,cap,mass)
        answer[start:stop]=H
        records.append((cycles,float(np.max(mapping,initial=0)),int(np.sum(mapping>tol))))
    maximum=max((r[1] for r in records),default=0.)
    failures=sum(r[2] for r in records)
    diagnostic=dict(features=n,outputs=m,factors=k,iterations=max((r[0] for r in records),default=0),
                    converged=failures==0,hit_iteration_cap=failures>0,relative_change=None,
                    solver='verified_quadratic',criterion='per-output projected-gradient residual',
                    projected_gradient_max=maximum,stationarity_tol=tol,unconverged_outputs=failures,
                    polished_outputs=polished,box_cap=cap,sum_cap=mass,
                    objective='weighted half-squared loss + phi/(2*lambda)*h^2 + 1e-10*sum(h)')
    return answer,diagnostic
