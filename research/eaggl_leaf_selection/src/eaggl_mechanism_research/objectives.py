"""Modular additive/union reconstruction with native ARD and concentration priors.

All scores are heuristic objectives, not independent-edge posterior likelihoods.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def union_loss_gradient(W, target):
    n,k=W.shape;gradient=np.zeros_like(W);loss=0.;prefix=np.ones(k+1)
    for i in range(n):
        for j in range(i):
            prefix[0]=1.
            for c in range(k):prefix[c+1]=prefix[c]*(1.-W[i,c]*W[j,c])
            residual=1.-prefix[k]-target[i,j];loss+=residual*residual
            suffix=1.
            for c in range(k-1,-1,-1):
                d=2.*residual*prefix[c]*suffix
                gradient[i,c]+=d*W[j,c];gradient[j,c]+=d*W[i,c]
                suffix*=1.-W[i,c]*W[j,c]
    return loss,gradient

@njit(cache=True)
def union_predict(W):
    n,k=W.shape;P=np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            s=1.
            for c in range(k):s*=1.-W[i,c]*W[j,c]
            P[i,j]=1.-s;P[j,i]=P[i,j]
    return P

class Objective:
    def __init__(self,target,*,link='additive',b0,C,phi_eff,column=0.,row=0.,penalty='l1l2',column_fraction=.25,row_effective=5.):
        self.target=np.asarray(target,dtype=float).copy();np.fill_diagonal(self.target,0.)
        if (self.target.ndim!=2 or self.target.shape[0]!=self.target.shape[1]
            or not np.isfinite(self.target).all() or not np.allclose(self.target,self.target.T,atol=1e-12)
            or min(b0,C)<=0 or min(phi_eff,column,row)<0 or link not in ['additive','union']):raise ValueError('Invalid objective')
        self.link=link;self.b0=float(b0);self.C=float(C);self.phi_eff=float(phi_eff);self.column=float(column);self.row=float(row)
        self.penalty=penalty;self.column_fraction=float(column_fraction);self.row_effective=float(row_effective)
        if penalty not in ['l1l2','diffuse'] or not 0<column_fraction<=1 or row_effective<=0:raise ValueError('Invalid concentration controls')
        self.norm2=float(np.sum(self.target*self.target));self.calls=0
    def evaluate(self,W):
        W=np.asarray(W,dtype=float);self.calls+=1
        if self.link=='union':data,gradient=union_loss_gradient(W,self.target)
        else:
            gram=W.T@W;MW=self.target@W;energy=(W*W).sum(axis=1)
            data=.5*(self.norm2-2*np.sum(W*MW)+np.sum(gram*gram)-np.sum(energy*energy))
            gradient=2*(W@gram-energy[:,None]*W-MW)
        energy=(W*W).sum(axis=0);base=.5*energy+self.b0
        value=data+self.phi_eff*self.C*np.log1p(.5*energy/self.b0).sum()
        gradient+=self.phi_eff*self.C*W/base
        # Machine-small smoothing defines the L2 derivative at zero only.
        eps=1e-12
        if self.column:
            norms=np.sqrt(energy+eps*eps)
            if self.penalty=='l1l2':
                value+=self.column*(W.sum(axis=0)-norms+eps).sum();gradient+=self.column*(1-W/norms)
            else:
                scale=np.sqrt(len(W)*self.column_fraction);hinge=np.maximum(W.sum(axis=0)-scale*norms,0)
                value+=self.column*np.sum(hinge*hinge);gradient+=2*self.column*hinge*(1-scale*W/norms)
        if self.row:
            norms=np.sqrt((W*W).sum(axis=1)+eps*eps)
            if self.penalty=='l1l2':
                value+=self.row*(W.sum(axis=1)-norms+eps).sum();gradient+=self.row*(1-W/norms[:,None])
            else:
                scale=np.sqrt(self.row_effective);hinge=np.maximum(W.sum(axis=1)-scale*norms,0)
                value+=self.row*np.sum(hinge*hinge);gradient+=2*self.row*hinge[:,None]*(1-scale*W/norms[:,None])
        return float(value),gradient
    def predict(self,W):
        if self.link=='union':return union_predict(W)
        P=W@W.T;np.fill_diagonal(P,0);return P
    def metadata(self):return dict(link=self.link,b0=self.b0,C=self.C,phi_eff=self.phi_eff,column=self.column,row=self.row,penalty=self.penalty,column_fraction=self.column_fraction,row_effective=self.row_effective)

def diagnostics(W,G,previous=None):
    d=dict(projected_gradient=float(np.linalg.norm(W-np.clip(W-G,0,1))/max(np.linalg.norm(W),1e-12)))
    if previous is not None:
        strong=np.argsort(np.sum(W*W,axis=0))[-min(5,W.shape[1]):]
        d.update(relative_change=float(np.linalg.norm(W-previous)/max(np.linalg.norm(W),1e-12)),strong_change=float(np.max(np.linalg.norm(W[:,strong]-previous[:,strong],axis=0)/np.maximum(np.linalg.norm(W[:,strong],axis=0),1e-12))))
    return d
