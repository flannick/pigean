"""Cancellation-controlled differences for the profiled pilot objective."""
import numpy as np
from .spectral_solver import OffDiagonalObjective


class PreciseObjective(OffDiagonalObjective):
    def descent_slack(self,value):
        return 32*np.finfo(float).eps*max(abs(value),1.)

    def change(self,W,D,gradient):
        G=W.T@W;A=W.T@D;B=D.T@D
        energy=np.sum(W*W,axis=1)
        diagonal_change=2*np.sum(W*D,axis=1)+np.sum(D*D,axis=1)
        squared_change=(2*np.sum(G*B)+2*np.trace(A@A)+4*np.trace(A@B)
                        +np.sum(B*B)-np.sum(diagonal_change**2))
        base=.5*np.sum(W*W,axis=0)+self.b0
        lambdas=base/self.C
        data_gradient=gradient-self.phi_eff*W/lambdas-self.sparsity
        RD=W@A-energy[:,None]*D-self.M@D
        loss_change=np.sum(data_gradient*D)+np.sum(D*RD)+.5*squared_change
        penalty_change=0.
        if self.phi_eff:
            ratio=(np.sum(W*D,axis=0)+.5*np.sum(D*D,axis=0))/base
            # Computing the new energy directly is safer when a column nearly vanishes.
            large_drop=ratio<-.5
            terms=np.empty_like(ratio)
            terms[~large_drop]=np.log1p(ratio[~large_drop])
            updated=.5*np.sum((W+D)**2,axis=0)+self.b0
            terms[large_drop]=np.log(updated[large_drop]/base[large_drop])
            penalty_change=self.phi_eff*self.C*np.sum(terms)
        return float(loss_change+penalty_change+self.sparsity*D.sum())
