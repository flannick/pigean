"""Tiny observed-response logistic target with a fixed intercept and proper prior."""
import numpy as np
from .fixed_evidence import FixedEvidenceTarget


class ObservedLogisticTarget(FixedEvidenceTarget):
    def __init__(self,design,response,offset,inclusion,slab_variance):
        super().__init__(design,np.zeros(len(response)),offset,inclusion,slab_variance)
        y=np.asarray(response,dtype=float)
        if y.shape!=self.offset.shape or np.any((y!=0)&(y!=1)):
            raise ValueError('Require binary observed response')
        object.__setattr__(self,'response',y)

    def log_likelihood(self,beta):
        eta=np.asarray(beta)@self.design.T+self.offset
        return np.sum(self.response*eta-np.logaddexp(0.,eta),axis=-1)
