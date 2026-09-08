import unittest
import numpy as np
from scipy import sparse, special
from audit_hdl_gene_dependence_20260907 import grouped_fit,covariance_stress

class AuditTests(unittest.TestCase):
    def test_grouped_fit_matches_dense_information(self):
        x=np.repeat([1,1,0,0],[12,13,8,67]).astype(bool);y=np.repeat([1.,0.,1.,0.],[12,13,8,67])
        fit=grouped_fit(x,y);D=np.column_stack([np.ones(101),np.r_[x,1]])
        mu=special.expit(D@np.array([special.logit(fit['p0']),fit['beta']]))
        cov=np.linalg.inv(D.T@((mu*(1-mu))[:,None]*D))
        self.assertAlmostEqual(np.sqrt(cov[1,1]),fit['se'])
    def test_sparse_stress_matches_dense_sandwich_and_binary_bound(self):
        x=np.repeat([1,1,0,0],[12,13,8,67]).astype(bool);y=np.repeat([1.,0.,1.,0.],[12,13,8,67])
        B=np.zeros((100,3));B[:40,0]=1;B[20:70,1]=1;B[60:90,2]=1
        degree=(B*B).sum(1);L=np.divide(B,np.sqrt(degree)[:,None],out=np.zeros_like(B),where=degree[:,None]>0)
        K=L@L.T+np.diag(degree==0)
        fit=grouped_fit(x,y);D=np.column_stack([np.ones(101),np.r_[x,1]])
        mu=np.r_[np.where(x,fit['p1'],fit['p0']),fit['p1']];v=mu*(1-mu)
        inv=np.linalg.inv(D.T@(v[:,None]*D))
        for row in covariance_stress(x,y,sparse.csr_matrix(B),weights=(0,.1,.5,1)):
            weight=row['shrinkage_weight'];R=sparse.block_diag([(1-weight)*np.eye(100)+weight*K,[[1]]]).toarray()
            cov=inv@D.T@(np.sqrt(v)[:,None]*R*np.sqrt(v)[None,:])@D@inv
            self.assertAlmostEqual(row['stress_se'],np.sqrt(cov[1,1]),places=12)
        self.assertFalse(row['pairwise_binary_bound_passes'])
    def test_dashboard_p_is_the_marginal_output_not_bayesian_beta(self):
        from pigean.dashboard import _merge_trait_projection_and_enrichment
        projection=[dict(trait='HDL',factor='Factor7',nnls_loading='0.5')]
        for beta in ['0','10']:
            enrichment={('HDL','Factor7'):dict(trait='HDL',Gene_Set='Factor7',P='1e-8',Z='5.7',SE='0.2',beta=beta,beta_uncorrected='2')}
            row=_merge_trait_projection_and_enrichment(projection,enrichment)[0]
            self.assertEqual(row['p_value'],'1e-8')
            self.assertEqual(row['beta'],beta)

    def test_undefined_deletion_fails_explicitly(self):
        self.assertEqual(grouped_fit(np.ones(8),np.arange(8)%2)['status'],'missing_predictor_group')
        self.assertEqual(grouped_fit(np.arange(8)%2,np.zeros(8))['status'],'separation')

if __name__=='__main__':unittest.main()
