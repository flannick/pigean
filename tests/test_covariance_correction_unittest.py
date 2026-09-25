import sys,unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pegs_shared.regression import compute_beta_tildes,compute_logistic_beta_tildes,compute_multivariate_beta_tildes
from pegs_shared.covariance import marginal_inflation,AnnotationCorrelation,annotation_correlations

class CovarianceTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(931);self.X=rng.uniform(.1,2,(90,3));self.X[rng.random((90,3))<.6]=0
  self.y=(rng.random(90)<.3).astype(float)
  block=np.repeat(np.arange(9),10);self.R=.4*(block[:,None]==block[None,:])+.6*np.eye(90)
 def test_logistic_identity_and_dense_sandwich(self):
  for pseudo in [False,True]:
   X=sparse.csc_matrix(self.X);a=compute_logistic_beta_tildes(X,self.y,append_pseudo=pseudo,rel_tol=1e-10)
   for R in [np.eye(90),self.R,sparse.csr_matrix(self.R)]:
    b=compute_logistic_beta_tildes(X,self.y,append_pseudo=pseudo,resid_correlation_matrix=R,rel_tol=1e-10)
    np.testing.assert_allclose(a[0],b[0])
    rr=R.toarray() if sparse.issparse(R) else R
    if pseudo:rr=sparse.block_diag([rr,np.ones((1,1))]).toarray()
    for j in range(3):
     x=self.X[:,j];scale=x.std();rawbeta=a[0][j]/scale
     if pseudo:x=np.append(x,1)
     D=np.column_stack([np.ones(len(x)),x]);mu=expit(D@np.array([a[5][j],rawbeta]));v=mu*(1-mu)
     H=D.T@(v[:,None]*D);inv=np.linalg.inv(H)
     cov=inv@D.T@(np.sqrt(v)[:,None]*rr*np.sqrt(v)[None,:])@D@inv
     self.assertAlmostEqual(b[1][j]/scale,np.sqrt(cov[1,1]),places=6)
 def test_linear_centering_dense_sparse_and_negative_correlations(self):
  for X in [self.X,sparse.csc_matrix(self.X)]:
   for R in [np.eye(90),self.R]:
    a=compute_beta_tildes(X,self.y);b=compute_beta_tildes(X,self.y,resid_correlation_matrix=R)
    c=self.X-self.X.mean(axis=0);ratio=np.sqrt(np.sum(c*(R@c),axis=0)/np.sum(c*c,axis=0))
    np.testing.assert_allclose(b[1]/a[1],ratio,rtol=1e-12)
   negative=np.eye(90);negative[0,1]=negative[1,0]=-.4
   self.assertTrue(np.all(np.isfinite(marginal_inflation(X,negative))))
 def test_trait_specific_permutation_and_degenerate_outcome(self):
  y=np.array([self.y,1-self.y,np.zeros(90)]);rs=[self.R,np.eye(90),np.eye(90)]
  a=compute_logistic_beta_tildes(sparse.csc_matrix(self.X),y,resid_correlation_matrix=rs)
  for i in [0,1]:
   b=compute_logistic_beta_tildes(sparse.csc_matrix(self.X),y[i],resid_correlation_matrix=rs[i])
   for field in range(4):np.testing.assert_allclose(a[field][i],b[field])
  perm=[2,0,1];b=compute_logistic_beta_tildes(sparse.csc_matrix(self.X),y[perm],resid_correlation_matrix=[rs[i] for i in perm])
  for field in range(4):np.testing.assert_allclose(b[field],a[field][perm])
 def test_annotation_dense_reference_psd_and_direct_diagonal(self):
  X=np.array([[1,1,0],[1,0,1],[0,0,0],[0,1,1]],float);beta=np.array([.7,.4,.3]);d=np.array([.2,.3,2.,-.4])
  op=AnnotationCorrelation(X,beta,d);L=X*beta;L*=op.keep[:,None];L-=L.mean(axis=1)[:,None]
  C=L@L.T/3+np.diag(d*d);v=np.diag(C);C+=np.diag(v==0);v=np.diag(C)
  expected=.5*np.eye(4)+.5*C/np.sqrt(v[:,None]*v[None,:]);actual=op@np.eye(4)
  np.testing.assert_allclose(actual,expected,atol=1e-12);np.testing.assert_allclose(np.diag(actual),1)
  self.assertGreaterEqual(np.linalg.eigvalsh(actual).min(),.5-1e-12)
  np.testing.assert_allclose(actual[2],[0,0,1,0])
  batch=annotation_correlations(sparse.csr_matrix(X),np.array([beta,beta*2]),np.array([d,d]))
  np.testing.assert_allclose(batch[0]@np.eye(4),actual)
 def test_invalid_correlations_rejected(self):
  for r in [np.eye(89),np.eye(90)*2,np.full((90,90),np.nan)]:
   with self.assertRaises(ValueError):marginal_inflation(self.X,r)
  r=np.eye(90);r[0,1]=.2
  with self.assertRaises(ValueError):marginal_inflation(self.X,r)
 def test_joint_identity_and_covariance_scale(self):
  y=np.array([self.y,self.y*7]);D=np.column_stack([self.X,np.ones(90)])
  a=compute_multivariate_beta_tildes(self.X,y)
  identity=compute_multivariate_beta_tildes(self.X,y,resid_correlation_matrix=[np.eye(90)]*2)
  np.testing.assert_allclose(a[1],identity[1],rtol=1e-12)
  b=compute_multivariate_beta_tildes(self.X,y,resid_correlation_matrix=[self.R]*2)
  inverse=np.linalg.inv(D.T@D);resid=y.T-D@np.linalg.lstsq(D,y.T,rcond=None)[0]
  expected=np.sqrt(np.diag(inverse@D.T@self.R@D@inverse)[None,:3]*(resid**2).sum(axis=0)[:,None]/86)
  np.testing.assert_allclose(b[1],expected,rtol=1e-12)
  np.testing.assert_allclose(b[1][1],b[1][0]*7)
 def test_single_trait_annotation_operator_joint(self):
  op=AnnotationCorrelation(sparse.csr_matrix(self.X),np.ones(3),self.y)
  a=compute_multivariate_beta_tildes(self.X,self.y[None,:],resid_correlation_matrix=op)
  b=compute_multivariate_beta_tildes(self.X,self.y[None,:],resid_correlation_matrix=[op@np.eye(90)])
  np.testing.assert_allclose(a[1],b[1])
 def test_huber_correlation_fails_explicitly(self):
  from eaggl.regression import compute_robust_betas
  from pegs_cli_errors import DataValidationError
  with self.assertRaisesRegex(DataValidationError,'not validated'):
   compute_robust_betas(None,self.X,self.y[None,:],resid_correlation_matrix=self.R,
     finalize_regression_fn=None,log_fn=lambda *args:None,debug_level=0)
 def test_large_annotation_operator_has_linear_storage(self):
  n=10000;X=sparse.eye(n,format='csr');op=AnnotationCorrelation(X,np.ones(n))
  result=op@np.ones((n,2));self.assertEqual(result.shape,(n,2));self.assertLess(op.storage_nnz,7*n)

if __name__=='__main__':unittest.main()
