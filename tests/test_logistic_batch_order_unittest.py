"""Trait batching must not change logistic gene-set enrichment estimates."""
import unittest
import numpy as np
from scipy import sparse
from pegs_shared.regression import compute_logistic_beta_tildes

class LogisticBatchOrderTest(unittest.TestCase):
    def test_batch_and_permutation_invariance(self):
        rng=np.random.default_rng(7052)
        X=sparse.csc_matrix(rng.uniform(size=(180,4))<np.array([.12,.25,.35,.5]))
        Y=np.vstack([rng.uniform(size=180)<p for p in [.08,.2,.4]]).astype(float)
        batch=compute_logistic_beta_tildes(X,Y.copy())
        for i in range(3):
            single=compute_logistic_beta_tildes(X,Y[i].copy())
            for field in [0,1,2,3,5]:np.testing.assert_allclose(batch[field][i],single[field],rtol=1e-10,atol=1e-12)
        order=[2,0,1];columns=[3,1,0,2]
        perm=compute_logistic_beta_tildes(X[:,columns],Y[order].copy())
        for field in [0,1,2,3,5]:np.testing.assert_allclose(perm[field],batch[field][order][:,columns],rtol=1e-10,atol=1e-12)

    def test_probabilistic_dichotomization_batch_invariance(self):
        rng=np.random.default_rng(17);X=sparse.csc_matrix(rng.random((120,3))<.3)
        Y=np.vstack([rng.beta(1,b,120) for b in [2,8,20]])
        batch=compute_logistic_beta_tildes(X,Y.copy())
        for i in range(3):
            single=compute_logistic_beta_tildes(X,Y[i].copy())
            for field in range(4):np.testing.assert_allclose(batch[field][i],single[field],rtol=1e-10,atol=1e-12)

if __name__=='__main__':unittest.main()
