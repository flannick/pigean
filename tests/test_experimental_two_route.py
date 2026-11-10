import importlib.util
from itertools import product
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit, logit, roots_hermitenorm

from pigean.experimental_two_route import (
    direct_without_annotations,
    main,
    pack_regions,
    project_gene_scores,
    sample,
)


P0=.05
INTERCEPT=logit(P0)
COUPLING=5.
Z=np.array([1.,-.7,.3,-.4])
REGIONS=[([0,1],[.88,.12]),([0,2],[.20,.80]),([2,3],[.65,.35])]


def oracle(n,pi,sd):
    """Sum gene states and assignments independently of the sampler."""
    nodes,mass=roots_hermitenorm(n)
    bs=np.r_[0.,sd*nodes]
    prior=np.r_[1.-pi,pi*mass/np.sqrt(2.*np.pi)]
    norm=0.
    inclusion=0.
    beta_mean=0.
    gene=np.zeros((3,len(Z)))
    for b,p in zip(bs,prior):
        q=expit(INTERCEPT+b*Z)
        for d in product((False,True),repeat=len(Z)):
            d=np.array(d)
            m=p*np.prod(np.where(d,q,1.-q))
            weight=np.exp(b*Z)*COUPLING**d
            for choices in product(*(range(len(ids)) for ids,_ in REGIONS)):
                prob=m
                assigned=np.zeros(len(Z),dtype=bool)
                for (ids,obs),j in zip(REGIONS,choices):
                    a=ids[j]
                    assigned[a]=True
                    prob*=obs[j]*weight[a]/weight[ids].sum()
                norm+=prob
                inclusion+=(b!=0.)*prob
                beta_mean+=b*prob
                gene+=prob*np.array([d,assigned,d|assigned])
    return inclusion/norm,beta_mean/norm,gene/norm


def design(neutral=0):
    # The uniform-evidence regions contribute constants after summing D,
    # but many additional D states make centered moves much harder.
    rs=REGIONS+[([j,j+1],[.5,.5]) for j in range(len(Z),len(Z)+neutral,2)]
    z=np.r_[Z,np.ones(neutral)]
    ids=np.array([g for gg,_ in rs for g in gg])
    w=np.array([v for _,ww in rs for v in ww])
    ptr=np.r_[0,np.cumsum([len(gg) for gg,_ in rs])]
    return z,pack_regions(ids,ptr,w,len(z))


@unittest.skipUnless(importlib.util.find_spec("numba"),"compiled sampler tests need Numba")
class TwoRouteSamplerTest(unittest.TestCase):
    def test_both_kernels_match_overlapping_exact_target(self):
        pi,sd=.43,1.3
        ref=oracle(48,pi,sd)
        fine=oracle(80,pi,sd)
        for a,b in zip(ref,fine):
            np.testing.assert_allclose(a,b,atol=2e-6,rtol=0)
        z,rs=design()
        for joint in (0.,.75,1.):
            rows=[sample(818+j,np.array([float(j>=2)]),z[None,:],np.array([pi]),np.array([sd]),rs,
                         np.arange(len(z)),COUPLING,INTERCEPT,25000,2000,2,joint) for j in range(4)]
            trace=np.stack([r[0][:,0] for r in rows])
            means=np.mean([r[2] for r in rows],axis=0)
            self.assertLess(abs(np.mean(trace!=0.)-fine[0]),.02)
            self.assertLess(abs(trace.mean()-fine[1]),.04)
            np.testing.assert_allclose(means,fine[2],atol=.012,rtol=0)

    def test_joint_update_recovers_hard_target(self):
        pi,sd=.43,1.3
        exact=oracle(80,pi,sd)
        z,rs=design(1800)
        traces=[]
        for joint in (0.,.75):
            fits=[sample(51+j,np.array([3. if j>=2 else 0.]),z[None,:],np.array([pi]),np.array([sd]),rs,
                         np.arange(len(Z)),COUPLING,INTERCEPT,3500,500,1,joint) for j in range(4)]
            traces.append(np.stack([row[0][:,0]!=0 for row in fits]))
        centered,joint=traces
        self.assertGreater(abs(centered[:2].mean()-centered[2:].mean()),.25)
        self.assertLess(abs(joint[:2].mean()-joint[2:].mean()),.03)
        self.assertLess(abs(joint.mean()-exact[0]),.02)
        self.assertGreater(np.count_nonzero(joint[:,1:]!=joint[:,:-1],axis=1).min(),500)

    def test_annotation_support_without_candidate_link(self):
        z,rs=design()
        beta=np.array([[.5],[1.5]])
        candidate=np.array([[.2,.2,.3,.3],[.4,.5,.3,.2],[.5,.6,.4,.4]])
        projected=project_gene_scores(beta,np.r_[z,1.][None,:],np.arange(len(z)),candidate,
                                      direct_without_annotations(rs),INTERCEPT)
        self.assertEqual(projected["direct_without_annotations"][-1],0.)
        self.assertEqual(projected["direct_with_annotations"][-1],0.)
        self.assertGreater(projected["indirect"][-1],P0)
        self.assertEqual(projected["either"][-1],projected["indirect"][-1])
        self.assertAlmostEqual(projected["direct_without_annotations"][0],1.-.12*.8)

    def test_cli_writes_all_routes_without_overwriting(self):
        z,rs=design()
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"input.npz"
            output=Path(tmp)/"output.npz"
            np.savez(source,mat=z[None,:],ids=rs[0],ptr=rs[1],weights=rs[2],pi=[.43],sd=[1.3],
                     initial=[0.],panel=[0,-1],all_gene_mat=np.r_[z,1.][None,:],keep=np.arange(len(z)))
            args=[str(source),str(output),"--random-seed","9","--steps","40","--burn","10"]
            main(args)
            with np.load(output) as f:
                self.assertEqual(f["beta"].shape,(6,1))
                self.assertEqual(f["indirect"].shape,(5,))
                self.assertEqual(f["either"][-1],f["indirect"][-1])
            with self.assertRaises(SystemExit):
                main(args)


if __name__=="__main__":
    unittest.main()
