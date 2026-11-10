"""Local research sampler for a fixed-SNP, two-route gene involvement model.

This module is deliberately separate from the native PIGEAN CLI. It does not
alter the model, defaults, scores or output schemas of ``pigean gibbs``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without the optional extra
    def njit(**_):
        return lambda f: f


def pack_regions(ids, ptr, weights, n_genes):
    ids = np.asarray(ids, dtype=np.int64)
    ptr = np.asarray(ptr, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if (ptr.ndim!=1 or ids.ndim!=1 or weights.ndim!=1 or len(ptr)<1
            or ptr[0]!=0 or ptr[-1]!=len(ids) or len(weights)!=len(ids)
            or np.any(np.diff(ptr)<=0) or np.any((ids<0)|(ids>=n_genes))
            or not np.isfinite(weights).all() or np.any(weights<0)):
        raise ValueError("invalid candidate-region arrays")
    for lo,hi in zip(ptr[:-1],ptr[1:]):
        if len(set(ids[lo:hi]))!=hi-lo or not np.isclose(weights[lo:hi].sum(),1.,atol=1e-10,rtol=0):
            raise ValueError("candidate IDs must be unique per region and weights must sum to one")
    edge_region = np.repeat(np.arange(len(ptr)-1),np.diff(ptr))
    order = np.argsort(ids,kind="stable")
    gene_ptr = np.r_[0,np.cumsum(np.bincount(ids,minlength=n_genes))]
    return ids,ptr,weights,gene_ptr,edge_region[order],weights[order]


@njit(nogil=True)
def _logistic(x):
    if x >= 0.:
        return 1./(1.+np.exp(-x))
    v=np.exp(x)
    return v/(1.+v)


@njit(nogil=True)
def _regional_sums(u, d, coupling, regions):
    ids,ptr,w,_,_,_=regions
    numerator=np.zeros(len(ptr)-1)
    denominator=np.zeros(len(ptr)-1)
    for k in range(len(numerator)):
        for e in range(ptr[k],ptr[k+1]):
            g=ids[e]
            v=u[g]*(coupling if d[g] else 1.)
            numerator[k]+=w[e]*v
            denominator[k]+=v
    return numerator,denominator


@njit(nogil=True)
def _log_likelihood(numerator,denominator):
    out=0.
    for k in range(len(numerator)):
        out+=np.log(numerator[k]/denominator[k])
    return out


@njit(nogil=True)
def _log_gene_prior(eta,d,intercept):
    total=0.
    for g in range(len(d)):
        v=intercept+eta[g]
        total+=(v if d[g] else 0.)-max(v,0.)-np.log1p(np.exp(-abs(v)))
    return total


@njit(nogil=True)
def _gene_sweep(eta,u,d,coupling,intercept,num,den,regions):
    _,_,_,gp,rk,gw=regions
    for g in np.random.permutation(len(d)):
        odds=intercept+eta[g]
        delta=u[g]*(coupling-1.)
        old=d[g]
        for e in range(gp[g],gp[g+1]):
            k,w=rk[e],gw[e]
            n0=num[k]-(w*delta if old else 0.)
            t0=den[k]-(delta if old else 0.)
            odds+=np.log1p(w*delta/n0)-np.log1p(delta/t0)
        new=np.random.random()<_logistic(odds)
        if new!=old:
            d[g]=new
            step=delta if new else -delta
            for e in range(gp[g],gp[g+1]):
                k=rk[e]
                num[k]+=gw[e]*step
                den[k]+=step


@njit(nogil=True)
def _transport_gene_states(eta,eta_new,d,intercept):
    # Refresh U conditional on the current D and beta; keep U fixed for the
    # proposed beta. Draw only when the threshold can change the gene state.
    out=d.copy()
    for g in range(len(d)):
        if eta[g]==eta_new[g]:
            continue
        p=_logistic(intercept+eta[g])
        q=_logistic(intercept+eta_new[g])
        if d[g] and q<p:
            out[g]=np.random.random()<q/p
        elif not d[g] and q>p:
            out[g]=np.random.random()<(q-p)/(1.-p)
    return out


@njit(nogil=True)
def _gene_summary(g,u,eta,d,coupling,intercept,num,den,regions):
    _,_,_,gp,rk,gw=regions
    odds=intercept+eta[g]
    miss0,miss1=1.,1.
    delta=u[g]*(coupling-1.)
    for e in range(gp[g],gp[g+1]):
        k,w=rk[e],gw[e]
        n0=num[k]-(w*delta if d[g] else 0.)
        t0=den[k]-(delta if d[g] else 0.)
        n1,t1=n0+w*delta,t0+delta
        odds+=np.log(n1/n0)+np.log(t0/t1)
        miss0*=max(0.,1.-w*u[g]/n0)
        miss1*=max(0.,1.-w*u[g]*coupling/n1)
    p=_logistic(odds)
    return p,1.-(1.-p)*miss0-p*miss1,1.-(1.-p)*miss0


@njit(nogil=True)
def sample(seed,initial,mat,pi,sd,regions,panel,coupling,intercept,steps,burn,thin,noncentered_probability):
    """Return beta traces, panel probabilities, gene means and acceptance counts.

    ``mat`` is annotations x candidate genes. Panel indices of -1 are absent
    from this mask; their probabilities should be computed as indirect-only
    from the sampled coefficients and their separate annotation rows.
    """
    np.random.seed(seed)
    beta=initial.copy()
    eta=beta@mat
    u=np.exp(eta)
    d=np.zeros(mat.shape[1],dtype=np.bool_)
    if coupling!=1.:
        for g in range(len(d)):
            d[g]=np.random.random()<_logistic(intercept+eta[g])
    odds=np.log(pi)-np.log1p(-pi)
    n=(steps-burn)//thin
    traces=np.empty((n,len(beta)))
    monitors=np.zeros((n,len(panel),3))
    sums=np.zeros((3,len(d)))
    attempts=np.zeros((2,2,len(beta)),dtype=np.int64)
    accepts=np.zeros_like(attempts)
    i=0
    for sweep in range(steps):
        num,den=_regional_sums(u,d,coupling,regions)
        if coupling!=1.:
            _gene_sweep(eta,u,d,coupling,intercept,num,den,regions)
        ll=_log_likelihood(num,den)
        gene_prior=_log_gene_prior(eta,d,intercept) if coupling!=1. else 0.
        for j in np.random.permutation(len(beta)):
            b=beta[j]
            if np.random.random()<.5:
                move=0
                candidate=np.random.normal()*sd[j] if b==0. else 0.
                offset=odds[j] if b==0. else -odds[j]
            elif b!=0.:
                move=1
                candidate=b+.4*sd[j]*np.random.normal()
                offset=-.5*((candidate/sd[j])**2-(b/sd[j])**2)
            else:
                continue
            joint=(coupling!=1. and noncentered_probability>0.
                   and np.random.random()<noncentered_probability)
            k=int(joint)
            attempts[k,move,j]+=1
            eta_new=eta+(candidate-b)*mat[j]
            d_new=_transport_gene_states(eta,eta_new,d,intercept) if joint else d
            u_new=np.exp(eta_new)
            nn,dd=_regional_sums(u_new,d_new,coupling,regions)
            cand_ll=_log_likelihood(nn,dd)
            prior_new=0.
            log_accept=cand_ll-ll+offset
            if coupling!=1. and not joint:
                prior_new=_log_gene_prior(eta_new,d_new,intercept)
                log_accept+=prior_new-gene_prior
            if np.log(np.random.random())<log_accept:
                beta[j]=candidate
                eta,u,d,num,den,ll=eta_new,u_new,d_new,nn,dd,cand_ll
                gene_prior=(_log_gene_prior(eta,d,intercept) if joint else prior_new)
                accepts[k,move,j]+=1
        if sweep>=burn and (sweep-burn+1)%thin==0:
            traces[i]=beta
            rows=np.empty((len(d),3))
            for g in range(len(d)):
                rows[g,0],rows[g,1],rows[g,2]=_gene_summary(g,u,eta,d,coupling,intercept,num,den,regions)
            for j in range(len(panel)):
                if panel[j]>=0:
                    monitors[i,j]=rows[panel[j]]
            sums+=rows.T
            i+=1
    return traces,monitors,sums/n,attempts,accepts


def direct_without_annotations(regions):
    """P(any assignment) for a fixed mask at beta=0, marginalizing gene states.

    This is the SNP-only reference with coupling disabled. In particular, it
    is not obtained by refitting the coupled model with beta fixed to zero.
    """
    ids,_,w,gene_ptr,_,_=regions
    miss=np.ones(len(gene_ptr)-1)
    np.multiply.at(miss,ids,1.-w)
    return 1.-miss


def project_gene_scores(beta, all_gene_mat, keep, candidate_means, snp_only, intercept):
    """Project onto the full gene universe; unlinked genes keep indirect support."""
    from scipy.special import expit

    indirect=np.zeros(all_gene_mat.shape[1])
    for start in range(0,len(beta),64):
        block=beta[start:start+64]
        active=np.any(block!=0.,axis=0)
        indirect+=expit(intercept+block[:,active]@all_gene_mat[active]).sum(axis=0)
    indirect/=len(beta)
    direct_snp=np.zeros_like(indirect)
    direct=np.zeros_like(indirect)
    involvement=indirect.copy()
    either=indirect.copy()
    direct_snp[keep]=snp_only
    involvement[keep],direct[keep],either[keep]=candidate_means
    return dict(direct_without_annotations=direct_snp,direct_with_annotations=direct,
                indirect=indirect,conditional_involvement=involvement,either=either)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path,help="NPZ with mat, ids, ptr, weights, pi, sd, initial and panel; optional all_gene_mat and keep")
    p.add_argument("output",type=Path)
    p.add_argument("--random-seed",type=int,required=True)
    p.add_argument("--steps",type=int,default=8000)
    p.add_argument("--burn",type=int,default=2000)
    p.add_argument("--thin",type=int,default=5)
    p.add_argument("--coupling",type=float,default=5.)
    p.add_argument("--involvement-baseline",type=float,default=.05)
    p.add_argument("--joint-move-probability",type=float,default=.75)
    args=p.parse_args(argv)
    if (not 0<=args.burn<args.steps or args.thin<=0 or (args.steps-args.burn)%args.thin
            or not 0<args.involvement_baseline<1 or not np.isfinite(args.coupling) or args.coupling<1
            or not 0<=args.joint_move_probability<=1 or not 0<=args.random_seed<2**32):
        p.error("invalid sampling settings")
    if args.output.exists():
        p.error("output already exists")
    with np.load(args.input) as f:
        mat=np.ascontiguousarray(f["mat"],dtype=float)
        initial,pi,sd=[np.ascontiguousarray(f[k],dtype=float) for k in ("initial","pi","sd")]
        panel=np.asarray(f["panel"],dtype=np.int64)
        if (mat.ndim!=2 or any(v.shape!=(mat.shape[0],) for v in (initial,pi,sd))
                or not all(np.isfinite(v).all() for v in (mat,initial,pi,sd))
                or np.any((pi<=0)|(pi>=1)) or np.any(sd<=0)
                or panel.ndim!=1 or np.any((panel< -1)|(panel>=mat.shape[1]))):
            p.error("invalid model arrays")
        regions=pack_regions(f["ids"],f["ptr"],f["weights"],mat.shape[1])
        if ("all_gene_mat" in f)!=("keep" in f):
            p.error("all_gene_mat and keep must be supplied together")
        if "all_gene_mat" in f:
            all_gene_mat=np.asarray(f["all_gene_mat"],dtype=float)
            keep=np.asarray(f["keep"],dtype=np.int64)
            if (all_gene_mat.ndim!=2 or all_gene_mat.shape[0]!=mat.shape[0] or not np.isfinite(all_gene_mat).all()
                    or keep.shape!=(mat.shape[1],) or len(np.unique(keep))!=len(keep)
                    or np.any((keep<0)|(keep>=all_gene_mat.shape[1]))
                    or not np.array_equal(all_gene_mat[:,keep],mat)):
                p.error("invalid all-gene projection arrays")
        else:
            all_gene_mat,keep=mat,np.arange(mat.shape[1])
    intercept=np.log(args.involvement_baseline)-np.log1p(-args.involvement_baseline)
    b,t,m,n,a=sample(args.random_seed,initial,mat,pi,sd,regions,panel,args.coupling,intercept,args.steps,
                     args.burn,args.thin,args.joint_move_probability)
    if not all(np.isfinite(v).all() for v in (b,t,m)):
        raise ArithmeticError("nonfinite model output")
    scores=project_gene_scores(b,all_gene_mat,keep,m,direct_without_annotations(regions),intercept)
    with args.output.open("xb") as f:
        np.savez_compressed(f,beta=b,panel_probabilities=t,gene_probabilities=m,attempts=n,accepts=a,
                            **scores)


if __name__=="__main__":
    main()
