"""Independent reference diagnostics and adapters to literal production updates.

These bounded experiments do not alter the production target or controller.
"""
from itertools import product
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import expit, logsumexp
from .quadratic import QuadraticTarget, enumerate_posterior


def gaussian_scan_stationary(beta_tilde, se, V, tau, order):
    """Exact stationary law of all-slab sequential linear Gaussian updates."""
    p=len(se); A=np.eye(p); b=np.zeros(p); noise=np.zeros((p,p))
    shrink=tau/(tau+se**2)
    for j in order:
        step=np.eye(p);step[j]=-shrink[j]*V[:,j];step[j,j]=0
        c=np.zeros(p);c[j]=shrink[j]*beta_tilde[j]
        n=np.zeros((p,p));n[j,j]=shrink[j]*se[j]**2
        A,b,noise=step@A,step@b+c,step@noise@step.T+n
    radius=float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius>=1:raise ValueError('Unstable Gaussian scan')
    mean=np.linalg.solve(np.eye(p)-A,b)
    cov=solve_discrete_lyapunov(A,noise)
    return dict(mean=mean,covariance=cov,spectral_radius=radius,
                invariance_error=float(np.max(np.abs(cov-A@cov@A.T-noise))))


def split_rhat(values):
    """Classical split R-hat; independent truth errors are reported alongside it."""
    n=values.shape[0]//2
    x=np.concatenate([values[:n],values[-n:]],axis=1)
    W=np.mean(np.var(x,axis=0,ddof=1),axis=0)
    B=n*np.var(np.mean(x,axis=0),axis=0,ddof=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.sqrt(((n-1)*W/n+B/n)/W)


def summarize_draws(draws,reference):
    supports=draws!=0
    codes=np.sum(supports*(1<<np.arange(draws.shape[-1])),axis=-1)
    weights=np.bincount(codes.ravel(),minlength=len(reference['weights']))/codes.size
    means=np.mean(draws,axis=(0,1))
    return dict(support_weights=weights,support_tv=float(np.sum(np.abs(weights-reference['weights']))/2),
                inclusion=np.mean(supports,axis=(0,1)),mean=means,
                max_mean_error=float(np.max(np.abs(means-reference['mean']))),
                beta_rhat=split_rhat(draws),support_rhat=split_rhat(supports.astype(float)),
                per_chain_inclusion=np.mean(supports,axis=0),
                support_changes_per_chain=np.sum(codes[1:]!=codes[:-1],axis=0))


def production_draws(beta_tilde,se,V,pi,tau,*,sweeps=10000,burn=2000,chains=8,seed=117,
                     batching='sequential',sparsify=False):
    """Use production's actual update function; fixed nuisance parameters/budget.

    Controller early stopping, restarts and hyperparameter updates are excluded.
    'production' batching calls the actual default partition builder.
    """
    from pigean.state import (PigeanState,_update_inner_beta_gene_set_batch,
                              _apply_inner_beta_sparsity_update)
    p=len(se); rng=np.random.default_rng(seed)
    sample=np.zeros((chains,1,p));sample[:chains//2,0,0]=beta_tilde[0]
    sample[chains//2:,0,-1]=beta_tilde[-1]
    pm=np.zeros_like(sample);prob=np.zeros_like(sample);res=np.zeros_like(sample)
    tau=np.asarray(tau)[None,:];se2=se[None,:]**2;shrink=tau/(tau+se2)
    if batching=='production':
        masks=PigeanState()._compute_gene_set_batches(V,use_sum=True)
    else:
        order=range(p) if batching=='sequential' else range(p-1,-1,-1)
        masks=[np.arange(p)==j for j in order]
    out=[]
    for iteration in range(sweeps):
        u=rng.random(sample.shape);z=rng.normal(size=sample.shape)
        for mask in masks:
            _update_inner_beta_gene_set_batch(compute_mask_m=np.asarray(mask)[None,:],
                compute_mask_v=np.array([True]),alpha_shrink=1.,rand_ps_t=u,rand_norms_t=z,
                hdmp_hdmpn_m=shrink,c_const_m=pi[None,:]/np.sqrt(tau+se2),
                d_const_m=(1-pi[None,:])/se[None,:],hdmpn_m=tau+se2,se2s_m=se2,
                norm_scale_m=np.sqrt(shrink*se2),assume_independent=False,
                beta_tildes_m=beta_tilde[None,:],sampled_beta_t=sample,V=V,
                multiple_V=False,sparse_V=False,use_X=False,X_orig=None,
                scale_factors_m=np.ones((1,p)),mean_shifts_m=np.zeros((1,p)),
                betas_trace_out=None,betas_trace_gene_sets=None,account_for_V_diag_m=False,
                V_diag_m=np.ones((1,p)),posterior_nonzero_prob_t=prob,
                posterior_beta_mean_t=pm,gauss_seidel=False,residual_beta_tilde_t=res)
        if sparsify:
            _apply_inner_beta_sparsity_update(True,None,prob,pi[None,:],pm,sample,np.array([True]))
        if iteration>=burn:out.append(sample[:,0].copy())
    return np.asarray(out),[np.flatnonzero(m).tolist() for m in masks]


def support_transition(weights,*,swap=False):
    """Random-site collapsed heat bath, optionally followed by a MH support swap.

    Full exact integration of Gaussian coefficients; no approximate weights.
    """
    n=len(weights); p=int(np.log2(n));T=np.zeros((n,n))
    for a in range(n):
        for j in range(p):
            b=a^(1<<j);prob=weights[b]/(weights[a]+weights[b])
            T[a,b]+=prob/p;T[a,a]+=(1-prob)/p
    if swap:
        if p!=2:raise ValueError('Bounded swap reference only supports two coefficients')
        S=np.eye(n)
        for a,b in ((1,2),(2,1)):
            acc=min(1.,weights[b]/weights[a]);S[a,a]=1-acc;S[a,b]=acc
        T=T@S
    return T


def transition_metrics(T,weights):
    eig=np.sort(np.abs(np.linalg.eigvals(T)))
    return dict(stationarity_error=float(np.max(np.abs(weights@T-weights))),
                spectral_gap=float(1-eig[-2]),
                worst_start_tv_500=float(np.max(np.sum(np.abs(np.linalg.matrix_power(T,500)-weights),axis=1)/2)))


def collapsed_draws(target,*,sweeps=10000,burn=2000,chains=8,seed=117,swap=True):
    ref=enumerate_posterior(target);T=support_transition(ref['weights'],swap=swap)
    rng=np.random.default_rng(seed);states=np.array([1]*(chains//2)+[2]*(chains-chains//2))
    mus=[];chols=[]
    for support in ref['supports']:
        _,mu,cov=target.support_moments(support);mus.append(mu)
        # Zero rows remain exactly zero for inactive coefficients.
        w,U=np.linalg.eigh(cov);chols.append(U*np.sqrt(np.maximum(w,0)))
    mus=np.array(mus);chols=np.array(chols);out=[]
    for it in range(sweeps):
        states=np.sum(rng.random(chains)[:,None]>np.cumsum(T[states],axis=1),axis=1)
        states=np.minimum(states,len(T)-1)
        draws=mus[states]+np.einsum('cij,cj->ci',chols[states],rng.normal(size=(chains,target.size)))
        if it>=burn:out.append(draws)
    return np.asarray(out),transition_metrics(T,ref['weights'])


def finite_outer_reference(design,offset,log_bf,*,levels=(-1.,0.,1.),inclusion=.2):
    """Exactly enumerable discrete-slab logistic model, not production's continuous slab."""
    design=np.asarray(design);n,p=design.shape
    beta=np.array(list(product(levels,repeat=p)));D=np.array(list(product([0.,1.],repeat=n)))
    prior=np.prod(np.where(beta==0,1-inclusion,inclusion/2),axis=1)
    eta=beta@design.T+offset
    logistic=eta@D.T-np.logaddexp(0,eta).sum(axis=1)[:,None]
    logjoint=logistic+np.log(prior)[:,None]+(D@log_bf)[None,:]
    joint=np.exp(logjoint-logsumexp(logjoint))
    bgd=np.exp((logistic+np.log(prior)[:,None])-logsumexp(logistic+np.log(prior)[:,None],axis=0))
    dgb=joint/joint.sum(axis=1)[:,None]
    return dict(beta=beta,D=D,joint=joint,beta_given_D=bgd,D_given_beta=dgb,eta=eta,
                design=design,offset=offset)


def conditional_cycle_error(beta_given_D,D_given_beta):
    """Positive finite conditionals are compatible iff log-ratio is additive.

    log B(b|d)-log A(d|b)=log p(b)-log p(d); mixed differences must vanish.
    """
    r=np.log(beta_given_D)-np.log(D_given_beta)
    interaction=r-r[:,[0]]-r[[0],:]+r[0,0]
    return float(np.max(np.abs(interaction)))


def finite_outer_metrics(ref,D_given_beta):
    """Exact D-sweep transition, its stationary law and resulting joint output."""
    B=ref['beta_given_D'];T=B.T@D_given_beta
    n=len(T);A=T.T-np.eye(n);A[-1]=1.;b=np.zeros(n);b[-1]=1
    stationary=np.linalg.solve(A,b)
    output=B*stationary[None,:]
    return dict(cycle_error=conditional_cycle_error(B,D_given_beta),
                D_stationarity_error=float(np.max(np.abs(stationary@T-stationary))),
                joint_tv_from_fixed_evidence=float(np.sum(np.abs(output-ref['joint']))/2),
                max_D_probability_error=float(np.max(np.abs(stationary@ref['D']-ref['joint'].sum(axis=0)@ref['D']))),
                stationary_D=stationary)


def mean_feedback_metrics(ref,log_bf):
    """Exact transition for plugging E[beta|D] into the next logistic D draw."""
    mu=ref['beta_given_D'].T@ref['beta']
    eta=mu@ref['design'].T+ref['offset']+log_bf
    T=np.exp(eta@ref['D'].T-np.logaddexp(0,eta).sum(axis=1)[:,None])
    n=len(T);A=T.T-np.eye(n);A[-1]=1.;b=np.zeros(n);b[-1]=1
    stationary=np.linalg.solve(A,b)
    output=ref['beta_given_D']*stationary[None,:]
    return dict(joint_tv_from_fixed_evidence=float(np.sum(np.abs(output-ref['joint']))/2),
                stationarity_error=float(np.max(np.abs(stationary@T-stationary))),
                stationary_D=stationary,
                max_D_probability_error=float(np.max(np.abs(stationary@ref['D']-ref['joint'].sum(axis=0)@ref['D']))))


def production_huge_conditionals(ref):
    """Actual HuGE competition/max and D-probability functions on two genes.

    Conditional target audit only. No new HuGE model; fixed q is never inferred.
    This does not reproduce full pipeline calibration/filtering.
    """
    from scipy import sparse
    from pigean.state import PigeanState,_compute_gibbs_iteration_y_terms
    state=PigeanState(background_prior=.05);state.genes=['a','b']
    evidence=sparse.csc_matrix([[9.,1.],[1.,9.]])
    q=np.array([.6,.8]);before=q.copy();terms=ref['beta']@ref['design'].T
    def distill(indices):
        return state._distill_huge_signal_bfs(evidence[:,indices],q[indices],
            np.ones(len(indices)),np.ones(len(indices)),.9,True,False,False,False,
            None,None,None,None,None,state.genes,rel_prior_log_bf=terms)[0]
    both=distill([0,1]);single=np.maximum(distill([0]),distill([1]))
    probs=_compute_gibbs_iteration_y_terms(terms,both,both.copy(),
              np.full(len(terms),float(ref['offset'][0])))['D_sample_m']
    eta=np.log(probs)-np.log1p(-probs)
    dgb=np.exp(eta@ref['D'].T-np.logaddexp(0,eta).sum(axis=1)[:,None])
    return dgb,dict(signal_probabilities_unchanged=bool(np.array_equal(before,q)),
            strongest_signal_error=float(np.max(np.abs(both-single))),
            duplicate_signal_error=float(np.max(np.abs(both-distill([0,1,0,1])))),
            order_error=float(np.max(np.abs(both-distill([1,0])))),
            log_bf_range_across_priors=np.ptp(both,axis=0),
            probability_formula_error=float(np.max(np.abs(probs-expit(ref['eta']+both)))))
