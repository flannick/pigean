"""Scoped research adapter; retains production outer estimator and HuGE refresh.

This is NOT an exact joint posterior sampler. Marginal coefficient corrections
are retained in the uncorrected output path but are not a logistic likelihood.
The experimental corrected fit therefore explicitly omits them. Production
source/defaults are not changed. Panels are restricted by the screen driver.
"""
from contextlib import contextmanager
import inspect
import time
import numpy as np
from .model import LogisticTarget
from .sampler import sample
from .diagnostics import summarize


def conditional_prior(state,iteration,c):
    scales=iteration['full_scale_factors_m'][c]
    pi=iteration['full_ps_m'][c]
    original=iteration['full_sigma2s_m'][c]
    if state.sigma_power is not None:
        variance=state.get_scaled_sigma2(scales,original,state.sigma_power,state.sigma_threshold_k,state.sigma_threshold_xo)
        dense=iteration['full_is_dense_gene_set_m'][c]
        if np.any(dense):
            ref=np.mean(scales[~dense]) if np.any(~dense) else np.mean(scales)
            variance[dense]=state.get_scaled_sigma2(ref,state.sigma2,state.sigma_power,state.sigma_threshold_k,state.sigma_threshold_xo)
    else:variance=original.copy()
    return pi,variance/pi


@contextmanager
def outer_backend(*,method='legacy',swaps=0,sweeps=120,burn=40,seed=117,records=None,capture=None):
    import pigean.state as module
    original=module._run_gibbs_corrected_betas_step
    prepare=module._prepare_gibbs_iteration_inputs
    signature=inspect.signature(prepare)
    info={} if records is None else records
    rng=np.random.default_rng(seed)
    offsets=None
    def wrapped_prepare(*args,**kwargs):
        nonlocal offsets
        bound=signature.bind(*args,**kwargs)
        offsets=np.array(bound.arguments['iteration_input_config']['cur_background_log_bf_v'],copy=True)
        return prepare(*args,**kwargs)
    def wrapped_step(state,iter_state,gene_set_mask_m,epoch_priors,iteration_update_config,inner_beta_kwargs):
        # The selected research panel stays fixed in every arm.
        mask=np.ones_like(gene_set_mask_m,dtype=bool)
        if capture is not None and not info.get('captured'):
            capture(state,iter_state,offsets)
            info['captured']=True
        if method=='legacy':
            return original(state,iter_state,mask,epoch_priors,iteration_update_config,inner_beta_kwargs)
        n,p=state.X_orig.shape
        if p>64:raise ValueError('Reduced-panel adapter limited to 64 annotations')
        X=state.X_orig.toarray()
        rows=[];samples=[];means=[];sample_probs=[];mean_probs=[]
        started=time.perf_counter()
        for c,y in enumerate(iter_state['p_sample_m']):
            design=(X-iter_state['full_mean_shifts_m'][c])/iter_state['full_scale_factors_m'][c]
            pi,tau=conditional_prior(state,iter_state,c)
            offset=float(offsets.ravel()[c] if offsets.size>1 else offsets.ravel()[0])
            target=LogisticTarget(design,y,np.full(n,offset),pi,tau)
            initial=epoch_priors['prev_warm_start_betas_m']
            initial=None if initial is None else np.tile(initial[c],(4,1))
            draws=sample(target,sweeps=sweeps,burn=burn,chains=4,seed=int(rng.integers(0,2**31)),
                         initial=initial,collapse=method=='collapsed',swaps=swaps)
            rows.append(summarize(draws));samples.append(draws[-1,0]);means.append(draws.mean(axis=(0,1)))
            sample_probs.append((draws[-1,0]!=0).astype(float));mean_probs.append((draws!=0).mean(axis=(0,1)))
        info.setdefault('inner',[]).append(dict(iteration=iter_state['total_iter_num'],seconds=time.perf_counter()-started,chains=rows))
        return np.array(samples),np.array(sample_probs),np.array(means),np.array(mean_probs)
    module._prepare_gibbs_iteration_inputs=wrapped_prepare
    module._run_gibbs_corrected_betas_step=wrapped_step
    try:yield info
    finally:
        module._prepare_gibbs_iteration_inputs=prepare
        module._run_gibbs_corrected_betas_step=original
