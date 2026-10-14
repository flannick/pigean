"""Bounded diagnostics; independent reference error accompanies classical R-hat."""
import numpy as np


def split_rhat(draws):
    n=len(draws)//2
    x=np.concatenate((draws[:n],draws[-n:]),axis=1)
    W=x.var(axis=0,ddof=1).mean(axis=0)
    B=n*x.mean(axis=0).var(axis=0,ddof=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.sqrt(((n-1)*W/n+B/n)/W)


def summarize(draws,reference=None):
    indicators=draws!=0
    result=dict(mean=draws.mean(axis=(0,1)),inclusion=indicators.mean(axis=(0,1)),
                beta_rhat=split_rhat(draws),inclusion_rhat=split_rhat(indicators.astype(float)),
                per_chain_inclusion=indicators.mean(axis=0),
                transitions=np.sum(np.any(indicators[1:]!=indicators[:-1],axis=2),axis=0))
    # Batch means includes positive temporal correlation; not an ESS certificate.
    count=20;size=len(draws)//count
    if size>1:
        batches=draws[:count*size].reshape(count,size,*draws.shape[1:]).mean(axis=1)
        result['mean_mcse_batch']=batches.reshape(-1,draws.shape[-1]).std(axis=0,ddof=1)/np.sqrt(count*draws.shape[1])
    if reference is not None:
        w=np.array([np.mean(np.all(indicators==s,axis=2)) for s in reference['supports']])
        result.update(support_weights=w,support_tv=float(np.abs(w-reference['weights']).sum()/2),
                      max_mean_error=float(np.max(abs(result['mean']-reference['mean']))),
                      max_inclusion_error=float(np.max(abs(result['inclusion']-reference['inclusion']))))
    return result


def clean(value):
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    return value
