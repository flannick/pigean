"""Compare reduced-panel estimator screens and prepare biological review tables."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def metrics(a,b):
    x=np.asarray(a,float);y=np.asarray(b,float);ok=np.isfinite(x)&np.isfinite(y)
    x=x[ok];y=y[ok]
    return dict(n=len(x),correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)*np.std(y)>0 else None,
        slope=float(np.polyfit(x,y,1)[0]) if np.std(x)>0 else None,
        intercept=float(np.polyfit(x,y,1)[1]) if np.std(x)>0 else None,
        rmse=float(np.sqrt(np.mean((x-y)**2))),max_abs_difference=float(np.max(abs(x-y))))


def rhat(draws):
    n=len(draws)//2;x=np.concatenate((draws[:n],draws[-n:]),axis=1)
    W=x.var(axis=0,ddof=1).mean(axis=0);B=n*x.mean(axis=0).var(axis=0,ddof=1)
    with np.errstate(divide='ignore',invalid='ignore'):return np.sqrt(((n-1)*W/n+B/n)/W)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();paths=[Path(args.baseline),Path(args.candidate)]
    receipts=[json.load(gzip.open(d/'screen.json.gz','rt')) for d in paths]
    for r in receipts:
        if r['status']!='COMPLETE_REDUCED_SCREEN':raise ValueError('Incomplete screen')
    a,b=receipts
    for key in ('fixture_sha256','kwargs','gene_sets','updates'):
        if a[key]!=b[key]:raise ValueError('Unmatched '+key)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    result=dict(scope='REDUCED_PANEL_FEASIBILITY_NOT_NONINFERIORITY',commits=[r['commit'] for r in receipts],
        updates=a['updates'],inference_seconds=[r['inference_seconds'] for r in receipts],
        speedup=a['inference_seconds']/b['inference_seconds'],peak_rss_bytes=[r['peak_rss_bytes'] for r in receipts],
        rss_change_percent=100*(b['peak_rss_bytes']/a['peak_rss_bytes']-1),metrics={},review={})
    for filename,idcol,fields in (('gene_stats.out.gz','Gene',['prior','log_bf','combined','combined_D']),
                                  ('gene_set_stats.out.gz','Gene_Set',['beta','beta_uncorrected'])):
        frames=[pd.read_csv(d/filename,sep='\t',compression='gzip').set_index(idcol) for d in paths]
        if any(f.index.duplicated().any() for f in frames):raise ValueError('Duplicate IDs')
        if set(frames[0].index)!=set(frames[1].index):raise ValueError('Mismatched IDs')
        base=frames[0];new=frames[1].reindex(base.index)
        for field in fields:
            x=base[field];y=new[field];result['metrics'][field]=metrics(x,y)
            review=pd.DataFrame({'baseline':x,'candidate':y,'difference':y-x,
                'baseline_rank':x.rank(ascending=False,method='min'),'candidate_rank':y.rank(ascending=False,method='min')})
            review['rank_gain']=review['baseline_rank']-review['candidate_rank']
            review.sort_values('candidate_rank').to_csv(out/(field+'_rank_changes.tsv.gz'),sep='\t',compression='gzip')
            for k in (20,200):
                k=min(k,len(x));result['metrics'][field]['top'+str(k)+'_overlap']=len(set(x.nlargest(k).index)&set(y.nlargest(k).index))/k
            result['review'][field]=dict(largest_absolute_changes=review.loc[review['difference'].abs().nlargest(15).index].reset_index().to_dict('records'),
                                        candidate_top=review.sort_values('candidate_rank').head(15).reset_index().to_dict('records'))
    result['last_diagnostics']=[r.get('last_diagnostics') for r in receipts]
    traces=[np.load(d/'iteration_arrays.npz',allow_pickle=True) for d in paths]
    if not np.array_equal(traces[0]['gene_sets'],traces[1]['gene_sets']):raise ValueError('Trace IDs differ')
    rhats=[rhat(t['beta'][10:]) for t in traces]
    common=pd.DataFrame({'Gene_Set':traces[0]['gene_sets'],'baseline_split_rhat':rhats[0],'candidate_split_rhat':rhats[1]})
    common.to_csv(out/'fixed_panel_rhat.tsv.gz',sep='\t',compression='gzip',index=False)
    result['fixed_panel_diagnostics']=dict(scope='All selected annotations; last ten outer mean-beta states; five draws per split',
        above_1_25=[int(np.sum(v>1.25)) for v in rhats],undefined=[int(np.sum(~np.isfinite(v))) for v in rhats],
        values=common.to_dict('records'))
    result['biological_verdict']='Pending expert inspection and agreed independent positive/negative controls'
    with gzip.open(out/'comparison.json.gz','wt') as f:json.dump(clean(result),f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('updates','speedup','rss_change_percent','metrics')},indent=2))


if __name__=='__main__':main()
