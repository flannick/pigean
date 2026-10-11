"""HuGE signal allocations before odds conversion, independent of target priors."""
from __future__ import annotations
import numpy as np


def signal_gene_allocations(var_gene_index, var_offset_prob, full_prob, gene_name_to_index):
    full=np.asarray(full_prob,dtype=float)
    total=full.sum()
    if total<=0: return {}
    linkage=np.divide(var_offset_prob,full[None,:],out=np.zeros_like(var_offset_prob),where=full[None,:]>0)
    linkage/=np.maximum(linkage.sum(axis=0),1)[None,:]
    allocation=linkage*(full/total)[None,:]
    names={i:g for g,i in gene_name_to_index.items()}
    valid=var_gene_index.ravel()>=0
    totals=np.bincount(var_gene_index.ravel()[valid],weights=allocation.ravel()[valid])
    return {names[int(i)]:float(totals[i]) for i in np.flatnonzero(totals) if int(i) in names}


def variant_metadata(columns,header,chrom,pos,*,build='',beta=None,freq=None,effect_col=None,other_col=None):
    lower=header if isinstance(header,dict) else {str(h).lower():i for i,h in enumerate(header)}
    def get(names):
        for name in names:
            if name in lower and lower[name]<len(columns): return columns[lower[name]]
        return ''
    ref=get(['reference','ref','other_allele','a2','allele0'])
    alt=get(['alt','effect_allele','a1','allele1'])
    if effect_col is not None:alt=columns[effect_col]
    if other_col is not None:ref=columns[other_col]
    if not ref or not alt:
        identity=get(['varid','variant_id','variant','snp','locus'])
        bits=identity.replace('_',':').split(':')
        if len(bits)==4: ref,alt=bits[2:]
    return dict(chrom=str(chrom),pos=int(pos),ref=ref.upper(),alt=alt.upper(),build=build or '',beta=beta,frequency=freq)
