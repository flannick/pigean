"""Target-independent construction and native/PLINK/PGS-style export."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from scipy import sparse
from .data import Weights,Variant,Source,fetch_selected,text_open,number,warn


def factor_basis(path=None,state=None):
    if path:
        # Reuse EAGGL's canonical raw Factor1..FactorK loading reader.
        from ..label import _read_wide_loading_table
        table=_read_wide_loading_table(path,kind='gene',id_column=None,id_column_candidates=['Gene','gene','Name','name'])
        return table.ids,table.factor_columns,table.loadings
    if state is None or getattr(state,'exp_gene_factors',None) is None:
        raise ValueError('SNP weights require gene factors or --factor-gene-clusters-in')
    matrix=np.asarray(state.exp_gene_factors)
    genes=list(state.genes)
    if len(genes)!=matrix.shape[0]:
        mask=getattr(state,'gene_factor_gene_mask',None)
        if mask is None or int(np.sum(mask))!=matrix.shape[0]: raise ValueError('Gene factor basis/IDs disagree')
        genes=list(np.asarray(genes)[mask])
    return genes,[f'Factor{i+1}' for i in range(matrix.shape[1])],matrix


def effector_records(paths,weight_column='signal_gene_weight',state=None):
    if not paths:
        records=getattr(state,'huge_signal_gene_records',None)
        if not records: raise ValueError('Provide --snp-factor-gene-effectors-in (PIGEAN gene-effectors-out)')
        for record in records:
            i=record['identity']; v=Variant(i['chrom'],int(i['pos']),i['ref'],i['alt'])
            for gene,weight in record['weights'].items():
                yield v,gene,float(weight),number(i.get('frequency')),i.get('build',''),record['signal_id'],'signal_gene_weight'
        return
    for path in paths:
        with text_open(path) as fh:
            reader=csv.DictReader(fh,delimiter='\t')
            column=weight_column
            if column not in (reader.fieldnames or []):
                if column=='signal_gene_weight' and 'cond_prob_huge' in (reader.fieldnames or []):
                    column='cond_prob_huge'; warn(f'{path}: legacy cond_prob_huge is a normalized evidence weight, not signal_gene_weight')
                else: raise ValueError(f'Missing effector weight column {column}')
            for row in reader:
                value=number(row.get(column))
                if value is None: raise ValueError(f'{path}: missing {column}; regenerate effectors or explicitly choose a legacy weight')
                identity=row.get('lead_variant') or row.get('Lead_locus')
                try: v=Variant.parse(identity)
                except (ValueError,TypeError): raise ValueError('Effectors need allele-resolved lead_variant; regenerate with current PIGEAN or annotate legacy leads')
                if value<0 or value>1: raise ValueError('Effector allocation must lie in [0,1]')
                build=row.get('genome_build','')
                if build=='NA': build=''
                yield v,row['Gene'],value,number(row.get('reference_frequency')),build,row.get('signal_id',v.id),column


def construct(options,state=None):
    genes,factors,matrix=factor_basis(getattr(options,'factor_gene_clusters_in',None),state)
    matrix=sparse.csr_matrix(matrix)
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data<0): raise ValueError('Factor loadings must be finite and nonnegative')
    gene_index={g:i for i,g in enumerate(genes)}
    if len(gene_index)!=len(genes):raise ValueError("Duplicate gene IDs in factor basis")
    variants=[]; lookup={}; rr=[]; cc=[]; dd=[]; freq=[]; builds=set(); seen={}; signals={}
    methods=set();allocations={}
    for v,g,q,f,build,signal,method in effector_records(getattr(options,'snp_factor_gene_effectors_in',None),getattr(options,'snp_factor_gene_weight','signal_gene_weight'),state):
        methods.add(method)
        if build: builds.add(build)
        key=(v,g)
        if key in seen:
            if seen[key]!=q: raise ValueError('Conflicting effector weights across sources; choose one mapping per signal')
            continue
        seen[key]=q
        allocations[v]=allocations.get(v,0)+q
        if v not in lookup:
            lookup[v]=len(variants); variants.append(v); freq.append(np.nan if f is None else f)
        signals.setdefault(v.id,[]).append(signal)
        if g in gene_index:
            rr.append(lookup[v]);cc.append(gene_index[g]);dd.append(q)
    if methods=={"signal_gene_weight"} and any(v>1+1e-8 for v in allocations.values()):raise ValueError("Signal gene allocations exceed unit mass")
    if len(builds)>1: raise ValueError('Mixed genome builds in effector inputs')
    build=getattr(options,'snp_factor_genome_build',None) or next(iter(builds),'')
    if builds and build not in builds: raise ValueError('Requested genome build disagrees with effectors')
    a=sparse.csr_matrix((dd,(rr,cc)),shape=(len(variants),len(genes)))@matrix
    # Keep only variants belonging to at least one factor; preserve complete factor roster.
    keep=np.asarray(a.sum(axis=1)).ravel()>0
    variants=[v for v,k in zip(variants,keep) if k]; a=a[keep].tocsr();freq=np.array(freq)[keep]
    if not len(variants): raise ValueError('No overlap between gene effectors and factor loadings')
    meta=dict(mode='unsigned',build=build,gene_weight=','.join(sorted(methods)),
        factor_basis=getattr(options,'factor_gene_clusters_in',None) or 'fitted_gene_factors',
        effector_sources=getattr(options,'snp_factor_gene_effectors_in',None) or ['runtime'],signals=signals,
        training_sources=getattr(options,'snp_factor_training_source',None) or [],normalization='membership',
        projection_method=getattr(options,'full_gene_projection_method',None))
    return Weights(variants,factors,a,a.copy(),freq,meta).validate()


def apply_anchor(weights,source,mode,ld):
    # An anchor only supplies orientation/effect; it does not select significance thresholds.
    coordinates=[v.coordinate for v in weights.variants]
    source=Source(source)
    if weights.metadata.get('build') and source.build and weights.metadata['build']!=source.build: raise ValueError('Anchor build mismatch')
    multiplier=np.zeros(len(weights.variants)); freqs=weights.frequencies.copy(); missing=[]
    with fetch_selected(source,coordinates) as records:
        for i,v in enumerate(weights.variants):
            f=ld.frequencies.get(v.id,freqs[i])
            f=None if not np.isfinite(f) else float(f)
            independent_frequency=f is not None
            r=records.lookup(v,signed=False,require_stat=False)
            if f is None and r and r['freq'] is not None and not r.get('freq_is_maf'):
                f=r['freq']
            if not r or r['beta'] in (None,0) or f is None or not 0<f<1:
                missing.append(v.id);continue
            freqs[i]=f
            # The anchor need not have a valid SE: only a finite, nonzero effect is required.
            r=records.lookup(v,signed=True,freq=f,require_stat=False,require_se=False)
            if {v.ref,v.alt} in ({'A','T'},{'C','G'}) and not independent_frequency:r=None
            if not r or r['beta'] in (None,0): missing.append(v.id);continue
            multiplier[i]=np.sign(r['beta']) if mode=='sign' else r['beta']*np.sqrt(2*f*(1-f))
    metadata=dict(weights.metadata,mode=mode,normalization='standardized_genotype',anchor=source.source,anchor_missing=missing,
                  frequency_reference=ld.provenance if ld.frequencies else 'effector_or_anchor_frequency_approximation')
    return Weights(weights.variants,weights.factors,weights.membership,
                   weights.membership.multiply(multiplier[:,None]).tocsr(),freqs,metadata).validate()


def export_scores(weights,plink=None,pgs=None):
    if not plink and not pgs:return
    if weights.metadata.get('mode')=='unsigned': raise ValueError('PLINK/PGS linear exports require --snp-factor-sign-anchor or --snp-factor-effect-weight-anchor; use native output for unsigned weights')
    weights.validate()
    if not weights.metadata.get('build'): raise ValueError('Standard score exports require --snp-factor-genome-build or documented input build')
    sd=np.sqrt(2*weights.frequencies*(1-weights.frequencies))
    inverse_sd=np.divide(1,sd,out=np.zeros_like(sd),where=np.isfinite(sd)&(sd>0))
    dosage=weights.coefficients.multiply(inverse_sd[:,None]).tocsr()
    offset=-np.asarray(dosage.T@(2*np.nan_to_num(weights.frequencies))).ravel()
    metadata=dict(weights.metadata,factors=weights.factors,centering_offset=offset.tolist(),
                  scoring='SUM(effect_allele_dosage * weight) + centering_offset; do not average; do not variance-standardize dosage again',
                  frequencies=[float(f) if np.isfinite(f) else None for f in weights.frequencies],variants=[v.id for v in weights.variants])
    if plink:
        with text_open(plink,'wt') as fh:
            writer=csv.writer(fh,delimiter='\t');writer.writerow(['ID','A1']+weights.factors)
            for i,v in enumerate(weights.variants):
                if dosage.getrow(i).nnz:writer.writerow([v.id,v.alt]+[repr(float(x)) for x in dosage.getrow(i).toarray().ravel()])
        Path(str(plink)+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if pgs:
        manifest=[]
        for j,factor in enumerate(weights.factors):
            name=re.sub('[^A-Za-z0-9_.-]','_',factor)
            path=f'{pgs}.{j+1}.{name}.txt.gz'
            column=dosage.getcol(j).tocoo()
            with text_open(path,'wt') as fh:
                fh.write(f'#format_version=2.0\n#pgs_name=EAGGL_{name}\n#genome_build={weights.metadata["build"]}\n#weight_type=linear_coefficient\n#variants_number={column.nnz}\n#source=EAGGL; not a PGS Catalog registration\n')
                writer=csv.writer(fh,delimiter='\t');writer.writerow(['chr_name','chr_position','effect_allele','other_allele','effect_weight'])
                for i,value in zip(column.row,column.data):
                    v=weights.variants[i];writer.writerow([v.chrom,v.pos,v.alt,v.ref,repr(float(value))])
            manifest.append(dict(factor=factor,path=path,centering_offset=float(offset[j])))
        Path(str(pgs)+'.manifest.json').write_text(json.dumps(dict(metadata,files=manifest),indent=2)+'\n')
