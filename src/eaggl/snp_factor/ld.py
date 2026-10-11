"""LD references and target-independent proxy selection (no target P ranking)."""
from __future__ import annotations
import csv
import math
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .data import Variant, text_open, number, allele_sign, warn


@dataclass
class LD:
    variants: dict = field(default_factory=dict)
    frequencies: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)  # canonical pair -> (signed r or None, r2)
    proxies: dict = field(default_factory=dict)
    ancestry: str = 'European'
    provenance: str = 'independence_approximation'
    memory_mb: float = 512

    def check_budget(self):
        # Conservative allowance for Python dictionaries, tuples and adjacency lists.
        pairs=sum(len(v) for v in self.proxies.values())
        if len(self.variants)*1024+pairs*1024 > self.memory_mb*1024**2/4:
            raise MemoryError('LD metadata exceeds its memory allocation; supply a regional reference or increase --snp-factor-memory-mb')

    def pair(self,a,b):
        if a==b: return 1.,1.
        return self.edges.get(tuple(sorted((a,b))), (0.,0.))

    def candidates(self,variant,min_r2=.8,window=1000000):
        items=[]
        for other,r,r2,rank in self.proxies.get(variant.id,[]):
            v=self.variants.get(other)
            if v is None: continue
            if v.chrom != variant.chrom or abs(v.pos-variant.pos)>window: continue
            if r2 is not None and r2<min_r2: continue
            key=(0,-r2,abs(v.pos-variant.pos),v.id) if r2 is not None else (1,rank,abs(v.pos-variant.pos),v.id)
            items.append((key,v,r,r2))
        return [(v,1.,1.) for v in [variant]]+[(v,r,r2) for _,v,r,r2 in sorted(items)]

    def blocks(self,variants,memory_mb=512):
        n=len(variants); lookup={v.id:i for i,v in enumerate(variants)}
        rows=[]; cols=[]
        for (a,b),(r,r2) in self.edges.items():
            if a in lookup and b in lookup and r2:
                rows.extend([lookup[a],lookup[b]]); cols.extend([lookup[b],lookup[a]])
        graph=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n))
        _,labels=connected_components(graph,directed=False)
        for label in np.unique(labels):
            ix=np.flatnonzero(labels==label)
            if len(ix)**2*8*8 > memory_mb*1024**2:
                raise MemoryError('One LD block exceeds memory budget; provide a smaller reference block or increase --snp-factor-memory-mb')
            R=np.eye(len(ix)); R2=np.eye(len(ix)); signed=True
            for i in range(len(ix)):
                for j in range(i):
                    r,r2=self.pair(variants[ix[i]].id,variants[ix[j]].id)
                    R2[i,j]=R2[j,i]=r2
                    if r is None and r2>0: signed=False
                    else: R[i,j]=R[j,i]=r or 0.
            if signed:
                eig=np.linalg.eigvalsh(R)
                if eig.min() < -1e-7: raise ValueError('LD correlation block is not positive semidefinite; check allele orientation and reference consistency')
            yield ix,R,R2,signed

    def add_edge(self,a,b,r,r2,rank=0):
        if a==b: return
        if r2 is not None and not 0<=r2<=1+1e-8: raise ValueError('Invalid LD r2')
        if r is not None:
            if abs(r)>1+1e-8: raise ValueError('Invalid LD r')
            if r2 is not None and abs(r*r-r2)>1e-3: raise ValueError('Inconsistent r and r2')
            r2=r*r
        pair=tuple(sorted((a,b)))
        existing=self.edges.get(pair)
        if existing is not None:
            if existing!=(r,r2): raise ValueError('Conflicting duplicate LD pair')
            return
        if r2 is None and any(v[0]==b for v in self.proxies.get(a,[])): return
        if r2 is not None: self.edges[pair]=(r,r2)
        self.proxies.setdefault(a,[]).append((b,r,r2,rank))
        self.proxies.setdefault(b,[]).append((a,r,r2,rank))
        if len(self.edges)%256==0:self.check_budget()


def read_variant_map(path,ld):
    """BIM or TSV: id chrom pos ref alt frequency. Alleles define signed LD coding."""
    if str(path).endswith('.bim'):
        with text_open(path) as fh:
            for line in fh:
                c,id,cm,p,a1,a2=line.split()[:6]
                ld.variants[id]=Variant(c,int(p),a2,a1)
                if len(ld.variants)%256==0:ld.check_budget()
    else:
        with text_open(path) as fh:
            for row in csv.DictReader(fh,delimiter='\t'):
                v=Variant(row['chrom'],int(row['pos']),row['ref'],row['alt'])
                ld.variants[row['id']]=v
                if len(ld.variants)%256==0:ld.check_budget()
                if number(row.get('frequency')) is not None: ld.frequencies[v.id]=float(row['frequency'])


def load_ld(path=None,variant_map=None,bfile=None,originals=(),memory_mb=512,window=1000000,ancestry='European',matrix_kind=None,proxy_min_r2=.8):
    ld=LD(ancestry=ancestry,memory_mb=memory_mb)
    for v in originals: ld.variants[v.id]=v
    if variant_map: read_variant_map(variant_map,ld)
    ld.check_budget()
    if bfile: return from_bed(bfile,originals,memory_mb,window,ancestry,proxy_min_r2)
    if not path: return ld
    ld.provenance='external_ld_approximation'
    def resolve(id):
        if id in ld.variants: return ld.variants[id]
        try: v=Variant.parse(id)
        except ValueError: raise ValueError(f'LD variant {id} needs an allele-resolved ID or --snp-factor-ld-variants-in')
        ld.variants[id]=v
        return v
    # PLINK square matrices require ordered .vars IDs and an allele map for rsIDs.
    if matrix_kind or any(token in str(path) for token in ('.vcor1','.vcor2')) or str(path).endswith(('.matrix','.ld.bin')) or Path(str(path)+'.vars').exists() or Path(str(path)+'.ids').exists():
        ids_path=str(path).removesuffix('.zst').removesuffix('.gz')+'.vars'
        if not Path(ids_path).exists():
            ids_path=str(path)+'.ids'
        with text_open(ids_path) as fh: ids=[l.strip().split()[0] for l in fh if l.strip()]
        n=len(ids)
        if n*n*8*3>memory_mb*1024**2: raise MemoryError('LD matrix exceeds budget; use a regional/table LD input')
        matrix=np.fromfile(path,dtype='<f8').reshape(n,n) if str(path).endswith('.bin') else np.loadtxt(path)
        if matrix_kind is None and not any(token in str(path) for token in ('.vcor1','.vcor2')):
            raise ValueError('Legacy LD matrices require --snp-factor-ld-matrix-kind r or r2; do not guess phase')
        squared=matrix_kind=='r2' or (matrix_kind is None and 'vcor2' in str(path))
        if not np.all(np.isfinite(matrix)) or matrix.shape!=(n,n) or not np.allclose(matrix,matrix.T) or not np.allclose(np.diag(matrix),1): raise ValueError('LD matrix/IDs mismatch or asymmetry')
        for i in range(n):
            for j in range(i):
                a,b=resolve(ids[i]),resolve(ids[j]); ld.variants[a.id]=a; ld.variants[b.id]=b
                if matrix[i,j]: ld.add_edge(a.id,b.id,None if squared else float(matrix[i,j]),float(matrix[i,j]) if squared else None)
        ld.check_budget()
        return ld
    with text_open(path) as fh:
        lines=(line for line in fh if line.strip() and not line.startswith('##'))
        header=next(lines).strip().lstrip('#').split()
        for line in lines:
            row=dict(zip(header,line.split()))
            # PLINK clump summary lists candidates but cannot establish phase or r2.
            if 'SP2' in row:
                a=resolve(row['SNP']); ld.variants[a.id]=a
                for rank,item in enumerate(row['SP2'].split(',')):
                    id=item.split('(')[0]
                    if id in ('NONE','.', ''): continue
                    b=resolve(id); ld.variants[b.id]=b; ld.add_edge(a.id,b.id,None,None,rank)
                ld.provenance='clump_independence_approximation'
                continue
            a_id=row.get('index') or row.get('SNP_A') or row.get('ID_A')
            b_id=row.get('proxy') or row.get('SNP_B') or row.get('ID_B')
            if not a_id or not b_id: raise ValueError('LD table needs index/proxy, SNP_A/SNP_B, or ID_A/ID_B')
            a,b=resolve(a_id),resolve(b_id); ld.variants[a.id]=a; ld.variants[b.id]=b
            r=number(row.get('r',row.get('R',row.get('UNPHASED_R'))))
            r2=number(row.get('r2',row.get('R2',row.get('UNPHASED_R2'))))
            ld.add_edge(a.id,b.id,r,r2,int(row.get('rank',0)))
    ld.check_budget()
    return ld


class BedLD(LD):
    """Lazy reference: bounded genotype cache, no candidate-by-candidate LD matrix."""
    def __init__(self,prefix,originals,memory_mb,window,ancestry,min_r2):
        super().__init__(ancestry=ancestry,provenance='reference_ld_approximation',memory_mb=memory_mb)
        from bisect import bisect_left
        from functools import lru_cache
        self.prefix=str(prefix);self.window=window;self.memory_mb=memory_mb
        with open(self.prefix+'.fam') as fh:self.samples=sum(1 for line in fh if line.strip())
        if self.samples<3:raise ValueError('LD reference needs at least three samples')
        self.step=(self.samples+3)//4
        with open(self.prefix+'.bed','rb') as fh:
            if fh.read(3)!=b'\x6c\x1b\x01':raise ValueError('BED must be PLINK variant-major format')
        points={}
        for v in originals:points.setdefault(v.chrom,[]).append(v.pos)
        for chrom in points:points[chrom].sort()
        self.catalog={};self.by_chrom={}
        with open(self.prefix+'.bim') as fh:
            for index,line in enumerate(fh):
                c,id,cm,p,a1,a2=line.split()[:6];c=c.removeprefix('chr');p=int(p)
                positions=points.get(c,[]);j=bisect_left(positions,p-window)
                if j==len(positions) or positions[j]>p+window:continue
                if a1=='0' or a2=='0' or a1==a2:continue
                v=Variant(c,p,a2,a1)
                if v.id in self.catalog:raise ValueError('Duplicate BED variant identity')
                self.catalog[v.id]=(index,v,1)
                self.variants[v.id]=v;self.by_chrom.setdefault(v.chrom,[]).append(v)
                if len(self.catalog)*700>memory_mb*1024**2/3:
                    raise MemoryError('Selected BED variant metadata exceeds budget; use a regional reference')
        for chrom in self.by_chrom:self.by_chrom[chrom].sort(key=lambda v:v.pos)
        cache_count=max(1,min(256,int(memory_mb*1024**2/8/max(self.samples*8,1))))
        if self.samples*8*4>memory_mb*1024**2/4:raise MemoryError('One reference genotype vector exceeds budget')
        self.genotype=lru_cache(maxsize=cache_count)(self._genotype)
        self.original_ids=set()
        for v in originals:
            matches=[b for b in self.by_chrom.get(v.chrom,[]) if b.pos==v.pos and allele_sign(b,v) is not None]
            if len(matches)==1:
                b=matches[0];index,_,_=self.catalog[b.id]
                self.catalog[v.id]=(index,b,allele_sign(b,v));self.variants[v.id]=v
                self.original_ids.add(v.id)
        # Construct only original-to-proxy links; covariance among actually used
        # variants is evaluated lazily below, after target-specific selection.
        for original in originals:
            if original.id not in self.catalog:continue
            x=self.genotype(original.id)
            if x is None:continue
            for v in self.by_chrom.get(original.chrom,[]):
                if v.pos<original.pos-window:continue
                if v.pos>original.pos+window:break
                if v.id==original.id:continue
                y=self.genotype(v.id)
                if y is None:continue
                r=float(np.clip(np.dot(x,y)/self.samples,-1,1))
                if r*r>=min_r2:
                    self.add_edge(original.id,v.id,r,r*r)
        self.genotype.cache_clear()

    def _genotype(self,id):
        index,v,sign=self.catalog[id]
        with open(self.prefix+'.bed','rb') as fh:
            fh.seek(3+index*self.step);raw=fh.read(self.step)
        if len(raw)!=self.step:raise ValueError('Truncated BED')
        data=np.frombuffer(raw,dtype=np.uint8)
        decode=np.array([2.,np.nan,1.,0.])
        x=decode[((data[:,None]>>(2*np.arange(4)))&3).ravel()[:self.samples]]
        if np.mean(np.isnan(x))>.02:return None
        f=float(np.nanmean(x)/2)
        if not .01<=f<=.99:return None
        x=np.where(np.isnan(x),2*f,x);sd=x.std()
        if sd==0:return None
        self.frequencies[id]=f if sign==1 else 1-f
        return sign*(x-x.mean())/sd

    def pair(self,a,b):
        if a==b:return 1.,1.
        if a not in self.catalog or b not in self.catalog:return super().pair(a,b)
        va,vb=self.variants[a],self.variants[b]
        if va.chrom!=vb.chrom or abs(va.pos-vb.pos)>self.window:return 0.,0.
        x,y=self.genotype(a),self.genotype(b)
        if x is None or y is None:return 0.,0.
        r=float(np.clip(np.dot(x,y)/self.samples,-1,1))
        return r,r*r

    def blocks(self,variants,memory_mb=512):
        ordered=sorted(range(len(variants)),key=lambda i:(variants[i].chrom,variants[i].pos))
        groups=[];group=[]
        for i in ordered:
            if group and (variants[i].chrom!=variants[group[-1]].chrom or variants[i].pos-variants[group[-1]].pos>self.window):
                groups.append(group);group=[]
            group.append(i)
        if group:groups.append(group)
        for group in groups:
            ix=np.array(group,dtype=int);n=len(ix)
            if n*n*8*8>memory_mb*1024**2:raise MemoryError('One selected LD block exceeds memory budget')
            R=np.eye(n)
            for i in range(n):
                for j in range(i):R[i,j]=R[j,i]=self.pair(variants[ix[i]].id,variants[ix[j]].id)[0]
            if np.linalg.eigvalsh(R).min() < -1e-7:raise ValueError('Window-truncated LD is not positive semidefinite; increase the LD window')
            yield ix,R,R*R,True


def from_bed(prefix,originals,memory_mb,window,ancestry,min_r2=.8):
    return BedLD(prefix,originals,memory_mb,window,ancestry,min_r2)
