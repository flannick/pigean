"""Versioned sparse weights and allele-aware selected-record GWAS access."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import sqlite3
import shutil
import subprocess
import tempfile
import urllib.request
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy import sparse, stats


def warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


@contextmanager
def text_open(path, mode='rt'):
    path = str(path)
    if path.startswith(('https://', 'http://', 's3://')):
        if mode != 'rt':
            raise ValueError('Remote writes are not supported')
        if path.startswith('s3://'):
            bucket, key = path[5:].split('/', 1)
            path = 'https://%s.s3.amazonaws.com/%s' % (bucket, key)
        with urllib.request.urlopen(path, timeout=60) as response:
            stream = io.BufferedReader(response)
            decoded = gzip.GzipFile(fileobj=stream) if stream.peek(2)[:2] == b'\x1f\x8b' else stream
            with io.TextIOWrapper(decoded) as handle:
                yield handle
    else:
        opener = gzip.open if path.endswith(('.gz', '.bgz')) else open
        with opener(path, mode) as handle:
            yield handle


def chrom_name(value):
    value = str(value).removeprefix('chr')
    return {'23': 'X', '24': 'Y', 'M': 'MT'}.get(value, value)


@dataclass(frozen=True, order=True)
class Variant:
    chrom: str
    pos: int
    ref: str
    alt: str

    def __post_init__(self):
        object.__setattr__(self, 'chrom', chrom_name(self.chrom))
        object.__setattr__(self, 'ref', self.ref.upper())
        object.__setattr__(self, 'alt', self.alt.upper())
        if self.pos < 1 or not self.ref or not self.alt or self.ref == self.alt:
            raise ValueError('Variant requires positive position and distinct alleles')

    @property
    def id(self):
        return f'{self.chrom}:{self.pos}:{self.ref}:{self.alt}'

    @classmethod
    def parse(cls, value):
        c, p, r, a = str(value).split(':')
        return cls(c, int(p), r, a)

    @property
    def coordinate(self):
        return self.chrom, self.pos


def allele_sign(observed, expected, *, signed=False, observed_freq=None, expected_freq=None):
    """Orientation to expected ALT dosage; unresolved palindromes fail signed use."""
    if observed.coordinate != expected.coordinate:
        return None
    pair, target = (observed.ref, observed.alt), (expected.ref, expected.alt)
    complement = str.maketrans('ACGT', 'TGCA')
    candidates = []
    if pair == target: candidates.append(1)
    if pair[::-1] == target: candidates.append(-1)
    if all(len(a) == 1 and a in 'ACGT' for a in pair + target):
        comp = tuple(a.translate(complement) for a in pair)
        if comp == target: candidates.append(1)
        if comp[::-1] == target: candidates.append(-1)
    candidates = set(candidates)
    if not candidates:
        return None
    if len(candidates) > 1 and signed:
        if observed_freq is None or expected_freq is None:
            return None
        compatible = [s for s in candidates if abs((observed_freq if s == 1 else 1-observed_freq)-expected_freq) < .1]
        return compatible[0] if len(compatible) == 1 else None
    return 1 if 1 in candidates else -1


@dataclass
class Source:
    source: str
    trait: str = ''
    ancestry: str = 'Mixed'
    build: str = ''
    units: str = ''
    study: str = ''
    n: float | None = None
    columns: dict = field(default_factory=dict)
    index: str = ''

    def __post_init__(self):
        if not self.trait:
            self.trait = self.source.split(':')[-1] if self.source.startswith('dig:') else Path(self.source).name
        if self.source.startswith('dig:'):
            _, self.ancestry, self.trait = self.source.split(':', 2)

    @property
    def uri(self):
        if not self.source.startswith('dig:'):
            return self.source
        from dig_open_data.catalog import build_key, DEFAULT_BUCKET
        return f's3://{DEFAULT_BUCKET}/{build_key(self.ancestry, self.trait)}'


ALIASES = {
    'chrom': ['chromosome', 'chr', 'chrom', '#chrom'], 'pos': ['position', 'pos', 'bp', 'genpos'],
    'id': ['varid', 'variant', 'variant_id', 'snp', 'snpid', 'locus', 'lead_variant'],
    'p': ['pvalue', 'p_value', 'p-value', 'p', 'pval'],
    'beta': ['beta', 'effect', 'effect_size', 'b'], 'se': ['stderr', 'se', 'standard_error', 'sebeta'],
    'alt': ['effect_allele', 'alt', 'a1', 'ea', 'allele1'],
    'ref': ['other_allele', 'reference', 'ref', 'a2', 'nea', 'allele0'],
    'freq': ['eaf', 'af', 'freq', 'frequency', 'allelefrequency_effect', 'a1freq'],
    'info': ['info', 'imputation_info', 'rsq'], 'n': ['n', 'n_total', 'samplesize'],
    'z': ['zscore', 'z', 'z_score'],
}


def header_columns(header, overrides=None):
    """One-based numeric overrides, like gwas-in; names are case insensitive."""
    lower = [h.lower() for h in header]
    result = {}
    from pegs_utils import infer_columns_from_table_file
    inferred = infer_columns_from_table_file(
        "header", open_text_fn=lambda _: io.StringIO("\t".join(header)+"\n"),
        log_fn=lambda *a: None, bail_fn=lambda message: None)
    for key, values in zip(('gene','id','chrom','pos','locus','p','beta','se','freq','n'), inferred[:-1]):
        if key in ALIASES and len(values)==1:
            value=values[0]
            if value in header: result[key]=header.index(value)

    for key, aliases in ALIASES.items():
        override = (overrides or {}).get(key)
        if override not in (None, ''):
            if str(override).isdigit():
                i = int(override)-1
                if not 0 <= i < len(header): raise ValueError(f'Column out of range: {override}')
            else:
                i = lower.index(str(override).lower())
            result[key] = i
        else:
            matches = [lower.index(a) for a in aliases if a in lower]
            if matches: result[key] = matches[0]
    if not ('id' in result or {'chrom', 'pos'} <= result.keys()):
        raise ValueError('GWAS needs chromosome/position or an allele-resolved variant ID')
    result['_freq_is_maf'] = 'freq' in result and lower[result['freq']] in ('maf','minor_allele_frequency')
    return result


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def parse_record(fields, columns, source, fallback_variant=None):
    get = lambda k: fields[columns[k]] if k in columns and columns[k] < len(fields) else None
    parsed = None
    if get('id'):
        try: parsed = Variant.parse(get('id'))
        except (ValueError, TypeError): pass
    c = get('chrom') or (parsed.chrom if parsed else None)
    p = get('pos') or (parsed.pos if parsed else None)
    ref = get('ref') or (parsed.ref if parsed else None)
    alt = get('alt') or (parsed.alt if parsed else None)
    allele_verified=bool(ref and alt)
    if c is None or p is None:raise ValueError('Missing variant coordinate')
    if not ref or not alt:
        if fallback_variant is None:raise ValueError('Missing allele-resolved variant identity')
        ref,alt=fallback_variant.ref,fallback_variant.alt
    variant = Variant(str(c), int(p), ref, alt)
    beta, se, pval, z = (number(get(k)) for k in ('beta', 'se', 'p', 'z'))
    if pval is not None and not 0 < pval <= 1: pval = None
    # Preserve usable P separately: negative SE must not be repaired.
    if pval is not None:
        z2 = float(stats.chi2.isf(pval, 1))
    elif se is not None and se > 0 and beta is not None:
        z2 = (beta/se)**2
    elif z is not None:
        z2 = z*z
    else:
        z2 = None
    return dict(variant=variant.id, allele_verified=allele_verified, beta=beta, se=se, p=pval, z2=z2,
                freq=number(get('freq')), freq_is_maf=columns.get('_freq_is_maf',False), info=number(get('info')),
                n=number(get('n')) or source.n)


class SelectedRecords:
    """Disk-backed selected rows. Memory is independent of source file size."""
    def __init__(self, memory_only=False, max_memory_mb=256):
        self.temp = None if memory_only else tempfile.TemporaryDirectory(prefix='eaggl-snps-')
        self.db = sqlite3.connect(':memory:' if memory_only else str(Path(self.temp.name)/'records.sqlite'))
        if memory_only:
            self.db.execute('PRAGMA temp_store=MEMORY')
            page_size=self.db.execute('PRAGMA page_size').fetchone()[0]
            self.db.execute('PRAGMA max_page_count=%d' % max(1,int(max_memory_mb*1024**2/page_size)))
        self.db.execute('PRAGMA cache_size=-2048')
        self.db.execute('CREATE TABLE records (chrom TEXT,pos INTEGER,id TEXT PRIMARY KEY,payload TEXT, duplicate INTEGER DEFAULT 0)')
        self.db.execute('CREATE INDEX coords ON records(chrom,pos)')
        self.qc = {'rows_scanned': 0, 'selected': 0, 'invalid': 0, 'duplicates': 0}
        self.access = 'stream'

    def add(self, record):
        v = Variant.parse(record['variant'])
        old = self.db.execute('SELECT payload FROM records WHERE id=?', (v.id,)).fetchone()
        payload = json.dumps(record, sort_keys=True)
        if old:
            if old[0] != payload:
                self.db.execute('UPDATE records SET duplicate=1 WHERE id=?', (v.id,))
                self.qc['duplicates'] += 1
            return
        self.db.execute('INSERT INTO records(chrom,pos,id,payload) VALUES (?,?,?,?)', (v.chrom,v.pos,v.id,payload))
        self.qc['selected'] += 1

    def lookup(self, variant, *, signed=False, freq=None, maf=.01, info=.8, require_stat=True, require_se=True):
        matches = []
        for payload, duplicate in self.db.execute('SELECT payload,duplicate FROM records WHERE chrom=? AND pos=?', variant.coordinate):
            if duplicate: continue
            r = json.loads(payload)
            if r['freq'] is not None and not maf <= r['freq'] <= 1-maf: continue
            if r['info'] is not None and r['info'] < info: continue
            sign = allele_sign(Variant.parse(r['variant']), variant, signed=signed, observed_freq=None if r.get('freq_is_maf') else r['freq'], expected_freq=freq)
            if sign is None or (require_stat and r['z2'] is None): continue
            if signed and not r.get('allele_verified',True):continue
            if signed and (r['beta'] is None or (require_se and (r['se'] is None or r['se'] <= 0))): continue
            r['beta'] = None if r['beta'] is None else sign*r['beta']
            if sign == -1 and r['freq'] is not None and not r.get('freq_is_maf'): r['freq'] = 1-r['freq']
            r['source_variant']=r['variant']
            r['variant']=variant.id
            matches.append(r)
        return matches[0] if len(matches) == 1 else None

    def close(self):
        self.db.close()
        if self.temp is not None:self.temp.cleanup()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def coalesced_regions(coordinates, gap=16384, max_span=1048576):
    """Merge nearby requests to reuse BGZF blocks; exact variants are filtered later."""
    current=None
    for chrom,pos in sorted(set(coordinates)):
        if current and chrom==current[0] and pos-current[2]<=gap and pos-current[1]<=max_span:
            current=(chrom,current[1],pos)
        else:
            if current:yield current
            current=(chrom,pos,pos)
    if current:yield current


def fetch_selected(source, coordinates, variants=(), *, memory_only=False, stream_only=False, max_memory_mb=256):
    """Use tabix when usable, otherwise stream once into a selected-only SQLite store."""
    selected = SelectedRecords(memory_only=memory_only,max_memory_mb=max_memory_mb)
    coords = set(coordinates)
    expected={}
    for v in variants:expected.setdefault(v.coordinate,set()).add(v)
    unique={c:next(iter(vs)) for c,vs in expected.items() if len(vs)==1}
    uri = source.uri
    discovered_index = source.index
    if not stream_only and source.source.startswith('dig:'):
        try:
            from dig_open_data.indexed import indexed_resource
            resource = indexed_resource(source.ancestry, source.trait)
            if resource:
                uri = resource['uri']; discovered_index = resource['index_uri']
                if resource.get('build'):
                    if source.build and source.build != resource['build']: raise ValueError('DIG build metadata conflict')
                    source.build = resource['build']
        except ImportError:
            pass
    if not stream_only and not discovered_index and uri.startswith(('http://','https://','s3://')):
        remote=uri
        if remote.startswith('s3://'):
            bucket,key=remote[5:].split('/',1); remote=f'https://{bucket}.s3.amazonaws.com/{key}'
        for suffix in ('.csi','.tbi'):
            try:
                with urllib.request.urlopen(urllib.request.Request(remote+suffix,method='HEAD'),timeout=10):
                    discovered_index=remote+suffix
                    break
            except OSError: pass
    header = None
    try:
        with text_open(uri) as fh:
            for line in fh:
                if line.startswith('##') or not line.strip(): continue
                header = line.strip().lstrip('#').split()
                break
        if header is None: raise ValueError('Empty GWAS file')
        columns = header_columns(header, source.columns)
        index = discovered_index
        if not index and not uri.startswith(('http', 's3:')):
            index = next((uri+s for s in ('.csi','.tbi') if os.path.exists(uri+s)), '')
        tabix = None
        if index and not stream_only:
            try:
                import pysam
                remote = uri
                if remote.startswith('s3://'):
                    bucket,key=remote[5:].split('/',1); remote=f'https://{bucket}.s3.amazonaws.com/{key}'
                tabix = pysam.TabixFile(remote, index=index)
            except (ImportError, OSError, ValueError) as exc:
                if not memory_only and shutil.which('tabix') and index in (uri+'.csi',uri+'.tbi'):
                    try: tabix = TabixCLI(uri, selected.temp.name)
                    except (OSError,ValueError): pass
                if tabix is None: warn(f'{source.trait}: index unavailable ({exc}); scanning; BGZF/tabix is recommended')
        if tabix is None:
            warn(f'{source.trait}: random access unavailable; streaming may be slow; BGZF/tabix is recommended')
            stream = text_open(uri)
            fh = stream.__enter__()
            lines = (line for line in fh if line.strip() and not line.startswith('#'))
        else:
            selected.access = 'tabix'
            contigs = {chrom_name(c): c for c in tabix.contigs}
            def indexed_lines():
                if isinstance(tabix, TabixCLI):
                    yield from tabix.fetch_many([(contigs[c],p) for c,p in sorted(coords) if c in contigs])
                    return
                for chrom,start,end in coalesced_regions(coords):
                    if chrom in contigs:
                        yield from tabix.fetch(contigs[chrom], start-1, end)
            lines = indexed_lines()
        try:
            for line in lines:
                selected.qc['rows_scanned'] += 1
                fields = line.split()
                if fields == header: continue
                # Reject nonrequested coordinates before parsing numeric association columns.
                try:
                    if 'chrom' in columns and 'pos' in columns:
                        coord = chrom_name(fields[columns['chrom']]), int(fields[columns['pos']])
                    else:
                        bits = fields[columns['id']].split(':')
                        coord = chrom_name(bits[0]), int(bits[1])
                    if coord not in coords: continue
                    selected.add(parse_record(fields, columns, source,unique.get(coord)))
                except (ValueError, IndexError, OverflowError):
                    selected.qc['invalid'] += 1
            selected.db.commit()
        finally:
            if tabix is not None: tabix.close()
            else: stream.__exit__(None,None,None)
        return selected
    except Exception:
        selected.close()
        raise


def source_list(path, default_ancestry='Mixed'):
    with text_open(path) as fh:
        lines = [line.strip() for line in fh if line.strip() and not line.startswith('#')]
    if not lines: return []
    if lines[0].split('\t')[0] == 'source':
        result=[]
        for row in csv.DictReader(lines, delimiter='\t'):
            columns={k[:-4]:v for k,v in row.items() if k.endswith('_col') and v}
            kwargs={k:v for k,v in row.items() if k in Source.__dataclass_fields__ and k!='columns' and v}
            if 'n' in kwargs: kwargs['n']=float(kwargs['n'])
            kwargs['columns']=columns
            result.append(Source(**kwargs))
        return result
    return [Source(line, ancestry=default_ancestry) for line in lines]


def resolve_sources(options):
    result=[]
    ancestries=getattr(options,'snp_factor_phewas_dig_ancestry',None) or ['Mixed']
    for src in getattr(options,'snp_factor_phewas_gwas_in',None) or []: result.append(Source(src))
    for path in getattr(options,'snp_factor_phewas_gwas_list',None) or []: result.extend(source_list(path))
    ids=list(getattr(options,'snp_factor_phewas_dig_id',None) or [])
    for path in getattr(options,'snp_factor_phewas_dig_id_list',None) or []:
        with text_open(path) as fh: ids.extend(l.strip() for l in fh if l.strip() and not l.startswith('#'))
    for item in ids:
        if item.startswith('dig:'): result.append(Source(item))
        else:
            for ancestry in ancestries: result.append(Source(f'dig:{ancestry}:{item}'))
    if getattr(options,'snp_factor_phewas_dig_all',False):
        from dig_open_data.catalog import list_traits
        for ancestry in ancestries:
            result.extend(Source(f'dig:{ancestry}:{trait}') for trait in list_traits(ancestry=ancestry))
    unique={}
    for source in result:
        key=(source.source,source.ancestry)
        if key in unique and asdict(unique[key]) != asdict(source):
            raise ValueError(f'Conflicting metadata for {source.source}')
        unique.setdefault(key,source)
    return list(unique.values())


@dataclass
class Weights:
    variants: list[Variant]
    factors: list[str]
    membership: sparse.csr_matrix
    coefficients: sparse.csr_matrix
    frequencies: np.ndarray
    metadata: dict = field(default_factory=dict)

    def validate(self):
        self.membership.eliminate_zeros()
        self.coefficients.eliminate_zeros()
        shape=(len(self.variants),len(self.factors))
        if self.membership.shape != shape or self.coefficients.shape != shape: raise ValueError('Weight dimensions disagree')
        if len(set(self.variants)) != shape[0] or len(set(self.factors)) != shape[1]: raise ValueError('Duplicate weight IDs')
        if len(self.frequencies)!=shape[0]: raise ValueError('Frequency dimension disagrees')
        if np.any(~np.isfinite(self.membership.data)) or np.any(self.membership.data<0): raise ValueError('Invalid membership')
        if np.any(~np.isfinite(self.coefficients.data)): raise ValueError('Invalid signed coefficient')
        active=np.diff(self.coefficients.indptr)>0
        if self.metadata.get('mode','unsigned') != 'unsigned' and np.any(active & (~np.isfinite(self.frequencies) | (self.frequencies<=0) | (self.frequencies>=1))):
            raise ValueError('Signed weights require valid reference allele frequencies')
        return self

    def write(self,path):
        self.validate()
        metadata=dict(self.metadata,version=1,factors=self.factors)
        with text_open(path,'wt') as fh:
            fh.write('#'+json.dumps(metadata,sort_keys=True)+'\n')
            writer=csv.writer(fh,delimiter='\t')
            writer.writerow(['variant','factor','membership','coefficient','frequency'])
            for i,v in enumerate(self.variants):
                row=self.membership.getrow(i)
                for j,a in zip(row.indices,row.data):
                    writer.writerow([v.id,self.factors[j],repr(float(a)),repr(float(self.coefficients[i,j])),repr(float(self.frequencies[i]))])

    @classmethod
    def read(cls,path):
        with text_open(path) as fh:
            meta=json.loads(next(fh).removeprefix('#'))
            if meta.get('version')!=1: raise ValueError('Unsupported SNP weight version')
            factors=meta['factors']; fmap={v:i for i,v in enumerate(factors)}
            variants=[]; lookup={}; rows=[]; cols=[]; aa=[]; uu=[]; freqs=[]; seen=set()
            for row in csv.DictReader(fh,delimiter='\t'):
                v=Variant.parse(row['variant'])
                if v not in lookup:
                    lookup[v]=len(variants); variants.append(v); freqs.append(float(row['frequency']))
                i=lookup[v]; j=fmap[row['factor']]
                f=float(row['frequency'])
                if not (f==freqs[i] or (np.isnan(f) and np.isnan(freqs[i]))): raise ValueError('Conflicting native weight frequencies')
                if (i,j) in seen: raise ValueError('Duplicate factor/variant weight')
                seen.add((i,j)); rows.append(i); cols.append(j); aa.append(float(row['membership'])); uu.append(float(row['coefficient']))
            shape=len(variants),len(factors)
            return cls(variants,factors,sparse.csr_matrix((aa,(rows,cols)),shape=shape),sparse.csr_matrix((uu,(rows,cols)),shape=shape),np.array(freqs),meta).validate()


class TabixCLI:
    """Optional system-HTSlib backend when pysam is not installed."""
    def __init__(self,uri,directory):
        self.uri=uri; self.directory=directory
        p=subprocess.run(['tabix','-l',uri],text=True,capture_output=True)
        if p.returncode: raise ValueError(p.stderr)
        self.contigs=p.stdout.splitlines()
    def fetch_many(self,coords):
        regions=Path(self.directory)/'regions.tsv'
        with regions.open('w') as fh:
            for c,start,end in coalesced_regions(coords): fh.write(f'{c}\t{start}\t{end}\n')
        with tempfile.TemporaryFile(mode='w+t') as errors:
            proc=subprocess.Popen(['tabix','-R',str(regions),self.uri],stdout=subprocess.PIPE,stderr=errors,text=True)
            try:
                yield from proc.stdout
                if proc.wait():
                    errors.seek(0);raise ValueError(errors.read(4096))
            finally:
                proc.stdout.close()
                if proc.poll() is None: proc.terminate();proc.wait()
    def close(self):pass
