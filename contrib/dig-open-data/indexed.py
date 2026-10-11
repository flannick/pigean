"""Additive dig_open_data.indexed module; existing streaming APIs are unchanged.

Copy into src/dig_open_data/indexed.py in DIG. Install the optional `indexed`
extra (pysam) to use iter_regions; discovery itself uses only the standard lib.
"""
from __future__ import annotations
import json
import urllib.error
from functools import lru_cache
from .api import open_text
from .catalog import DEFAULT_BUCKET, DEFAULT_PREFIX


@lru_cache(maxsize=4)
def _manifest(bucket,prefix):
    uri=f's3://{bucket}/{prefix.rstrip("/")}/indexed/manifest.json'
    try:
        with open_text(uri,retries=0) as fh: value=json.load(fh)
    except urllib.error.HTTPError as exc:
        if exc.code==404:return {}
        raise
    if value.get('version')!=1:raise ValueError('Unsupported DIG indexed manifest version')
    return value.get('traits',{})


def indexed_resource(ancestry,trait,*,bucket=DEFAULT_BUCKET,prefix=DEFAULT_PREFIX):
    """Return the validated data/index descriptor, or None if not published."""
    entry=_manifest(bucket,prefix).get(f'{ancestry}:{trait}')
    if entry is None:return None
    if not entry.get('uri') or not entry.get('index_uri'):raise ValueError('Incomplete DIG indexed resource')
    return dict(entry)


def _http(uri):
    if uri.startswith('s3://'):
        bucket,key=uri[5:].split('/',1)
        return f'https://{bucket}.s3.amazonaws.com/{key}'
    return uri


def iter_regions(uri,regions,*,index_uri=None):
    """Yield unique records for (chrom,start,end) zero-based half-open regions.

    Merge overlapping requests before fetching. No fallback or full download:
    callers decide whether to stream when the capability is unavailable.
    """
    try:import pysam
    except ImportError as exc:raise ImportError('Install dig-open-data[indexed] for region queries') from exc
    merged=[]
    for chrom,start,end in sorted(regions):
        if start<0 or end<=start:raise ValueError('Invalid half-open interval')
        if merged and merged[-1][0]==chrom and start<=merged[-1][2]:
            merged[-1]=(chrom,merged[-1][1],max(end,merged[-1][2]))
        else:merged.append((chrom,start,end))
    with pysam.TabixFile(_http(uri),index=_http(index_uri) if index_uri else None) as tabix:
        names=set(tabix.contigs)
        for chrom,start,end in merged:
            if chrom in names:yield from tabix.fetch(chrom,start,end)


def iter_trait_regions(ancestry,trait,regions,**kwargs):
    entry=indexed_resource(ancestry,trait,**kwargs)
    if entry is None:raise FileNotFoundError('Indexed access has not been published for this DIG trait')
    yield from iter_regions(entry['uri'],regions,index_uri=entry['index_uri'])
