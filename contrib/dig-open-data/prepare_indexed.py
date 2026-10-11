"""Prepare a local BGZF/CSI companion and manifest fragment; never uploads.

Run with the project's configured Python interpreter. Requires native bgzip,
tabix and sort executables. Work space is bounded by external sort settings.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('output');p.add_argument('--ancestry',required=True);p.add_argument('--trait',required=True)
    p.add_argument('--build',required=True);p.add_argument('--published-uri',required=True);p.add_argument('--sort-memory',default='256M')
    args=p.parse_args()
    output=Path(args.output)
    if output.exists() or Path(str(output)+'.csi').exists():raise ValueError('Refusing to overwrite existing companions')
    digest=hashlib.sha256()
    with open(args.source,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='dig-index-') as temp:
        rows=Path(temp)/'rows.tsv';sorted_path=Path(temp)/'sorted.tsv';packed=Path(temp)/'sumstats.tsv.gz'
        opener=gzip.open if args.source.endswith('.gz') else open
        with opener(args.source,'rt') as source,rows.open('w') as out:
            header=source.readline().rstrip('\n').split('\t')
            chrom=header.index('chromosome')+1;position=header.index('position')+1
            for line in source:out.write(line)
        with sorted_path.open('w') as out:
            out.write('#'+'\t'.join(header)+'\n');out.flush()
            subprocess.run(['sort','-S',args.sort_memory,'-T',temp,'-t','\t',f'-k{chrom},{chrom}',f'-k{position},{position}n',str(rows)],stdout=out,check=True)
        with packed.open('wb') as out:subprocess.run(['bgzip','-c',str(sorted_path)],stdout=out,check=True)
        subprocess.run(['tabix','-C','-s',str(chrom),'-b',str(position),'-e',str(position),str(packed)],check=True)
        subprocess.run(['tabix','-l',str(packed)],check=True,capture_output=True)
        packed.rename(output);Path(str(packed)+'.csi').rename(str(output)+'.csi')
    entry=dict(uri=args.published_uri,index_uri=args.published_uri+'.csi',build=args.build,
               coordinate_system='1-based',chrom_column=chrom,position_column=position,header=header,
               source_sha256=digest.hexdigest())
    Path(str(output)+'.manifest-fragment.json').write_text(json.dumps({'version':1,'traits':{f'{args.ancestry}:{args.trait}':entry}},indent=2)+'\n')


if __name__=='__main__':main()
