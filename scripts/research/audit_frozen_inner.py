"""Audit trusted captured production conditionals without fitting a new model."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'research'))
from conditional_compatibility import audit_slab_precision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-coefficients',type=int,default=128)
    a=p.parse_args()
    if a.max_coefficients < 2:p.error('Need at least two coefficients')
    source=a.capture/'conditional.pkl.gz';meta=json.loads((a.capture/'capture.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != meta['conditional_sha256']:raise ValueError('Capture checksum mismatch')
    with gzip.open(source,'rb') as f:k=pickle.load(f)
    if k['use_X'] or k['assume_independent'] or k['gauss_seidel']:raise ValueError('Require captured correlated V path')
    V=k['V'].toarray() if k['sparse_V'] else k['V']
    rows=[]
    for chain in range(k['se2s_m'].shape[0]):
        s=k['hdmp_hdmpn_m'][chain];v=k['norm_scale_m'][chain]**2
        valid=np.flatnonzero(np.isfinite(v)&(v>0)&np.isfinite(s)&(s>0))
        if len(valid)<2:continue
        ids=valid[np.linspace(0,len(valid)-1,min(len(valid),a.max_coefficients),dtype=int)]
        matrix=V[chain] if k['multiple_V'] else V
        if k['sparse_V']:
            matrix=matrix.T  # The production sparse branch multiplies by rows.
        result=audit_slab_precision(matrix[np.ix_(ids,ids)],s[ids],v[ids])
        result.update(chain=chain,selected_indices=ids.tolist())
        w=result['witness'];w.update(original_i=int(ids[w['i']]),original_j=int(ids[w['j']]))
        rows.append(result)
    if not rows:raise ValueError('No eligible conditionals')
    report=dict(capture=meta,conditional_sha256=digest,checks=rows,
                joint_target_gate='FAIL' if any(not r['symmetric'] or not r['positive_definite'] for r in rows) else 'NECESSARY_CHECKS_ONLY',
                scope='Frozen all-slab conditional subsets; rejection is a witness, passing is not a complete target proof.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.out,'wt') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(gate=report['joint_target_gate'],chains=len(rows),max_relative_asymmetry=max(r['max_relative_asymmetry'] for r in rows))))

if __name__=='__main__':main()
