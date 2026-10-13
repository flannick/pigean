"""Compare scientific summaries across seeds/configurations, without timing gates."""
import argparse
import json
from pathlib import Path
from compare_inference_screen import table,metric


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();a,b=Path(args.reference),Path(args.candidate)
    receipts=[json.loads((x/'screen.json').read_text()) for x in (a,b)]
    for key in ('fixture_sha256','inference_kwargs','outer_updates_executed'):
        assert receipts[0][key]==receipts[1][key],key
    assert all(x['status']=='COMPLETE_FIXED_BUDGET' for x in receipts)
    result=dict(status='SINGLE_CONTROL_SENSITIVITY_NOT_STATISTICAL_EQUIVALENCE',
        reference=str(a),candidate=str(b),seeds=[x['args']['seed'] for x in receipts],
        commits=[x['commit'] for x in receipts],exports={},
        diagnostics=[x.get('latest_post_burn_diagnostics') for x in receipts])
    for file,key,value in (('gene_stats.out.gz','Gene','prior'),('gene_set_stats.out.gz','Gene_Set','beta_uncorrected'),('gene_set_stats.out.gz','Gene_Set','beta')):
        x,y=[table(path/file).set_index(key)[value] for path in (a,b)]
        assert x.index.is_unique and y.index.is_unique
        result['exports'][value]=metric(x,y)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
