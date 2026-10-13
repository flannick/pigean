"""Archive compact stage-2 evidence; raw data remain in the analysis workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1];repo=Path(args.repo).resolve()
    destination=repo/'docs/experiments/optimization-milestone-2'
    destination.mkdir(parents=True,exist_ok=False)
    selected=[]
    for name in ('optimization_tracker.md','autonomous_optimization_20261012.md','stage2_validation_20261012.md','next_inference_directions_20261012.md','next_inference_directions_20261013.md'):
        selected.append(root/'plans'/name)
    names=('2026-10-13_optimization_stage2_closeout.md','2026-10-12_autonomous_optimization.md','2026-10-12_inner_target_audit.md','2026-10-12_tracker_history.md',
        '2026-10-12_T2D_full_budget.md','2026-10-11_bounded_optimization.md','2026-10-11_grouped_regression.md',
        '2026-10-12_annotation_structure_reuse.md','2026-10-12_compact_covariance.md','2026-10-12_lean_regression.md',
        '2026-10-12_inner_gibbs_arithmetic.md','2026-10-12_compiled_covariance.md','2026-10-12_parallel_covariance.md',
        '2026-10-11_convergence_runtime_research.md')
    selected.extend(root/'logs'/name for name in names)
    runs=root/'results/autonomous_20261012'
    for path in runs.rglob('*'):
        if not path.is_file() or 'numba_cache' in path.parts:continue
        if (path.suffix=='.json' and path.name not in ('progress.json',)) or path.name=='profile.txt' or path.name.endswith('_tests.txt') or path.name=='static_manifest_test.txt':
            selected.append(path)
    selected.extend((root/'results/T2D_full_budget_20261012'/name) for name in ('comparison.json','legacy/screen.json','optimized/screen.json'))
    scripts=('screen_inference.py','compare_inference_screen.py','compare_output_sensitivity.py',
        'capture_inner_workload.py','capture_prefilter_workload.py','benchmark_inner_conditional.py','benchmark_inner_sweeps.py',
        'benchmark_inner_cache_capacity.py','benchmark_inner_batching.py','benchmark_prefilter.py','benchmark_huge_prior.py',
        'benchmark_static_correlation.py','run_autonomous_T2D_pair.sh','run_stage2_full.sh','run_final_arithmetic.sh',
        'run_stage2_NAFLD_pair.sh','run_NAFLD_huge_ablation.sh','run_NAFLD_seed_control.sh','run_static_correlation_full.sh',
        'run_inner_effort_screen.sh','archive_optimization_stage2.py')
    selected.extend(root/'scripts'/name for name in scripts)
    # An optional full probe exists only if the short-screen decision warranted it.
    selected.extend((root/'scripts').glob('run_inner_effort_full*.sh'))
    for trait in ('T2D','NAFLD'):
        for name in ('fixture.pkl.gz.json','command.json','receipt.json'):
            selected.append(root/'results/feature_foundation_20261011/fixtures'/trait/name)
    manifest=[]
    for path in sorted(set(selected)):
        if not path.is_file():raise FileNotFoundError(path)
        relative=path.relative_to(root);target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        manifest.append(dict(path=str(relative),sha256=digest(path),bytes=path.stat().st_size))
    raw=[]
    for path in sorted(runs.rglob('*.gz')):
        if 'numba_cache' in path.parts:continue
        raw.append(dict(path=str(path.relative_to(root)),sha256=digest(path),bytes=path.stat().st_size))
    (destination/'raw_artifacts.json').write_text(json.dumps(raw,indent=2)+'\n')
    (destination/'manifest.json').write_text(json.dumps(dict(original_analysis=str(root),
        tested_source_before_archive=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        files=manifest),indent=2)+'\n')
    print(json.dumps(dict(destination=str(destination),files=len(manifest),copied_bytes=sum(x['bytes'] for x in manifest),raw_files_bound=len(raw))))


if __name__=='__main__':main()
