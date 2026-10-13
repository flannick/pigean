"""Bounded, explicitly non-convergence screening of trusted local Gibbs fixtures."""
import argparse
import cProfile
from contextlib import nullcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import pstats
import random
import resource
import signal
import subprocess
import sys
import time
import tracemalloc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--seconds', type=float, default=140)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--engine', choices=['legacy', 'sparse-regression', 'grouped-regression', 'cached-grouped-regression', 'compact-covariance', 'lean-regression', 'cached-inner', 'compiled-covariance', 'parallel-prefilter'], default='legacy')
    p.add_argument('--prepare-engine', action='store_true', help='Measure engine preparation separately before inference; whole-screen time still includes it')
    p.add_argument('--memory-stages', action='store_true', help='Diagnostic allocation tracing; timings are not performance-screen estimates')
    p.add_argument('--write-outputs', action='store_true')
    p.add_argument('--fixed-budget-validation', action='store_true', help='Label a longer matched-budget validation; completion is not convergence')
    p.add_argument('--trace-stride', type=int, default=1, help='Keep first, every Nth and budget-final update arrays')
    p.add_argument('--monitor', action='store_true', help='Save scalar progress, diagnostics and inner-fit timing')
    args = p.parse_args()
    if args.iterations < 1 or args.seconds <= 0 or args.trace_stride < 1:
        p.error('Iterations and seconds must be positive')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(args.repo).resolve()
    fixture = Path(args.fixture).resolve()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise ValueError('Screening requires a clean committed source checkout')
    sys.path.insert(0, str(repo/'src'))
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    receipt = dict(status='RUNNING', purpose='SCREENING_ONLY', args=vars(args), commit=commit,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.fixed_budget_validation:
        receipt['purpose'] = 'MATCHED_FIXED_BUDGET_VALIDATION_NOT_CONVERGENCE'
    receipt['pid'] = os.getpid()
    receipt['started_unix'] = time.time()
    started = time.perf_counter()
    if args.memory_stages:
        tracemalloc.start(1)
    memory_originals = {}
    state_module = None
    inner_class = None
    original_inner = None
    monitor_originals = {}
    progress_handle = None
    trace_indices = []
    def json_default(value):
        if hasattr(value, 'item'):
            return value.item()
        raise TypeError(type(value).__name__)
    def write_progress():
        snapshot = {**receipt, 'elapsed_seconds': time.perf_counter() - started,
                    'peak_rss_bytes': peak_rss()}
        temp = out/'progress.json.tmp'
        temp.write_text(json.dumps(snapshot, indent=2, default=json_default)+'\n')
        temp.replace(out/'progress.json')
    def peak_rss():
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    def memory_sample(stage):
        if args.memory_stages:
            current, peak = tracemalloc.get_traced_memory()
            receipt.setdefault('memory_stages', []).append(dict(
                stage=stage, peak_rss_bytes=peak_rss(), traced_current_bytes=current, traced_peak_bytes=peak))
    memory_sample('before_imports')
    profiler = cProfile.Profile() if args.profile else None
    def timeout(signum, frame):
        raise TimeoutError('Screening wall-time budget exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        import numpy as np
        metadata = json.loads(fixture.with_suffix(fixture.suffix+'.json').read_text())
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if metadata['format'] != 1 or digest != metadata['sha256']:
            raise ValueError('Fixture checksum/format mismatch')
        receipt['fixture_sha256'] = digest
        memory_sample('before_fixture_load')
        with gzip.open(fixture, 'rb') as handle:
            data = pickle.load(handle)
        memory_sample('after_fixture_load')
        np.random.seed(args.seed)
        random.seed(args.seed)
        kwargs = dict(data['kwargs'])
        kwargs.update(max_num_iter=args.iterations, total_num_iter=args.iterations)
        receipt['inference_kwargs'] = kwargs
        receipt['setup_seconds'] = time.perf_counter() - started
        trace = {'prior': [], 'beta_uncorrected': [], 'beta': []}
        import pigean.state as state_module
        if args.monitor:
            progress_handle = gzip.open(out/'updates.jsonl.gz', 'wt')
            receipt['inner_fit_seconds'] = 0.0
            receipt['inner_fit_calls'] = 0
            inner_class = type(data['state'])
            original_inner = inner_class._calculate_non_inf_betas
            def timed_inner(*positional, **inputs):
                inner_started = time.perf_counter()
                try:
                    return original_inner(*positional, **inputs)
                finally:
                    receipt['inner_fit_seconds'] += time.perf_counter() - inner_started
                    receipt['inner_fit_calls'] += 1
            inner_class._calculate_non_inf_betas = timed_inner
            name = '_log_gibbs_post_burn_diagnostics'
            monitor_originals[name] = getattr(state_module, name)
            def record_diagnostics(*positional, **inputs):
                receipt['latest_post_burn_diagnostics'] = inputs.copy()
                return monitor_originals['_log_gibbs_post_burn_diagnostics'](*positional, **inputs)
            setattr(state_module, name, record_diagnostics)
            name = '_apply_gibbs_final_state'
            monitor_originals[name] = getattr(state_module, name)
            def record_final_state(state, final_summary, adjust_priors):
                receipt['final_summary_scalars'] = {
                    key: value for key, value in final_summary.items()
                    if value is None or isinstance(value, (str, bool, int, float, np.generic))}
                return monitor_originals['_apply_gibbs_final_state'](state, final_summary, adjust_priors)
            setattr(state_module, name, record_final_state)
            write_progress()
        if args.memory_stages:
            def wrap_memory(fn, name):
                def wrapped(*positional, **inputs):
                    memory_sample(name+':before')
                    try:
                        return fn(*positional, **inputs)
                    finally:
                        memory_sample(name+':after')
                return wrapped
            for name in ('_initialize_gibbs_epoch_state', '_prepare_gibbs_iteration_inputs',
                         '_compute_gibbs_linear_prefilter_betas', '_compute_gibbs_logistic_outputs_for_batches'):
                memory_originals[name] = getattr(state_module, name)
                setattr(state_module, name, wrap_memory(memory_originals[name], name))
        original_update = state_module._compute_gibbs_iteration_betas_and_priors
        def record_update(*positional, **inputs):
            memory_sample('coefficients_and_priors:before')
            result = original_update(*positional, **inputs)
            memory_sample('coefficients_and_priors:after')
            update = receipt.get('outer_updates_executed', 0) + 1
            if args.write_outputs and (update == 1 or update % args.trace_stride == 0 or update == args.iterations - 1):
                trace['prior'].append(inputs['epoch_priors']['priors_mean_m'].copy())
                trace['beta_uncorrected'].append(inputs['iter_state']['uncorrected_betas_mean_m'].copy())
                trace['beta'].append(result['full_betas_mean_m'].copy())
                trace_indices.append(update)
            receipt['outer_updates_executed'] = update
            if args.monitor:
                receipt['inference_seconds_so_far'] = time.perf_counter() - t0
                receipt['inference_cpu_seconds_so_far'] = cpu_seconds() - inference_cpu_start
                receipt['last_epoch_iteration'] = inputs['iter_state'].get('epoch_iter_num')
                receipt['last_global_iteration'] = inputs['iter_state'].get('total_iter_num')
                progress_handle.write(json.dumps({
                    key: receipt[key] for key in ('outer_updates_executed', 'last_epoch_iteration',
                    'last_global_iteration', 'inference_seconds_so_far', 'inference_cpu_seconds_so_far',
                    'inner_fit_seconds', 'inner_fit_calls')}, default=json_default)+'\n')
                if update % 10 == 0:
                    progress_handle.flush()
                write_progress()
            return result
        state_module._compute_gibbs_iteration_betas_and_priors = record_update
        def cpu_seconds():
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_utime + usage.ru_stime
        preparation_cpu_start = cpu_seconds()
        preparation_start = time.perf_counter()
        compiled_module = None
        if args.prepare_engine and args.engine in ('compiled-covariance', 'parallel-prefilter'):
            os.environ['NUMBA_CACHE_DIR'] = str(out/'numba_cache')
            import pegs_shared.compiled_covariance as compiled_module
            import numba
            import llvmlite
            compiled_module.prepare_compiled_covariance()
            receipt['compiler_versions'] = dict(numba=numba.__version__, llvmlite=llvmlite.__version__)
            receipt['compiled_signatures_before'] = [str(x) for x in compiled_module._edge_kernel.signatures]
        receipt['engine_preparation_seconds'] = time.perf_counter() - preparation_start
        receipt['engine_preparation_cpu_seconds'] = cpu_seconds() - preparation_cpu_start
        receipt['engine_prepared'] = True
        receipt['setup_seconds'] = time.perf_counter() - started
        inference_cpu_start = cpu_seconds()
        t0 = time.perf_counter()
        backend = nullcontext()
        if args.engine in ('sparse-regression', 'grouped-regression', 'cached-grouped-regression', 'compact-covariance', 'lean-regression', 'cached-inner', 'compiled-covariance', 'parallel-prefilter'):
            from pegs_shared.sparse_regression import marginal_backend
            backend = marginal_backend({'sparse-regression': 'sparse', 'grouped-regression': 'grouped',
                                        'cached-grouped-regression': 'grouped-cached',
                                        'compact-covariance': 'compact-covariance',
                                        'lean-regression': 'lean-regression', 'cached-inner': 'cached-inner',
                                        'compiled-covariance': 'compiled-covariance', 'parallel-prefilter': 'parallel-prefilter'}[args.engine])
        if profiler:
            profiler.enable()
        try:
            receipt['peak_rss_before_inference_bytes'] = peak_rss()
            memory_sample('before_inference')
            with backend as structure_cache:
                data['state'].run_gibbs(**kwargs)
            if structure_cache is not None:
                receipt['structure_cache'] = dict(hits=structure_cache.hits, misses=structure_cache.misses,
                                                entries=len(structure_cache.entries),
                                                payload_bytes=structure_cache.payload_bytes)
                if hasattr(structure_cache, 'covariance_stats'):
                    receipt['covariance_stats'] = structure_cache.covariance_stats
        finally:
            state_module._compute_gibbs_iteration_betas_and_priors = original_update
            if profiler:
                profiler.disable()
            receipt['inference_seconds'] = time.perf_counter() - t0
            receipt['inference_cpu_seconds'] = cpu_seconds() - inference_cpu_start
            receipt['peak_rss_through_inference_bytes'] = peak_rss()
            if compiled_module is not None:
                receipt['compiled_signatures_after'] = [str(x) for x in compiled_module._edge_kernel.signatures]
                receipt['compilation_during_inference'] = (receipt['compiled_signatures_before'] != receipt['compiled_signatures_after'])
            memory_sample('after_inference')
        if args.write_outputs:
            with gzip.open(out/'iteration_arrays.pkl.gz', 'wb') as handle:
                pickle.dump({**{key: np.array(value) for key, value in trace.items()},
                             'update_indices': trace_indices,
                             'genes': data['state'].genes, 'gene_sets': data['state'].gene_sets}, handle, protocol=5)
            data['state'].write_gene_statistics(str(out/'gene_stats.out.gz'), output_detail='full')
            data['state'].write_gene_set_statistics(str(out/'gene_set_stats.out.gz'), output_detail='full')
        if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != commit or subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Source changed during execution')
        receipt['trace_update_indices'] = trace_indices
        receipt['status'] = 'COMPLETE_FIXED_BUDGET' if args.fixed_budget_validation else 'COMPLETE_SCREEN'
    except TimeoutError as error:
        receipt.update(status='TIME_LIMIT', error=str(error))
    except BaseException as error:
        receipt.update(status='ERROR', error=repr(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if state_module is not None:
            for name, original in memory_originals.items():
                setattr(state_module, name, original)
            for name, original in monitor_originals.items():
                setattr(state_module, name, original)
        if original_inner is not None:
            inner_class._calculate_non_inf_betas = original_inner
        if progress_handle is not None:
            progress_handle.close()
        if args.memory_stages:
            tracemalloc.stop()
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(out/'profile.pstats'))
            with (out/'profile.txt').open('w') as handle:
                stats = pstats.Stats(profiler, stream=handle)
                stats.sort_stats('cumulative').print_stats(45)
                stats.sort_stats('tottime').print_stats(35)
        receipt['wall_seconds'] = time.perf_counter() - started
        receipt['peak_rss_bytes'] = peak_rss()
        (out/'screen.json').write_text(json.dumps(receipt, indent=2, default=json_default)+'\n')
        if args.monitor:
            write_progress()
    print(json.dumps({k: receipt[k] for k in ('status', 'wall_seconds')}))
    if receipt['status'] not in ('COMPLETE_SCREEN', 'COMPLETE_FIXED_BUDGET'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
