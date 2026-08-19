"""Launch the same PIGEAN command once per random seed.

Two execution strategies are supported:

* ``sequential`` — one run at a time, in seed order.
* ``parallel``   — up to ``--workers`` runs at once.

Both produce byte-for-byte the same per-run outputs; the strategy only changes
wall-clock and peak memory. Each run is a separate ``python -m pigean``
subprocess, so a crash in one seed cannot corrupt another, and a partially
finished sweep can be resumed with ``--resume``.

Seeding note (this is load-bearing, not cosmetic): every run is given BOTH
``--deterministic`` and ``--seed N``. ``--seed`` alone only pins the legacy
global ``random``/``numpy.random`` state; several pre-Gibbs paths (gene-set
batching, hyper subsampling) draw from an unseeded ``default_rng``. Without
``--deterministic`` those draws vary independently of the seed and run-to-run
spread would be confounded with un-pinned sampling rather than measuring the
seed effect we are after.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Serialises console writes so interleaved worker lines stay whole.
_CONSOLE_LOCK = threading.Lock()


def _log(message: str) -> None:
    """Print and flush.

    A sweep is usually launched into a redirect or a background task, where
    bare print() is block-buffered and nothing appears until the process exits
    -- which is exactly when progress output stops being useful.
    """
    print(message, flush=True)

# Output flag -> file basename written into each run directory.
OUTPUT_FLAGS = {
    "gene_stats": ("--gene-stats-out", "gene_stats.tsv.gz"),
    "gene_set_stats": ("--gene-set-stats-out", "gene_set_stats.tsv.gz"),
    "gene_gene_set_stats": ("--gene-gene-set-stats-out", "gene_gene_set_stats.tsv.gz"),
    "gene_set_overlap_stats": ("--gene-set-overlap-stats-out", "gene_set_overlap_stats.tsv.gz"),
    "params": ("--params-out", "params.tsv"),
    "gene_covs": ("--gene-covs-out", "gene_covs.tsv.gz"),
    "gene_effectors": ("--gene-effectors-out", "gene_effectors.tsv.gz"),
}

DEFAULT_OUTPUTS = ("gene_stats", "gene_set_stats", "gene_gene_set_stats", "params")

# Applied to every run unless the sweep config sets them itself. These only
# affect logging verbosity, never the numbers.
QUIET_FLAGS = ("--hide-opts", "--hide-progress")

# BLAS/OpenMP thread caps. PIGEAN is largely single-threaded numpy, but the
# linear algebra underneath is not; leaving it uncapped means W parallel workers
# each spawn N threads and the box thrashes.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class SweepConfig:
    """A sweep definition: one PIGEAN command, many seeds."""

    def __init__(self, raw: dict, config_path: Path | None = None):
        self.raw = raw
        self.config_path = config_path
        self.name = raw.get("name") or (config_path.stem if config_path else "sweep")
        self.mode = raw.get("mode", "gibbs")
        self.args = raw.get("args", {})
        self.seeds = list(raw.get("seeds", []))
        self.outputs = list(raw.get("outputs", DEFAULT_OUTPUTS))
        self.description = raw.get("description", "")
        if not isinstance(self.args, dict):
            raise ValueError("sweep config 'args' must be an object")
        unknown = [name for name in self.outputs if name not in OUTPUT_FLAGS]
        if unknown:
            raise ValueError(
                "unknown outputs %s; known: %s" % (unknown, sorted(OUTPUT_FLAGS))
            )

    @classmethod
    def load(cls, path: Path) -> "SweepConfig":
        with open(path) as fh:
            return cls(json.load(fh), config_path=path)


def _flagify(key: str) -> str:
    key = key if key.startswith("-") else "--" + key
    return key.replace("_", "-")


def build_command(
    config: SweepConfig,
    seed: int,
    run_dir: Path,
    python: str,
    repo_root: Path,
    *,
    stream: bool = False,
    remote_map: dict | None = None,
) -> list[str]:
    """Assemble the argv for one seeded run.

    Config args are resolved relative to ``repo_root`` when they look like
    existing repo-relative paths, so a sweep config stays portable and does not
    have to hardcode absolute paths.
    """
    cmd = [python, "-m", "pigean", config.mode]

    for flag in QUIET_FLAGS:
        if flag.lstrip("-").replace("-", "_") not in {k.lstrip("-").replace("-", "_") for k in config.args}:
            cmd.append(flag)

    for key, value in config.args.items():
        flag = _flagify(key)
        if flag in ("--seed", "--deterministic"):
            # Seeding is owned by the sweep, not the config.
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is True:
                cmd.append(flag)
            elif item is False or item is None:
                continue
            else:
                text = str(item)
                if remote_map and text in remote_map:
                    text = remote_map[text]
                cmd.extend([flag, _resolve_path_like(text, repo_root)])

    # Sweep-owned seeding. See the module docstring for why both flags.
    cmd.extend(["--deterministic", "--seed", str(seed)])

    for name in config.outputs:
        flag, basename = OUTPUT_FLAGS[name]
        cmd.extend([flag, str(run_dir / basename)])
    if stream:
        # --log-file diverts PIGEAN's progress off stderr entirely (stderr.txt
        # comes out empty), so there would be nothing to stream. When streaming,
        # the runner takes the log over: it tees the piped output to run.log
        # itself, so the file still exists and the console still sees the run.
        pass
    else:
        cmd.extend(["--log-file", str(run_dir / "run.log.gz")])
    cmd.extend(["--warnings-file", str(run_dir / "warnings.log.gz")])

    return cmd


def _resolve_path_like(value: str, repo_root: Path) -> str:
    """Expand a repo-relative path, leaving non-path values untouched."""
    if os.path.isabs(value) or value.startswith("dig-open-data:") or is_remote(value):
        return value
    candidate = repo_root / value
    if candidate.exists():
        return str(candidate)
    return value


def is_remote(value: str) -> bool:
    return isinstance(value, str) and value.lower().startswith(("http://", "https://", "ftp://"))


def _remote_size(url: str):
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError, OSError):
        return None


def download_once(url: str, cache_dir: Path, log=_log) -> str:
    """Fetch a remote input into ``cache_dir``, reusing a complete prior copy.

    PIGEAN reads https URLs natively, so a sweep *could* let every seed stream
    its own copy. It should not: the T2D bottom-line sumstats is 8.3 GB, so
    three seeds would pull 25 GB and twenty would pull 166 GB, and each run
    would be gated on the network rather than on PIGEAN. Fetching once also
    removes any doubt that every seed saw identical input bytes, which matters
    when the whole point is attributing differences to the seed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / os.path.basename(urllib.parse.urlparse(url).path)
    expected = _remote_size(url)

    if target.exists():
        actual = target.stat().st_size
        if expected is None or actual == expected:
            log("cached input: %s (%s, %.1f GB)" % (target.name, url, actual / 1e9))
            return str(target)
        log(
            "cached %s is %d bytes but remote is %d; re-fetching"
            % (target.name, actual, expected)
        )

    partial = target.with_suffix(target.suffix + ".part")
    log(
        "fetching %s -> %s%s"
        % (url, target, "" if expected is None else " (%.1f GB)" % (expected / 1e9))
    )
    started = time.time()
    with urllib.request.urlopen(url) as response, open(partial, "wb") as fh:
        while True:
            chunk = response.read(8 << 20)
            if not chunk:
                break
            fh.write(chunk)
    # Rename only after the body is fully written, so an interrupted fetch can
    # never be mistaken for a usable cache entry on the next run.
    partial.replace(target)
    log("fetched %s in %.0fs" % (target.name, time.time() - started))
    return str(target)


def materialize_remote_inputs(config: SweepConfig, cache_dir: Path, *, log=_log) -> dict:
    """Download every remote value in the config once; return url -> local path."""
    mapping = {}
    for value in config.args.values():
        for item in value if isinstance(value, list) else [value]:
            if is_remote(item) and item not in mapping:
                mapping[item] = download_once(item, cache_dir, log=log)
    return mapping


def _run_env(repo_root: Path, threads_per_worker: int) -> dict:
    env = dict(os.environ)
    src_root = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
    )
    # PYTHONHASHSEED matters: PIGEAN iterates over sets in a few places, and
    # unpinned hash randomization would inject a second, uncontrolled source of
    # run-to-run variation on top of the seed we are actually studying.
    env["PYTHONHASHSEED"] = "0"
    if threads_per_worker > 0:
        for var in THREAD_ENV_VARS:
            env[var] = str(threads_per_worker)
    return env


def _tee(pipe, sink, prefix: str, pattern, echo: bool) -> None:
    """Copy one pipe to a file, echoing matching lines to the console prefixed.

    The file always gets every line; ``pattern`` only filters what reaches the
    terminal, so a grep never costs you the full log.
    """
    for line in pipe:
        sink.write(line)
        if echo and (pattern is None or pattern.search(line)):
            with _CONSOLE_LOCK:
                sys.stdout.write(prefix + line.rstrip("\n") + "\n")
                sys.stdout.flush()
    pipe.close()


def _run_streaming(cmd, run_dir: Path, cwd: str, env: dict, seed: int, pattern) -> int:
    """Run one seed with its output piped back so several runs can be watched.

    stdout and stderr are merged into one ordered stream — PIGEAN interleaves
    them and splitting them would scramble the narrative of a single run.
    """
    prefix = "[seed %d] " % seed
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    # buffering=1 (line buffered) so run.log can be tailed during the run
    # rather than appearing all at once when the process exits.
    with open(run_dir / "run.log", "w", buffering=1) as sink:
        _tee(proc.stdout, sink, prefix, pattern, echo=True)
    return proc.wait()


def run_one_seed(
    config: SweepConfig,
    seed: int,
    out_dir: Path,
    *,
    python: str,
    repo_root: Path,
    threads_per_worker: int,
    resume: bool,
    dry_run: bool,
    remote_map: dict | None = None,
    stream: bool = False,
    stream_pattern=None,
    log=_log,
) -> dict:
    run_dir = out_dir / "runs" / ("seed_%d" % seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"

    if resume and status_path.exists():
        with open(status_path) as fh:
            prior = json.load(fh)
        if prior.get("returncode") == 0:
            log("seed %d: reusing completed run at %s" % (seed, run_dir))
            return prior

    cmd = build_command(
        config, seed, run_dir, python, repo_root, stream=stream, remote_map=remote_map
    )
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")

    if dry_run:
        log("seed %d (dry run): %s" % (seed, " ".join(cmd)))
        return {"seed": seed, "run_dir": str(run_dir), "returncode": None, "dry_run": True}

    log("seed %d: starting" % seed)
    started = time.time()
    env = _run_env(repo_root, threads_per_worker)
    if stream:
        returncode = _run_streaming(cmd, run_dir, str(repo_root), env, seed, stream_pattern)
    else:
        with open(run_dir / "stdout.txt", "w") as out_fh, open(run_dir / "stderr.txt", "w") as err_fh:
            returncode = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=out_fh,
                stderr=err_fh,
                check=False,
            ).returncode
    elapsed = time.time() - started

    status = {
        "seed": seed,
        "run_dir": str(run_dir),
        "returncode": returncode,
        "wall_seconds": round(elapsed, 2),
        "command": cmd,
        "outputs": {
            name: str(run_dir / OUTPUT_FLAGS[name][1]) for name in config.outputs
        },
    }
    with open(status_path, "w") as fh:
        json.dump(status, fh, indent=2)

    if returncode == 0:
        log("seed %d: done in %.1fs" % (seed, elapsed))
    else:
        where = run_dir / ("run.log" if stream else "stderr.txt")
        log("seed %d: FAILED rc=%d (see %s)" % (seed, returncode, where))
    return status


def run_sweep(
    config: SweepConfig,
    out_dir: Path,
    *,
    strategy: str = "parallel",
    workers: int = 3,
    python: str | None = None,
    repo_root: Path | None = None,
    threads_per_worker: int = 1,
    resume: bool = False,
    dry_run: bool = False,
    stream: bool = False,
    stream_grep: str | None = None,
    cache_dir: Path | None = None,
    stream_remote: bool = False,
    log=_log,
) -> dict:
    python = python or sys.executable
    stream_pattern = re.compile(stream_grep) if stream_grep else None
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(config.seeds)
    if not seeds:
        raise ValueError("no seeds to run")

    if strategy == "sequential":
        effective_workers = 1
    elif strategy == "parallel":
        effective_workers = max(1, min(workers, len(seeds)))
    else:
        raise ValueError("strategy must be 'sequential' or 'parallel', got %r" % strategy)

    log(
        "sweep %r: %d seed(s) %s, strategy=%s workers=%d"
        % (config.name, len(seeds), seeds, strategy, effective_workers)
    )

    # Remote inputs are fetched once, before any seed starts, so the workers
    # never race to download the same file into the same cache path.
    remote_map = {}
    if not dry_run and not stream_remote:
        remote_map = materialize_remote_inputs(
            config, cache_dir or (out_dir / "_inputs"), log=log
        )

    started = time.time()

    def _work(seed: int) -> dict:
        return run_one_seed(
            config,
            seed,
            out_dir,
            python=python,
            repo_root=repo_root,
            threads_per_worker=threads_per_worker,
            resume=resume,
            dry_run=dry_run,
            remote_map=remote_map,
            stream=stream,
            stream_pattern=stream_pattern,
            log=log,
        )

    if effective_workers == 1:
        statuses = [_work(seed) for seed in seeds]
    else:
        # Threads, not processes: each task spends its whole life blocked in
        # subprocess.run, so the GIL is released and a thread pool is the
        # cheapest correct way to cap concurrency.
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            statuses = list(pool.map(_work, seeds))

    elapsed = time.time() - started
    manifest = {
        "name": config.name,
        "description": config.description,
        "mode": config.mode,
        "config_path": str(config.config_path) if config.config_path else None,
        "seeds": seeds,
        "strategy": strategy,
        "workers": effective_workers,
        "threads_per_worker": threads_per_worker,
        "stream": stream,
        "remote_inputs": remote_map,
        "outputs": config.outputs,
        "python": python,
        "repo_root": str(repo_root),
        "args": config.args,
        "sweep_wall_seconds": round(elapsed, 2),
        "runs": statuses,
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    failed = [s["seed"] for s in statuses if s.get("returncode") not in (0, None)]
    if failed:
        log("sweep finished in %.1fs with FAILURES on seeds %s" % (elapsed, failed))
    else:
        log("sweep finished in %.1fs, all %d seed(s) ok" % (elapsed, len(statuses)))
    return manifest
