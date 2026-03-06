#!/usr/bin/env python3
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time


def _to_kb(ru_maxrss: float) -> float:
    if platform.system() == "Darwin":
        return float(ru_maxrss) / 1024.0
    return float(ru_maxrss)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command and record elapsed/max RSS")
    parser.add_argument("--metrics-out", required=True, help="Path to JSON metrics output")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run after --")
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided")

    start = time.time()
    proc = subprocess.run(cmd, check=False)
    end = time.time()

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    metrics = {
        "command": cmd,
        "cwd": os.getcwd(),
        "exit_code": int(proc.returncode),
        "elapsed_sec": float(end - start),
        "max_rss_kb": float(_to_kb(usage.ru_maxrss)),
    }

    with open(args.metrics_out, "w") as out_fh:
        json.dump(metrics, out_fh, indent=2, sort_keys=True)
        out_fh.write("\n")

    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
