#!/usr/bin/env python3
"""Parallel-local baseline: 12-round local search from each original seed.

The local-only comparison: skip global search and run local search directly
on every seed. clockchain is skipped (the main run's
local search IS its baseline); the other 14 seeds run as parallel
subprocesses with per-run caches.

Usage: python3 analysis/run_baseline15.py --run-id <id> [--per-run-concurrency 6]
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--firm-set", default="in/firm-set-15-evalrank.txt")
    ap.add_argument("--skip", default="clockchain",
                    help="Seed whose baseline already exists (main local run).")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--per-run-concurrency", type=int, default=6)
    ap.add_argument("--model", default="deepseek/deepseek-v3.2")
    args = ap.parse_args()

    seeds = [l.split("#")[0].strip() for l in open(BASE / args.firm_set)
             if l.split("#")[0].strip()]
    seeds = [s for s in seeds if s != args.skip]
    print(f"baseline: {len(seeds)} local searches (skipping {args.skip})")

    def run(seed):
        prefix = f"{args.run_id}-baseline15-{seed}"
        narrative = BASE / "out" / f"{prefix}-local-search-narrative.txt"
        if narrative.exists() and f"Step {args.steps}" in narrative.read_text():
            print(f"{seed}: already complete")
            return seed, 0
        env = os.environ.copy()
        env["AI_ENTREP_SEED_PROJECT_ID"] = seed
        env["AI_ENTREP_CALL_CACHE"] = str(BASE / "out" / f"{prefix}-call-cache.sqlite3")
        env["AI_ENTREP_MAX_CONCURRENT_CALLS"] = str(args.per_run_concurrency)
        cmd = [sys.executable, "src/local_search.py",
               "--input-file", "in/projects.csv",
               "--seed-id", seed,
               "--steps", str(args.steps),
               "--model", args.model,
               "--run-prefix", prefix,
               "--output-dir", "out"]
        r = subprocess.run(cmd, cwd=BASE, env=env,
                           capture_output=True, text=True)
        tag = "done" if r.returncode == 0 else f"FAILED rc={r.returncode}"
        print(f"{seed}: {tag}")
        if r.returncode != 0:
            print(r.stdout[-500:], r.stderr[-500:])
        return seed, r.returncode
    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        results = list(ex.map(run, seeds))
    bad = [s for s, rc in results if rc != 0]
    print("all done" if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
