#!/usr/bin/env python3
"""Polish the final global population with local search.

Takes the 15 plans of global search's last-step population, runs an
independent local search (12 steps, 5 variants per step) on each, and
prepares the entrants file for the held-out placement championship.
Rationale: local search is cheap and reliable at improving a plan; global
search discovers promising positions; combining them -- global discovers,
local polishes -- is a division of labor worth measuring.

Phases (each idempotent/resumable):
  extract  - write in/polish-seeds.csv (g12-s00..s14) + out/<run>-polish-map.csv
  run      - launch all 15 local_search.py subprocesses in parallel
  collect  - gather each run's final plan; write out/<run>-polish-entrants.csv
             containing the 15 step-12 PARENTS (kind=g12-parent) and the 15
             polished FINALS (kind=polished-descendant)
"""

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

csv.field_size_limit(10**9)

HERE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Polish the final global population.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--per-run-concurrency", type=int, default=8)
    ap.add_argument("--model", default="deepseek/deepseek-v3.2")
    ap.add_argument("--phase", choices=("extract", "run", "collect", "all"),
                    default="all")
    ap.add_argument("--mock", action="store_true")
    return ap.parse_args()


def seed_id(slot: int) -> str:
    return f"g12-s{slot:02d}"


def extract(run_id: str) -> list[dict]:
    src = HERE / "out" / f"{run_id}-global-candidates.csv"
    rows = [r for r in csv.DictReader(open(src, newline="", encoding="utf-8"))
            if r["step"] == "12"]
    assert len(rows) == 15, f"expected 15 step-12 rows, got {len(rows)}"
    rows.sort(key=lambda r: int(r["slot"]))

    seeds_path = HERE / "in" / "polish-seeds.csv"
    with open(seeds_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "description"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({"id": seed_id(int(r["slot"])), "description": r["plan_text"]})

    map_path = HERE / "out" / f"{run_id}-polish-map.csv"
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "seed_id", "slot", "rank", "win_rate", "origin", "root_lineage",
            "candidate_id"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({
                "seed_id": seed_id(int(r["slot"])), "slot": r["slot"],
                "rank": r["rank"], "win_rate": r["win_rate"],
                "origin": r["origin"][:40], "root_lineage": r["root_lineage"],
                "candidate_id": r["candidate_id"],
            })
    print(f"extract: wrote {seeds_path.name} and {map_path.name}")
    return rows


def run_one(run_id: str, slot: int, steps: int, concurrency: int,
            model: str, mock: bool) -> tuple[int, int]:
    sid = seed_id(slot)
    prefix = f"{run_id}-polish-s{slot:02d}"
    marker = HERE / "out" / f"{prefix}-local-search-narrative.txt"
    if marker.exists():
        print(f"{sid}: already complete, skipping")
        return slot, 0
    env = dict(os.environ)
    env["AI_ENTREP_MAX_CONCURRENT_CALLS"] = str(concurrency)
    env["AI_ENTREP_CALL_CACHE"] = str(HERE / "out" / f"{prefix}-call-cache.sqlite3")
    if mock:
        env["AI_ENTREP_MOCK_LLM"] = "1"
    cmd = [sys.executable, str(HERE / "src" / "local_search.py"),
           "--seed-id", sid,
           "--input-file", str(HERE / "in" / "polish-seeds.csv"),
           "--model", model,
           "--steps", str(steps),
           "--run-prefix", prefix,
           "--output-dir", str(HERE / "out")]
    proc = subprocess.run(cmd, env=env, cwd=str(HERE),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"{sid}: FAILED rc={proc.returncode}\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
    else:
        print(f"{sid}: done")
    return slot, proc.returncode


def collect(run_id: str) -> None:
    map_path = HERE / "out" / f"{run_id}-polish-map.csv"
    parents = list(csv.DictReader(open(map_path, newline="", encoding="utf-8")))
    seeds = {r["id"]: r["description"] for r in csv.DictReader(
        open(HERE / "in" / "polish-seeds.csv", newline="", encoding="utf-8"))}

    entrants_path = HERE / "out" / f"{run_id}-polish-entrants.csv"
    with open(entrants_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "kind", "plan_text"],
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        for p in parents:
            sid = p["seed_id"]
            w.writerow({"id": f"{sid}-parent (step-12 rank {p['rank']})",
                        "kind": "g12-parent", "plan_text": seeds[sid]})
            cand = HERE / "out" / f"{run_id}-polish-{sid.replace('g12-', '')}-local-candidates.csv"
            last = None
            for row in csv.DictReader(open(cand, newline="", encoding="utf-8")):
                if row["selected"] == "True":
                    last = row
            assert last is not None, f"no selected state in {cand}"
            w.writerow({"id": f"{sid}-polished ({last['candidate_id']})",
                        "kind": "polished-descendant",
                        "plan_text": last["plan_text"]})
    print(f"collect: wrote {entrants_path.name} with {2 * len(parents)} entrants")


def main() -> None:
    args = parse_args()
    if args.phase in ("extract", "all"):
        extract(args.run_id)
    if args.phase in ("run", "all"):
        with ThreadPoolExecutor(max_workers=15) as pool:
            results = list(pool.map(
                lambda slot: run_one(args.run_id, slot, args.steps,
                                     args.per_run_concurrency, args.model,
                                     args.mock),
                range(15)))
        failures = [s for s, rc in results if rc != 0]
        if failures:
            raise SystemExit(f"polish runs failed for slots: {failures} "
                             "(rerun --phase run; completed runs are skipped)")
    if args.phase in ("collect", "all"):
        collect(args.run_id)


if __name__ == "__main__":
    main()
