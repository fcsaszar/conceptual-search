#!/usr/bin/env python3
"""Restricted win rates from a championship's match records.

For every plan in the championship, computes its win rate counting only
matches against the 30 original ventures (for an original, the other 29).
Writes analysis/universe-champ-wr.json ("universe" is the record-level name of
the 30-venture reference set), the data source for the polish figure and the
seed rates in the map and trajectory figures.

Usage: python3 analysis/compute_universe_wr.py --run-id <id>
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    BASE = Path(args.base_dir)
    csv.field_size_limit(10**9)
    R = BASE / "out" / args.run_id

    champ = list(csv.DictReader(open(f"{R}-championship.csv")))
    kinds = {r["id"]: r["kind"] for r in champ}
    originals = {i for i, k in kinds.items() if k == "original"}
    matches = list(csv.DictReader(open(f"{R}-championship-matches.csv")))

    wins, n = defaultdict(float), defaultdict(int)
    for m in matches:
        a, b, win = m["plan_a_id"], m["plan_b_id"], m["winner_id"]
        for me, opp in ((a, b), (b, a)):
            if opp in originals and me != opp:
                n[me] += 1
                if win == me:
                    wins[me] += 1
                elif win not in (a, b):    # tie recorded by the evaluator
                    wins[me] += 0.5

    out = {}
    for r in sorted(champ, key=lambda r: float(r["win_rate"]), reverse=True):
        i = r["id"]
        out[i] = {"kind": r["kind"], "champ_wr": float(r["win_rate"]),
                  "universe_wr": round(wins[i] / n[i], 4), "n_vs_orig": n[i]}
    path = BASE / "analysis" / "universe-champ-wr.json"
    path.write_text(json.dumps(out, indent=1))
    orig = sorted(((v["universe_wr"], k) for k, v in out.items()
                   if v["kind"] == "original"), reverse=True)
    print(f"wrote {path} ({len(out)} plans; originals top3: "
          + ", ".join(f"{k} {wr:.1%}" for wr, k in orig[:3]) + ")")


if __name__ == "__main__":
    main()
