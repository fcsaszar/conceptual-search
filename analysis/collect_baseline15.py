#!/usr/bin/env python3
"""Local-only comparison: assemble its tournament entrants and summarize it.

Phase ``entrants`` (run after analysis/run_baseline15.py): collect the final
concept of each of the 14 additional local searches into
out/<run-id>-baseline15-entrants.csv, the extra-entrants file for the
baseline tournament (src/championship.py --run-prefix <run-id>-baseline15).

Phase ``summary`` (run after that tournament and src/semantic_scores.py):
compute each local-only final's win rate against the 30 reference ventures
from the tournament's match records, add the focal clockchain local final
(measured in the main tournament), write analysis/baseline15-wr.json, and
print the performance and semantic-distance summaries reported in the paper's
robustness section.

Usage:
  python3 analysis/collect_baseline15.py --run-id <id> --phase entrants
  python3 analysis/collect_baseline15.py --run-id <id> --phase summary
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10**9)


def sq(text) -> str:
    return " ".join(str(text).split())


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_seeds(path: Path) -> list[str]:
    seeds = [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [seed for seed in seeds if seed]


def final_state(candidates_path: Path) -> dict:
    last = None
    for row in read_rows(candidates_path):
        if row["selected"] == "True":
            last = row
    if last is None:
        raise SystemExit(f"No selected state in {candidates_path}")
    return last


def write_entrants(base: Path, run_id: str, seeds: list[str], skip: str) -> Path:
    out_path = base / "out" / f"{run_id}-baseline15-entrants.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "kind", "plan_text"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for seed in sorted(seed for seed in seeds if seed != skip):
            last = final_state(base / "out" / f"{run_id}-baseline15-{seed}-local-candidates.csv")
            writer.writerow(
                {
                    "id": f"baseline-{seed} ({last['candidate_id']})",
                    "kind": "baseline-local",
                    "plan_text": last["plan_text"],
                }
            )
    print(f"wrote {out_path.relative_to(base)}")
    return out_path


def restricted_win_rates(base: Path, prefix: str) -> dict[str, float]:
    """Win rate of each non-original entrant counting only matches against the
    30 original ventures (the same rule as compute_universe_wr.py)."""
    entrants = read_rows(base / "out" / f"{prefix}-championship.csv")
    kinds = {row["id"]: row["kind"] for row in entrants}
    originals = {plan_id for plan_id, kind in kinds.items() if kind == "original"}
    wins: dict[str, float] = defaultdict(float)
    matches: dict[str, int] = defaultdict(int)
    for match in read_rows(base / "out" / f"{prefix}-championship-matches.csv"):
        a, b, winner = match["plan_a_id"], match["plan_b_id"], match["winner_id"]
        for me, opponent in ((a, b), (b, a)):
            if opponent in originals and me != opponent:
                matches[me] += 1
                if winner == me:
                    wins[me] += 1
                elif winner not in (a, b):
                    wins[me] += 0.5
    return {
        plan_id: round(wins[plan_id] / matches[plan_id], 4)
        for plan_id in sorted(kinds)
        if kinds[plan_id] == "baseline-local"
    }


def summarize(base: Path, run_id: str, seeds: list[str], skip: str) -> None:
    rates = restricted_win_rates(base, f"{run_id}-baseline15")
    main_rates = json.loads((base / "analysis" / "universe-champ-wr.json").read_text(encoding="utf-8"))
    focal = [value for value in main_rates.values() if value["kind"] == "evolved-local"]
    if len(focal) != 1:
        raise SystemExit("Expected exactly one focal local final in analysis/universe-champ-wr.json")
    rates[f"baseline-{skip} (main local run)"] = focal[0]["universe_wr"]
    out_path = base / "analysis" / "baseline15-wr.json"
    out_path.write_text(json.dumps(rates, indent=1))
    print(f"wrote {out_path.relative_to(base)} ({len(rates)} local-only finals)")

    originals = {key: value["universe_wr"] for key, value in main_rates.items() if value["kind"] == "original"}
    values = list(rates.values())
    gains = []
    for key, value in rates.items():
        seed = re.match(r"baseline-(.+?) \(", key).group(1)
        gains.append(value - originals[seed])
    print(
        "local-only finals: median %.1f%%, best %.1f%%, min %.1f%%; %d/%d improved on their seed "
        "(median gain %.1f pp)"
        % (
            100 * statistics.median(values),
            100 * max(values),
            100 * min(values),
            sum(gain > 0 for gain in gains),
            len(gains),
            100 * statistics.median(gains),
        )
    )

    # Semantic distances (component-level BGE-M3 distance to the nearest seed).
    distances: dict[str, float] = {}
    for name in (f"{run_id}-baseline15-semantic-scores.csv", f"{run_id}-semantic-scores.csv",
                 f"{run_id}-polish-semantic-scores.csv"):
        path = base / "out" / name
        if path.exists():
            for row in read_rows(path):
                distances.setdefault(sq(row["plan_text"]), float(row["semantic_novelty"]))

    def report(label: str, texts: list[str]) -> None:
        found = [distances[sq(text)] for text in texts if sq(text) in distances]
        if len(found) != len(texts):
            print(f"{label}: distances available for {len(found)} of {len(texts)} concepts")
        if found:
            print("%s: median distance %.3f, max %.3f (n=%d)" % (label, statistics.median(found), max(found), len(found)))

    baseline_texts = [row["plan_text"] for row in read_rows(base / "out" / f"{run_id}-baseline15-entrants.csv")]
    baseline_texts.append(final_state(base / "out" / f"{run_id}-local-candidates.csv")["plan_text"])
    report("local-only finals (15)", baseline_texts)
    polish_rows = read_rows(base / "out" / f"{run_id}-polish-entrants.csv")
    report("global search final generation, before polish (15)",
           [row["plan_text"] for row in polish_rows if row["kind"] == "g12-parent"])
    report("polished descendants (15)",
           [row["plan_text"] for row in polish_rows if row["kind"] == "polished-descendant"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--firm-set", default="in/firm-set-15-evalrank.txt")
    ap.add_argument("--skip", default="clockchain", help="Seed searched by the main run's local stage.")
    ap.add_argument("--phase", choices=("entrants", "summary"), required=True)
    args = ap.parse_args()
    base = Path(args.base_dir)
    seeds = read_seeds(base / args.firm_set)
    if args.phase == "entrants":
        write_entrants(base, args.run_id, seeds, args.skip)
    else:
        summarize(base, args.run_id, seeds, args.skip)


if __name__ == "__main__":
    main()
