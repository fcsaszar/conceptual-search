#!/usr/bin/env python3
"""Held-out championship: evolved plans vs. the FULL 30-project population.

The searches only ever saw the 15 seeds (the bottom half of the reference set
under the evaluator's own ranking).  This stage places the committed global
champion, the final local-search concept, the best hybrid and two commit-round
runners-up, and any extra entrants (for example the polished descendants and
their pre-polish parents) into a full double round robin with all 30 original
ventures, and reports where the generated concepts rank against a population
that includes the withheld top half.  Reference-set win rates are then derived
from the match records by analysis/compute_universe_wr.py.
"""

import argparse
import asyncio
import csv
import logging
import os
import time

import config
from agents_common import Evaluator, get_call_stats, get_usage_stats
from global_search import append_match_records, load_seed_plans
from records import MATCH_FIELDNAMES, initialize_csv

LOG_FORMAT = "%(asctime)s %(message)s"
logging.Formatter.default_msec_format = "%s.%03d"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CHAMPIONSHIP_FIELDNAMES = ["rank", "id", "kind", "wins", "matches", "win_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-population held-out championship.")
    parser.add_argument("--input-file", default=config.INPUT_FILE)
    parser.add_argument("--commit-round", required=True,
                        help="The run's commit-round CSV; the selected row is the champion.")
    parser.add_argument("--local-candidates", required=True,
                        help="The run's local-candidates CSV; the last selected row is the local final.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", default="out")
    parser.add_argument("--entrants-csv", default=None,
                        help="Optional CSV (id, kind, plan_text) of extra "
                             "entrants, e.g. polished descendants and their "
                             "step-12 parents.")
    return parser.parse_args()


def read_champion(path: str) -> tuple[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["selected"] == "True":
                return row["plan_id"], row["plan_text"]
    raise SystemExit(f"No selected champion found in {path}")


def read_commit_runners_up(path: str, champ_plan: str, k: int = 2) -> list[tuple[str, str]]:
    """Top-k non-champion commit entrants by wins (distinct texts): controls
    for the archive benchmark saturating at the top."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: -float(r["wins"]))
    out, seen = [], {" ".join(champ_plan.split())}
    for row in rows:
        key = " ".join(row["plan_text"].split())
        if key in seen:
            continue
        seen.add(key)
        out.append((f"evolved-commit-runnerup ({row['plan_id'][:36]})", row["plan_text"]))
        if len(out) == k:
            break
    return out


def read_best_hybrid(path: str) -> tuple[str, str] | None:
    """Best commit-round entrant whose lineage braids multiple seeds."""
    best = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "X" not in row["plan_id"]:
                continue
            if best is None or float(row["wins"]) > float(best["wins"]):
                best = row
    if best is None:
        return None
    return f"evolved-best-hybrid ({best['plan_id'][:40]}...)", best["plan_text"]


def read_local_final(path: str) -> tuple[str, str]:
    last = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["selected"] == "True":
                last = row
    if last is None:
        raise SystemExit(f"No selected local state found in {path}")
    return f"local-final ({last['candidate_id']})", last["plan_text"]


async def main() -> None:
    args = parse_args()
    if args.model:
        config.set_all_agent_models(args.model)
    if not config.MOCK_LLM:
        config.require_openrouter_api_key()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{args.run_prefix}-championship.csv")
    match_csv = os.path.join(args.output_dir, f"{args.run_prefix}-championship-matches.csv")
    log_path = os.path.join(args.output_dir, f"{args.run_prefix}-championship.log")
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)

    with open(args.input_file, newline="", encoding="utf-8") as f:
        n_all = sum(1 for _ in csv.DictReader(f))
    seeds = load_seed_plans(args.input_file, n_all)

    champ_id, champ_plan = read_champion(args.commit_round)
    local_id, local_plan = read_local_final(args.local_candidates)
    hybrid = read_best_hybrid(args.commit_round)

    ids = [f"evolved-champion ({champ_id})", local_id] + [s[0] for s in seeds]
    plans = [champ_plan, local_plan] + [s[1] for s in seeds]
    kinds = ["evolved-global", "evolved-local"] + ["original"] * len(seeds)
    if hybrid is not None and " ".join(hybrid[1].split()) != " ".join(champ_plan.split()):
        ids.insert(2, hybrid[0])
        plans.insert(2, hybrid[1])
        kinds.insert(2, "evolved-best-hybrid")
    for rid, rplan in read_commit_runners_up(args.commit_round, champ_plan):
        ids.insert(2, rid)
        plans.insert(2, rplan)
        kinds.insert(2, "evolved-commit-runnerup")
    if args.entrants_csv:
        csv.field_size_limit(10**9)
        seen_texts = {" ".join(p.split()) for p in plans}
        with open(args.entrants_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = " ".join(row["plan_text"].split())
                if key in seen_texts:
                    logger.info("Entrant %s duplicates an existing plan; skipped.",
                                row["id"])
                    continue
                seen_texts.add(key)
                ids.append(row["id"])
                plans.append(row["plan_text"])
                kinds.append(row["kind"])

    n = len(plans)
    logger.info("Held-out championship: %d plans, %d matches "
                "(evolved champion + local final + all %d originals)...",
                n, n * (n - 1), len(seeds))
    evaluator = Evaluator()
    ranks, win_rates, match_details = await evaluator.evaluate(plans)

    initialize_csv(match_csv, MATCH_FIELDNAMES)
    append_match_records(match_csv, "championship", 0, ids, match_details)

    order = sorted(range(n), key=lambda i: ranks[i])
    n_matches = 2 * (n - 1)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHAMPIONSHIP_FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for i in order:
            writer.writerow({
                "rank": ranks[i],
                "id": ids[i],
                "kind": kinds[i],
                "wins": f"{win_rates[i] * n_matches:.1f}",
                "matches": n_matches,
                "win_rate": f"{win_rates[i]:.6f}",
            })

    for i in order[:8]:
        logger.info("  rank %2d  %-42s %s  win rate %.0f%%",
                    ranks[i], ids[i][:42], kinds[i], win_rates[i] * 100)
    for i in range(n):
        if kinds[i] == "original":
            continue
        logger.info("RESULT %s (%s): rank %d of %d, win rate %.1f%%",
                    kinds[i], ids[i][:48], ranks[i], n, win_rates[i] * 100)
    logger.info("Championship record: %s", out_csv)
    logger.info("Calls: %s | usage: %s", get_call_stats(), get_usage_stats())
    logger.info("Elapsed: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    asyncio.run(main())
