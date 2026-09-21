#!/usr/bin/env python3
"""Calibrate plan quality against one shared seed benchmark set.

Every unique concept text in the local and global search records is compared
with each venture in the given firm set, in both prompt orders.  With
`--scope all-points`, `--firm-set in/firm-set-30.txt`, and
`AI_ENTREP_EXPECTED_FIRM_COUNT=30`, this produces the reference-set win rates
that the figure builders join to the records by normalized text.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime
import logging
import os
import re
import time

import config
from agents_common import (
    Evaluator,
    get_cache_stats,
    get_call_stats,
    get_error_count,
    get_usage_stats,
)


LOG_FORMAT = "%(asctime)s %(message)s"
MATCH_FIELDNAMES = [
    "concept_key",
    "concept_index",
    "benchmark_id",
    "direction",
    "winner",
    "analysis",
]
SUMMARY_FIELDNAMES = [
    "text_key",
    "concept_index",
    "plan_text",
    "wins",
    "matches",
    "benchmark_quality",
]

logging.Formatter.default_msec_format = "%s.%03d"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score selected plans against one seed benchmark set.")
    parser.add_argument("--local-csv", required=True, help="Local-search CSV to calibrate.")
    parser.add_argument("--global-csv", required=True, help="Global-search CSV to calibrate.")
    parser.add_argument("--firm-set", required=True, help="Text file with benchmark seed ids, one per line.")
    parser.add_argument("--model", default=config.DEFAULT_OPENROUTER_MODEL, help="OpenRouter model for evaluator calls.")
    parser.add_argument("--run-prefix", default=None, help="Run label used in logs.")
    parser.add_argument("--input-file", default=config.INPUT_FILE, help="Seed-project CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory for common-quality result files.")
    parser.add_argument(
        "--scope",
        choices=("frontier", "all-points"),
        default="frontier",
        help="Which concepts to calibrate. `frontier` is local accepted states plus global winners.",
    )
    return parser.parse_args()


def normalize_text_key(text: str | None) -> str:
    return " ".join((text or "").split())


def read_firm_set(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        ids = [line.split("#", 1)[0].strip() for line in f]
    return [firm_id for firm_id in ids if firm_id]


def load_seed_descriptions(csv_path: str, firm_ids: list[str]) -> list[tuple[str, str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        by_id = {row["id"]: row["description"] for row in csv.DictReader(f)}

    missing = [firm_id for firm_id in firm_ids if firm_id not in by_id]
    if missing:
        raise ValueError(f"Firm ids not found in {csv_path}: {', '.join(missing)}")

    return [(firm_id, by_id[firm_id]) for firm_id in firm_ids]


def numbered_columns(fieldnames: list[str], pattern: str) -> list[str]:
    regex = re.compile(pattern)
    cols = []
    for name in fieldnames:
        match = regex.fullmatch(name)
        if match:
            cols.append((int(match.group(1)), name))
    return [name for _, name in sorted(cols)]


def lineage_mentions_any(plan_id: str, firm_ids: list[str]) -> bool:
    if not firm_ids:
        return True
    tokens = re.split(r"[().X]", plan_id or "")
    return any(firm_id in tokens for firm_id in firm_ids)


def collect_unique_concepts(local_csv: str, global_csv: str, firm_ids: list[str], scope: str) -> list[dict[str, str | int]]:
    concepts_by_key: dict[str, dict[str, str | int]] = {}

    def add_concept(text: str) -> None:
        text_key = normalize_text_key(text)
        if not text_key or text_key in concepts_by_key:
            return
        concepts_by_key[text_key] = {
            "text_key": text_key,
            "concept_index": len(concepts_by_key) + 1,
            "plan_text": text,
        }

    with open(local_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        plan_cols = numbered_columns(reader.fieldnames or [], r"plan_(\d+)")
        seed_added = False
        for row in reader:
            if scope == "all-points":
                for col in plan_cols:
                    add_concept(row.get(col, ""))
            else:
                if not seed_added and plan_cols:
                    add_concept(row.get("plan_0", ""))
                    seed_added = True
                winner_key = row.get("winner", "")
                if re.fullmatch(r"plan_\d+", winner_key or ""):
                    add_concept(row.get(winner_key, ""))

    with open(global_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        plan_cols = numbered_columns(fieldnames, r"plan_(\d+)")
        plan_id_cols = numbered_columns(fieldnames, r"plan_id_(\d+)")
        for row in reader:
            if scope == "all-points":
                for plan_col, plan_id_col in zip(plan_cols, plan_id_cols):
                    if lineage_mentions_any(row.get(plan_id_col, ""), firm_ids):
                        add_concept(row.get(plan_col, ""))
            else:
                candidates = []
                for plan_col, plan_id_col in zip(plan_cols, plan_id_cols):
                    plan_id = row.get(plan_id_col, "")
                    if not lineage_mentions_any(plan_id, firm_ids):
                        continue
                    suffix = plan_col.rsplit("_", 1)[1]
                    rank = int(row.get(f"rank_{suffix}") or 10**9)
                    candidates.append((rank, plan_col))
                if candidates:
                    _rank, winner_col = min(candidates, key=lambda item: item[0])
                    add_concept(row.get(winner_col, ""))

    return list(concepts_by_key.values())


def read_existing_matches(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def completed_match_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {
        (row["concept_key"], row["benchmark_id"], row["direction"])
        for row in rows
        if row.get("concept_key") and row.get("benchmark_id") and row.get("direction")
    }


def append_match_rows(path: str, rows: list[dict[str, str | int]]) -> None:
    if not rows:
        return
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_FIELDNAMES, quoting=csv.QUOTE_ALL)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def score_row(row: dict[str, str]) -> float:
    winner = (row.get("winner") or "").lower()
    if winner == "concept":
        return 1.0
    if winner == "benchmark":
        return 0.0
    return 0.5


def write_summary(path: str, concepts: list[dict[str, str | int]], match_rows: list[dict[str, str]], n_benchmarks: int) -> None:
    rows_by_key: dict[str, list[dict[str, str]]] = {}
    for row in match_rows:
        rows_by_key.setdefault(row["concept_key"], []).append(row)

    expected_matches = 2 * n_benchmarks
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for concept in concepts:
            concept_key = str(concept["text_key"])
            rows = rows_by_key.get(concept_key, [])
            wins = sum(score_row(row) for row in rows)
            matches = len(rows)
            quality = wins / expected_matches if expected_matches else 0.0
            writer.writerow(
                {
                    "text_key": concept_key,
                    "concept_index": concept["concept_index"],
                    "plan_text": concept["plan_text"],
                    "wins": f"{wins:.1f}",
                    "matches": matches,
                    "benchmark_quality": f"{quality:.6f}",
                }
            )


async def score_matches(
    match_path: str,
    concepts: list[dict[str, str | int]],
    benchmarks: list[tuple[str, str]],
) -> list[dict[str, str]]:
    evaluator = Evaluator()
    match_rows = read_existing_matches(match_path)
    done = completed_match_keys(match_rows)

    exact_rows: list[dict[str, str | int]] = []
    specs = []
    for concept in concepts:
        concept_key = str(concept["text_key"])
        concept_text = str(concept["plan_text"])
        concept_index = int(concept["concept_index"])
        for benchmark_id, benchmark_text in benchmarks:
            benchmark_key = normalize_text_key(benchmark_text)
            for direction in ("concept_first", "seed_first"):
                key = (concept_key, benchmark_id, direction)
                if key in done:
                    continue
                if concept_key == benchmark_key:
                    exact_rows.append(
                        {
                            "concept_key": concept_key,
                            "concept_index": concept_index,
                            "benchmark_id": benchmark_id,
                            "direction": direction,
                            "winner": "tie",
                            "analysis": "Exact same normalized text; skipped LLM call.",
                        }
                    )
                    done.add(key)
                    continue
                specs.append((concept_key, concept_index, concept_text, benchmark_id, benchmark_text, direction))

    append_match_rows(match_path, exact_rows)
    if exact_rows:
        match_rows.extend({key: str(value) for key, value in row.items()} for row in exact_rows)
        logger.info("Recorded %d exact-text tie matches without LLM calls.", len(exact_rows))

    total = len(specs)
    logger.info("Common-quality matches to run: %d", total)
    for batch_start in range(0, total, config.MATCH_BATCH_SIZE):
        batch_specs = specs[batch_start:batch_start + config.MATCH_BATCH_SIZE]
        tasks = []
        for concept_key, _concept_index, concept_text, benchmark_id, benchmark_text, direction in batch_specs:
            if direction == "concept_first":
                plan_a, plan_b = concept_text, benchmark_text
                label_a, label_b = f"concept {concept_key[:12]}", f"seed {benchmark_id}"
            else:
                plan_a, plan_b = benchmark_text, concept_text
                label_a, label_b = f"seed {benchmark_id}", f"concept {concept_key[:12]}"
            tasks.append(evaluator._match(plan_a, plan_b, label_a=label_a, label_b=label_b))

        results = await asyncio.gather(*tasks)
        batch_rows: list[dict[str, str | int]] = []
        for spec, (winner_letter, analysis) in zip(batch_specs, results):
            concept_key, concept_index, _concept_text, benchmark_id, _benchmark_text, direction = spec
            if winner_letter == "TIE":
                winner = "tie"
            elif direction == "concept_first":
                winner = "concept" if winner_letter == "A" else "benchmark"
            else:
                winner = "benchmark" if winner_letter == "A" else "concept"
            batch_rows.append(
                {
                    "concept_key": concept_key,
                    "concept_index": concept_index,
                    "benchmark_id": benchmark_id,
                    "direction": direction,
                    "winner": winner,
                    "analysis": analysis,
                }
            )

        append_match_rows(match_path, batch_rows)
        match_rows.extend({key: str(value) for key, value in row.items()} for row in batch_rows)
        logger.info(
            "Common-quality batch %d-%d / %d done.",
            batch_start + 1,
            batch_start + len(batch_specs),
            total,
        )

    return match_rows


async def main(args: argparse.Namespace) -> None:
    config.set_all_agent_models(args.model)
    if not config.MOCK_LLM:
        config.require_openrouter_api_key()

    t0 = time.time()
    run_prefix = args.run_prefix or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    match_path = os.path.join(args.output_dir, f"{run_prefix}-common-quality-matches.csv")
    summary_path = os.path.join(args.output_dir, f"{run_prefix}-common-quality.csv")
    log_path = os.path.join(args.output_dir, f"{run_prefix}-common-quality.log")

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(file_handler)

    firm_ids = read_firm_set(args.firm_set)
    try:
        config.validate_firm_set(firm_ids)
    except ValueError as exc:
        raise SystemExit(str(exc))
    benchmarks = load_seed_descriptions(args.input_file, firm_ids)
    concepts = collect_unique_concepts(args.local_csv, args.global_csv, firm_ids, args.scope)

    logger.info("Run prefix: %s", run_prefix)
    logger.info("Artifacts: matches=%s, summary=%s, log=%s", match_path, summary_path, log_path)
    logger.info("Model: %s", args.model)
    logger.info("Scope: %s", args.scope)
    logger.info("Max concurrent calls: %d", config.MAX_CONCURRENT_CALLS)
    logger.info("Benchmark seeds (%d): %s", len(benchmarks), [firm_id for firm_id, _ in benchmarks])
    logger.info("Unique concepts to calibrate: %d", len(concepts))

    match_rows = await score_matches(match_path, concepts, benchmarks)
    write_summary(summary_path, concepts, match_rows, len(benchmarks))

    planned, total = get_call_stats().get("Evaluator", [0, 0])
    logger.info("Evaluator calls: %d planned, %d total including retries.", planned, total)
    logger.info("Cache hits: %s", get_cache_stats())
    logger.info("Live-call usage: %s", get_usage_stats())
    logger.info("Errors: %d", get_error_count())
    logger.info("Summary: %s", summary_path)
    logger.info("Time elapsed: %.1f minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
