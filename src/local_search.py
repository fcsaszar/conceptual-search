#!/usr/bin/env python3
"""Run hill-climbing search from the locked entrepreneurial-plan seed."""

import argparse
import asyncio
import csv
from datetime import datetime
import logging
import os
import sys
import time

import config
from local_search_agents import Evaluator, Generator, Narrator, SpecializedGenerator, get_call_stats, get_error_count
from agents_common import get_cache_stats, get_usage_stats
from local_search_prompts import FACTUAL_FIDELITY_RULES, SPECIALIZED_GENERATOR_PROMPT
from records import (
    AUDIT_FIELDNAMES,
    CANDIDATE_FIELDNAMES,
    MATCH_FIELDNAMES,
    append_rows,
    claims_json,
    initialize_csv,
)

LOG_FORMAT = "%(asctime)s %(message)s"

logging.Formatter.default_msec_format = "%s.%03d"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM-guided startup-plan search.")
    parser.add_argument("--seed-id", default=config.SEED_PROJECT_ID, help="Seed project id from the input CSV.")
    parser.add_argument("--model", default=None, help="OpenRouter model id to use for every agent.")
    parser.add_argument("--steps", type=int, default=None, help="Number of local-search steps.")
    parser.add_argument("--num-ideas", type=int, default=None, help="Improvement ideas per step.")
    parser.add_argument("--input-file", default=None, help="Seed-project CSV path.")
    parser.add_argument("--run-prefix", default=None, help="Run label used in logs.")
    parser.add_argument("--output-dir", default=None, help="Directory for all local-search result files.")
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace | None) -> None:
    if args is None:
        return
    config.SEED_PROJECT_ID = args.seed_id
    if args.model:
        config.set_all_agent_models(args.model)
    if args.steps is not None:
        config.NUM_STEPS = args.steps
    if args.num_ideas is not None:
        config.NUM_IDEAS = args.num_ideas
    if args.input_file:
        config.INPUT_FILE = args.input_file
    config.sync_step_aliases()


def _build_match_pairs(num_plans: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(num_plans) for j in range(num_plans) if i != j]


def _build_fieldnames(num_ideas: int) -> list[str]:
    num_plans = num_ideas + 1
    match_pairs = _build_match_pairs(num_plans)
    return (
        ["step"]
        + [f"plan_{i}" for i in range(num_plans)]
        + ["improvements"]
        + [f"rank_{i}" for i in range(num_plans)]
        + ["winner"]
        + [f"winner_{i}_vs_{j}" for i, j in match_pairs]
        + [f"analysis_{i}_vs_{j}" for i, j in match_pairs]
    )


def _plan_snippet(text: str, head: int = 30, tail: int = 30) -> str:
    """Return a short preview: first and last few chars + total length."""
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= head + tail + 10:
        return f'{len(text)} chars, "{text}"'
    return f'{len(text)} chars, "{text[:head]} ... {text[-tail:]}"'


def _col_letter(col_idx: int) -> str:
    """Convert 0-based column index to Excel letter(s): 0→A, 25→Z, 26→AA."""
    result = ""
    idx = col_idx
    while True:
        result = chr(ord("A") + idx % 26) + result
        idx = idx // 26 - 1
        if idx < 0:
            break
    return result


def _csv_cell(fieldnames: list[str], col_name: str, step: int) -> str:
    """Excel-style cell ref, e.g. 'B2'. Header=row 1, step 1=row 2."""
    return f"{_col_letter(fieldnames.index(col_name))}{step + 1}"


def load_seed_plan(csv_path: str, project_id: str = config.SEED_PROJECT_ID) -> str:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] == project_id:
                return row["description"]
    print(f"ERROR: Project '{project_id}' not found in {csv_path}")
    sys.exit(1)


def write_header(output_path: str, fieldnames: list[str]):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(fieldnames)


def append_row(
    output_path: str,
    match_pairs: list[tuple[int, int]],
    step: int,
    plans: list[str],
    improvements: str,
    ranks: list[int],
    winner: str,
    match_details: dict,
):
    match_winners = [match_details[(i, j)][0] for i, j in match_pairs]
    match_analyses = [match_details[(i, j)][1] for i, j in match_pairs]
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([step] + plans + [improvements] + ranks + [winner] + match_winners + match_analyses)


async def write_narrative(history, narrator, narrative_path: str, fieldnames: list[str]):
    """Write the local search narrative from iteration history."""
    lines = []

    # Step 0: seed plan (no CSV ref since seed row doesn't exist)
    lines.append("* Step 0: Initial state\n")
    lines.append(history[0][1] + "\n")

    prev_best_text = history[0][1]
    prev_best_step = 0

    for step, plan_text, label, winner_col in history[1:]:
        changed = (plan_text != prev_best_text)

        lines.append(f"* Step {step}\n")

        if not changed:
            lines.append("(no change -- incumbent wins)\n")
        else:
            cell_ref = _csv_cell(fieldnames, winner_col, step)
            lines.append(f"** Description (cell {cell_ref} in CSV)\n")
            lines.append(plan_text + "\n")
            lines.append(f"** Difference from step {prev_best_step}\n")
            summary = await narrator.summarize_diff(prev_best_text, plan_text)
            lines.append(summary + "\n")

            prev_best_text = plan_text
            prev_best_step = step

    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def main(args: argparse.Namespace | None = None):
    apply_cli_overrides(args)
    config.validate_local_search_config()
    t0 = time.time()
    run_prefix = args.run_prefix if args and args.run_prefix else datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = (
        args.output_dir
        if args and args.output_dir
        else "out"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{run_prefix}-{config.OUTPUT_FILE}")
    log_file = os.path.join(output_dir, f"{run_prefix}-{config.LOG_FILE}")
    narrative_file = os.path.join(output_dir, f"{run_prefix}-{config.NARRATIVE_FILE}")
    candidate_file = os.path.join(output_dir, f"{run_prefix}-local-candidates.csv")
    match_file = os.path.join(output_dir, f"{run_prefix}-local-matches.csv")
    audit_file = os.path.join(output_dir, f"{run_prefix}-local-fidelity-audits.csv")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(file_handler)

    logger.info("Run prefix: %s", run_prefix)
    logger.info(
        "Artifacts: csv=%s, log=%s, narrative=%s, candidates=%s, matches=%s, audits=%s",
        output_file,
        log_file,
        narrative_file,
        candidate_file,
        match_file,
        audit_file,
    )

    seed = load_seed_plan(config.INPUT_FILE, config.SEED_PROJECT_ID)
    stuck_label = "disabled" if config.STUCK_THRESHOLD is None else str(config.STUCK_THRESHOLD)
    logger.info('Seed plan "%s" loaded (%d chars)', config.SEED_PROJECT_ID, len(seed))
    logger.info("Config: steps=%d, stuck=%s, ideas=%d, max_concurrent_calls=%d",
                config.NUM_STEPS, stuck_label, config.NUM_IDEAS,
                config.MAX_CONCURRENT_CALLS)
    fieldnames = _build_fieldnames(config.NUM_IDEAS)
    match_pairs = _build_match_pairs(config.NUM_IDEAS + 1)
    n_gen = config.NUM_STEPS
    n_spec = config.NUM_STEPS * config.NUM_IDEAS
    n_eval = config.NUM_STEPS * (config.NUM_IDEAS + 1) * config.NUM_IDEAS
    n_fidelity = config.NUM_STEPS * config.NUM_IDEAS
    n_narr = config.NUM_STEPS
    n_total = n_gen + n_spec + n_fidelity + n_eval + n_narr
    logger.info(
        "Expected # of calls: Generator (%s): %d, Specialized Generator (%s): %d, "
        "Fidelity Auditor (%s): %d, Evaluator (%s): %d, Narrator (%s): %d. Total: %d",
        config.GENERATOR_MODEL, n_gen,
        config.SPECIALIZED_GENERATOR_MODEL, n_spec,
        config.FIDELITY_AUDITOR_MODEL, n_fidelity,
        config.EVALUATOR_MODEL, n_eval,
        config.NARRATOR_MODEL, n_narr,
        n_total,
    )

    generator = Generator()
    specialized = SpecializedGenerator()
    evaluator = Evaluator()
    narrator = Narrator()

    incumbent = seed
    consecutive_wins = 0
    incumbent_id = config.SEED_PROJECT_ID
    history = [(0, seed, "seed", None)]
    write_header(output_file, fieldnames)
    initialize_csv(candidate_file, CANDIDATE_FIELDNAMES)
    initialize_csv(match_file, MATCH_FIELDNAMES)
    initialize_csv(audit_file, AUDIT_FIELDNAMES)
    append_rows(
        candidate_file,
        CANDIDATE_FIELDNAMES,
        [
            {
                "mode": "local",
                "step": 0,
                "slot": 0,
                "candidate_id": incumbent_id,
                "parent_a": "",
                "parent_b": "",
                "origin": "seed",
                "transformation": "seed",
                "requested_change": "",
                "plan_text": seed,
                "generated_plan": seed,
                "rank": "",
                "win_rate": "",
                "selected": True,
                "fidelity_passed": True,
                "fidelity_action": "source",
                "unsupported_claims": "[]",
                "fidelity_analysis": "Original source description.",
                "evidence_notes": "",
                "prompt": f"seed from {config.INPUT_FILE}",
                "root_lineage": config.SEED_PROJECT_ID,
            }
        ],
    )

    for iteration in range(1, config.NUM_STEPS + 1):
        header = f"── Step {iteration}/{config.NUM_STEPS} "
        logger.info(header + "─" * (52 - len(header)))

        # Step 1: Generate improvement ideas
        logger.info("Generating ideas...")
        gen_extra = f"; cell {_csv_cell(fieldnames, 'improvements', iteration)} in CSV"
        ideas = await generator.generate(incumbent, log_extra=gen_extra)
        for i, idea in enumerate(ideas, 1):
            logger.info("    idea %d: %.120s", i, idea)

        # Step 2: Generate plan variations
        logger.info("Rewriting plan variations...")
        spec_extras = [
            f"; cell {_csv_cell(fieldnames, f'plan_{i+1}', iteration)} in CSV"
            for i in range(len(ideas))
        ]
        variation_records = await specialized.generate_all_variations(
            incumbent, ideas, log_extras=spec_extras
        )
        unavailable_audits = [
            idx
            for idx, record in enumerate(variation_records, 1)
            if record["audit"].get("unavailable", False)
        ]
        if unavailable_audits:
            slots = ", ".join(str(idx) for idx in unavailable_audits)
            raise RuntimeError(
                "Fidelity auditing was unavailable for local candidate slot(s) "
                f"{slots} at step {iteration}. The run is stopping instead of "
                "treating service failures as substantive rejections; rerun the "
                "same run ID to resume from cached successful calls."
            )
        variations = [record["plan"] for record in variation_records]
        candidate_ids = [incumbent_id] + [f"L{iteration}.{idx}" for idx in range(1, len(ideas) + 1)]

        # Step 3: Evaluate all plans (incumbent + variations)
        all_plans = [incumbent] + variations
        n = len(all_plans)
        logger.info("Evaluating %d plans (%d matches)...", n, n * (n - 1))
        eval_extras = {
            (i, j): f"; cell {_csv_cell(fieldnames, f'analysis_{i}_vs_{j}', iteration)} in CSV"
            for i, j in match_pairs
        }
        ranks, win_rates, match_details = await evaluator.evaluate(all_plans, match_log_extras=eval_extras)
        logger.info("  win_rates=%s", [f"{w:.0%}" for w in win_rates])
        logger.info("  ranks=%s (cells %s:%s in CSV)", ranks,
                    _csv_cell(fieldnames, "rank_0", iteration),
                    _csv_cell(fieldnames, f"rank_{len(all_plans) - 1}", iteration))

        # Step 4: Determine best plan and check stopping
        best_idx = ranks.index(1)
        winner_col = f"plan_{best_idx}"

        append_rows(
            match_file,
            MATCH_FIELDNAMES,
            [
                {
                    "mode": "local",
                    "step": iteration,
                    "plan_a_slot": i,
                    "plan_a_id": candidate_ids[i],
                    "plan_b_slot": j,
                    "plan_b_id": candidate_ids[j],
                    "winner_id": (
                        candidate_ids[int(match_details[(i, j)][0].split("_")[1])]
                        if match_details[(i, j)][0].startswith("plan_")
                        else "tie"
                    ),
                    "winner_label": match_details[(i, j)][0],
                    "analysis": match_details[(i, j)][1],
                }
                for i, j in match_pairs
            ],
        )

        candidate_rows = [
            {
                "mode": "local",
                "step": iteration,
                "slot": 0,
                "candidate_id": incumbent_id,
                "parent_a": incumbent_id,
                "parent_b": "",
                "origin": "incumbent",
                "transformation": "retained",
                "requested_change": "",
                "plan_text": incumbent,
                "generated_plan": incumbent,
                "rank": ranks[0],
                "win_rate": f"{win_rates[0]:.6f}",
                "selected": best_idx == 0,
                "fidelity_passed": True,
                "fidelity_action": "retained",
                "unsupported_claims": "[]",
                "fidelity_analysis": "Previously accepted source-preserving state.",
                "evidence_notes": "",
                "prompt": "",
                "root_lineage": config.SEED_PROJECT_ID,
            }
        ]
        audit_rows = []
        for idx, (idea, record) in enumerate(zip(ideas, variation_records), 1):
            audit = record["audit"]
            prompt = SPECIALIZED_GENERATOR_PROMPT.format(
                plan=incumbent,
                idea=idea,
                fidelity_rules=FACTUAL_FIDELITY_RULES,
            )
            candidate_rows.append(
                {
                    "mode": "local",
                    "step": iteration,
                    "slot": idx,
                    "candidate_id": candidate_ids[idx],
                    "parent_a": incumbent_id,
                    "parent_b": "",
                    "origin": "local_variant",
                    "transformation": "specialized_rewrite",
                    "requested_change": idea,
                    "plan_text": record["plan"],
                    "generated_plan": record["generated_plan"],
                    "rank": ranks[idx],
                    "win_rate": f"{win_rates[idx]:.6f}",
                    "selected": best_idx == idx,
                    "fidelity_passed": audit["passed"],
                    "fidelity_action": record["action"],
                    "unsupported_claims": claims_json(audit["unsupported_claims"]),
                    "fidelity_analysis": audit["analysis"],
                    "evidence_notes": record["evidence_notes"],
                    "prompt": prompt,
                    "root_lineage": config.SEED_PROJECT_ID,
                }
            )
            audit_rows.append(
                {
                    "mode": "local",
                    "step": iteration,
                    "candidate_id": candidate_ids[idx],
                    "operator": "specialized_rewrite",
                    "parent_a": incumbent_id,
                    "parent_b": "",
                    "model_passed": audit.get("model_passed", audit["passed"]),
                    "decision_basis": audit.get("decision_basis", "model_audit"),
                    "model_unsupported_claims": claims_json(
                        audit.get("model_unsupported_claims", audit["unsupported_claims"])
                    ),
                    "passed": audit["passed"],
                    "action": record["action"],
                    "unsupported_claims": claims_json(audit["unsupported_claims"]),
                    "analysis": audit["analysis"],
                    "generated_plan": record["generated_plan"],
                    "evaluated_plan": record["plan"],
                }
            )
        append_rows(candidate_file, CANDIDATE_FIELDNAMES, candidate_rows)
        append_rows(audit_file, AUDIT_FIELDNAMES, audit_rows)

        # Step 5: Write row
        improvements_text = "\n".join(f"• {idea}" for idea in ideas)
        append_row(output_file, match_pairs, iteration, all_plans, improvements_text, ranks, winner_col, match_details)
        logger.info("  winner=%s (cell %s in CSV)", winner_col, _csv_cell(fieldnames, winner_col, iteration))

        if best_idx == 0:
            consecutive_wins += 1
            history.append((iteration, incumbent, "incumbent", "plan_0"))
            if (
                config.STUCK_THRESHOLD is not None
                and consecutive_wins >= config.STUCK_THRESHOLD
            ):
                logger.info(">> Incumbent wins (streak %d/%d) -- stopping (cell %s in CSV; %s)",
                            consecutive_wins, config.STUCK_THRESHOLD,
                            _csv_cell(fieldnames, "plan_0", iteration), _plan_snippet(incumbent))
                break
            elif config.STUCK_THRESHOLD is not None:
                logger.info(">> Incumbent wins (streak %d/%d) (cell %s in CSV; %s)",
                            consecutive_wins, config.STUCK_THRESHOLD,
                            _csv_cell(fieldnames, "plan_0", iteration), _plan_snippet(incumbent))
            else:
                logger.info(">> Incumbent wins (streak %d; threshold disabled) (cell %s in CSV; %s)",
                            consecutive_wins,
                            _csv_cell(fieldnames, "plan_0", iteration), _plan_snippet(incumbent))
        else:
            consecutive_wins = 0
            incumbent = all_plans[best_idx]
            incumbent_id = candidate_ids[best_idx]
            history.append((iteration, incumbent, f"alt {best_idx}", f"plan_{best_idx}"))
            logger.info(">> New incumbent: alt %d (cell %s in CSV; %s)",
                        best_idx, _csv_cell(fieldnames, f"plan_{best_idx}", iteration),
                        _plan_snippet(incumbent))

    if (
        config.STUCK_THRESHOLD is not None
        and consecutive_wins >= config.STUCK_THRESHOLD
    ):
        logger.info("Done. Best plan unchanged for %d steps.", consecutive_wins)
    else:
        logger.info("Done. Reached max steps (%d).", config.NUM_STEPS)
    logger.info("Output: %s", output_file)

    # Generate narrative
    await write_narrative(history, narrator, narrative_file, fieldnames)
    logger.info("Narrative: %s", narrative_file)

    # Call summary
    role_models = {
        "Generator": config.GENERATOR_MODEL,
        "Specialized Generator": config.SPECIALIZED_GENERATOR_MODEL,
        "Fidelity Auditor": config.FIDELITY_AUDITOR_MODEL,
        "Evaluator": config.EVALUATOR_MODEL,
        "Narrator": config.NARRATOR_MODEL,
    }
    stats = get_call_stats()
    parts = []
    actual_total = 0
    for role in ("Generator", "Specialized Generator", "Fidelity Auditor", "Evaluator", "Narrator"):
        planned, total = stats.get(role, [0, 0])
        retries = total - planned
        actual_total += total
        parts.append(f"{role} ({role_models[role]}): {planned}+{retries}")
    logger.info("# of calls: %s. Total: %d", ", ".join(parts), actual_total)
    logger.info("Cache hits: %s", get_cache_stats())
    logger.info("Live-call usage: %s", get_usage_stats())
    elapsed = (time.time() - t0) / 60
    logger.info("Time elapsed: %.1f minutes", elapsed)
    errors = get_error_count()
    logger.info('ERROR SUMMARY: %d errors occurred (search the log for "ERROR")', errors)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
