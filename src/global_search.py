#!/usr/bin/env python3
"""Run the locked population-based entrepreneurial-plan search."""

import argparse
import asyncio
import csv
from datetime import datetime
import logging
import os
import random
import sys
import time

import config
from agents_common import (
    Evaluator,
    Narrator,
    get_cache_stats,
    get_call_stats,
    get_error_count,
    get_usage_stats,
)
from global_search_agents import Crossover, Mutator, rank_proportional_select
from local_search_prompts import EVALUATION_CRITERIA, FACTUAL_FIDELITY_RULES
from global_search_prompts import (
    MUTATION_COMPONENT_GUIDANCE,
    MUTATOR_PROMPT, CROSSOVER_PROMPT,
    NARRATIVE_MUTATION_PROMPT, NARRATIVE_CROSSOVER_PROMPT,
)
from records import (
    AUDIT_FIELDNAMES,
    CANDIDATE_FIELDNAMES,
    MATCH_FIELDNAMES,
    append_rows,
    claims_json,
    initialize_csv,
)
from semantic_novelty import (
    COMPONENTS,
    SemanticNovelty,
    SelectionMetrics,
    combine_quality_and_novelty,
)


def _norm_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def pick_recombination_partner(
    a_id: str,
    a_plan: str,
    a_roots: frozenset[str],
    eligible: list[dict],
    used_pairs: set[tuple[str, str]],
) -> dict | None:
    """Choose crossover parent B: the best-ranked eligible candidate that adds
    at least one founding-seed lineage parent A does not already carry.

    ``eligible`` is ordered by preference (current pool by selection rank, then
    archive seeds by their step-0 quality order).  Pairs already used this step
    are skipped on the first pass so the five crossovers explore distinct
    pairings; if every lineage-adding candidate is used, a repeat is allowed.
    A same-lineage candidate is only ever a last resort.
    """
    a_norm = _norm_text(a_plan)

    def usable(candidate: dict) -> bool:
        return (
            candidate["candidate_id"] != a_id
            and _norm_text(str(candidate["plan"])) != a_norm
        )

    for candidate in eligible:
        if not usable(candidate):
            continue
        if (a_id, candidate["candidate_id"]) in used_pairs:
            continue
        if not frozenset(candidate["roots"]).issubset(a_roots):
            return candidate
    for candidate in eligible:
        if not usable(candidate):
            continue
        if not frozenset(candidate["roots"]).issubset(a_roots):
            return candidate
    for candidate in eligible:
        if usable(candidate):
            return candidate
    return None


def select_elites_with_lineage_cap(
    sorted_indices: list[int],
    root_sets: list[frozenset[str]],
    n_elites: int,
    cap: int,
) -> list[int]:
    """Pick elites in selection-rank order, holding each founding lineage to at
    most ``cap`` elite slots; top up by rank if the cap cannot fill all slots."""
    chosen: list[int] = []
    lineage_counts: dict[str, int] = {}
    for index in sorted_indices:
        if len(chosen) == n_elites:
            break
        roots = root_sets[index]
        if all(lineage_counts.get(root, 0) < cap for root in roots):
            chosen.append(index)
            for root in roots:
                lineage_counts[root] = lineage_counts.get(root, 0) + 1
    if len(chosen) < n_elites:
        for index in sorted_indices:
            if index not in chosen:
                chosen.append(index)
                if len(chosen) == n_elites:
                    break
    return chosen

LOG_FORMAT = "%(asctime)s %(message)s"

logging.Formatter.default_msec_format = "%s.%03d"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global evolutionary LLM-guided startup-plan search.")
    parser.add_argument(
        "--firm-set",
        default=config.LOCKED_FIRM_SET_PATH,
        help="Locked text file with the 15 ex ante seed project ids.",
    )
    parser.add_argument("--preset", choices=("15",), default="15", help="Locked population preset.")
    parser.add_argument("--model", default=None, help="OpenRouter model id to use for every agent.")
    parser.add_argument("--steps", type=int, default=None, help="Number of global-search steps.")
    parser.add_argument("--input-file", default=None, help="Seed-project CSV path.")
    parser.add_argument("--run-prefix", default=None, help="Run label used in logs.")
    parser.add_argument("--output-dir", default=None, help="Directory for all global-search result files.")
    return parser.parse_args()


def read_firm_set(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        firm_ids = [
            line.split("#", 1)[0].strip()
            for line in f
        ]
    return [firm_id for firm_id in firm_ids if firm_id]


def apply_cli_overrides(args: argparse.Namespace | None) -> list[str] | None:
    if args is None:
        return None
    if args.model:
        config.set_all_agent_models(args.model)
    if args.steps is not None:
        config.NUM_STEPS = args.steps
    if args.input_file:
        config.INPUT_FILE = args.input_file
    if args.preset:
        config.apply_global_preset(args.preset)

    firm_ids = read_firm_set(args.firm_set)
    config.validate_firm_set(firm_ids)
    config.sync_step_aliases()
    return firm_ids


# --- CSV helpers ---------------------------------------------------------------

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
    """Excel-style cell ref, e.g. 'B2'. Header=row 1, step 0=row 2."""
    return f"{_col_letter(fieldnames.index(col_name))}{step + 2}"


def _build_fieldnames() -> list[str]:
    n = config.POP_SIZE
    return (
        ["step"]
        + ["best_id", "best_plan"]
        + [f"plan_id_{i}" for i in range(n)]
        + [f"plan_{i}" for i in range(n)]
        + [f"rank_{i}" for i in range(n)]
        + [f"win_rate_{i}" for i in range(n)]
        + [f"quality_percentile_{i}" for i in range(n)]
        + [f"novelty_{i}" for i in range(n)]
        + [f"novelty_rank_{i}" for i in range(n)]
        + [f"novelty_percentile_{i}" for i in range(n)]
        + [f"combined_score_{i}" for i in range(n)]
        + [f"selection_rank_{i}" for i in range(n)]
        + [f"origin_{i}" for i in range(n)]
        + [f"prompt_{i}" for i in range(n)]
    )


def write_header(fieldnames: list[str], output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(fieldnames)


def append_row(fieldnames: list[str], output_path: str, step: int, best_id: str, best_plan: str,
               plan_ids: list[str], plans: list[str], ranks: list[int],
               win_rates: list[float], novelty_scores: list[float],
               selection: SelectionMetrics, origins: list[str], provenance: list[str]):
    row = (
        [step]
        + [best_id, best_plan]
        + plan_ids
        + plans
        + ranks
        + [f"{w:.3f}" for w in win_rates]
        + [f"{value:.6f}" for value in selection.quality_percentiles]
        + [f"{value:.6f}" for value in novelty_scores]
        + [f"{value:.3f}" for value in selection.novelty_ranks]
        + [f"{value:.6f}" for value in selection.novelty_percentiles]
        + [f"{value:.6f}" for value in selection.combined_scores]
        + selection.selection_ranks
        + origins
        + provenance
    )
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(row)


def append_match_records(
    path: str,
    mode: str,
    step: int | str,
    plan_ids: list[str],
    match_details: dict,
) -> None:
    append_rows(
        path,
        MATCH_FIELDNAMES,
        [
            {
                "mode": mode,
                "step": step,
                "plan_a_slot": i,
                "plan_a_id": plan_ids[i],
                "plan_b_slot": j,
                "plan_b_id": plan_ids[j],
                "winner_id": (
                    plan_ids[int(winner_label.split("_")[1])]
                    if winner_label.startswith("plan_")
                    else "tie"
                ),
                "winner_label": winner_label,
                "analysis": analysis,
            }
            for (i, j), (winner_label, analysis) in match_details.items()
        ],
    )


def append_candidate_records(
    path: str,
    step: int,
    plan_ids: list[str],
    plans: list[str],
    origins: list[str],
    ranks: list[int],
    win_rates: list[float],
    novelty_scores: list[float],
    selection: SelectionMetrics,
    metadata: list[dict],
    root_sets: list[frozenset[str]] | None = None,
) -> None:
    append_rows(
        path,
        CANDIDATE_FIELDNAMES,
        [
            {
                "mode": "global",
                "step": step,
                "slot": slot,
                "candidate_id": plan_ids[slot],
                "parent_a": metadata[slot]["parent_a"],
                "parent_b": metadata[slot]["parent_b"],
                "origin": origins[slot],
                "transformation": metadata[slot]["transformation"],
                "requested_change": metadata[slot]["requested_change"],
                "plan_text": plans[slot],
                "generated_plan": metadata[slot]["generated_plan"],
                "rank": ranks[slot],
                "win_rate": f"{win_rates[slot]:.6f}",
                "quality_percentile": f"{selection.quality_percentiles[slot]:.6f}",
                "novelty": f"{novelty_scores[slot]:.6f}",
                "novelty_rank": f"{selection.novelty_ranks[slot]:.3f}",
                "novelty_percentile": f"{selection.novelty_percentiles[slot]:.6f}",
                "combined_score": f"{selection.combined_scores[slot]:.6f}",
                "selection_rank": selection.selection_ranks[slot],
                "selected": selection.selection_ranks[slot] <= config.POP_ELITE,
                "fidelity_passed": metadata[slot]["fidelity_passed"],
                "fidelity_action": metadata[slot]["fidelity_action"],
                "unsupported_claims": claims_json(metadata[slot]["unsupported_claims"]),
                "fidelity_analysis": metadata[slot]["fidelity_analysis"],
                "evidence_notes": metadata[slot]["evidence_notes"],
                "prompt": metadata[slot]["prompt"],
                "root_lineage": (
                    ",".join(sorted(root_sets[slot])) if root_sets is not None else ""
                ),
            }
            for slot in range(len(plans))
        ],
    )


# --- Seed loading --------------------------------------------------------------

def load_seed_plans(csv_path: str, n: int, firm_ids: list[str] | None = None) -> list[tuple[str, str]]:
    """Load first n plans from CSV. Returns list of (id, description)."""
    seeds = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if firm_ids is not None:
            by_id = {row["id"]: row["description"] for row in reader}
            missing = [firm_id for firm_id in firm_ids if firm_id not in by_id]
            if missing:
                print(f"ERROR: Firm ids not found in {csv_path}: {', '.join(missing)}")
                sys.exit(1)
            return [(firm_id, by_id[firm_id]) for firm_id in firm_ids]
        for row in reader:
            seeds.append((row["id"], row["description"]))
            if len(seeds) >= n:
                break
    if len(seeds) < n:
        print(f"ERROR: Only {len(seeds)} projects in {csv_path}, need {n}")
        sys.exit(1)
    return seeds


# --- Narrative -----------------------------------------------------------------

async def write_narrative(history: list, narrator: Narrator, fieldnames: list[str],
                          narrative_path: str, seed_tournament=None,
                          seed_lookup: dict[str, str] | None = None,
                          commit_summary: dict | None = None):
    """Write the global search narrative file.

    history entries: (step, best_id, best_plan, plan_ids, plans, ranks, win_rates, origins)
    """
    lines = []

    # Step 0 is the initial seed-population evaluation.
    step0, best_id0, best_plan0, ids0, plans0, ranks0, wr0, origins0 = history[0]
    lines.append(f"* Step {step0}: Initial ranking\n")

    sorted_indices = sorted(range(len(ids0)), key=lambda k: ranks0[k])
    for idx in sorted_indices:
        cell = _csv_cell(fieldnames, f"plan_{idx}", step0)
        lines.append(f"- {ids0[idx]} (win rate {wr0[idx]:.0%}, cell {cell} in CSV)")
    lines.append("")

    best_cell0 = _csv_cell(fieldnames, f"plan_{ranks0.index(1)}", step0)
    lines.append(f"** Best firm: {best_id0} (cell {best_cell0} in CSV)\n")
    lines.append(best_plan0 + "\n")

    prev_best_text = best_plan0
    prev_step = step0

    for i, entry in enumerate(history[1:], start=1):
        step, best_id, best_plan, ids, pls, ranks, wr, origins = entry
        final = (i == len(history) - 1)
        subtitle = ": Final evaluation" if final else ""
        lines.append(f"* Step {step}{subtitle}\n")

        lines.append("** Ranking\n")
        sorted_indices = sorted(range(len(ids)), key=lambda k: ranks[k])
        for idx in sorted_indices:
            cell = _csv_cell(fieldnames, f"plan_{idx}", step)
            lines.append(f"- {ids[idx]} (win rate {wr[idx]:.0%}, cell {cell} in CSV)")
        lines.append("")

        changed = (best_plan != prev_best_text)
        if changed:
            best_idx = ranks.index(1)
            best_cell = _csv_cell(fieldnames, f"plan_{best_idx}", step)
            lines.append(f"** Best firm: {best_id} (cell {best_cell} in CSV)\n")
            lines.append(best_plan + "\n")
            lines.append(f"** Difference from step {prev_step}\n")
            summary = await narrator.summarize_diff(prev_best_text, best_plan)
            lines.append(summary + "\n")

            # Lineage
            best_origin = origins[best_idx]
            if best_origin != "seed":
                prev_ids = history[i - 1][3]
                prev_pls = history[i - 1][4]
                prev_step_num = history[i - 1][0]
                lineage = None

                if best_origin == "elite":
                    lineage = (f"This plan was carried unchanged as an elite "
                               f"from step {prev_step_num}.")
                elif best_origin.startswith("mutant("):
                    inner = best_origin[len("mutant("):-1]
                    parent_id, dimension = inner.rsplit(",", 1)
                    if parent_id in prev_ids:
                        parent_plan = prev_pls[prev_ids.index(parent_id)]
                        prompt = NARRATIVE_MUTATION_PROMPT.format(
                            parent_plan=parent_plan, child_plan=best_plan,
                            dimension=dimension)
                        lineage = await narrator.narrate(
                            prompt, context=f"lineage mutation ({dimension})")
                elif best_origin.startswith("crossover("):
                    inner = best_origin[len("crossover("):-1]
                    a_id, b_id = inner.rsplit(",", 1)
                    seed_lookup = seed_lookup or {}
                    a_plan = (
                        prev_pls[prev_ids.index(a_id)]
                        if a_id in prev_ids
                        else seed_lookup.get(a_id)
                    )
                    b_plan = (
                        prev_pls[prev_ids.index(b_id)]
                        if b_id in prev_ids
                        else seed_lookup.get(b_id)
                    )
                    if a_plan is not None and b_plan is not None:
                        prompt = NARRATIVE_CROSSOVER_PROMPT.format(
                            a_id=a_id, b_id=b_id, a_plan=a_plan,
                            b_plan=b_plan, child_plan=best_plan)
                        lineage = await narrator.narrate(
                            prompt, context=f"lineage crossover ({a_id} x {b_id})")

                if lineage:
                    lines.append("** Lineage\n")
                    lines.append(lineage + "\n")

            prev_best_text = best_plan
            prev_step = step
        else:
            lines.append(f"** Best firm (same plan as step {prev_step})\n")
            lines.append(f"** Difference from step {prev_step} (no change)\n")

    # Commit round
    if commit_summary is not None:
        lines.append("")
        lines.append("* Commit Round\n")
        lines.append(
            f"After the search steps, {commit_summary['n_candidates']} unique plans that "
            f"held an elite slot or won a step were each verified against the seed "
            f"archive ({commit_summary['n_matches']} matches per plan). The champion is "
            f"the verification-best plan: {commit_summary['champion_id']} "
            f"({commit_summary['champion_wins']:.1f}/{commit_summary['n_matches']} wins)."
        )
        if commit_summary["polish_adopted"]:
            lines.append(
                "A polish mutation of the verification leader outperformed it and was "
                "adopted as the champion.")
        lines.append("")

    # Seed tournament
    if seed_tournament is not None:
        st_ids, st_ranks, st_win_rates = seed_tournament
        n_st = len(st_ids)
        best_id = st_ids[0]
        lines.append("")
        lines.append("* Seed Tournament\n")
        lines.append(
            f"The evolved winner ({best_id}) competed against all {n_st - 1} original seeds "
            f"({n_st} plans, {n_st * (n_st - 1)} matches).\n")

        lines.append("** Ranking\n")
        sorted_indices = sorted(range(n_st), key=lambda k: st_ranks[k])
        for idx in sorted_indices:
            suffix = " <- evolved winner" if idx == 0 else ""
            lines.append(f"- {st_ids[idx]} (win rate {st_win_rates[idx]:.0%}){suffix}")
        lines.append("")

        lines.append("** Result\n")
        lines.append(f"Evolved winner {best_id}: rank {st_ranks[0]} of {n_st}, "
                     f"win rate {st_win_rates[0]:.0%}.\n")

    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --- Plan snippet for logging --------------------------------------------------

def _plan_snippet(text: str, head: int = 30, tail: int = 30) -> str:
    text = " ".join(text.split())
    if len(text) <= head + tail + 10:
        return f'{len(text)} chars, "{text}"'
    return f'{len(text)} chars, "{text[:head]} ... {text[-tail:]}"'


def _unavailable_audit_slots(
    mutant_results: list[dict],
    offspring_results: list[dict],
) -> list[str]:
    """Identify generated slots whose fidelity audit did not complete."""
    unavailable = []
    for operator, results in (
        ("mutation", mutant_results),
        ("crossover", offspring_results),
    ):
        unavailable.extend(
            f"{operator} {slot}"
            for slot, result in enumerate(results, 1)
            if result.get("audit", {}).get("unavailable", False)
        )
    return unavailable


# --- Commit-by-verification round ---------------------------------------------

COMMIT_FIELDNAMES = [
    "phase",
    "plan_id",
    "role",
    "first_step",
    "wins",
    "matches",
    "benchmark_quality",
    "novelty",
    "selected",
    "plan_text",
]


async def _benchmark_entries(
    evaluator: Evaluator,
    entries: list[dict],
    seed_ids: list[str],
    seed_plans: list[str],
    match_file: str,
    phase: str,
) -> list[float]:
    """Score each entry against every archive seed in both orders.

    Returns per-entry wins out of ``2 * len(seed_ids)`` matches.  Every match
    is appended to the global matches record under mode
    ``global_commit_<phase>``.
    """
    specs = [
        (eidx, sid, splan, direction)
        for eidx in range(len(entries))
        for sid, splan in zip(seed_ids, seed_plans)
        for direction in ("candidate_first", "seed_first")
    ]
    wins = [0.0] * len(entries)
    match_rows = []
    for start in range(0, len(specs), config.MATCH_BATCH_SIZE):
        batch = specs[start:start + config.MATCH_BATCH_SIZE]
        tasks = []
        for eidx, sid, splan, direction in batch:
            entry = entries[eidx]
            if direction == "candidate_first":
                tasks.append(evaluator._match(
                    entry["plan"], splan,
                    label_a=f"commit {entry['plan_id']}", label_b=f"seed {sid}"))
            else:
                tasks.append(evaluator._match(
                    splan, entry["plan"],
                    label_a=f"seed {sid}", label_b=f"commit {entry['plan_id']}"))
        results = await asyncio.gather(*tasks)
        for (eidx, sid, _splan, direction), (winner, analysis) in zip(batch, results):
            entry = entries[eidx]
            if winner == "TIE":
                wins[eidx] += 0.5
                winner_id, winner_label = "tie", f"tie:{direction}"
            elif (winner == "A") == (direction == "candidate_first"):
                wins[eidx] += 1.0
                winner_id, winner_label = entry["plan_id"], f"candidate:{direction}"
            else:
                winner_id, winner_label = sid, f"seed:{direction}"
            match_rows.append(
                {
                    "mode": f"global_commit_{phase}",
                    "step": "commit",
                    "plan_a_slot": eidx,
                    "plan_a_id": entry["plan_id"],
                    "plan_b_slot": -1,
                    "plan_b_id": sid,
                    "winner_id": winner_id,
                    "winner_label": winner_label,
                    "analysis": analysis,
                }
            )
    append_rows(match_file, MATCH_FIELDNAMES, match_rows)
    return wins


async def run_commit_round(
    evaluator: Evaluator,
    mutator,
    commit_pool: dict[str, dict],
    seed_ids: list[str],
    seed_plans: list[str],
    novelty_measure,
    rng: random.Random,
    match_file: str,
    commit_file: str,
) -> tuple[str, str, dict]:
    """Select the champion by verification against the seed archive.

    Every plan that ever held an elite slot or won a step plays 30 matches
    against the archive; the champion is the benchmark-best plan (ties broken
    by semantic novelty, then earlier discovery).  A short polish round then
    mutates the champion and keeps the best verified result.
    """
    header = "── Commit Round (verification vs. seed archive) "
    logger.info(header + "─" * (60 - len(header)))
    entries = sorted(
        commit_pool.values(), key=lambda e: (e["first_step"], e["plan_id"])
    )
    n_matches = 2 * len(seed_ids)
    logger.info(
        "Commit candidates: %d unique plans x %d matches each (%d total)...",
        len(entries), n_matches, len(entries) * n_matches,
    )
    wins = await _benchmark_entries(
        evaluator, entries, seed_ids, seed_plans, match_file, "search"
    )
    novelty_scores = novelty_measure.novelty_scores([e["plan"] for e in entries])

    order = sorted(
        range(len(entries)),
        key=lambda i: (
            -wins[i],
            -novelty_scores[i],
            entries[i]["first_step"],
            entries[i]["plan_id"],
        ),
    )
    champ_idx = order[0]
    champion_id = entries[champ_idx]["plan_id"]
    champion_plan = entries[champ_idx]["plan"]
    champion_wins = wins[champ_idx]
    logger.info(
        "  Commit ranking (top 5): %s",
        [
            (entries[i]["plan_id"], f"{wins[i]:.1f}/{n_matches}",
             f"nov={novelty_scores[i]:.3f}")
            for i in order[:5]
        ],
    )
    logger.info(
        "  Champion by verification: %s (%.1f/%d vs. archive)",
        champion_id, champion_wins, n_matches,
    )

    commit_rows = [
        {
            "phase": "search",
            "plan_id": entries[i]["plan_id"],
            "role": entries[i]["role"],
            "first_step": entries[i]["first_step"],
            "wins": f"{wins[i]:.1f}",
            "matches": n_matches,
            "benchmark_quality": f"{wins[i] / n_matches:.6f}",
            "novelty": f"{novelty_scores[i]:.6f}",
            "selected": i == champ_idx,
            "plan_text": entries[i]["plan"],
        }
        for i in range(len(entries))
    ]

    # Polish round: a few extra single-component mutations of the champion,
    # each audited, then verified against the archive under the same rule.
    polish_adopted = False
    n_polish = min(config.COMMIT_POLISH_VARIANTS, len(COMPONENTS))
    if n_polish > 0:
        polish_components = rng.sample(list(COMPONENTS), k=n_polish)
        logger.info("  Polish round: mutating champion on %s", polish_components)
        polish_results = await mutator.mutate_batch(
            [champion_plan] * n_polish,
            [champion_id] * n_polish,
            polish_components,
            config.NUM_STEPS + 1,
        )
        unavailable = [
            f"polish {slot}"
            for slot, result in enumerate(polish_results, 1)
            if result.get("audit", {}).get("unavailable", False)
        ]
        if unavailable:
            raise RuntimeError(
                "Fidelity auditing was unavailable for commit polish slot(s) "
                + ", ".join(unavailable)
                + ". Rerun the same run ID to resume from cached calls."
            )
        polish_entries = []
        for result in polish_results:
            if result["action"] != "accepted":
                continue
            if _norm_text(result["plan"]) == _norm_text(champion_plan):
                continue
            polish_entries.append(
                {
                    "plan_id": result["child_id"],
                    "plan": result["plan"],
                    "role": f"polish({result['dimension']})",
                    "first_step": config.NUM_STEPS + 1,
                }
            )
        if polish_entries:
            polish_wins = await _benchmark_entries(
                evaluator, polish_entries, seed_ids, seed_plans, match_file,
                "polish",
            )
            polish_novelty = novelty_measure.novelty_scores(
                [e["plan"] for e in polish_entries]
            )
            best_polish = max(range(len(polish_entries)), key=lambda i: polish_wins[i])
            for i, entry in enumerate(polish_entries):
                commit_rows.append(
                    {
                        "phase": "polish",
                        "plan_id": entry["plan_id"],
                        "role": entry["role"],
                        "first_step": entry["first_step"],
                        "wins": f"{polish_wins[i]:.1f}",
                        "matches": n_matches,
                        "benchmark_quality": f"{polish_wins[i] / n_matches:.6f}",
                        "novelty": f"{polish_novelty[i]:.6f}",
                        "selected": False,
                        "plan_text": entry["plan"],
                    }
                )
            if polish_wins[best_polish] > champion_wins:
                polish_adopted = True
                for row in commit_rows:
                    row["selected"] = False
                commit_rows_index = len(commit_rows) - len(polish_entries) + best_polish
                commit_rows[commit_rows_index]["selected"] = True
                champion_id = polish_entries[best_polish]["plan_id"]
                champion_plan = polish_entries[best_polish]["plan"]
                champion_wins = polish_wins[best_polish]
                logger.info(
                    "  Polish adopted: %s (%.1f/%d vs. archive)",
                    champion_id, champion_wins, n_matches,
                )
            else:
                logger.info("  Polish round did not beat the champion; keeping it.")

    initialize_csv(commit_file, COMMIT_FIELDNAMES)
    append_rows(commit_file, COMMIT_FIELDNAMES, commit_rows)
    logger.info("Commit record: %s", commit_file)

    summary = {
        "n_candidates": len(entries),
        "n_matches": n_matches,
        "champion_id": champion_id,
        "champion_wins": champion_wins,
        "polish_adopted": polish_adopted,
    }
    return champion_id, champion_plan, summary


# --- Main loop -----------------------------------------------------------------

async def main(args: argparse.Namespace | None = None):
    firm_ids = apply_cli_overrides(args)
    config.validate_global_search_config()
    t0 = time.time()
    run_prefix = args.run_prefix if args and args.run_prefix else datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = (
        args.output_dir
        if args and args.output_dir
        else "out"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{run_prefix}-{config.GS_OUTPUT_FILE}")
    log_file = os.path.join(output_dir, f"{run_prefix}-{config.GS_LOG_FILE}")
    narrative_file = os.path.join(output_dir, f"{run_prefix}-{config.GS_NARRATIVE_FILE}")
    candidate_file = os.path.join(output_dir, f"{run_prefix}-global-candidates.csv")
    match_file = os.path.join(output_dir, f"{run_prefix}-global-matches.csv")
    audit_file = os.path.join(output_dir, f"{run_prefix}-global-fidelity-audits.csv")
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

    rng = random.Random(config.RANDOM_SEED)
    selection_pool_size = config.get_selection_pool_size()

    # Load seed population
    seeds = load_seed_plans(config.INPUT_FILE, config.POP_SIZE, firm_ids=firm_ids)
    plan_ids = [s[0] for s in seeds]
    plans = [s[1] for s in seeds]
    origins = ["seed"] * config.POP_SIZE
    seed_prompt = f"seed from {config.INPUT_FILE}"
    provenance = [seed_prompt] * config.POP_SIZE
    candidate_meta = [
        {
            "parent_a": "",
            "parent_b": "",
            "transformation": "seed",
            "requested_change": "",
            "generated_plan": plan,
            "fidelity_passed": True,
            "fidelity_action": "source",
            "unsupported_claims": [],
            "fidelity_analysis": "Original source description.",
            "evidence_notes": "",
            "prompt": seed_prompt,
        }
        for plan in plans
    ]

    seed_ids = list(plan_ids)
    seed_plans = list(plans)
    seed_lookup = dict(zip(seed_ids, seed_plans))

    # Founding-seed lineage of every population slot (diversity-preserving
    # elitism and the recombination partner rule both read this).
    root_sets: list[frozenset[str]] = [frozenset([pid]) for pid in plan_ids]
    # Archive seeds ordered by step-0 quality; filled after the first evaluation.
    seed_rank_order: list[str] = list(seed_ids)
    # Every plan that ever holds an elite slot or wins a step is a candidate
    # for the commit-by-verification round after the search steps.
    commit_pool: dict[str, dict] = {}

    def add_to_commit_pool(plan_id: str, plan_text: str, step: int, role: str) -> None:
        key = _norm_text(plan_text)
        if key not in commit_pool:
            commit_pool[key] = {
                "plan_id": plan_id,
                "plan": plan_text,
                "first_step": step,
                "role": role,
            }

    novelty_measure = SemanticNovelty(
        seed_ids,
        seed_plans,
        model_name=config.NOVELTY_EMBEDDING_MODEL,
        model_revision=config.NOVELTY_EMBEDDING_REVISION,
        device=config.NOVELTY_EMBEDDING_DEVICE,
        mock=config.MOCK_LLM,
    )

    logger.info("Loaded %d seed plans: %s", config.POP_SIZE, plan_ids)
    logger.info("Config: steps=%d, pop=%d, elite=%d, mutant=%d, offspring=%d, pool=%d, max_concurrent_calls=%d",
                config.NUM_STEPS, config.POP_SIZE, config.POP_ELITE,
                config.POP_MUTANT, config.POP_OFFSPRING, selection_pool_size,
                config.MAX_CONCURRENT_CALLS)
    logger.info(
        "Novelty: nearest original seed; mean cosine distance across %s; "
        "backend=%s; model=%s; revision=%s; device=%s",
        ", ".join(COMPONENTS),
        novelty_measure.backend,
        config.NOVELTY_EMBEDDING_MODEL,
        config.NOVELTY_EMBEDDING_REVISION,
        config.NOVELTY_EMBEDDING_DEVICE,
    )

    n_eval_per_gen = config.POP_SIZE * (config.POP_SIZE - 1)
    n_seed_tourney = (config.POP_SIZE + 1) * config.POP_SIZE  # winner + POP_SIZE seeds
    n_eval_total = n_eval_per_gen * (config.NUM_STEPS + 1) + n_seed_tourney
    n_mut_total = config.POP_MUTANT * config.NUM_STEPS
    n_cross_total = config.POP_OFFSPRING * config.NUM_STEPS
    n_fidelity_total = (config.POP_MUTANT + config.POP_OFFSPRING) * config.NUM_STEPS
    n_narr = config.NUM_STEPS * 2  # diff + lineage per search step (upper bound)
    logger.info(
        "Expected # of calls: Evaluator (%s): %d, Mutator (%s): %d, "
        "Crossover (%s): %d, Fidelity Auditor (%s): %d, Narrator (%s): ≤%d. Total: ~%d",
        config.EVALUATOR_MODEL, n_eval_total,
        config.MUTATOR_MODEL, n_mut_total,
        config.CROSSOVER_MODEL, n_cross_total,
        config.FIDELITY_AUDITOR_MODEL, n_fidelity_total,
        config.NARRATOR_MODEL, n_narr,
        n_eval_total + n_mut_total + n_cross_total + n_fidelity_total + n_narr,
    )

    evaluator = Evaluator()
    mutator = Mutator()
    crossover = Crossover()
    narrator = Narrator()

    fieldnames = _build_fieldnames()
    write_header(fieldnames, output_file)
    initialize_csv(candidate_file, CANDIDATE_FIELDNAMES)
    initialize_csv(match_file, MATCH_FIELDNAMES)
    initialize_csv(audit_file, AUDIT_FIELDNAMES)

    history = []  # (step, best_id, best_plan, plan_ids, plans, ranks, win_rates, origins)

    for generation in range(1, config.NUM_STEPS + 1):
        current_step = generation - 1
        header = f"── Step {current_step}/{config.NUM_STEPS} "
        logger.info(header + "─" * (60 - len(header)))

        # STEP 1: Evaluate fitness via double round-robin
        n = len(plans)
        logger.info("Evaluating %d plans (%d matches)...", n, n * (n - 1))

        # Build match_log_extras with CSV cell refs for each pairwise comparison
        match_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        eval_extras = {
            (i, j): (
                f"; comparing cells {_csv_cell(fieldnames, f'plan_{i}', current_step)} "
                f"and {_csv_cell(fieldnames, f'plan_{j}', current_step)} in CSV"
            )
            for i, j in match_pairs
        }
        ranks, win_rates, match_details = await evaluator.evaluate(
            plans,
            match_log_extras=eval_extras,
            tie_break_rng=rng,
        )
        novelty_scores = novelty_measure.novelty_scores(plans)
        selection = combine_quality_and_novelty(
            ranks,
            novelty_scores,
            quality_weight=config.QUALITY_SELECTION_WEIGHT,
            novelty_weight=config.NOVELTY_SELECTION_WEIGHT,
        )

        best_idx = ranks.index(1)
        best_id = plan_ids[best_idx]
        best_plan = plans[best_idx]
        mean_wr = sum(win_rates) / len(win_rates)
        max_wr = max(win_rates)

        if current_step == 0:
            seed_rank_order = [
                plan_ids[i] for i in sorted(range(n), key=lambda i: ranks[i])
            ]
        add_to_commit_pool(best_id, best_plan, current_step, "step_winner")

        logger.info("  Ranks: %s (cells %s:%s in CSV)",
                    list(zip(plan_ids, ranks)),
                    _csv_cell(fieldnames, "rank_0", current_step),
                    _csv_cell(fieldnames, f"rank_{n - 1}", current_step))
        logger.info("  Win rates: %s (cells %s:%s in CSV)",
                    [f"{plan_ids[i]}={win_rates[i]:.0%}" for i in range(n)],
                    _csv_cell(fieldnames, "win_rate_0", current_step),
                    _csv_cell(fieldnames, f"win_rate_{n - 1}", current_step))
        logger.info("  Best: %s (win rate %.0f%%, rank 1, cell %s in CSV)",
                    best_id, win_rates[best_idx] * 100,
                    _csv_cell(fieldnames, f"plan_{best_idx}", current_step))
        logger.info("  Mean win rate: %.0f%%, Max: %.0f%%", mean_wr * 100, max_wr * 100)
        logger.info(
            "  Novelty and combined selection: %s",
            [
                (
                    plan_ids[i],
                    f"novelty={novelty_scores[i]:.3f}",
                    f"combined={selection.combined_scores[i]:.3f}",
                    f"selection_rank={selection.selection_ranks[i]}",
                )
                for i in range(n)
            ],
        )

        history.append((current_step, best_id, best_plan,
                        list(plan_ids), list(plans), list(ranks),
                        list(win_rates), list(origins)))

        append_match_records(match_file, "global", current_step, plan_ids, match_details)
        append_candidate_records(
            candidate_file,
            current_step,
            plan_ids,
            plans,
            origins,
            ranks,
            win_rates,
            novelty_scores,
            selection,
            candidate_meta,
            root_sets=root_sets,
        )

        # Write CSV row
        append_row(fieldnames, output_file, current_step, best_id, best_plan,
                   plan_ids, plans, ranks, win_rates, novelty_scores,
                   selection, origins, provenance)

        # STEP 2: Equal-weight quality-novelty selection
        sorted_indices = sorted(range(n), key=lambda i: selection.selection_ranks[i])
        pool_indices = sorted_indices[:selection_pool_size]
        pool_ids = [plan_ids[i] for i in pool_indices]
        pool_plans = [plans[i] for i in pool_indices]
        pool_ranks = [selection.selection_ranks[i] for i in pool_indices]
        logger.info("  Combined-rank pool (top %d): %s", selection_pool_size, pool_ids)

        # STEP 3: Build next generation

        # 3a. Elite: keep the highest combined-ranked plans unchanged, holding
        # each founding lineage to at most ELITE_LINEAGE_CAP slots so several
        # lineages stay alive and available for recombination.
        elite_indices = select_elites_with_lineage_cap(
            sorted_indices,
            root_sets,
            config.POP_ELITE,
            config.ELITE_LINEAGE_CAP,
        )
        uncapped = sorted_indices[:config.POP_ELITE]
        if elite_indices != uncapped:
            displaced = [plan_ids[i] for i in uncapped if i not in elite_indices]
            admitted = [plan_ids[i] for i in elite_indices if i not in uncapped]
            logger.info(
                "  Lineage cap (max %d per founding seed): displaced %s, admitted %s",
                config.ELITE_LINEAGE_CAP, displaced, admitted,
            )
        new_ids = [plan_ids[i] for i in elite_indices]
        new_plans = [plans[i] for i in elite_indices]
        new_root_sets = [root_sets[i] for i in elite_indices]
        new_origins = ["elite"] * config.POP_ELITE
        new_provenance = [
            f"elite -- copied unchanged from cell {_csv_cell(fieldnames, f'plan_{i}', current_step)} in CSV"
            for i in elite_indices
        ]
        for i in elite_indices:
            add_to_commit_pool(plan_ids[i], plans[i], current_step, "elite")
        new_candidate_meta = [
            {
                "parent_a": plan_ids[i],
                "parent_b": "",
                "transformation": "elite",
                "requested_change": "",
                "generated_plan": plans[i],
                "fidelity_passed": True,
                "fidelity_action": "retained",
                "unsupported_claims": [],
                "fidelity_analysis": "Copied unchanged from an audited or source state.",
                "evidence_notes": "",
                "prompt": new_provenance[pos],
            }
            for pos, i in enumerate(elite_indices)
        ]
        for ei, i in enumerate(elite_indices):
            logger.info("  Elite: %s (cell %s in CSV)",
                        plan_ids[i], _csv_cell(fieldnames, f"plan_{i}", current_step))

        # 3b. Mutants: pick POP_MUTANT plans from pool (rank-proportional)
        mutant_parent_ids = rank_proportional_select(pool_ids, pool_ranks, config.POP_MUTANT, rng)
        mutant_parent_plans = [pool_plans[pool_ids.index(pid)] for pid in mutant_parent_ids]
        mutant_parent_roots = [root_sets[plan_ids.index(pid)] for pid in mutant_parent_ids]
        mutant_components = [rng.choice(COMPONENTS) for _ in range(config.POP_MUTANT)]

        # Precompute mutant prompts
        mutant_prompts = [
            MUTATOR_PROMPT.format(
                plan=parent_plan,
                component=component,
                component_guidance=MUTATION_COMPONENT_GUIDANCE[component],
                fidelity_rules=FACTUAL_FIDELITY_RULES,
            )
            for parent_plan, component in zip(mutant_parent_plans, mutant_components)
        ]

        # Compute parent cell refs for log
        mutant_parent_cells = [
            _csv_cell(fieldnames, f"plan_{plan_ids.index(pid)}", current_step)
            for pid in mutant_parent_ids
        ]
        logger.info("  Mutating: %s (parent cells %s in CSV)",
                    list(zip(mutant_parent_ids, mutant_components)),
                    [f"{pid}={c}" for pid, c in zip(mutant_parent_ids, mutant_parent_cells)])

        # 3c. Offspring: first parent from the combined-rank pool; second is
        # the best-ranked eligible plan from a different founding lineage
        # (current pool by selection rank, then archive seeds by step-0
        # quality), so recombination pairs good, complementary material.
        first_parent_ids = rank_proportional_select(
            pool_ids, pool_ranks, config.POP_OFFSPRING, rng
        )
        eligible_second_parents: list[dict[str, object]] = []
        seen_candidates: set[tuple[str, str]] = set()
        for index in pool_indices:
            key = (plan_ids[index], _norm_text(plans[index]))
            if key not in seen_candidates:
                eligible_second_parents.append(
                    {
                        "candidate_id": plan_ids[index],
                        "plan": plans[index],
                        "roots": root_sets[index],
                        "source": "current",
                        "slot": index,
                    }
                )
                seen_candidates.add(key)
        for seed_id in seed_rank_order:
            seed_plan = seed_lookup[seed_id]
            key = (seed_id, _norm_text(seed_plan))
            if key not in seen_candidates:
                eligible_second_parents.append(
                    {
                        "candidate_id": seed_id,
                        "plan": seed_plan,
                        "roots": frozenset([seed_id]),
                        "source": "seed_archive",
                        "slot": -1,
                    }
                )
                seen_candidates.add(key)

        offspring_pairs = []
        offspring_second_sources = []
        offspring_crossover_prompts = []
        offspring_a_roots = []
        offspring_b_roots = []
        used_pairs: set[tuple[str, str]] = set()
        for a_id in first_parent_ids:
            a_index = plan_ids.index(a_id)
            a_plan = plans[a_index]
            a_roots = root_sets[a_index]
            second = pick_recombination_partner(
                a_id,
                a_plan,
                a_roots,
                eligible_second_parents,
                used_pairs,
            )
            if second is None:
                raise RuntimeError(
                    f"No eligible crossover partner for {a_id} at step {current_step}"
                )
            b_id = str(second["candidate_id"])
            b_plan = str(second["plan"])
            used_pairs.add((a_id, b_id))
            offspring_pairs.append((a_plan, b_plan, a_id, b_id))
            offspring_second_sources.append(second)
            offspring_a_roots.append(a_roots)
            offspring_b_roots.append(frozenset(second["roots"]))
            offspring_crossover_prompts.append(
                CROSSOVER_PROMPT.format(
                    plan_a=a_plan,
                    plan_b=b_plan,
                    criteria=EVALUATION_CRITERIA,
                    fidelity_rules=FACTUAL_FIDELITY_RULES,
                )
            )

        offspring_parent_cells = []
        for pair, second in zip(offspring_pairs, offspring_second_sources):
            first_index = plan_ids.index(pair[2])
            first_ref = _csv_cell(fieldnames, f"plan_{first_index}", current_step)
            second_ref = (
                _csv_cell(fieldnames, f"plan_{int(second['slot'])}", current_step)
                if second["source"] == "current"
                else "original seed archive"
            )
            offspring_parent_cells.append(
                (f"{pair[2]}={first_ref}", f"{pair[3]}={second_ref}")
            )
        logger.info("  Crossing: %s (parent cells %s in CSV)",
                    [(p[2], p[3]) for p in offspring_pairs], offspring_parent_cells)

        # Build log_extras for agent calls (child cell refs in next generation's row)
        mutant_log_extras = [
            f"; cell {_csv_cell(fieldnames, f'plan_{config.POP_ELITE + k}', current_step + 1)} in CSV"
            for k in range(config.POP_MUTANT)
        ]
        offspring_log_extras = [
            f"; cell {_csv_cell(fieldnames, f'plan_{config.POP_ELITE + config.POP_MUTANT + k}', current_step + 1)} in CSV"
            for k in range(config.POP_OFFSPRING)
        ]

        # Run mutation and crossover concurrently
        mutant_results, offspring_results = await asyncio.gather(
            mutator.mutate_batch(mutant_parent_plans, mutant_parent_ids,
                                 mutant_components, generation,
                                 log_extras=mutant_log_extras),
            crossover.cross_batch(offspring_pairs, generation,
                                  log_extras=offspring_log_extras),
        )

        unavailable_audits = _unavailable_audit_slots(mutant_results, offspring_results)
        if unavailable_audits:
            slots = ", ".join(unavailable_audits)
            raise RuntimeError(
                "Fidelity auditing was unavailable for global candidate slot(s) "
                f"{slots} at step {current_step + 1}. The run is stopping instead "
                "of treating service failures as substantive rejections; rerun "
                "the same run ID to resume from cached successful calls."
            )

        audit_rows = []
        for k, result in enumerate(mutant_results):
            plan_text = result["plan"]
            dimension = result["dimension"]
            child_id = result["child_id"]
            new_ids.append(child_id)
            new_plans.append(plan_text)
            new_root_sets.append(mutant_parent_roots[k])
            parent_id = mutant_parent_ids[k]
            new_origins.append(f"mutant({parent_id},{dimension})")
            new_provenance.append(mutant_prompts[k])
            audit = result["audit"]
            new_candidate_meta.append(
                {
                    "parent_a": parent_id,
                    "parent_b": "",
                    "transformation": "mutation",
                    "requested_change": dimension,
                    "generated_plan": result["generated_plan"],
                    "fidelity_passed": audit["passed"],
                    "fidelity_action": result["action"],
                    "unsupported_claims": audit["unsupported_claims"],
                    "fidelity_analysis": audit["analysis"],
                    "evidence_notes": "",
                    "prompt": mutant_prompts[k],
                }
            )
            audit_rows.append(
                {
                    "mode": "global",
                    "step": current_step + 1,
                    "candidate_id": child_id,
                    "operator": "mutation",
                    "parent_a": parent_id,
                    "parent_b": "",
                    "model_passed": audit.get("model_passed", audit["passed"]),
                    "decision_basis": audit.get("decision_basis", "model_audit"),
                    "model_unsupported_claims": claims_json(
                        audit.get("model_unsupported_claims", audit["unsupported_claims"])
                    ),
                    "passed": audit["passed"],
                    "action": result["action"],
                    "unsupported_claims": claims_json(audit["unsupported_claims"]),
                    "analysis": audit["analysis"],
                    "generated_plan": result["generated_plan"],
                    "evaluated_plan": result["plan"],
                }
            )
            child_cell = _csv_cell(fieldnames, f"plan_{config.POP_ELITE + k}", current_step + 1)
            logger.info("    Mutant: %s (cell %s in CSV) -> %s (cell %s in CSV) (dim: %s; %s)",
                        parent_id, mutant_parent_cells[k],
                        child_id, child_cell,
                        dimension, _plan_snippet(plan_text))

        for k, result in enumerate(offspring_results):
            plan_text = result["plan"]
            rationale = result["rationale"]
            child_id = result["child_id"]
            new_ids.append(child_id)
            new_plans.append(plan_text)
            # A reverted crossover evaluates parent A's text, so it carries
            # only parent A's founding lineage; an accepted blend carries both.
            if result["action"] == "accepted":
                new_root_sets.append(offspring_a_roots[k] | offspring_b_roots[k])
            else:
                new_root_sets.append(offspring_a_roots[k])
            pair = offspring_pairs[k]
            new_origins.append(f"crossover({pair[2]},{pair[3]})")
            new_provenance.append(offspring_crossover_prompts[k])
            audit = result["audit"]
            new_candidate_meta.append(
                {
                    "parent_a": pair[2],
                    "parent_b": pair[3],
                    "transformation": "crossover",
                    "requested_change": rationale,
                    "generated_plan": result["generated_plan"],
                    "fidelity_passed": audit["passed"],
                    "fidelity_action": result["action"],
                    "unsupported_claims": audit["unsupported_claims"],
                    "fidelity_analysis": audit["analysis"],
                    "evidence_notes": rationale,
                    "prompt": offspring_crossover_prompts[k],
                }
            )
            audit_rows.append(
                {
                    "mode": "global",
                    "step": current_step + 1,
                    "candidate_id": child_id,
                    "operator": "crossover",
                    "parent_a": pair[2],
                    "parent_b": pair[3],
                    "model_passed": audit.get("model_passed", audit["passed"]),
                    "decision_basis": audit.get("decision_basis", "model_audit"),
                    "model_unsupported_claims": claims_json(
                        audit.get("model_unsupported_claims", audit["unsupported_claims"])
                    ),
                    "passed": audit["passed"],
                    "action": result["action"],
                    "unsupported_claims": claims_json(audit["unsupported_claims"]),
                    "analysis": audit["analysis"],
                    "generated_plan": result["generated_plan"],
                    "evaluated_plan": result["plan"],
                }
            )
            child_cell = _csv_cell(fieldnames, f"plan_{config.POP_ELITE + config.POP_MUTANT + k}", current_step + 1)
            logger.info("    Offspring: %s (cell %s in CSV, from %s x %s; %s)",
                        child_id, child_cell,
                        pair[2], pair[3], _plan_snippet(plan_text))

        append_rows(audit_file, AUDIT_FIELDNAMES, audit_rows)

        # Lineage stats
        origin_types = {}
        for o in new_origins:
            key = o.split("(")[0]
            origin_types[key] = origin_types.get(key, 0) + 1
        logger.info("  Step %d composition: %s", current_step + 1, origin_types)

        # Update population
        plan_ids = new_ids
        plans = new_plans
        origins = new_origins
        provenance = new_provenance
        candidate_meta = new_candidate_meta
        root_sets = new_root_sets

    # FINAL EVALUATION
    final_step = config.NUM_STEPS
    header = f"── Step {final_step}/{config.NUM_STEPS}: Final Evaluation "
    logger.info(header + "─" * (60 - len(header)))
    n = len(plans)
    logger.info("Final evaluation: %d plans (%d matches)...", n, n * (n - 1))

    match_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    eval_extras = {
        (i, j): (
            f"; comparing cells {_csv_cell(fieldnames, f'plan_{i}', final_step)} "
            f"and {_csv_cell(fieldnames, f'plan_{j}', final_step)} in CSV"
        )
        for i, j in match_pairs
    }
    ranks, win_rates, match_details = await evaluator.evaluate(
        plans,
        match_log_extras=eval_extras,
        tie_break_rng=rng,
    )
    novelty_scores = novelty_measure.novelty_scores(plans)
    selection = combine_quality_and_novelty(
            ranks,
            novelty_scores,
            quality_weight=config.QUALITY_SELECTION_WEIGHT,
            novelty_weight=config.NOVELTY_SELECTION_WEIGHT,
        )

    best_idx = ranks.index(1)
    best_id = plan_ids[best_idx]
    best_plan = plans[best_idx]
    logger.info("  Final ranks: %s (cells %s:%s in CSV)",
                list(zip(plan_ids, ranks)),
                _csv_cell(fieldnames, "rank_0", final_step),
                _csv_cell(fieldnames, f"rank_{n - 1}", final_step))
    logger.info("  Final win rates: %s (cells %s:%s in CSV)",
                [f"{plan_ids[i]}={win_rates[i]:.0%}" for i in range(n)],
                _csv_cell(fieldnames, "win_rate_0", final_step),
                _csv_cell(fieldnames, f"win_rate_{n - 1}", final_step))
    logger.info("  Final best: %s (win rate %.0f%%, cell %s in CSV)",
                best_id, win_rates[best_idx] * 100,
                _csv_cell(fieldnames, f"plan_{best_idx}", final_step))

    history.append((final_step, best_id, best_plan,
                    list(plan_ids), list(plans), list(ranks),
                    list(win_rates), list(origins)))

    append_match_records(match_file, "global", final_step, plan_ids, match_details)
    append_candidate_records(
        candidate_file,
        final_step,
        plan_ids,
        plans,
        origins,
        ranks,
        win_rates,
        novelty_scores,
        selection,
        candidate_meta,
        root_sets=root_sets,
    )

    # Write final CSV row
    append_row(fieldnames, output_file, final_step, best_id, best_plan,
               plan_ids, plans, ranks, win_rates, novelty_scores,
               selection, origins, provenance)

    logger.info("Output: %s", output_file)

    # The final population's step winner and top-of-pool plans are commit
    # candidates too.
    add_to_commit_pool(best_id, best_plan, final_step, "step_winner")
    final_sorted = sorted(range(n), key=lambda i: selection.selection_ranks[i])
    for i in final_sorted[:config.POP_ELITE]:
        add_to_commit_pool(plan_ids[i], plans[i], final_step, "final_top")

    # COMMIT ROUND: verification against the seed archive selects the champion
    champion_id, champion_plan, commit_summary = await run_commit_round(
        evaluator,
        mutator,
        commit_pool,
        seed_ids,
        seed_plans,
        novelty_measure,
        rng,
        match_file,
        os.path.join(output_dir, f"{run_prefix}-commit-round.csv"),
    )

    # SEED TOURNAMENT: committed champion vs original seeds
    header = "── Seed Tournament "
    logger.info(header + "─" * (60 - len(header)))
    best_id = champion_id
    best_plan = champion_plan
    seed_tourney_ids = [best_id] + seed_ids
    seed_tourney_plans = [best_plan] + seed_plans
    n_st = len(seed_tourney_plans)
    logger.info("Seed tournament: evolved winner + %d original seeds (%d plans, %d matches)...",
                config.POP_SIZE, n_st, n_st * (n_st - 1))
    st_ranks, st_win_rates, st_match_details = await evaluator.evaluate(
        seed_tourney_plans,
        tie_break_rng=rng,
    )
    append_match_records(
        match_file,
        "global_seed_tournament",
        "seed_tournament",
        seed_tourney_ids,
        st_match_details,
    )
    logger.info("  Seed tournament ranks: %s", list(zip(seed_tourney_ids, st_ranks)))
    logger.info("  Seed tournament win rates: %s",
                [f"{seed_tourney_ids[i]}={st_win_rates[i]:.0%}" for i in range(n_st)])
    st_winner_rank = st_ranks[0]
    logger.info("  Evolved winner %s: rank %d, win rate %.0f%% (vs seeds)",
                best_id, st_winner_rank, st_win_rates[0] * 100)

    # Narrative
    seed_tournament = (seed_tourney_ids, st_ranks, st_win_rates)
    await write_narrative(
        history,
        narrator,
        fieldnames,
        narrative_file,
        seed_tournament=seed_tournament,
        seed_lookup=seed_lookup,
        commit_summary=commit_summary,
    )
    logger.info("Narrative: %s", narrative_file)

    # Call summary
    role_models = {
        "Evaluator": config.EVALUATOR_MODEL,
        "Mutator": config.MUTATOR_MODEL,
        "Crossover": config.CROSSOVER_MODEL,
        "Fidelity Auditor": config.FIDELITY_AUDITOR_MODEL,
        "Narrator": config.NARRATOR_MODEL,
    }
    stats = get_call_stats()
    parts = []
    actual_total = 0
    for role in ("Evaluator", "Mutator", "Crossover", "Fidelity Auditor", "Narrator"):
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
