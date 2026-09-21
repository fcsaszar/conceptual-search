#!/usr/bin/env python3
"""One-command search workflow: global search, local search, seed-benchmark
scoring, and semantic scores, with a run manifest that records every setting.

The stages that follow (polish, the local-only comparison, reference-set
scoring, and the final comparison tournament) are separate commands; see the
Makefile and README.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sqlite3
import sys
import time
from typing import Any

sys.dont_write_bytecode = True

import config
import estimate_cost
from global_search_prompts import CROSSOVER_PROMPT, MUTATOR_PROMPT
from local_search_prompts import (
    EVALUATION_CRITERIA,
    EVALUATOR_PROMPT,
    FACTUAL_FIDELITY_RULES,
    FIDELITY_AUDITOR_PROMPT,
    GENERATOR_PROMPT,
    SPECIALIZED_GENERATOR_PROMPT,
)
from semantic_novelty import COMPONENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the conceptual-search pipeline.")
    parser.add_argument("--firm-set", default=config.LOCKED_FIRM_SET_PATH)
    parser.add_argument("--model", default=config.DEEPSEEK_V32_MODEL)
    parser.add_argument("--quality-calibration", choices=("seed-benchmark", "none"), default="seed-benchmark")
    parser.add_argument("--quality-scope", choices=("frontier", "all-points"), default="all-points")
    parser.add_argument("--run-id", default=None, help="Stable prefix shared by all output files.")
    parser.add_argument(
        "--run-kind",
        choices=("smoke", "full"),
        default="full",
        help="Classify the run in its manifest; smoke runs still use all 15 locked firms.",
    )
    parser.add_argument("--steps", type=int, default=config.NUM_STEPS)
    parser.add_argument(
        "--max-concurrent-calls",
        type=int,
        default=40,
        help="Maximum parallel OpenRouter calls used by LLM stages.",
    )
    parser.add_argument(
        "--seed-id",
        default="first",
        help="Local-search seed id. Use `first` to use the first id in --firm-set.",
    )
    parser.add_argument("--input-file", default=config.INPUT_FILE)
    parser.add_argument("--output-dir", default="out", help="Directory for all run records.")
    parser.add_argument("--force", action="store_true", help="Rerun stages even if their expected outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest and print commands without running them.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run a deterministic offline integration smoke without external API calls.",
    )
    return parser.parse_args()


def read_firm_set(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        ids = [line.split("#", 1)[0].strip() for line in f]
    return [firm_id for firm_id in ids if firm_id]


def shlex_join(command: list[str]) -> str:
    return shlex.join(command)


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def search_outputs(output_dir: Path, run_id: str, mode: str) -> dict[str, str]:
    prefix = f"{run_id}-{mode}"
    return {
        "csv": str(output_dir / f"{prefix}-search.csv"),
        "log": str(output_dir / f"{prefix}-search.log"),
        "narrative": str(output_dir / f"{prefix}-search-narrative.txt"),
        "candidates": str(output_dir / f"{prefix}-candidates.csv"),
        "matches": str(output_dir / f"{prefix}-matches.csv"),
        "fidelity_audits": str(output_dir / f"{prefix}-fidelity-audits.csv"),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_input_hash(input_file: str, firm_ids: list[str]) -> str:
    with open(input_file, newline="", encoding="utf-8-sig") as handle:
        by_id = {row["id"]: row["description"] for row in csv.DictReader(handle)}
    missing = [firm_id for firm_id in firm_ids if firm_id not in by_id]
    if missing:
        raise SystemExit(f"Firm ids missing from {input_file}: {', '.join(missing)}")
    payload = "\n".join(f"{firm_id}\t{by_id[firm_id]}" for firm_id in firm_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_usage(path: Path, model: str) -> dict[str, int | float]:
    """Summarize the run's call ledger (every live response, with token use and
    provider-reported cost).  When a ledger is warm-started from an earlier run,
    the totals include the inherited entries."""
    if not path.exists():
        return {
            "unique_live_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "provider_reported_cost_usd": 0.0,
            "token_price_estimate_usd": 0.0,
        }
    connection = sqlite3.connect(path)
    row = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), "
        "COALESCE(SUM(output_tokens), 0), COALESCE(SUM(reported_cost_usd), 0) FROM calls"
    ).fetchone()
    connection.close()
    count, input_tokens, output_tokens, reported_cost = row
    prices = estimate_cost.MODEL_PRICES[model]
    token_estimate = (
        float(input_tokens) / 1_000_000 * prices["input"]
        + float(output_tokens) / 1_000_000 * prices["output"]
    )
    return {
        "unique_live_calls": int(count),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "provider_reported_cost_usd": round(float(reported_cost), 6),
        "token_price_estimate_usd": round(token_estimate, 6),
    }


def refresh_manifest_usage(manifest: dict[str, Any]) -> dict[str, int | float]:
    total = cache_usage(Path(manifest["artifacts"]["call_cache"]), manifest["model"])
    manifest["cache_ledger_usage"] = total
    manifest["actual_usage"] = total
    return total


def outputs_exist(paths: dict[str, str] | list[str]) -> bool:
    values = paths.values() if isinstance(paths, dict) else paths
    return all(Path(path).exists() for path in values)


def model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "").replace("-", "")


def resolve_seed_id(seed_id: str, firm_ids: list[str]) -> str:
    if seed_id == "first":
        if not firm_ids:
            raise SystemExit("Cannot use --seed-id first because the firm set is empty.")
        return firm_ids[0]
    return seed_id


def run_stage(
    manifest_path: Path,
    manifest: dict[str, Any],
    name: str,
    command: list[str],
    expected_outputs: dict[str, str] | list[str],
    force: bool,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> None:
    stage = {
        "name": name,
        "command": shlex_join(command),
        "expected_outputs": expected_outputs,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pending",
    }
    if env:
        stage["environment"] = dict(sorted(env.items()))
    manifest["stages"].append(stage)
    write_manifest(manifest_path, manifest)

    if outputs_exist(expected_outputs) and not force:
        stage["status"] = "skipped_existing_outputs"
        stage["finished_at"] = datetime.now().isoformat(timespec="seconds")
        refresh_manifest_usage(manifest)
        write_manifest(manifest_path, manifest)
        print(f"[skip] {name}: expected outputs already exist")
        return

    if dry_run:
        stage["status"] = "dry_run"
        stage["finished_at"] = datetime.now().isoformat(timespec="seconds")
        refresh_manifest_usage(manifest)
        write_manifest(manifest_path, manifest)
        print(f"[dry-run] {shlex_join(command)}")
        return

    print(f"[run] {name}: {shlex_join(command)}", flush=True)
    t0 = time.time()
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(command, env=run_env)
    stage["returncode"] = result.returncode
    stage["elapsed_minutes"] = round((time.time() - t0) / 60, 2)
    stage["finished_at"] = datetime.now().isoformat(timespec="seconds")
    stage["status"] = "completed" if result.returncode == 0 else "failed"
    refresh_manifest_usage(manifest)
    write_manifest(manifest_path, manifest)

    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    if args.max_concurrent_calls < 1:
        raise SystemExit("--max-concurrent-calls must be at least 1")
    firm_ids = read_firm_set(args.firm_set)
    try:
        config.validate_firm_set(firm_ids)
    except ValueError as exc:
        raise SystemExit(str(exc))
    run_id = args.run_id or (
        f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}-"
        f"{model_slug(args.model)}-locked15-{args.run_kind}"
    )
    local_seed_id = resolve_seed_id(args.seed_id, firm_ids)
    firms = len(firm_ids)
    if local_seed_id != config.SEED_PROJECT_ID:
        raise SystemExit(
            f"The configured local-search seed is {config.SEED_PROJECT_ID!r}; got "
            f"{local_seed_id!r}. Set AI_ENTREP_SEED_PROJECT_ID to change the seed."
        )

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / f"{run_id}-manifest.json"
    call_cache_path = output_dir / f"{run_id}-call-cache.sqlite3"

    cost_inputs_m, cost_outputs_m = estimate_cost.estimate_tokens_m(
        firms,
        args.steps,
        common_quality=(args.quality_calibration == "seed-benchmark"),
        common_quality_scope=args.quality_scope,
    )
    prices = estimate_cost.MODEL_PRICES[args.model]
    estimated_cost = cost_inputs_m * prices["input"] + cost_outputs_m * prices["output"]

    local_outputs = search_outputs(output_dir, run_id, "local")
    global_outputs = search_outputs(output_dir, run_id, "global")
    quality_outputs = {
        "summary": str(output_dir / f"{run_id}-common-quality.csv"),
        "matches": str(output_dir / f"{run_id}-common-quality-matches.csv"),
        "log": str(output_dir / f"{run_id}-common-quality.log"),
    }
    semantic_outputs = {
        "scores": str(output_dir / f"{run_id}-semantic-scores.csv"),
    }

    design_files = [
        "src/local_search_prompts.py",
        "src/global_search_prompts.py",
        "src/semantic_novelty.py",
        "src/semantic_scores.py",
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": args.run_kind,
        "execution_mode": (
            "dry_run"
            if args.dry_run
            else "offline_mock"
            if args.mock
            else "live_openrouter"
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "working_directory": str(Path.cwd()),
        "git_commit": current_git_commit(),
        "model": args.model,
        "provider": "OpenRouter",
        "api_base_url": config.OPENROUTER_BASE_URL,
        "input_file": args.input_file,
        "firm_set": args.firm_set,
        "firm_ids": firm_ids,
        "population_design": {
            "size": config.POP_SIZE,
            "elites": config.POP_ELITE,
            "mutants": config.POP_MUTANT,
            "crossover_children": config.POP_OFFSPRING,
            "selection_pool_fraction": config.SELECTION_POOL_FRAC,
            "selection_rule": "average of quality and novelty percentile ranks",
            "quality_weight": config.QUALITY_SELECTION_WEIGHT,
            "novelty_weight": config.NOVELTY_SELECTION_WEIGHT,
            "elite_lineage_cap": config.ELITE_LINEAGE_CAP,
            "mutation_rule": "one randomly selected component",
            "crossover_first_parent": "rank-proportional draw from the top-ten selection ranks",
            "crossover_second_parent": (
                "best-ranked eligible plan (current parent pool, then the original "
                "seeds) that adds a founding lineage the first parent lacks"
            ),
        },
        "semantic_novelty": {
            "construct": "mean component-wise cosine distance from the nearest original seed",
            "components": list(COMPONENTS),
            "model": config.NOVELTY_EMBEDDING_MODEL,
            "revision": config.NOVELTY_EMBEDDING_REVISION,
            "device": config.NOVELTY_EMBEDDING_DEVICE,
            "backend": "mock-hash" if args.mock else "sentence-transformers",
        },
        "local_seed_id": local_seed_id,
        "local_seed_arg": args.seed_id,
        "steps": args.steps,
        "random_seed": config.RANDOM_SEED,
        "max_concurrent_calls": args.max_concurrent_calls,
        "quality_calibration": args.quality_calibration,
        "quality_scope": args.quality_scope if args.quality_calibration == "seed-benchmark" else "none",
        "cost": {
            "scope": "global search, local search, seed-benchmark scoring, semantic scores",
            "estimated_usd": round(estimated_cost, 4),
            "estimated_tokens_m": {
                "input": round(cost_inputs_m, 3),
                "output": round(cost_outputs_m, 3),
            },
            "price_usd_per_million_tokens": estimate_cost.MODEL_PRICES[args.model],
            "price_snapshot_date": estimate_cost.PRICE_SNAPSHOT_DATE,
            "price_source": estimate_cost.MODEL_PRICE_SOURCES.get(args.model, ""),
        },
        "model_settings": {
            "generator": {"model": args.model, "temperature": config.GENERATOR_TEMPERATURE},
            "specialized_generator": {"model": args.model, "temperature": config.SPECIALIZED_GENERATOR_TEMPERATURE},
            "mutator": {"model": args.model, "temperature": config.MUTATOR_TEMPERATURE},
            "crossover": {"model": args.model, "temperature": config.CROSSOVER_TEMPERATURE},
            "fidelity_auditor": {"model": args.model, "temperature": config.FIDELITY_AUDITOR_TEMPERATURE},
            "evaluator": {
                "model": args.model,
                "temperature": config.EVALUATOR_TEMPERATURE,
                "design": "double round robin with prompt order reversed",
                "criteria": EVALUATION_CRITERIA,
            },
            "narrator": {"model": args.model, "temperature": config.NARRATOR_TEMPERATURE},
            "max_response_tokens": config.MAX_RESPONSE_TOKENS,
            "reasoning_effort": config.REASONING_EFFORT,
        },
        "prompts": {
            "schema_version": config.PROMPT_SCHEMA_VERSION,
            "factual_fidelity_rules": FACTUAL_FIDELITY_RULES,
            "generator": GENERATOR_PROMPT,
            "specialized_generator": SPECIALIZED_GENERATOR_PROMPT,
            "mutator": MUTATOR_PROMPT,
            "crossover": CROSSOVER_PROMPT,
            "fidelity_auditor": FIDELITY_AUDITOR_PROMPT,
            "evaluator": EVALUATOR_PROMPT,
        },
        "input_hashes_sha256": {
            args.input_file: sha256_file(args.input_file),
            args.firm_set: sha256_file(args.firm_set),
            "selected_firm_records": selected_input_hash(args.input_file, firm_ids),
            **{path: sha256_file(path) for path in design_files},
        },
        "artifacts": {
            "local": local_outputs,
            "global": global_outputs,
            "quality": quality_outputs,
            "semantic_scores": semantic_outputs,
            "call_cache": str(call_cache_path),
            "output_directory": str(output_dir),
        },
        "actual_usage": {},
        "cache_ledger_usage": {},
        "stages": [],
    }
    refresh_manifest_usage(manifest)
    write_manifest(manifest_path, manifest)
    child_env = {
        "AI_ENTREP_MAX_CONCURRENT_CALLS": str(args.max_concurrent_calls),
        "AI_ENTREP_CALL_CACHE": str(call_cache_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if args.mock:
        child_env["AI_ENTREP_MOCK_LLM"] = "1"

    global_cmd = [
        sys.executable,
        "src/global_search.py",
        "--firm-set",
        args.firm_set,
        "--preset",
        str(firms),
        "--model",
        args.model,
        "--steps",
        str(args.steps),
        "--input-file",
        args.input_file,
        "--run-prefix",
        run_id,
        "--output-dir",
        str(output_dir),
    ]
    local_cmd = [
        sys.executable,
        "src/local_search.py",
        "--seed-id",
        local_seed_id,
        "--model",
        args.model,
        "--steps",
        str(args.steps),
        "--input-file",
        args.input_file,
        "--run-prefix",
        run_id,
        "--output-dir",
        str(output_dir),
    ]
    quality_cmd = [
        sys.executable,
        "src/common_quality.py",
        "--local-csv",
        local_outputs["csv"],
        "--global-csv",
        global_outputs["csv"],
        "--firm-set",
        args.firm_set,
        "--model",
        args.model,
        "--run-prefix",
        run_id,
        "--input-file",
        args.input_file,
        "--output-dir",
        str(output_dir),
        "--scope",
        args.quality_scope,
    ]
    semantic_cmd = [
        sys.executable,
        "src/semantic_scores.py",
        "--input-file",
        args.input_file,
        "--firm-set",
        args.firm_set,
        "--candidate-csv",
        local_outputs["candidates"],
        "--candidate-csv",
        global_outputs["candidates"],
        "--output",
        semantic_outputs["scores"],
    ]
    if args.mock:
        semantic_cmd.append("--mock")

    run_stage(manifest_path, manifest, "global-search", global_cmd, global_outputs, args.force, args.dry_run, child_env)
    run_stage(manifest_path, manifest, "local-search", local_cmd, local_outputs, args.force, args.dry_run, child_env)
    if args.quality_calibration == "seed-benchmark":
        run_stage(manifest_path, manifest, "common-quality", quality_cmd, quality_outputs, args.force, args.dry_run, child_env)
    run_stage(manifest_path, manifest, "semantic-scores", semantic_cmd, semantic_outputs, args.force, args.dry_run, child_env)

    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["status"] = "dry_run_complete" if args.dry_run else "completed"
    refresh_manifest_usage(manifest)
    write_manifest(manifest_path, manifest)
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] output directory: {output_dir}")


if __name__ == "__main__":
    main()
